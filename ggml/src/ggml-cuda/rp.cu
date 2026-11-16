// Q8_0 weight repack for gfx906 (two-plane layout: int8 quants, then f16 scales).
// The layout and the kernels in rp-kernels.cuh / rp-common.cuh come from mxxm-t/mx-llama.cpp
// (ggml/src/ggml-cuda/q8_repack, Q8_0 path contributed there by DENEB1312), MIT licensed.
// This file is the glue for this fork: instead of a separate buffer type, each eligible weight is
// repacked in place on the device the first time a graph uses it as src0 of MUL_MAT.
//   GGML_CUDA_Q8_REPACK=0   turn the repack off (default on, gfx906 only)
//   GGML_CUDA_RP_SPLIT=0    9-16 token batches: use the tiled kernel instead of two mat-vec launches
#include "rp.cuh"
#include "rp-common.cuh"
#include "rp-kernels.cuh"
#include "quantize.cuh"
#include "mmq.cuh"
#include "ggml-backend-impl.h"

#include <cstring>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#if defined(GGML_USE_HIP)

struct rp_registry {
    std::mutex mutex;
    std::unordered_map<const void *, size_t> repacked;   // tensor data -> nbytes
    std::unordered_set<const void *> canonical;          // must stay canonical (seen with another consumer)
    size_t n_bytes = 0;
    bool   logged  = false;
};
// Device pointers are unique across GPUs in one process (unified addressing), so one registry serves all
// devices, including virtual devices that share a physical GPU.
static rp_registry g_rp_all;

bool ggml_cuda_rp_enabled() {
    static const bool on = []() {
        const char * e = getenv("GGML_CUDA_Q8_REPACK");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

bool ggml_cuda_rp_is_repacked(int device, const ggml_tensor * t) {
    if (t == nullptr || t->type != GGML_TYPE_Q8_0 || t->view_src != nullptr || !ggml_cuda_rp_enabled()) {
        return false;
    }
    rp_registry & r = g_rp_all; GGML_UNUSED(device);
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.repacked.find(t->data) != r.repacked.end();
}

void ggml_cuda_rp_buffer_freed(int device, const void * base, size_t size) {
    rp_registry & r = g_rp_all; GGML_UNUSED(device);
    std::lock_guard<std::mutex> lock(r.mutex);
    const char * b = (const char *) base;
    for (auto it = r.repacked.begin(); it != r.repacked.end();) {
        const char * p = (const char *) it->first;
        if (p >= b && p < b + size) { r.n_bytes -= it->second; it = r.repacked.erase(it); } else { ++it; }
    }
    for (auto it = r.canonical.begin(); it != r.canonical.end();) {
        const char * p = (const char *) *it;
        if (p >= b && p < b + size) { it = r.canonical.erase(it); } else { ++it; }
    }
}

void ggml_cuda_rp_check_access(const ggml_tensor * t, const char * what) {
    if (t == nullptr || !ggml_cuda_rp_enabled()) {
        return;
    }
    const ggml_tensor * base = t->view_src ? t->view_src : t;
    if (base->type != GGML_TYPE_Q8_0) {
        return;
    }
    rp_registry & r = g_rp_all;
    std::lock_guard<std::mutex> lock(r.mutex);
    if (r.repacked.find(base->data) != r.repacked.end()) {
        GGML_ABORT("q8 repack: %s on repacked weight '%s' (set GGML_CUDA_Q8_REPACK=0)", what, base->name);
    }
}

// Host read of a repacked weight (e.g. the drafter's mirror of the output head): fetch the two planes
// for the blocks that cover [offset, offset + size) and rebuild the canonical bytes.
bool ggml_cuda_rp_get_tensor(const ggml_tensor * t, void * data, size_t offset, size_t size) {
    if (t == nullptr || t->type != GGML_TYPE_Q8_0 || !ggml_cuda_rp_enabled()) {
        return false;
    }
    {
        rp_registry & r = g_rp_all;
        std::lock_guard<std::mutex> lock(r.mutex);
        const ggml_tensor * base = t->view_src ? t->view_src : t;
        if (r.repacked.find(base->data) == r.repacked.end()) {
            return false;
        }
        if (t->view_src != nullptr) {
            GGML_ABORT("q8 repack: get_tensor on a view of repacked weight '%s' (set GGML_CUDA_Q8_REPACK=0)", base->name);
        }
    }
    if (size == 0) {
        return true;
    }
    const size_t ne0 = t->ne[0], ne1 = t->ne[1];
    const size_t b0 = offset / 34, b1 = (offset + size + 33) / 34;
    const size_t nb = b1 - b0;
    std::vector<uint8_t>  qs(nb * 32);
    std::vector<uint16_t> ds(nb);
    const uint8_t * dev = (const uint8_t *) t->data;
    CUDA_CHECK(cudaMemcpy(qs.data(), dev + b0 * 32, nb * 32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ds.data(), dev + ne1 * ne0 + b0 * 2, nb * 2, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> canon(nb * 34);
    for (size_t i = 0; i < nb; i++) {
        memcpy(canon.data() + i * 34, &ds[i], 2);
        memcpy(canon.data() + i * 34 + 2, qs.data() + i * 32, 32);
    }
    memcpy(data, canon.data() + (offset - b0 * 34), size);
    return true;
}

static __global__ void rp_extract_d_kernel(const uint8_t * __restrict__ src, uint16_t * __restrict__ dst, const int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const uint8_t * sb = src + i * 34;
    dst[i] = (uint16_t) sb[0] | ((uint16_t) sb[1] << 8);
}

static __global__ void rp_pack_qs_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, const int64_t n_blocks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) {
        return;
    }
    const uint8_t * sb = src + i * 34 + 2;
    uint8_t * d = dst + i * 32;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        d[k] = sb[k];
    }
}

// In place: canonical rows of ne0/32 blocks x 34 B  ->  [ne1 x ne0 quants][ne1 x ne0/32 f16 scales].
// The quant rows only move towards lower addresses, so they are compacted front to back through a
// small scratch; the scales are parked in a second scratch and written to the freed tail.
static void rp_repack_in_place(ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
    const int64_t n_sub = ne0 / 32, n_blocks = ne1 * n_sub;
    uint8_t * data = (uint8_t *) t->data;
    cudaStream_t stream = ctx.stream();

    const int64_t chunk_rows = std::max<int64_t>(1, std::min<int64_t>(ne1, (16ll << 20) / ne0));
    uint16_t * dscr = nullptr;
    uint8_t  * qscr = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &dscr, n_blocks * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc((void **) &qscr, chunk_rows * ne0));

    rp_extract_d_kernel<<<(n_blocks + 255) / 256, 256, 0, stream>>>(data, dscr, n_blocks);
    for (int64_t r0 = 0; r0 < ne1; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, ne1 - r0);
        const int64_t nb = nr * n_sub;
        rp_pack_qs_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(data + r0 * n_sub * 34, qscr, nb);
        CUDA_CHECK(cudaMemcpyAsync(data + r0 * ne0, qscr, nr * ne0, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(data + ne1 * ne0, dscr, n_blocks * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dscr));
    CUDA_CHECK(cudaFree(qscr));
}

static int rp_ineligible_reason(const ggml_tensor * node, const ggml_tensor * w) {
    if (w->type != GGML_TYPE_Q8_0 || w->view_src != nullptr || w->data == nullptr || w->buffer == nullptr) {
        return 1;
    }
    if (ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return 2;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] % 128 != 0 || w->ne[0] < 256 || !ggml_is_contiguous(w)) {
        return 3;
    }
    if (((uintptr_t) w->data & 15) != 0) {
        return 4;
    }
    const ggml_tensor * x = node->src[1];
    if (x == nullptr || x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 || x->ne[0] != w->ne[0]) {
        return 5;
    }
    if (ggml_get_op_params_i32(node, 1) != 0) {   // hinted special cases keep their own path
        return 6;
    }
    return 0;
}

static bool rp_eligible(int device, const ggml_tensor * node, const ggml_tensor * w) {
    GGML_UNUSED(device);
    const int why = rp_ineligible_reason(node, w);
    static const bool dbg = getenv("GGML_CUDA_RP_DEBUG") != nullptr;
    if (dbg && why != 0 && w->type == GGML_TYPE_Q8_0) {
        static int n = 0;
        if (n++ < 12) {
            GGML_LOG_WARN("q8 repack: skip '%s' reason %d (ne %lld x %lld, view %d, usage %d, data %p)\n", w->name, why,
                (long long) w->ne[0], (long long) w->ne[1], w->view_src != nullptr,
                w->buffer ? (int) ggml_backend_buffer_get_usage(w->buffer) : -1, w->data);
        }
    }
    return why == 0;
}

// Runs before graph capture. Pass 1 pins every Q8_0 weight that has a consumer other than
// MUL_MAT src0 to the canonical layout; pass 2 repacks the remaining MUL_MAT weights.
void ggml_cuda_rp_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    {
        static const bool dbg = getenv("GGML_CUDA_RP_DEBUG") != nullptr;
        static int n = 0;
        if (dbg && n++ < 3) {
            GGML_LOG_WARN("q8 repack: prepare_graph dev %d nodes %d enabled %d\n", ctx.device, cgraph->n_nodes, (int) ggml_cuda_rp_enabled());
        }
    }
    if (!ggml_cuda_rp_enabled()) {
        return;
    }
    rp_registry & r = g_rp_all;
    bool any_q8 = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * t = node->src[s];
            if (t == nullptr) {
                continue;
            }
            const ggml_tensor * base = t->view_src ? t->view_src : t;
            if (base->type != GGML_TYPE_Q8_0 || base->data == nullptr) {
                continue;
            }
            any_q8 = true;
            if (node->op == GGML_OP_MUL_MAT && s == 0 && rp_eligible(ctx.device, node, t)) {
                continue;
            }
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.repacked.find(base->data) != r.repacked.end()) {
                GGML_ABORT("q8 repack: op %s reads repacked weight '%s' (set GGML_CUDA_Q8_REPACK=0)",
                    ggml_op_name(node->op), base->name);
            }
            r.canonical.insert(base->data);
        }
    }
    if (!any_q8) {
        return;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || !rp_eligible(ctx.device, node, node->src[0])) {
            continue;
        }
        const ggml_tensor * w = node->src[0];
        {
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.repacked.find(w->data) != r.repacked.end() || r.canonical.find(w->data) != r.canonical.end()) {
                continue;
            }
        }
        rp_repack_in_place(ctx, w);
        std::lock_guard<std::mutex> lock(r.mutex);
        r.repacked[w->data] = ggml_nbytes(w);
        r.n_bytes += ggml_nbytes(w);
        if (!r.logged) {
            r.logged = true;
            GGML_LOG_WARN("q8 repack: device %d repacks Q8_0 weights in place (first: %s, %lld x %lld)\n",
                ctx.device, w->name, (long long) w->ne[0], (long long) w->ne[1]);
        }
    }
}

size_t ggml_cuda_rp_stats(int device, size_t * n_tensors) {
    rp_registry & r = g_rp_all; GGML_UNUSED(device);
    std::lock_guard<std::mutex> lock(r.mutex);
    if (n_tensors) {
        *n_tensors = r.repacked.size();
    }
    return r.n_bytes;
}

template <ggml_type WT>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream);

static void rp_gemm(const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11, cudaStream_t stream) {
    const int nrl    = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    if (ne11 >= 128) {
        const int bn = 64 * MMQ_RP_Q8_TN;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm, (ne11 + bn - 1) / bn, 1);
        mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q8_0><<<grid, dim3(64, nrl), 0, stream>>>(
            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
    } else {
        const int bn = 32;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm, (ne11 + bn - 1) / bn, 1);
        mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q8_0><<<grid, dim3(32, nrl*2), 0, stream>>>(
            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
    }
}

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();
    const uint8_t * w = (const uint8_t *) src0->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    static const bool split = []() { const char * e = getenv("GGML_CUDA_RP_SPLIT"); return e == nullptr || atoi(e) != 0; }();
    const int64_t mmv_max = split ? 2 * MMQ_RP_Q8_MMV_MAX_TOKENS : MMQ_RP_Q8_MMV_MAX_TOKENS;

    if (ne11 <= mmv_max) {
        ggml_cuda_pool_alloc<block_q8_1> xq_alloc(ctx.pool(), (size_t) (ne13 * ne12 * ne11 * x_stride));
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, xq_alloc.get(),
            src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1 * xq = xq_alloc.get() + (i3 * ne12 + i2) * ne11 * x_stride;
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            if (ne11 == 1) {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_q8_0_repacked<16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } else if (ne11 <= MMQ_RP_Q8_MMV_MAX_TOKENS) {
                ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q8_0>(w, xq, dst_d, ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
            } else {
                const int64_t n1 = (ne11 + 1) / 2;
                ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q8_0>(w, xq, dst_d, ne00, ne01, n1, (uint32_t) x_stride, dst_s1, stream);
                ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q8_0>(w, xq + n1 * x_stride, dst_d + n1 * dst_s1, ne00, ne01, ne11 - n1,
                    (uint32_t) x_stride, dst_s1, stream);
            }
        }
        }
        return;
    }

    const uint64_t n_groups = (uint64_t) ne10_padded / (4 * QK8_1);
    ggml_cuda_pool_alloc<block_q8_1_mmq_h> xq_alloc(ctx.pool(), (size_t) (ne13 * ne12 * ne11 * n_groups));
    quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, xq_alloc.get(),
        src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = reinterpret_cast<const block_q8_1 *>(xq_alloc.get() + (i3 * ne12 + i2) * ne11 * n_groups);
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        rp_gemm(w, xq, dst_d, ne00, ne01, ne11, stream);
    }
    }
}

// Narrow batch (2..8 tokens): one multi-column mat-vec pass per slice.
template <ggml_type WT>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream) {
    // Geometry per width, measured on Qwen3.8-27B (pp512 at that ubatch, 4x
    // MI50 -sm tensor). The row count per lane is small once the tensor is
    // split, so scheduling granularity beats per-group reuse and 64-thread
    // workgroups spread over the CUs win. Two rows per lane reuse the
    // activation block across rows and pay from 3 tokens up; at 2 there is
    // too little to amortize, and 4 rows spills.
    // Lane width per row, chosen from the ROW LENGTH and the batch width.
    //
    // LANES is the number of lanes cooperating on one row. The strided loop
    // runs ne0/32/LANES iterations and is followed by a log2(LANES)-step DPP
    // reduction, so the useful knob is how much work a lane gets before it has
    // to reduce. Narrowing the lane group raises that, and raises occupancy,
    // but it also cuts the block count that hides memory latency.
    //
    // Which effect wins is decided by the row length, and two models disagree
    // about the batch width alone, so that is the wrong axis. Measured pp512
    // at the ubatch, 2x MI50 -sm layer, narrow-lane against the 64-lane shape:
    //
    //                       ne0    K-blocks   width 2   width 3
    //   Qwen3.6-27B        5120       160      +13.8%    +2.1%
    //   Qwen3.6-35B-A3B    2048        64       -9.5%    -0.7%
    //
    // A 2048-wide row is 64 K-blocks, so splitting it over 16 lanes leaves 4
    // iterations against a 4-step reduction while cutting the grid by 16x -
    // the row is simply too short to pay for the lost parallelism. Gate on the
    // block count so the choice follows the shape and not the model.
    // Narrowing the lane group raises the work a lane does before it reduces,
    // and multiplies the rows a block covers, which shrinks the grid. Both
    // have a floor and BOTH must hold.
    //
    // A lane accumulates n_blocks/LANES times, then pays a log2(LANES)-step
    // reduction, so require at least four accumulations at the widest group we
    // would narrow to. Measured on two models: 160 and 128 blocks win, 64
    // loses at every lane width, so the break is between them.
    //
    // The grid condition is what the row length alone cannot see - a 48-row
    // tensor is a 2-block grid however long its rows are.
    constexpr int RP_NC_MIN_ACCUM   = 4;
    constexpr int RP_NC_LANES_WIDE  = 32;   // widest group we ever narrow to
    const int     nsm       = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t n_blocks  = ne00 / 32;
    const int64_t min_grid  = 2LL * nsm;
    const bool    work_ok   = n_blocks >= RP_NC_MIN_ACCUM * RP_NC_LANES_WIDE;
    const bool    lanes16_ok = work_ok && (ne01 + 31) / 32 >= min_grid;
    const bool    lanes32_ok = work_ok && (ne01 +  7) /  8 >= min_grid;
    switch (ne11) {
        case 2: {
            if (lanes16_ok) {
                // 16 lanes to a row, 32 rows per block, one row per lane.
                const dim3 grid((ne01 + 31) / 32, 1, 1);
                mul_mat_vec_repacked_nc<32, 8, 2, 1, 16, WT><<<grid, 512, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 2, 2, 1, 64, WT><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 3: {
            if (lanes32_ok) {
                // 32 lanes to a row, 8 rows per block, one row per lane.
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_repacked_nc<8, 4, 3, 1, 32, WT><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 1, 3, 2, 64, WT><<<grid, 64, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 4: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 4, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 5: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 5, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 6: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 6, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 7: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 7, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 8: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 8, 2, 64, WT><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        default: GGML_ABORT("nc mat-vec: unsupported batch width");
    }
}


#else  // !GGML_USE_HIP

bool ggml_cuda_rp_enabled() { return false; }
bool ggml_cuda_rp_is_repacked(int, const ggml_tensor *) { return false; }
void ggml_cuda_rp_buffer_freed(int, const void *, size_t) {}
void ggml_cuda_rp_check_access(const ggml_tensor *, const char *) {}
bool ggml_cuda_rp_get_tensor(const ggml_tensor *, void *, size_t, size_t) { return false; }
void ggml_cuda_rp_prepare_graph(ggml_backend_cuda_context &, ggml_cgraph *) {}
size_t ggml_cuda_rp_stats(int, size_t *) { return 0; }
void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("q8 repack is HIP only");
}

#endif
