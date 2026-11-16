// Q8_0 weight repack for gfx906: see rp.cu.
#pragma once

#include "common.cuh"

bool ggml_cuda_rp_enabled();
// True when t (not a view) holds repacked bytes on this device.
bool ggml_cuda_rp_is_repacked(int device, const ggml_tensor * t);
// Before graph capture: pin weights with other consumers, repack the MUL_MAT weights of this graph.
void ggml_cuda_rp_prepare_graph(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);
// A device buffer is going away: forget everything inside it.
void ggml_cuda_rp_buffer_freed(int device, const void * base, size_t size);
// A raw read/write/copy of tensor bytes: aborts if t is repacked.
void ggml_cuda_rp_check_access(const ggml_tensor * t, const char * what);
// Host read of tensor bytes: returns true (and fills data with canonical bytes) when t is repacked.
bool ggml_cuda_rp_get_tensor(const ggml_tensor * t, void * data, size_t offset, size_t size);
size_t ggml_cuda_rp_stats(int device, size_t * n_tensors);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
