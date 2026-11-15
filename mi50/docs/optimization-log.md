# Optimization log

Every speed idea tried on this setup (2x MI50 32 GB, Qwen3.8-27B Q8_0, DFlash2 Q8 drafter at depth 4, f16 KV,
`-c 409600 --parallel 2`), what happened, and where the code lives. Newest first within each section. Settings
never change between rows; only the code or one switch does.

Status: **KEPT** = in use or recommended, **NO GAIN** = measured, within noise, **WORSE** = measured slower or
lower quality, **BROKEN** = crashes or can't run, **OPEN** = in progress or not tried yet.

## Current baseline

| Measure | Value |
|---|---|
| Generation with drafter: replay / short / 20k / 40k / 60k | 62.8 / 50.5 / 43.0 / 38.5 / 36.3 t/s |
| Two users at once | ~25 t/s each |
| Raw decode, no drafter (llama-bench tg128) | 28.9 t/s |
| Prefill (llama-bench pp512 / pp4096) | 332 / 357 t/s (361 / 393 with the A3 table) |
| Quality: KLD vs gfx906-turbo baseline / same top token | 0.000596 / 99.42% |

## Tried

| Date | Idea | Result | Status | Where |
|---|---|---|---|---|
| 2026-11-15 | **A2b: aligned Q8_0 weight reads** (dword loads + `v_alignbyte` instead of 16-bit load pairs) and 2/4/8 ints per thread (`VDR`) | in progress | OPEN | `opt/a2-mmvq-tune` |
| 2026-11-15 | **A2a: mmvq launch shape** (thread groups per row, rows per block) for 1-token and 5-token (drafter check) passes | 1-token best: +1.8% (29.35 vs 28.84). 5-token: current setting already best. With the drafter on, almost all passes are 5-token, so no real gain | NO GAIN | `opt/a2-mmvq-tune` (knobs in `mmvq.cu`) |
| 2026-11-15 | **A3b: upstream GCN prefill table** (ggml-org/llama.cpp#30021) with milpster's vega table off | prefill +8.5% (pp512 332 -> 361), +10% (pp4096 357 -> 393), server 20k prompt 317 -> 344 t/s; generation unchanged; KLD identical | **KEPT** | `opt/a3-upstream-sync` (this branch) |
| 2026-11-15 | **A3a: plain upstream llama.cpp** instead of milpster | raw tg +1%, prefill +13%, same quality, but **crashes at startup with DFlash + `-sm tensor`** (`ggml-backend-meta.cpp:573 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`) | BROKEN | `~/llama-upstream` test build |
| 2026-11-15 | A3: pull upstream HIP AllReduce (#27825) and drafter meta wrapper (#28390) | already in milpster's tree; nothing to do | - | - |
| 2026-11-15 | **A1: queue both GPUs' work from two CPU threads** (`GGML_META_PARALLEL_LAUNCH=1`) | tg 28.67 -> 28.84, server short 51.9 -> 52.6, 20k 44.6 -> 44.5; output identical. GPU1's late start comes from the host-staged PCIe sync, not the CPU | NO GAIN | `opt/a1-allreduce-wait` (local) |
| 2026-11-15 | A1 original idea: sleep inside the AllReduce wait loop | already done in the code (`__builtin_amdgcn_s_sleep(4)`) | - | - |
| 2026-11-14 | GPU clock level `high` | cards run at the 225 W cap, not clock-ramp limited | NO GAIN (not needed) | - |
| 2026-11-14 | RCCL AllReduce (`-DGGML_HIP_RCCL=ON`) | 28.8 vs 29.0 t/s | NO GAIN | separate build |
| 2026-11-14 | `GGML_CUDA_ALLREDUCE=none` (butterfly) | crashes | BROKEN | - |
| 2026-11-14 | HIP graphs off | within noise | NO GAIN | - |
| 2026-11-14 | Drafter depth 3 / 5 / 6 | all slower than 4 at short and 20k | WORSE | - |
| 2026-11-14 | **Tensor split** (`-sm tensor`) + drafter mirror fix | +37-44% with the drafter, +54% raw decode, +43% prefill; KLD unchanged; 40/40 suite | **KEPT** | `main` |
| 2026-10-19 | `-ub 1024` | prefill -20%, no VRAM left on GPU0 | WORSE | - |
| 2026-10-19 | q8_0 KV cache | slower at every depth (-14% at 64k) | WORSE | - |
| 2026-10-19 | n-gram + DFlash combos (layer split) | +27% on file replay, -6 to -8% on new text | WORSE overall | - |
| 2026-10-19 | Q4_K_M vs Q8_0 drafter | same speed, byte-identical output | NO GAIN (Q8 kept) | - |
| 2026-10-19 | **milpster fork + DFlash2 drafter** (vs gfx906-turbo) | ~1.7-2x generation, same quality | **KEPT** | `main` |
| 2026-10-12 | MXFP4 weights | +25% speed but worst quality tested (KLD ~0.05) | WORSE | - |
| 2026-10 | Q6_K / Q4 weights | Q6_K no faster; Q4 faster but worse in real use | WORSE | - |

## Not tried yet

| Idea | Expected | Effort |
|---|---|---|
| A2c: full Q8_0 re-layout at load time (aligned quants, separate scales) + wide-load kernel | depends on A2b | days |
| A5: re-test n-gram drafting under tensor split | file edits only | ~1 hour |
| A4: fuse more of the ~1,600 small kernels per token | +10-15% | days to weeks |
| B1: Infinity Fabric bridge (hardware) | +10-15%, removes the PCIe sync cost A1 found | research first |
| Fix upstream's DFlash + tensor-split crash, then move to upstream | maintained base, +13% prefill | unknown |
