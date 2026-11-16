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
| Raw decode, no drafter (llama-bench tg128) | 28.9 t/s (30.4 with A2b aligned reads) |
| Prefill (llama-bench pp512 / pp4096) | 332 / 357 t/s (361 / 393 with the A3 table) |
| Quality: KLD vs gfx906-turbo baseline / same top token | 0.000596 / 99.42% |

## Tried

| Date | Idea | Result | Status | Where |
|---|---|---|---|---|
| 2026-11-15 | **A2d: 5-token pass, matmul shape with aligned reads** (7 variants) | current setting (1 group, 2 rows) best at pp5 101.8 / 102.0; others 68.7-99.0 | NO GAIN | `opt/a2-mmvq-tune` (knobs) |
| 2026-11-15 | A2d: tiled kernel (MMQ) for 2-8 token batches instead of mmvq | pp5 41.6 vs 101.8 (2.5x slower) | WORSE | `opt/a2-mmvq-tune` (`GGML_MMVQ_GCN_Q8_0_MAX_BATCH`, default 8) |
| 2026-11-15 | HIP graphs off, re-checked in the server with drafter | 52.8 / 44.9 vs 52.7 / 45.3 (short / 20k); the earlier llama-bench +8% was run-to-run noise | NO GAIN | - |
| 2026-11-15 | **Profile: 1-token vs 5-token (drafter check) pass** (rocprofv3, per card) | Q8 matmul 22.6 -> 36.8 ms (1.6x for the same weight bytes: compute-bound at 5 columns), GPU0 sync wait 6.3 -> 11.4 ms, small kernels 7.3 -> 8.5 ms, delta-net + attention 1.1 -> 2.0 ms. The 5-column matmul is the biggest lever with the drafter on. Side finding: HIP graphs off gave pp5 101.6 vs 94 t/s in llama-bench | finding | - |
| 2026-11-15 | **A2b: aligned Q8_0 weight reads** (dword loads + `v_alignbyte` instead of 16-bit load pairs), 2 ints per thread | raw 1-token decode +5.2% (28.85 -> 30.36, 2x5 interleaved runs); 5-token drafter check +0.5% (93.9 -> 94.4); server with drafter unchanged (53.3 / 45.2 vs 53.2 / 44.9); output **bit-identical** with and without drafter. Kept as default: free gain without the drafter | **KEPT** (no gain with drafter) | `opt/a2-mmvq-tune` (`GGML_MMVQ_Q8_0_ALIGNED=1`) |
| 2026-11-15 | A2b variants: 4 or 8 ints per thread (`VDR`) | 8 + aligned: 1-token +8% but 5-token drafter check -17% (75.6 vs 90.7); 4 + aligned: between. Different sum order, so not bit-identical | WORSE (with drafter) | `opt/a2-mmvq-tune` (knob) |
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
| A2f: new gfx906 kernel for 2-8 token Q8_0 matmuls (activations in LDS, wide weight reads, register reuse across tokens). Settings can't fix it: the 5-token pass loses ~14 ms/card to instruction overhead, not compute (~3 ms of dot4 work) | up to +20-30% with drafter | days |
| A2e: per-pass width (8 ints per thread for 1-token passes, 2 for drafter checks) | +3% without drafter only | hours |
| A2c: full Q8_0 re-layout at load time (aligned quants, separate scales) | small: A2b already gets the aligned-read benefit | days |
| A5: re-test n-gram drafting under tensor split | file edits only | ~1 hour |
| A4: fuse more of the ~1,600 small kernels per token | +10-15% | days to weeks |
| B1: Infinity Fabric bridge (hardware) | +10-15%, removes the PCIe sync cost A1 found | research first |
| Fix upstream's DFlash + tensor-split crash, then move to upstream | maintained base, +13% prefill | unknown |
