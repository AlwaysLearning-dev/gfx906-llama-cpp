# Optimization log

Every speed idea tried on this setup (2x MI50 32 GB, Qwen3.8-27B Q8_0, DFlash2 Q8 drafter at depth 4, f16 KV,
`-c 409600 --parallel 2`), what happened, and where the code lives. Newest first within each section. Settings
never change between rows; only the code or one switch does.

Kept results are merged into `main`; branches of experiments that didn't help were deleted, and this log is their record.

Status: **KEPT** = in use or recommended, **NO GAIN** = measured, within noise, **WORSE** = measured slower or
lower quality, **BROKEN** = crashes or can't run, **OPEN** = in progress or not tried yet.

## Current baseline

| Measure | Value |
|---|---|
| Generation with drafter: replay / short / 20k / 40k / 60k | 62.8 / 50.5 / 43.0 / 38.5 / 36.3 t/s |
| Two users at once | ~25 t/s each |
| Raw decode, no drafter (llama-bench tg128) | 30.4 t/s (28.9 before the A2b aligned reads) |
| Prefill (llama-bench pp512 / pp4096) | 361 / 393 t/s (332 / 357 before the A3 table) |
| Quality: KLD vs gfx906-turbo baseline / same top token | 0.000596 / 99.42% |

## Tried

| Date | Idea | Result | Status | Where |
|---|---|---|---|---|
| 2026-11-15 | **T3: two ARCTIC S8038-10K fans (one per card, SYS_FAN4 + SYS_FAN2) with a two-fan junction curve** | throttle gone: GPU0 holds 1725 MHz at 85-88 C (was 1485-1606 MHz at 99-101 C), GPU1 69-78 C, fans at ~8200 of 10000 RPM. Same 6000-token text: 48.0 -> 49.2 t/s (+2.5%). Per-card mode (`MAP=split`): GPU0 86 C with its fan at ~8200 RPM, GPU1 72-79 C with its fan at 4700-6500 RPM, same 49.2 t/s | **KEPT** (cooler cards, small speed gain) | `opt/t1-thermal-balance` (`mi50/setup/fan/gpu-fan.sh`) |
| 2026-11-15 | **T1: drafter on GPU1** (`DRAFT_DEV=ROCm1`), same 6000-token generation | 48.0 -> 49.0 t/s. Heat follows the load: GPU1 now also reaches 100 C. VRAM is better balanced (GPU0 27.9 GB, GPU1 30.9 GB used instead of 31.7 / 27.1) | NO GAIN (speed), better VRAM balance | `opt/t1-thermal-balance` |
| 2026-11-15 | Control: no drafter, both cards doing identical work | both cards reach 100-101 C junction and throttle to ~1620-1700 MHz. Cooling is short for both cards; GPU1 only looked cooler because it idles while GPU0 drafts | finding | - |
| 2026-11-15 | **T2: fan curve on the junction sensor** (full speed at 85 C) | 49.0 -> 49.2 t/s; GPU0 peak 97 C instead of 100 C, GPU1 still 100 C at full fan | KEPT (safer, not faster) | `opt/t1-thermal-balance` (`mi50/setup/fan/gpu-fan.sh`) |
| 2026-11-15 | **D1: performance level `high`** | 49.2 -> 49.7 t/s (within noise); raises idle power | NO GAIN | - |
| 2026-11-15 | **A2f step 2 v1: LDS-staged kernel for 2-8 token Q8_0 matmuls** (256-thread blocks, 8 rows share one LDS copy of the tokens' activations) | slower: pp5 93.7 -> 84.5, server with drafter 53.3 / 44.8 -> 46.6 / 38.4. Not bit-identical. Overheads (integer division per staged value, barriers per chunk, LDS bank conflicts) outweigh the saved loads | WORSE (v1) | branch removed (code not kept) |
| 2026-11-15 | **A2f step 1: load each weight row once, reuse for all 5 tokens** (and the fused gate) | bit-identical but slower: pp5 92.7 -> 90.3, server with drafter 53.3 / 44.8 -> 50.3 / 42.6. The compiler already removed the duplicate loads; holding more values raised register use | WORSE | branch removed (code not kept) |
| 2026-11-15 | Finding: why the 5-token matmul is slow | 60 registers per thread (4 waves per SIMD vs 10 for 1-token), 64-thread blocks, ~6 memory loads per 4 weight bytes (1 weight + 5 activations from cache) vs 2 for 1-token. Limited by load-instruction count and latency, not bandwidth or math. Gate/up fusion is not used at 5 tokens | finding | - |
| 2026-11-15 | **A2d: 5-token pass, matmul shape with aligned reads** (7 variants) | current setting (1 group, 2 rows) best at pp5 101.8 / 102.0; others 68.7-99.0 | NO GAIN | `opt/a2-mmvq-tune` (knobs) |
| 2026-11-15 | A2d: tiled kernel (MMQ) for 2-8 token batches instead of mmvq | pp5 41.6 vs 101.8 (2.5x slower) | WORSE | `opt/a2-mmvq-tune` (`GGML_MMVQ_GCN_Q8_0_MAX_BATCH`, default 8) |
| 2026-11-15 | HIP graphs off, re-checked in the server with drafter | 52.8 / 44.9 vs 52.7 / 45.3 (short / 20k); the earlier llama-bench +8% was run-to-run noise | NO GAIN | - |
| 2026-11-15 | **Profile: 1-token vs 5-token (drafter check) pass** (rocprofv3, per card) | Q8 matmul 22.6 -> 36.8 ms (1.6x for the same weight bytes: compute-bound at 5 columns), GPU0 sync wait 6.3 -> 11.4 ms, small kernels 7.3 -> 8.5 ms, delta-net + attention 1.1 -> 2.0 ms. The 5-column matmul is the biggest lever with the drafter on. Side finding: HIP graphs off gave pp5 101.6 vs 94 t/s in llama-bench | finding | - |
| 2026-11-15 | **A2b: aligned Q8_0 weight reads** (dword loads + `v_alignbyte` instead of 16-bit load pairs), 2 ints per thread | raw 1-token decode +5.2% (28.85 -> 30.36, 2x5 interleaved runs); 5-token drafter check +0.5% (93.9 -> 94.4); server with drafter unchanged (53.3 / 45.2 vs 53.2 / 44.9); output **bit-identical** with and without drafter. Kept as default: free gain without the drafter | **KEPT** (no gain with drafter) | `opt/a2-mmvq-tune`, `main` (`GGML_MMVQ_Q8_0_ALIGNED=1`) |
| 2026-11-15 | A2b variants: 4 or 8 ints per thread (`VDR`) | 8 + aligned: 1-token +8% but 5-token drafter check -17% (75.6 vs 90.7); 4 + aligned: between. Different sum order, so not bit-identical | WORSE (with drafter) | `opt/a2-mmvq-tune` (knob) |
| 2026-11-15 | **A2a: mmvq launch shape** (thread groups per row, rows per block) for 1-token and 5-token (drafter check) passes | 1-token best: +1.8% (29.35 vs 28.84). 5-token: current setting already best. With the drafter on, almost all passes are 5-token, so no real gain | NO GAIN | `opt/a2-mmvq-tune` (knobs in `mmvq.cu`) |
| 2026-11-15 | **A3b: upstream GCN prefill table** (ggml-org/llama.cpp#30021) with milpster's vega table off | prefill +8.5% (pp512 332 -> 361), +10% (pp4096 357 -> 393), server 20k prompt 317 -> 344 t/s; generation unchanged; KLD identical | **KEPT** | `opt/a3-upstream-sync`, `main` |
| 2026-11-15 | **A3a: plain upstream llama.cpp** instead of milpster | raw tg +1%, prefill +13%, same quality, but **crashes at startup with DFlash + `-sm tensor`** (`ggml-backend-meta.cpp:573 GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)`) | BROKEN | `~/llama-upstream` test build |
| 2026-11-15 | A3: pull upstream HIP AllReduce (#27825) and drafter meta wrapper (#28390) | already in milpster's tree; nothing to do | - | - |
| 2026-11-15 | **A1: queue both GPUs' work from two CPU threads** (`GGML_META_PARALLEL_LAUNCH=1`) | tg 28.67 -> 28.84, server short 51.9 -> 52.6, 20k 44.6 -> 44.5; output identical. GPU1's late start comes from the host-staged PCIe sync, not the CPU | NO GAIN | branch removed (code not kept) |
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

## Findings from the 2026-11-15 research round (measured on the live server)

- **GPU0 is thermally throttled on any answer longer than ~40 s.** 3-minute generation: GPU0 junction reaches 99-101 C (critical is 100 C) after 40 s and stays there; its clock drops from 1725 to 1485-1606 MHz at 150-190 W (below the 225 W cap). GPU1 stays at 1725 MHz, 86-90 C, ~205 W. With tensor split the slower card sets the pace.
- GPU0 carries the drafter (17% of each round) on top of its half of the model; GPU1 idles during that time and its clock drops to 925 MHz.
- The fan script follows the edge sensor, which reads ~25 C below junction: the fan reaches full speed only after ~80 s, and full speed does not pull GPU0 below 100 C.
- A round costs ~69 ms: ~12 ms drafting (17%), ~55 ms for the 5-token check. Rounds yield 3.2-4.5 tokens depending on content.

## Not tried yet

| Idea | Evidence | Expected | Effort |
|---|---|---|---|
| S1: MTP drafting (`--spec-type draft-mtp`, `mtp-Qwen3.8-27B-Q8_0.gguf`, 3.0 GB) instead of DFlash2 | others: 2x R9700 ROCm tensor split MTP n3 56.5 vs DFlash2 n3 53.3 t/s; single large GPU DFlash2 187 vs MTP 159 | -10% to +10% | ~1 hour; may not run with tensor split on this fork |
| S2: `ngram-map-k4v` + DFlash2 (never tested here; the October test used ngram-mod) | others: +68% on iterative code edits, +1% on fresh code | large for file re-emission, ~0 otherwise | ~1 hour |
| S3: `--spec-draft-p-min` 0.3-0.6 to cut unsure draft tails (smaller checks) | others: helps slow cards, hurts fast ones; may be ignored for DFlash2 in this code version | -5% to +5% | 30 min |
| K1: verify-kernel v2 after Luce-Org/lucebox#859 (whole columns staged in LDS once, 2 rows per wave, per-wave full-K traversal) | +5.5% on gfx1151, bit-identical; never run on GCN. Our v1 lost 10% | ~+5% | days |
| K2: HIP graph cache keyed on graph shape (halo-box/strix-llama.cpp#106) | +9.4% with speculative decoding on gfx1151; here graphs on/off measure the same, so graphs may not be reused | 0-9% | half a day to diagnose |
| M1: two users: batch both slots' checks in one pass (needs mmvq beyond 8 columns, even-length drafts) | two users total 50 t/s = one user | up to +30% total when both slots are busy | days |
| P1: raise the power limit by pp_table injection (UPP) | reported on a 16 GB Pro VII (190 -> 300 W); untested on MI50 32 GB | none until cooling is fixed | not recommended |
| B1: Infinity Fabric bridge | no source found for a purchasable bridge or ROCm support | - | parked |
| U1: move to upstream llama.cpp | upstream fix for DFlash2 + tensor split (ggml-org/llama.cpp#27858) is still a draft | +13% prefill | wait for merge |
| A4: fuse more of the small kernels | ~8 ms of a 35 ms pass | +10-15% without drafter | days to weeks |
