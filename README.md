# gfx906-llama-cpp (dual MI50 tensor split)

llama.cpp for **two AMD Instinct MI50 32 GB** cards (gfx906): Qwen3.8-27B at full **Q8_0** with a DFlash2
speculative drafter and **tensor split** across both cards.

> **This is the `dev_v2` branch:** everything in `dev` plus the Q8_0 weight repack, which makes prompt processing
> **18-25% faster** (layout and kernels from [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp)). **~58 tokens/s** average generation and
> **~435 tokens/s** prompt processing (20k prompt), same quality. It is the newest branch and still in testing.

## Which branch?

All three run the same model, quality and settings (Qwen3.8-27B Q8_0, DFlash2 drafter, tensor split). Speeds are tokens/s
on the reference box.

| Branch | What it adds | Avg generation (40-task suite) | Generation at 20k context | Prompt processing (20k prompt) | Two users at once, total |
|---|---|---|---|---|---|
| [`main`](../../tree/main) | first published state: tensor split, upstream prefill table, aligned Q8_0 reads | 49.6 | 43.0 | 344 | ~50 |
| [`dev`](../../tree/dev) | `main` + faster verify kernel, two-user kernel, two-fan cooling | 58.1 | 50.4 | 346 | 63-65 |
| [`dev_v2`](../../tree/dev_v2) | `dev` + Q8_0 weight repack, with the layout and kernels from [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) (newest, in testing) | 58.5 | 53.0 | **434** | 65-67 |

- Pick **`dev_v2`** for the fastest build, above all for long prompts (coding agents, large contexts).
- Pick **`dev`** for the same generation speed without the newest change.
- Pick **`main`** for the smallest set of changes on top of milpster's fork.

Credit: the gfx906 kernels and DFlash2 come from [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp),
the base from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), and the weight repack in `dev_v2` from
[mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) (Q8_0 path contributed there by DENEB1312). Details under [Credit](#credit).

Quality is the same on all three: KLD 0.000596 against the gfx906-turbo baseline, same top token 99.42%, task suite
40/40.

## Results side by side

Same model (Qwen3.8-27B Q8_0), same settings and the same tests in every column. Speeds are tokens/s.

| Test | gfx906-turbo, layer split, no drafter | milpster fork + DFlash2, layer split | `main`: + tensor split | `dev` | **`dev_v2`** | `dev_v2` vs `dev` (same session) |
|---|---|---|---|---|---|---|
| **Average over the 40-task suite** | 19.0 | 36.3 | 49.6 | 58.1 | **58.5** | same |
| One user, short prompt | - | ~36 | 50.5 | 55.8 | **58.7** | same (58.1 -> 58.7) |
| One user, 20k context | - | 30.5 | 43.0 | 50.4 | **53.0** | +6% (50.1 -> 53.0) |
| One user, 60k context | - | 26.5 | 36.3 | 42.7 | **41.7** | same (42.5 -> 41.7) |
| One user, re-generating an existing file | - | 43.7 | 62.8 | 69.9 | **72.6** | +3% (70.3 -> 72.6) |
| Two users at once, total | - | ~36 (18 each) | ~50 (25 each) | 63-65 (31-34 each) | **65-67 (32-35 each)** | same |
| Prompt processing, 20k-token prompt | - | - | 344 | 346 | **434** | **+25%** |
| Prompt processing, 2.5k / 40k-token prompt | - | - | - | 309 / 282 | **366 / 341** | **+18% / +21%** |
| Quality: KLD vs gfx906-turbo (lower is closer) | 0 (reference) | - | 0.000596 | 0.000596 | 0.000596 | same |
| Quality: same top token as gfx906-turbo | 100% (reference) | - | 99.42% | 99.42% | 99.42% | same |
| Quality: task suite score | 39/40 | 40/40 | 40/40 | 40/40 | 40/40 | same |

"-" = not measured for that setup. One-user rows are 3-seed averages with the drafter at depth 4. `dev_v2` is **3.1x** the
gfx906-turbo starting point on the suite average.

The `dev` column is from its own full check; the last column compares `dev` and `dev_v2` measured back to back in one
session (the numbers in brackets), which is the fair comparison because speeds drift a few percent between sessions.

### Tests behind the `dev` column (2026-11-16)

| Test | How | Result |
|---|---|---|
| Speed, one user | `REPS=3 mi50/bench/bench2.py PORT label -1 0 20000 60000` | 69.9 / 55.8 / 50.4 / 42.7 t/s (replay / short / 20k / 60k) |
| Speed, two users | `mi50/bench/np2-check.py PORT`, 3 runs | 63.4 / 64.0 / 65.4 t/s total |
| Task suite, one user | `mi50/bench/qual-suite.py PORT label 5 out.json` (40 tasks) | 40/40, 58.1 t/s average |
| Task suite, two users at once | two copies of the suite in parallel (80 tasks) | 79/80, 32 t/s each (the miss is a logic puzzle that gfx906-turbo also misses 1 in 5) |
| KLD vs gfx906-turbo | `llama-perplexity -sm tensor --kl-divergence-base turbo.kld --kl-divergence` | mean 0.000596, same top token 99.42% (identical to `main`) |
| KLD, every pass forced to 10 tokens (the two-user path) | same with `-b 10 -ub 10` | 0.000632 / 99.34%: the same to the last digit as `main` with 5-token passes |
| Verify kernel on / off | temp-0 output at draft depth 4, 5, 6, 7 | identical text |
| Heat, 6000-token answer | junction temperature and clock every 5 s | GPU0 peaks at 81 C, GPU1 at 77 C, both hold 1725 MHz (before the second fan: 99-101 C and throttling) |

### Tests behind the `dev_v2` column (2026-11-16)

| Test | How | Result |
|---|---|---|
| Speed, one user | `REPS=3 mi50/bench/bench2.py PORT label -1 0 20000 60000` | 72.6 / 58.7 / 53.0 / 41.7 t/s (replay / short / 20k / 60k); prompt processing 366 / 434 / 341 t/s at 2.5k / 20k / 40k |
| Speed, two users | `mi50/bench/np2-check.py PORT`, 3 runs | 65.4 / 67.1 / 67.4 t/s total |
| Task suite, one user | `mi50/bench/qual-suite.py PORT label 5 out.json` (40 tasks) | 40/40, 58.5 t/s average |
| Task suite, two users at once | two copies of the suite in parallel (80 tasks) | 79/80, 33 t/s each (the miss is the same logic puzzle as on `dev`) |
| KLD vs gfx906-turbo | `llama-perplexity -sm tensor --kl-divergence-base turbo.kld --kl-divergence` | mean 0.000596, same top token 99.42% (identical to `dev` and `main`) |
| KLD, every pass forced to 10 or 5 tokens | same with `-b 10 -ub 10`, `-b 5 -ub 5` | 0.000628 / 99.41% (`dev`: 0.000632 / 99.34%) |
| Raw speed, no drafter | `llama-bench -p 512 -n 128`, tensor split | pp512 545 (`dev`: 430 with the same command), tg128 30.7 (`dev`: 30.4) |
| Image request | one chat request with an image | works |
| VRAM | used memory per card after load | unchanged from `dev` |

### What `dev` adds to `main`

| Change | Effect |
|---|---|
| **Verify kernel** (LDS-staged Q8_0 mat-vec for the 5-8 token check pass, `ggml/src/ggml-cuda/mmvq.cu`) | one user **+9 to +11%**, output identical |
| **Two-user kernel** (9-14 token passes on the same kernel instead of the tiled fallback) | two users at once **+28 to +34%** total |
| **Two fans, one per card, junction-based curve with smoothing and ramp limits** (`mi50/setup/fan/gpu-fan.sh`) | no thermal throttling: GPU0 81-87 C instead of 99-101 C, +2.5% on long answers |
| `DRAFT_DEV` option in `mi50/run-server.sh` | drafter on either card (same speed, better VRAM balance on GPU1) |

**Start here: [mi50/README.md](mi50/README.md)** (requirements, quick start, settings, benchmarks, docs).

## Optimizations since the first version

After tensor split, every further idea was measured on the same box and settings (results, method and dead ends:
[mi50/docs/optimization-log.md](mi50/docs/optimization-log.md)). Two made it into `main`:

| Change | Effect | Quality |
|---|---|---|
| **Upstream GCN prefill table** ([ggml-org/llama.cpp#30021](https://github.com/ggml-org/llama.cpp/pull/30021)) instead of milpster's vega MMQ table (`-DGGML_CUDA_VEGA_TUNE_MMQ=OFF`) | prompt processing **+8.5-10%**: pp512 332 -> 361, pp4096 357 -> 393, a 20k-token prompt 317 -> 344 t/s | KLD unchanged (0.000596) |
| **Aligned Q8_0 weight reads** in the mat-vec kernel (dword loads + `v_alignbyte` instead of 16-bit load pairs) | generation without the drafter **+5.2%** (28.9 -> 30.4 t/s); with the drafter unchanged | output bit-identical |

Tried and dropped (all measured, details in the log): queuing both GPUs from two CPU threads, mat-vec launch-shape
tuning, the tiled MMQ kernel for 2-8 token batches (2.5x slower), a load-once multi-token mat-vec path, an
LDS-staged multi-token kernel, HIP graphs off, RCCL, plain upstream llama.cpp (crashes with DFlash + tensor split).

Branches that were tried, brought no improvement and were deleted (results are in the log):

| Branch | What it tried | Outcome |
|---|---|---|
| `opt/a1-allreduce-wait` | queue both GPUs' work from two CPU threads | no gain |
| `opt/a2f-mmvq-multicol` | load-once and LDS-staged multi-token mat-vec kernels | 3-10% slower |
| `opt/k2-graph-cache_v6` | checked whether HIP graph recordings are reused | they already are (97%); graphs on or off measure the same |
| `opt/s1-drafting-sweep_v2` | lookup drafters with DFlash2, draft cut-off (p-min), MTP drafting at depth 2-4 | nothing beats DFlash2 depth 4 overall; `ngram-mod` + DFlash2 is +13% only when re-emitting files and 4-7% slower otherwise |
| `dev_v3` | on top of `dev_v2`: the verify kernel adapted to the repacked layout, mx-llama.cpp's whole-token graph capture (measured on its own build), other lane shapes for the 4-8 token kernel, and a runtime overclock | kernel 4-20% slower; token graph about +2% with the drafter, not ported; no better lane shape; overclock +1.5-2% with the drafter, not adopted |

The verify kernel and the two-user kernel are described at the top of this page; before / after for the verify kernel
alone (3 seeds, same build and session): replay 64.0 -> 70.5, short 52.9 -> 58.6, 20k 45.1 -> 49.4, 60k 38.4 -> 41.7.

### `dev_v2`: Q8_0 weight repack

Builds on `dev`. Weights are repacked on the card into two planes (quants, then scales), using the layout and kernels
from [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) (its `q8_repack` code; the Q8_0 path there was contributed by DENEB1312). 3 seeds, same session:

| Test | `dev` | with repack | Change |
|---|---|---|---|
| Prompt processing, 2.5k / 20k / 40k prompt | 309 / 346 / 282 | 366 / 434 / 341 | **+18% / +25% / +21%** |
| Generation: replay / short / 20k / 60k | 70.3 / 58.1 / 50.1 / 42.5 | 72.6 / 58.7 / 53.0 / 41.7 | +3% / same / +6% / same |
| Two users at once, total | 63-68 | 65-67 | same |
| Quality: KLD / same top token / suite | 0.000596 / 99.42% / 40/40 | 0.000596 / 99.42% / 40/40 | same |

VRAM use is unchanged (the repack is done in place). Turn it off with `GGML_CUDA_Q8_REPACK=0`.

### `dev_v2`: probabilistic draft sampling (2026-11-17)

An optional switch, `--spec-draft-sampling probabilistic` (`SAMPLING=probabilistic` with `mi50/run-server.sh`; the default
stays `greedy`). The drafter samples each drafted token at the request's temperature instead of always taking its top
choice, and the target verifies by rejection sampling, which keeps the target's output distribution. It is a port of
upstream llama.cpp [#27694](https://github.com/ggml-org/llama.cpp/pull/27694) and
[#30663](https://github.com/ggml-org/llama.cpp/pull/30663). Same binary, each mode run twice, alternating:

| Test (temperature 0.6) | greedy | probabilistic |
|---|---|---|
| Draft tokens accepted | 67.0% | 69.0% |
| Generation over the whole run | 55.6 t/s | 56.9 t/s (+2.0-2.5%) |
| Generation at 20k / 60k context | 49.3 / 42.1 t/s | 52.1 / 43.9 t/s |
| Generation, short prompt | 57.0 t/s | 56.3 t/s |
| Two users at once, total | 64.8-66.9 t/s | 66.5-69.0 t/s |
| 40-task coding suite | 40/40, 40/40 | 39/40, 39/40 |

Both misses were on one logic puzzle. Run alone 40 times per mode it passed 33 and 34 times, so that is the puzzle's
own failure rate. There is no effect at temperature 0.

### Long conversations and host memory (2026-11-17)

One slot grown step by step to 171k tokens, drafter on, prose replies (code is faster at every size):

| Context | 16k | 35k | 55k | 81k | 111k | 143k | 171k |
|---|---|---|---|---|---|---|---|
| Generation t/s | 39.8 | 38.3 | 36.5 | 33.2 | 30.8 | 29.5 | 29.3 |
| Reading new tokens, t/s | 435 | 370 | 318 | 274 | 233 | 202 | 178 |

Host RAM at 171k: 13.7 GB in total. That needs `--ctx-checkpoints 6 --cache-ram 4096` (the defaults in
`mi50/run-server.sh`): with 30 checkpoints per slot, long agent sessions grew the server to 29 GB and it was killed on
a 30 GB host. Details: [server-flags.md](mi50/docs/server-flags.md#host-memory-system-ram-not-vram).

### Compared with mx-llama.cpp (2026-11-17)

[mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) (`cdc81da`) is the fork the weight repack comes from. Both builds were run back to back on the reference
box with the same model, DFlash2 drafter (depth 4) and settings. Speeds are tokens/s.

| Test | mx-llama.cpp | `dev_v2` |
|---|---|---|
| No drafter: prompt processing, 512 / 4096 tokens | **604 / 576** | 550 / 536 |
| No drafter: generation | **34.2** | 32.2 |
| Drafter, 2 x 65K context: generation, short / 20k / 60k | 52.4 / 46.7 / 39.9 | **59.0 / 53.1 / 42.0** |
| Drafter, 2 x 65K: re-generating an existing file | 67.7 | **73.0** |
| Drafter, 2 x 65K: two users at once, total | 38-40 | **66-68** |
| Drafter, 2 x 65K: prompt processing, 2.5k / 20k / 40k prompt | **385 / 454 / 350** | 367 / 431 / 339 |
| Host RAM after the 2 x 65K run | 22.3 GB | **11.2 GB** |
| One slot of 200K: generation, short / 20k / 60k / 120k | 48.8 / 45.5 / 37.7 / out of host memory | **59.5 / 53.1 / 40.6 / 35.9** |
| One slot of 200K: host RAM | 22.9 GB right after loading | **10.0 GB** at the end of the run |

- mx-llama.cpp is still ahead on prompt processing (3-5% with the drafter, 8-10% without) and on generation without a
  drafter. Two parts of its repack are not ported here: the row padding (it needs a larger buffer than an in-place
  repack has) and the fused single-token path.
- On this box (30 GB of host RAM, two cards without peer-to-peer access) mx-llama.cpp needs
  `-DGGML_CUDA_NO_PEER_COPY=ON`, cannot run 2 x 200K with the drafter, and reserves about 23 GB of host RAM for one
  200K slot. Its 200K numbers were taken with `--cache-ram 0 --ctx-checkpoints 6`. On hosts with more RAM, or cards
  with peer access, its results may differ.
- Method and raw numbers: [mi50/docs/optimization-log.md](mi50/docs/optimization-log.md).

### Branches

| Branch | Contents |
|---|---|
| `main` | first published state: tensor split, prefill table, aligned reads + docs |
| **`dev`** | `main` + the verify kernel (**+9-11% generation**), the two-user kernel (**two users: +28-34% total**) and the two-fan cooling |
| **`dev_v2`** | `dev` + the Q8_0 weight repack (**prompt processing +18-25%**): the current fastest build, in testing, with the full test results on this page |

Each step was developed on its own `opt/<topic>` branch (`opt/a3-upstream-sync`, `opt/a2-mmvq-tune`,
`opt/k1-verify-kernel_v4`, `opt/m1-two-user-batch_v6`). Those branches were folded into `main` and `dev` and are no
longer published; the log still refers to them by name.

## Tested environment

Everything needed to reproduce the results by hand, without the `mi50/` scripts.

**Hardware**

| Part | Tested with | Notes |
|---|---|---|
| GPUs | 2x AMD Instinct MI50 32 GB (gfx906) | two different vBIOS versions; power cap 225 W each |
| PCIe | 4.0 x8 / x8, both on CPU lanes | |
| Board / BIOS | Gigabyte X570 (AM4) | **Above 4G Decoding on, CSM off**, Re-Size BAR off |
| CPU / RAM | AMD Ryzen 9 (12 cores, Zen 2), 32 GB | |
| Display | separate UEFI (GOP) card | needed so the board POSTs with CSM off (no iGPU) |

**Software**

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 LTS |
| Kernel | 7.0 (HWE) |
| Kernel parameters | `amdgpu.rebar=0 amdgpu.runpm=0` (+ `processor.max_cstate=1` only for an idle-freeze fix) |
| GPU driver | in-kernel `amdgpu` (no amdgpu-dkms) |
| ROCm | 7.1.1 (`rocm 7.1.1.70101-38~24.04`) from `repo.radeon.com/rocm/apt/7.1.1` + `amdgpu/30.20.1` repos |
| HIP / rocBLAS / hipBLAS | 7.1.52802 / 5.1.1 / 3.1.0 |
| Compiler | ROCm clang 20.0.0git (roc-7.1.1), gcc 13.3.0, cmake 3.28.3 |
| llama.cpp source | this repo = milpster/gfx906-llama-cpp `1373e889` + the changes under [What this repo adds](#what-this-repo-adds) |

**Models** (exact files, sha256 in [mi50/models.sha256](mi50/models.sha256))

| File | Hugging Face repo @ revision |
|---|---|
| `Qwen3.8-27B-Q8_0.gguf` + `mmproj-Qwen3.8-27B-Q8_0.gguf` | `ggml-org/Qwen3.8-27B-GGUF` @ `0669b98607d4` |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` (drafter) | `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f8ce46` |

**Build** (cmake options)

```
-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906
-DCMAKE_HIP_COMPILER=/opt/rocm-7.1.1/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm-7.1.1
-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
-DGGML_CUDA_VEGA_TUNE=ON -DGGML_CUDA_VEGA_TUNE_FATTN=ON -DGGML_CUDA_VEGA_TUNE_MMQ=OFF
-DGGML_CUDA_VEGA_TUNE_TOPK=ON -DGGML_CUDA_VEGA_TUNE_GRAPHS=ON
-DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON
-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
```

**Run**

| Setting | Value |
|---|---|
| Environment | `HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=0,1 HSA_XNACK=0 LLAMA_DFLASH_MIRROR_OUTPUT=1 GGML_CUDA_FATTN_PATH=force_convert GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 GPU_SINGLE_ALLOC_PERCENT=100 GPU_MAX_ALLOC_PERCENT=100 HSA_ENABLE_SDMA=1 HIP_GRAPH=1` |
| Split | `--device ROCm0,ROCm1 -sm tensor -mg 0 --pipeline-parallel off` |
| Drafter | `-md Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4 --spec-draft-override-tensor .*=ROCm0 --spec-draft-device ROCm0 -ngld 99` |
| Context | `--parallel 2 -c 409600` (2 x 204,800), f16 KV cache |
| Batching | `-fa on -ngl 999 -b 2048 -ub 384 --ctx-checkpoints 6 --cache-ram 4096` (the last two limit host RAM, see [server-flags.md](mi50/docs/server-flags.md#host-memory-system-ram-not-vram)) |
| Loading | `--load-mode none` (this fork has no `--no-mmap`) |
| Model / sampling | `--jinja --reasoning on --reasoning-budget 8192 --chat-template-kwargs '{"reasoning_effort":"medium"}' --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` |

The full command is in [mi50/run-server.sh](mi50/run-server.sh); every flag is explained in
[mi50/docs/server-flags.md](mi50/docs/server-flags.md).

## Credit

This is not original llama.cpp work. Almost all of the code comes from these projects:

- **[milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp)**: the gfx906 (Radeon VII / MI50 / MI60)
  fork with wave64 kernel tuning, DFlash2 speculative decoding and the drafter output-head mirror. This repo is a
  snapshot of it at commit
  [`1373e889`](https://github.com/milpster/gfx906-llama-cpp/commit/1373e889a4d5c69f453fe82c96332585f3d2b960).
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**, which milpster's fork is built on.
- **[mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp)**: the Q8_0 weight repack layout and its mat-vec and
  tiled kernels (`ggml/src/ggml-cuda/rp-common.cuh`, `rp-kernels.cuh`; the Q8_0 path there was contributed by
  DENEB1312). MIT licensed. Used on the `dev_v2` branch.

All are MIT licensed. The [LICENSE](LICENSE), [AUTHORS](AUTHORS) and [licenses/](licenses/) are kept unchanged.

## What this repo adds

- **Source changes** (all measured):
  - **Upstream GCN prefill table** (`ggml/src/ggml-cuda/mmq-config-gcn.cuh` from ggml-org/llama.cpp#30021), used via
    `-DGGML_CUDA_VEGA_TUNE_MMQ=OFF` (prefill +8.5-10%).
  - **Aligned Q8_0 weight reads** in `ggml/src/ggml-cuda/vecdotq.cuh` / `mmvq.cu` (generation +5% without drafter,
    bit-identical), plus mat-vec tuning knobs at the top of `mmvq.cu`.
  - **Verify kernel and two-user kernel** in `ggml/src/ggml-cuda/mmvq.cu` (LDS-staged Q8_0 mat-vec for 5-8 token
    passes; 9-14 token passes as two launches of it): one user +9-11%, two users at once +28-34%.
  - **In-place Q8_0 weight repack** (`ggml/src/ggml-cuda/rp.cu`, written for this fork; the layout and the kernels in
    `rp-common.cuh` / `rp-kernels.cuh` come from mx-llama.cpp): prompt processing +18-25%. With the repack on, 2-16
    token passes run mx-llama.cpp's mat-vec kernel in place of the verify and two-user kernels above, which stay
    as the path for `GGML_CUDA_Q8_REPACK=0`.
- **One fix** in `src/llama-context.cpp`: the DFlash2 drafter's output-head mirror verified its copy with
  partial reads, which tensor-split buffers don't support, so `-sm tensor` with a drafter aborted at startup. It now
  verifies with one full read. Also available as [mi50/mirror-verify-tensor-split.patch](mi50/mirror-verify-tensor-split.patch).
- **[mi50/](mi50/)**: OS and ROCm setup, build script, pinned model downloads, the tested server settings, a systemd
  unit, benchmark and quality tools, and docs.

## What was left out

To keep the repo small, only what's needed to build and run the server on MI50s is included (~56 MB, no history):

- milpster's own test scripts, journals, experiment patches, chat templates and results files;
- upstream's tests and test vocab files, examples, docs, Python converters (`convert_*.py`, `gguf-py`), CI and
  packaging files.

Get any of those from milpster's fork or upstream llama.cpp. To convert models yourself, use upstream llama.cpp's tools.
