# Performance: results, how to reproduce, quality checks, profile

All numbers: two MI50 32 GB, ROCm 7.1.1, Qwen3.8-27B **Q8_0** target in every row, measured November 2026.

## Headline

| Average over the 40-task suite | t/s | Score |
|---|---|---|
| gfx906-turbo, layer split, no drafter | 19.0 | 39/40 |
| milpster + DFlash2 Q8 drafter (depth 4), layer split | 36.3 | 40/40 |
| **milpster + DFlash2 Q8 drafter (depth 4), tensor split** | **49.6** | **40/40** |

## Speed by context depth

`bench2.py`, 3 seeds, 768 generated tokens (1536 for replay), temperature 0.6, prompt cache on.

| Generation t/s | turbo | milpster, no drafter | + drafter, layer | **+ drafter, tensor** |
|---|---|---|---|---|
| file replay (re-emit a long file) | - | - | 43.7 | **62.8** |
| short prompt | 19.5 | 18.7 | ~36 | **50.5** |
| 20k context | 18.3 | 17.6 | 30.5 | **43.0** |
| 40k context | - | - | - | **38.5** |
| 60k context | 15.4 | 14.9 | 26.5 | **36.3** |

| Raw (`llama-bench`, no drafter) | layer | tensor | tensor + prefill table + aligned reads |
|---|---|---|---|
| tg128 | 18.8 | **28.9** | **30.4** |
| pp512 | 236 | **336** | **361** |
| pp4096 | - | 357 | **393** |

Two users at once (`np2-check.py`): layer ~18 t/s each, **tensor ~25 t/s each** (single user 48.6).

## Reproduced from a clean clone

The repo itself was re-tested on 2026-11-15: a fresh clone, built with `mi50/build.sh`, run with `mi50/run-server.sh`
defaults (tensor split, Q8 drafter, depth 4, 2 x 204,800 context) on the same two MI50s:

| Check | Clean clone | Original measurement |
|---|---|---|
| file replay / short / 20k (2 seeds) | 62.2 / 50.8 / 44.7 t/s | 62.8 / 50.5 / 43.0 |
| two users at once | 22.6 + 23.5 t/s | ~25 each |
| single user, long answer | 48.7 t/s | 48.6 |
| answers (arithmetic, code), image input | correct | correct |

## Sweeps (what didn't help)

| Under tensor split, 2 seeds | d3 | **d4** | d5 | d6 | d4, HIP graphs off |
|---|---|---|---|---|---|
| replay | 56.2 | **62.9** | 61.1 | 62.4 | 62.5 |
| short | 48.7 | **51.3** | 47.9 | 46.2 | 51.4 |
| 20k | 43.7 | **44.4** | 41.5 | 39.7 | 44.7 |

Earlier sweeps (layer split, still valid conclusions):
- `-ub 1024`: prefill -20%, no VRAM left on GPU0.
- q8_0 KV cache: slower at every depth (-14% at 64k).
- n-gram + DFlash combos: +27% on file replay but -6 to -8% on new text. Plain DFlash depth 4 is the best overall.
- Shorter context (`-c`) doesn't speed up generation; only the tokens actually in the context matter.
- RCCL instead of the internal AllReduce: no change. Q4 vs Q8 drafter: same speed, identical text.

## Quality checks

Two independent checks, both run for each server change:

1. **KL divergence** vs a fixed baseline: `llama-perplexity` on a 1.5 MB code corpus (llama.cpp's own C++ sources,
   40 chunks x 512 tokens), first saving the baseline logits with `--kl-divergence-base`, then comparing with
   `--kl-divergence`. The numbers below used the gfx906-turbo fork as the baseline; `mi50/bench/kld.sh` compares
   tensor split against layer split of this build.

   | vs turbo baseline | mean KLD | same top token | max KLD |
   |---|---|---|---|
   | milpster, layer split | 0.000593 | 99.34% | 0.178 |
   | **milpster, tensor split** | **0.000596** | **99.42%** | **0.094** |

   For scale, going from Q8_0 to Q4_K_M weights is roughly 20x larger. These differences are rounding noise.
2. **Task suite** (`mi50/bench/qual-suite.py`): 8 tasks x 5 runs at medium reasoning: sigma-rule math, JSON extraction,
   primes, quicksort, a bug fix, a weather tool call, a logic puzzle, arithmetic. Auto-graded. 40/40 for both
   milpster configs.

## Reproducing

Production uses both GPUs, so stop any running server first. From the repo root:

```bash
PORT=8090 mi50/run-server.sh &                              # SM=layer|tensor, DRAFT=q8|q4|none, NMAX=4
REPS=3 python3 mi50/bench/bench2.py 8090 mylabel -1 0 20000 60000 > results.jsonl   # depth -1 = file replay
python3 mi50/bench/np2-check.py 8090                        # two concurrent users
python3 mi50/bench/qual-suite.py 8090 mylabel 5 suite.json  # 40-task suite
mi50/bench/sweep.sh                                         # layer vs tensor vs no drafter, one go
mi50/bench/kld.sh                                           # KL divergence, tensor vs layer
# raw decode/prefill without the server
build-mi50/bin/llama-bench -m ~/models/Qwen3.8-27B-Q8_0.gguf -ngl 999 -fa 1 -ub 384 -sm tensor -p 512 -n 128
```
(`llama-bench` needs the same `HSA_OVERRIDE_GFX_VERSION=9.0.6` and `LD_LIBRARY_PATH` as `run-server.sh`.)

## Where the time goes now

`rocprofv3 --kernel-trace` on llama-bench decode with tensor split. Set `GGML_CUDA_DISABLE_GRAPHS=1` first: the
profiler segfaults while HIP graphs are being captured. Per token, per card:

| Part | Kernels | Time | Notes |
|---|---|---|---|
| Q8_0 mat-vec (weights) | 433 | ~24 ms | ~595 GB/s, **74%** of the ~800 GB/s a plain read kernel reaches |
| Small kernels (norms, adds, quantize, rope, delta-net...) | 1,612 | ~7.9 ms | ~4-5 us each, near the launch minimum; common pairs are already fused |
| AllReduce | 128 | 5.9 ms on GPU0, 1.3 ms on GPU1 | GPU0 arrives first and spin-waits for GPU1 |

Both cards run at their 225 W power cap during decode, with clocks between 1485 and 1725 MHz. The cap, not the
clock policy, is the limit. What's left needs kernel work: see [roadmap.md](roadmap.md).

## Measuring your cards' bandwidth

A 2 GB read kernel (HIP, one `int4` load per thread per iteration, 240-3840 blocks) reached 792-808 GB/s on each
reference card. Token generation can't beat bandwidth divided by bytes per token, so measure it to know your ceiling.
