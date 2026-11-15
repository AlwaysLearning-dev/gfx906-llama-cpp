# Roadmap: what could still make it faster

Ideas ranked by expected gain per hour of work, with the measurements behind them in [performance.md](performance.md).
None are implemented yet.

**Rule for every item:** keep the same settings (Q8_0 target, Q8 DFlash2 drafter at depth 4, f16 KV, `-c 409600`,
`--parallel 2`). A change only counts if it beats today's baseline on the same benchmark and passes the quality checks.

---

## Today's baseline (to beat)

| Benchmark | Command / script | Result (2026-11-14, tensor split) |
|---|---|---|
| Raw decode, no drafter | `llama-bench -sm tensor -fa 1 -ub 384 -p 512 -n 128` | tg 28.9, pp 336 t/s |
| Server + drafter, 3 seeds | `REPS=3 python3 mi50/bench/bench2.py 8090 <label> -1 0 20000 60000` | 62.8 / 50.5 / 43.0 / 36.3 t/s |
| Two users at once | `python3 mi50/bench/np2-check.py 8090` | 25.2 + 25.0 t/s |
| Quality: KLD vs turbo | `mi50/bench/kld.sh` | mean 0.000596, same-top 99.42% |
| Quality: task suite | `python3 mi50/bench/qual-suite.py 8090 <label> 5 out.json` | 40/40 |

Run test servers with `PORT=8090 mi50/run-server.sh`. Stop production first (`sudo systemctl stop llama-server`):
the model fills both GPUs.

## Where the time goes now (profile, per token per card, no drafter)

| Part | Kernels | Time | Headroom |
|---|---|---|---|
| Q8 matmuls (reading the weights) | 433 | ~24 ms | runs at ~595 GB/s = **74%** of the measured 807 GB/s peak |
| Small kernels (norms, adds, quantize, rope, delta-net...) | 1,612 | ~7.9 ms | each ~4-5 us, close to the GPU's minimum. Common pairs are already fused |
| AllReduce (two per layer, card-to-card sync) | 128 | 5.9 ms GPU0 / 1.3 ms GPU1 | GPU0 spin-waits for GPU1, and the spin burns power |

Both cards sit at their **225 W cap** (the maximum) during decode, with clocks dipping from 1725 to 1485-1606 MHz.
Profiled with `rocprofv3 --kernel-trace` and `GGML_CUDA_DISABLE_GRAPHS=1` (the profiler crashes with HIP graphs on).

---

## A. Speed (software)

### A1. Make the AllReduce wait less wasteful (small, worth trying first)
- **What:** the fork's internal AllReduce (`ggml/src/ggml-cuda/allreduce.cu`) busy-spins on a host-memory flag
  while waiting for the other card. Add `__builtin_amdgcn_s_sleep(n)` inside the spin loop, or tune the
  chunked-kernel vs copy-engine size threshold.
- **Why:** both cards are power-capped. Spinning at full clock steals watts from the matmuls on the same card.
- **Expected:** +2-6% if clocks hold higher. **Effort:** a few hours. **Risk:** low; easy to revert.
- **Also check:** why GPU0 always arrives first. It could be launch order in the meta backend (GPU0's kernels are
  queued before GPU1's in every segment) or the vBIOS difference (see B3).

### A2. Tune the Q8_0 mat-vec kernel for gfx906 (medium)
- **What:** `mul_mat_vec_q<Q8_0>` (`ggml/src/ggml-cuda/mmvq.cu`) reaches 74% of peak bandwidth. Try rows per block,
  warps per block, and wider loads for the decode shapes (5120 x 17408 etc., batch 1 and the drafter's verify batch
  of 5). Measure with `test-backend-ops perf -o MUL_MAT` on the real shapes, then llama-bench.
- **Expected:** reaching 85% would save ~3 ms/token, about **+8-12%**. **Effort:** 1-3 days. **Risk:** low, gated by
  the KLD check. The fork's VEGA_TUNE tables currently cover mmq (prefill) and flash attention, not mmvq.

### A3. Sync the fork with newer upstream tensor-parallel work (medium, recurring)
- **Upstream PRs the author parked** because their rig uses layer split, but which matter for ours:
  - **#27825**: HIP-native AllReduce.
  - **#28390**: drafter on the meta backend. The drafter would also split across both cards, which frees GPU0 VRAM
    (it has ~1.0 GB free today, the tightest margin in the stack).
- **How:** merge upstream (or a newer milpster commit), keep the mirror-verify fix, rebuild in a *separate* build
  dir, run the baseline table above, then switch.
- **Expected:** unknown; could be anything from 0 to +10%. **Effort:** half a day per sync plus testing.
- **Risk:** medium (this fork moves fast). Keep the previous build dir as the rollback.

### A4. Fewer, bigger kernels (large)
- **What:** fuse more of the 1,612 small kernels per token: `quantize_q8_1` into the matmul prologue, the
  delta-net pre/post ops, and add + norm across the tensor-split segment boundaries.
- **Expected:** saving 3-4 ms is about **+10-15%**. **Effort:** days to weeks. **Risk:** medium (correctness).
  Best done upstream, or after A3, so the work isn't lost at the next sync.

### A5. Re-test n-gram drafting under tensor split (small, workload-specific)
- In the October sweep, `ngram-mod + dflash` was +27% on file replay but 6-8% slower on new text. With tensor
  split the balance may shift. Worth one sweep only if most of your use is Claude Code editing existing files.

### Already tested, no gain (don't repeat)
- Drafter depth 3, 5 or 6 (4 wins), HIP graphs off (noise), RCCL (`src/build-rccl`: same speed).
- `GGML_CUDA_ALLREDUCE=none` (crashes), `-ub 1024` (slower prefill, no VRAM), q8_0 KV cache (slower, rejected).
- GPU clock mode `high` (moot: the cards are power-capped, not clock-ramp limited).

---

## B. Speed (hardware)

### B1. Infinity Fabric (xGMI) bridge (speculative)
- MI50s have IF-link connectors. A 2-card bridge would replace the PCIe 4.0 x8, host-staged AllReduce with a
  direct link and largely remove the 5.9 ms sync cost (**+10-15%** if it works).
- **Unknown:** whether the bridge is obtainable and works on this board and kernel. Research before buying.

### B2. Platform (not recommended for speed alone)
- Both cards are on CPU lanes at PCIe 4.0 x8/x8, which is fine. A board with x16/x16 would help prefill a little,
  not decode.

### B3. vBIOS (not recommended)
- GPU0 and GPU1 run different vBIOS versions ("AMD Radeon Graphics" and "MI50/MI60"). Bandwidth measured the same
  (792 vs 808 GB/s). Flashing a different vBIOS for a higher power cap risks bricking a card. Not worth it.

---

## C. Reliability

- **C1. Long-run stability.** Upstream marks `-sm tensor` EXPERIMENTAL. Watch `journalctl -u llama-server` for
  crashes. Fallback: `SM=layer` (still about 36 t/s with the drafter).
- **C2. Deep context.** Tested to 60k context with the drafter. A 150k+ session hasn't been measured.
- **C3. Warm-up.** The first request after a start is slow; a warm-up request in `ExecStartPost` would hide it.
- **C4. Claude Code system messages.** Qwen3.8's chat template rejects `role=system` entries inside `messages[]`,
  which Claude Code sends. The reference box works around it with a small proxy; fixing it in the server's template
  handling would remove the need.
