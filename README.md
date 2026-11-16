# gfx906-llama-cpp (dual MI50 tensor split)

llama.cpp for **two AMD Instinct MI50 32 GB** cards (gfx906): Qwen3.8-27B at full **Q8_0** with a DFlash2
speculative drafter and **tensor split** across both cards, at **~50 tokens/s** average generation, same quality.

| Avg generation over a 40-task suite, Qwen3.8-27B Q8_0 | t/s |
|---|---|
| gfx906-turbo, layer split, no drafter | 19.0 |
| milpster gfx906 fork + DFlash2 drafter, layer split | 36.3 |
| **this repo: + tensor split** | **49.6** |

Generation speed by how much context is already in the conversation (this repo, 3-seed averages):

| Context | short prompt | 20k | 40k | 60k |
|---|---|---|---|---|
| t/s | 50.5 | 43.0 | 38.5 | 36.3 |

**Start here: [mi50/README.md](mi50/README.md)** (requirements, quick start, settings, benchmarks, docs).

## Optimizations since the first version

After tensor split, every further idea was measured on the same box and settings (results, method and dead ends:
[mi50/docs/optimization-log.md](mi50/docs/optimization-log.md)). Two made it in:

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
| `opt/s1-drafting-sweep_v2` | lookup drafters with DFlash2, draft cut-off (p-min), MTP drafting at depth 2-4 | nothing beats DFlash2 depth 4 overall; `ngram-mod` + DFlash2 is +13% only when re-emitting files and 4-7% slower otherwise |

With the drafter on, generation time is dominated by the 5-token verify pass. A second, redesigned LDS-staged kernel
for that pass is on the branch `opt/k1-verify-kernel_v4` (not in `main` yet):

| Generation with drafter, 3 seeds | replay | short | 20k | 60k |
|---|---|---|---|---|
| kernel off | 64.0 | 52.9 | 45.1 | 38.4 |
| **kernel on** | **70.5** | **58.6** | **49.4** | **41.7** |

That is **+9 to +11%**, with identical temp-0 output at draft depths 4-7. Two users at once are unchanged (~25 t/s each).

### Branches

| Branch | Contents |
|---|---|
| `main` | everything kept (tensor split, prefill table, aligned reads) + docs |
| `opt/a3-upstream-sync` | the upstream GCN prefill table on its own |
| `opt/a2-mmvq-tune` | `opt/a3-upstream-sync` + aligned Q8_0 reads + mat-vec tuning knobs (`ggml/src/ggml-cuda/mmvq.cu`) |
| `opt/t1-thermal-balance` | `main` + drafter device option (`DRAFT_DEV`), two-fan junction-based fan script |
| `opt/t2-fan-smoothing_v3` | previous + fan smoothing (peak hold) |
| `opt/k1-verify-kernel_v4` | previous + LDS-staged Q8_0 mat-vec for the 5-8 token verify pass (**+9-11% generation**) |
| `opt/t3-fan-ramp_v5` | previous + smooth fan curve with ramp limits (quieter ramp, same temperatures) |

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
| llama.cpp source | this repo = milpster/gfx906-llama-cpp `1373e889` + the tensor-split fix |

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
| Batching | `-fa on -ngl 999 -b 2048 -ub 384 --ctx-checkpoints 30` |
| Loading | `--load-mode none` (this fork has no `--no-mmap`) |
| Model / sampling | `--jinja --reasoning on --reasoning-budget 32768 --chat-template-kwargs '{"reasoning_effort":"medium"}' --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` |

The full command is in [mi50/run-server.sh](mi50/run-server.sh); every flag is explained in
[mi50/docs/server-flags.md](mi50/docs/server-flags.md).

## Credit

This is not original llama.cpp work. Almost all of the code comes from two projects:

- **[milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp)**: the gfx906 (Radeon VII / MI50 / MI60)
  fork with wave64 kernel tuning, DFlash2 speculative decoding and the drafter output-head mirror. This repo is a
  snapshot of it at commit
  [`1373e889`](https://github.com/milpster/gfx906-llama-cpp/commit/1373e889a4d5c69f453fe82c96332585f3d2b960).
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**, which milpster's fork is built on.

Both are MIT licensed. Their [LICENSE](LICENSE), [AUTHORS](AUTHORS) and [licenses/](licenses/) are kept unchanged.

## What this repo adds

- **Source changes** (all small, all measured):
  - **Upstream GCN prefill table** (`ggml/src/ggml-cuda/mmq-config-gcn.cuh` from ggml-org/llama.cpp#30021), used via
    `-DGGML_CUDA_VEGA_TUNE_MMQ=OFF` (prefill +8.5-10%).
  - **Aligned Q8_0 weight reads** in `ggml/src/ggml-cuda/vecdotq.cuh` / `mmvq.cu` (generation +5% without drafter,
    bit-identical), plus mat-vec tuning knobs at the top of `mmvq.cu`.
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

Get any of those from the two projects above. To convert models yourself, use upstream llama.cpp's tools.
