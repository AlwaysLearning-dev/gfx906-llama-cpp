# Server flags and environment, explained

What `mi50/run-server.sh` runs, and why each setting is there.

## Build

`mi50/build.sh` configures with:

```
-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 -DCMAKE_HIP_COMPILER=/opt/rocm-7.1.1/lib/llvm/bin/clang++
-DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_VULKAN=OFF
-DGGML_CUDA_VEGA_TUNE=ON (+ _FATTN, _MMQ, _TOPK, _GRAPHS)
-DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON
```

The upstream fork's own scripts pin ROCm 6.1 with a private rocBLAS build; stock ROCm 7.1.1 with Vulkan off works.
Building one tool on its own can relink the shared libraries, so `build.sh` builds `llama-server` last.

## Models

| File | Source | Role |
|---|---|---|
| `Qwen3.8-27B-Q8_0.gguf` (28.6 GB) | [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) @ `0669b986` | target |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` (629 MB) | same | vision |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) @ `2d9571f8` | drafter |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (optional) | same | smaller drafter, identical output |

- **ggml-org GGUFs only.** unsloth and lmstudio Qwen3.8 builds embed an MTP layer this fork can't load
  (`missing tensor 'blk.64.ssm_conv1d.weight'`).
- **The Q8 is pinned** to the revision that was tested. ggml-org re-uploaded it later with a different checksum.
- **Why Q8_0:** Q6_K was no faster on gfx906; Q4 variants were faster but noticeably worse in real use.
- **The drafter's quant doesn't affect output:** the target checks every drafted token. Q4 and Q8 drafters gave
  byte-identical text.

## Flags

| Flag | Value | Why |
|---|---|---|
| `-m`, `--mmproj` | Q8_0 + mmproj | Full-quality target, image input |
| `--load-mode none` | | This fork rejects `--no-mmap`; this is its equivalent |
| `--jinja --reasoning on` | | Qwen3.8 chat template with thinking |
| `--chat-template-kwargs` | `{"reasoning_effort":"medium"}` | `high`/`xhigh` think far longer and feel slow |
| `--reasoning-budget` | 8192 | Cap on thinking tokens per reply. With 32768 a reply that loops in its thinking can run for 10 minutes; coding tasks in a test used under 3,000 |
| `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0` | | Qwen's recommended sampling for thinking mode |
| `--parallel 2 -c 409600` | | Two slots of 204,800 context each |
| `-fa on -ngl 999` | | Flash attention, all layers on GPU |
| `-b 2048 -ub 384` | | `-ub 1024` cut prefill 20% and left GPU0 with no headroom |
| `--ctx-checkpoints 6` | | Resume points for the hybrid (gated delta-net) layers, kept in host RAM at about 330 MB each. One is added per turn of a conversation. With the default of 30, a long agent session held about 10 GB per slot and the server was killed for running out of host memory (see below) |
| `--cache-ram 4096` | | Host RAM limit (MiB) for saved conversations of idle slots. The default is 8192 |
| `--device ROCm0,ROCm1 -sm tensor` | | Tensor split across both cards |
| `-mg 0 --pipeline-parallel off` | | Main GPU 0, no pipeline parallelism |
| `-md <drafter>` | | DFlash2 drafter |
| `--spec-type draft-dflash --spec-draft-n-max 4` | | 4 drafted tokens per step (3, 5, 6 all slower) |
| `--spec-draft-sampling probabilistic` | | Optional (`SAMPLING=probabilistic` in `run-server.sh`; default `greedy`). The drafter samples each drafted token at the request's temperature and the target verifies by rejection sampling, so the output distribution stays the target's. Measured: draft acceptance 67.0% to 69.0%, generation +2.0-2.5% overall and +4-6% at 20k-60k prompts, no change on short prompts or at temperature 0. Port of upstream llama.cpp #27694 and #30663 |
| `--spec-draft-override-tensor .*=ROCm0 --spec-draft-device ROCm0 -ngld 99` | | Drafter fully on GPU0 |
| KV cache | f16 (default) | q8_0 KV was slower at every depth on gfx906 |

## Host memory (system RAM, not VRAM)

The reference box has 30 GB of RAM. With `--ctx-checkpoints 30` and the default prompt cache, `llama-server` grew to
about 29 GB during long agent sessions and the kernel killed it twice. Measured on this model:

| What | Size | Controlled by |
|---|---|---|
| Context checkpoint | about 330 MB, one per turn of a conversation | `--ctx-checkpoints` (per slot) |
| Saved conversation of an idle slot | about 350 MB for a short chat, 64 KiB per token of KV plus the slot's checkpoints | `--cache-ram` |
| Pinned staging buffers (ROCm) | 1.7 GB at start; grows in doubling steps with long prompts, never released | - |

With the settings above, one conversation grown to 171k tokens used 13.7 GB of host RAM in total. Fewer checkpoints
only cost anything when a client rewrites old parts of a conversation: the server then re-reads from an older point.

## Environment

| Variable | Why |
|---|---|
| `HSA_OVERRIDE_GFX_VERSION=9.0.6` | gfx906 is outside ROCm 7's official list |
| `HIP_VISIBLE_DEVICES=0,1` | The two MI50s (set it if other GPUs are present) |
| `HSA_XNACK=0` | No XNACK on these cards |
| `LLAMA_DFLASH_MIRROR_OUTPUT=1` | Gives the drafter a GPU-local copy of the target's output head (needs the patch under tensor split) |
| `GGML_CUDA_FATTN_PATH=force_convert`, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` | Fork's tested gfx906 paths |
| `GPU_SINGLE_ALLOC_PERCENT=100`, `GPU_MAX_ALLOC_PERCENT=100` | Allow large single allocations |
| `HSA_ENABLE_SDMA=1`, `HIP_GRAPH=1` | Tested settings (HIP graphs on vs off measured the same) |

## VRAM

With these settings GPU0 (drafter + mirror + half the model + half the KV) has about **1.0 GB free**; GPU1 has
more. If GPU0 runs out, use `DRAFT=q4` (0.9 GB smaller) or lower `CTX`.

## Card-to-card sync under tensor split

| `GGML_CUDA_ALLREDUCE` | Result |
|---|---|
| unset (internal, host-staged) | 29.0 t/s raw decode, the default |
| RCCL build (`-DGGML_HIP_RCCL=ON`) | 28.8 t/s, no gain |
| `none` (butterfly fallback) | crashes |

## Speculative decoding behaviour

Acceptance depends on content: ~70% on normal prose and code, 80-88% when re-emitting existing text, under 40%
on unpredictable output (for example generated HTML/JS effects). Speed tracks acceptance: check
`draft_n_accepted / draft_n` in a response's `timings`. Output isn't guaranteed byte-identical to no-drafter
decoding (the batched verify can flip near-tie tokens); quality measured the same.
