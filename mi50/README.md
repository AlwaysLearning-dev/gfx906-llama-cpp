# gfx906-mi50: Qwen3.8-27B Q8_0 at ~58 tokens/s on two AMD MI50 32 GB

This snapshot of [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) is tuned for one setup:
**two matched AMD Instinct MI50 32 GB cards** running **Qwen3.8-27B at full Q8_0** with a DFlash2 speculative
drafter and **tensor split** across both cards.

Everything in this folder is what makes it work: the source fix, build script, model downloads with checksums,
the exact server settings, a systemd unit, OS and ROCm setup, and the benchmark and quality tools used to measure it.

## Results

Same 40-task suite (coding, JSON, tool calls, logic, arithmetic; 5 runs each), Qwen3.8-27B Q8_0 in every row:

| Setup | Avg t/s | Score |
|---|---|---|
| gfx906-turbo fork, layer split, no drafter (well-tuned baseline) | 19.0 | 39/40 |
| milpster fork + DFlash2 drafter, layer split | 36.3 | 40/40 |
| this repo, `main`: milpster + DFlash2 drafter + tensor split | 49.6 | 40/40 |
| this repo, `dev`: + verify kernel, two-user kernel, two-fan cooling | 58.1 | 40/40 |
| **this repo, `dev_v2`: + Q8_0 weight repack (prompt processing +18-25%)** | **58.5** | **40/40** |

| Generation t/s, drafter at depth 4 | layer split | tensor split (`main`) | `dev` | **`dev_v2`** |
|---|---|---|---|---|
| short prompt | ~36 | 50.5 | **55.8** | **58.7** |
| 20k context | 30.5 | 43.0 | **50.4** | **53.0** |
| 40k context | not measured | 38.5 | not measured | not measured |
| 60k context | 26.5 | 36.3 | **42.7** | **41.7** |
| re-generating an existing file | 43.7 | 62.8 | **69.9** | **72.6** |
| two users at once | ~18 each | 25 each | **31-34 each** | **32-35 each** |
| raw decode, no drafter (llama-bench tg128) | 18.8 | 28.9 (30.4 with aligned reads) | 30.4 | 30.7 |
| prefill (llama-bench pp512) | 236 | 336 (361 with the GCN prefill table) | 361 | 545 with the default batch (`dev`: 430 with the same command) |
| prompt processing, 20k-token prompt (server) | not measured | 344 | 346 | **434** |

Quality is unchanged: KL divergence vs the baseline is 0.0006 (same top token 99.4% of the time) for both split
modes. Details: [docs/performance.md](docs/performance.md).

## What this repo changes

1. **One source fix** ([mirror-verify-tensor-split.patch](mirror-verify-tensor-split.patch), `src/llama-context.cpp`).
   With `-sm tensor` plus the DFlash2 drafter, startup aborted in `init_mirror_output` with
   `ggml-backend-meta.cpp:1631: GGML_ASSERT(size % chunk_size_full == 0)`. The drafter's output-head mirror was
   verified with two 4 KiB partial reads, and tensor-split ("meta") buffers only support whole-tensor reads. It now
   verifies with one full read. Layer split behaves the same as before.
2. **The `mi50/` folder** (this one): scripts, settings, docs. Nothing else in the source is changed.

### Why tensor split is faster

Token generation reads every weight once per token, so it's limited by memory bandwidth. With layer split
(llama.cpp's default) the two cards take turns: each token reads all 28.6 GB at **one** card's ~800 GB/s. With
tensor split each card holds half of *every* layer and both read at the same time, joined by two small syncs per
layer over PCIe. Qwen3.8-27B's 4 KV heads split cleanly 2+2 across two identical cards.

The upstream fork author tested tensor split on a 3-GPU ROCm + Vulkan rig, where 4 heads can't divide evenly and
the output was garbage. That problem doesn't apply to two matched cards.

## Requirements

- **2x AMD Instinct MI50 32 GB** (gfx906). Passive cards: **they need added airflow** ([fan guide](docs/gpu-fan-control-guide.md)).
- A board with **Above 4G Decoding on and CSM off**, and a UEFI display card if the CPU has no iGPU
  ([hardware-and-bios.md](docs/hardware-and-bios.md)).
- **Ubuntu 24.04**, **ROCm 7.1.1** (in-kernel amdgpu, no amdgpu-dkms).
- ~35 GB disk for models, a few GB for the build.
- Internet during the build: llama.cpp's built-in web UI is downloaded from ggml-org's Hugging Face bucket and
  checksum-verified.


## Quick start

```bash
git clone <this repo> ~/gfx906-mi50 && cd ~/gfx906-mi50
sudo mi50/setup/01-os.sh            # kernel parameters + build tools (FREEZE_FIX=1 only if you get idle freezes)
sudo mi50/setup/02-rocm.sh          # ROCm 7.1.1
sudo reboot
rocm-smi --showproductname          # expect two MI50s
cd ~/gfx906-mi50
mi50/build.sh                       # -> build-mi50/bin/llama-server
mi50/download-models.sh             # ~31 GB to ~/models, pinned revisions, sha256-checked
mi50/run-server.sh                  # http://127.0.0.1:8080 (built-in web UI + OpenAI/Anthropic APIs)
```

To run it as a service (binds `0.0.0.0:8080`, so firewall it to your LAN):

```bash
sudo -E mi50/systemd/install-service.sh
```

The model takes 1-2 minutes to load, and the first request after a start is slower (warm-up).

## Server settings

`run-server.sh` holds the tested settings. Override any of them with environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SM` | `tensor` | `tensor` or `layer` split |
| `DRAFT` | `q8` | `q8`, `q4` (identical output, 0.9 GB less VRAM on GPU0) or `none` |
| `NMAX` | `4` | drafted tokens per step (3, 5 and 6 were all slower) |
| `CTX` / `NP` | `409600` / `2` | total context / parallel slots (2 x 204,800) |
| `EFFORT` | `medium` | Qwen3.8 reasoning effort (`high`/`xhigh` think much longer) |
| `HOST` / `PORT` | `127.0.0.1` / `8080` | bind address |
| `MODELS` | `~/models` | model directory |

Extra arguments are passed to llama-server. Every flag is explained in [docs/server-flags.md](docs/server-flags.md).

## Verify it yourself

Stop any running server first: the model fills both GPUs.

```bash
PORT=8090 mi50/run-server.sh &                         # test instance
python3 mi50/bench/bench2.py 8090 mytest -1 0 20000    # speed: replay, short, 20k context
python3 mi50/bench/np2-check.py 8090                   # two users at once
python3 mi50/bench/qual-suite.py 8090 mytest 5 out.json   # 40-task quality suite
mi50/bench/sweep.sh                                    # layer vs tensor vs no drafter
mi50/bench/kld.sh                                      # KL divergence, tensor vs layer
```

## Docs

- [hardware-and-bios.md](docs/hardware-and-bios.md): MI50 requirements, BIOS, kernel parameters, cooling, power
- [server-flags.md](docs/server-flags.md): every flag and environment variable, and why
- [performance.md](docs/performance.md): all measurements, method, profile of where time goes now
- [troubleshooting.md](docs/troubleshooting.md): problems hit and fixes
- [roadmap.md](docs/roadmap.md): what could still be faster, with the baseline to beat
- [optimization-log.md](docs/optimization-log.md): every speed idea tried, what worked and what didn't
- [extras.md](docs/extras.md): the optional services around the server (HTTPS, voice input, web search, memory, Claude Code)
- [gpu-fan-control-guide.md](docs/gpu-fan-control-guide.md): MI50 cooling with a motherboard fan header

## Credits

[milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) (gfx906 kernels, DFlash2),
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), [ggml-org Qwen3.8 GGUFs](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF),
[z-lab DFlash2 drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF). MIT licensed, like llama.cpp.
