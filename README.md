# gfx906-llama-cpp (dual MI50 tensor split)

llama.cpp for **two AMD Instinct MI50 32 GB** cards (gfx906): Qwen3.8-27B at full **Q8_0** with a DFlash2
speculative drafter and **tensor split** across both cards, at **~50 tokens/s** average generation, same quality.

| Avg generation over a 40-task suite, Qwen3.8-27B Q8_0 | t/s |
|---|---|
| gfx906-turbo, layer split, no drafter | 19.0 |
| milpster gfx906 fork + DFlash2 drafter, layer split | 36.3 |
| **this repo: + tensor split** | **49.6** |

**Start here: [mi50/README.md](mi50/README.md)** (requirements, quick start, settings, benchmarks, docs).

## Credit

This is not original llama.cpp work. Almost all of the code comes from two projects:

- **[milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp)**: the gfx906 (Radeon VII / MI50 / MI60)
  fork with wave64 kernel tuning, DFlash2 speculative decoding and the drafter output-head mirror. This repo is a
  snapshot of it at commit
  [`1373e889`](https://github.com/milpster/gfx906-llama-cpp/commit/1373e889a4d5c69f453fe82c96332585f3d2b960).
- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**, which milpster's fork is built on.

Both are MIT licensed. Their [LICENSE](LICENSE), [AUTHORS](AUTHORS) and [licenses/](licenses/) are kept unchanged.

## What this repo adds

- **One source fix** in `src/llama-context.cpp`: the DFlash2 drafter's output-head mirror verified its copy with
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
