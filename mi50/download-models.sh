#!/usr/bin/env bash
# Download the models used here (~31 GB) at pinned Hugging Face revisions and verify sha256.
# Usage: mi50/download-models.sh [DIR]   (default: $MODELS or ~/models). Needs only curl + sha256sum.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=${1:-${MODELS:-$HOME/models}}; mkdir -p "$D"; cd "$D"
get() { [ -s "$3" ] && { echo "have $3"; return; }
        curl -fL --retry 5 -C - -o "$3.part" "https://huggingface.co/$1/resolve/$2/$3" && mv "$3.part" "$3"; }
# Target + vision projector. ggml-org only (unsloth/lmstudio builds carry an MTP layer this fork can't load).
# Pinned to 0669b986 = the files tested here; ggml-org later re-uploaded the Q8 with a different checksum.
T=ggml-org/Qwen3.8-27B-GGUF; TR=0669b98607d47046c7c2b3f801011d54a08cfccf
get $T $TR Qwen3.8-27B-Q8_0.gguf
get $T $TR mmproj-Qwen3.8-27B-Q8_0.gguf
# DFlash2 drafter (z-lab; NOT incoai = older format). Q8_0 is used; Q4_K_M gives identical output, 0.9 GB smaller.
Z=z-lab/Qwen3.8-27B-DFlash2-GGUF; ZR=2d9571f8ce46e151f61c6499c99dee6079e1d610
get $Z $ZR Qwen3.8-27B-DFlash2-Q8_0.gguf
if [ "${WITH_Q4_DRAFTER:-0}" = 1 ]; then get $Z $ZR Qwen3.8-27B-DFlash2-Q4_K_M.gguf; fi
sha256sum -c --ignore-missing "$HERE/models.sha256"
echo "Models in $D"
