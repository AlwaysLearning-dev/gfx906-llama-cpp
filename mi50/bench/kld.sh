#!/usr/bin/env bash
# Quality check: KL divergence of -sm tensor vs -sm layer on the same model (code corpus = this repo's C++ sources).
# Stop any running server first (both GPUs are needed). Usage: mi50/bench/kld.sh   (MODELS=~/models by default)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
B=$ROOT/build-mi50/bin; M=${MODELS:-$HOME/models}/Qwen3.8-27B-Q8_0.gguf; O=${OUT:-$ROOT/mi50-results}
mkdir -p "$O"
[ -s "$O/corpus.txt" ] || cat "$ROOT"/common/*.cpp "$ROOT"/src/*.cpp | head -c 1600000 > "$O/corpus.txt"
export HSA_OVERRIDE_GFX_VERSION=9.0.6 LD_LIBRARY_PATH=$B:${ROCM:-/opt/rocm-7.1.1}/lib
run() { "$B/llama-perplexity" -m "$M" -f "$O/corpus.txt" -c 512 --chunks 40 -ngl 999 -fa on --device ROCm0,ROCm1 "$@"; }
echo "baseline (-sm layer) ..."; run -sm layer --kl-divergence-base "$O/layer.kld" > "$O/kld-base.log" 2>&1
echo "tensor split ..."; run -sm tensor --kl-divergence-base "$O/layer.kld" --kl-divergence > "$O/kld-tensor.log" 2>&1
grep -E "Mean    KLD|Maximum KLD|Same top p" "$O/kld-tensor.log"
echo "Expect mean KLD around 0.0006 or lower and same-top above 99%. The .kld file is several GB: delete it after."
