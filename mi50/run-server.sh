#!/usr/bin/env bash
# Run llama-server with the tested dual-MI50 settings: Qwen3.8-27B Q8_0 + vision + DFlash2 drafter, tensor split.
# Every setting can be overridden by env var; extra arguments are passed through to llama-server.
#   MODELS   model dir (default ~/models)          SM      tensor | layer (default tensor)
#   HOST     bind address (default 127.0.0.1)      PORT    default 8080
#   CTX      total context (default 409600)        NP      parallel slots (default 2, so 2 x 204800)
#   DRAFT    q8 | q4 | none (default q8)           NMAX    drafted tokens per step (default 4)
#   DRAFT_DEV  GPU that runs the drafter and its mirror (default ROCm0)
#   EFFORT   reasoning effort low|medium|high (default medium)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$ROOT/build-mi50/bin/llama-server}
M=${MODELS:-$HOME/models}
SM=${SM:-tensor}; DRAFT=${DRAFT:-q8}
SPEC=()
case $DRAFT in
  none) ;;
  q8|q4)
    D=$M/Qwen3.8-27B-DFlash2-$( [ "$DRAFT" = q4 ] && echo Q4_K_M || echo Q8_0 ).gguf
    export LLAMA_DFLASH_MIRROR_OUTPUT=1   # GPU-local copy of the target's output head for the drafter
    SPEC=(-md "$D" --spec-type draft-dflash --spec-draft-n-max "${NMAX:-4}"
          --spec-draft-override-tensor ".*=${DRAFT_DEV:-ROCm0}" --spec-draft-device "${DRAFT_DEV:-ROCm0}" -ngld 99) ;;
  *) echo "DRAFT must be q8, q4 or none"; exit 1 ;;
esac
export HSA_OVERRIDE_GFX_VERSION=9.0.6 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1} HSA_XNACK=0 \
  HIP_GRAPH=1 AMD_LOG_LEVEL=0 GGML_CUDA_FATTN_PATH=force_convert GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 \
  GPU_SINGLE_ALLOC_PERCENT=100 GPU_MAX_ALLOC_PERCENT=100 HSA_ENABLE_SDMA=1 \
  LD_LIBRARY_PATH=$(dirname "$BIN"):${ROCM:-/opt/rocm-7.1.1}/lib
exec "$BIN" -m "$M/Qwen3.8-27B-Q8_0.gguf" --mmproj "$M/mmproj-Qwen3.8-27B-Q8_0.gguf" \
  --load-mode none --alias Qwen3.8-27B --jinja \
  --chat-template-kwargs "{\"reasoning_effort\":\"${EFFORT:-medium}\"}" --reasoning on --reasoning-budget 32768 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  -fa on -ngl 999 -b 2048 -ub "${UB:-384}" --ctx-checkpoints 30 \
  --device ROCm0,ROCm1 -sm "$SM" -mg 0 --pipeline-parallel off \
  --parallel "${NP:-2}" -c "${CTX:-409600}" --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  "${SPEC[@]}" "$@"
