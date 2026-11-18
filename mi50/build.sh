#!/usr/bin/env bash
# Build llama-server (+ llama-bench, llama-cli, llama-perplexity) for gfx906 against ROCm 7.1.1.
# Output: <repo>/build-mi50/bin. Usage: mi50/build.sh   (ROCM=/opt/rocm-7.1.1 by default)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
R=${ROCM:-/opt/rocm-7.1.1}
BUILD=${BUILD:-$ROOT/build-mi50}
[ -x "$R/lib/llvm/bin/clang++" ] || { echo "ROCm not found at $R (run mi50/setup/02-rocm.sh)"; exit 1; }
export PATH=$R/bin:$PATH HSA_OVERRIDE_GFX_VERSION=9.0.6
# Web UI assets are downloaded at build time. Without a pin the build takes "latest", and newer asset bundles
# have a layout this tree rejects ("UI: missing required asset(s)"). b12445 is the bundle the tested builds used.
export HF_UI_VERSION=${HF_UI_VERSION:-b12445}
cmake -B "$BUILD" -S "$ROOT" \
  -DCMAKE_HIP_COMPILER=$R/lib/llvm/bin/clang++ \
  -Dhip_DIR=$R/lib/cmake/hip -Dhipblas_DIR=$R/lib/cmake/hipblas \
  -Drocblas_DIR=$R/lib/cmake/rocblas -Dhsa-runtime64_DIR=$R/lib/cmake/hsa-runtime64 \
  -Damd_comgr_DIR=$R/lib/cmake/amd_comgr -DAMDDeviceLibs_DIR=$R/lib/cmake/AMDDeviceLibs \
  -DCMAKE_PREFIX_PATH=$R \
  -DGGML_CUDA_FA_QUANTS=all -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DGGML_VULKAN=OFF \
  -DGGML_CUDA_VEGA_TUNE=ON -DGGML_CUDA_VEGA_TUNE_FATTN=ON -DGGML_CUDA_VEGA_TUNE_MMQ=${VEGA_TUNE_MMQ:-OFF} \
  -DGGML_CUDA_VEGA_TUNE_TOPK=ON -DGGML_CUDA_VEGA_TUNE_GRAPHS=ON \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_GRAPHS=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$BUILD" -j"$(nproc)" --target llama-bench llama-cli llama-perplexity
# a single-target build can relink the shared libs, so build the server last
cmake --build "$BUILD" -j"$(nproc)" --target llama-server
echo "Built: $BUILD/bin/llama-server"
