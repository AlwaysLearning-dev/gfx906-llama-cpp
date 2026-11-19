#!/usr/bin/env bash
# ROCm 7.2.4 runtime libraries in a side folder (no sudo, the system ROCm 7.1.1 stays as it is).
# Why: the 7.1.1 runtime keeps host memory while llama-server generates (about 250 KiB per written token with
# tensor split: recorded HIP graphs and the AllReduce event record). With the 7.2.4 runtime the heap stays flat and
# speed and output are the same. mi50/run-server.sh picks the folder up by itself.
# Usage: mi50/setup/03-rocm-runtime.sh [dir, default ~/rocm-runtime-7.2.4]
set -euo pipefail
D=${1:-$HOME/rocm-runtime-7.2.4}; B=https://repo.radeon.com/rocm/apt/7.2.4
mkdir -p "$D/debs"; cd "$D/debs"
while read -r sha path; do
  f=$(basename "$path")
  [ -s "$f" ] || curl -fL --retry 3 -o "$f" "$B/$path"
  echo "$sha  $f" | sha256sum -c -
  dpkg-deb -x "$f" "$D"
done <<'LIST'
70961d0f79708148a54f23bdde2ebab5c389d5ab4373170d8df94f3e54d87a0b pool/main/h/hip-runtime-amd/hip-runtime-amd_7.2.53211.70204-93~24.04_amd64.deb
6af02a99d15efbf21ba0e20a333aac32e2d712aa4b325dd0617842f4cde64ab4 pool/main/h/hsa-rocr/hsa-rocr_1.18.0.70204-93~24.04_amd64.deb
02951341120d23ebf34b066a84ab172e4813cba4a601f910545c6658989bee87 pool/main/c/comgr/comgr_3.0.0.70204-93~24.04_amd64.deb
3b13874e567fa40b6eaf9f8d572d5c9a4ac780eb37496b31565755463a378295 pool/main/r/rocprofiler-register/rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb
LIST
ls -l "$D/opt/rocm-7.2.4/lib/libamdhip64.so.7"
echo "Runtime libraries: $D/opt/rocm-7.2.4/lib (set ROCM_RT to this path if you used another folder)"
