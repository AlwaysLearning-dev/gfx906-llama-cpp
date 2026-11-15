#!/usr/bin/env bash
# ROCm 7.1.1 userspace for Ubuntu 24.04 (noble), using the in-kernel amdgpu driver (no amdgpu-dkms). Run with sudo.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
U=${SUDO_USER:?run via sudo from the user that will run llama-server}
mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor --yes -o /etc/apt/keyrings/rocm.gpg
install -m 0644 "$HERE/rocm.list" /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update
apt-get install -y rocm      # tested: rocm 7.1.1.70101-38~24.04
usermod -aG render,video "$U"
echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf; ldconfig
echo "ROCm installed. REBOOT, then check: rocm-smi --showproductname (expect two MI50s)"
