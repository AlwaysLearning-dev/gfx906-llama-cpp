#!/usr/bin/env bash
# Kernel parameters and build packages for MI50s on Ubuntu 24.04. Run with sudo, then reboot after 02-rocm.sh.
#   amdgpu.rebar=0 amdgpu.runpm=0   always (BAR resizing and runtime power-down cause trouble on these cards)
#   FREEZE_FIX=1                    also processor.max_cstate=1 + lockup-panic sysctl: fixed idle whole-system freezes
#                                   on a Ryzen 3000 box. Only use it if you see idle freezes.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
apt-get update
apt-get install -y build-essential cmake git curl python3 pciutils
PARAMS="amdgpu.rebar=0 amdgpu.runpm=0"
if [ "${FREEZE_FIX:-0}" = 1 ]; then
  PARAMS="$PARAMS processor.max_cstate=1"
  install -m 0644 "$HERE/99-lockup-panic.conf" /etc/sysctl.d/ && sysctl --system >/dev/null
fi
cp /etc/default/grub "/etc/default/grub.bak-$(date +%F)"
for p in $PARAMS; do
  grep -q "GRUB_CMDLINE_LINUX_DEFAULT=.*${p%%=*}=" /etc/default/grub || \
    sed -i "s/^GRUB_CMDLINE_LINUX_DEFAULT=\"\(.*\)\"/GRUB_CMDLINE_LINUX_DEFAULT=\"\1 $p\"/" /etc/default/grub
done
update-grub
grep ^GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub
echo "OS step done. Next: sudo mi50/setup/02-rocm.sh, then reboot."
