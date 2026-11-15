#!/usr/bin/env bash
# Optional: frankcrawford it87 DKMS driver + gpu-fan.sh (MI50 temps -> SYS_FAN4 on the it8792 chip). Run with sudo.
# BOARD-SPECIFIC: written for Gigabyte X570 board (it8688 + it8792). On a different board,
# run `sensors-detect` and read mi50/docs/gpu-fan-control-guide.md, then edit gpu-fan.sh before enabling gpu-fan.service.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -d /usr/src/it87-frankcrawford ]; then
  git clone https://github.com/frankcrawford/it87 /usr/src/it87-frankcrawford
  git -C /usr/src/it87-frankcrawford checkout 490f76f    # tested version
fi
cd /usr/src/it87-frankcrawford && make dkms && depmod -a
echo it87 > /etc/modules-load.d/it87.conf   # load the driver at boot
install -m 0755 "$HERE/gpu-fan.sh" /usr/local/bin/gpu-fan.sh
install -m 0644 "$HERE/gpu-fan.service" /etc/systemd/system/gpu-fan.service && systemctl daemon-reload
echo "Installed, NOT enabled. After a reboot and checking the fan mapping: sudo systemctl enable --now gpu-fan"
echo "Check after reboot: dkms status | grep it87 ; cat /sys/module/it87/srcversion (must NOT be CC614881D45ED7F90D9FED9 = in-kernel driver)"
