#!/bin/bash
# MI50 GPU-temp fan control for SYS_FAN4.
# Resolves hwmon paths by NAME, not index: hwmon numbers shuffle across reboots
# and driver changes, and this board has TWO ITE chips (it8688 + it8792) -- only
# the it8792 carries SYS_FAN4 on pwm3 (pwm1/pwm2 are unconnected).
set -euo pipefail

find_hwmon_by_name() {
  local want="$1" h
  for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/name" ] || continue
    if [ "$(cat "$h/name")" = "$want" ]; then echo "$h"; return 0; fi
  done
  return 1
}

# --- Fan controller: the it8792 Super-I/O chip, channel 3 = SYS_FAN4 ---
FANCHIP="$(find_hwmon_by_name it8792)" || { echo "FATAL: it8792 hwmon not found" >&2; exit 1; }
PWM_PATH="$FANCHIP/pwm3"
PWM_ENABLE="$FANCHIP/pwm3_enable"
[ -w "$PWM_PATH" ] || { echo "FATAL: $PWM_PATH not writable" >&2; exit 1; }

# --- Temp sources: every amdgpu instance (both MI50s, wherever they enumerate) ---
GPU_TEMPS=()
for h in /sys/class/hwmon/hwmon*; do
  [ -r "$h/name" ] || continue
  [ "$(cat "$h/name")" = "amdgpu" ] || continue
  [ -r "$h/temp1_input" ] && GPU_TEMPS+=("$h/temp1_input")
done
[ "${#GPU_TEMPS[@]}" -gt 0 ] || { echo "FATAL: no amdgpu temp sensors found" >&2; exit 1; }

echo "Fan header : $PWM_PATH ($(cat "$FANCHIP/name"))"
echo "GPU sensors: ${GPU_TEMPS[*]}"

# On stop, hand the header back to hardware/auto control.
trap 'echo 2 > "$PWM_ENABLE" 2>/dev/null; exit 0' SIGTERM SIGINT

echo 1 > "$PWM_ENABLE"

while true; do
  T=0
  for tf in "${GPU_TEMPS[@]}"; do
    v=$(cat "$tf" 2>/dev/null || echo 0)
    (( v > T )) && T=$v
  done
  T=$(( T / 1000 ))
  # Measured fan: starts at duty 110 (~1300 RPM), stalls below ~100, max ~2850
  # RPM at 255. Floor stays >=115 so it always spins and never drops into the
  # dead zone where it would need to re-start from a stop.
  if   (( T < 45 )); then PWM=115   # idle/light  ~1350 RPM
  elif (( T < 55 )); then PWM=145   # light load  ~1850 RPM
  elif (( T < 65 )); then PWM=180   # moderate    ~2250 RPM
  elif (( T < 75 )); then PWM=220   # heavy       ~2600 RPM
  else                    PWM=255   # max         ~2850 RPM
  fi
  echo "$PWM" > "$PWM_PATH"
  sleep 5
done
