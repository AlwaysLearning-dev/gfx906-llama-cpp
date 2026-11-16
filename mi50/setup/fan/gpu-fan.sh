#!/bin/bash
# MI50 fan control: two PWM fans, one per card, each following a card's junction temperature.
#   fan A = SYS_FAN4 (it8792 pwm3)      fan B = SYS_FAN2 (it8688 pwm3)
# MAP=shared (default): both fans follow the hotter card. MAP=split: fan A follows the card at
# PCI_A, fan B the card at PCI_B. Set these in /etc/default/gpu-fan.
# hwmon numbers change across boots, so chips are found by name.
# Curve is for ARCTIC S8038-10K (about 40 RPM per PWM step, 500-10000 RPM).
set -euo pipefail
MAP=shared; PCI_A=; PCI_B=   # empty: the first and second MI50 found, in PCI order
[ -r /etc/default/gpu-fan ] && . /etc/default/gpu-fan
if [ -z "$PCI_A" ] || [ -z "$PCI_B" ]; then
  mapfile -t MI50S < <(for d in /sys/bus/pci/drivers/amdgpu/0000:*; do grep -qi '^0x66a' "$d/device" 2>/dev/null && basename "$d"; done | sort)
  PCI_A=${PCI_A:-${MI50S[0]:-}}; PCI_B=${PCI_B:-${MI50S[1]:-}}
fi

find_hwmon_by_name() {
  local h
  for h in /sys/class/hwmon/hwmon*; do
    [ -r "$h/name" ] && [ "$(cat "$h/name")" = "$1" ] && { echo "$h"; return 0; }
  done
  return 1
}
A_CHIP=$(find_hwmon_by_name it8792) || { echo "FATAL: it8792 hwmon not found" >&2; exit 1; }
B_CHIP=$(find_hwmon_by_name it8688) || { echo "FATAL: it8688 hwmon not found" >&2; exit 1; }
FAN_PWM=("$A_CHIP/pwm3" "$B_CHIP/pwm3")
FAN_RPM=("$A_CHIP/fan3_input" "$B_CHIP/fan3_input")
for p in "${FAN_PWM[@]}"; do [ -w "$p" ] || { echo "FATAL: $p not writable" >&2; exit 1; }; done

gpu_hwmon() { ls -d "/sys/bus/pci/devices/$1"/hwmon/hwmon* 2>/dev/null | head -1; }
GPU_A=$(gpu_hwmon "$PCI_A"); GPU_B=$(gpu_hwmon "$PCI_B")
[ -n "$GPU_A" ] && [ -n "$GPU_B" ] || { echo "FATAL: MI50 temp sensors not found at $PCI_A / $PCI_B" >&2; exit 1; }

temp() {  # junction (temp2) if present, else edge + 25
  if [ -r "$1/temp2_input" ]; then echo $(( $(cat "$1/temp2_input") / 1000 ))
  else echo $(( $(cat "$1/temp1_input") / 1000 + 25 )); fi
}
curve() {  # junction C -> PWM
  local t=$1
  if   (( t < 50 )); then echo 30    # ~1200 RPM, idle
  elif (( t < 60 )); then echo 50    # ~2000
  elif (( t < 70 )); then echo 80    # ~3200
  elif (( t < 78 )); then echo 115   # ~4600
  elif (( t < 85 )); then echo 160   # ~6400
  elif (( t < 92 )); then echo 205   # ~8200
  else                    echo 255   # full, ~10000
  fi
}

echo "Fan A: ${FAN_PWM[0]}  Fan B: ${FAN_PWM[1]}  map: $MAP"
echo "GPU A: $GPU_A ($PCI_A)  GPU B: $GPU_B ($PCI_B)"

# On stop, hand both headers back to the BIOS (set them to Full Speed there as a failsafe).
trap 'for p in "${FAN_PWM[@]}"; do echo 2 > "${p}_enable" 2>/dev/null; done; exit 0' SIGTERM SIGINT
for p in "${FAN_PWM[@]}"; do echo 1 > "${p}_enable"; done

CUR=(255 255); STALL=(0 0)
HIST_A=(); HIST_B=()   # last 5 readings (15 s) per card: the fan follows the peak, so short dips do not slow it
while true; do
  TA=$(temp "$GPU_A" 2>/dev/null || echo 99); TB=$(temp "$GPU_B" 2>/dev/null || echo 99)
  HIST_A+=("$TA"); HIST_B+=("$TB")
  (( ${#HIST_A[@]} > 5 )) && HIST_A=("${HIST_A[@]:1}")
  (( ${#HIST_B[@]} > 5 )) && HIST_B=("${HIST_B[@]:1}")
  for v in "${HIST_A[@]}"; do (( v > TA )) && TA=$v; done
  for v in "${HIST_B[@]}"; do (( v > TB )) && TB=$v; done
  if [ "$MAP" = split ]; then T=("$TA" "$TB"); else M=$(( TA > TB ? TA : TB )); T=("$M" "$M"); fi
  for i in 0 1; do
    want=$(curve "${T[$i]}")
    # step down only when 6 C below the threshold, so the fan does not hunt
    if (( want < CUR[i] )); then want=$(curve $(( T[i] + 6 ))); (( want > CUR[i] )) && want=${CUR[$i]}; fi
    # a fan that reads 0 RPM for 3 loops gets full speed
    rpm=$(cat "${FAN_RPM[$i]}" 2>/dev/null || echo 0)
    if (( rpm < 200 )); then STALL[i]=$(( STALL[i] + 1 )); else STALL[i]=0; fi
    (( STALL[i] >= 3 )) && want=255
    CUR[i]=$want
    echo "$want" > "${FAN_PWM[$i]}"
  done
  sleep 3
done
