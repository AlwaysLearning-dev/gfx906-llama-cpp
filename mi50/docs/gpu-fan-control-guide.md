# MI50 GPU Fan Control - Setup, Locations & Reference

Temperature-based control of the 120 mm case fan on **SYS_FAN4**, ramped by the
hotter of the two AMD MI50 GPUs. Built and debugged on the reference box
(Ubuntu 24.04, Gigabyte X570 board). Install with `sudo mi50/setup/fan/install-fan.sh`.

Last updated: 2026-07-30.

---

## 1. Quick status / health check

```bash
# Is the service running and what did it bind to?
systemctl is-active gpu-fan.service
journalctl -u gpu-fan.service -n 5 --no-pager | grep -E 'Fan header|GPU sensors'

# Live view: duty / enable(1=manual) / fan RPM on the it8792 chip
IT=$(for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = it8792 ] && echo $h && break; done)
watch -n1 "cat $IT/pwm3; cat $IT/pwm3_enable; cat $IT/fan3_input"

# GPU temps (both MI50s)
for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = amdgpu ] && \
  printf '%s %sC\n' "$h" $(($(cat $h/temp1_input)/1000)); done
```

Healthy looks like: service `active`, `pwm3_enable`=1, `pwm3` matching the curve
below for the current temp, and a **non-zero** `fan3_input`.

---

## 2. File locations

| Path | What it is |
|------|-----------|
| `/usr/local/bin/gpu-fan.sh` | The control loop (bash). Resolves hwmon by name, reads GPU temps, writes PWM. |
| `/etc/systemd/system/gpu-fan.service` | systemd unit. `enable`d -> starts on boot, `Restart=always`. |
| `/usr/src/it87-frankcrawford/` | Cloned source of the out-of-tree it87 driver. |
| `/var/lib/dkms/it87/` | DKMS-built driver artifacts (auto-rebuilds on kernel updates). |
| `/lib/modules/$(uname -r)/updates/dkms/it87.ko.zst` | The driver that actually loads (overrides the in-kernel one). |
| `/etc/modules-load.d/it87.conf` | Contains `it87` -> force-loads the driver at boot. |
| `mi50/docs/gpu-fan-control-guide.md` | This document. |

---

## 3. Hardware map (IMPORTANT: indices are NOT stable)

`hwmon*` numbers **shuffle across reboots and driver changes** - never hardcode
them. Resolve by `name` instead. As last observed:

| Device | name | Role |
|--------|------|------|
| ITE Super-I/O #1 | `it8688` | Other fan headers (CPU, chassis) - BIOS-controlled. **Do not touch.** |
| ITE Super-I/O #2 | `it8792` | **SYS_FAN4 = `pwm3` / `fan3_input`.** pwm1/pwm2 are unconnected (always 0 RPM). |
| GPU 1 | `amdgpu` (first card) | MI50 - `temp1_input` (edge temp, milli- degC). |
| GPU 2 | `amdgpu` (second card) | MI50 - `temp1_input`. |

> This board has **two** ITE chips. The stock in-kernel `it87` driver only ever
> detected one of them; the out-of-tree driver exposes both. SYS_FAN4 lives on
> the **it8792**, identifiable by pwm1/pwm2 being dead and pwm3 driving the fan.

---

## 4. The fan curve

Control temp = **max of the two MI50 edge temps**. Duty is 0-255.

| GPU edge temp | PWM duty | ~RPM |
|---------------|----------|------|
| < 45  degC (idle/light) | 115 | ~1350 |
| 45-54  degC | 145 | ~1850 |
| 55-64  degC | 180 | ~2250 |
| 65-74  degC | 220 | ~2600 |
| >= 75  degC | 255 | ~2850 |

### Measured fan characteristics (why the floor is 115)

This specific fan has hysteresis:

- **Won't start** from a stop below duty **110** (0 RPM at 60-100, jumps to
  ~1300 RPM at 110).
- **Stalls** when spinning down below duty **~100**.

So the curve **never writes below 115** - the fan stays spinning at all times and
never falls into the dead zone where it would need to re-start from a stop under
rising load. To change the curve, edit the `if/elif` block in
`/usr/local/bin/gpu-fan.sh` and keep every active step >= 115.

---

## 5. How it works

1. `gpu-fan.service` starts `gpu-fan.sh` at boot as root.
2. The script finds the fan chip by name (`it8792`) and collects **all** `amdgpu`
   temp sensors, logging the resolved paths (visible in `journalctl`).
3. It sets `pwm3_enable=1` (manual mode) and loops every 5 s: read the hottest
   GPU, map to a duty via the curve, write `pwm3`.
4. On stop (`SIGTERM`/`SIGINT`) it sets `pwm3_enable=2`, handing the header back
   to hardware/auto control.

**Why the out-of-tree driver is required:** the it8792's on-chip SmartGuardian
firmware controller re-asserts its own PWM value about once per second, and no
BIOS Q-Fan mode fully releases the header. The stock in-kernel `it87` driver
loses that fight (writes revert within ~1 s). The maintained
[`frankcrawford/it87`](https://github.com/frankcrawford/it87) driver takes true
manual control, so written values hold.

---

## 6. Reinstalling / rebuilding the driver

DKMS auto-rebuilds on kernel updates. To do it manually (e.g. after a failed
kernel upgrade):

```bash
cd /usr/src/it87-frankcrawford
sudo make dkms          # build + register + install via DKMS
sudo depmod -a
sudo reboot             # clean reload so hwmon numbering settles
```

Verify the out-of-tree build is the one loaded (srcversion should NOT be the
in-kernel `CC614881D45ED7F90D9FED9`):

```bash
dkms status | grep it87
modinfo it87 | grep -E '^(filename|srcversion):'   # filename must be under updates/dkms/
cat /sys/module/it87/srcversion                     # loaded module's srcversion
```

If a fresh clone is ever needed: `git clone https://github.com/frankcrawford/it87`.

---

## 7. Troubleshooting

**Service `active` but fan not tracking / wrong RPM**
Check the resolved paths in the log - a driver or kernel change may have moved
the chips:
```bash
journalctl -u gpu-fan.service -n 5 --no-pager | grep -E 'Fan header|GPU sensors'
```
The script resolves by name, so it self-corrects across reboots; if it can't find
`it8792` or any `amdgpu` it exits with a `FATAL:` line in the journal.

**PWM writes revert (129/136/255 creeping back)**
The out-of-tree driver isn't loaded - you're back on the in-kernel one. Confirm
with the `modinfo`/`srcversion` checks in section 6 and reinstall.

**Fan stuck at 0 RPM while duty is set**
Duty is below the ~110 start threshold, or the fan is on the wrong chip. Confirm
`pwm3` on the **it8792** (not it8688) and that duty >= 115.

**`systemctl start` hangs / service dead with no journal entries**
That was the original ordering-cycle bug: do **not** add `After=multi-user.target`
to the unit. `WantedBy=multi-user.target` in `[Install]` is the only ordering it
needs.

---


## 8. BIOS note

Leave **SYS_FAN4 on "Full Speed"** in BIOS. The driver holds manual control so it
doesn't interfere, and it acts as a **failsafe**: if the DKMS driver ever fails to
load (e.g. a broken kernel upgrade), the service exits cleanly and the fan falls
back to firmware full-speed - maximum cooling rather than off.

---

## 9. History - what was fixed (2026-07-30)

1. **systemd ordering cycle** - removed `After=multi-user.target` (it deadlocked
   the boot transaction and made `systemctl start` hang forever, which looked
   like SSH flakiness).
2. **Firmware PWM override** - installed the out-of-tree `frankcrawford/it87`
   DKMS driver so manual writes hold.
3. **Wrong-chip / hwmon fragility** - rewrote the script to resolve the fan chip
   (`it8792`) and both GPUs (`amdgpu`) by name instead of hardcoded `hwmon2/5/6`,
   which had drifted to the it8688 + the wifi sensor after a reboot.
4. **Fan stall** - measured the 110 start threshold and set the curve floor to
   115 so the fan always spins.
