# Hardware, BIOS and kernel settings

## The cards

**AMD Instinct MI50 32 GB** (Vega 20, gfx906): 32 GB HBM2 at about 1 TB/s theoretical. A simple read kernel
measured **~800 GB/s** per card on the reference box. That bandwidth is what makes them good for LLM token
generation, which is memory-bound. They are old datacenter cards, so:

- **No fans.** They're passive and expect server airflow. Add blower fans or ducted case fans, and drive them from
  GPU temperature ([gpu-fan-control-guide.md](gpu-fan-control-guide.md), `mi50/setup/fan/`). Unmanaged, they overheat quickly.
- **Power cap 225 W each**, which is also the maximum on these vBIOSes. Under tensor-split decode both cards sit at
  the cap. Budget your PSU for 2 x 225 W plus the rest of the system.
- **ROCm support:** gfx906 is out of AMD's official support list, but ROCm 7.1.1 still runs it with
  `HSA_OVERRIDE_GFX_VERSION=9.0.6`. The in-kernel amdgpu driver is used (no amdgpu-dkms).
- **vBIOS varies.** The two reference cards report different SKUs (one "AMD Radeon Graphics" and one
  "MI50/MI60") and measured the same bandwidth. Mixed vBIOS works fine. Flashing for a higher power cap is risky and
  wasn't tried.

## Reference board and slots

- **Gigabyte X570 board**, Ryzen 9 (Zen 2), 32 GB RAM.
- Both MI50s are on CPU lanes at **PCIe 4.0 x8/x8**. That's plenty: decode traffic between the cards is small
  (~128 syncs per token), and the tensor-split sync turned out not to be bandwidth-limited.
- A display card sits in the chipset slot so it takes no lanes from the MI50s.

## BIOS settings (required)

| Setting | Value | Why |
|---|---|---|
| **Above 4G Decoding** | **Enabled** | The MI50s need a 64-bit memory window above 4 GB |
| **CSM Support** | **Disabled** | With CSM on, the firmware gives **no** above-4G window even with Above 4G "Enabled", and both MI50s fail (`amdgpu ... error -12`, `can't assign; no space`) |
| Re-Size BAR | Disabled | The kernel parameter `amdgpu.rebar=0` is set anyway |
| GPU fan header | Full speed | The fan script takes manual PWM control |

**Keep a UEFI (GOP-capable) display card installed** if your CPU has no iGPU. Without one, the reference board
still booted Linux, but the firmware **switched CSM back on by itself**, which killed the MI50s. Legacy-only cards
need CSM, so they don't work either. After *any* card change, re-check CSM Disabled + Above 4G Enabled.

Check from Linux:
```bash
journalctl -k -b | grep -E 'shadowed ROM|root bus resource \[mem 0x8'   # want ONLY the "root bus resource [mem 0x8..." line
rocm-smi --showproductname | grep -c 'Card Series'                    # want 2
```

## Kernel parameters

Set by `mi50/setup/01-os.sh` in `/etc/default/grub`:

| Parameter | Why |
|---|---|
| `amdgpu.rebar=0` | Avoids BAR resizing problems with these cards |
| `amdgpu.runpm=0` | No runtime power-down of the cards (wake-up hangs) |
| `processor.max_cstate=1` (only with `FREEZE_FIX=1`) | Fix for whole-system **idle freezes** on the reference Ryzen (Zen 2): the deepest CPU idle state (CC6) hung the machine every few weeks. No freeze since this was set. Leave it off unless you see idle freezes |

`FREEZE_FIX=1` also installs a lockup-panic sysctl, so a hang reboots the box after 10 s instead of freezing forever.

## Network

Use **ethernet**. On the reference box, every "the server is down" turned out to be a USB WiFi adapter dropping
(hundreds of disconnects a week) while the server itself was healthy. Check `curl 127.0.0.1:8080/health` on the
box before restarting anything.
