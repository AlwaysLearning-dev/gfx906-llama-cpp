# Troubleshooting: every problem hit so far

Newest first. Each row is a real problem from building and running the reference box.

| Symptom | Cause | Fix |
|---|---|---|
| Tensor split + drafter aborts at startup: `ggml-backend-meta.cpp:1631: GGML_ASSERT(size % chunk_size_full == 0)` in `init_mirror_output` | The drafter's output-head mirror verified its copy with 4 KiB partial reads; split buffers only allow whole-tensor reads | Fixed in this repo (`mi50/mirror-verify-tensor-split.patch`). **Re-apply it if you update from upstream** |
| `rocprofv3` segfaults profiling llama.cpp | HIP graph capture under the profiler | Profile with `GGML_CUDA_DISABLE_GRAPHS=1` |
| "Server unreachable" from other machines | Network, not llama-server (on the reference box: a flaky USB WiFi adapter) | On the box: `curl 127.0.0.1:8080/health`. Use ethernet |
| milpster fork won't start with `--no-mmap` | The fork removed that flag | `--load-mode none` |
| `curl http://host:8080/` returns 415 | The server only sends the gzip-compressed UI | Use a browser, or `curl --compressed` |
| Building one tool rebuilt all the libraries | A system header update made every object look stale | Harmless; rebuild `llama-server` last after any single-target build |
| Community Qwen3.8 GGUFs won't load (`missing tensor 'blk.64.ssm_conv1d.weight'`) | unsloth/lmstudio builds include an MTP layer | ggml-org GGUFs only |
| MI50s missing after a hardware change (`Fatal error during GPU init`, `error -12`, `bridge window ... can't assign; no space`) | CSM got switched on (some boards do it by themselves when no UEFI display card is present); with CSM on there's no above-4G window | BIOS: CSM Disabled, Above 4G Enabled; keep a UEFI/GOP display card installed ([hardware-and-bios.md](hardware-and-bios.md)) |
| Whole-system **freezes at idle** every few weeks, journal cut mid-line, no panic | Ryzen 3000 idle C-state (CC6) hang. `amdgpu.runpm=0` alone didn't fix it | `sudo FREEZE_FIX=1 mi50/setup/01-os.sh` (`processor.max_cstate=1` + lockup-panic sysctl). No freeze since |
| After a kernel upgrade "the GPUs broke" / fans stopped tracking | The it87 DKMS module didn't rebuild for the new kernel. amdgpu itself was fine | `dkms status`, `rocm-smi`, `systemctl status gpu-fan llama-server`. Reinstall it87 via `mi50/setup/fan/install-fan.sh` if needed |
| Generation suddenly slower (24-25 t/s) | Low drafter acceptance on unpredictable content (e.g. generated HTML/JS effects) | Normal: check `draft_n_accepted / draft_n` in the response timings |
| The first reply after a restart is slow (~15-40 t/s) | Warm-up | Normal; the second request is full speed |

## Standing gotchas

- **Test servers need production stopped**: the model fills both GPUs.
- **`pgrep -f` / `pkill -f` with a plain pattern matches your own shell** and kills it. Use the bracket trick:
  `pkill -f '[l]lama-server'`.
- **The fork rejects `--no-mmap`**: use `--load-mode none`.
- **Port 8080 is plain HTTP without auth.** Keep it on localhost or firewall it to your LAN.

## Useful checks

```bash
curl -s 127.0.0.1:8080/health                         # server up?
journalctl -u llama-server -b --no-pager | tail -50   # load errors
rocm-smi --showmeminfo vram                           # both cards present and loaded?
rocm-smi -t --showpower                               # temperatures and power under load
curl -s 127.0.0.1:8080/slots | python3 -m json.tool   # what each slot is doing
```
