# memtrace: where does llama-server's host memory go?

- `livealloc.c`: `LD_PRELOAD` shim that tracks live heap blocks (>= `LIVEALLOC_MIN` bytes, default 4096) with their call
  stacks. `kill -USR1 <pid>` writes live bytes per call stack and the allocator's statistics to `$LIVEALLOC_OUT.<n>`
  at the next allocation. Build: `gcc -O2 -fPIC -shared -o livealloc.so livealloc.c -ldl -lpthread`.
- `memprobe.py gen1|gen3|gen|read`: sends new chats with long replies (or long prompts) to the server on `$PORT` and
  prints the server's own memory before and after each.
- `report.py dump1 dump2 [n]`: the call stacks that grew most between two dumps, with function names (`addr2line`).

Typical run: start the server with `LD_PRELOAD=livealloc.so LIVEALLOC_OUT=out/live LIVEALLOC_MIN=96`, run
`memprobe.py gen1`, send `USR1`, run `memprobe.py gen3`, send `USR1`, then `report.py out/live.1 out/live.2`.
This is how the HIP graph leak in the optimization log (2026-11-18) was found.
