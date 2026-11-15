#!/usr/bin/env python3
"""(bench2: REPS env = seeds 1..N, prompt cache on so long prompts fill once)
Generation-speed bench against a llama-server. Usage: bench.py PORT LABEL [depths...]
depth 0 = short coding prompt; N = ~N-token code context + question. Prints JSON lines."""
import json, sys, time, urllib.request, glob, os, hashlib
port, label = sys.argv[1], sys.argv[2]
depths = [int(x) for x in sys.argv[3:]] or [0]
URL = f"http://127.0.0.1:{port}"
SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root: its sources are the prompt corpus
files = sorted(glob.glob(f"{SRC}/src/*.cpp")) + sorted(glob.glob(f"{SRC}/tools/server/*.cpp"))
CODE = "".join(f"\n// ===== {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)

def post(path, body, timeout=3600):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])

def ctx_text(n):
    lo, hi = 0, len(CODE)
    while hi - lo > 2000:  # binary search on chars for ~n tokens
        mid = (lo + hi) // 2
        if ntok(CODE[:mid]) < n: lo = mid
        else: hi = mid
    return CODE[:lo]

SHORT = ("Write a Python module implementing a thread-safe LRU cache class with get/put, "
         "a max size, TTL expiry, and hit/miss stats. Include type hints, docstrings and "
         "a pytest test file covering eviction, TTL and concurrency.")
QUESTION = ("Above is C++ source from llama.cpp. Explain how the server schedules slots and "
            "speculative decoding, citing function names, then write a refactored version of "
            "the slot-selection function with comments.")

# depth -1 = "replay": Claude Code-style turn where the model re-emits a file it already wrote
REPLAY_FILE = open(f"{SRC}/common/ngram-cache.cpp", errors="ignore").read()[:9000]
REPLAY = [{"role": "user", "content": "Write common/ngram-cache.cpp for me."},
          {"role": "assistant", "content": "```cpp\n" + REPLAY_FILE + "\n```"},
          {"role": "user", "content": "Rename every occurrence of `ngram_cache` to `ngram_store` and "
           "output the complete updated file in one code block, nothing else."}]

for d in depths:
    for temp, seed in [(0.6, k) for k in range(1, int(os.environ.get("REPS", "3")) + 1)]:
        if seed == 1:
            msgs = REPLAY if d == -1 else [{"role": "user",
                   "content": SHORT if d == 0 else ctx_text(d) + "\n\n" + QUESTION}]
        t = time.time()
        r = post("/v1/chat/completions", {"messages": msgs,
                 "max_tokens": 768, "temperature": temp, "seed": seed, "cache_prompt": True,
                 "chat_template_kwargs": {"reasoning_effort": "low"}, **({"max_tokens": 1536} if d == -1 else {})})
        tm = r.get("timings", {})
        ch = r["choices"][0]["message"]
        out = (ch.get("reasoning_content") or "") + "\n" + (ch.get("content") or "")
        print(json.dumps({"label": label, "depth": d, "temp": temp, "seed": seed,
            "prompt_n": tm.get("prompt_n"), "pp_tps": round(tm.get("prompt_per_second", 0), 1),
            "gen_n": tm.get("predicted_n"), "tg_tps": round(tm.get("predicted_per_second", 0), 2),
            "draft_n": tm.get("draft_n"), "draft_acc": tm.get("draft_n_accepted"),
            "wall_s": round(time.time() - t, 1), "out_hash": hashlib.sha1(out.encode()).hexdigest()[:12],
            "out_head": out[:80].replace("\n", " ")}), flush=True)
