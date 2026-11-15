#!/usr/bin/env python3
"""--parallel 2 check: one request alone, then two at once. Usage: np2-check.py PORT"""
import json, sys, threading, urllib.request
URL = f"http://127.0.0.1:{sys.argv[1]}"
P = ["Write a Python module implementing a thread-safe LRU cache with TTL and pytest tests.",
     "Write a bash script that rotates logs: gzip files older than 1 day, delete .gz older than 30 days."]
def one(p, out, i):
    b = {"messages": [{"role": "user", "content": p}], "max_tokens": 768, "temperature": 0.6,
         "chat_template_kwargs": {"reasoning_effort": "low"}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions",
        json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=900))
    out[i] = round(r["timings"]["predicted_per_second"], 1)
o = [None]; one(P[0], o, 0); print("single:", o[0], "t/s")
o = [None, None]; ts = [threading.Thread(target=one, args=(P[i], o, i)) for i in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
print("concurrent:", o, "t/s per slot, total", round(sum(o), 1))
