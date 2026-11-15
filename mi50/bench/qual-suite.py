#!/usr/bin/env python3
"""Run the qwen3.8-effort-bench.py tasks (same folder) at medium effort REPS times. Usage: qual-suite.py PORT LABEL REPS OUT"""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("eb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3.8-effort-bench.py"))
eb = importlib.util.module_from_spec(spec); spec.loader.exec_module(eb)
port, label, reps, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
runs = []
for rep in range(reps):
    for t in eb.TASKS:
        r = eb.run_one(f"http://127.0.0.1:{port}", t, "medium")
        runs.append({"label": label, "rep": rep, "task": t["id"], **r})
        print(f"{label} rep{rep} {t['id']:18} {r['verdict']:6} {r['wall']:6.1f}s gen={r['completion_tokens']}", flush=True)
        json.dump(runs, open(out, "w"), indent=1, default=str)
