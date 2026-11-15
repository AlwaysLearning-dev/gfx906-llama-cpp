#!/usr/bin/env bash
# Speed sweep with the server settings: split mode x draft depth. Stop the production server first.
# Usage: mi50/bench/sweep.sh   Results: mi50-results/sweep.jsonl (bench2.py rows; depth -1 = file replay)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); O=${OUT:-$ROOT/mi50-results}; mkdir -p "$O"; P=${PORT:-8090}
one() { local label=$1; shift
  echo "=== $label"
  env "$@" PORT=$P "$ROOT/mi50/run-server.sh" > "$O/server-$label.log" 2>&1 & local pid=$!
  for _ in $(seq 1 180); do curl -sf "127.0.0.1:$P/health" >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 2; done
  REPS=${REPS:-2} python3 "$ROOT/mi50/bench/bench2.py" "$P" "$label" -1 0 20000 >> "$O/sweep.jsonl" 2>>"$O/sweep.err"
  kill $pid; wait $pid 2>/dev/null; sleep 5; }
one layer_d4 SM=layer NMAX=4
one tensor_d4 SM=tensor NMAX=4
one tensor_nodraft SM=tensor DRAFT=none
python3 - "$O/sweep.jsonl" <<'PY'
import json, sys, collections
r = collections.defaultdict(list)
for l in open(sys.argv[1]):
    d = json.loads(l); r[(d["label"], d["depth"])].append(d["tg_tps"])
for k in sorted(r): print(k, round(sum(r[k]) / len(r[k]), 1), "t/s")
PY
