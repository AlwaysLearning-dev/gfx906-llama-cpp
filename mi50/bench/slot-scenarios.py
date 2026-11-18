#!/usr/bin/env python3
"""Slot pinning + checkpoint cost scenarios. Usage: slot-scenarios.py PORT SERVER_LOG SERVER_PID REPO
Prints one line per request: route, slot chosen and how, prompt tokens processed, wall, overhead (wall - prompt - gen)."""
import json, sys, time, urllib.request, glob, os, re
port, LOG, PID, SRC = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
URL = f"http://127.0.0.1:{port}"
files = sorted(glob.glob(f"{SRC}/src/*.cpp")) + sorted(glob.glob(f"{SRC}/tools/server/*.cpp")) + sorted(glob.glob(f"{SRC}/common/*.cpp"))
CODE = "".join(f"\n// ===== {os.path.basename(f)}\n" + open(f, errors="ignore").read() for f in files)
def chunk(start_tok, n_tok):   # ~3.2 chars per token of C++
    return CODE[int(start_tok*3.2):int((start_tok+n_tok)*3.2)]
def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))
def hwm():
    for l in open(f"/proc/{PID}/status"):
        if l.startswith("VmHWM"): return int(l.split()[1]) / 1048576
def rss():
    for l in open(f"/proc/{PID}/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 1048576
def req(name, msgs, route="oai", slot=None, max_tokens=48):
    off = os.path.getsize(LOG)
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.6, "seed": 1}
    if slot is not None: body["id_slot"] = slot
    if route == "oai":
        body.update({"cache_prompt": True, "chat_template_kwargs": {"reasoning_effort": "low"}}); path = "/v1/chat/completions"
    else:
        body["model"] = "Qwen3.8-27B"; path = "/v1/messages"
    t = time.time(); r = post(path, body); wall = time.time() - t
    time.sleep(0.5)
    new = open(LOG, errors="ignore").read()[off:]
    sel = re.findall(r"id\s+(\d+) \| task [-\d]+ \| selected slot by ([\w ]+)", new)
    pe = re.findall(r"prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens", new)
    ev = re.findall(r"\|\s+eval time =\s+([\d.]+) ms /\s+(\d+) tokens", new)
    pms, pn = (float(pe[-1][0]), int(pe[-1][1])) if pe else (0, -1)
    gms = float(ev[-1][0]) if ev else 0
    how = ("slot %s by %s" % sel[-1]) if sel else "slot ?"
    print("   %-26s %-4s %-22s prompt %6d tok  wall %6.1f s  overhead %5.2f s  rss %4.1f GB" % (name, route, how.strip(), pn, wall, wall - pms/1000 - gms/1000, rss()), flush=True)
    if route == "oai": return r["choices"][0]["message"].get("content") or "ok"
    return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text") or "ok"

LONG=int(os.environ.get("LONG","20000")); PIN=os.environ.get("PIN","1")=="1"; P0,P1=(0,1) if PIN else (None,None)
if os.environ.get("S1","1")=="1":
  print(" S1 checkpoint cost: grow a conversation to several checkpoints on slot 0, displace it, come back")
  A = [{"role": "user", "content": chunk(0, 6000) + "\n\nSummarise this file in two sentences."}]
  for i in range(6):
      a = req(f"A turn {i+1}", A, slot=0)
      A += [{"role": "assistant", "content": a}, {"role": "user", "content": chunk(6000 + i*2500, 2500) + "\n\nAnd this part, two sentences."}]
  B = [{"role": "user", "content": chunk(40000, 3000) + "\n\nWhat does this code do? One sentence."}]
  for k in range(3):
      req("B displaces A (slot 0)", B, slot=0)
      a = req("A returns (slot 0)", A, slot=0)
  print("   peak server RSS after S1: %.2f GB" % hwm())
print(" S2 eviction replay on the Anthropic route: long conversation pinned to 0, two other clients pinned to 1")
L = [{"role": "user", "content": chunk(60000, LONG) + "\n\nSummarise this in two sentences."}]
a = req("long turn 1 (want 0)", L, route="anth", slot=P0)
L += [{"role": "assistant", "content": a}, {"role": "user", "content": "Now name the three most important functions."}]
req("other client X (want 1)", [{"role": "user", "content": chunk(2000, 2000) + "\n\nOne sentence summary."}], route="anth", slot=P1)
req("other client Y (want 1)", [{"role": "user", "content": chunk(9000, 2000) + "\n\nOne sentence summary."}], route="anth", slot=P1)
req("other client Z (want 1)", [{"role": "user", "content": chunk(15000, 2000) + "\n\nOne sentence summary."}], route="anth", slot=P1)
req("long turn 2 (want 0)", L, route="anth", slot=P0)
print("   peak server RSS at end: %.2f GB" % hwm())
