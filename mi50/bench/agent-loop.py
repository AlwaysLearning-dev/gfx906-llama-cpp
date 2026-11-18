#!/usr/bin/env python3
"""Does dropping old thinking help? Simulated agent session (grep/read tools on a source tree, several user tasks in
one conversation), run with preserve_reasoning on or off per request. Usage: agent-loop.py PORT on|off SEED REPO [SLOT]
Prints per-task and total: prompt tokens the server had to process, context size, generated tokens, wall time, answers."""
import json, sys, time, urllib.request, subprocess, os, re
port, mode, seed, REPO = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
slot = int(sys.argv[5]) if len(sys.argv) > 5 else None
URL = f"http://127.0.0.1:{port}/v1/chat/completions"
TOOLS = [{"type": "function", "function": {"name": "grep", "description": "Search for a regular expression in a file or directory of the repository. Returns matching lines with line numbers (max 40).",
          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "file or directory, relative to the repository root"}}, "required": ["pattern", "path"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read lines of a file in the repository.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start": {"type": "integer", "description": "first line, 1-based"}, "lines": {"type": "integer", "description": "how many lines (max 120)"}}, "required": ["path"]}}}]
def safe(p):
    full = os.path.realpath(os.path.join(REPO, p)); return full if full.startswith(os.path.realpath(REPO)) else None
def run_tool(name, a):
    try:
        p = safe(a.get("path", "."))
        if not p or not os.path.exists(p): return "error: no such path"
        if name == "grep":
            o = subprocess.run(["grep", "-rnE", "--include=*.cpp", "--include=*.h", "--include=*.sh", "--include=*.md", "-e", a["pattern"], p], capture_output=True, text=True, timeout=20).stdout
            return "\n".join(l.replace(os.path.realpath(REPO) + "/", "")[:300] for l in o.split("\n")[:40]) or "no matches"
        if name == "read_file":
            s = max(1, int(a.get("start", 1))); n = min(120, int(a.get("lines", 80)))
            L = open(p, errors="ignore").read().split("\n")
            return "\n".join("%d: %s" % (s + i, l[:300]) for i, l in enumerate(L[s-1:s-1+n]))
    except Exception as e: return f"error: {e}"
    return "error: unknown tool"
TASKS = [
 ("How many context checkpoints does mi50/run-server.sh keep by default, and which environment variable changes that? Answer in one line.", ["6", "CKPT"]),
 ("Which environment variable can set --spec-draft-sampling instead of the command line flag? Look in common/arg.cpp. Answer in one line.", ["LLAMA_ARG_SPEC_DRAFT_SAMPLING"]),
 ("In tools/server/server-chat.cpp, which request keys does the Anthropic to OpenAI conversion pass through unchanged in its 'common params' loop? List them.", ["id_slot", "message_dedup", "top_k"]),
 ("Without using any tool: from what you found in the first task, write the exact flag and value run-server.sh would pass if I set that variable to 12, and tell me the default prompt cache limit in MiB that the same script uses.", ["--ctx-checkpoints 12", "4096"]),
 ("In common/sampling.cpp find the function that does the rejection-sampling accept for drafted tokens. What is its name and what does it return? One or two lines.", ["common_sampler_sample_and_accept_n_rejection"]),
 ("Without using any tool: list every file you have opened or searched in this conversation and, for each, the one fact you took from it.", ["run-server.sh", "arg.cpp", "server-chat.cpp", "sampling.cpp"]),
]
msgs = [{"role": "system", "content": "You are a coding agent working in a C++ repository. Use the tools to look things up before answering. Be brief."}]
tot = dict(pp=0, gen=0, think=0, wall=0.0, steps=0, ok=0, pms=0.0)
for ti, (q, keys) in enumerate(TASKS):
    msgs.append({"role": "user", "content": q}); t_pp = t_gen = 0; t0 = time.time(); ctx = 0; ans = ""
    for step in range(12):
        body = {"messages": msgs, "tools": TOOLS, "max_tokens": 4096, "temperature": 0.6, "seed": seed + ti * 100 + step, "cache_prompt": True,
                "chat_template_kwargs": {"preserve_reasoning": mode == "on"}}
        if slot is not None: body["id_slot"] = slot
        r = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
        tm = r["timings"]; m = r["choices"][0]["message"]
        t_pp += tm["prompt_n"]; t_gen += tm["predicted_n"]; tot["pms"] += tm["prompt_ms"]; ctx = r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]; tot["steps"] += 1
        a = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("reasoning_content"): a["reasoning_content"] = m["reasoning_content"]; tot["think"] += len(m["reasoning_content"])
        if m.get("tool_calls"):
            a["tool_calls"] = m["tool_calls"]; msgs.append(a)
            for tc in m["tool_calls"]:
                try: args = json.loads(tc["function"]["arguments"])
                except Exception: args = {}
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": run_tool(tc["function"]["name"], args)})
            continue
        msgs.append(a); ans = a["content"]; break
    ok = all(k.lower() in ans.lower() for k in keys); tot["ok"] += ok
    w = time.time() - t0; tot["pp"] += t_pp; tot["gen"] += t_gen; tot["wall"] += w
    print("   task %d  %-4s steps %2d  prompt processed %6d  generated %5d  context %6d  wall %5.1f s  | %s" % (ti + 1, "PASS" if ok else "FAIL", step + 1, t_pp, t_gen, ctx, w, ans.replace("\n", " ")[:110]), flush=True)
print("   TOTAL %s seed %d: %d/%d correct, %d steps, prompt processed %d tok (%.1f s), generated %d tok, final context %d, thinking kept by client %d chars, wall %.0f s"
      % (mode, seed, tot["ok"], len(TASKS), tot["steps"], tot["pp"], tot["pms"] / 1000, tot["gen"], ctx, tot["think"], tot["wall"]), flush=True)
json.dump(msgs, open(f"{os.path.dirname(os.path.abspath(__file__))}/out/msgs-{mode}-{seed}.json", "w"))
