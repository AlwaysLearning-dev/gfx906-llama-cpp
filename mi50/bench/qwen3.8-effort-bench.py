#!/usr/bin/env python3
"""
Qwen3.8-27B reasoning_effort head-to-head: low vs medium vs xhigh.

Runs a small fixed suite of 8 tasks (structured output, coding with runnable
asserts, tool-use format check, and tricky reasoning) against the local
llama-server's OpenAI-compat endpoint. Each task runs once per tier; results
are scored PASS / FAIL / BUDGET (hit max_tokens mid-think) with objective
checkers where possible. Prints a live scorecard and dumps raw responses
to a JSON file for later inspection.

Usage:
  python3 qwen3.8-effort-bench.py                              # all three tiers
  python3 qwen3.8-effort-bench.py --tiers low medium           # subset
  python3 qwen3.8-effort-bench.py --url http://localhost:8080 --out /tmp/bench.json

Talks straight to llama-server on :8080 by default (not through any proxy in
front of it) - we want to measure the model, not a proxy.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

# ------------------------------ HTTP client ------------------------------

def chat(url, messages, max_tokens, effort, tools=None, timeout=900):
    body = {
        "model": "Qwen3.8-27B",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ch = d["choices"][0]
    msg = ch["message"]
    usage = d.get("usage", {})
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "wall": dt,
    }

# ------------------------------ helpers ------------------------------

def extract_code(text, lang=None):
    pat = r"```(?:%s)?\s*\n(.*?)```" % (lang or r"python|py|")
    m = re.findall(pat, text, re.DOTALL)
    return m[-1] if m else None

def run_python(code, appended_test, timeout=15):
    src = code + "\n\n" + appended_test
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr).strip()[-200:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try: os.unlink(path)
        except: pass

def extract_first_json(text):
    """Return first well-formed JSON object found in text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return None
    return None

# ------------------------------ task definitions ------------------------------
# Each task: dict with id, category, prompt (str or messages), checker(response)
#   -> returns (passed: bool, note: str)
# tools optional for tool-use tasks.

def check_sigma_rule(r):
    c = r["content"]
    # Look for a YAML fenced block containing key Sigma fields
    m = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", c, re.DOTALL)
    body = m.group(1) if m else c
    needed = ["title:", "logsource:", "detection:", "condition:"]
    hits = sum(1 for k in needed if k in body)
    return hits == 4, f"{hits}/4 required keys"

def check_json_shape(r):
    obj = extract_first_json(r["content"])
    if not isinstance(obj, dict):
        return False, "no top-level JSON object"
    required = {"user_id", "email", "roles", "created_at"}
    missing = required - set(obj.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    if not isinstance(obj.get("roles"), list):
        return False, "roles is not a list"
    return True, "shape ok"

def check_primes(r):
    code = extract_code(r["content"])
    if not code:
        return False, "no python block"
    ok, out = run_python(code, "assert [n for n in range(50) if is_prime(n)] == [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]; print('OK')")
    return ok, out

def check_qsort(r):
    code = extract_code(r["content"])
    if not code:
        return False, "no python block"
    ok, out = run_python(code, "import random; random.seed(0); a=[random.randint(0,99) for _ in range(50)]; assert qsort(list(a))==sorted(a); print('OK')")
    return ok, out

def check_bugfix(r):
    code = extract_code(r["content"])
    if not code:
        return False, "no python block"
    ok, out = run_python(code, "assert my_max([-5,-2,-9])==-2; assert my_max([3,1,4,1,5])==5; assert my_max([-1])==-1; print('OK')")
    return ok, out

def check_weather_toolcall(r):
    if r["tool_calls"]:
        tc = r["tool_calls"][0]
        name = tc.get("function", {}).get("name") if isinstance(tc, dict) else None
        args_raw = tc.get("function", {}).get("arguments") if isinstance(tc, dict) else None
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw or {}
        except Exception:
            args = {}
        city_ok = str(args.get("city", "")).strip().lower() == "paris"
        return (name == "get_weather" and city_ok), f"name={name} city={args.get('city')!r}"
    # Fallback: sometimes model emits tool call as text; look for the JSON
    obj = extract_first_json(r["content"])
    if isinstance(obj, dict):
        args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
        if str(args.get("city", "")).strip().lower() == "paris":
            return True, "tool call in content JSON, city=paris"
    return False, "no valid tool call found"

def check_logic_alice(r):
    # Answer: Carol is the liar. The prompt asks for ONLY the name.
    txt = (r["content"] or "").strip()
    if not txt:
        return False, "empty content"
    # Strip markdown/bold and take the last non-empty line
    lines = [l.strip("* _`").strip() for l in txt.splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    # 1) final line is exactly "Carol" (in any case, possibly with punctuation)
    if re.fullmatch(r"[*_`]*carol[*_`.!:]*", last, re.I):
        return True, "final line = Carol"
    # 2) tail mentions Carol + (liar|lying|lies) close together, and does NOT
    #    conclude with Alice or Dave as the named liar
    tail = txt[-400:].lower()
    carol_close_to_liar = bool(re.search(r"\bcarol\b[^.]{0,60}\b(liar|lying|lies)\b|\b(liar|lying|lies)\b[^.]{0,60}\bcarol\b", tail))
    named_wrong = re.search(r"\b(alice|dave)\b\s+is\s+(?:the\s+)?(liar|lying)", tail) is not None
    if carol_close_to_liar and not named_wrong:
        return True, "Carol identified as liar in conclusion"
    return False, f"tail: {tail[-140:]!r}"

def check_arithmetic(r):
    # A train leaves A at 60 mph; a second leaves B (300 mi apart) at 40 mph toward A. When do they meet? -> 3 hours.
    txt = r["content"] or ""
    # Strip LaTeX commands and formatting so `\boxed{3}`, `$$...$$`, `\text{ hours}` etc. don't hide the answer.
    stripped = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", txt)   # unwrap \boxed{3}, \text{ hours}
    stripped = re.sub(r"\\[a-zA-Z]+", " ", stripped)             # drop bare \frac, \cdot etc
    stripped = re.sub(r"[$*_`]", " ", stripped)                   # drop math delimiters + markdown
    stripped = re.sub(r"\s+", " ", stripped)
    tail = stripped[-500:]
    # 1) "3 hours" / "3 hrs" / "3 h" in the tail
    if re.search(r"\b3\s*(hours?|hrs?|h)\b", tail, re.I):
        return True, "3 hours"
    # 2) written form
    if re.search(r"\bthree\s+hours\b", tail, re.I):
        return True, "three hours"
    # 3) 180 minutes
    if re.search(r"\b180\s*(minutes?|mins?)\b", tail, re.I):
        return True, "180 minutes"
    # 4) bare "3" appearing near an "hour" mention (e.g. "= 3\n\nhours" from LaTeX)
    if "hour" in tail.lower() and re.search(r"(?<![\d.])3(?![\d.])", tail):
        # ensure the 3 isn't inside speed 60/40/300 or similar
        # split around numeric context
        return True, "3 near 'hour'"
    return False, f"answer not found (tail: {tail[-120:]!r})"

TASKS = [
    {
        "id": "sigma_rule",
        "category": "structured",
        "prompt": (
            "Write a Sigma rule (YAML) that detects Windows PowerShell downloading "
            "a file from an external HTTP(S) URL via Invoke-WebRequest, "
            "Invoke-RestMethod, or the WebClient .NET class. Include title, "
            "description, logsource (product windows, category process_creation), "
            "detection block with selection + condition, false positives, level, "
            "and MITRE ATT&CK tags. Output only the YAML in a fenced block."
        ),
        "checker": check_sigma_rule,
        "max_tokens": 32000,
    },
    {
        "id": "json_user",
        "category": "structured",
        "prompt": (
            "Produce a single JSON object describing a new user with fields: "
            "user_id (int), email (string), roles (array of strings), created_at "
            "(ISO-8601 string). Use plausible example values. Output only the "
            "JSON object, no prose."
        ),
        "checker": check_json_shape,
        "max_tokens": 32000,
    },
    {
        "id": "primes",
        "category": "coding",
        "prompt": (
            "Write a Python function is_prime(n) that returns True iff n is a "
            "prime number. Handle n < 2 correctly. Return ONLY the function in "
            "a ```python fenced block."
        ),
        "checker": check_primes,
        "max_tokens": 32000,
    },
    {
        "id": "qsort",
        "category": "coding",
        "prompt": (
            "Write a Python function qsort(lst) that returns a NEW sorted list "
            "using the quicksort algorithm. Do not use list.sort or sorted(). "
            "Return ONLY the function in a ```python fenced block."
        ),
        "checker": check_qsort,
        "max_tokens": 32000,
    },
    {
        "id": "bugfix",
        "category": "coding",
        "prompt": (
            "The following function should return the max of a list, but it has "
            "a bug (fails on all-negative lists). Fix it and return the "
            "corrected full function in a ```python block, no explanation.\n\n"
            "```python\n"
            "def my_max(xs):\n"
            "    m = 0\n"
            "    for x in xs:\n"
            "        if x > m: m = x\n"
            "    return m\n"
            "```"
        ),
        "checker": check_bugfix,
        "max_tokens": 32000,
    },
    {
        "id": "toolcall_weather",
        "category": "tool-use",
        "prompt": "What's the weather in Paris right now?",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "City name"}},
                    "required": ["city"],
                },
            },
        }],
        "checker": check_weather_toolcall,
        "max_tokens": 32000,
    },
    {
        "id": "logic_alice",
        "category": "reasoning",
        "prompt": (
            "Alice, Dave, and Carol each say one statement. Exactly one of them is "
            "lying (the other two tell the truth). Alice says: 'Dave is telling the "
            "truth.' Dave says: 'Carol is telling the truth.' Carol says: 'Dave is "
            "lying.' Who is the liar? Give ONLY the name."
        ),
        "checker": check_logic_alice,
        "max_tokens": 32000,
    },
    {
        "id": "arithmetic_train",
        "category": "reasoning",
        "prompt": (
            "Train A leaves station X heading east at 60 mph. At the same moment, "
            "train B leaves station Y heading west toward X at 40 mph. The two "
            "stations are 300 miles apart. How many hours until the trains meet? "
            "Give the final numeric answer in hours."
        ),
        "checker": check_arithmetic,
        "max_tokens": 32000,
    },
]

# ------------------------------ runner ------------------------------

def run_one(url, task, effort):
    messages = [{"role": "user", "content": task["prompt"]}]
    try:
        resp = chat(url, messages, task["max_tokens"], effort, tools=task.get("tools"))
    except Exception as e:
        return {"passed": False, "verdict": "ERROR", "note": str(e)[:200],
                "wall": 0, "prompt_tokens": None, "completion_tokens": None,
                "reasoning_len": 0, "content_len": 0, "raw": None}
    # BUDGET: hit max_tokens without emitting content (stuck in <think>)
    if resp["finish"] == "length" and not resp["content"] and not resp["tool_calls"]:
        return {"passed": False, "verdict": "BUDGET", "note": "hit max_tokens in reasoning",
                "wall": resp["wall"], "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "reasoning_len": len(resp["reasoning"]),
                "content_len": len(resp["content"]),
                "raw": {"reasoning_preview": resp["reasoning"][:400]}}
    passed, note = task["checker"](resp)
    return {"passed": passed, "verdict": "PASS" if passed else "FAIL",
            "note": note, "wall": resp["wall"],
            "prompt_tokens": resp["prompt_tokens"],
            "completion_tokens": resp["completion_tokens"],
            "reasoning_len": len(resp["reasoning"]),
            "content_len": len(resp["content"]),
            "raw": {"content": resp["content"][:800],
                    "reasoning_preview": resp["reasoning"][:300],
                    "tool_calls": resp["tool_calls"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--tiers", nargs="+", default=["low", "medium", "xhigh"],
                    choices=["low", "medium", "xhigh"])
    ap.add_argument("--out", default=os.path.expanduser("~/qwen3.8-effort-bench-results.json"))
    ap.add_argument("--tasks", nargs="+", help="Subset of task ids to run")
    args = ap.parse_args()

    tasks = TASKS
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in TASKS if t["id"] in wanted]

    print(f"URL: {args.url}   tiers: {args.tiers}   tasks: {len(tasks)}")
    print("-" * 90)

    results = {"url": args.url, "tiers": args.tiers, "runs": []}
    tally = {t: {"pass": 0, "fail": 0, "budget": 0, "error": 0,
                 "wall_total": 0.0, "gen_tokens_total": 0} for t in args.tiers}

    for task in tasks:
        for tier in args.tiers:
            print(f"  [{task['category']:11s}] {task['id']:22s} tier={tier:7s} ...", end="", flush=True)
            r = run_one(args.url, task, tier)
            results["runs"].append({"task": task["id"], "category": task["category"],
                                    "tier": tier, **r})
            v = r["verdict"].lower()
            if v == "pass": tally[tier]["pass"] += 1
            elif v == "budget": tally[tier]["budget"] += 1
            elif v == "error": tally[tier]["error"] += 1
            else: tally[tier]["fail"] += 1
            tally[tier]["wall_total"] += r["wall"] or 0
            tally[tier]["gen_tokens_total"] += r["completion_tokens"] or 0
            note = (r["note"] or "")[:60]
            print(f" {r['verdict']:6s}  {r['wall']:5.1f}s  gen={r['completion_tokens'] or '?':>5}  note={note}")

    print("-" * 90)
    print("SUMMARY")
    print(f"{'tier':>8s}  {'pass':>4s} {'fail':>4s} {'budget':>6s} {'err':>3s}   {'wall':>7s}   {'gen_tok':>8s}")
    for tier in args.tiers:
        t = tally[tier]
        n = t["pass"] + t["fail"] + t["budget"] + t["error"]
        print(f"{tier:>8s}  {t['pass']:>4d} {t['fail']:>4d} {t['budget']:>6d} {t['error']:>3d}   "
              f"{t['wall_total']:>6.1f}s   {t['gen_tokens_total']:>8d}   ({n} tasks)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results: {args.out}")


if __name__ == "__main__":
    main()
