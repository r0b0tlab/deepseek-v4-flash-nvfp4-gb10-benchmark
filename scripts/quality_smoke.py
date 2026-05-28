#!/usr/bin/env python3
"""Quality smoke for DSV4-Flash. Must PASS before any perf number (GATE 0/4)."""
import json, urllib.request, sys
BASE = "http://127.0.0.1:8000/v1"; MODEL = "deepseek-v4-flash"
CASES = [
    ("The capital of France is", ["paris"]),
    ("Hello, world! In Python that prints", ["print"]),
    ("Jupiter is the largest planet. It is so large that more than", ["earth", "1,3", "130", "1300"]),
    ("2 + 2 =", ["4", "four"]),
    ("def fibonacci(n):", ["return", "fib", "n-1", "n - 1"]),
]
def gen(prompt, n=40):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": n,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(BASE + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["text"]
def main():
    passed = 0
    for prompt, needles in CASES:
        try:
            out = gen(prompt)
            ok = any(nd.lower() in out.lower() for nd in needles)
            print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r} -> {out[:80]!r}", flush=True)
            passed += ok
        except Exception as e:
            print(f"[ERR ] {prompt!r} -> {e}", flush=True)
    print(f"\n{passed}/{len(CASES)} passed", flush=True)
    sys.exit(0 if passed >= 4 else 1)
if __name__ == "__main__":
    main()
