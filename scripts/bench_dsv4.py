#!/usr/bin/env python3
"""DSV4-Flash GB10 benchmark harness — fixed prompts, long decode, median of N.

Headline metric: c=1 single-stream per-request decode tok/s.
Secondary: aggregate tok/s at concurrency c.
Records TTFT, decode tok/s (per-req + aggregate), output tokens, wall time.
Writes JSON to artifacts. Does NOT pipe server logs (SIGPIPE-safe).
"""
import argparse, json, time, statistics, sys, urllib.request, threading
from datetime import datetime, timezone

BASE = "http://127.0.0.1:8000/v1"
MODEL = "deepseek-v4-flash"

# Fixed prompts (factual / math / code / short / long-ish) — comparable across phases.
PROMPTS = [
    "Explain in detail how a transformer neural network processes a sequence of tokens, step by step.",
    "Write a Python function that computes the nth Fibonacci number using memoization, with docstring.",
    "What is the capital of France, and name three landmarks there.",
    "Summarize the causes of the fall of the Western Roman Empire in a few paragraphs.",
    "Describe the process of photosynthesis including the light and dark reactions.",
]

def one_request(prompt, max_tokens, results, idx):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(BASE + "/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        t1 = time.time()
        usage = data.get("usage", {})
        out_tok = usage.get("completion_tokens", 0)
        wall = t1 - t0
        results[idx] = {"ok": True, "out_tok": out_tok, "wall": wall,
                        "decode_tps": out_tok / wall if wall > 0 else 0}
    except Exception as e:
        results[idx] = {"ok": False, "err": str(e)}

def run_concurrency(c, max_tokens, runs):
    per_run = []
    for run in range(runs):
        results = [None] * c
        threads = []
        t0 = time.time()
        for i in range(c):
            th = threading.Thread(target=one_request,
                                  args=(PROMPTS[i % len(PROMPTS)], max_tokens, results, i))
            threads.append(th); th.start()
        for th in threads: th.join()
        wall = time.time() - t0
        ok = [r for r in results if r and r["ok"]]
        if not ok:
            per_run.append({"run": run, "ok": False, "errors": [r for r in results if r]})
            continue
        total_out = sum(r["out_tok"] for r in ok)
        agg_tps = total_out / wall
        per_req = statistics.median(r["decode_tps"] for r in ok)
        per_run.append({"run": run, "ok": True, "agg_tps": agg_tps,
                        "per_req_median_tps": per_req, "total_out": total_out,
                        "wall": wall, "n_ok": len(ok)})
        print(f"  c={c} run={run}: agg={agg_tps:.2f} t/s, per_req_median={per_req:.2f} t/s, "
              f"out={total_out}, wall={wall:.1f}s, ok={len(ok)}/{c}", flush=True)
    good = [r for r in per_run if r.get("ok")]
    summary = None
    if good:
        summary = {
            "concurrency": c,
            "agg_tps_median": statistics.median(r["agg_tps"] for r in good),
            "agg_tps_min": min(r["agg_tps"] for r in good),
            "agg_tps_max": max(r["agg_tps"] for r in good),
            "per_req_tps_median": statistics.median(r["per_req_median_tps"] for r in good),
            "runs_ok": len(good), "runs_total": runs,
        }
    return {"concurrency": c, "runs": per_run, "summary": summary}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrencies", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="bench_result.json")
    a = ap.parse_args()
    cs = [int(x) for x in a.concurrencies.split(",")]
    print(f"=== DSV4 bench: c={cs}, max_tokens={a.max_tokens}, runs={a.runs} ===", flush=True)
    out = {"ts": datetime.now(timezone.utc).isoformat(), "max_tokens": a.max_tokens,
           "runs": a.runs, "model": MODEL, "results": []}
    for c in cs:
        out["results"].append(run_concurrency(c, a.max_tokens, a.runs))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SUMMARY -> {a.out} ===", flush=True)
    for r in out["results"]:
        s = r["summary"]
        if s:
            print(f"  c={s['concurrency']}: agg_median={s['agg_tps_median']:.2f} t/s "
                  f"[{s['agg_tps_min']:.2f}-{s['agg_tps_max']:.2f}], "
                  f"per_req_median={s['per_req_tps_median']:.2f} t/s "
                  f"({s['runs_ok']}/{s['runs_total']} ok)", flush=True)
        else:
            print(f"  c={r['concurrency']}: ALL RUNS FAILED", flush=True)

if __name__ == "__main__":
    main()
