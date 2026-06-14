#!/usr/bin/env python3
"""DSV4 Flash benchmark — full concurrency matrix."""
import json, time, concurrent.futures, statistics, subprocess, os

SERVER = "http://localhost:8000"
MODEL = "deepseek-v4-flash"
MAX_TOKENS = 256
N_RUNS = 3
OUTPUT = "/home/r0b0tdgx/spark-vllm-docker/results/benchmark_post_fix_driver580159.json"

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list of dictionaries by key.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis step by step.",
    "Write a short poem about artificial intelligence.",
    "How does a transformer neural network architecture work?",
    "Explain the difference between TCP and UDP protocols.",
    "What is the significance of the Turing test in AI?",
    "Write a SQL query to find the top 10 customers by revenue.",
    "Describe the water cycle and its importance.",
    "Explain the concept of recursion in programming with an example.",
    "What are the key principles of agile software development?",
    "How do vaccines work to protect against diseases?",
    "Write a JavaScript function to debounce another function.",
    "Explain the theory of general relativity in layman's terms.",
    "What is blockchain technology and how does it work?",
]

def send_request(prompt):
    import urllib.request
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "ignore_eos": True
    }).encode()
    req = urllib.request.Request(f"{SERVER}/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
            elapsed = time.time() - start
            tokens = body.get("usage", {}).get("completion_tokens", 0)
            return tokens, elapsed, tokens > 0
    except Exception as e:
        return 0, time.time() - start, False

def bench_concurrency(c):
    runs = []
    for run in range(N_RUNS):
        prompts = [PROMPTS[(run * c + i) % len(PROMPTS)] for i in range(c)]
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
            results = list(pool.map(send_request, prompts))
        elapsed = time.time() - start

        total_tokens = sum(r[0] for r in results)
        sr = sum(1 for r in results if r[2]) / c
        agg = total_tokens / elapsed if elapsed > 0 else 0
        per_req = agg / c

        runs.append({"tokens": total_tokens, "elapsed": round(elapsed, 1),
                      "agg_tps": round(agg, 2), "per_req_tps": round(per_req, 2), "sr": sr})
        print(f"  c={c} run {run+1}: {total_tokens} tok {elapsed:.1f}s = {agg:.1f} agg ({per_req:.1f}/req) SR={sr:.0%}")
        time.sleep(3)

    return {
        "concurrency": c,
        "agg_median": round(statistics.median([r["agg_tps"] for r in runs]), 2),
        "per_req_median": round(statistics.median([r["per_req_tps"] for r in runs]), 2),
        "sr_median": statistics.median([r["sr"] for r in runs]),
        "runs": runs
    }

# Warmup
print("Warming up...", flush=True)
send_request("Hello world")
time.sleep(3)

print(f"\n=== DSV4 Flash Benchmark (MoE+IB fix, driver 580.159.03, MAX_NUM_SEQS=16) ===", flush=True)
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

all_results = []
for c in [1, 2, 4, 8, 16]:
    print(f"\n--- c={c} ---", flush=True)
    result = bench_concurrency(c)
    all_results.append(result)
    time.sleep(5)

print("\n=== SUMMARY ===", flush=True)
print(f"{'C':>4} | {'Per-req':>10} | {'Aggregate':>10} | {'SR':>5}", flush=True)
print("-" * 45, flush=True)
for r in all_results:
    print(f"{r['concurrency']:>4} | {r['per_req_median']:>10.2f} | {r['agg_median']:>10.2f} | {r['sr_median']:>5.0%}", flush=True)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump({
        "date": "2026-06-14",
        "driver": "580.159.03",
        "kernel": "6.17.0-1021-nvidia",
        "mods": ["deepgemm-sm121", "moe-padding-fix"],
        "config": {"max_num_seqs": 16, "max_model_len": 65536, "mtp": 2, "tp": 2, "ep": True},
        "results": all_results
    }, f, indent=2)
print(f"\nSaved to {OUTPUT}", flush=True)
