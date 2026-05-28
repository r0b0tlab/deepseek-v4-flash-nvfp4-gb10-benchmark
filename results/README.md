# Results

Raw benchmark JSON (median-of-3, fixed prompts, 256 decode tokens):
- `latency_profile_200k_seqs2.json` — 200K ctx, 2 seqs (best single-stream)
- `throughput_profile_65k_seqs16.json` — 65K ctx, 16 seqs (best aggregate)

Each file: per-run agg/per-req tok/s + median/min/max summary per concurrency.
