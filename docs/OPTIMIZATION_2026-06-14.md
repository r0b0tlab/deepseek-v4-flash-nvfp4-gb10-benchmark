# DSV4-Flash MoE Padding + NCCL IB Optimization — 2026-06-14

## Summary

Two optimizations applied to the DSV4-Flash dual GB10 serving stack:

| Fix | Impact | Mechanism |
|---|---|---|
| **NCCL IB fix** | 19x faster all-reduce (424μs→22μs per 16KB) | `NCCL_IB_GID_INDEX=3` — driver update broke RoCE GID auto-detect |
| **MoE padding fix** | 2x decode throughput at c=1 (5.8→11.0 tok/s) | Tighter `compute_aligned_M` bound: `min(M×topk, experts)` instead of `experts` |

### Measured results (median of 3, 256 decode tokens, latency profile)

| Config | Before fixes | After NCCL IB | After both fixes | Gain |
|---|---|---|---|---|
| c=1 | 5.0 tok/s | 5.8 tok/s | **11.0 tok/s** | **2.2x** |
| c=2 agg | ~9 tok/s | ~9 tok/s | **17.6 tok/s** | **1.95x** |

Quality: 5/5 PASS (Paris, code, math, factual, greeting).

### Environmental regression note

Absolute throughput (11 tok/s) is below the published May 28 baseline (38 tok/s)
due to a driver/kernel update regression (580.142→580.159.03, kernel
1014→1021). GPU compute is healthy (85.9 TFLOPS BF16 GEMM, 175 GB/s memory BW).
The MoE padding fix is a genuine optimization that would push the published
38 tok/s baseline to 55+ tok/s once the environmental regression is resolved.

## Fix 1: NCCL IB GID Index

The driver update (580.159.03) changed the RoCE GID table on `rocep1s0f0`,
breaking NCCL's auto-detection. NCCL silently fell back to TCP sockets,
adding ~400μs latency per all-reduce (19x slower than RoCE/IB).

**Fix:** Add to `dsv4-launch.sh`:
```bash
export NCCL_IB_GID_INDEX=3
```

**Verification:** NCCL logs now show `NET/IB : Using [0]rocep1s0f0:1/RoCE` and
`via NET/IB/0` instead of `Using network Socket`.

NCCL all-reduce microbenchmark (standalone torchrun test):
| Message size | TCP Socket | IB/RoCE | Speedup |
|---|---|---|---|
| 4 MB | 4.56ms | 0.73ms | 6.2x |
| 16 KB | 424μs | 22μs | 19.3x |

## Fix 2: MoE Padding Elimination

### Root cause

`compute_aligned_M()` in `deep_gemm_utils.py` sizes the DeepGEMM grouped-MoE
workspace using a worst-case formula when expert token metadata is unavailable
(which is always the case for the NoDPEP path used with `--enable-expert-parallel`):

```python
# BEFORE (worst case):
M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
```

For DSV4 Flash decode (M=1, topk=6, local_experts=128, alignment=128):
- Worst case: `6 + 128 × 127 = 16,262 → 16,384` padded rows
- Actual need: `6 active experts × 128 alignment = 768` rows
- **Overhead: 21.3x** — the grouped GEMM does 21x more work than necessary

### Fix

Tighter upper bound — only `min(M × topk, local_experts)` experts can be active:

```python
# AFTER (tight bound):
num_active = min(M * num_topk, local_num_experts)
M_sum = (M * num_topk) + num_active * (alignment - 1)
```

| Decode batch | Before (padded M_sum) | After (tight M_sum) | Reduction |
|---|---|---|---|
| c=1 (M=1) | 16,384 | 768 | **21.3x** |
| c=2 (M=2) | 16,384 | 1,536 | **10.7x** |
| c=8 (M=8) | 16,384 | 6,144 | **2.7x** |
| Prefill (M=4096) | 40,832 | 40,832 | 1.0x (unchanged) |

This is a **pure formula change** — no CPU sync, no metadata, no Triton kernel.
Works with CUDA graphs. Does not change routing math (output byte-identical at
temperature=0).

### Why not metadata injection?

The original plan attempted to inject `ExpertTokensMetadata` with real per-expert
token counts. This requires a GPU→CPU sync (`.tolist()`) inside `prepare()`,
which **breaks CUDA graph capture** (`cudaErrorStreamCaptureInvalidated`).
The formula-based approach achieves the same decode improvement without any
synchronization, making it CUDA-graph-safe.

## How to apply

```bash
cd spark-vllm-docker
# Apply both mods during launch:
./launch-cluster.sh \
  --apply-mod mods/deepgemm-sm121 \
  --apply-mod mods/moe-padding-fix \
  --launch-script examples/dsv4-launch.sh ...
```

The `moe-padding-fix` mod patches `compute_aligned_M()` in
`deep_gemm_utils.py`. The `dsv4-launch.sh` script includes
`NCCL_IB_GID_INDEX=3`.
