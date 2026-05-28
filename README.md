# DeepSeek-V4-Flash on Dual DGX Spark (GB10 / SM121) — Native Blackwell FP8 Benchmark

Reproducible benchmark of **DeepSeek-V4-Flash** (official FP8) served across
**2× NVIDIA DGX Spark (GB10, SM121 Blackwell)** with tensor-parallel TP=2 over
RoCE, using the **fully native Blackwell tensor-core path** — DeepGEMM block-scaled
FP8 + MXFP4 MoE, sparse MLA, Lightning Indexer, MTP speculative decode. No Marlin,
no emulation, no CPU fallback.

## Headline results (median of 3 runs, fixed prompts, 256 decode tokens)

### Latency profile — 200K context, 2 seqs (CUDA graphs)
| Concurrency | Per-req decode (t/s) | Aggregate (t/s) | Reference | % of ref |
|---|---|---|---|---|
| 1 | **38.4** | 38.4 | 44 | 87% |
| 2 | 28.0 | **53.6** | ~45 | **119%** |
| 4 | 20.7 | 54.0 | — | — |
| 8 | 11.7 | 54.7 | — | — |

### Throughput profile — 65K context, 16 seqs (CUDA graphs)
| Concurrency | Per-req decode (t/s) | Aggregate (t/s) | Reference | % of ref |
|---|---|---|---|---|
| 1 | 36.1 | 36.1 | 44 | 82% |
| 2 | 28.8 | 57.0 | ~45 | 127% |
| 4 | 15.7 | 62.8 | — | — |
| 8 | 14.5 | **101.5** | ~96 | **106%** |
| 16 | 9.5 | **144.6** | — | — |

Quality: 5/5 smoke (factual / math / code) PASS. GPUs 33W, 68°C, full 2.5 GHz,
no thermal throttling.

## Why it's fast: the build is the lever

The decisive factor for DSV4-Flash decode throughput on GB10 is the **vLLM build
commit**, not runtime flags. We build vLLM from the pinned commit
`dda4668b59567416f86956cfe7bbc1eab371a61e` (jasl/vllm) for `sm_121a`, which:

- Recognizes SM121 (capability 12.1) as Blackwell **family-120** via
  `is_device_capability_family(120)` → routes the full native FP8/MXFP4 path.
- Includes the **rowwise paged-MQA logits decode kernel** tuned for long context.
- Needs **no custom patches/mods**.

An earlier branch-alias build lacking this path floored decode at ~5 tok/s.
Pinning the commit lifted single-stream to ~38 tok/s — a **7.5× improvement**.

## Reproduce

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the full build + launch recipe,
including the three build/launch fixes (CUDA-torch re-pin, env vars, container
restart for port reuse).

## Reference baseline

NVIDIA Developer Forums — "DeepSeek-V4-Flash (official FP8) running across 2x DGX
Spark — TP=2, MTP, 200K ctx": ~44 tok/s c=1, ~45 tok/s c=2 aggregate, ~96 tok/s
c=8 aggregate, on identical hardware.

---
Built and benchmarked by [@mr-r0b0t](https://x.com/mr-r0b0t) · r0b0tlab
