# TCQ3 set_rows optimization notes, 2026-04-29

These notes summarize the CUDA TCQ3 KV-cache optimization work on branch
`optimize-tcq3-set-rows`. The main benchmark target was:

```bash
./build-cuda-tcq-v10/bin/llama-bench \
    -hf Ununnilium/Qwen3.6-27B-IQ4_XS-pure-GGUF \
    -hff qwen3.6-27b-IQ4_XS-pure.gguf \
    -ngl 999 -fa 1 \
    -ctk turbo3_tcq -ctv turbo3_tcq \
    -p 15000 -n 128
```

Local GPU:

- NVIDIA GeForce RTX 4070 Ti SUPER, compute capability 8.9, 16 GB VRAM

Remote cross-check GPU:

- NVIDIA A100-SXM4-80GB, compute capability 8.0

## Current state

The squashed PR state is commit `018092c45`:

```text
cuda: optimize turbo3 tcq set_rows
```

The useful optimized state includes:

- Compact TCQ3 backtrace from `128 * 512` bytes per group to `128 * 64` bytes per group.
- Compute predecessor minima once for the 64 low-state groups.
- Use an 8 KiB shared-memory backtrace by default when available.
- Keep `TURBO_TCQ_SHARED_BT=0` as a global-memory fallback.
- Use warp shuffles for the first FWHT stages.
- Preserve exact encode output for the committed path.

Measured 4070 Ti SUPER performance after the committed cleanup was approximately:

| State | pp15000 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| Pre-optimization TCQ3/TCQ3 | ~1491 | ~36.5 |
| Optimized committed state | ~1580 to ~1595 | ~37.2 |
| q8_0/q8_0 comparison | ~1698 | ~38.0 |

The remaining 4070 Ti SUPER prefill gap to `q8_0/q8_0` is roughly 6 percent.

## A100 validation

IQ4_XS weights on A100:

| KV type | pp15000 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| q8_0/q8_0 | 1292.62 | 54.84 |
| turbo3_tcq/turbo3_tcq, shared BT | 1220.18 | 52.57 |
| turbo3_tcq/turbo3_tcq, `TURBO_TCQ_SHARED_BT=0` | 1216.17 | 52.49 |

Q8_0 weights on A100:

| KV type | pp15000 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| q8_0/q8_0 | 1260.68 | 38.69 |
| q8_0/turbo3_tcq | 1226.40 | 38.20 |
| turbo3_tcq/q8_0 | 1227.66 | 38.19 |
| turbo3_tcq/turbo3_tcq, shared BT | 1193.08 | 37.56 |
| turbo3_tcq/turbo3_tcq, `TURBO_TCQ_SHARED_BT=0` | 1190.18 | 37.50 |

The shared backtrace path is approximately neutral to mildly positive on both
tested GPUs. The fallback should remain available because a single-side Q8_0
A100 run was noisy and did not clearly favor shared memory.

## Negative experiments

Several bigger-looking optimizations were tried and removed because they
regressed or failed to beat the current path.

| Experiment | Result | Interpretation |
| --- | ---: | --- |
| Graph-level current K/V attention bypass | ~1579 pp tok/s | Graph-side padding/copies cost more than TCQ attention-side savings. |
| Dedicated current-only KQ mask | crashed during setup | A `15000 x 15000` mask is too large for this route. |
| Fused TCQ3 MMA prefill, serial row loader | ~1591 pp tok/s | Avoids fp16 staging but under-parallelizes TCQ decode. |
| Fused TCQ3 MMA prefill, parallel half2 loader | ~1594 pp tok/s | Still does not beat the existing staging path. |
| Move reconstruction norm work into serial backtrack | ~1589 pp tok/s | Parallel reconstruction pass is cheaper than serializing the work. |
| 8-thread subgroup Viterbi ownership reshape | ~1503 pp tok/s | Removing syncs damaged memory/layout behavior enough to dominate. |
| `__launch_bounds__(512, 2)` | ~1593 pp tok/s | Higher occupancy target is worse than the compiler's current register/block tradeoff. |
| Per-state predecessor recompute | 1528.54 pp tok/s, 36.86 tg tok/s | Removing the exchange by repeating the 8-way min in every output state multiplies shared-memory reads too much. |
| 64-thread owner-computes-eight layout | 1249.85 pp tok/s, 35.05 tg tok/s | One thread per low-state group removes the exchange, but underutilizes the block badly. |
| Low-state-major cost layout with 8-lane shuffle reduction | 1466.32 pp tok/s, 36.64 tg tok/s | Contiguous writes and one sync per step were not enough; reindexing plus subgroup shuffles were slower than the shared predecessor exchange. |
| Hoist TCQ codebook value out of Viterbi loop | 1573.96 pp tok/s, 37.19 tg tok/s | Likely increased register pressure or defeated a better compiler choice. |
| Drop common `xt*xt` term from Viterbi cost | 1569.00 to 1574.88 pp tok/s, 37.20 to 37.21 tg tok/s | Exact output symbols matched the baseline trace, but performance stayed within noise and was not a real improvement. |

The graph bypass also showed an important constraint: the cache writes are still
needed for subsequent decode, so simply consuming current K/V for attention does
not remove the TCQ encode work from prompt processing.

## Profiling constraints

`ncu` is installed, but hardware performance counters were not available in the
local environment:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

Without counters, the most reliable signal came from paired `llama-bench` runs
and Nsight Systems kernel timing.

Previous useful timing snapshots:

| State | `k_set_rows_turbo3_tcq` total | Average launch time |
| --- | ---: | ---: |
| Original | 7.262 s | 1.254 ms |
| v4 compact shared BT | 4.489 s | 0.775 ms |
| Cleanup/v8 state | 4.215 s | 0.728 ms |

## Validation note

The legacy `TURBO_TCQ_DUMP_ERRORS` path is useful for autocorrelation analysis,
but it is not sufficient for exact equivalence checks. The dump is indexed only
by `group`, and group indices are reused by many `set_rows` calls across layers
and K/V tensors. Timing changes can alter the last writer for a dump slot,
producing false mismatches.

The validation harness now has a non-racy call-indexed trace:

```bash
TURBO_TCQ_DUMP_TRACE=32:4 \
TURBO_TCQ_DUMP_TRACE_PATH=/tmp/tcq_base.bin \
./build-cuda-tcq-v10/bin/llama-bench \
    -hf Ununnilium/Qwen3.6-27B-IQ4_XS-pure-GGUF \
    -hff qwen3.6-27b-IQ4_XS-pure.gguf \
    -ngl 999 -fa 1 \
    -ctk turbo3_tcq -ctv turbo3_tcq \
    -p 128 -n 1
```

`TURBO_TCQ_DUMP_TRACE=CALLS:GROUPS` captures the first `GROUPS` TCQ groups for
each of the first `CALLS` TCQ `set_rows` launches. The dump includes per-call
metadata and stores post-FWHT normalized values plus output symbols in
`[call][group][128]` layout.

Compare two runs with:

```bash
python3 scripts/compare_tcq_trace.py /tmp/tcq_base.bin /tmp/tcq_candidate.bin --check-x
```

Same-build smoke test result:

```text
PASS: output symbols match exactly for 32 calls / 128 captured groups
x max abs diff: 0
```

The common-term cost experiment also passed exact trace validation:

```text
PASS: output symbols match exactly for 32 calls / 128 captured groups
x max abs diff: 0
```

It replaced `(xt - c)^2` with `c * (c - 2*xt)`, dropping the `xt*xt` term that
is common to all states at a timestep. The symbol path stayed exact, but
paired `pp15000/tg128` runs were noise-level:

| State | pp15000 tok/s | tg128 tok/s |
| --- | ---: | ---: |
| Candidate run 1 | 1574.88 | 37.21 |
| Baseline same-session run | 1571.56 | 37.21 |
| Candidate run 2 | 1569.00 | 37.20 |

The experiment was reverted.

## Likely next real optimization

The obvious local edits are mostly exhausted. The current kernel is tuned around
contiguous state ownership:

- `sid` owns one state.
- The 64 predecessor minima are written contiguously.
- The eight output states read a shared predecessor minimum.
- The extra sync is expensive, but attempts to remove it by changing ownership
  made accesses less favorable and regressed badly.

A meaningful next step likely needs something more structural than the no-sync
layouts tried so far:

- The current shared predecessor exchange is cheaper than recomputing minima,
  using only 64 active low-state threads, or transposing the cost layout for
  8-lane subgroup reductions.
- Future layout work needs to avoid shared-memory bank conflicts, subgroup
  shuffle overhead, and reduced active-lane utilization at the same time.
- Use the call-indexed trace before keeping any deeper rewrite.

In short: the remaining big improvement is probably not another small sync
removal. It needs a cost/codebook/state layout that makes the no-exchange
ownership model cheap.
