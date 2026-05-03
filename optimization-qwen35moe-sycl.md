# Qwen3.6 35B A3B APEX SYCL optimization log

## Goal

Optimize the requested llama.cpp benchmark until there is at least a 5% measured improvement.

Requested benchmark:

```sh
./build-f16/bin/llama-bench -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

## Environment

- Repository: `/home/sroecker/src/llama.cpp`
- Benchmark binary: `./build-f16/bin/llama-bench`
- Build reported by benchmark: `0754b7b6f (9009)`
- Backend: SYCL
- GPU: Intel Arc A770 Graphics through Level Zero
- Device discovery:
  - `llama-ls-sycl-device`: `Intel Arc A770 Graphics`, 512 compute units, 16225M global memory, driver `1.14.37435+12`
  - `sycl-ls`: `level_zero:gpu:0` plus OpenCL CPU/GPU devices
- Existing git state before changes: only untracked `old/`

The first benchmark attempt failed inside the initial sandbox with:

```text
ptrace: Operation not permitted
can not find preferred GPU platform
```

Running with normal device access fixed discovery and allowed the benchmark to run.

## Baseline

Command:

```sh
./build-f16/bin/llama-bench -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Result:

| test | t/s |
| --- | ---: |
| `pp512` | `109.04 +/- 1.90` |
| `tg128` | `9.22 +/- 0.11` |

## Experiments

| Experiment | Command or build change | `pp512` t/s | `tg128` t/s | Finding |
| --- | --- | ---: | ---: | --- |
| Enable SYCL graph runtime | `GGML_SYCL_DISABLE_GRAPH=0` | `106.16 +/- 1.23` | `9.03 +/- 0.08` | Worse than baseline. Keep graphs disabled. |
| Prioritize DMMV | `GGML_SYCL_PRIORITIZE_DMMV=1` | `107.95 +/- 2.06` | `9.09 +/- 0.05` | Worse than baseline. |
| Disable oneDNN path | `GGML_SYCL_DISABLE_DNN=1` | `104.75 +/- 1.73` | `9.11 +/- 0.09` | Worse than baseline. Keep oneDNN enabled. |
| Disable SYCL optimize/reorder | `GGML_SYCL_DISABLE_OPT=1` | `108.52 +/- 1.90` | `9.11 +/- 0.01` | Flat to worse. Keep default optimization enabled. |
| Disable SYCL flash-attention runtime gate | `GGML_SYCL_ENABLE_FLASH_ATTN=0` | `107.68 +/- 1.88` | `8.74 +/- 0.07` | Clearly worse. Keep flash attention enabled. |
| Force Level Zero selector and Sysman | `ZES_ENABLE_SYSMAN=1 ONEAPI_DEVICE_SELECTOR=level_zero:0` | `108.93 +/- 1.97` | `9.11 +/- 0.02` | No improvement. |
| Disable pinned host buffers | `GGML_SYCL_NO_PINNED=1` | `108.88 +/- 1.74` | `9.15 +/- 0.02` | No meaningful improvement. |
| Force MMQ compile-time path | Separate build with `-DCMAKE_CXX_FLAGS=-DGGML_SYCL_FORCE_MMQ` | `109.35 +/- 2.12` | `9.18 +/- 0.02` | Essentially flat; does not meet 5%. |
| Increase MMV rows per workgroup | Separate build with `-DCMAKE_CXX_FLAGS=-DGGML_SYCL_MMV_Y=2` | `109.46 +/- 1.41` | `9.14 +/- 0.04` | Flat to worse; reject. |
| Disable fused MoE MMVQ path | Guarded build with `-DGGML_SYCL_DISABLE_MOE_MMVQ_FUSED` | `111.11 +/- 1.37` | `7.54 +/- 0.07` | Prompt improves, generation regresses badly; reject. |
| Level Zero immediate command lists | `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | `108.03 +/- 1.53` | `9.15 +/- 0.05` | No improvement. |
| Level Zero batch size | `UR_L0_BATCH_SIZE=256` | `108.18 +/- 1.62` | `9.15 +/- 0.05` | No improvement. |
| UR immediate command-list batching | `UR_L0_IMMEDIATE_COMMANDLISTS_BATCH_MAX=256 UR_L0_IMMEDIATE_COMMANDLISTS_EVENTS_PER_BATCH=256 UR_L0_USE_IMMEDIATE_COMMANDLISTS=1` | `108.52 +/- 2.13` | `9.12 +/- 0.03` | No improvement. |
| Allow `MUL_MAT_ID` graph capture | Removed graph-compatibility rejection and ran with `GGML_SYCL_DISABLE_GRAPH=0` | n/a | n/a | Crashed during `pp512`; fallback `MUL_MAT_ID` still calls `stream->wait()` while recording a graph. Rejected and reverted. |
| Disable host buffers | `--no-host 1` | `109.21 +/- 1.17` | `9.19 +/- 0.03` | Flat; does not meet target. |
| Disable non-op offload | `-nopo 1` | `109.31 +/- 1.94` | `9.17 +/- 0.03` | Flat; does not meet target. |

## Code-path notes

- The model is reported as `qwen35moe 35B.A3B Q3_K - Medium`.
- Relevant hot paths are in `ggml/src/ggml-sycl`.
- Regular Q3_K matvec can use DMMV through `ggml_sycl_op_dequantize_mul_mat_vec`.
- MoE single-token work uses `GGML_OP_MUL_MAT_ID` and has a fused MMVQ path:
  - `ggml_sycl_mul_mat_id_mmvq_fused`
  - `ggml_sycl_mul_mat_vec_q_id`
  - `launch_mul_mat_vec_q_moe`
- `ggml_sycl_supports_mmq()` currently returns `false`, so the generic MMQ path is not normally used despite MMQ kernels existing.
- Runtime switches did not produce the needed 5% gain. The winning area was reducing `MUL_MAT_ID` prompt-path launch overhead by extending the fused MoE MMVQ path to batched tokens.

## Current progress

- Baseline established and final target exceeded.
- Clean baseline rerun after stopping the stale `build-tune-acm-g10` compiler jobs confirmed the original target:
  - `pp512`: `109.40 +/- 1.18` t/s
  - `tg128`: `9.20 +/- 0.04` t/s
- Runtime switches tested and rejected.
- A force-MMQ build was tested and rejected.
- A separate `build-tune-y2` build was created with `-DCMAKE_CXX_FLAGS=-DGGML_SYCL_MMV_Y=2`, benchmarked, and rejected.
- A separate `build-tune-no-fused-moe` build disabled the fused MoE MMVQ fast path. It improved `pp512` but badly hurt `tg128`, so it was rejected.
- VTune 2026.0.0 is available at `/opt/intel/oneapi/vtune/2026.0/bin64/vtune`; API/offload traces were used because GPU Hotspots hardware metrics require a missing Metrics Discovery library.

## System observations

- Stopped stale `build-tune-acm-g10` `gmake`/`icpx` jobs before continuing measurements.
- Arc A770 sysfs frequency limits:
  - `gt_RP0_freq_mhz`: `2400`
  - `gt_max_freq_mhz`: `2400`
  - `gt_boost_freq_mhz`: `2400`
  - `gt_min_freq_mhz`: `300`
- Power limit is already `210 W` (`power1_max=210000000`) and idle temperature was about `45 C`.
- No immediate sysfs power/frequency tuning opportunity was visible without privileged writes, and the reported max/boost clocks are already at the card maximum.
- `--prio 2` was rejected by the OS with `Permission denied`, so high process priority cannot be used from this user session.
- `--poll 100` screened at `-r 3`:
  - `pp512`: `110.08 +/- 1.48` t/s
  - `tg128`: `9.17 +/- 0.03` t/s
  - Result: no meaningful improvement; reject.
- `build-tune-k1` with `-DCMAKE_CXX_FLAGS=-DK_QUANTS_PER_ITERATION=1` failed to compile:
  - `ggml/src/ggml-sycl/dmmv.cpp` references `float4` in this macro path.
  - Intel SYCL compilation reports `use of undeclared identifier 'float4'`.
  - Result: not a usable tuning path without fixing that experimental code path first.
- `build-tune-y4` with `-DCMAKE_CXX_FLAGS=-DGGML_SYCL_MMV_Y=4`:
  - `pp512`: `108.82 +/- 1.79` t/s
  - `tg128`: `9.14 +/- 0.02` t/s
  - Result: worse than baseline; reject.
- Batched fused MoE MMVQ source patch in `build-f16`:
  - Extended `ggml_sycl_mul_mat_vec_q_id` / `mul_mat_vec_q_moe` from a single token to explicit token and selected-expert dimensions.
  - `ggml_sycl_mul_mat_id_mmvq_fused` now handles prompt batches where `ids->ne[1] == src1->ne[2]`, quantizing all routed rows and launching one fused matvec kernel per `MUL_MAT_ID` op instead of falling back to the host-synchronized per-expert loop.
  - Exact benchmark result:
    - `pp512`: `164.66 +/- 3.70` t/s
    - `tg128`: `9.24 +/- 0.05` t/s
  - Versus clean baseline `pp512 109.40`, this is about `+50.5%`; `tg128` is flat-to-slightly-up versus `9.20`.
  - Result: target achieved; now validating correctness/regression risk.
- After adding tighter layout guards and rebuilding `build-f16`, final exact benchmark result:
  - `pp512`: `164.13 +/- 3.48` t/s
  - `tg128`: `9.14 +/- 0.03` t/s
  - Versus clean baseline `pp512 109.40`, final prompt throughput improvement is about `+50.0%`.
- Correctness validation:
  - `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0`
  - Result: `690/690 tests passed`.

## Post-patch VTune check

Collected another GPU offload/API trace:

```sh
vtune -collect gpu-offload \
  -knob collect-programming-api=true \
  -knob enable-characterization-insights=false \
  -knob enable-stack-collection=false \
  -knob dump-compute-task-binaries=false \
  -result-dir vtune-qwen35moe-gpu-offload-batched-moe -- \
  ./build-f16/bin/llama-bench -r 1 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Profiled benchmark rows with VTune overhead:

| test | t/s |
| --- | ---: |
| `pp512` | `160.08 +/- 0.00` |
| `tg128` | `8.82 +/- 0.00` |

Key API/task-count deltas versus the original VTune trace:

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `zeCommandListAppendLaunchKernel` calls | `653,871` | `543,213` | `-16.9%` |
| `zeCommandListAppendMemoryFill` calls | `37,942` | `18,031` | `-52.5%` |
| `ggml_sycl_mul_mat_id` GPU task instances | `25,292` | `12,015` | `-52.5%` |

The new fused prompt kernel shows up directly in VTune:

| GPU task | Total time | Count |
| --- | ---: | ---: |
| `launch_mul_mat_vec_q_moe<...block_q3_K...>` | `1.301s` | `76` |
| `mul_mat_vec_iq2_s_q8_1_sycl` | `0.808s` | `62,780` |
| `bin_bcast_sycl<op_add>` | `0.459s` | `51,620` |
| `ggml_sycl_mul_mat_id` | `0.387s` | `12,015` |
| `quantize_row_q8_1_sycl` | `0.352s` | `54,608` |

Interpretation: the winning change removed a large fraction of prompt-path `MUL_MAT_ID` fallback work and host launch overhead by using one fused batched MoE matvec per supported `MUL_MAT_ID` op.

## VTune findings

Attempted `gpu-hotspots` first:

```sh
vtune -collect gpu-hotspots ...
```

It failed because GPU hardware metrics are blocked by the current i915 setting:

```text
Cannot collect GPU hardware metrics due to a lack of permissions.
/proc/sys/dev/i915/perf_stream_paranoid value is set to 0.
```

Current user is already in the `render` group. `/proc/sys/dev/i915/perf_stream_paranoid` was initially `1`; it was later set to `0`, enabling GPU hardware metrics collection.

After that setting was fixed, `gpu-hotspots` still could not collect hardware metrics because the Intel Metrics Discovery library is not installed:

```text
Cannot collect GPU hardware metrics because neither libigdmd.so nor libmd.so was found.
```

Collected a non-metric GPU offload/API trace instead:

```sh
vtune -collect gpu-offload \
  -knob collect-programming-api=true \
  -knob enable-characterization-insights=false \
  -knob enable-stack-collection=false \
  -knob dump-compute-task-binaries=false \
  -result-dir vtune-qwen35moe-gpu-offload -- \
  ./build-f16/bin/llama-bench -r 1 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Profiled benchmark rows with VTune overhead:

| test | t/s |
| --- | ---: |
| `pp512` | `106.30 +/- 0.00` |
| `tg128` | `8.58 +/- 0.00` |

Important VTune summary numbers:

| Host task | Task time | Count |
| --- | ---: | ---: |
| `zeCommandListAppendLaunchKernel` | `15.120s` | `653,871` |
| `zeEventHostSynchronize` | `2.993s` | `10,080` |
| `zeCommandListAppendMemoryCopy` | `2.268s` | `13,845` |
| `zeCommandListAppendMemoryFill` | `0.966s` | `37,942` |

Top GPU tasks by total time:

| GPU task | Total time | Count | Notes |
| --- | ---: | ---: | --- |
| `mul_mat_vec_iq2_s_q8_1_sycl` | `0.861s` | `62,880` | Many tiny matvec launches. |
| `ggml_sycl_mul_mat_id` | `0.852s` | `25,292` | MoE ID path/copy task. |
| `mul_mat_vec_q3_K_q8_1_sycl` | `0.648s` | `21,014` | Q3_K MMVQ matvec. |
| `bin_bcast_sycl<op_add>` | `0.492s` | `51,620` | Small elementwise launches. |
| `quantize_row_q8_1_sycl` | `0.378s` | `55,026` | Activation quantization overhead. |

Interpretation: this workload is dominated by a very high number of very small GPU/API tasks. The best remaining optimization direction is reducing launch/API overhead or fusing adjacent tiny operations, especially around MoE matvec, activation quantization, and elementwise add/mul paths.

## Prompt-path outcome before TG follow-up

- The requested `./build-f16/bin/llama-bench ...` benchmark now reaches `164.13 +/- 3.48` t/s on `pp512`, up from the clean baseline `109.40 +/- 1.18` t/s.
- Final measured `pp512` improvement: about `+50.0%`.
- `tg128` remains near baseline: final `9.14 +/- 0.03` t/s versus clean baseline `9.20 +/- 0.04` t/s.
- `MUL_MAT_ID` correctness validation passed on SYCL0: `690/690 tests passed`.

## TG128 follow-up

New target: improve isolated `tg128` by at least 5%.

TG-only baseline command:

```sh
./build-f16/bin/llama-bench -p 0 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

TG-only baseline result: `9.37 +/- 0.06` t/s. The +5% target for isolated TG is about `9.84` t/s.

After installing Fedora's Metrics Discovery package, VTune GPU Hotspots works:

```sh
vtune -collect gpu-hotspots \
  -result-dir vtune-qwen35moe-tg-gpu-hotspots -- \
  ./build-f16/bin/llama-bench -r 1 -p 0 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Profiled TG row with VTune overhead: `9.02` t/s.

Important VTune GPU Hotspots findings:

| Metric | Value |
| --- | ---: |
| GPU time | `15.046s` |
| GPU time as elapsed | `53.7%` |
| XVE Array Stalled/Idle | `95.2%` |
| GPU L3 bandwidth bound | `0.6% of peak` |
| Occupancy | `14.3% of peak` |

Top low-occupancy GPU tasks:

| GPU task | Total time | Peak XVE threads occupancy | Occupancy |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_iq2_s_q8_1_sycl` | `0.525s` | `12.5%` | `36.2%` |
| `bin_bcast_sycl<op_add>` | `0.483s` | `0.8%` | `24.1%` |
| `quantize_row_q8_1_sycl` | `0.364s` | `1.6%` | `29.2%` |

Interpretation: TG is not L3 bandwidth bound. It is dominated by many small low-occupancy kernels and host/offload overhead, so the next useful direction is reducing launch count and tiny elementwise/quantization kernels in generation.

### TG experiment: MoE weighted-sum graph fusion

The decode graph materializes each MoE down projection as:

```text
ffn_moe_down-* = MUL_MAT_ID(... selected experts ...)
ffn_moe_weighted-* = ffn_moe_down-* * ffn_moe_weights_norm-*
ffn_moe_out-* = ADD chain over 8 expert views
```

Initial attempt: add a graph rewrite for the `MUL` plus seven-`ADD` reduction after `ffn_moe_down-*`.

Finding: this was too weak for the target, and the first matcher did not account for no-op `VIEW` nodes in the cgraph. Debugging with `GGML_SYCL_DEBUG=1` showed that the original `ffn_moe_weighted-*` and add chain were still executing.

Accepted implementation:

- Added a graph helper that walks to the next executable node while skipping empty, `RESHAPE`, `TRANSPOSE`, `VIEW`, `PERMUTE`, and `NONE` nodes, matching the existing SYCL executor skip behavior.
- Added a direct MoE down fast path that recognizes `MUL_MAT_ID -> MUL(weights) -> ADD...` and launches one weighted-sum MMVQ kernel.
- The new kernel loops over selected experts inside one row/token workgroup, multiplies each expert dot product by `ffn_moe_weights_norm`, and writes the final `ffn_moe_out-*` tensor directly.
- Added cgraph use-count checks so the rewrite only skips tensors consumed by this exact weighted-sum chain.
- Kept a post-`MUL` weighted-sum fusion fallback for layers where direct down fusion is not selected.

TG-only result after the active direct fusion:

| command | baseline | result | improvement |
| --- | ---: | ---: | ---: |
| `llama-bench -p 0 ...` | `9.37 +/- 0.06` | `10.07 +/- 0.03` | `+7.5%` |

Original combined benchmark after the active direct fusion:

| test | previous optimized result | final result | change |
| --- | ---: | ---: | ---: |
| `pp512` | `164.13 +/- 3.48` | `163.18 +/- 4.41` | about unchanged |
| `tg128` | `9.14 +/- 0.03` | `9.75 +/- 0.03` | `+6.7%` |

Correctness / smoke validation:

| validation | result |
| --- | --- |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0` | `690/690 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID_FUSION -b SYCL0` | `13/13 tests passed` |
| `llama-bench -r 1 -p 0 -n 1 --no-warmup ...` | completed |
| `GGML_SYCL_DEBUG=1 ... \| rg "ffn_moe_down-0\|ffn_moe_weighted-0\|ffn_moe_out-0"` | no compute trace for the skipped chain |

Final VTune GPU Hotspots pass with the active fusion:

| Metric | Value |
| --- | ---: |
| Profiled TG row | `9.75` t/s |
| GPU time | `13.530s` |
| GPU time as elapsed | `50.4%` |
| XVE Array Stalled/Idle | `94.9%` |
| GPU L3 bandwidth bound | `0.6% of peak` |
| Occupancy | `14.6% of peak` |

Final top low-occupancy GPU tasks:

| GPU task | Total time | Peak XVE threads occupancy | Occupancy |
| --- | ---: | ---: | ---: |
| `mul_mat_vec_iq2_s_q8_1_sycl` | `0.510s` | `12.5%` | `3.1%` |
| `quantize_row_q8_1_sycl` | `0.349s` | `1.6%` | `7.7%` |
| `dequantize_mul_mat_vec_q3_K_sycl` | `0.321s` | `50.0%` | `27.8%` |

The prior `bin_bcast_sycl<op_add>` hotspot (`0.483s`) is no longer in the final top low-occupancy list, which matches the intended effect of removing the MoE weighted add-reduction chain from decode.

### TG follow-up: remaining VTune candidates

After the weighted-sum fusion, the remaining VTune candidates were:

- `mul_mat_vec_iq2_s_q8_1_sycl`
- `quantize_row_q8_1_sycl`
- `dequantize_mul_mat_vec_q3_K_sycl`

Clean starting point for this pass:

```sh
./build-f16/bin/llama-bench -p 0 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Result: `10.01 +/- 0.03` t/s.

Debug trace finding:

- The `iq2_s` MoE layers were still falling back from `MUL_MAT_ID` into per-expert `ggml_sycl_mul_mat` calls.
- One-token debug trace count for normal `ggml_sycl_mul_mat` calls with `type=iq2_s`: `480` before, `0` after adding IQ2_S to fused MoE ID dispatch.
- Q8_1 quantization trace entries in the same one-token run dropped from `1282` to `322` after the IQ2_S fused path was enabled.

Accepted changes:

- Added `GGML_TYPE_IQ2_S` to `ggml_sycl_mul_mat_vec_q_id`.
- Added `GGML_TYPE_IQ2_S` to `ggml_sycl_mul_mat_vec_q_id_weighted_sum`.
- Prefer MMVQ over DMMV for Q3_K single-vector decode when `MUL_MAT_VEC_Q` is available and `GGML_SYCL_PRIORITIZE_DMMV` is not set.

Screened experiments:

| experiment | result | decision |
| --- | ---: | --- |
| Add IQ2_S to fused MoE ID and weighted-sum dispatch | `14.27 +/- 0.06` t/s | Keep. Removes the top IQ2_S per-expert fallback. |
| Prefer Q3_K MMVQ over DMMV for decode | `14.43 +/- 0.01` t/s | Keep. Small but repeatable positive sample and removes `dequantize_mul_mat_vec_q3_K_sycl` from the top VTune list. |
| Use two rows per fused MoE workgroup | `14.34 +/- 0.07` t/s | Do not keep. It did not beat the best kept Q3_K sample. |

Final clean isolated TG sample after keeping IQ2_S fusion and Q3_K MMVQ dispatch:

| command | previous committed result | final result | improvement |
| --- | ---: | ---: | ---: |
| `llama-bench -p 0 ...` | `10.01 +/- 0.03` | `14.47 +/- 0.02` | `+44.6%` |

Original combined benchmark after the remaining-candidates pass:

| test | previous weighted-sum result | final result | change |
| --- | ---: | ---: | ---: |
| `pp512` | `163.18 +/- 4.41` | `340.12 +/- 4.47` | `+108.4%` |
| `tg128` | `9.75 +/- 0.03` | `14.20 +/- 0.04` | `+45.6%` |

VTune GPU Hotspots after IQ2_S fusion:

| Metric | Value |
| --- | ---: |
| Profiled TG row | `14.22` t/s |
| GPU time | `9.814s` |
| XVE Array Stalled/Idle | `92.6%` |
| GPU L3 bandwidth bound | `0.9% of peak` |
| Occupancy | `21.1% of peak` |

Top low-occupancy GPU tasks after IQ2_S fusion:

| GPU task | Total time | Peak XVE threads occupancy | Occupancy |
| --- | ---: | ---: | ---: |
| `dequantize_mul_mat_vec_q3_K_sycl` | `0.327s` | `50.0%` | `22.8%` |
| `launch_mul_mat_vec_q_moe<..., block_q3_K, ...>` | `0.317s` | `100.0%` | `19.4%` |
| `dequantize_mul_mat_vec_q3_K_sycl` | `0.305s` | `100.0%` | `17.6%` |

VTune GPU Hotspots after Q3_K MMVQ dispatch:

| Metric | Value |
| --- | ---: |
| Profiled TG row | `14.21` t/s |
| GPU time | `9.742s` |
| XVE Array Stalled/Idle | `92.1%` |
| GPU L3 bandwidth bound | `0.6% of peak` |
| Occupancy | `18.9% of peak` |

Top low-occupancy GPU tasks after Q3_K MMVQ dispatch:

| GPU task | Total time | Peak XVE threads occupancy | Occupancy |
| --- | ---: | ---: | ---: |
| `launch_mul_mat_vec_q_moe<..., block_q3_K, ...>` | `0.316s` | `100.0%` | `17.5%` |
| `mul_mat_vec_q3_K_q8_1_sycl` | `0.219s` | `50.0%` | `18.9%` |
| `reorder_mul_mat_vec_q6_k_q8_1_sycl` | `0.219s` | `100.0%` | `73.8%` |

Correctness / validation:

| validation | result |
| --- | --- |
| `cmake --build build-f16 --target llama-bench test-backend-ops -j6` | passed |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0` | `690/690 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID_FUSION -b SYCL0` | `13/13 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT -b SYCL0` | `911/911 tests passed` |

The original `mul_mat_vec_iq2_s_q8_1_sycl` and `quantize_row_q8_1_sycl` candidates are no longer top named VTune tasks after adding IQ2_S to the fused MoE path. The Q3_K DMMV candidate was addressed by dispatching Q3_K decode through MMVQ; the remaining visible hot paths are Q3_K MMVQ/MoE and Q6 output projection work.

### TG follow-up: Q3_K and Q6_K VDR tuning

Fresh VTune/API check before this pass:

```sh
vtune -collect gpu-offload \
  -knob collect-programming-api=true \
  -knob enable-characterization-insights=false \
  -knob enable-stack-collection=false \
  -knob dump-compute-task-binaries=false \
  -result-dir vtune-qwen35moe-gpu-offload-current -- \
  ./build-f16/bin/llama-bench -r 1 -p 0 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Profiled TG row with VTune overhead: `14.19` t/s.

Top findings from that trace:

| area | finding |
| --- | --- |
| Host/API overhead | `zeCommandListAppendLaunchKernel`: `6.176s`, `293,756` calls |
| Host sync | `zeEventHostSynchronize`: `1.980s`, `2,088` calls |
| Q3_K MoE | `launch_mul_mat_vec_q_moe<..., block_q3_K, ...>`: `0.318s`, `5,160` instances |
| Q3_K MMVQ | `mul_mat_vec_q3_K_q8_1_sycl`: `0.220s` and `0.198s` groups |
| Q6_K output | `reorder_mul_mat_vec_q6_k_q8_1_sycl`: `0.219s`, `129` instances |

Screened experiments:

| experiment | result | decision |
| --- | ---: | --- |
| Allow compatible `MUL_MAT_ID` graph capture and run with `GGML_SYCL_DISABLE_GRAPH=0` | `14.47` t/s standalone, `14.30` t/s under VTune; launch count still `293,756` | Reject. No host launch-count reduction for this graph. |
| Q3_K MMVQ/MoE VDR=2 | `15.12 +/- 0.06` first sample; `14.97 +/- 0.01` final Q3-only sample | Keep. VTune Q3 MoE time dropped from `0.318s` to `0.209s`. |
| Q3_K MMVQ/MoE VDR=4 | `14.99 +/- 0.02` | Reject. Did not beat Q3_K VDR=2. |
| Disable Q6_K reordered MMVQ | `14.86 +/- 0.10` | Reject. Slower than kept path. |
| Force Q6_K vector matmul to DMMV | `14.78 +/- 0.02` | Reject. Slower than kept path. |
| Q6_K MMVQ VDR=2 | `15.08 +/- 0.01` | Keep until VDR=4 screening. |
| Q6_K MMVQ VDR=4, non-reordered macro only | `15.19 +/- 0.01` | Superseded by full Q6_K VDR=4. |
| Q4_K reordered VDR=4 accidental screen | `909/911` `MUL_MAT` tests passed | Reject. It fails Q4_K correctness and is not part of the final patch. |
| Q6_K MMVQ VDR=4, reordered trait plus non-reordered macro | `15.22 +/- 0.05` | Keep. Best screened TG result and correctness passes. |

Final accepted changes in this pass:

- Q3_K MMVQ/MoE now evaluates two dot-product rows per lane through `VDR_Q3_K_Q8_1_MMVQ = 2`.
- Q6_K MMVQ/MoE now evaluates four rows per lane through `VDR_Q6_K_Q8_1_MMVQ = 4`.
- Q6_K reordered MMVQ now uses `block_q_t<GGML_TYPE_Q6_K>::traits::vdr_mmvq = 4`.
- The attempted graph-capture and Q4_K/Q6 fallback dispatch changes were reverted.

Final clean isolated TG sample:

| command | fresh profiled baseline | final result | improvement |
| --- | ---: | ---: | ---: |
| `llama-bench -p 0 ...` | `14.19` t/s under VTune | `15.22 +/- 0.05` | `+7.3%` versus profiled baseline |

Original combined benchmark after this pass:

| test | previous remaining-candidates result | final result | change |
| --- | ---: | ---: | ---: |
| `pp512` | `340.12 +/- 4.47` | `398.54 +/- 6.44` | `+17.2%` |
| `tg128` | `14.20 +/- 0.04` | `15.17 +/- 0.04` | `+6.8%` |

Final VTune GPU Hotspots pass:

```sh
vtune -collect gpu-hotspots \
  -result-dir vtune-qwen35moe-tg-gpu-hotspots-vdr-final -- \
  ./build-f16/bin/llama-bench -r 1 -p 0 -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

| Metric | Value |
| --- | ---: |
| Profiled TG row | `15.01` t/s |
| GPU time | `8.877s` |
| GPU time as elapsed | `41.5%` |
| XVE Array Stalled/Idle | `92.7%` |
| GPU L3 bandwidth bound | `0.6% of peak` |
| Occupancy | `15.7% of peak` |

Top low-occupancy GPU tasks after VDR tuning:

| GPU task | Total time | Peak XVE threads occupancy | Occupancy |
| --- | ---: | ---: | ---: |
| `launch_mul_mat_vec_q_moe<..., block_q3_K, VDR=2, ...>` | `0.210s` | `100.0%` | `12.9%` |
| `flash_attn_tile<...>` | `0.191s` | `1.6%` | `8.5%` |
| `bin_bcast_sycl<op_mul>` | `0.169s` | `0.8%` | `28.0%` |

Correctness / validation after final VDR settings:

| validation | result |
| --- | --- |
| `cmake --build build-f16 --target llama-bench test-backend-ops -j6` | passed |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT -b SYCL0` | `911/911 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0` | `690/690 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID_FUSION -b SYCL0` | `13/13 tests passed` |

Interpretation: Q3_K and Q6_K VDR tuning clears the remaining Q3/MMVQ and Q6 output-projection candidates from the previous VTune list and gives another measured `tg128` gain over the already optimized branch. The main visible next candidates are now the residual Q3_K fused MoE kernel, flash-attention tile occupancy, and small `op_mul` broadcasts.

## Two-GPU long-context KV-cache benchmark

Goal: benchmark the same model on both Arc A770 devices with a more realistic longer prompt/generation shape and quantized KV cache.

Device discovery:

| device | name | free memory | reorder |
| --- | --- | ---: | --- |
| `SYCL0` | Intel Arc A770 Graphics | `15473 MiB` | yes |
| `SYCL1` | Intel Arc A770 Graphics | `15473 MiB` | yes |

Benchmark shape:

- Prompt tokens: `4096`
- Generated tokens: `512`
- Repetitions: `3`
- Flash attention: enabled
- GPU layers: `99`
- KV cache variants: `q8_0/q8_0` and `q4_0/q4_0`
- Devices tested: `SYCL0`, `SYCL1`, and `SYCL0/SYCL1` with default layer split

Representative command:

```sh
./build-f16/bin/llama-bench -r 3 -p 4096 -n 512 \
  -ctk q8_0 -ctv q8_0 -dev SYCL0 \
  -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

Results:

| devices | KV cache | `pp4096` t/s | `tg512` t/s |
| --- | --- | ---: | ---: |
| `SYCL0` | `q8_0/q8_0` | `325.19 +/- 0.52` | `14.27 +/- 0.12` |
| `SYCL1` | `q8_0/q8_0` | `319.66 +/- 0.65` | `14.72 +/- 0.05` |
| `SYCL0` | `q4_0/q4_0` | `324.43 +/- 0.50` | `14.27 +/- 0.12` |
| `SYCL1` | `q4_0/q4_0` | `319.26 +/- 0.89` | `14.61 +/- 0.05` |
| `SYCL0/SYCL1` | `q8_0/q8_0` | `514.94 +/- 0.59` | `14.51 +/- 0.01` |
| `SYCL0/SYCL1` | `q4_0/q4_0` | `514.43 +/- 0.24` | `14.37 +/- 0.02` |

Interpretation:

- The two A770s are close, but not identical: `SYCL0` is slightly faster on prompt processing, while `SYCL1` is faster on generation in this run.
- Dual-GPU layer split gives a large prompt-processing gain: best dual `pp4096` is `514.94` t/s versus best single-GPU `325.19` t/s, about `+58%`.
- Dual-GPU layer split does not improve generation for this decode-heavy shape: best single-GPU `tg512` is `14.72` t/s on `SYCL1`, while dual q8 is `14.51` t/s.
- `q4_0` KV did not improve throughput at this context length. It was flat to slightly slower than `q8_0`, so `q8_0` is the better performance choice here unless memory pressure requires q4.

Going-forward two-GPU optimization baseline:

| benchmark target | baseline |
| --- | ---: |
| Prompt throughput, `SYCL0/SYCL1`, `q8_0` KV | `pp4096 514.94 +/- 0.59` t/s |
| Decode throughput, `SYCL0/SYCL1`, `q8_0` KV | `tg512 14.51 +/- 0.01` t/s |
| Best single-GPU decode reference, `SYCL1`, `q8_0` KV | `tg512 14.72 +/- 0.05` t/s |

For future two-GPU work, use `SYCL0/SYCL1` with `q8_0` KV as the main benchmark target. A useful decode optimization should beat both the dual-GPU baseline and the single-GPU `SYCL1` reference; otherwise it may only be improving prompt processing or moving overhead between devices.

## Two-GPU decode optimization pass

Goal: improve the two-GPU long-context q8 KV decode baseline by at least 5%.

Baseline:

```sh
./build-f16/bin/llama-bench -r 3 -p 4096 -n 512 \
  -ctk q8_0 -ctv q8_0 -dev SYCL0/SYCL1 \
  -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-I-Mini.gguf -fa 1 -ngl 99
```

| metric | baseline |
| --- | ---: |
| `pp4096` | `514.94 +/- 0.59` t/s |
| `tg512` | `14.51 +/- 0.01` t/s |

Kernel expert guidance received:

- Treat the previous profile as occupancy / tiny-kernel limited, not memory-bandwidth limited.
- Keep Q3_K VDR at 2 for now; the next gain should come from exposing more independent work or removing small kernels.
- Try FA vector for single-token decode.
- Try Q3_K MoE MMVQ row grouping at 4 and 8 rows per work-group.
- Prefer explicit hot-path `op_mul` fusion before SYCL graph replay.

Experiments:

| experiment | result | decision |
| --- | --- | --- |
| Force FA vector for non-quantized `Q->ne[1] == 1` decode | f16 KV `tg512 14.95 +/- 0.02` vs f16 baseline `14.82 +/- 0.03`; q8 path unchanged and measured `14.43 +/- 0.02` | Keep selector simplification. Positive for f16 KV, neutral/noisy for q8 because q8 already used vector for `nq <= 2`. |
| Q3_K MoE MMVQ `ROWS_PER_WG=4` | q8 two-GPU `tg512 14.45 +/- 0.05` | Reject. Correct but slower than baseline. |
| Q3_K MoE MMVQ `ROWS_PER_WG=8` | q8 two-GPU `tg512 14.43 +/- 0.03` | Reject. Correct but slower than baseline. |
| Fuse `RMS_NORM + MUL` for F32 norm-scale patterns | q8 two-GPU `pp4096 520.33 +/- 0.37`, `tg512 15.08 +/- 0.02` | Keep. Removes standalone `attn_norm`, `attn_post_norm`, `Qcur_normed`, `Kcur_normed`, and `result_norm` multiply launches. |
| Fuse F32 vector-by-scalar multiply plus add for shared expert gate | q8 two-GPU `pp4096 516.25 +/- 0.46`, `tg512 15.22 +/- 0.05` | Keep for decode. Removes standalone `ffn_shexp_gated` multiply and folds the following add. |
| Fuse F32 `SIGMOID + MUL` for exact same-shape gated attention | q8 two-GPU `pp4096 519.21 +/- 0.56`, `tg512 15.34 +/- 0.00` | Keep. Removes standalone `attn_gated` multiply and reaches the target. |

Notes:

- Although the build directory is `build-f16`, these fused paths are intentionally F32. The debug graph shows the target activation and norm tensors as `type=f32`; the build merely enables F16-capable SYCL support and does not make all runtime tensors F16.
- The Q3_K row-grouping experiment matched the expert's suggested shape, but it did not improve this two-GPU q8 decode target. It may still be worth revisiting with a single-GPU decode target or a fused expert-dimension launch, but the simple work-group shape change is not enough here.

Final accepted two-GPU result:

| metric | baseline | final | improvement |
| --- | ---: | ---: | ---: |
| `pp4096` | `514.94 +/- 0.59` | `519.21 +/- 0.56` | `+0.8%` |
| `tg512` | `14.51 +/- 0.01` | `15.34 +/- 0.00` | `+5.7%` |

Validation:

| validation | result |
| --- | --- |
| `cmake --build build-f16 --target llama-bench test-backend-ops -j6` | passed |
| `./build-f16/bin/test-backend-ops test -o FLASH_ATTN_EXT -b SYCL0` | `2592/2592 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID_FUSION -b SYCL0` | `13/13 tests passed` |
| `./build-f16/bin/test-backend-ops test -o RMS_NORM -b SYCL0` | `21/21 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL -b SYCL0` | `91/91 tests passed` |
| `./build-f16/bin/test-backend-ops test -o SIGMOID -b SYCL0` | `8/8 tests passed` |

## Balanced GGUF two-GPU baseline

Goal: check whether `Qwen3.6-35B-A3B-APEX-Balanced.gguf` is viable now that two GPUs are available.

Before measurement, the rough estimate based on local GGUF size was pessimistic:

| file | cached size |
| --- | ---: |
| `Qwen3.6-35B-A3B-APEX-I-Mini.gguf` | `14G` |
| `Qwen3.6-35B-A3B-APEX-Balanced.gguf` | `24G` |

Initial estimate: `pp4096 300-360` t/s and `tg512 8.8-10.5` t/s, with a less pessimistic upper case around `pp4096 380` and `tg512 11-12` if the larger file did not scale the active decode path linearly.

Measured command:

```sh
./build-f16/bin/llama-bench -r 3 -p 4096 -n 512 \
  -ctk q8_0 -ctv q8_0 -dev SYCL0/SYCL1 \
  -hf mudler/Qwen3.6-35B-A3B-APEX-GGUF \
  -hff Qwen3.6-35B-A3B-APEX-Balanced.gguf -fa 1 -ngl 99
```

Measured result:

| model | size | weights | `pp4096` | `tg512` |
| --- | ---: | --- | ---: | ---: |
| `Qwen3.6-35B-A3B-APEX-Balanced.gguf` | `23.85 GiB` | `Q5_K - Medium` | `474.84 +/- 0.65` | `15.01 +/- 0.03` |
| `Qwen3.6-35B-A3B-APEX-I-Mini.gguf` final local build | `14G cached` | mixed lower quant | `519.21 +/- 0.56` | `15.34 +/- 0.00` |

Interpretation:

- Balanced prompt processing is slower than Mini by about `8.5%`, but decode is only about `2.2%` slower than the current optimized Mini result.
- The file-size-based estimate was too pessimistic for decode. Treat Balanced as a viable two-GPU candidate.
- The next Balanced tests should focus on runtime flags first, because the existing decode fusions already carry over and the remaining gap may be split / cache / batching related.

## Balanced Q5_K MMVQ VDR tuning

Goal: apply the same MMVQ-style VDR tuning used for the lower-quant Mini path to Balanced's `Q5_K - Medium` weights.

Implementation note:

- The existing Q5_K MMVQ dot body already represented a 2-wide chunk.
- The accepted experiment keeps that body as `vec_dot_q5_K_q8_1_vdr2` and makes `VDR_Q5_K_Q8_1_MMVQ = 4` sum two adjacent 2-wide chunks.
- This is narrower than changing scheduling or adding a new op; only the Q5_K MMVQ dot granularity changes.

Screened results on the Balanced two-GPU q8 KV target:

| Q5_K MMVQ setting | `pp4096` | `tg512` | decision |
| --- | ---: | ---: | --- |
| VDR=2 baseline | `474.84 +/- 0.65` | `15.01 +/- 0.03` | Baseline |
| VDR=4 | `497.90 +/- 0.40` | `15.11 +/- 0.03` | Keep |
| VDR=8 | `459.45 +/- 0.10` | `15.08 +/- 0.01` | Reject; prompt regressed and decode did not beat VDR=4 |

Accepted improvement:

| metric | baseline | VDR=4 | improvement |
| --- | ---: | ---: | ---: |
| `pp4096` | `474.84 +/- 0.65` | `497.90 +/- 0.40` | `+4.9%` |
| `tg512` | `15.01 +/- 0.03` | `15.11 +/- 0.03` | `+0.7%` |

Validation for the VDR=4 candidate:

| validation | result |
| --- | --- |
| `cmake --build build-f16 --target llama-bench test-backend-ops -j6` | passed |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT -b SYCL0` | `911/911 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0` | `690/690 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID_FUSION -b SYCL0` | `13/13 tests passed` |

Interpretation: Q5_K behaves differently from the lower-quant Mini path. VDR=4 is useful, but mostly for prompt throughput; decode is only slightly better. VDR=8 appears to create too much per-lane work or register pressure for this shape.

## Active-expert MoE MMVQ scheduling experiments

Goal: apply the GPU-kernel guidance to expose more independent active-expert decode work per launch without changing the existing graph ops.

Baseline comparison points:

| target | baseline state | `pp4096` | `tg512` |
| --- | --- | ---: | ---: |
| Mini two-GPU q8 KV | after decode fusions | `519.21 +/- 0.56` | `15.34 +/- 0.00` |
| Balanced two-GPU q8 KV | after Q5_K VDR=4 | `497.90 +/- 0.40` | `15.11 +/- 0.03` |

Screened candidates:

| candidate | Mini `pp4096` | Mini `tg512` | Balanced `pp4096` | Balanced `tg512` | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Weighted-sum Q3_K/Q5_K split by active expert with destination zero + global atomic add | `523.93 +/- 1.07` | `15.26 +/- 0.02` | `497.37 +/- 0.53` | `15.00 +/- 0.02` | Reject. Correct, but decode regressed; atomics and the extra zero kernel outweighed added parallelism. |
| Weighted-sum Q3_K/Q5_K local reduction, one workgroup containing all active-expert subgroups for one row | `522.07 +/- 0.95` | `15.25 +/- 0.01` | `499.13 +/- 0.20` | `15.10 +/- 0.03` | Reject. Removed global atomics but still did not improve decode. |
| Non-weighted Q5_K MoE MMVQ with `ROWS_PER_WG=4` | not run | not run | `495.84 +/- 1.88` | `15.14 +/- 0.03` | Reject. Tiny decode movement with prompt regression. |
| Non-weighted Q5_K MoE MMVQ with `ROWS_PER_WG=2` | `510.57 +/- 0.93` | `15.01 +/- 0.06` | `498.14 +/- 0.69` | `15.15 +/- 0.01` | Reject. Balanced decode was slightly higher, but Mini regressed in the final cleaned-state check. |
| Weighted-sum register-private `EXPERT_TILE=2` with one final subgroup reduction | `521.68 +/- 0.09` | `15.22 +/- 0.02` | not run | not run | Reject. Keeps one subgroup per row and avoids SLM/atomics, but two-expert interleaving still regressed Mini decode. |
| Weighted-sum register-private `EXPERT_TILE=1` with one final subgroup reduction | `521.15 +/- 0.74` | `15.34 +/- 0.02` | `478.19 +/- 1.18` | `14.88 +/- 0.03` | Reject. Mini was neutral, but Balanced regressed badly; per-expert subgroup reductions are better for Q5_K Balanced. |

Validation while screening:

| validation | result |
| --- | --- |
| `cmake --build build-f16 --target llama-bench test-backend-ops -j6` | passed for each built candidate |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID_FUSION -b SYCL0` | `13/13 tests passed` |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0 -p 'type_a=q5_K'` | `2/2 tests passed` for Q5_K row grouping |
| `./build-f16/bin/test-backend-ops test -o MUL_MAT_ID -b SYCL0 -p 'type_a=q[35]_K'` | `4/4 tests passed` for weighted-sum screens' related non-weighted coverage |

Finding: simply increasing scheduled expert/row granularity is not enough for this workload. The weighted-sum variants create extra synchronization, local-memory, atomic, or zeroing cost, and the Q5_K row-grouping variants are too small to justify keeping when Mini safety is considered. The later register-private tests show that even removing repeated subgroup reductions is not universally better: Mini tolerates a single final reduction, but Balanced/Q5_K strongly prefers the original per-expert reduction order. All active-expert scheduling and expert-tiling source changes from this section were backed out; keep the previous VDR/fusion commits as the current best local state.

## Local-only flag candidates

These flags are useful for local performance testing and do not imply upstreamable code changes.

Recommended next test order:

| area | flags | purpose |
| --- | --- | --- |
| KV cache type | `-ctk q8_0 -ctv q8_0`, `-ctk q4_0 -ctv q4_0`, `-ctk f16 -ctv f16` | Check whether Balanced decode is cache bandwidth, precision, or occupancy limited at longer contexts. |
| Split mode | `-sm layer`, `-sm row`, `-sm tensor` | Test whether decode benefits from row/tensor splitting instead of default layer splitting on two A770s. |
| Tensor split | `-ts 1/1`, then skewed splits such as `-ts 0.45/0.55` and `-ts 0.4/0.6` | Check whether giving slightly more work to the faster decode GPU improves dual-GPU generation. |
| Main GPU | `-mg 0`, `-mg 1` | Relevant when row split or tensor split changes where intermediate work lands. |
| Batch sizing | `-b 1024`, `-b 2048`, `-b 4096`; `-ub 256`, `-ub 512`, `-ub 1024` | Tune prompt throughput and see whether decode-side launch packing changes. |
| Flash attention | `-fa 1`, `-fa 0` | Keep `-fa 1` as default, but use `-fa 0` as a diagnostic for FA selector and KV-cache interactions. |
| Memory fit | `-fitt 512`, `-fitt 1024`, `-fitc 4096`, `-fitc 8192` | Find practical local memory ceilings and prevent accidental host spillover. |
| Host buffer avoidance | `--no-host` | Force failure instead of hidden host-buffer use; useful for reproducible GPU-only timing. |
| Op offload diagnostic | `-nopo 1` | Check whether tiny offloaded ops are hurting elapsed time. This is diagnostic, not a likely final setting. |
| KV offload diagnostic | `-nkvo 1` | Quantifies the cost of moving KV cache off GPU. Expected to be slower, but useful as a bound. |
| MoE CPU fallback | `-ncmoe` | Memory fallback only. Expected to hurt throughput for these GPU-focused tests. |

For realistic interactive testing with `llama-cli`, also use `--offline` to avoid Hub checks and `--perf` to collect timing output. `llama-cli` exposes runtime flags such as `--no-repack`, `--op-offload`, `--no-op-offload`, and speculative decoding options that are not part of the current `llama-bench` loop.
