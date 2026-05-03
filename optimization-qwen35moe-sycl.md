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
