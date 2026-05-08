# Native NVFP4 CUDA Progress

Date: 2026-05-08

Branch: `feature/native-nvfp4-scale-semantics-local`

Target GPU: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16 GB class VRAM.

This is local progress for native NVFP4 CUDA kernels without dequantizing weights to a wider format before MMQ. The target is Blackwell consumer SM120 behavior, not DGX Spark or GB200 server behavior.

## Current Scope

- GGUF conversion support for compressed-tensors NVFP4 packing.
- CUDA MMQ native FP4 path for MXFP4 and NVFP4 on Blackwell.
- NVFP4 block-scaled MMA plumbing using SM120-compatible warp-level MMA.
- Experimental NVFP4 MMQ+GLU fusion guarded by `GGML_CUDA_NVFP4_MMQ_GLU=1`.
- 5070 Ti memory pressure reduction by sizing native FP4 activation scratch as FP4 blocks instead of Q8 blocks.
- 5070 Ti NVFP4 MMQ X tile cap of 64 by default on SM120, with `GGML_CUDA_NVFP4_MMQ_X_MAX` available for local tuning.
- 5070 Ti NVFP4 MMQ now uses a 4-warp, Y=64 Blackwell tile with a 4-CTA launch-bound hint. Shared memory still limits the hot kernel to three CTAs per SM, but the stricter hint reduces register allocation from 168 to 128 registers/thread and improves prefill throughput without changing other quantized MMQ types.
- The SM120 NVFP4 weight loader now maps the 8 lanes assigned to a row over 32-bit words within each FP4 block instead of assigning one 36-byte `block_nvfp4` to each lane. This keeps the shared tile layout unchanged while making the packed weight loads and shared stores less strided.
- An opt-in CUDA-side NVFP4 MMQ repack cache is available through `GGML_CUDA_NVFP4_REPACK_CACHE_MB`. It keeps canonical GGUF/GGML tensor storage untouched and creates a budgeted per-tensor sidecar laid out as 64 packed FP4 words plus 8 scale words for each 512-value MMQ row segment. Compute buffers are skipped, and sidecars are marked stale on partial uploads or memset.
- 5070 Ti NVFP4 activation quantization uses direct max-derived subblock scales by default on SM120, with `GGML_CUDA_NVFP4_QUANT_SCALE_RADIUS=2` available to restore the previous scale search.
- NVFP4 `.scale` tensors are now applied to the base matmul result before bias and before LoRA deltas for the qwen35moe paths covered here.
- Eligible NVFP4 `.scale` multiplies are fused into the native CUDA MMQ write-back epilogue for dense `MUL_MAT` and MoE `MUL_MAT_ID` prefill/batch cases. Decode-sized batches stay on MMVQ to avoid regressing token generation.
- `.input_scale` tensors are loaded, attached to NVFP4 `MUL_MAT`/`MUL_MAT_ID` nodes as metadata, and consumed by the native MMQ activation quantizer. They are not represented as post-matmul graph multipliers.
- The activation quantizer divides F32 activations by the static `.input_scale` before choosing per-16 UE4M3 scales; the MMQ epilogue multiplies the FP32 accumulator by the product of weight `.scale` and activation `.input_scale`.
- Dense `.input_scale` uses a scalar tensor. MoE `MUL_MAT_ID` uses the existing `expert_bounds` mapping to select the per-expert activation input scale for the sorted activation rows.
- `GGML_CUDA_NVFP4_DEBUG=1` now emits a model-load readiness summary for NVFP4 tensors, including `.scale`, `.input_scale`, and K multiple-of-64 coverage.
- `GGML_CUDA_NVFP4_DEBUG=1` also logs when dense or MoE `.scale` fusion is selected for the native MMQ epilogue and when `.input_scale` is consumed by the MMQ activation quantizer.
- The opt-in MMQ+GLU fusion is disabled for graph nodes carrying `.input_scale` metadata until that fused path handles the activation tensor scale explicitly.
- `GGML_CUDA_NVFP4_NATIVE=1` forces the native SM120 path to fail if unavailable. `GGML_CUDA_NVFP4_DEBUG=1` logs the native path. `GGML_CUDA_NVFP4_NATIVE=0` fails closed on this build because there is no safe non-native NVFP4 MMQ fallback specialization in the Blackwell-compiled CUDA path.

## Local Validation

Build:

```sh
cmake --build build-cuda --target test-backend-ops llama-bench llama-cli llama-completion -j 8
```

CUDA backend correctness:

```sh
./build-cuda/bin/test-backend-ops -o MUL_MAT -b CUDA0 -p 'type_a=nvfp4'
./build-cuda/bin/test-backend-ops -o MUL_MAT_ID -b CUDA0 -p 'type_a=nvfp4'
./build-cuda/bin/test-llama-graph
./build-cuda/bin/test-backend-ops -o MUL_MAT_SCALE -b CUDA0
./build-cuda/bin/test-backend-ops -o MUL_MAT_ID_SCALE -b CUDA0
GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/test-backend-ops -o MUL_MAT_SCALE -b CUDA0
GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/test-backend-ops -o MUL_MAT_ID_SCALE -b CUDA0
GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/test-backend-ops -o MUL_MAT_INPUT_SCALE -b CUDA0
GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/test-backend-ops -o MUL_MAT_ID_INPUT_SCALE -b CUDA0
GGML_CUDA_NVFP4_REPACK_CACHE_MB=64 GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/test-backend-ops -o MUL_MAT -b CUDA0 -p 'type_a=nvfp4'
```

Results:

- `MUL_MAT type_a=nvfp4`: 41/41 passed.
- `MUL_MAT_ID type_a=nvfp4`: 72/72 passed.
- `test-llama-graph`: passed. This structurally verifies that `input_scale` is attached to the matmul node as metadata and is not used as a post-matmul multiplier in `build_lora_mm` or `build_lora_mm_id`.
- `MUL_MAT_SCALE`: 2/2 passed.
- `MUL_MAT_ID_SCALE`: 2/2 passed.
- `MUL_MAT_INPUT_SCALE`: 1/1 passed.
- `MUL_MAT_ID_INPUT_SCALE`: 1/1 passed.
- Debug scale-fusion smoke: dense and `MUL_MAT_ID` scale tests both logged MMQ epilogue fusion.
- Debug input-scale smoke: dense and `MUL_MAT_ID` input-scale tests both logged activation `input_scale` consumption in the MMQ quantizer.
- `GGML_CUDA_NVFP4_REPACK_CACHE_MB=64` `MUL_MAT type_a=nvfp4`: 41/41 passed, with the debug log confirming that the sidecar MMQ weight loader was selected on eligible k=1024 cases.

Debug readiness smoke test:

```sh
GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/llama-cli \
    -hf sroecker/Qwen3.6-35B-REAP-Pruned-ratio-0.5-NVFP4-GGUF \
    -ngl 999 -fa 1 -st --reasoning off --no-display-prompt --temp 0 --seed 1 \
    -p 'Paris is the capital of' -n 1
```

Observed readiness summary:

```text
CUDA NVFP4 native readiness: nvfp4 tensors = 280, known matmul weights = 280, with .scale = 280, with .input_scale metadata = 280
CUDA NVFP4 native readiness: input_scale is activation-quantization metadata and is not applied as a post-matmul multiplier
```

Benchmark command:

```sh
./build-cuda/bin/llama-bench \
    -hf sroecker/Qwen3.6-35B-REAP-Pruned-ratio-0.5-NVFP4-GGUF \
    -ngl 999 -fa 1 -p 15000 -n 128
```

5070 Ti results:

| Mode | pp15000 | tg128 |
| --- | ---: | ---: |
| `GGML_CUDA_NVFP4_MMQ_GLU=0` | `5596.62 +/- 6.98 t/s` | `127.35 +/- 0.64 t/s` |
| `GGML_CUDA_NVFP4_MMQ_GLU=1` | `5586.75 +/- 4.66 t/s` | `126.18 +/- 0.57 t/s` |
| SM120 default `NVFP4_MMQ_X_MAX=64` | `5611.50 +/- 11.64 t/s` | `127.04 +/- 0.80 t/s` |
| SM120 default cap 64 + quant radius 0 | `6165.73 +/- 4.41 t/s` | `127.12 +/- 0.70 t/s` |
| Scale-order graph pass | `6171.70 +/- 8.81 t/s` | `127.10 +/- 0.62 t/s` |
| Readiness/input-scale guard pass | `6167.98 +/- 9.29 t/s` | `127.14 +/- 0.69 t/s` |
| Batch-gated `.scale` MMQ epilogue fusion | `6294.69 +/- 3.68 t/s` | `127.01 +/- 0.75 t/s` |
| Static activation `.input_scale` quantizer | `6248.47 +/- 9.66 t/s` | `127.13 +/- 0.65 t/s` |
| Source-aligned rerun after profiling | `6244.19 +/- 5.15 t/s` | `127.07 +/- 0.64 t/s` |
| SM120 NVFP4 MMQ 4-warp Y64 3-CTA launch-bound pass | `6321.87 +/- 8.76 t/s` | `127.02 +/- 0.68 t/s` |
| SM120 NVFP4 MMQ 4-warp Y64 4-CTA launch-bound hint | `6433.62 +/- 8.94 t/s` | `127.00 +/- 0.67 t/s` |
| SM120 NVFP4 lane-remapped weight loader | `6480.10 +/- 6.03 t/s` | `127.06 +/- 0.66 t/s` |
| Current source, repack cache disabled | `6470.57 +/- 3.89 t/s` | `126.91 +/- 0.89 t/s` |
| `GGML_CUDA_NVFP4_REPACK_CACHE_MB=64` | `6479.81 +/- 17.12 t/s` | `126.96 +/- 0.92 t/s` |
| `GGML_CUDA_NVFP4_REPACK_CACHE_MB=256` | `6469.15 +/- 14.25 t/s` | `126.89 +/- 0.95 t/s` |

The GLU fusion path was slightly slower in this benchmark, so it remains opt-in.

Prompt sanity check:

```sh
GGML_CUDA_NVFP4_MMQ_GLU=1 ./build-cuda/bin/llama-cli \
    -hf sroecker/Qwen3.6-35B-REAP-Pruned-ratio-0.5-NVFP4-GGUF \
    -ngl 999 -fa 1 -st --reasoning off --no-display-prompt --simple-io \
    -p 'Paris is the capital of' -n 32 --temp 0 --seed 1
```

Observed completion:

```text
Paris is the capital of **France**.
```

After the scale-order pass, the same prompt completed correctly:

```text
Paris is the capital of **France**.
```

Default SM120 sanity check:

```sh
./build-cuda/bin/llama-cli \
    -hf sroecker/Qwen3.6-35B-REAP-Pruned-ratio-0.5-NVFP4-GGUF \
    -ngl 999 -fa 1 -st --simple-io --no-display-prompt \
    --reasoning off --temp 0 --seed 1 \
    -p 'Paris is the capital of' -n 32
```

Observed completion:

```text
Paris is the capital of **France**.
```

After adding static activation `.input_scale` quantization, the single-turn `llama-cli` check again completed correctly:

```sh
GGML_CUDA_NVFP4_DEBUG=1 ./build-cuda/bin/llama-cli \
    -hf sroecker/Qwen3.6-35B-REAP-Pruned-ratio-0.5-NVFP4-GGUF \
    -ngl 999 -fa 1 --simple-io --reasoning off -st \
    -p 'Paris is the capital of' -n 24 --temp 0
```

Observed completion:

```text
Paris is the capital of **France**.
```

After the 4-warp Y64 MMQ pass, the same prompt again completed correctly:

```text
Paris is the capital of **France**.
```

After the 4-CTA launch-bound hint, the single-turn `llama-cli` prompt again completed correctly with `--reasoning off`:

```text
Paris is the capital of **France**.
```

## Profiling

`nsys` on the requested benchmark with GLU disabled showed these leading CUDA kernel costs on the 5070 Ti:

| Kernel | Time | Share |
| --- | ---: | ---: |
| `mul_mat_q<NVFP4,128>` | `1.318 s` | `25.3%` |
| `gated_delta_net_cuda<128>` | `0.902 s` | `17.3%` |
| `quantize_mmq_nvfp4` | `0.648 s` | `12.4%` |
| `flash_attn_ext_f16` | `0.349 s` | `6.7%` |

`ncu` for `mul_mat_q<NVFP4,128>` reported 255 registers/thread, about 63 KiB shared memory/block, 16.6% achieved occupancy, and only about 0.28 eligible warps/scheduler. The dominant stalls were long scoreboard, wait, math pipe throttle, MIO throttle, and LG throttle.

A local sweep of `GGML_CUDA_NVFP4_MMQ_X_MAX` found the best repeatable result at 64 for this card:

| Cap | pp15000 | tg128 |
| --- | ---: | ---: |
| default before cap | `5542.15 +/- 10.72 t/s` | `127.02 +/- 0.68 t/s` |
| 64 | `5580.62 +/- 7.29 t/s` | `126.94 +/- 0.65 t/s` |
| 80 | `5552.74 +/- 3.75 t/s` | `126.95 +/- 0.64 t/s` |
| 96 | `5553.77 +/- 5.95 t/s` | `126.89 +/- 0.64 t/s` |

`ncu` for `mul_mat_q<NVFP4,64>` still reported 255 registers/thread and 16.5% achieved occupancy, but shared memory/block dropped to 53.5 KiB. The win is small and prompt-processing specific, so the cap is restricted to SM120 rather than all Blackwell devices.

The original NVFP4 activation quantizer tested five FP8 scale codes per 16-value subblock (`+/- 2` around the max-derived code). On this 5070 Ti that cost was visible in `nsys`, so a local sweep tested smaller search radii. Radius 0 was then confirmed with five repeats:

| Quant scale radius | pp15000 | tg128 |
| --- | ---: | ---: |
| 0 | `6163.84 +/- 3.33 t/s` | `127.19 +/- 0.68 t/s` |
| 1 | `5843.59 t/s` | `125.90 t/s` |
| 2 | `5615.88 t/s` | `126.00 t/s` |

Radius 0 passed the focused CUDA backend tests and the `Paris is the capital of` sanity check, so SM120 defaults to radius 0. Radius 2 remains available through `GGML_CUDA_NVFP4_QUANT_SCALE_RADIUS=2` for comparing against the previous error-search behavior.

Final `nsys` with the no-env SM120 defaults:

| Kernel | Time | Share |
| --- | ---: | ---: |
| `mul_mat_q<NVFP4,64>` | `1.366 s` | `28.7%` |
| `gated_delta_net_cuda<128>` | `0.905 s` | `19.1%` |
| `flash_attn_ext_f16` | `0.352 s` | `7.4%` |
| `quantize_mmq_nvfp4<0>` | `0.192 s` | `4.0%` |

`ncu` for `quantize_mmq_nvfp4<0>` reported 40 registers/thread, 1.02 KiB shared memory/block, 100% theoretical occupancy, 85.5% achieved occupancy, and 1.93 eligible warps/scheduler. The remaining dominant stall is long scoreboard, but the kernel is now a much smaller part of the benchmark.

After adding static activation `.input_scale`, a fresh `nsys` run of the requested benchmark captured three benchmark repetitions. Per-launch comparison is therefore more useful than total time:

| Kernel | Avg before static `.input_scale` | Avg with static `.input_scale` |
| --- | ---: | ---: |
| `mul_mat_q<NVFP4,64>` | `83.1 us` | `85.3 us` |
| `quantize_mmq_nvfp4<0>` | `11.7 us` | `12.8 us` |

The matching `ncu` samples showed:

| Kernel | Duration | Registers/thread | Achieved occupancy | Eligible warps/scheduler |
| --- | ---: | ---: | ---: | ---: |
| `quantize_mmq_nvfp4<0>` with `.input_scale` | `39.94 us` | `39` | `86.98%` | `2.17` |
| `mul_mat_q<NVFP4,64,apply_scale>` with `.scale * .input_scale` | `177.31 us` | `255` | `16.53%` | `0.29` |

A warp-broadcast experiment for the activation input-scale lookup was tested locally. It did not improve the profiled quantizer launch (`40.06 us`) or the requested bench (`6244.45 +/- 6.67 t/s` pp15000, `127.04 +/- 0.68 t/s` tg128), so it was dropped. The next useful optimization target is the native MMQ kernel's register/shared-memory pressure, not the scalar input-scale lookup.

An input-scale-era sweep of `GGML_CUDA_NVFP4_MMQ_X_MAX` kept the same conclusion: cap 64 is the best conservative setting on the 5070 Ti, while smaller caps lose prefill throughput and larger caps do not recover a repeatable win.

| Cap | pp15000 | tg128 |
| --- | ---: | ---: |
| 32 | `5535.27 +/- 7.93 t/s` | `127.09 +/- 0.65 t/s` |
| 40 | `5827.45 +/- 2.22 t/s` | `126.94 +/- 0.67 t/s` |
| 48 | `6002.63 +/- 9.93 t/s` | `126.83 +/- 0.66 t/s` |
| 56 | `5994.98 +/- 5.40 t/s` | `126.86 +/- 0.68 t/s` |
| 64 | `6212.81 +/- 3.01 t/s` | `126.83 +/- 0.67 t/s` |
| 72 | `6209.48 +/- 4.06 t/s` | `126.83 +/- 0.65 t/s` |
| 80 | `6189.56 +/- 7.22 t/s` | `126.80 +/- 0.65 t/s` |
| 96 | `6190.20 +/- 9.18 t/s` | `126.87 +/- 0.64 t/s` |

A temporary local switch to disable stream-k for NVFP4 MMQ was also tested. The forced-off path was far slower than the default and was interrupted before completing the first benchmark row, so no code was kept. Stream-k should stay enabled for this workload; the next optimization has to reduce the native MMQ kernel's 255-register pressure or its shared-memory footprint rather than relying on dispatch knobs.

The first structural Y64 experiment changed only `mmq_y` and failed at compile time: the current MMA writeback requires `nwarps * tile_C::I == mmq_y`. With the original 8-warp shape, `mmq_y=64` violates that invariant. The kept version therefore changes NVFP4 on Blackwell as a matched shape: 4 warps and `mmq_y=64`. A scale-mode specialization that split `none/output/input/both` epilogues was also tested and dropped; it did not reduce the active kernel's register pressure or improve the benchmark.

The first kept Y64 pass used a 3-CTA launch-bound target. A follow-up launch-bound sweep showed that the looser 2-CTA hint regressed prefill, while the stricter 4-CTA hint improved prefill even though shared memory still caps the active blocks at three CTAs per SM:

| Launch-bound hint | Active kernel registers/thread | Stack | pp15000 | tg128 |
| --- | ---: | ---: | ---: | ---: |
| 2 CTAs | `255` | `8 B` | `6235.99 +/- 3.87 t/s` | `126.88 +/- 0.94 t/s` |
| 3 CTAs | `168` | `48 B` | `6321.87 +/- 8.76 t/s` | `127.02 +/- 0.68 t/s` |
| 4 CTAs | `128` | `72 B` | `6433.62 +/- 8.94 t/s` | `127.00 +/- 0.67 t/s` |

Fresh `ncu` for the 4-CTA hint on `mul_mat_q<NVFP4,64,apply_scale>` showed that registers are no longer the active occupancy limiter:

| Kernel shape | Registers/thread | Dynamic shared/block | Theoretical occupancy | Achieved occupancy | Active warps/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8-warp Y128 baseline | `255` | `~52 KiB` | `16.67%` | `16.53%` | `~7.9` |
| 8-warp Y128 with 2-CTA launch bound | `128` | `~52 KiB` | `16.67%` | `16.49%` | `~7.9` |
| 4-warp Y64, no register cap | `255` | `30.98 KiB` | `16.67%` | `15.63%` | `7.50` |
| 4-warp Y64 with 3-CTA launch bound | `168` | `30.98 KiB` | `25.00%` | `22.94%` | `11.01` |
| 4-warp Y64 with 4-CTA launch-bound hint | `128` | `30.98 KiB` | `25.00%` | `22.95%` | `11.02` |

The 4-CTA hint did not raise occupancy beyond the 3-CTA result because shared memory is now the limiting resource, but it shortened the sampled 8192-grid MMQ launch from about `157.15 us` to `148.03 us`. The fresh `nsys` trace for the requested benchmark with three repetitions showed `mul_mat_q<NVFP4,64,apply_scale>` at `26.8%` of CUDA kernel time, `gated_delta_net_cuda<128>` at `20.1%`, `flash_attn_ext_f16` at `7.8%`, and `quantize_mmq_nvfp4<0>` at `4.7%`.

The next kept loader pass remapped the 8 row lanes from "one lane owns one `block_nvfp4`" to "one lane owns one 32-bit packed-q word while the row lanes iterate over the 8 blocks together." This does not change the canonical GGUF/GGML storage layout and does not add a repack cache, but it makes each row group load contiguous words from one NVFP4 block at a time and store contiguous words into the existing shared tile layout.

Fresh `ncu` on the same sampled `mul_mat_q<NVFP4,64,apply_scale>` launch showed:

| Loader | Duration | Global-load useful bytes/sector | Global excessive sectors | Shared-store bank conflicts | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-CTA hint baseline | `148.03 us` | `7.0 / 32 B` | `19,738,160 / 25,614,640` | `2.4-way`, `848,500` conflicts | `22.95%` |
| Lane-remapped loader | `143.33 us` | `17.2 / 32 B` | `4,685,360 / 10,561,840` | `1.6-way`, `798,866` conflicts | `22.87%` |

The active specialization remains resource-stable at 128 registers/thread and 30.98 KiB dynamic shared memory/block. `cuobjdump` reports the no-check/apply-scale specialization stack at `64 B`, down from `72 B` in the previous kept source.

A frag-major MMA staging experiment was also tested to reduce live A/B fragments. It passed the focused NVFP4 tests, but it increased the active no-check/apply-scale stack to `80 B` and regressed the r3 benchmark to `6456.25 +/- 9.05 t/s` pp15000 versus `6475.53 +/- 8.04 t/s` for the kept lane-remapped loader, so that staging change was dropped.

The first opt-in repack-cache prototype passed correctness but did not produce a reliable benchmark win. A 64 MiB cache budget stores sidecars for 47 tensors in this model load and the `nsys` pass showed the cached specialization as real but small: `mul_mat_q<NVFP4,64,apply_scale,x_repacked=false>` remained about `25.3%` of CUDA kernel time, while `x_repacked=true` was about `1.1%`. The upload-time `nvfp4_repack_mmq_kernel` was negligible after model load.

`ncu` on the cached specialization reported 128 registers/thread and 30.98 KiB dynamic shared memory/block, matching the current uncached specialization's resource shape. The sampled cached launch had only a 70-block grid, so achieved occupancy was low (`8.32%`) due to underfilled work rather than a new resource cliff. Because the r3 benchmark is within noise at 64 MiB and slightly worse at 256 MiB, the cache stays opt-in rather than becoming the default.

## Notes

- `llama-cli --no-conversation` is rejected for this chat-template model; `-st` was used for a single-turn `llama-cli` check.
- `llama-cli -st --reasoning off` with `GGML_CUDA_NVFP4_REPACK_CACHE_MB=64` completed `Paris is the capital of` as `Paris is the capital of **France**.`
- `llama-completion -no-cnv` with `GGML_CUDA_NVFP4_REPACK_CACHE_MB=64` completed the raw prompt with `France`.
- The first `.scale` epilogue fusion draft applied to decode-sized workloads too and dropped `tg128` to about `107 t/s`; the current version mirrors normal dispatch and only fuses cases that should use MMQ rather than MMVQ.
- The first priority for the 5070 Ti is keeping the native path within the 16 GB memory envelope while preserving correctness.
- Profiling artifacts are local under `profiles/native-nvfp4/` and are not intended for upstream submission.
