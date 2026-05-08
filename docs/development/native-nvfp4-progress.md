# Native NVFP4 CUDA Progress

Date: 2026-05-08

Branch: `feature/native-nvfp4-5070ti-local`

Target GPU: NVIDIA GeForce RTX 5070 Ti, compute capability 12.0, 16 GB class VRAM.

This is local progress for native NVFP4 CUDA kernels without dequantizing weights to a wider format before MMQ. The target is Blackwell consumer SM120 behavior, not DGX Spark or GB200 server behavior.

## Current Scope

- GGUF conversion support for compressed-tensors NVFP4 packing.
- CUDA MMQ native FP4 path for MXFP4 and NVFP4 on Blackwell.
- NVFP4 block-scaled MMA plumbing using SM120-compatible warp-level MMA.
- Experimental NVFP4 MMQ+GLU fusion guarded by `GGML_CUDA_NVFP4_MMQ_GLU=1`.
- 5070 Ti memory pressure reduction by sizing native FP4 activation scratch as FP4 blocks instead of Q8 blocks.
- 5070 Ti NVFP4 MMQ X tile cap of 64 by default on SM120, with `GGML_CUDA_NVFP4_MMQ_X_MAX` available for local tuning.
- 5070 Ti NVFP4 activation quantization uses direct max-derived subblock scales by default on SM120, with `GGML_CUDA_NVFP4_QUANT_SCALE_RADIUS=2` available to restore the previous scale search.

## Local Validation

Build:

```sh
cmake --build build-cuda --target test-backend-ops llama-bench llama-cli -j 8
```

CUDA backend correctness:

```sh
./build-cuda/bin/test-backend-ops -o MUL_MAT -b CUDA0 -p 'type_a=nvfp4'
./build-cuda/bin/test-backend-ops -o MUL_MAT_ID -b CUDA0 -p 'type_a=nvfp4'
```

Results:

- `MUL_MAT type_a=nvfp4`: 41/41 passed.
- `MUL_MAT_ID type_a=nvfp4`: 72/72 passed.

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

Default SM120 sanity check:

```sh
./build-cuda/bin/llama-cli \
    -hf sroecker/Qwen3.6-35B-REAP-Pruned-ratio-0.5-NVFP4-GGUF \
    -ngl 999 -fa 1 --reasoning off --no-warmup --temp 0 \
    -p 'Paris is the capital of' -n 16 --simple-io
```

Observed completion:

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

## Notes

- `llama-cli --no-conversation` is rejected for this chat-template model; `-st` was used for a single-turn `llama-cli` check.
- `llama-completion -no-cnv` also completed the raw prompt with `France`.
- The first priority for the 5070 Ti is keeping the native path within the 16 GB memory envelope while preserving correctness.
- Profiling artifacts are local under `profiles/native-nvfp4/` and are not intended for upstream submission.
