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

## Notes

- `llama-cli --no-conversation` is rejected for this chat-template model; `-st` was used for a single-turn `llama-cli` check.
- `llama-completion -no-cnv` also completed the raw prompt with `France`.
- The first priority for the 5070 Ti is keeping the native path within the 16 GB memory envelope while preserving correctness.
