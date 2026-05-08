#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdlib>
#include <cstring>

static int ggml_cuda_nvfp4_native_mode() {
    const char * env = getenv("GGML_CUDA_NVFP4_NATIVE");
    if (!env || strcmp(env, "auto") == 0) {
        return -1;
    }
    return std::atoi(env) != 0 ? 1 : 0;
}

static bool ggml_cuda_nvfp4_debug_enabled() {
    const char * env = getenv("GGML_CUDA_NVFP4_DEBUG");
    return env && std::atoi(env) != 0;
}

static bool ggml_cuda_nvfp4_native_available(const int cc) {
    const int mode = ggml_cuda_nvfp4_native_mode();

    if (mode == 0) {
        GGML_ABORT("GGML_CUDA_NVFP4_NATIVE=0 requested, but this build has no safe non-native NVFP4 MMQ fallback");
    }

    if (!blackwell_mma_available(cc)) {
        if (mode == 1) {
            GGML_ABORT("GGML_CUDA_NVFP4_NATIVE=1 requested, but SM120/SM121 FP4 MMA is unavailable");
        }
        static bool logged = false;
        if (!logged && ggml_cuda_nvfp4_debug_enabled()) {
            GGML_LOG_INFO("CUDA NVFP4 native: disabled, unsupported compute capability sm_%d\n", cc / 10);
            logged = true;
        }
        return false;
    }

    static bool logged = false;
    if (!logged && (mode == 1 || ggml_cuda_nvfp4_debug_enabled())) {
        GGML_LOG_INFO("CUDA NVFP4 native: enabled, arch sm_%d, kernel mma.sync mxf4nvf4 scale_vec::4X\n", cc / 10);
        logged = true;
    }
    return true;
}

static __global__ void nvfp4_repack_mmq_kernel(const block_nvfp4 * src, uint32_t * dst, const int64_t ngroups) {
    const int64_t group = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= ngroups) {
        return;
    }

    const block_nvfp4 * src_group = src + group * (MMQ_ITER_K_FP4 / QK_NVFP4);
    uint32_t * dst_group = dst + group * (MMQ_ITER_K_FP4 / 8 + MMQ_ITER_K_FP4 / QK_NVFP4);

#pragma unroll
    for (int kb = 0; kb < MMQ_ITER_K_FP4 / QK_NVFP4; ++kb) {
        const uint32_t * src_qs = (const uint32_t *) src_group[kb].qs;
#pragma unroll
        for (int w = 0; w < QK_NVFP4 / 8; ++w) {
            dst_group[(QK_NVFP4 / 8) * kb + w] = src_qs[w];
        }

        dst_group[MMQ_ITER_K_FP4 / 8 + kb] = *((const uint32_t *) src_group[kb].d);
    }
}

void ggml_cuda_nvfp4_repack_mmq_cuda(const char * src, void * dst, const int64_t nblocks, cudaStream_t stream) {
    GGML_ASSERT(nblocks % (MMQ_ITER_K_FP4 / QK_NVFP4) == 0);

    const int64_t ngroups = nblocks / (MMQ_ITER_K_FP4 / QK_NVFP4);
    if (ngroups == 0) {
        return;
    }

    const int block_size = 128;
    const dim3 block_nums((ngroups + block_size - 1) / block_size);
    nvfp4_repack_mmq_kernel<<<block_nums, block_size, 0, stream>>>((const block_nvfp4 *) src, (uint32_t *) dst, ngroups);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static size_t ggml_cuda_mmq_src1_nbytes(
        const ggml_type type_x, const int cc, const int64_t nrows, const int64_t ncols_padded, const bool use_native_fp4) {
    const size_t row_size = use_native_fp4
        ? ncols_padded * sizeof(block_fp4_mmq) / QK_K
        : ncols_padded * sizeof(block_q8_1)    / QK8_1;

    return nrows * row_size + get_mmq_x_max_for_type(type_x, cc) * sizeof(block_q8_1_mmq);
}

static int64_t ggml_cuda_mmq_src1_stride(
        const int64_t nrows, const int64_t ncols_padded, const bool use_native_fp4) {
    return use_native_fp4
        ? nrows * ncols_padded * sizeof(block_fp4_mmq) / (QK_K  * sizeof(int))
        : nrows * ncols_padded * sizeof(block_q8_1)    / (QK8_1 * sizeof(int));
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_tensor * output_scale, const ggml_tensor * input_scale) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.
    GGML_ASSERT(!output_scale || output_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(!output_scale || src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(!input_scale  || input_scale->type  == GGML_TYPE_F32);
    GGML_ASSERT(!input_scale  || src0->type == GGML_TYPE_NVFP4);

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    const bool use_native_fp4 = src0->type == GGML_TYPE_MXFP4 ? blackwell_mma_available(cc) :
                                src0->type == GGML_TYPE_NVFP4 ? ggml_cuda_nvfp4_native_available(cc) : false;
    const ggml_cuda_nvfp4_repack_cache * repack_cache =
        src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? ggml_cuda_nvfp4_get_repack_cache(src0) : nullptr;
    const bool use_repack_cache = repack_cache && repack_cache->ready;
    if (use_repack_cache) {
        src0_d = (const char *) repack_cache->data;

        if (ggml_cuda_nvfp4_debug_enabled()) {
            static bool logged = false;
            if (!logged) {
                GGML_LOG_INFO("CUDA NVFP4 repack cache: using MMQ sidecar weights\n");
                logged = true;
            }
        }
    }

    if (output_scale) {
        GGML_ASSERT(use_native_fp4);
        GGML_ASSERT(ids || ggml_nelements(output_scale) == 1);
        GGML_ASSERT(!ids || output_scale->ne[0] == ne02);
    }
    if (input_scale) {
        GGML_ASSERT(use_native_fp4);
        GGML_ASSERT(ids || ggml_nelements(input_scale) == 1);
        GGML_ASSERT(!ids || input_scale->ne[0] == ne02);
    }

    const float * output_scale_d      = output_scale ? (const float *) output_scale->data : nullptr;
    const int     output_scale_stride = output_scale ? (ids ? 1 : 0) : 0;
    const float * input_scale_d       = input_scale  ? (const float *) input_scale->data  : nullptr;
    const int     input_scale_stride  = input_scale  ? (ids ? 1 : 0) : 0;
    if (input_scale && ggml_cuda_nvfp4_debug_enabled()) {
        static bool logged = false;
        if (!logged) {
            GGML_LOG_INFO("CUDA NVFP4 native: using activation input_scale in MMQ quantizer\n");
            logged = true;
        }
    }
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4 && ggml_cuda_nvfp4_debug_enabled()) {
        const int mode = (use_repack_cache ? 1 : 0) | (output_scale ? 2 : 0) | (input_scale ? 4 : 0);
        static bool logged[8] = {};
        if (!logged[mode]) {
            GGML_LOG_INFO("CUDA NVFP4 native MMQ dispatch: weights=%s, output_scale=%s, input_scale=%s\n",
                use_repack_cache ? "repacked" : "canonical",
                output_scale ? "yes" : "no",
                input_scale ? "yes" : "no");
            logged[mode] = true;
        }
    }

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ggml_cuda_mmq_src1_nbytes(
            src0->type, cc, ne13*ne12*ne11, ne10_padded, use_native_fp4);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, input_scale_d, nullptr, input_scale ? 1 : 0,
                                        input_scale_stride, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        const int64_t s12 = ggml_cuda_mmq_src1_stride(ne11, ne10_padded, use_native_fp4);
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            output_scale_d, output_scale_stride,
            input_scale_d, input_scale_stride,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1,
            use_repack_cache};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ggml_cuda_mmq_src1_nbytes(
        src0->type, cc, ne12*n_expert_used, ne10_padded, use_native_fp4);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, input_scale_d, expert_bounds.get(),
                                    input_scale ? ne02 : 0, input_scale_stride, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = ggml_cuda_mmq_src1_stride(ne11, ne10_padded, use_native_fp4);
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        output_scale_d, output_scale_stride,
        input_scale_d, input_scale_stride,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12,
        use_repack_cache};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_mul_mat_q_glu(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(src0->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(fusion && fusion->gate);
    GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias);
    GGML_ASSERT(fusion->gate->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(!fusion->x_scale || fusion->x_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(!fusion->gate_scale || fusion->gate_scale->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    GGML_ASSERT(blackwell_mma_available(cc));

    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: using experimental NVFP4 MMQ+GLU path\n", __func__);
        logged = true;
    }

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(nb00       == ts_src0);
    GGML_ASSERT(nb10       == ts_src1);
    GGML_ASSERT(nb0        == ts_dst);
    GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const char  * gate_d = (const char  *) fusion->gate->data;
    const float * x_scale_d    = fusion->x_scale    ? (const float *) fusion->x_scale->data    : nullptr;
    const float * gate_scale_d = fusion->gate_scale ? (const float *) fusion->gate_scale->data : nullptr;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);
    GGML_ASSERT(!fusion->x_scale    || fusion->x_scale->ne[0]    == ne02);
    GGML_ASSERT(!fusion->gate_scale || fusion->gate_scale->ne[0] == ne02);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);
    GGML_ASSERT(ne12 == 1);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_fp4 =
        ne12*n_expert_used*ne10_padded * sizeof(block_fp4_mmq)/QK_K +
        8*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_fp4(ctx.pool(), nbytes_src1_fp4);

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_fp4.get(), src0->type, ne10, s11, s12, s13,
                              ne10_padded, ne_get_rows, 1, 1, nullptr, nullptr, 0, 0, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int));
    const int64_t s13 = ne12 * s12;

    const mmq_glu_args args = {
        src0_d, gate_d, x_scale_d, gate_scale_d,
        (const int *) src1_fp4.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, fusion->glu_op};

    launch_mul_mat_q_glu_nvfp4<8>(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, bool src0_repacked_i,
    const float * src1_ddf_i, const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        nullptr, 0,
        nullptr, 0,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols,
        src0_repacked_i};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
