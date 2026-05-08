#include "../src/llama-cparams.h"
#include "../src/llama-graph.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static llama_hparams make_hparams() {
    llama_hparams hparams = {};
    hparams.n_ctx_train        = 16;
    hparams.n_embd             = 16;
    hparams.n_layer            = 1;
    hparams.n_embd_head_k_full = 8;
    hparams.n_embd_head_v_full = 8;
    hparams.n_rot_full         = 8;
    hparams.n_head_arr[0]      = 2;
    hparams.n_head_kv_arr[0]   = 1;
    hparams.n_ff_arr[0]        = 32;
    hparams.f_norm_eps         = 1.0e-5f;
    hparams.f_norm_rms_eps     = 1.0e-5f;
    return hparams;
}

static llama_cparams make_cparams() {
    llama_cparams cparams = {};
    cparams.n_ctx            = 16;
    cparams.rope_freq_base   = 10000.0f;
    cparams.rope_freq_scale  = 1.0f;
    cparams.yarn_ext_factor  = -1.0f;
    cparams.yarn_attn_factor = 1.0f;
    cparams.yarn_beta_fast   = 32.0f;
    cparams.yarn_beta_slow   = 1.0f;
    cparams.causal_attn      = true;
    cparams.pooling_type     = LLAMA_POOLING_TYPE_NONE;
    return cparams;
}

static void init_graph_params(llm_graph_params & params, llm_graph_result & res, const llama_adapter_loras & loras) {
    params = {};
    params.arch      = LLM_ARCH_LLAMA;
    params.hparams   = make_hparams();
    params.cparams   = make_cparams();
    params.gtype     = LLM_GRAPH_TYPE_DEFAULT;
    params.loras     = &loras;
    params.n_outputs = 1;
    params.res       = &res;
}

static bool graph_contains(const ggml_tensor * cur, const ggml_tensor * needle) {
    if (cur == nullptr) {
        return false;
    }

    if (cur == needle) {
        return true;
    }

    for (const ggml_tensor * src : cur->src) {
        if (graph_contains(src, needle)) {
            return true;
        }
    }

    return false;
}

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static void test_build_lora_mm_input_scale_not_postmul() {
    llm_graph_result res(64);
    const llama_adapter_loras loras;
    llm_graph_params params;
    init_graph_params(params, res, loras);
    llm_graph_context graph(params);
    ggml_context * ctx = res.get_ctx();

    ggml_tensor * w      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * cur    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    ggml_tensor * w_s    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * w_in_s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_tensor * out = graph.build_lora_mm(w, cur, w_s, w_in_s);

    require(out->op == GGML_OP_MUL, "build_lora_mm must apply w_s as an output MUL");
    require(out->src[0] != nullptr && out->src[0]->op == GGML_OP_MUL_MAT, "build_lora_mm scale input must be MUL_MAT");
    require(out->src[1] == w_s, "build_lora_mm must multiply by w_s");
    require(!graph_contains(out, w_in_s), "build_lora_mm must not apply input_scale as a post-matmul multiplier");

    out = graph.build_lora_mm(w, cur, nullptr, w_in_s);
    require(out->op == GGML_OP_MUL_MAT, "build_lora_mm without w_s must return MUL_MAT");
    require(!graph_contains(out, w_in_s), "build_lora_mm without w_s must still ignore input_scale");
}

static void test_build_lora_mm_id_input_scale_not_postmul() {
    llm_graph_result res(96);
    const llama_adapter_loras loras;
    llm_graph_params params;
    init_graph_params(params, res, loras);
    llm_graph_context graph(params);
    ggml_context * ctx = res.get_ctx();

    ggml_tensor * w      = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 5);
    ggml_tensor * cur    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 3);
    ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 3);
    ggml_tensor * w_s    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    ggml_tensor * w_in_s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);

    ggml_tensor * out = graph.build_lora_mm_id(w, cur, ids, w_s, w_in_s);

    require(out->op == GGML_OP_MUL, "build_lora_mm_id must apply w_s as an output MUL");
    require(out->src[0] != nullptr && out->src[0]->op == GGML_OP_MUL_MAT_ID, "build_lora_mm_id scale input must be MUL_MAT_ID");
    require(out->src[1] != nullptr && out->src[1]->op == GGML_OP_GET_ROWS, "build_lora_mm_id scale branch must select per-expert rows");
    require(out->src[1]->src[0] != nullptr && out->src[1]->src[0]->op == GGML_OP_REPEAT, "build_lora_mm_id scale branch must repeat w_s");
    require(graph_contains(out->src[1], w_s), "build_lora_mm_id scale branch must use w_s");
    require(!graph_contains(out, w_in_s), "build_lora_mm_id must not apply input_scale as a post-matmul multiplier");

    out = graph.build_lora_mm_id(w, cur, ids, nullptr, w_in_s);
    require(out->op == GGML_OP_MUL_MAT_ID, "build_lora_mm_id without w_s must return MUL_MAT_ID");
    require(!graph_contains(out, w_in_s), "build_lora_mm_id without w_s must still ignore input_scale");
}

int main() {
    test_build_lora_mm_input_scale_not_postmul();
    test_build_lora_mm_id_input_scale_not_postmul();

    return 0;
}
