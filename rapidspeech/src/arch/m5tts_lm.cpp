// M5-TTS LM graphs: backbone prefill/step (llama-30M), MTP depth decoder
// steps (llama-20M), next-frame embedding. Mirrors onnx/graphs.py
// (Backbone + DepthFrame) with explicit tensor KV caches.

#include "arch/m5tts_graph.h"

#include "utils/rs_log.h"

#include <cstring>
#include <vector>

namespace m5tts {

// ---------------------------------------------------------------------------
// Prefill: backbone over [text frames ‖ audio-prompt frames]
//   text_ids: {Tt} i32 (raw vocab ids)
//   prompt_codes: {Tp * 16} i32 interleaved [t][k] codec codes 0..2047
// Outputs: last_hidden {hidden} f32 (post final norm, last column),
//          c0_logits {audio_vocab} f32, n_past = Tt + Tp.
// ---------------------------------------------------------------------------
bool Model::Prefill(ggml_backend_sched_t sched, const int32_t *text_ids,
                    int Tt, const int32_t *prompt_codes, int Tp,
                    std::vector<float> &last_hidden,
                    std::vector<float> &c0_logits, int &n_past) {
    const int D = hp_.hidden;
    const int T = Tt + Tp;
    if (T < 1) {
        RS_LOG_ERR("M5TTS: empty prefill sequence");
        return false;
    }

    graph::Scratch sc(1024, 8192);
    ggml_context *c = sc.ctx;
    ggml_cgraph *gf = sc.make_graph();

    ggml_tensor *ids_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, Tt);
    ggml_set_name(ids_in, "text_ids");
    ggml_set_input(ids_in);
    // 16 per-codebook 1-D index tensors (get_rows requires 1-D indices)
    ggml_tensor *codes_in[16];
    for (int k = 0; k < 16; k++) {
        codes_in[k] = ggml_new_tensor_1d(c, GGML_TYPE_I32, Tp);
        ggml_set_name(codes_in[k], ("codes_" + std::to_string(k)).c_str());
        ggml_set_input(codes_in[k]);
    }
    ggml_tensor *pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(pos_in, "positions");
    ggml_set_input(pos_in);

    // hidden = [text_emb(ids) ‖ Σ_k audio_emb(codes_k + 1 + k*2051)]
    ggml_tensor *hidden = nullptr;
    if (Tt > 0) hidden = ggml_get_rows(c, text_emb_, ids_in);  // {D, Tt}
    if (Tp > 0) {
        ggml_tensor *acc = nullptr;
        for (int k = 0; k < 16; k++) {
            ggml_tensor *row = ggml_get_rows(c, audio_emb_, codes_in[k]);  // {D, Tp}
            acc = (acc == nullptr) ? row : ggml_add(c, acc, row);
        }
        hidden = (hidden == nullptr) ? acc : ggml_concat(c, hidden, acc, 1);
    }
    if (hidden->ne[1] != T) {
        RS_LOG_ERR("M5TTS: prefill length mismatch");
        return false;
    }

    // causal mask {T, T} f16
    std::vector<ggml_fp16_t> mask = graph::build_mask(T, T, 0, true);
    ggml_tensor *mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, T, T);
    ggml_set_name(mask_in, "causal_mask");
    ggml_set_input(mask_in);

    for (int il = 0; il < hp_.bb_layers; il++) {
        graph::AttnCfg cfg;
        cfg.n_head = hp_.bb_heads;
        cfg.n_kv = hp_.bb_kv_heads;
        cfg.head_dim = hp_.bb_head_dim;
        cfg.inv_freq = bb_inv_freq_;
        cfg.kv_k = bb_kv_k_;
        cfg.kv_v = bb_kv_v_;
        cfg.il = il;
        cfg.n_past = 0;
        cfg.mask = mask_in;
        cfg.positions = pos_in;
        cfg.eps = hp_.norm_eps;
        hidden = graph::llama_block(c, gf, bb_layers_[il], hidden, cfg);
        }
    hidden = graph::rmsnorm(c, hidden, bb_norm_w_, hp_.norm_eps);

    // last column → c0 logits
    ggml_tensor *last =
        ggml_view_2d(c, hidden, D, 1, hidden->nb[1],
                     (size_t)(T - 1) * hidden->nb[1]);
    last = ggml_cont(c, last);
    ggml_tensor *c0 = graph::linear(c, codebook0_head_, last);  // {2051, 1}

    ggml_set_output(last);
    ggml_set_output(c0);
    ggml_build_forward_expand(gf, last);
    ggml_build_forward_expand(gf, c0);

    if (!AllocGraph(sched, gf, "prefill")) return false;
    SetInputI32(gf, "text_ids", text_ids, Tt);
    // apply +1 + codebook offsets here (matches infer_onnx.py prefix lookup)
    for (int k = 0; k < 16; k++) {
        std::vector<int32_t> ck((size_t)Tp);
        for (int t = 0; t < Tp; t++)
            ck[(size_t)t] = prompt_codes[(size_t)t * 16 + k] + 1 +
                            k * hp_.audio_vocab;
        SetInputI32(gf, ("codes_" + std::to_string(k)).c_str(), ck.data(), Tp);
    }
    std::vector<int32_t> pos((size_t)T);
    for (int i = 0; i < T; i++) pos[i] = i;
    SetInputI32(gf, "positions", pos.data(), T);
    {
        ggml_tensor *mt = ggml_graph_get_tensor(gf, "causal_mask");
        if (mt) ggml_backend_tensor_set(mt, mask.data(), 0,
                                        mask.size() * sizeof(ggml_fp16_t));
    }
    if (!ComputeGraph(sched, gf, "prefill")) return false;

    last_hidden.resize(D);
    ggml_backend_tensor_get(last, last_hidden.data(), 0, D * sizeof(float));
    c0_logits.resize(hp_.audio_vocab);
    ggml_backend_tensor_get(c0, c0_logits.data(), 0,
                            hp_.audio_vocab * sizeof(float));
    n_past = T;
    return true;
}

// ---------------------------------------------------------------------------
// Backbone step: one generated-frame token at position pos.
// embed: {hidden} f32. Outputs: last_hidden {hidden}, c0_logits {2051}.
// ---------------------------------------------------------------------------
bool Model::BackboneStep(ggml_backend_sched_t sched, const float *embed,
                         int pos, std::vector<float> &last_hidden,
                         std::vector<float> &c0_logits) {
    const int D = hp_.hidden;

    graph::Scratch sc(512, 4096);
    ggml_context *c = sc.ctx;
    ggml_cgraph *gf = sc.make_graph();

    ggml_tensor *embed_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 1);
    ggml_set_name(embed_in, "embed");
    ggml_set_input(embed_in);
    ggml_tensor *pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_name(pos_in, "positions");
    ggml_set_input(pos_in);

    ggml_tensor *hidden = embed_in;
    for (int il = 0; il < hp_.bb_layers; il++) {
        graph::AttnCfg cfg;
        cfg.n_head = hp_.bb_heads;
        cfg.n_kv = hp_.bb_kv_heads;
        cfg.head_dim = hp_.bb_head_dim;
        cfg.inv_freq = bb_inv_freq_;
        cfg.kv_k = bb_kv_k_;
        cfg.kv_v = bb_kv_v_;
        cfg.il = il;
        cfg.n_past = pos;
        cfg.mask = nullptr;
        cfg.positions = pos_in;
        cfg.eps = hp_.norm_eps;
        hidden = graph::llama_block(c, gf, bb_layers_[il], hidden, cfg);
    }
    hidden = graph::rmsnorm(c, hidden, bb_norm_w_, hp_.norm_eps);
    ggml_tensor *c0 = graph::linear(c, codebook0_head_, hidden);  // {2051, 1}
    ggml_set_output(hidden);
    ggml_set_output(c0);
    ggml_build_forward_expand(gf, hidden);
    ggml_build_forward_expand(gf, c0);

    if (!AllocGraph(sched, gf, "backbone step")) return false;
    SetInputF32(gf, "embed", embed, D);
    SetInputI32(gf, "positions", &pos, 1);
    if (!ComputeGraph(sched, gf, "backbone step")) return false;

    last_hidden.resize(D);
    ggml_backend_tensor_get(hidden, last_hidden.data(), 0, D * sizeof(float));
    c0_logits.resize(hp_.audio_vocab);
    ggml_backend_tensor_get(c0, c0_logits.data(), 0,
                            hp_.audio_vocab * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// MTP depth decoder step. Builds the decoder KV cache within one frame:
//   cb_idx = 0: two tokens [projection(last_hidden) ‖ projection(emb(c0))]
//               at positions [pos_base, pos_base+1]; logits for codebook 1.
//   cb_idx = k: one token projection(emb(code + (k-1)*2051)) at position
//               pos_base; logits for codebook k+1 (audio_head slice k).
// last_hidden: {hidden} f32 (only used for cb_idx == 0).
// codes: the codebook token to embed (c0 for step 0, c_{k-1} for step k).
// Output: logits {audio_vocab} f32.
// ---------------------------------------------------------------------------
bool Model::DepthStep(ggml_backend_sched_t sched, int cb_idx, int n_in,
                      const float *last_hidden, const int32_t *codes,
                      int pos_base, std::vector<float> &logits) {
    const int D = hp_.hidden;

    graph::Scratch sc(1024, 8192);
    ggml_context *c = sc.ctx;
    ggml_cgraph *gf = sc.make_graph();

    ggml_tensor *pos_in =
        ggml_new_tensor_1d(c, GGML_TYPE_I32, n_in > 1 ? 2 : 1);
    ggml_set_name(pos_in, "positions");
    ggml_set_input(pos_in);

    ggml_tensor *local = nullptr;
    if (n_in > 1) {
        ggml_tensor *lh = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 1);
        ggml_set_name(lh, "last_hidden");
        ggml_set_input(lh);
        ggml_tensor *code0 = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_name(code0, "code0");
        ggml_set_input(code0);
        ggml_tensor *emb = ggml_get_rows(c, audio_emb_, code0);  // {D, 1}
        ggml_tensor *p1 = graph::linear(c, projection_w_, lh);
        ggml_tensor *p2 = graph::linear(c, projection_w_, emb);
        local = ggml_concat(c, p1, p2, 1);  // {D, 2}
    } else {
        ggml_tensor *code_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        ggml_set_name(code_in, "code_in");
        ggml_set_input(code_in);
        ggml_tensor *emb = ggml_get_rows(c, audio_emb_, code_in);
        local = graph::linear(c, projection_w_, emb);  // {D, 1}
    }

    for (int il = 0; il < hp_.dc_layers; il++) {
        graph::AttnCfg cfg;
        cfg.n_head = hp_.dc_heads;
        cfg.n_kv = hp_.dc_kv_heads;
        cfg.head_dim = hp_.dc_head_dim;
        cfg.inv_freq = dc_inv_freq_;
        cfg.kv_k = dc_kv_k_;
        cfg.kv_v = dc_kv_v_;
        cfg.il = il;
        cfg.n_past = pos_base;
        cfg.mask = nullptr;
        cfg.positions = pos_in;
        cfg.eps = hp_.norm_eps;
        local = graph::llama_block(c, gf, dc_layers_[il], local, cfg);
    }
    local = graph::rmsnorm(c, local, dc_norm_w_, hp_.norm_eps);

    // last column → logits via audio_head slice (cb_idx - 1)
    const int T = (int)local->ne[1];
    ggml_tensor *last =
        ggml_view_2d(c, local, D, 1, local->nb[1],
                     (size_t)(T - 1) * local->nb[1]);
    last = ggml_cont(c, last);
    // audio_head GGUF ne = [2051, 512, 15] (PT [15, 512, 2051] reversed);
    // mul_mat needs ne[0] = K = 512, so transpose the codebook slice.
    ggml_tensor *head = ggml_view_3d(c, audio_head_, hp_.audio_vocab, D, 1,
                                     audio_head_->nb[1], audio_head_->nb[2],
                                     (size_t)(cb_idx - 1) * audio_head_->nb[2]);
    head = ggml_cont(c, ggml_transpose(c, head));  // {512, 2051}
    ggml_tensor *lg = graph::linear(c, head, last);  // {2051, 1}
    ggml_set_output(lg);
    ggml_build_forward_expand(gf, lg);

    if (!AllocGraph(sched, gf, "depth step")) return false;
    if (n_in > 1) {
        SetInputF32(gf, "last_hidden", last_hidden, D);
        SetInputI32(gf, "code0", codes, 1);
    } else {
        // host-side offset: code + (cb_idx - 1) * audio_vocab
        const int32_t idx = codes[0] + (cb_idx - 1) * hp_.audio_vocab;
        SetInputI32(gf, "code_in", &idx, 1);
    }
    std::vector<int32_t> pos((size_t)n_in);
    for (int i = 0; i < n_in; i++) pos[i] = pos_base + i;
    SetInputI32(gf, "positions", pos.data(), pos.size());
    if (!ComputeGraph(sched, gf, "depth step")) return false;

    logits.resize(hp_.audio_vocab);
    ggml_backend_tensor_get(lg, logits.data(), 0,
                            hp_.audio_vocab * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Next-frame input embedding: Σ_k audio_emb(codes[k] + k*2051) → {hidden}.
// codes: raw sampled tokens (0..2050) for all 16 codebooks.
// ---------------------------------------------------------------------------
bool Model::NextHidden(ggml_backend_sched_t sched, const int32_t *codes,
                       std::vector<float> &out) {
    const int D = hp_.hidden;

    graph::Scratch sc(256, 1024);
    ggml_context *c = sc.ctx;
    ggml_cgraph *gf = sc.make_graph();

    ggml_tensor *codes_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 16);
    ggml_set_name(codes_in, "codes");
    ggml_set_input(codes_in);

    ggml_tensor *rows = ggml_get_rows(c, audio_emb_, codes_in);  // {D, 16}
    ggml_tensor *tt = ggml_cont(c, ggml_transpose(c, rows));     // {16, D}
    ggml_tensor *sum = ggml_sum_rows(c, tt);                     // {1, D}
    ggml_tensor *out_t = ggml_reshape_2d(c, sum, D, 1);
    ggml_set_output(out_t);
    ggml_build_forward_expand(gf, out_t);

    if (!AllocGraph(sched, gf, "next hidden")) return false;
    std::vector<int32_t> idx(16);
    for (int k = 0; k < 16; k++) idx[k] = codes[k] + k * hp_.audio_vocab;
    SetInputI32(gf, "codes", idx.data(), 16);
    if (!ComputeGraph(sched, gf, "next hidden")) return false;

    out.resize(D);
    ggml_backend_tensor_get(out_t, out.data(), 0, D * sizeof(float));
    return true;
}

}  // namespace m5tts
