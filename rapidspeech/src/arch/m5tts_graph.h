// Shared ggml graph-building helpers for the M5-TTS implementation
// (m5tts_lm.cpp / m5tts_codec.cpp). All tensors are feature-major
// {features, T} unless noted; conv data is channels-first {C, T}.

#pragma once

#include "arch/m5tts.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace m5tts {
namespace graph {

// scratch context for one graph build+run; metadata must outlive compute
struct Scratch {
    std::vector<uint8_t> meta;
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    size_t n_nodes_budget = 0;

    Scratch(size_t n_tensors, size_t n_graph_nodes)
        : n_nodes_budget(n_graph_nodes) {
        const size_t bytes = ggml_tensor_overhead() * n_tensors +
                             ggml_graph_overhead_custom(n_graph_nodes, false) +
                             64 * 1024;
        meta.resize(bytes);
        ggml_init_params ip = {meta.size(), meta.data(), true};
        ctx = ggml_init(ip);
    }
    ~Scratch() {
        if (ctx) ggml_free(ctx);
    }
    ggml_cgraph *make_graph(size_t n_nodes = 0) {
        gf = ggml_new_graph_custom(ctx, n_nodes ? n_nodes : n_nodes_budget,
                                   false);
        return gf;
    }
};

// ---------------------------------------------------------------------------
// basic blocks
// ---------------------------------------------------------------------------

// RMSNorm over ne[0]; x {D, T} → {D, T}
inline ggml_tensor *rmsnorm(ggml_context *c, ggml_tensor *x, ggml_tensor *w,
                            float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

// LayerNorm with affine; x {D, T} → {D, T}
inline ggml_tensor *layernorm(ggml_context *c, ggml_tensor *x, ggml_tensor *w,
                              ggml_tensor *b, float eps) {
    ggml_tensor *y = ggml_norm(c, x, eps);
    y = ggml_mul(c, y, w);
    if (b) y = ggml_add(c, y, b);
    return y;
}

// bias broadcast: b {out} → {out, 1}
inline ggml_tensor *bias_2d(ggml_context *c, ggml_tensor *b) {
    return ggml_reshape_2d(c, b, b->ne[0], 1);
}

// linear: x {in, T} → {out, T}  (w ne=[in, out])
inline ggml_tensor *linear(ggml_context *c, ggml_tensor *w, ggml_tensor *x) {
    return ggml_mul_mat(c, w, x);
}

inline ggml_tensor *linear_bias(ggml_context *c, ggml_tensor *w,
                                ggml_tensor *b, ggml_tensor *x) {
    ggml_tensor *y = ggml_mul_mat(c, w, x);
    if (b) y = ggml_add(c, y, bias_2d(c, b));
    return y;
}

// ---------------------------------------------------------------------------
// manual rotate-half RoPE (supports the llama3 frequency scaling: the scaled
// inv_freq table is precomputed in the GGUF).
//   x: {hd, H, T} f32 → returns contiguous rotated {hd, H, T}
//   positions: i32 {T}
// ---------------------------------------------------------------------------
inline ggml_tensor *rope_neox(ggml_context *c, ggml_tensor *x,
                              ggml_tensor *inv_freq, ggml_tensor *positions) {
    const int64_t hd = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t T = x->ne[2];
    const int64_t half = hd / 2;

    ggml_tensor *pos = ggml_cast(c, positions, GGML_TYPE_F32);  // {T}
    pos = ggml_reshape_2d(c, pos, 1, T);
    // freqs[d, t] = pos[t] * inv_freq[d]  (ggml broadcast needs the repeated
    // operand first: {1,T} → {half,T}, then {half,1} divides cleanly)
    ggml_tensor *shape = ggml_new_tensor_2d(c, GGML_TYPE_F32, half, T);
    ggml_tensor *pos_rep = ggml_repeat(c, pos, shape);  // {half, T}
    ggml_tensor *freqs =
        ggml_mul(c, pos_rep, ggml_reshape_2d(c, inv_freq, half, 1));
    ggml_tensor *cos_h = ggml_cos(c, freqs);
    ggml_tensor *sin_h = ggml_sin(c, freqs);
    ggml_tensor *cos_full =
        ggml_reshape_3d(c, ggml_concat(c, cos_h, cos_h, 0), hd, 1, T);
    ggml_tensor *sin_full =
        ggml_reshape_3d(c, ggml_concat(c, sin_h, sin_h, 0), hd, 1, T);

    // rotate_half: concat(-x[half:], x[:half])
    const size_t nb0 = x->nb[0], nb1 = x->nb[1], nb2 = x->nb[2];
    ggml_tensor *x1 = ggml_view_3d(c, x, half, H, T, nb1, nb2, 0);
    ggml_tensor *x2 = ggml_view_3d(c, x, half, H, T, nb1, nb2,
                                   (size_t)half * nb0);
    ggml_tensor *rot =
        ggml_concat(c, ggml_neg(c, ggml_cont(c, x2)), ggml_cont(c, x1), 0);

    ggml_tensor *out = ggml_add(c, ggml_mul(c, x, cos_full),
                                ggml_mul(c, rot, sin_full));
    return ggml_cont(c, out);
}

// ---------------------------------------------------------------------------
// llama-style self attention with persistent KV cache (f16 caches laid out
// {head_dim, max_ctx, n_kv, n_layers}). cur: {D, T}; positions: i32 {T};
// mask: {Lk, T} f16 or nullptr (single-token step). gf is needed to wire the
// cache cpy ops into the graph.
// ---------------------------------------------------------------------------
struct AttnCfg {
    int n_head = 0, n_kv = 0, head_dim = 0;
    ggml_tensor *inv_freq = nullptr;
    ggml_tensor *kv_k = nullptr, *kv_v = nullptr;
    int il = 0;
    int n_past = 0;      // first cache write position
    ggml_tensor *mask = nullptr;
    ggml_tensor *positions = nullptr;
    float eps = 1e-5f;
};

inline ggml_tensor *llama_block(ggml_context *c, ggml_cgraph *gf,
                                const LlamaBlock &w, ggml_tensor *cur,
                                const AttnCfg &cfg) {
    const int hd = cfg.head_dim;
    const int H = cfg.n_head;
    const int Hkv = cfg.n_kv;
    const int T = (int)cur->ne[1];
    const int Lk = cfg.n_past + T;

    // --- self attention ---
    ggml_tensor *residual = cur;
    ggml_tensor *x = rmsnorm(c, cur, w.ln1_w, cfg.eps);
    fprintf(stderr, "blk: cur{%lld,%lld} q_w{%lld,%lld} k_w{%lld,%lld} x{%lld,%lld}\n",
            (long long)cur->ne[0], (long long)cur->ne[1],
            (long long)w.q_w->ne[0], (long long)w.q_w->ne[1],
            (long long)w.k_w->ne[0], (long long)w.k_w->ne[1],
            (long long)x->ne[0], (long long)x->ne[1]);
    ggml_tensor *q = ggml_reshape_3d(c, linear(c, w.q_w, x), hd, H, T);
    ggml_tensor *k = ggml_reshape_3d(c, linear(c, w.k_w, x), hd, Hkv, T);
    ggml_tensor *v = ggml_reshape_3d(c, linear(c, w.v_w, x), hd, Hkv, T);
    q = rope_neox(c, q, cfg.inv_freq, cfg.positions);
    k = rope_neox(c, k, cfg.inv_freq, cfg.positions);

    // append K/V into the persistent cache
    ggml_tensor *k_new = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    ggml_tensor *v_new = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));
    const size_t koff = (size_t)cfg.il * cfg.kv_k->nb[3] +
                        (size_t)cfg.n_past * cfg.kv_k->nb[1];
    ggml_tensor *k_slot = ggml_view_4d(c, cfg.kv_k, hd, T, Hkv, 1,
                                       cfg.kv_k->nb[1], cfg.kv_k->nb[2],
                                       cfg.kv_k->nb[3], koff);
    ggml_tensor *v_slot = ggml_view_4d(c, cfg.kv_v, hd, T, Hkv, 1,
                                       cfg.kv_v->nb[1], cfg.kv_v->nb[2],
                                       cfg.kv_v->nb[3], koff);
    ggml_build_forward_expand(gf, ggml_cpy(c, k_new, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(c, v_new, v_slot));

    ggml_tensor *k_full = ggml_view_3d(c, cfg.kv_k, hd, Lk, Hkv,
                                       cfg.kv_k->nb[1], cfg.kv_k->nb[2],
                                       (size_t)cfg.il * cfg.kv_k->nb[3]);
    ggml_tensor *v_full = ggml_view_3d(c, cfg.kv_v, hd, Lk, Hkv,
                                       cfg.kv_v->nb[1], cfg.kv_v->nb[2],
                                       (size_t)cfg.il * cfg.kv_v->nb[3]);

    ggml_tensor *q_attn =
        ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));  // {hd, T, H}
    ggml_tensor *attn = ggml_flash_attn_ext(c, q_attn, k_full, v_full, cfg.mask,
                                            1.0f / std::sqrt((float)hd), 0.0f,
                                            0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(c, attn, hd * H, T);

    ggml_tensor *proj = linear(c, w.o_w, attn);
    if (w.attn_gamma) proj = ggml_mul(c, proj, w.attn_gamma);
    cur = ggml_add(c, residual, proj);

    // --- MLP (SwiGLU) ---
    residual = cur;
    x = rmsnorm(c, cur, w.ln2_w, cfg.eps);
    ggml_tensor *gate = linear(c, w.gate_w, x);
    ggml_tensor *up = linear(c, w.up_w, x);
    ggml_tensor *ff = ggml_mul(c, ggml_silu(c, gate), up);
    ff = linear(c, w.down_w, ff);
    if (w.mlp_gamma) ff = ggml_mul(c, ff, w.mlp_gamma);
    return ggml_add(c, residual, ff);
}

// ---------------------------------------------------------------------------
// channels-first causal conv (offline / full sequence)
//   x: {C, T} f32; w ne=[k, C, OC]; pad_left = span - s; pad_right makes the
//   output ceil(T/s) frames (matches the ONNX reference padding).
// ---------------------------------------------------------------------------
// causal Conv1d (channels-first): x {C, T} → pad time → transpose →
// ggml_conv_1d → transpose back → {OC, T_out}. Output matches the ONNX
// reference (CausalConvNet pad_left = span − stride, pad_right = stride-exact).
inline ggml_tensor *conv1d_full(ggml_context *c, const Conv1D &w,
                                ggml_tensor *x, int stride = 1,
                                int dilation = 1) {
    const int T = (int)x->ne[1];
    const int span = (w.k - 1) * dilation + 1;
    const int pad_l = span - stride;
    int pad_r = 0;
    if (stride > 1) {
        int frames = (T + stride - 1) / stride;
        pad_r = frames * stride - T;
    }
    ggml_tensor *xp = ggml_pad_ext(c, x, 0, 0, pad_l, pad_r, 0, 0, 0, 0);
    ggml_tensor *xt = ggml_cont(c, ggml_transpose(c, xp));  // {T', C}
    ggml_tensor *wf = (w.w->type == GGML_TYPE_F16)
                          ? w.w
                          : ggml_cast(c, w.w, GGML_TYPE_F16);
    ggml_tensor *y = ggml_conv_1d(c, wf, xt, stride, 0, dilation);
    y = ggml_reshape_2d(c, y, y->ne[0], y->ne[1]);
    y = ggml_cont(c, ggml_transpose(c, y));  // {OC, T_out}
    if (w.b) y = ggml_add(c, y, bias_2d(c, w.b));
    return y;
}

// streaming causal conv (stride 1): conv over [state {C,S} ‖ x {C,T}] with
// no left padding (the state provides the history) → {OC, T};
// *new_state = last S samples of the concat.
inline ggml_tensor *conv1d_streaming(ggml_context *c, ggml_cgraph *gf,
                                     const Conv1D &w, ggml_tensor *x,
                                     ggml_tensor *state,
                                     ggml_tensor **new_state) {
    const int T = (int)x->ne[1];
    const int S = (int)state->ne[1];
    ggml_tensor *cat = ggml_concat(c, state, x, 1);  // {C, S+T}
    // update state: last S samples
    ggml_tensor *st = ggml_view_2d(c, cat, cat->ne[0], S, cat->nb[1],
                                   (size_t)T * cat->nb[1]);
    *new_state = ggml_cpy(c, st, state);
    ggml_build_forward_expand(gf, *new_state);

    ggml_tensor *xt = ggml_cont(c, ggml_transpose(c, cat));
    ggml_tensor *wf = (w.w->type == GGML_TYPE_F16)
                          ? w.w
                          : ggml_cast(c, w.w, GGML_TYPE_F16);
    ggml_tensor *y;
    if (w.dw) {
        y = ggml_conv_1d_dw(c, wf, xt, 1, 0, w.dilation);
    } else {
        y = ggml_conv_1d(c, wf, xt, 1, 0, w.dilation);
    }
    y = ggml_reshape_2d(c, y, y->ne[0], y->ne[1]);
    y = ggml_cont(c, ggml_transpose(c, y));
    if (w.b) y = ggml_add(c, y, bias_2d(c, w.b));
    return y;
}

// channels-first ConvTranspose1d, no padding: x {C, T} → {OC, (T-1)*s + k}
inline ggml_tensor *conv_transpose_1d(ggml_context *c, const Conv1D &w,
                                      ggml_tensor *x, int stride) {
    ggml_tensor *xt = ggml_cont(c, ggml_transpose(c, x));  // {T, C}
    ggml_tensor *wf = (w.w->type == GGML_TYPE_F16)
                          ? w.w
                          : ggml_cast(c, w.w, GGML_TYPE_F16);
    ggml_tensor *y = ggml_conv_transpose_1d(c, wf, xt, stride, 0, 1);
    y = ggml_reshape_2d(c, y, y->ne[0], y->ne[1]);
    y = ggml_cont(c, ggml_transpose(c, y));  // {OC, T_unpad}
    if (w.b) y = ggml_add(c, y, bias_2d(c, w.b));
    return y;
}

// SnakeBeta: y = x + sin(x·α)² / (exp(β) + 1e-9), α/β params {C}.
// eps_t is a persistent scalar tensor (1e-9) owned by the model.
inline ggml_tensor *snake_beta(ggml_context *c, ggml_tensor *x,
                               const SnakeAct &act, ggml_tensor *eps_t) {
    ggml_tensor *alpha = ggml_exp(c, act.alpha);  // {C}
    ggml_tensor *beta = ggml_exp(c, act.beta);
    ggml_tensor *alpha_b = ggml_reshape_2d(c, alpha, alpha->ne[0], 1);
    ggml_tensor *beta_b = ggml_reshape_2d(c, beta, beta->ne[0], 1);
    ggml_tensor *s = ggml_sin(c, ggml_mul(c, x, alpha_b));
    s = ggml_mul(c, s, s);
    return ggml_add(c, x, ggml_div(c, s, ggml_add(c, beta_b, eps_t)));
}

inline ggml_tensor *elu(ggml_context *c, ggml_tensor *x) {
    return ggml_elu(c, x);
}

// build a f16 attention mask {Lk, T} on the host: 0 allowed, -inf masked.
// allowed[q, k] = k <= q_pos [causal] && q_pos - k < window [sliding]
inline std::vector<ggml_fp16_t> build_mask(int Lk, int T, int n_past,
                                           bool causal, int window = 0) {
    std::vector<ggml_fp16_t> mask((size_t)Lk * T);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < T; q++) {
        const int q_pos = n_past + q;
        for (int k = 0; k < Lk; k++) {
            bool ok = true;
            if (causal && k > q_pos) ok = false;
            if (ok && window > 0 && q_pos - k >= window) ok = false;
            mask[(size_t)q * Lk + k] = ok ? zero : ninf;
        }
    }
    return mask;
}

}  // namespace graph
}  // namespace m5tts
