// M5-TTS codec graphs: Qwen3-TTS 12Hz encoder (SEANet + Mimi transformer +
// RVQ argmin quantization) and stateful streaming decoder (RVQ lookup →
// causal conv → 8L sliding-window transformer → conv upsampler).
// Mirrors onnx/graphs.py (CodecEncoder) + codec_decoder_cached.onnx
// (conv/tail streaming states, transconv add-carry with bias compensation).

#include "arch/m5tts_graph.h"

#include "utils/rs_log.h"

#include <cstring>
#include <vector>

namespace m5tts {

namespace {

// Mimi encoder transformer block: LayerNorm (bias), no-bias qkvo, GELU-erf
// MLP (fc1/fc2), LayerScale gammas, sliding-window causal attention.
ggml_tensor *mimi_block(ggml_context *c, ggml_cgraph *gf, const MimiBlock &w,
                        ggml_tensor *cur, ggml_tensor *inv_freq,
                        ggml_tensor *positions, ggml_tensor *mask, int n_head,
                        int n_kv, int head_dim, float eps) {
    const int T = (int)cur->ne[1];
    const int hd = head_dim;
    const int H = n_head;
    const int Hkv = n_kv;

    ggml_tensor *residual = cur;
    ggml_tensor *x = graph::layernorm(c, cur, w.ln1_w, w.ln1_b, eps);
    ggml_tensor *q = ggml_reshape_3d(c, graph::linear(c, w.q_w, x), hd, H, T);
    ggml_tensor *k = ggml_reshape_3d(c, graph::linear(c, w.k_w, x), hd, Hkv, T);
    ggml_tensor *v = ggml_reshape_3d(c, graph::linear(c, w.v_w, x), hd, Hkv, T);
    q = graph::rope_neox(c, q, inv_freq, positions);
    k = graph::rope_neox(c, k, inv_freq, positions);

    ggml_tensor *q_attn = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    ggml_tensor *k_attn = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    ggml_tensor *v_attn = ggml_cont(c, ggml_permute(c, v, 0, 2, 1, 3));
    ggml_tensor *attn = ggml_flash_attn_ext(
        c, q_attn, k_attn, v_attn, mask, 1.0f / std::sqrt((float)hd), 0.0f,
        0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(c, attn, hd * H, T);

    ggml_tensor *proj = graph::linear(c, w.o_w, attn);
    if (w.attn_gamma) proj = ggml_mul(c, proj, w.attn_gamma);
    cur = ggml_add(c, residual, proj);

    residual = cur;
    x = graph::layernorm(c, cur, w.ln2_w, w.ln2_b, eps);
    ggml_tensor *h = graph::linear(c, w.fc1_w, x);
    h = ggml_gelu_erf(c, h);
    h = graph::linear(c, w.fc2_w, h);
    if (w.mlp_gamma) h = ggml_mul(c, h, w.mlp_gamma);
    return ggml_add(c, residual, h);
}

// conv k1 as a matmul (weight ne=[1, in, out] → {in, out})
ggml_tensor *conv1x1(ggml_context *c, const Conv1D &w, ggml_tensor *x) {
    ggml_tensor *w2 = ggml_reshape_2d(c, w.w, w.w->ne[1], w.w->ne[2]);
    ggml_tensor *y = graph::linear(c, w2, x);
    if (w.b) y = ggml_add(c, y, graph::bias_2d(c, w.b));
    return y;
}

}  // namespace

// ---------------------------------------------------------------------------
// Codec encoder: pcm {n} f32 → codes {T, 16} i32 (interleaved [t][k]).
// ---------------------------------------------------------------------------
bool Model::EncodeReference(ggml_backend_sched_t sched, const float *pcm,
                            int n, std::vector<int32_t> &codes, int &T_out) {
    graph::Scratch sc(8192, 32768);
    ggml_context *c = sc.ctx;
    ggml_cgraph *gf = sc.make_graph(32768);

    ggml_tensor *audio = ggml_new_tensor_1d(c, GGML_TYPE_F32, n);
    ggml_set_name(audio, "audio");
    ggml_set_input(audio);
    ggml_tensor *x = ggml_reshape_2d(c, audio, 1, n);  // {1, n} channels-first
    {
        // keep the padded+transposed conv input + stem weight in the graph
        ggml_tensor *xtp = ggml_cont(c, ggml_transpose(c, ggml_pad_ext(
            c, x, 0, 0, 6, 0, 0, 0, 0, 0)));
        ggml_tensor *acpy = ggml_cpy(c, xtp, ggml_new_tensor_2d(c, GGML_TYPE_F32, n + 6, 1));
        ggml_set_name(acpy, "enc_padded_in");
        ggml_build_forward_expand(gf, acpy);
        ggml_tensor *wcpy = ggml_cpy(c, cew_.stem.w,
                                     ggml_new_tensor_3d(c, GGML_TYPE_F32, 7, 1, 64));
        ggml_set_name(wcpy, "enc_stem_w");
        ggml_build_forward_expand(gf, wcpy);
        {
            // dump the RAW gguf weight tensor (first 448 floats)
            ggml_tensor *wraw = cew_.stem.w;
            ggml_tensor *wraw_c = ggml_cont(c, wraw);
            ggml_set_name(wraw_c, "enc_stem_w_raw");
            ggml_build_forward_expand(gf, wraw_c);
        }
    }

    // SEANet stem
    x = graph::conv1d_full(c, cew_.stem, x, cew_.stem.stride, 1);
    ggml_set_name(x, "enc_stem");
    if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(x);

    // downsample stages
    for (size_t si = 0; si < cew_.stages.size(); si++) {
        const auto &st = cew_.stages[si];
        ggml_tensor *h = graph::conv1d_full(c, st.res1, graph::elu(c, x),
                                            st.res1.stride, 1);
        if (si == 0) ggml_set_name(h, "enc_s0_res1");
        h = graph::elu(c, h);
        h = graph::conv1d_full(c, st.res2, h, st.res2.stride, 1);
        if (si == 0) ggml_set_name(h, "enc_s0_res2");
        ggml_tensor *xadd = ggml_add(c, x, h);
        if (si == 0) { ggml_set_name(xadd, "enc_s0_add"); if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(xadd); }
        x = graph::elu(c, xadd);
        if (si == 0) { ggml_set_name(x, "enc_s0_elu_add"); if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(x); }
        x = graph::conv1d_full(c, st.down, x, st.down.stride, 1);
        ggml_set_name(x, ("enc_stage" + std::to_string(si)).c_str());
        if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(x);

    }
    x = graph::elu(c, x);
    x = graph::conv1d_full(c, cew_.final_conv, x, cew_.final_conv.stride, 1);
    ggml_set_name(x, "enc_final");
    if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(x);


    // transformer (offline, causal + sliding window)
    const int Ttf = (int)x->ne[1];
    ggml_tensor *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, Ttf);
    ggml_set_name(pos, "positions");
    ggml_set_input(pos);
    std::vector<ggml_fp16_t> mask =
        graph::build_mask(Ttf, Ttf, 0, true, hp_.ce_window);
    ggml_tensor *mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, Ttf, Ttf);
    ggml_set_name(mask_in, "enc_mask");
    ggml_set_input(mask_in);
    for (int il = 0; il < hp_.ce_layers; il++) {
        x = mimi_block(c, gf, cew_.tf_layers[il], x, ce_inv_freq_, pos, mask_in,
                       hp_.ce_heads, hp_.ce_kv_heads, hp_.ce_head_dim,
                       hp_.ce_eps);
        }

    // downsample (compress) → quantizer
    ggml_set_name(x, "enc_tf_out");
    if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(x);
    x = graph::conv1d_full(c, cew_.downsample, x, cew_.downsample.stride, 1);
    ggml_set_name(x, "enc_down_out");
    if (std::getenv("RS_M5TTS_ENC_DEBUG")) ggml_set_output(x);
    const int Tc = (int)x->ne[1];

    // quantizer encode
    ggml_tensor *xs = conv1x1(c, Conv1D{cew_.sem_in_proj_w, nullptr}, x);
    ggml_tensor *ra = xs;  // {256, Tc}

    ggml_tensor *codes_out = nullptr;
    auto quantize_layer = [&](ggml_tensor *inp, std::vector<ggml_tensor *> stack) {
        // per layer: argmin over ||E_v - r_t||² ; r -= E[codes]
        for (ggml_tensor *E : stack) {
            const int64_t V = E->ne[1];
                    ggml_tensor *sim = graph::linear(c, E, inp);  // {V, Tc}
            ggml_tensor *e2 = ggml_sum_rows(c, ggml_mul(c, E, E));  // {1, V}
                    e2 = ggml_reshape_2d(c, e2, V, 1);
                    ggml_tensor *r2 = ggml_sum_rows(c, ggml_mul(c, inp, inp));  // {1,Tc}
            // dist = 2·sim − E² − r²  (broadcast-legal chain against {V,Tc})
            ggml_tensor *t1 = ggml_add(c, sim, sim);
            ggml_tensor *dist =
                ggml_sub(c, ggml_sub(c, t1, e2), r2);
                    ggml_tensor *idx = ggml_argmax(c, dist);  // dist = −true dist; argmax → argmin
            idx = ggml_reshape_1d(c, idx, Tc);
                    ggml_tensor *q = ggml_get_rows(c, E, idx);  // {256, Tc}
            inp = ggml_sub(c, inp, q);
            ggml_tensor *row = ggml_reshape_2d(c, idx, 1, Tc);
            codes_out = (codes_out == nullptr)
                            ? row
                            : ggml_concat(c, codes_out, row, 0);
        }
    };
    quantize_layer(ra, {cew_.sem_embed});
    // acoustic RVQ re-encodes the ORIGINAL x (not the semantic residual);
    // its input_proj maps 512 → 256, residual subtracted per layer in 256-space
    ra = conv1x1(c, Conv1D{cew_.acou_in_proj_w, nullptr}, x);
    if (std::getenv("RS_M5TTS_ENC_DEBUG")) {
        ggml_tensor *racpy = ggml_cpy(c, ra, ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, Tc));
        ggml_set_name(racpy, "enc_acou_in");
        ggml_set_output(racpy);
        ggml_build_forward_expand(gf, racpy);
        ggml_tensor *semsub = ggml_cpy(c, xs, ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, Tc));
        ggml_set_name(semsub, "enc_sem_in");
        ggml_set_output(semsub);
        ggml_build_forward_expand(gf, semsub);
    }
    quantize_layer(ra, cew_.acou_embed);

    ggml_set_output(codes_out);
    ggml_build_forward_expand(gf, codes_out);

    if (!AllocGraph(sched, gf, "codec encode")) return false;
    ggml_backend_tensor_set(audio, pcm, 0, (size_t)n * sizeof(float));
    {
        std::vector<int32_t> p((size_t)Ttf);
        for (int i = 0; i < Ttf; i++) p[i] = i;
        SetInputI32(gf, "positions", p.data(), Ttf);
        ggml_tensor *mt = ggml_graph_get_tensor(gf, "enc_mask");
        if (mt) ggml_backend_tensor_set(mt, mask.data(), 0,
                                        mask.size() * sizeof(ggml_fp16_t));
    }
    if (!ComputeGraph(sched, gf, "codec encode")) return false;

    // optional numeric dumps (align with ONNX taps)
    if (const char *dp = std::getenv("RS_M5TTS_ENC_DEBUG")) {
        for (const char *nm : {"enc_stem", "enc_s0_res1", "enc_s0_res2", "enc_padded_in", "enc_stem_w", "enc_stem_w_raw", "enc_tf_out", "enc_down_out", "enc_s0_add", "enc_s0_elu_add", "enc_acou_in", "enc_sem_in"}) {
            ggml_tensor *t = ggml_graph_get_tensor(gf, nm);
            if (!t) { fprintf(stderr, "ENC DBG: tensor %s NOT FOUND\n", nm); continue; }
            size_t total = 1;
            for (int d = 0; d < 4; d++) total *= (size_t)t->ne[d];
            std::vector<float> buf(total);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            FILE *f = fopen((std::string(dp) + std::string(nm).erase(0,4) + ".f32").c_str(), "wb");
            if (f) { fwrite(buf.data(), 4, buf.size(), f); fclose(f); }
        }
        for (int i = 0; i < (int)cew_.stages.size(); i++) {
            ggml_tensor *t = ggml_graph_get_tensor(gf, ("enc_stage" + std::to_string(i)).c_str());
            if (!t) continue;
            std::vector<float> buf((size_t)t->ne[0] * (size_t)t->ne[1]);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            FILE *f = fopen((std::string(dp) + "_stage" + std::to_string(i) + ".f32").c_str(), "wb");
            if (f) { fwrite(buf.data(), 4, buf.size(), f); fclose(f); }
        }
        {
            // dump the downsample conv's im2col (conv_idx 14)
            ggml_tensor *t = ggml_graph_get_tensor(gf, "im2col_14");
            if (!t) fprintf(stderr, "ENC DBG: im2col_14 NOT FOUND\n");
            if (t) {
                size_t total = (size_t)t->ne[0] * t->ne[1];
                std::vector<uint8_t> raw(ggml_nbytes(t));
                ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
                FILE *f = fopen("/tmp/m5_im2col_ds.f32", "wb");
                if (f) {
                    // F32 im2col in this path
                    fwrite(raw.data(), 1, raw.size(), f);
                    fclose(f);
                }
            }
        }
        ggml_tensor *t = ggml_graph_get_tensor(gf, "enc_final");
        if (t) {
            std::vector<float> buf((size_t)t->ne[0] * (size_t)t->ne[1]);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            FILE *f = fopen((std::string(dp) + "_final.f32").c_str(), "wb");
            if (f) { fwrite(buf.data(), 4, buf.size(), f); fclose(f); }
        }
    }

    std::vector<int32_t> raw((size_t)16 * Tc);
    ggml_backend_tensor_get(codes_out, raw.data(), 0,
                            (size_t)16 * Tc * sizeof(int32_t));
    T_out = (int)((n + hp_.hop_length - 1) / hp_.hop_length);
    if (T_out > Tc) T_out = Tc;
    codes.resize((size_t)T_out * 16);
    // codes_out rows are [k][t] (concat along dim0 of {1, Tc} rows)
    for (int k = 0; k < 16; k++)
        for (int t = 0; t < T_out; t++)
            codes[(size_t)t * 16 + k] = raw[(size_t)k * Tc + t];
    return true;
}

// ---------------------------------------------------------------------------
// Codec decoder: chunked streaming decode. codes: {T, 16} interleaved.
// ---------------------------------------------------------------------------
bool Model::DecodeCodes(ggml_backend_sched_t sched, const int32_t *codes,
                        int T, std::vector<float> &pcm_out) {
    const int chunk = 32;
    pcm_out.clear();
    pcm_out.reserve((size_t)T * hp_.hop_length);
    for (int start = 0; start < T; start += chunk) {
        const int n = std::min(chunk, T - start);
        std::vector<float> pcm_chunk;
        if (!DecodeChunk(sched, codes + (size_t)start * 16, n, pcm_chunk))
            return false;
        pcm_out.insert(pcm_out.end(), pcm_chunk.begin(), pcm_chunk.end());
    }
    return true;
}

bool Model::DecodeChunk(ggml_backend_sched_t sched, const int32_t *codes,
                        int n, std::vector<float> &pcm_out) {
    const int L = hp_.cd_latent;

    graph::Scratch sc(4096, 32768);
    ggml_context *c = sc.ctx;
    ggml_cgraph *gf = sc.make_graph(32768);

    // 16 per-codebook 1-D index tensors (get_rows requires 1-D indices)
    ggml_tensor *codes_in[16];
    for (int k = 0; k < 16; k++) {
        codes_in[k] = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
        ggml_set_name(codes_in[k], ("codes_" + std::to_string(k)).c_str());
        ggml_set_input(codes_in[k]);
    }

    // --- RVQ lookup: hidden = sem_out(emb[c0]) + acou_out(Σ_i emb[ci]) ---
    ggml_tensor *sem_e = ggml_get_rows(c, cdw_.sem_embed, codes_in[0]);  // {256, n}
    ggml_tensor *hidden =
        graph::linear(c, ggml_reshape_2d(c, cdw_.sem_out_proj_w,
                                         cdw_.sem_out_proj_w->ne[1],
                                         cdw_.sem_out_proj_w->ne[2]),
                      sem_e);  // {512, n}
    ggml_tensor *acc = nullptr;
    for (int i = 1; i < hp_.codebooks; i++) {
        ggml_tensor *e =
            ggml_get_rows(c, cdw_.acou_embed[i - 1], codes_in[i]);  // {256, n}
        acc = (acc == nullptr) ? e : ggml_add(c, acc, e);
    }
    ggml_tensor *acou_q =
        graph::linear(c, ggml_reshape_2d(c, cdw_.acou_out_proj_w,
                                         cdw_.acou_out_proj_w->ne[1],
                                         cdw_.acou_out_proj_w->ne[2]),
                      acc);
    hidden = ggml_add(c, hidden, acou_q);  // {512, n}

    // --- pre_conv (streaming, state 2) ---
    ggml_tensor *st0 = nullptr;
    hidden = graph::conv1d_streaming(c, gf, cdw_.pre_conv, hidden,
                                     conv_states_[0].t, &st0);

    // --- transformer ---
    // hidden {512, n} → input_proj {512→... wait input_proj maps latent→hidden
    ggml_tensor *tf_in = graph::linear_bias(c, cdw_.tf_input_proj_w,
                                            cdw_.tf_input_proj_b, hidden);
    ggml_tensor *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(pos, "positions");
    ggml_set_input(pos);
    const int Lk = cd_offset_ + n;
    std::vector<ggml_fp16_t> mask =
        graph::build_mask(Lk, n, cd_offset_, true, hp_.cd_window);
    ggml_tensor *mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, Lk, n);
    ggml_set_name(mask_in, "dec_mask");
    ggml_set_input(mask_in);
    for (int il = 0; il < hp_.cd_layers; il++) {
        graph::AttnCfg cfg;
        cfg.n_head = hp_.cd_heads;
        cfg.n_kv = hp_.cd_kv_heads;
        cfg.head_dim = hp_.cd_head_dim;
        cfg.inv_freq = cd_inv_freq_;
        cfg.kv_k = cd_kv_k_;
        cfg.kv_v = cd_kv_v_;
        cfg.il = il;
        cfg.n_past = cd_offset_;
        cfg.mask = mask_in;
        cfg.positions = pos;
        cfg.eps = hp_.cd_eps;
        tf_in = graph::llama_block(c, gf, cdw_.tf_layers[il], tf_in, cfg);
    }
    tf_in = graph::rmsnorm(c, tf_in, cdw_.tf_norm_w, hp_.cd_eps);
    ggml_tensor *tf_out = graph::linear_bias(c, cdw_.tf_output_proj_w,
                                             cdw_.tf_output_proj_b, tf_in);
    // {1024, n} feature-major → channels-first (same ne, just semantics)
    ggml_tensor *x = ggml_cont(c, tf_out);  // {L, n}

    // --- upsample ×2: transconv (k=s=2) + ConvNeXt ---
    int si = 1;  // conv state index (pre_conv=0, up0.dwconv=1, up1.dwconv=2)
    for (int u = 0; u < (int)hp_.upsampling_ratios.size(); u++) {
        const auto &up = cdw_.upstage[u];
        ggml_tensor *y =
            graph::conv_transpose_1d(c, up.transconv, x, up.transconv.stride);
        ggml_tensor *st = nullptr;
        ggml_tensor *h = graph::conv1d_streaming(c, gf, up.dwconv, y,
                                                 conv_states_[si].t, &st);
        si++;
        h = graph::layernorm(c, h, up.norm_w, up.norm_b, 1e-6f);
        h = graph::linear_bias(c, up.pw1_w, up.pw1_b, h);
        h = ggml_gelu_erf(c, h);
        h = graph::linear_bias(c, up.pw2_w, up.pw2_b, h);
        h = ggml_mul(c, h, up.gamma);
        x = ggml_add(c, y, h);
    }

    // --- decoder conv chain ---
    ggml_tensor *st = nullptr;
    x = graph::conv1d_streaming(c, gf, cdw_.decoder_in, x,
                                conv_states_[si].t, &st);
    si++;

    for (int b = 0; b < (int)hp_.upsample_rates.size(); b++) {
        const auto &blk = cdw_.blocks[b];
        const int rate = hp_.upsample_rates[b];
        ggml_tensor *act = graph::snake_beta(c, x, blk.act0, sc_eps9_);
        ggml_tensor *y = graph::conv_transpose_1d(c, blk.transconv, act,
                                                  blk.transconv.stride);
        const int64_t T_unpad = y->ne[1];
        const int64_t ch = y->ne[0];
        const int64_t T_in = (T_unpad - rate) / blk.transconv.stride;  // frames

        // tail state: y[T_in*s : T_in*s + rate] - bias  (bias-free carry)
        ggml_tensor *prev_tail = ggml_cont(c, conv_states_[si].t);
        ggml_tensor *tail = ggml_view_2d(
            c, y, ch, rate, y->nb[1],
            (size_t)T_in * blk.transconv.stride * y->nb[1]);
        ggml_tensor *tail_nb =
            ggml_sub(c, tail, graph::bias_2d(c, blk.transconv.b));
        ggml_tensor *st_tail = ggml_cpy(c, tail_nb, conv_states_[si].t);
        si++;

        // emission = [prev_tail + y[0:rate]] ‖ y[rate : T_in*s]
        ggml_tensor *overlap = ggml_view_2d(c, y, ch, rate, y->nb[1], 0);
        ggml_tensor *emission = ggml_concat(
            c,
            ggml_add(c, prev_tail, overlap),
            ggml_view_2d(c, y, ch, T_in * blk.transconv.stride - rate,
                         y->nb[1], (size_t)rate * y->nb[1]),
            1);
        ggml_build_forward_expand(gf, st_tail);

        // residual units (dilations 1, 3, 9)
        static const int dils[3] = {1, 3, 9};
        for (int ru = 0; ru < 3; ru++) {
            ggml_tensor *a1 = graph::snake_beta(c, emission, blk.ru[ru].act1, sc_eps9_);
            ggml_tensor *st_ru = nullptr;
            ggml_tensor *cv = graph::conv1d_streaming(
                c, gf, blk.ru[ru].conv1, a1, conv_states_[si].t, &st_ru);
            si++;
            ggml_tensor *a2 = graph::snake_beta(c, cv, blk.ru[ru].act2, sc_eps9_);
            ggml_tensor *c2 = conv1x1(c, blk.ru[ru].conv2, a2);
            emission = ggml_add(c, emission, c2);
        }
        x = emission;
    }

    x = graph::snake_beta(c, x, cdw_.act_final, sc_eps9_);
    {
        ggml_tensor *st_f = nullptr;
        x = graph::conv1d_streaming(c, gf, cdw_.decoder_out, x,
                                    conv_states_[si].t, &st_f);
    }
    ggml_tensor *audio = ggml_clamp(c, x, -1.0f, 1.0f);  // {1, 1920n}
    ggml_tensor *audio_out = ggml_reshape_1d(c, audio, (int64_t)n * hp_.hop_length);
    ggml_set_output(audio_out);
    ggml_build_forward_expand(gf, audio_out);

    if (!AllocGraph(sched, gf, "codec decode")) return false;
    for (int k = 0; k < 16; k++) {
        std::vector<int32_t> ck((size_t)n);
        for (int t = 0; t < n; t++) ck[(size_t)t] = codes[(size_t)t * 16 + k];
        SetInputI32(gf, ("codes_" + std::to_string(k)).c_str(), ck.data(), n);
    }
    std::vector<int32_t> pos_ids((size_t)n);
    for (int i = 0; i < n; i++) pos_ids[i] = cd_offset_ + i;
    SetInputI32(gf, "positions", pos_ids.data(), n);
    {
        ggml_tensor *mt = ggml_graph_get_tensor(gf, "dec_mask");
        if (mt) ggml_backend_tensor_set(mt, mask.data(), 0,
                                        mask.size() * sizeof(ggml_fp16_t));
    }
    if (!ComputeGraph(sched, gf, "codec decode")) return false;

    pcm_out.resize((size_t)n * hp_.hop_length);
    ggml_backend_tensor_get(audio_out, pcm_out.data(), 0,
                            pcm_out.size() * sizeof(float));
    cd_offset_ += n;
    return true;
}

}  // namespace m5tts
