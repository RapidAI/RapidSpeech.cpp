// kokoro.cpp — merged TU: ISpeechModel adapter + CrispASR-port engine
//
// This file is the result of mechanically concatenating:
//   * kokoro_core_activation.h  (namespace core_act)
//   * kokoro_core_align.h       (namespace core_align)
//   * kokoro_core_conv.h        (namespace core_convt)
//   * kokoro_core_lstm.h        (namespace core_lstm)
//   * kokoro_core_attention.h   (namespace core_attn)
//   * kokoro_core_gguf_loader.cpp (namespace core_gguf)
//   * kokoro_engine.h           (structs)
//   * kokoro_engine.cpp         (engine body, fns demoted to static)
//   * kokoro.cpp                (ISpeechModel adapter)
//
// All external linkage (extern "C") has been demoted to static; engine
// symbols are TU-local and reached only through the KokoroModel adapter.

#include "arch/kokoro.h"
#include "core/rs_context.h"
#include "frontend/kokoro_g2p_zh.h"
#include "frontend/kokoro_g2p_en.h"
#include "frontend/wetext_normalizer.h"
#include "utils/rs_log.h"
#include "utils/rs_data_paths.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <cerrno>
#include <cstdarg>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <cfenv>
#include <cmath>
#include <random>
#include <unordered_map>
#include <memory>
#include <functional>

#ifdef RS_USE_METAL_KOKORO
#include "arch/kokoro_metal.h"
#endif


// ======================================================================
// BEGIN kokoro_core_activation.h
// ======================================================================

// src/core/activation.h — non-FFN activation primitives for ggml graphs.
//
// FFN-shaped activations (gelu, swiglu, geglu, silu_ffn, gelu_erf_ffn)
// live in `ffn.h` because they're combined with their up/down projections.
// This header collects the standalone activations that get applied
// inside per-block residual paths or vocoder generator stacks.
//
// Currently:
//   snake_alpha  — Snake-α activation, used by BigVGAN-family vocoders
//                  (Kokoro generator, SNAC decoder, future iSTFTNet /
//                  ConvNeXt-vocoder ports). Per-channel learnable α, init=1.
//   snake_beta   — SnakeBeta activation, used by the qwen3-tts codec
//                  decoder (BigVGAN-style with separate α frequency and
//                  β amplitude scales).



namespace core_act {

// Snake-α activation: y = x + (1 / α) · sin²(α · x).
//
// Per-channel learnable scalar α with shape ne = (1, C, 1) F16 in the
// GGUF (matches PyTorch's `nn.Parameter(torch.ones(1, C, 1))`). We
// reshape to (C, 1) F32 so it broadcasts over the time axis of x.
//
// Input  x:     (C, T)     F32, channel-major.
// Input  alpha: (1, C, 1)  F16, the per-channel learnable scale.
// Output:       (C, T)     F32, same shape as x.
//
// α is initialised to 1 in training and stays bounded away from zero
// in practice, so the 1/α has no special-casing.
static inline ggml_tensor* snake_alpha(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    const int C = (int)x->ne[0];
    ggml_tensor* a = ggml_reshape_2d(ctx, alpha, C, 1);
    a = ggml_cast(ctx, a, GGML_TYPE_F32);  // (C, 1) F32 (Metal needs F32×F32)
    ggml_tensor* ax = ggml_mul(ctx, x, a); // α·x with α broadcast on T
    ggml_tensor* sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ax));
    ggml_tensor* div = ggml_div(ctx, sin_sq, a); // sin²(α·x) / α
    return ggml_add(ctx, x, div);
}

// SnakeBeta activation: y = x + exp(-β) · sin²(x · exp(α)).
//
// BigVGAN's two-parameter variant — α controls the frequency, β the
// amplitude, both per-channel. Used by the qwen3-tts codec decoder
// (and likely by Chatterbox / VoxCPM2 when those land).
//
// Input  x:     (C, T)  F32, channel-major.
// Input  alpha: (C,)    F32, per-channel frequency log-scale.
// Input  beta:  (C,)    F32, per-channel amplitude log-scale.
// Output:       (C, T)  F32.
static inline ggml_tensor* snake_beta(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta) {
    ggml_tensor* ea = ggml_exp(ctx, alpha);                   // (C,)
    ggml_tensor* inv_eb = ggml_exp(ctx, ggml_neg(ctx, beta)); // (C,) = 1 / exp(β)
    ggml_tensor* xa = ggml_mul(ctx, x, ea);                   // (C, T)
    ggml_tensor* s = ggml_sin(ctx, xa);
    ggml_tensor* s2 = ggml_mul(ctx, s, s); // (C, T)
    ggml_tensor* scaled = ggml_mul(ctx, s2, inv_eb);
    return ggml_add(ctx, x, scaled);
}

} // namespace core_act

// ======================================================================
// BEGIN kokoro_core_align.h
// ======================================================================

// src/core/align.h — duration-based alignment helpers for TTS pipelines.
//
// Every TTS with explicit duration prediction (FastSpeech, VITS,
// StyleTTS, Kokoro, qwen3-tts talker) needs the same primitive: take a
// per-token feature matrix and a per-token integer duration vector,
// and return a per-frame feature matrix where each token's features
// are repeated `dur[i]` times. PyTorch reference:
//
//   indices = torch.repeat_interleave(torch.arange(L), durations)
//   en      = features.transpose(-1, -2) @ one_hot(indices, L)
//
// The matmul-with-one-hot formulation is what the reference dumpers
// emit, but it's a wasteful way to compute it on CPU (O(L · T_frames)
// memory for the alignment matrix) when the same answer comes from a
// straight memcpy loop.
//
// This helper does the memcpy version. Output buffer is malloc'd —
// caller frees with `free()`.



namespace core_align {

// Repeat-interleave a (D, L) channel-major feature matrix according to
// per-token integer durations. Returns a malloc'd (D, T_frames) F32
// buffer with T_frames = sum(durations); writes T_frames to
// *out_T_frames. Returns nullptr (and *out_T_frames=0) on empty input
// or allocation failure.
//
// Layout: features[i] is the i-th token's column at offset i·D in the
// flat buffer, matching ggml's ne=(D, L) storage. The output preserves
// the same channel-major layout.
static inline float* repeat_interleave(const float* features, int D, int L, const int* durations, int* out_T_frames) {
    int T = 0;
    for (int i = 0; i < L; i++)
        T += durations[i];
    if (out_T_frames)
        *out_T_frames = T;
    if (T <= 0)
        return nullptr;
    float* out = (float*)std::malloc((size_t)D * T * sizeof(float));
    if (!out)
        return nullptr;
    int j = 0;
    for (int i = 0; i < L; i++) {
        const int n = durations[i];
        const float* col = features + (size_t)i * D;
        for (int k = 0; k < n; k++) {
            std::memcpy(out + (size_t)j * D, col, (size_t)D * sizeof(float));
            j++;
        }
    }
    return out;
}

} // namespace core_align

// ======================================================================
// BEGIN kokoro_core_conv.h
// ======================================================================

// src/core/conv.h — convolution helpers that work around ggml limitations.
//
// ggml has no `groups` argument on `ggml_conv_1d` or
// `ggml_conv_transpose_1d`, so any depthwise / grouped conv has to be
// open-coded. This header collects the specific shapes that come up
// repeatedly across the BigVGAN-family vocoder ports (Kokoro, future
// iSTFTNet variants, possibly mimo codec).
//
// Currently:
//   convt1d_depthwise_2x_k3  — depthwise ConvTranspose1d with kernel=3,
//                              stride=2, padding=1, output_padding=1.
//                              Used for 2× upsamples in iSTFTNet-style
//                              vocoder pool layers.
//   convt1d_crop             — channels-first ConvTranspose1d wrapper
//                              that handles the (C,T) ↔ (T,C) transpose
//                              dance and lets the caller specify how
//                              many time samples to crop from each end
//                              (causal vs symmetric padding).



namespace core_convt {

// Depthwise ConvTranspose1d with parameters (k=3, s=2, p=1, op=1).
// Output length = 2 · T_in.
//
// PyTorch ConvTranspose1d emits `y[i] = sum input[j] · weight[k]` over
// (j, k) satisfying `j·stride + k − padding = i`. For our config:
//
//   y[c, 2t]   = w[c, 1] · x[c, t]                                  (j=t,   k=1)
//   y[c, 2t+1] = w[c, 2] · x[c, t] + w[c, 0] · x[c, t+1]            (j=t,k=2 + j=t+1,k=0)
//                                                                   (x[c, T]=0 boundary)
//
// **Critical**: `w[2]` and `w[0]` are NOT interchangeable in the odd
// case — getting the kernel ends swapped produces plausible-but-wrong
// audio that can survive informal QA. The Kokoro M11 diff harness
// caught exactly this bug (commit 448c1af); see LEARNINGS.md
// "Kokoro / StyleTTS2 lessons" Lesson 2.
//
// Inputs:
//   x        : (C, T)        F32, channel-major.
//   w_kernel : (K=3, 1, C)   F16, depthwise kernel (PyTorch
//              `nn.ConvTranspose1d(C, C, k=3, s=2, p=1, op=1, groups=C)`
//              stores weights as `(C, 1, K)` and the converter
//              transposes to `(K, 1, C)` for ggml).
//   w_bias   : (C,)          F32, optional per-channel bias (broadcast
//              over time). Pass nullptr to skip.
//
// Output: (C, 2·T) F32.
static inline ggml_tensor* convt1d_depthwise_2x_k3(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_kernel,
                                                   ggml_tensor* w_bias) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];

    // Permute kernel (K=3, 1, C) → (C, 3, 1), cast to F32 (F16 view + F32
    // mul fails on Metal at the kernel-dispatch level), reshape to
    // (C, 3), then take three column views w0/w1/w2.
    ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_kernel, 2, 0, 1, 3)); // (C, 3, 1) F16
    ggml_tensor* w_perm_f32 = ggml_cast(ctx, w_perm, GGML_TYPE_F32);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w_perm_f32, C, 3); // (C, 3) F32
    const size_t row_b = w_2d->nb[1];
    ggml_tensor* w0 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)0 * row_b);
    ggml_tensor* w1 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)1 * row_b);
    ggml_tensor* w2 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)2 * row_b);

    // x_shifted[c, t] = x[c, t+1] for t < T-1, 0 for t = T-1.
    // Take x[:, 1:] (C, T-1) and zero-pad on the right to (C, T).
    ggml_tensor* x_tail = ggml_view_2d(ctx, x, C, T - 1, x->nb[1], x->nb[1]);   // (C, T-1)
    x_tail = ggml_cont(ctx, x_tail);                                            // contiguous
    ggml_tensor* x_shifted = ggml_pad_ext(ctx, x_tail, 0, 0, 0, 1, 0, 0, 0, 0); // (C, T)

    // y_even (C, T) = w1 ⊙ x  (broadcast w1 over T)
    ggml_tensor* y_even = ggml_mul(ctx, x, w1);
    // y_odd (C, T) = w2 ⊙ x + w0 ⊙ x_shifted   (PyTorch ConvTranspose1d
    // kernel indexing — see derivation note above)
    ggml_tensor* y_odd = ggml_add(ctx, ggml_mul(ctx, x, w2), ggml_mul(ctx, x_shifted, w0));

    // Interleave: reshape both to (C, 1, T), concat dim=1 → (C, 2, T),
    // reshape to (C, 2T). Memory layout means consecutive time positions
    // alternate even/odd, which is the desired interleaving.
    ggml_tensor* even_3d = ggml_reshape_3d(ctx, y_even, C, 1, T);
    ggml_tensor* odd_3d = ggml_reshape_3d(ctx, y_odd, C, 1, T);
    ggml_tensor* stacked = ggml_concat(ctx, even_3d, odd_3d, /*dim=*/1);      // (C, 2, T)
    ggml_tensor* y = ggml_cont(ctx, ggml_reshape_2d(ctx, stacked, C, 2 * T)); // (C, 2T)

    if (w_bias)
        y = ggml_add(ctx, y, w_bias);
    return y;
}

// Channels-first ConvTranspose1d (groups=1) with caller-controlled
// time-axis cropping.
//
// ggml_conv_transpose_1d expects (T, Cin) input and emits T_unpad =
// (T_in - 1)·stride + K samples; it has no padding parameter. Most
// callers want a smaller T_out and crop the excess from the ends:
//
//   - **Causal upsamplers** (qwen3-tts codec) trim the right tail only:
//     `crop_left=0, crop_right=K-stride` so T_out = T_in · stride.
//   - **Symmetric-pad upsamplers** (SNAC, with k=2s, p=s/2) crop the
//     same amount from each end: `crop_left=crop_right=stride/2`,
//     giving T_out = T_in · stride.
//
// Inputs:
//   x         : (Cin, T_in)   F32, channel-major.
//   w         : (K, Cout, Cin) F16/F32, ggml weight layout (PyTorch
//               numpy `(Cin, Cout, K)` transposed by the converter).
//   b         : (Cout,)       F32 or nullptr.
//   stride    : positive integer.
//   crop_left : samples to crop from the start of the time axis (≥ 0).
//   crop_right: samples to crop from the end of the time axis (≥ 0).
//
// Output: (Cout, T_unpad - crop_left - crop_right) F32.
static inline ggml_tensor* convt1d_crop(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int stride,
                                        int crop_left, int crop_right) {
    const int Cout = (int)w->ne[1];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, x));          // (T_in, Cin)
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xT, stride, 0, 1); // (T_unpad, Cout, 1, 1)
    const int T_unpad = (int)y->ne[0];
    const int T_out = T_unpad - crop_left - crop_right;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout);
    if (crop_left > 0 || crop_right > 0) {
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_unpad * sizeof(float), (size_t)crop_left * sizeof(float));
        y = ggml_cont(ctx, y); // (T_out, Cout)
    }
    y = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

} // namespace core_convt

// Conv1d using a pre-flattened 2D weight `w_2d` with ggml shape [K*Cin, Cout].
// The convert script (convert_kokoro_to_gguf.py) emits all Conv1d weights in
// this layout so the quantizer's pick() can engage k-quants on ne[0]=K*Cin
// instead of demoting tiny K-axis weights to F16. Conv geometry (K, Cin) is
// known from the model config at graph-build time and passed explicitly.
//
// in    : (T_in, Cin) F32 — post-transpose, as ggml_conv_1d expects.
// w_2d  : ne=[K*Cin, Cout, 1, 1], any dtype the runtime can mul_mat (F16,
//         Q8_0, Q4_K, ...). Cout is read from w_2d->ne[1].
// K, Cin: kernel size and input channels.
// s,p,d : stride, padding, dilation along the time axis.
//
// Returns (T_out, Cout, 1) — identical layout to ggml_conv_1d.
static inline ggml_tensor* kokoro_conv_1d_2d(ggml_context* ctx, ggml_tensor* w_2d, ggml_tensor* in, int K, int Cin,
                                             int s, int p, int d) {
    // Shape-reference tensor for ggml_im2col — only its ne[0]=K and ne[1]=Cin
    // are read (data is never accessed during compute).
    ggml_tensor* shape_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, Cin, 1);
    ggml_tensor* col = ggml_im2col(ctx, shape_ref, in,
                                   /*s0*/ s, /*s1*/ 0, /*p0*/ p, /*p1*/ 0,
                                   /*d0*/ d, /*d1*/ 0, /*is_2D*/ false,
                                   GGML_TYPE_F16);                         // [K*Cin, Tout, N]
    const int64_t Tout = col->ne[1];
    const int64_t N    = col->ne[2];
    const int64_t Cout = w_2d->ne[1];
    ggml_tensor* col_2d = ggml_reshape_2d(ctx, col, col->ne[0], Tout * N); // [K*Cin, Tout*N]
    // mul_mat(w, col): keep the quantized weight in position A so the Metal
    // backend can find a kernel — its `mul_mm_<quant>_f16` family only
    // covers (A=quant, B=f16). The result has ne[0]=Cout, ne[1]=Tout*N
    // (i.e., memory layout is Cout-fastest within each Tout*N row), so we
    // view it as (Cout, Tout, N) and transpose the leading two dims back
    // to (Tout, Cout, N) to match ggml_conv_1d's output contract.
    ggml_tensor* out = ggml_mul_mat(ctx, w_2d, col_2d);                    // [Cout, Tout*N]
    out = ggml_reshape_3d(ctx, out, Cout, Tout, N);                       // (Cout, Tout, N)
    return ggml_cont(ctx, ggml_permute(ctx, out, 1, 0, 2, 3));            // (Tout, Cout, N)
}

// ======================================================================
// BEGIN kokoro_core_lstm.h
// ======================================================================

// src/core/lstm.h — bidirectional LSTM helpers for ggml graphs.
//
// PyTorch nn.LSTM forward (single layer, single direction):
//
//   pre_t = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh   ∈ R^{4H}
//   i, f, g, o = pre_t.split(H)                          (PyTorch gate order)
//   c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
//   h_t = sigmoid(o) * tanh(c_t)
//
// Storage convention used by every Kokoro / StyleTTS2 LSTM in the GGUF:
//
//   weight_ih_l0          ne = (input_size,  4 * hidden_size)   F16
//   weight_hh_l0          ne = (hidden_size, 4 * hidden_size)   F16
//   bias_ih_l0            ne = (4 * hidden_size,)               F32
//   bias_hh_l0            ne = (4 * hidden_size,)               F32
//   weight_ih_l0_reverse  same shape as weight_ih_l0 (bidir layer 0)
//   ... etc for hh + biases
//
// Both biases are added (PyTorch keeps them separate even though their
// sum is mathematically equivalent — we follow the same pattern so the
// graph is bit-equivalent to the reference dump).
//
// We optimise the per-step graph by hoisting the input projection out of
// the loop:  proj_x = (W_ih @ X) + b_ih, computed once for all T at
// once.  Per timestep we still need W_hh @ h_{t-1} (the recurrence is
// fundamentally sequential).  At t=0 the helper skips the W_hh matmul
// entirely (h_{-1} = 0), which matches PyTorch's default zero initial
// state and avoids needing an externally-zeroed buffer.
//
// Output assembly: each step's h_t is written into a column of a
// pre-allocated `output` tensor (shape (H, T)) via ggml_view_2d +
// ggml_cpy + ggml_build_forward_expand.  This is the same pattern used
// by core_attn::kv_self_attn for the persistent KV cache write — the
// scheduler's view-tracking sequences the cpys before any downstream
// read of the output.




namespace core_lstm {

// Single-direction LSTM forward over T timesteps.
//
//   ctx          per-graph ggml context (no_alloc=true)
//   gf           graph being built — cpy ops are appended via
//                ggml_build_forward_expand so the writes are guaranteed
//                to run before any downstream read of the returned tensor
//   X            input, ne = (input_size, T)  F32
//   W_ih         ne = (input_size,  4H)        F16/F32
//   W_hh         ne = (hidden_size, 4H)        F16/F32
//   b_ih, b_hh   ne = (4H,)                     F32
//   H            hidden_size
//   reverse      iterate t = T-1 .. 0 (use the *_reverse weights)
//
// Returns the LSTM output as a contiguous (H, T) F32 tensor.
static inline ggml_tensor* lstm_unidir(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* X, ggml_tensor* W_ih,
                                       ggml_tensor* W_hh, ggml_tensor* b_ih, ggml_tensor* b_hh, int H, bool reverse) {
    const int T = (int)X->ne[1];
    const int H4 = 4 * H;

    // Input projection over all T timesteps at once.
    // proj_x ne = (4H, T)
    ggml_tensor* proj_x = ggml_mul_mat(ctx, W_ih, X);
    proj_x = ggml_add(ctx, proj_x, b_ih);

    // Pre-allocated output container (H, T) F32.
    ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);

    ggml_tensor* h = nullptr;
    ggml_tensor* c = nullptr;

    const size_t row_stride_4h = proj_x->nb[1];
    const size_t row_stride_h = output->nb[1];
    const size_t f32_size = ggml_type_size(GGML_TYPE_F32);

    const int t0 = reverse ? T - 1 : 0;
    const int dt = reverse ? -1 : 1;
    for (int step = 0; step < T; step++) {
        const int t = t0 + step * dt;

        // proj_x slice for this timestep: (4H, 1) F32, contiguous.
        ggml_tensor* px = ggml_view_2d(ctx, proj_x, H4, 1, row_stride_4h, (size_t)t * row_stride_4h);

        // pre = px + (W_hh @ h + b_hh) — at step 0 the W_hh@0 term vanishes.
        ggml_tensor* pre;
        if (h) {
            ggml_tensor* ph = ggml_mul_mat(ctx, W_hh, h); // (4H, 1)
            ph = ggml_add(ctx, ph, b_hh);
            pre = ggml_add(ctx, px, ph);
        } else {
            pre = ggml_add(ctx, px, b_hh);
        }

        // Split (4H, 1) into 4 gates of shape (H, 1).  Gate order is
        // PyTorch's (i, f, g, o), each H-wide stripe along ne[0].
        const size_t pre_stride = pre->nb[1];
        ggml_tensor* gi = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)0 * H * f32_size);
        ggml_tensor* gf_gate = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)1 * H * f32_size);
        ggml_tensor* gg = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)2 * H * f32_size);
        ggml_tensor* go = ggml_view_2d(ctx, pre, H, 1, pre_stride, (size_t)3 * H * f32_size);

        ggml_tensor* sig_i = ggml_sigmoid(ctx, gi);
        ggml_tensor* sig_f = ggml_sigmoid(ctx, gf_gate);
        ggml_tensor* sig_o = ggml_sigmoid(ctx, go);
        ggml_tensor* tanh_g = ggml_tanh(ctx, gg);

        // c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
        if (c) {
            c = ggml_add(ctx, ggml_mul(ctx, sig_f, c), ggml_mul(ctx, sig_i, tanh_g));
        } else {
            c = ggml_mul(ctx, sig_i, tanh_g);
        }
        // h_t = sigmoid(o) * tanh(c_t)
        h = ggml_mul(ctx, sig_o, ggml_tanh(ctx, c));

        // Write h_t into column t of output.
        ggml_tensor* slot = ggml_view_2d(ctx, output, H, 1, row_stride_h, (size_t)t * row_stride_h);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h, slot));
    }
    return output;
}

// Bidirectional LSTM. Runs two unidirectional passes (forward over the
// natural order, backward over the reversed order using the
// *_reverse weights), then concatenates along the feature dim:
//   output ne = (2H, T)  with forward H first, then backward H.
static inline ggml_tensor* lstm_bidir(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* X, ggml_tensor* W_ih_f,
                                      ggml_tensor* W_hh_f, ggml_tensor* b_ih_f, ggml_tensor* b_hh_f,
                                      ggml_tensor* W_ih_r, ggml_tensor* W_hh_r, ggml_tensor* b_ih_r,
                                      ggml_tensor* b_hh_r, int H) {
    ggml_tensor* fwd = lstm_unidir(ctx, gf, X, W_ih_f, W_hh_f, b_ih_f, b_hh_f, H, /*reverse=*/false);
    ggml_tensor* bwd = lstm_unidir(ctx, gf, X, W_ih_r, W_hh_r, b_ih_r, b_hh_r, H, /*reverse=*/true);
    // Concat along feature axis (ne[0] == H), result (2H, T).
    return ggml_concat(ctx, fwd, bwd, /*dim=*/0);
}

} // namespace core_lstm

// ======================================================================
// BEGIN kokoro_core_attention.h
// ======================================================================

// src/core/attention.h — shared multi-head attention helpers (header-only).
//
// Replaces the Q/K/V-projection + reshape + RoPE + GQA-expand + flash-attn +
// output-projection block that every LLM-based model in src/ has 1–2 copies
// of. The helper is header-only so the compiler inlines it straight into
// each caller, producing the exact same ggml op sequence as the original
// inline code and preserving bit-identical graph execution.
//
// Scope of the initial version (this commit):
//
//   core_attn::llama_self_attn_kv()  — the classic Llama / Mistral LLM
//     attention block: RMSNorm weights applied by caller, no biases on
//     Q/K/V/O, NEOX RoPE, optional GQA expansion, ggml_flash_attn_ext with
//     a caller-supplied causal-or-sliding-window mask, reshape + output
//     projection. Used by voxtral, voxtral4b, qwen3 (without Q/K norm),
//     and granite LLM decoders.
//
// Follow-up variants (to be added when their first consumer migrates):
//
//   * post-projection Q/K RMSNorm (qwen3)
//   * separate audio-encoder variant with biases + no RoPE (voxtral audio)
//   * adaptive scale / residual_multiplier (granite µP)
//   * KV-cache lookup that returns a (K, V) pair instead of taking
//     pre-permuted inputs (needed when KV cache is stored in a different
//     layout than (head_dim, T, n_heads))
//
// This staged approach keeps the first commit narrow and verifiable. Every
// new caller either fits the existing helper or adds a new sibling helper
// with its own regression test.




namespace core_attn {

// PLAN #60e: KV cache dtype selection from `CRISPASR_KV_QUANT`.
//
// Default returns `GGML_TYPE_F16` so any backend that calls this in
// its `*_kv_init` is bit-identical to legacy behaviour until the user
// opts in. Recognised values: `f16` (default), `q8_0`, `q4_0`. Anything
// else logs a warning to stderr and falls back to F16.
//
// Pairs with the `core_attn::kv_self_attn` write- and read-path
// quant-safety: when the cache type is quantised, the helper switches
// to `ggml_set_rows` for the write (vs `ggml_cpy` for F16, which
// requires contig dst that quant slices never satisfy) and uses
// `ggml_cast(...,F32)` to dequantise on read (CPU-backend safe; F16
// would be metal-only).
//
// `backend_tag` is the prefix on the warning line so a misconfigured
// env var points at a specific backend rather than a generic message.
// Parse a single KV-quant string. Internal helper.
inline ggml_type kv_dtype_parse(const char* s, const char* backend_tag, const char* env_name, ggml_type fallback) {
    if (!s || !*s)
        return fallback;
    if (std::strcmp(s, "f16") == 0 || std::strcmp(s, "F16") == 0)
        return GGML_TYPE_F16;
    if (std::strcmp(s, "f32") == 0 || std::strcmp(s, "F32") == 0)
        return GGML_TYPE_F32;
    if (std::strcmp(s, "q8_0") == 0 || std::strcmp(s, "Q8_0") == 0)
        return GGML_TYPE_Q8_0;
    if (std::strcmp(s, "q4_0") == 0 || std::strcmp(s, "Q4_0") == 0)
        return GGML_TYPE_Q4_0;
    std::fprintf(stderr, "%s: %s='%s' unrecognised, defaulting to f16\n", backend_tag, env_name, s);
    return GGML_TYPE_F16;
}

inline ggml_type kv_dtype_from_env(const char* backend_tag) {
    return kv_dtype_parse(std::getenv("CRISPASR_KV_QUANT"), backend_tag, "CRISPASR_KV_QUANT", GGML_TYPE_F16);
}

// Asymmetric K/V cache types. PLAN #69e — llama.cpp-style independent
// `--cache-type-k` / `--cache-type-v`. The two halves of the KV cache
// have very different sensitivity profiles:
//
//   * V quantises down well: it gets used as `softmax(QK^T) · V`, where
//     softmax already concentrates probability mass and per-element
//     errors get averaged across attended positions. q4_0 V is usually
//     indistinguishable from F16.
//   * K is the fragile half: errors in `QK^T / sqrt(d)` distort *which*
//     positions get attended to (the softmax exponentiates errors).
//     K typically wants q8_0 or higher for the same PPL floor.
//
// Common llama.cpp recipe is `-ctk q8_0 -ctv q4_0`, ~40 % KV memory
// savings vs symmetric Q8_0 with PPL barely moved on Llama-class
// models. The legacy CRISPASR_KV_QUANT remains the default for both
// halves; the per-half overrides take precedence.
struct kv_dtype_pair {
    ggml_type k;
    ggml_type v;
};

inline kv_dtype_pair kv_dtype_pair_from_env(const char* backend_tag) {
    const ggml_type both = kv_dtype_from_env(backend_tag);
    const ggml_type k = kv_dtype_parse(std::getenv("CRISPASR_KV_QUANT_K"), backend_tag, "CRISPASR_KV_QUANT_K", both);
    const ggml_type v = kv_dtype_parse(std::getenv("CRISPASR_KV_QUANT_V"), backend_tag, "CRISPASR_KV_QUANT_V", both);
    return {k, v};
}

// PLAN #69b — pick the backend on which to allocate the KV cache.
// Default: same backend as the model weights (`gpu_backend`). When
// `CRISPASR_KV_ON_CPU=1` is set, allocate on `cpu_backend` instead so
// users with very long context + tight VRAM can spill the cache to
// system RAM. The cost is per-step GPU↔CPU copy of the KV slice, which
// is typically slower than just running with `CRISPASR_KV_QUANT=q4_0`
// to fit KV in VRAM — try KV_QUANT first.
//
// `backend_tag` is the prefix on the warning line. Returns gpu_backend
// when neither offload is requested, cpu_backend when it is.
inline ggml_backend_t kv_backend_from_env(ggml_backend_t gpu_backend, ggml_backend_t cpu_backend,
                                          const char* backend_tag) {
    const char* s = std::getenv("CRISPASR_KV_ON_CPU");
    if (!s || !*s || std::strcmp(s, "0") == 0)
        return gpu_backend;
    if (!cpu_backend) {
        std::fprintf(stderr, "%s: CRISPASR_KV_ON_CPU=%s requested but no CPU backend available, falling back to GPU\n",
                     backend_tag, s);
        return gpu_backend;
    }
    return cpu_backend;
}

// PLAN #73 — quant-safe per-step KV cache write. Replaces the inline
// `ggml_cpy(K_perm, ggml_view_4d(kv_k, …))` pattern that several
// backends use (canary, cohere, kyutai_stt, …). Works for any cache
// dtype: F16 / F32 take the strided-view + ggml_cpy path (preserved
// for bit-exactness with the legacy code), Q8_0 / Q4_0 take a
// ggml_set_rows path keyed by a runtime indices tensor.
//
// Caller responsibilities:
//   * `K_perm` / `V_perm` shape: [head_dim, T, n_kv_heads], i.e.
//     already permuted from the QKV layout into cache layout.
//   * `kv_k` / `kv_v` shape: [head_dim, max_ctx, n_kv_heads, n_layers].
//   * `indices` is an I32 tensor of length T containing the cache
//     positions to write to (typically `[n_past, n_past+T)` — the
//     same tensor used as RoPE positions). Required for quant cache;
//     may be nullptr for F16/F32 (the fast static-offset path will
//     be used).
//
// Adds the K and V writes to `gf` via ggml_build_forward_expand. The
// returned bool is informational: true if the quant path was taken,
// false for the F16 fast path.
inline bool kv_cache_write(ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* K_perm, ggml_tensor* V_perm,
                           ggml_tensor* kv_k, ggml_tensor* kv_v, int layer_idx, int n_past, int T,
                           ggml_tensor* indices) {
    const bool quant_k = ggml_is_quantized(kv_k->type);
    const bool quant_v = ggml_is_quantized(kv_v->type);
    const bool quant_any = quant_k || quant_v;

    if (!quant_any) {
        // Legacy F16/F32 path: strided view + ggml_cpy. Bit-identical
        // to the inline code that callers used to have.
        const int hd = (int)kv_k->ne[0];
        const int nh = (int)kv_k->ne[2];
        ggml_tensor* k_dst = ggml_view_4d(ctx, kv_k, hd, T, nh, 1, kv_k->nb[1], kv_k->nb[2], kv_k->nb[3],
                                          (size_t)layer_idx * kv_k->nb[3] + (size_t)n_past * kv_k->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K_perm, k_dst));
        ggml_tensor* v_dst = ggml_view_4d(ctx, kv_v, hd, T, nh, 1, kv_v->nb[1], kv_v->nb[2], kv_v->nb[3],
                                          (size_t)layer_idx * kv_v->nb[3] + (size_t)n_past * kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V_perm, v_dst));
        return false;
    }

    // Quant path: ggml_set_rows expects rows along ne[0] of the dst.
    // The cache layout is [hd, max_ctx, n_kv, n_layers]; rows are along
    // max_ctx (dim 1). So we view the layer slice as [hd, max_ctx, nh],
    // then ggml_set_rows writes T rows at positions given by `indices`.
    GGML_ASSERT(indices && "kv_cache_write: quant cache requires indices tensor");
    const int hd = (int)kv_k->ne[0];
    const int nh = (int)kv_k->ne[2];
    ggml_tensor* k_layer =
        ggml_view_3d(ctx, kv_k, hd, (int)kv_k->ne[1], nh, kv_k->nb[1], kv_k->nb[2], (size_t)layer_idx * kv_k->nb[3]);
    ggml_tensor* v_layer =
        ggml_view_3d(ctx, kv_v, hd, (int)kv_v->ne[1], nh, kv_v->nb[1], kv_v->nb[2], (size_t)layer_idx * kv_v->nb[3]);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_layer, K_perm, indices));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_layer, V_perm, indices));
    (void)n_past; // unused on quant path; indices carries position info
    return true;
}

// Parameters that differ from call to call. Everything here is a plain
// value type so the compiler can inline the caller's constants into the
// helper's ggml_* op chain.
struct LlamaSelfAttnParams {
    int n_heads;    // query heads (== n_kv_heads for MHA)
    int n_kv_heads; // key/value heads; with GQA, n_heads / n_kv_heads > 1
    int head_dim;   // per-head dimension
    int n_kv_grp;   // == n_heads / n_kv_heads (caller precomputes)
    int n_ctx_orig; // rope n_ctx_orig (usually llm_max_pos)
    float rope_theta;
    float attn_scale; // usually 1 / sqrt(head_dim); pass explicitly
};

// Llama / Mistral-style self-attention with optional GQA, NEOX RoPE, and
// ggml_flash_attn_ext.
//
// Inputs:
//   x              [d_model, T]  — RMSNormed input for this layer (the
//                                   caller does the norm + the learned
//                                   scale multiplication)
//   q_w,k_w,v_w    Q/K/V weight tensors (no biases in the LLM case)
//   o_w            output projection weight
//   positions      [T]           — RoPE position ids
//   mask           [ctx, T] F16  — causal / sliding-window mask or nullptr
//                                   for no-mask (voxtral audio case)
//
// Output:
//   attn           [d_model, T]  — post-output-projection tensor. The
//                                   caller adds it to the residual.
// Fused QKV variant: if qkv_w is non-null, do a single matmul and split.
// qkv_w shape: [d_model, n_q*hd + 2*n_kv*hd] — concatenated Q, K, V weights.
// Falls back to 3 separate matmuls when qkv_w is null (backward compat).
static inline ggml_tensor* llama_self_attn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* q_w, ggml_tensor* k_w,
                                           ggml_tensor* v_w, ggml_tensor* o_w, ggml_tensor* positions,
                                           ggml_tensor* mask, const LlamaSelfAttnParams& p,
                                           ggml_tensor* qkv_w = nullptr) {
    const int hd = p.head_dim;
    const int n_q = p.n_heads;
    const int n_kv = p.n_kv_heads;
    const int n_ctx = p.n_ctx_orig;
    const int grp = p.n_kv_grp;

    ggml_tensor* Q;
    ggml_tensor* K;
    ggml_tensor* V;

    if (qkv_w) {
        // Single fused matmul: one mul_mat instead of three.
        // qkv_w: [d_model, q_dim + k_dim + v_dim]
        // Output: [q_dim + k_dim + v_dim, T]
        ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, x);
        const int q_dim = n_q * hd;
        const int kv_dim = n_kv * hd;
        const int T = (int)x->ne[1];
        // Split along ne[0]: Q=[0..q_dim), K=[q_dim..q_dim+kv_dim), V=[q_dim+kv_dim..)
        Q = ggml_view_2d(ctx, qkv, q_dim, T, qkv->nb[1], 0);
        K = ggml_view_2d(ctx, qkv, kv_dim, T, qkv->nb[1], q_dim * ggml_type_size(qkv->type));
        V = ggml_view_2d(ctx, qkv, kv_dim, T, qkv->nb[1], (q_dim + kv_dim) * ggml_type_size(qkv->type));
    } else {
        // Standard 3 separate matmuls (backward compat).
        Q = ggml_mul_mat(ctx, q_w, x);
        K = ggml_mul_mat(ctx, k_w, x);
        V = ggml_mul_mat(ctx, v_w, x);
    }

    // T is the time dim of x; ggml stores [d_model, T] as ne = [d_model, T].
    const int T = (int)x->ne[1];

    Q = ggml_reshape_3d(ctx, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx, V, hd, n_kv, T);

    // NEOX RoPE on Q and K. Same args as the original inline code in
    // voxtral / voxtral4b / qwen3 / granite LLM blocks.
    Q = ggml_rope_ext(ctx, Q, positions, /*freq_factors*/ nullptr, hd, GGML_ROPE_TYPE_NEOX, n_ctx, p.rope_theta,
                      /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f,
                      /*attn_factor*/ 1.0f, /*beta_fast*/ 32.0f, /*beta_slow*/ 1.0f);
    K = ggml_rope_ext(ctx, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, n_ctx, p.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f,
                      1.0f);

    // GQA expansion: replicate each KV head `grp` times along a new dim so
    // K/V have n_heads rows instead of n_kv_heads, then flatten back.
    if (grp > 1) {
        ggml_tensor* K4 = ggml_reshape_4d(ctx, K, hd, 1, n_kv, T);
        ggml_tensor* V4 = ggml_reshape_4d(ctx, V, hd, 1, n_kv, T);
        K4 = ggml_repeat_4d(ctx, K4, hd, grp, n_kv, T);
        V4 = ggml_repeat_4d(ctx, V4, hd, grp, n_kv, T);
        K = ggml_cont(ctx, ggml_reshape_3d(ctx, K4, hd, n_q, T));
        V = ggml_cont(ctx, ggml_reshape_3d(ctx, V4, hd, n_q, T));
    }

    // Permute to flash-attention layout: (head_dim, T, n_heads).
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    // Flash attention. Output shape = (head_dim, n_heads, T, 1).
    ggml_tensor* attn =
        ggml_flash_attn_ext(ctx, Q, K, V, mask, p.attn_scale, /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);

    // Back to (d_model, T).
    attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);

    // Output projection (no bias).
    return ggml_mul_mat(ctx, o_w, attn);
}

// ---------------------------------------------------------------------------
// Encoder self-attention — biased Q/K/V/O projections, optional RoPE.
//
// Covers architectures like the Whisper audio encoder (voxtral 3B) and the
// causal RoPE+SwiGLU audio encoder (voxtral4b). Key differences from the
// LLM llama_self_attn():
//   - Q, K, V, O projections can each have an optional bias (nullptr = skip)
//   - RoPE is optional: pass positions == nullptr to skip
//   - GQA expansion is included for architectures that use it
//
// The caller still handles the pre-attention norm and post-attention
// residual add.
// ---------------------------------------------------------------------------

struct EncoderSelfAttnParams {
    int n_heads;    // query heads
    int n_kv_heads; // key/value heads (usually == n_heads for encoders)
    int head_dim;
    int n_kv_grp;     // n_heads / n_kv_heads (1 for MHA)
    float attn_scale; // usually 1/sqrt(head_dim)
    // RoPE params (only used when positions != nullptr)
    int n_ctx_orig;
    float rope_theta;
    // When true (default), wrap ggml_permute() in ggml_cont() before
    // flash_attn_ext. voxtral 3B needs this; voxtral4b does not (its
    // encoder was written without cont and changing it would alter the
    // ggml graph structure). Set to false for voxtral4b compatibility.
    bool permute_cont = true;
};

static inline ggml_tensor* encoder_self_attn(ggml_context* ctx, ggml_tensor* x, ggml_tensor* q_w, ggml_tensor* q_b,
                                             ggml_tensor* k_w, ggml_tensor* k_b, ggml_tensor* v_w, ggml_tensor* v_b,
                                             ggml_tensor* o_w, ggml_tensor* o_b, ggml_tensor* positions,
                                             ggml_tensor* mask, const EncoderSelfAttnParams& p) {
    const int hd = p.head_dim;
    const int n_q = p.n_heads;
    const int n_kv = p.n_kv_heads;
    const int grp = p.n_kv_grp;
    const int T = (int)x->ne[1];

    // Q/K/V projections with optional biases.
    ggml_tensor* Q = ggml_mul_mat(ctx, q_w, x);
    if (q_b)
        Q = ggml_add(ctx, Q, q_b);
    ggml_tensor* K = ggml_mul_mat(ctx, k_w, x);
    if (k_b)
        K = ggml_add(ctx, K, k_b);
    ggml_tensor* V = ggml_mul_mat(ctx, v_w, x);
    if (v_b)
        V = ggml_add(ctx, V, v_b);

    Q = ggml_reshape_3d(ctx, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx, V, hd, n_kv, T);

    // Optional RoPE (skip for encoders with learned positional embeddings).
    if (positions) {
        Q = ggml_rope_ext(ctx, Q, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, p.n_ctx_orig, p.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, positions, nullptr, hd, GGML_ROPE_TYPE_NEOX, p.n_ctx_orig, p.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
    }

    // GQA expansion (when n_kv_heads < n_heads).
    if (grp > 1) {
        ggml_tensor* K4 = ggml_reshape_4d(ctx, K, hd, 1, n_kv, T);
        ggml_tensor* V4 = ggml_reshape_4d(ctx, V, hd, 1, n_kv, T);
        K4 = ggml_repeat_4d(ctx, K4, hd, grp, n_kv, T);
        V4 = ggml_repeat_4d(ctx, V4, hd, grp, n_kv, T);
        K = ggml_cont(ctx, ggml_reshape_3d(ctx, K4, hd, n_q, T));
        V = ggml_cont(ctx, ggml_reshape_3d(ctx, V4, hd, n_q, T));
    }

    // Permute to flash-attention layout: (head_dim, T, n_heads).
    Q = ggml_permute(ctx, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx, K, 0, 2, 1, 3);
    V = ggml_permute(ctx, V, 0, 2, 1, 3);
    if (p.permute_cont) {
        Q = ggml_cont(ctx, Q);
        K = ggml_cont(ctx, K);
        V = ggml_cont(ctx, V);
    }

    // Flash attention (bidirectional if mask==nullptr, causal/SWA otherwise).
    ggml_tensor* attn = ggml_flash_attn_ext(ctx, Q, K, V, mask, p.attn_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, hd * n_q, T);

    // Output projection with optional bias.
    attn = ggml_mul_mat(ctx, o_w, attn);
    if (o_b)
        attn = ggml_add(ctx, attn, o_b);
    return attn;
}

// ---------------------------------------------------------------------------
// KV-cached self-attention for the LLM decoders (qwen3-asr, voxtral,
// voxtral4b, granite-speech).
//
// Replaces the Q/K/V-proj + [optional Q/K norm] + RoPE + persistent-KV-cache
// write + cache read + [manual GQA expansion] + flash-attn-ext + output-proj
// block that each of the four models has its own copy of. The helper does
// NOT do the pre-attention RMSNorm or the post-attention residual add —
// callers do those inline so the helper stays focused on the attention
// block proper (which is where the per-model knobs live).
//
// KV cache layout convention: ne = (head_dim, max_ctx, n_kv_heads, n_layers).
// Every consumer already stores its cache this way, which is why this helper
// is shareable in the first place.
// ---------------------------------------------------------------------------

// GQA expansion strategy.
//
// qwen3 and voxtral (3b) manually expand each KV head into `n_kv_grp` query
// heads via reshape_4d -> repeat_4d -> reshape_3d, and then wrap the final
// reshape in ggml_cont. voxtral4b also manually expands, but skips the final
// ggml_cont. granite skips the manual expansion entirely and relies on
// ggml_flash_attn_ext's native GQA support (pass Kfull/Vfull with n_kv heads
// directly, flash-attn handles the repeat internally).
//
// Each mode produces slightly different graph ops. Picking the wrong one
// breaks bit-identity on the regression sweep, so this is an explicit knob.
enum GqaMode {
    GQA_MANUAL_CONT = 0,   // reshape_4d / repeat_4d / reshape_3d + ggml_cont
    GQA_MANUAL_NOCONT = 1, // reshape_4d / repeat_4d / reshape_3d, no final cont
    GQA_NATIVE = 2,        // no expansion; flash_attn_ext handles GQA itself
};

struct KvSelfAttnParams {
    int n_heads;    // query heads
    int n_kv_heads; // key/value heads (== n_heads for MHA)
    int head_dim;   // per-head dimension
    int n_kv_grp;   // n_heads / n_kv_heads (caller precomputes)
    int n_ctx_orig; // RoPE n_ctx_orig (some models 0, some llm_max_pos)
    float rope_theta;
    float rope_beta_fast; // RoPE extrapolation beta_fast (qwen3/voxtral: 32, others: 0)
    float rope_beta_slow; // RoPE extrapolation beta_slow (qwen3/voxtral: 1,  others: 0)
    float attn_scale;     // usually 1/sqrt(head_dim); granite uses µP scale
    float qk_norm_eps;    // RMSNorm epsilon for optional Q/K norm (qwen3); unused otherwise
    GqaMode gqa_mode;
    int rope_type = GGML_ROPE_TYPE_NEOX; // NEOX for most models, NORMAL for fairseq2/omniasr
    // Partial-rotary RoPE: number of head-dim entries to rotate. The
    // remaining `head_dim - n_rot` entries pass through unchanged. Used
    // by Gemma4 full-attention layers (`partial_rotary_factor=0.25`,
    // i.e. n_rot = head_dim/4) and Phi-3-style models. Default 0 means
    // rotate the entire head_dim — matches every existing caller.
    int n_rot = 0;
    // Apply RMSNorm-without-learned-weight to V before the cache write.
    // Gemma4's `v_norm` is constructed with `with_scale=False`, i.e.
    // RMSNorm with no learned scale tensor — there is no weight to load,
    // we just need to run the normalisation op on V. Default false → no
    // op, matches every other consumer.
    bool v_rms_norm = false;
    // Optional per-dimension RoPE frequency factors (e.g. Llama 3 scaling).
    ggml_tensor* rope_freq_factors = nullptr;
};

// KV-cached self-attention. Writes the new K/V into the persistent cache
// slice at [n_past, n_past + T) for layer `il`, then reads the full history
// [0, n_past + T) back out and runs flash-attention against Q.
//
// Inputs:
//   x            [d_model, T]  — pre-attention normalized activations
//   q_w,k_w,v_w  projection weights (no biases for the Llama case)
//   o_w          output projection weight (no bias)
//   q_norm_w     [head_dim] Q-norm weight, or nullptr to skip (non-qwen3)
//   k_norm_w     [head_dim] K-norm weight, or nullptr to skip
//   positions    [T] I32 — absolute positions n_past, n_past+1, ...
//   causal_mask  [Lk, T] F16 or nullptr (decode path uses nullptr)
//   kv_k, kv_v   persistent cache, ne = (hd, max_ctx, n_kv, n_layers)
//   il           layer index into the cache's trailing dim
//   n_past       number of tokens already in the cache
//
// Output:
//   attn         [d_model, T] — post-output-projection tensor. Caller adds
//                                it to the residual.
// fixed_kv_len > 0: override the KV-read length (Lk) to a constant, keeping
// topology identical across calls with different n_past.  Unwritten slots are
// masked to -inf by causal_mask so they never affect output.
//
// kv_indices != nullptr: scatter the new K/V into the cache via ggml_set_rows
// keyed by the runtime indices tensor instead of the default static-offset
// ggml_cpy.  Required for graph-cache reuse across calls at different n_past:
// the static-offset path bakes n_past into the graph as a literal byte offset,
// so a cached graph built at n_past=A would write to slot A even when reused at
// n_past=B; the dynamic-index path makes the destination a runtime input. Pass
// the same `positions` tensor that's already populated with [n_past, n_past+T)
// for RoPE — the indices required for set_rows are bit-equivalent.
//
// q_b/k_b/v_b/o_b: optional projection biases. Qwen2 (mimo-asr LM) sets
// `attention_bias=true` and ships per-layer Q/K/V biases; Qwen3 / Llama /
// granite / voxtral / gemma4 do not. Default nullptr keeps the graph
// bit-identical for those callers.
//
// qkv_b: optional fused-bias for the Qwen2 fused-QKV path. When qkv_w is
// non-null and the GGUF stores a fused `attn.qkv.bias` (length q_dim +
// 2*kv_dim), pass it here — it's added to the fused matmul output before
// the Q/K/V split. q_b/k_b/v_b should be nullptr in that case (the
// caller emits one fused tensor instead of three). Algebraically
// identical to per-projection bias adds; one ggml_add op instead of
// three.
static inline ggml_tensor* kv_self_attn(ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* x, ggml_tensor* q_w,
                                        ggml_tensor* k_w, ggml_tensor* v_w, ggml_tensor* o_w, ggml_tensor* q_norm_w,
                                        ggml_tensor* k_norm_w, ggml_tensor* positions, ggml_tensor* causal_mask,
                                        ggml_tensor* kv_k, ggml_tensor* kv_v, int il, int n_past,
                                        const KvSelfAttnParams& p, ggml_tensor* qkv_w = nullptr, int fixed_kv_len = 0,
                                        ggml_tensor* kv_indices = nullptr, ggml_tensor* q_b = nullptr,
                                        ggml_tensor* k_b = nullptr, ggml_tensor* v_b = nullptr,
                                        ggml_tensor* o_b = nullptr, ggml_tensor* qkv_b = nullptr) {
    const int hd = p.head_dim;
    const int n_q = p.n_heads;
    const int n_kv = p.n_kv_heads;
    const int grp = p.n_kv_grp;
    const int T = (int)x->ne[1];
    const int Lk = fixed_kv_len > 0 ? fixed_kv_len : (n_past + T);

    // ---- Q/K/V projections ----
    ggml_tensor* Q;
    ggml_tensor* K;
    ggml_tensor* V;

    if (qkv_w) {
        // Fused: one matmul, then split output. The 2D views below are
        // strided (each T-row leaves gaps for the other Q/K/V), so for T>1
        // the downstream ggml_reshape_3d would fail its contiguity assert.
        // ggml_cont materialises each into its own contiguous buffer; for
        // T=1 the cont is a no-op (single row is already contiguous).
        ggml_tensor* qkv = ggml_mul_mat(ctx0, qkv_w, x);
        if (qkv_b) {
            // Fused bias (1D, length q_dim + 2*kv_dim) added before the
            // split so each Q/K/V chunk picks up its own slice. One add
            // instead of three; algebraically identical to per-proj adds.
            qkv = ggml_add(ctx0, qkv, qkv_b);
        }
        const int q_dim = n_q * hd;
        const int kv_dim = n_kv * hd;
        const size_t ts = ggml_type_size(qkv->type);
        Q = ggml_view_2d(ctx0, qkv, q_dim, T, qkv->nb[1], 0);
        K = ggml_view_2d(ctx0, qkv, kv_dim, T, qkv->nb[1], q_dim * ts);
        V = ggml_view_2d(ctx0, qkv, kv_dim, T, qkv->nb[1], (q_dim + kv_dim) * ts);
        if (T > 1) {
            Q = ggml_cont(ctx0, Q);
            K = ggml_cont(ctx0, K);
            V = ggml_cont(ctx0, V);
        }
    } else {
        Q = ggml_mul_mat(ctx0, q_w, x);
        K = ggml_mul_mat(ctx0, k_w, x);
        V = ggml_mul_mat(ctx0, v_w, x);
    }

    // Optional Q/K/V projection biases (Qwen2). Applied before reshape so
    // the bias broadcasts along the time dim; q_b/k_b/v_b are 1D.
    if (q_b)
        Q = ggml_add(ctx0, Q, q_b);
    if (k_b)
        K = ggml_add(ctx0, K, k_b);
    if (v_b)
        V = ggml_add(ctx0, V, v_b);

    Q = ggml_reshape_3d(ctx0, Q, hd, n_q, T);
    K = ggml_reshape_3d(ctx0, K, hd, n_kv, T);
    V = ggml_reshape_3d(ctx0, V, hd, n_kv, T);

    // ---- Optional Q/K RMSNorm (qwen3) ----
    if (q_norm_w) {
        Q = ggml_rms_norm(ctx0, Q, p.qk_norm_eps);
        Q = ggml_mul(ctx0, Q, q_norm_w);
    }
    if (k_norm_w) {
        K = ggml_rms_norm(ctx0, K, p.qk_norm_eps);
        K = ggml_mul(ctx0, K, k_norm_w);
    }

    // ---- Optional V RMSNorm without learned weight (gemma4) ----
    // gemma4's v_norm is `Gemma4RMSNorm(head_dim, with_scale=False)`,
    // so there is no weight tensor — we just normalise V along its
    // last (head_dim) axis, exactly as ggml_rms_norm does.
    if (p.v_rms_norm) {
        V = ggml_rms_norm(ctx0, V, p.qk_norm_eps);
    }

    // ---- RoPE (NEOX for most models, NORMAL for fairseq2/omniasr) ----
    // p.n_rot > 0 selects partial-rotary mode (only the first n_rot
    // entries of each head are rotated; the rest pass through). 0
    // means rotate the entire head_dim, which matches every existing
    // caller's prior behaviour.
    const int n_rot = p.n_rot > 0 ? p.n_rot : hd;
    Q = ggml_rope_ext(ctx0, Q, positions, p.rope_freq_factors, n_rot, p.rope_type, p.n_ctx_orig, p.rope_theta,
                      /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f,
                      /*attn_factor*/ 1.0f, p.rope_beta_fast, p.rope_beta_slow);
    K = ggml_rope_ext(ctx0, K, positions, p.rope_freq_factors, n_rot, p.rope_type, p.n_ctx_orig, p.rope_theta, 1.0f,
                      0.0f, 1.0f, p.rope_beta_fast, p.rope_beta_slow);

    // ---- Permute new K/V to (hd, T, n_kv) for cache write ----
    ggml_tensor* K_new_perm = ggml_permute(ctx0, K, 0, 2, 1, 3);
    ggml_tensor* V_new_perm = ggml_permute(ctx0, V, 0, 2, 1, 3);

    // ---- Write into the persistent KV cache at [n_past, n_past+T) ----
    // The default ggml_cpy(F32, slice-of-cache) path requires the
    // destination to be contiguous when the source/dst types differ
    // (CPU backend's `dup_to_q` aborts otherwise, and Metal's CPY also
    // skips non-contiguous quantised dst). For a quantised cache —
    // PLAN #60e CRISPASR_KV_QUANT={q8_0,q4_0} — we instead always use
    // `ggml_set_rows` with a per-token row-index tensor, which both
    // backends accept for F32→Q* directly. When the caller already
    // supplies `kv_indices` (cached-graph reuse path) we honour that;
    // otherwise we synthesise the indices from `positions` (which is
    // [n_past..n_past+T) by construction for RoPE — exactly the row
    // ids set_rows needs).
    const bool quant_kv = ggml_is_quantized(kv_k->type);
    if (kv_indices || quant_kv) {
        ggml_tensor* eff_idx = kv_indices ? kv_indices : positions;
        ggml_tensor* k_layer =
            ggml_view_3d(ctx0, kv_k, hd, kv_k->ne[1], n_kv, kv_k->nb[1], kv_k->nb[2], (size_t)il * kv_k->nb[3]);
        ggml_tensor* v_layer =
            ggml_view_3d(ctx0, kv_v, hd, kv_v->ne[1], n_kv, kv_v->nb[1], kv_v->nb[2], (size_t)il * kv_v->nb[3]);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, k_layer, K_new_perm, eff_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx0, v_layer, V_new_perm, eff_idx));
    } else {
        ggml_tensor* k_view = ggml_view_4d(ctx0, kv_k, hd, T, n_kv, 1, kv_k->nb[1], kv_k->nb[2], kv_k->nb[3],
                                           (size_t)il * kv_k->nb[3] + (size_t)n_past * kv_k->nb[1]);
        ggml_tensor* v_view = ggml_view_4d(ctx0, kv_v, hd, T, n_kv, 1, kv_v->nb[1], kv_v->nb[2], kv_v->nb[3],
                                           (size_t)il * kv_v->nb[3] + (size_t)n_past * kv_v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, K_new_perm, k_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, V_new_perm, v_view));
    }

    // ---- Read full K/V history from cache ----
    // Cache may be allocated as F16 (default) or as a quantized type
    // (Q8_0 / Q4_0 / etc., per CRISPASR_KV_QUANT — PLAN #60e). For the
    // default F16 path the strided per-layer view becomes a contiguous
    // F16 tensor via ggml_cont (a CPY F16→F16 op). For a quantized
    // cache the equivalent CPY (Q8_0→Q8_0 etc.) isn't supported by
    // Metal, so we use ggml_cast(...,F32) which lowers to CPY Q*→F32
    // — supported on both Metal and the CPU backend (the CPU `dup`
    // dispatch only implements `Q*→F32` for the dequant-on-read path,
    // not `Q*→F16`; so F32 is the only safe target if the scheduler
    // splits the op). The cache *storage* still uses ~half the bytes
    // (for Q8_0); reads pay one dequant pass per layer per step.
    // Flash-attn-ext on Metal accepts F32 K/V natively (and F16 / quant
    // too) but mixing types across K and V isn't supported, so both
    // sides cast to the same dtype.
    ggml_tensor* k_layer_view =
        ggml_view_3d(ctx0, kv_k, hd, Lk, n_kv, kv_k->nb[1], kv_k->nb[2], (size_t)il * kv_k->nb[3]);
    ggml_tensor* v_layer_view =
        ggml_view_3d(ctx0, kv_v, hd, Lk, n_kv, kv_v->nb[1], kv_v->nb[2], (size_t)il * kv_v->nb[3]);
    // CRISPASR_KV_READ_F32=1 forces the cache read to dequantise (or
    // upcast F16) to F32 before flash_attn. Useful when F16 attention
    // accumulator drift on Metal sends the sampler off the rails for
    // sensitive models (chatterbox T3 — see LEARNINGS §82). Default
    // off to preserve legacy bit-exactness with the F16 fast path.
    static const bool s_kv_read_f32 = []() {
        const char* s = std::getenv("CRISPASR_KV_READ_F32");
        return s && *s && std::strcmp(s, "0") != 0;
    }();
    const bool need_dequant_k = ggml_is_quantized(kv_k->type) || (s_kv_read_f32 && kv_k->type != GGML_TYPE_F32);
    const bool need_dequant_v = ggml_is_quantized(kv_v->type) || (s_kv_read_f32 && kv_v->type != GGML_TYPE_F32);
    ggml_tensor* Kfull = need_dequant_k ? ggml_cast(ctx0, k_layer_view, GGML_TYPE_F32) : ggml_cont(ctx0, k_layer_view);
    ggml_tensor* Vfull = need_dequant_v ? ggml_cast(ctx0, v_layer_view, GGML_TYPE_F32) : ggml_cont(ctx0, v_layer_view);

    // ---- GQA expansion ----
    if (p.gqa_mode != GQA_NATIVE && grp > 1) {
        ggml_tensor* K4 = ggml_reshape_4d(ctx0, Kfull, hd, Lk, 1, n_kv);
        ggml_tensor* V4 = ggml_reshape_4d(ctx0, Vfull, hd, Lk, 1, n_kv);
        K4 = ggml_repeat_4d(ctx0, K4, hd, Lk, grp, n_kv);
        V4 = ggml_repeat_4d(ctx0, V4, hd, Lk, grp, n_kv);
        if (p.gqa_mode == GQA_MANUAL_CONT) {
            Kfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, K4, hd, Lk, n_q));
            Vfull = ggml_cont(ctx0, ggml_reshape_3d(ctx0, V4, hd, Lk, n_q));
        } else {
            Kfull = ggml_reshape_3d(ctx0, K4, hd, Lk, n_q);
            Vfull = ggml_reshape_3d(ctx0, V4, hd, Lk, n_q);
        }
    }

    // CrispASR debug hook (#83 bisect): when CRISPASR_CORE_ATTN_DUMP_FA_LAYER
    // matches the current layer index, name + add the FA inputs and output as
    // graph outputs so chatterbox.cpp's run_t3_kv post-compute dumper can
    // fetch them. Negligible perf cost when the env knob is unset.
    auto dbg_dump_il_env = std::getenv("CRISPASR_CORE_ATTN_DUMP_FA_LAYER");
    const int dbg_dump_il = dbg_dump_il_env ? (int)std::strtol(dbg_dump_il_env, nullptr, 10) : -1;
    const bool dbg_dump = ((int)il == dbg_dump_il);
    if (dbg_dump) {
        ggml_tensor* Q_pre = ggml_cont(ctx0, Q);
        ggml_set_name(Q_pre, "DBG_Q_post_rope");
        ggml_set_output(Q_pre);
        ggml_build_forward_expand(gf, Q_pre);

        ggml_tensor* Kfull_dbg = ggml_cast(ctx0, Kfull, GGML_TYPE_F32);
        ggml_set_name(Kfull_dbg, "DBG_Kfull");
        ggml_set_output(Kfull_dbg);
        ggml_build_forward_expand(gf, Kfull_dbg);

        ggml_tensor* Vfull_dbg = ggml_cast(ctx0, Vfull, GGML_TYPE_F32);
        ggml_set_name(Vfull_dbg, "DBG_Vfull");
        ggml_set_output(Vfull_dbg);
        ggml_build_forward_expand(gf, Vfull_dbg);
    }

    // ---- Permute Q to (hd, T, n_q) for flash-attn ----
    Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

    // ---- Flash attention + reshape + output projection ----
    ggml_tensor* attn = ggml_flash_attn_ext(ctx0, Q, Kfull, Vfull, causal_mask, p.attn_scale, /*max_bias*/ 0.0f,
                                            /*logit_softcap*/ 0.0f);
    if (dbg_dump) {
        ggml_set_name(attn, "DBG_fa_out");
        ggml_set_output(attn);
        ggml_build_forward_expand(gf, attn);
    }
    attn = ggml_reshape_2d(ctx0, attn, hd * n_q, T);

    if (dbg_dump) {
        ggml_set_name(attn, "DBG_fa_reshaped");
        ggml_set_output(attn);
        ggml_build_forward_expand(gf, attn);
    }

    ggml_tensor* out = ggml_mul_mat(ctx0, o_w, attn);
    if (o_b)
        out = ggml_add(ctx0, out, o_b);
    return out;
}

} // namespace core_attn

// ======================================================================
// BEGIN kokoro_core_gguf_loader.h
// ======================================================================

// src/core/gguf_loader.h — shared GGUF weight loading scaffolding.
//
// Every model implementation in src/ has its own copy of the "open a
// GGUF file, read its hyperparameters, allocate a backend buffer, mmap
// the weight data, and build a name -> tensor lookup map" dance. The
// code is ~40-60 lines per model and is essentially identical across
// them, with only the model-specific prefix and tensor naming scheme
// changing.
//
// This helper extracts the shared scaffolding. What stays model-specific:
//
//   * Hyperparameter reading (each model has its own hparams struct
//     and GGUF key prefix, e.g. "parakeet.n_layers" vs "voxtral.n_layers").
//   * Vocabulary / tokenizer loading (varies by tokenizer type).
//   * The actual per-field assignment loop that pulls tensors out of
//     the map and stores them in per-layer struct fields.
//
// What this helper does for the model:
//
//   * Opens the GGUF file in two passes (metadata, then tensor alloc).
//   * Provides scalar / string / array reader helpers with defaults.
//   * Allocates the backend buffer and mmap-copies the weight data.
//   * Builds the std::map<std::string, ggml_tensor *> tensor
//     lookup map and returns it in a WeightLoad struct.
//   * Provides require() / try_get() tensor lookup helpers that log a
//     sensible error message when a required tensor is missing.
//
// Usage pattern (each model's *_model_load function):
//
//   // 1. Metadata pass — read hyperparameters.
//   gguf_context * meta = core_gguf::open_metadata(path);
//   if (!meta) return false;
//   hp.n_layers = core_gguf::kv_u32(meta, "mymodel.n_layers", hp.n_layers);
//   // ... other hparams
//   core_gguf::load_vocab_array(meta, "tokenizer.ggml.tokens", vocab);
//   core_gguf::free_metadata(meta);
//
//   // 2. Weight pass — allocate backend buffer, mmap, build tensor map.
//   core_gguf::WeightLoad wl;
//   if (!core_gguf::load_weights(path, backend, wl)) return false;
//   model.ctx     = wl.ctx;
//   model.buf     = wl.buf;
//   model.tensors = std::move(wl.tensors);
//
//   // 3. Bind named tensors into struct fields.
//   model.attn.q_w = core_gguf::require(model.tensors, "encoder.attn.q.weight", "mymodel");
//   ... etc.




namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata (hyperparameters + vocab).
// ---------------------------------------------------------------------------

// Open the GGUF for metadata-only reading. Returns a gguf_context owned
// by the caller — free with free_metadata() when done reading keys.
// Returns nullptr and prints an error to stderr on failure.
gguf_context* open_metadata(const char* path);

// Free a gguf_context obtained from open_metadata().
void free_metadata(gguf_context* gctx);

// Scalar key readers with defaults. All return the default value when
// the key is absent or the type doesn't match.
uint32_t kv_u32(gguf_context* gctx, const char* key, uint32_t default_val);
int32_t kv_i32(gguf_context* gctx, const char* key, int32_t default_val);
float kv_f32(gguf_context* gctx, const char* key, float default_val);
bool kv_bool(gguf_context* gctx, const char* key, bool default_val);
std::string kv_str(gguf_context* gctx, const char* key, const char* default_val);

// Read a string array (e.g. tokenizer.ggml.tokens). Returns an empty
// vector when the key is missing or has the wrong type.
std::vector<std::string> kv_str_array(gguf_context* gctx, const char* key);

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

struct WeightLoad {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    // PLAN #69a layer offload: optional second backend buffer for tensors
    // routed off-GPU. Non-null only when load_weights_split() was used.
    ggml_backend_buffer_t buf_cpu = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// Load all tensor metadata + weights into a new ggml_context backed by
// a newly-allocated backend buffer. On success the WeightLoad struct is
// populated and the caller takes ownership of ctx/buf (typically moving
// them into the model struct).
//
// model_tag is used only in error messages ("parakeet: ...").
bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag, WeightLoad& out);

// PLAN #69a: layer-residency-aware weight loader. Tensors for which
// `is_gpu(tensor_name, user) == true` go on the GPU backend; the rest
// go on the CPU backend. ggml_backend_sched then auto-routes ops to
// follow weight residency, giving llama.cpp's `--n-gpu-layers` behaviour.
//
// Caller takes ownership of `out.ctx`, `out.buf` (gpu partition), and
// `out.buf_cpu` (cpu partition). All three must be freed by the caller
// or via free_weights() / free_weights_split() at shutdown.
//
// Falls back to the legacy alloc+copy path internally — the mmap
// optimisations in load_weights() require contiguous tensor regions
// that the split partition can't satisfy. Acceptable trade-off: users
// who set N_GPU_LAYERS are accepting the extra RAM hit to fit the
// model at all.
//
// Returns false on any allocation / load failure with a logged
// stderr message; partial state is freed before returning.
using IsGpuTensor = bool (*)(const char* tensor_name, void* user);
bool load_weights_split(const char* path, ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, IsGpuTensor is_gpu,
                        void* user, const char* model_tag, WeightLoad& out);

// PLAN #69a — generic predicate for the "<prefix><N>." tensor naming
// used by every LLM-decode backend in src/. Each backend has its own
// prefix:
//   "blk."          (voxtral, voxtral4b, qwen3_asr, granite_speech, gemma4_e2b, mimo_asr)
//   "llm.blk."      (glm_asr)
//   "talker.blk."   (orpheus)
//   "dec."          (omniasr)
// Returns -1 if the tensor name doesn't match `<prefix><integer>.`.
int blk_layer_of_with_prefix(const char* tensor_name, const char* prefix);

// Convenience for backends using bare "blk.<N>." (the most common
// scheme). Equivalent to blk_layer_of_with_prefix(name, "blk.").
int blk_layer_of(const char* tensor_name);

// Configurable predicate: tensors named `<cfg.prefix><N>.<rest>` go to
// CPU iff N >= cfg.threshold; anything else (different prefix, no
// integer, threshold-violating layer) stays on GPU. Pass a pointer
// to a LayerSplitConfig as `user`.
struct LayerSplitConfig {
    const char* prefix; // e.g. "blk.", "llm.blk.", "talker.blk.", "dec."
    int threshold;      // N — first CPU-resident layer
};
bool is_gpu_tensor_with_prefix(const char* tensor_name, void* user);

// Bare "blk." convenience: pass a pointer to an int (the threshold)
// as `user`. Equivalent to is_gpu_tensor_with_prefix() with
// LayerSplitConfig{ "blk.", *N }.
bool is_gpu_tensor_blk(const char* tensor_name, void* user);

// Free a WeightLoad's resources. Call when the model is being destroyed
// and the buffer/context are not held elsewhere.
void free_weights(WeightLoad& wl);

// PLAN #60g: hint the kernel that the mmap'd weight region is now being
// accessed in random order (e.g., the per-layer KV revisit pattern of
// decode steps), and that readahead is therefore wasted IO. No-op if
// the buffer wasn't allocated through one of our mmap paths (e.g.,
// when `CRISPASR_GGUF_MMAP=0` opts out of the default mmap loader).
// Safe to call multiple times.
//
// Recommended use: after prefill completes, before entering the decode
// loop. See PLAN #60g for the rationale.
void mmap_advise_random(ggml_backend_buffer_t buf);

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

// Look up a tensor by name. Returns nullptr (silently) if missing.
ggml_tensor* try_get(const std::map<std::string, ggml_tensor*>& tensors, const char* name);

// Look up a tensor by name. Prints an error to stderr if missing but
// still returns nullptr — the caller decides whether a missing tensor
// is fatal.
ggml_tensor* require(const std::map<std::string, ggml_tensor*>& tensors, const char* name, const char* model_tag);

// Build a shell command that produces the formatted tensor name for a
// per-layer lookup. Avoids the snprintf(buf, sizeof(buf), "...", i) line
// that every loader repeats.
std::string format_layer_name(const char* fmt, int i);
std::string format_layer_name(const char* fmt, int i, int j);

} // namespace core_gguf

// ======================================================================
// BEGIN kokoro_core_gguf_loader.cpp
// ======================================================================

// src/core/gguf_loader.cpp — implementation of core_gguf:: helpers.
// See gguf_loader.h for the interface contract.


#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif


#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata
// ---------------------------------------------------------------------------

gguf_context* open_metadata(const char* path) {
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "core_gguf: failed to open '%s' for metadata read\n", path);
    }
    return g;
}

void free_metadata(gguf_context* gctx) {
    if (gctx)
        gguf_free(gctx);
}

// Type-checked scalar readers. The GGUF format stores types explicitly so
// we can validate; if the file has a mismatched type the reader silently
// returns the default rather than crashing, matching the existing inline
// helpers in each model.

uint32_t kv_u32(gguf_context* gctx, const char* key, uint32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
    case GGUF_TYPE_UINT32:
        return gguf_get_val_u32(gctx, k);
    case GGUF_TYPE_INT32:
        return (uint32_t)gguf_get_val_i32(gctx, k);
    case GGUF_TYPE_UINT64:
        return (uint32_t)gguf_get_val_u64(gctx, k);
    case GGUF_TYPE_INT64:
        return (uint32_t)gguf_get_val_i64(gctx, k);
    case GGUF_TYPE_UINT16:
        return gguf_get_val_u16(gctx, k);
    case GGUF_TYPE_INT16:
        return (uint32_t)gguf_get_val_i16(gctx, k);
    case GGUF_TYPE_UINT8:
        return gguf_get_val_u8(gctx, k);
    case GGUF_TYPE_INT8:
        return (uint32_t)gguf_get_val_i8(gctx, k);
    default:
        return default_val;
    }
}

int32_t kv_i32(gguf_context* gctx, const char* key, int32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
    case GGUF_TYPE_INT32:
        return gguf_get_val_i32(gctx, k);
    case GGUF_TYPE_UINT32:
        return (int32_t)gguf_get_val_u32(gctx, k);
    case GGUF_TYPE_INT64:
        return (int32_t)gguf_get_val_i64(gctx, k);
    case GGUF_TYPE_UINT64:
        return (int32_t)gguf_get_val_u64(gctx, k);
    default:
        return default_val;
    }
}

float kv_f32(gguf_context* gctx, const char* key, float default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    if (t == GGUF_TYPE_FLOAT32)
        return gguf_get_val_f32(gctx, k);
    if (t == GGUF_TYPE_FLOAT64)
        return (float)gguf_get_val_f64(gctx, k);
    return default_val;
}

bool kv_bool(gguf_context* gctx, const char* key, bool default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL)
        return default_val;
    return gguf_get_val_bool(gctx, k);
}

std::string kv_str(gguf_context* gctx, const char* key, const char* default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val ? default_val : "";
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_STRING)
        return default_val ? default_val : "";
    const char* s = gguf_get_val_str(gctx, k);
    return s ? std::string(s) : std::string(default_val ? default_val : "");
}

std::vector<std::string> kv_str_array(gguf_context* gctx, const char* key) {
    std::vector<std::string> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY)
        return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_STRING)
        return out;
    const int n = gguf_get_arr_n(gctx, k);
    out.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(gctx, k, i));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

namespace {

// Read a file slice into a backend tensor. Uses mmap on POSIX; falls back
// to pread/lseek+read when mmap is unavailable (rare in practice).
//
// On POSIX the mmap lives for the duration of one load call — we copy via
// ggml_backend_tensor_set then unmap. No mmap persists past load_weights().
struct MappedFile {
    int fd = -1;
    void* base = nullptr;
    size_t size = 0;
    bool ok = false;

    // When `writable` is true, the mapping is created with copy-on-write
    // semantics (POSIX MAP_PRIVATE + PROT_READ|PROT_WRITE / Win32
    // FILE_MAP_COPY). Reads share the file's page cache; writes get a
    // private anonymous duplicate of the touched page. This lets backends
    // that mutate weights post-load (e.g. parakeet's BN-into-conv fold) run
    // unchanged on the zero-copy path without modifying the underlying file.
    explicit MappedFile(const char* path, bool writable = false) {
#if defined(_WIN32)
        const DWORD page_protect = writable ? PAGE_WRITECOPY : PAGE_READONLY;
        const DWORD view_access = writable ? FILE_MAP_COPY : FILE_MAP_READ;
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (hFile == INVALID_HANDLE_VALUE)
            return;
        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(hFile, &fsize)) {
            CloseHandle(hFile);
            return;
        }
        size = (size_t)fsize.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, nullptr, page_protect, 0, 0, nullptr);
        CloseHandle(hFile);
        if (!hMap)
            return;
        base = MapViewOfFile(hMap, view_access, 0, 0, 0);
        CloseHandle(hMap);
        if (!base)
            return;
        ok = true;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0)
            return;
        struct stat st;
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            fd = -1;
            return;
        }
        size = (size_t)st.st_size;
        const int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
        const int flags = writable ? MAP_PRIVATE : MAP_SHARED;
        base = ::mmap(nullptr, size, prot, flags, fd, 0);
        ::close(fd);
        fd = -1;
        if (base == MAP_FAILED) {
            base = nullptr;
            return;
        }
        ok = true;
#endif
    }
    ~MappedFile() {
#if defined(_WIN32)
        if (base)
            UnmapViewOfFile(base);
#else
        if (base)
            ::munmap(base, size);
#endif
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    // Transfer ownership of the mmap region out of the RAII handle so it
    // outlives the destructor. Used by the CRISPASR_GGUF_MMAP=1 path to
    // hand the mapping to a backend buffer that owns it for the model's
    // lifetime.
    void release() {
        base = nullptr;
        size = 0;
    }
};

// PLAN #51a: a CPU backend buffer whose backing memory is an mmap'd file
// region. On free_buffer the mapping is unmapped — that's the entire
// reason this buffer type exists. Tensors must be bound with
// ggml_backend_tensor_alloc(); we do not provide an init_tensor path.
//
// We reuse ggml_backend_cpu_buffer_type() so ggml_backend_buffer_is_host()
// returns true on this buffer (some scheduler paths key off that).
struct mmap_buffer_ctx {
    void* mmap_base = nullptr;   // page-aligned start of the mmap
    size_t mmap_size = 0;        // length of the mmap
    void* tensor_base = nullptr; // mmap_base + data_off, 32-byte aligned
};

static void* mmap_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return ((mmap_buffer_ctx*)buffer->context)->tensor_base;
}

static void mmap_buffer_free(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    auto* mctx = (mmap_buffer_ctx*)buffer->context;
    if (mctx->mmap_base) {
#if defined(_WIN32)
        UnmapViewOfFile(mctx->mmap_base);
#else
        ::munmap(mctx->mmap_base, mctx->mmap_size);
#endif
    }
    delete mctx;
}

static void mmap_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, uint8_t value, size_t offset,
                                      size_t size) {
    GGML_ASSERT(tensor);
    memset((char*)tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void mmap_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, const void* data, size_t offset,
                                   size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char*)tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void mmap_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* tensor, void* data, size_t offset,
                                   size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char*)tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool mmap_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor* src, ggml_tensor* dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void mmap_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto* mctx = (mmap_buffer_ctx*)buffer->context;
    memset(mctx->tensor_base, value, buffer->size);
}

static const ggml_backend_buffer_i mmap_buffer_iface = {
    /* .free_buffer    = */ mmap_buffer_free,
    /* .get_base       = */ mmap_buffer_get_base,
    /* .init_tensor    = */ nullptr,
    /* .memset_tensor  = */ mmap_buffer_memset_tensor,
    /* .set_tensor     = */ mmap_buffer_set_tensor,
    /* .get_tensor     = */ mmap_buffer_get_tensor,
    /* .set_tensor_2d  = */ nullptr,
    /* .get_tensor_2d  = */ nullptr,
    /* .cpy_tensor     = */ mmap_buffer_cpy_tensor,
    /* .clear          = */ mmap_buffer_clear,
    /* .reset          = */ nullptr,
};

// PLAN #51a (Metal variant): non-CPU backends use `buffer_from_host_ptr`
// to wrap our mmap region directly into a backend buffer (e.g. an
// MTLResourceStorageModeShared MTLBuffer on Apple-Silicon Metal). We
// CANNOT wrap that inner buffer with our own iface to attach a munmap
// callback: ggml-metal pierces the iface abstraction and casts
// `buffer->context` straight to its private `ggml_metal_buffer_t` (see
// `ggml_metal_get_buffer_id` in ggml-metal-context.m), so any wrapper
// makes Metal read garbage and emit "tensor 'X' buffer is nil" for
// every weight — the kokoro Metal gibberish-audio regression.
//
// Instead we hand the inner buffer back as-is and track the mmap region
// in this static side-map. When the buffer is freed elsewhere (model
// shutdown) the inner backend's free callback releases its device-side
// reference, but the host mmap stays mapped — Metal's
// `newBufferWithBytesNoCopy:options:deallocator:nil` doesn't own the
// host pages, so there's no MTLBuffer-side teardown that could munmap.
// We deliberately leak the mmap; on macOS the kernel can still evict
// file-backed pages under pressure (they're not anonymous), and process
// exit reclaims everything. Address-space-wise this costs nothing past
// the model's working set, which we'd be holding anyway.
struct gpu_mmap_handle {
    void* base = nullptr;
    size_t size = 0;
};
static std::mutex g_gpu_mmap_mu;
static std::map<ggml_backend_buffer_t, gpu_mmap_handle> g_gpu_mmap;

static void register_gpu_mmap(ggml_backend_buffer_t buf, void* base, size_t size) {
    std::lock_guard<std::mutex> lk(g_gpu_mmap_mu);
    g_gpu_mmap[buf] = {base, size};
}
static gpu_mmap_handle lookup_gpu_mmap(ggml_backend_buffer_t buf) {
    std::lock_guard<std::mutex> lk(g_gpu_mmap_mu);
    auto it = g_gpu_mmap.find(buf);
    return it != g_gpu_mmap.end() ? it->second : gpu_mmap_handle{};
}

// Issue #94 (chatterbox-turbo segfault during init on macOS / Apple
// Silicon): the legacy alloc+copy load path takes 30-60 s for the
// chatterbox-turbo T3 (658 MB Q8_0) on slow disks and reproducibly
// fails for some users. The zero-copy mmap path completes the same
// load in ~5-10 s and uses half the peak RSS. The CPU mmap path has
// been validated for every backend that goes through this loader
// (mimo-asr, voxtral, voxtral4b, chatterbox base + turbo, kokoro,
// qwen3-tts, vibevoice, parakeet, granite, …) since the PLAN #51a
// rollout in late April; flipping the default after a month of opt-in
// testing matches llama.cpp's behaviour and resolves the slow-load
// reports.
//
// Opt out with `CRISPASR_GGUF_MMAP=0` for users whose model files live
// on volumes that may disappear mid-run (network mounts, removable
// disks); mmap-backed weights SIGBUS if the underlying file vanishes.
static bool mmap_loader_enabled() {
    const char* v = std::getenv("CRISPASR_GGUF_MMAP");
    if (!v || !*v)
        return true;
    return *v != '0';
}

// PLAN #60c: opt-in preload — page-walk the entire mmap region so every
// page is resident before we return. Trades cold-start *load* time for
// cold-start *prefill* time; useful for benchmarking and for users with
// enough RAM to keep the working set resident. POSIX path uses a 1-byte
// volatile read per page; Linux MADV_POPULATE_READ would be a cleaner
// single-syscall version when available.
static bool preload_enabled() {
    const char* v = std::getenv("CRISPASR_GGUF_PRELOAD");
    return v && *v && *v != '0';
}
static void preload_pages(void* base, size_t size) {
#if !defined(_WIN32)
    const long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0)
        return;
    volatile const unsigned char* p = (const unsigned char*)base;
    size_t touched = 0;
    for (size_t off = 0; off < size; off += (size_t)pg) {
        (void)p[off];
        touched++;
    }
    (void)touched;
#else
    (void)base;
    (void)size;
#endif
}

// PLAN #60f: opt-in mlock — pin the mmap region in physical RAM so the
// kernel can't evict pages under memory pressure. Risky on RAM-tight
// hosts (a 16 GB model on a 16 GB box would starve the rest of the
// system). Useful as opt-in for users with comfortable headroom (32+
// GB). Failure (typically RLIMIT_MEMLOCK exceeded) prints a warning
// and continues — mmap'd weights still work, just without the pin.
static bool mlock_enabled() {
    const char* v = std::getenv("CRISPASR_MLOCK");
    return v && *v && *v != '0';
}
static void try_mlock(const char* tag, void* base, size_t size) {
#if !defined(_WIN32)
    if (::mlock(base, size) != 0) {
        fprintf(stderr,
                "%s: mlock(%zu MiB) failed (errno=%d) — pages may still be evicted under "
                "memory pressure. Raise RLIMIT_MEMLOCK (`ulimit -l unlimited`) if you want "
                "the pin to take effect.\n",
                tag, size / (1024 * 1024), errno);
    }
#else
    (void)tag;
    (void)base;
    (void)size;
#endif
}

} // namespace

bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag, WeightLoad& out) {
    const char* tag = model_tag ? model_tag : "core_gguf";

    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&out.ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
        return false;
    }

    // PLAN #51a: zero-copy CPU path. Skip ggml_backend_alloc_ctx_tensors
    // (which would allocate a fresh backend-side buffer) and instead bind
    // each tensor directly into the mmap'd file. Saves one full copy of
    // the weights — the difference between a 14.9 GB F16 GGUF loading on
    // a 16 GB Mac and thrashing swap. Default-on as of issue #94 (slow /
    // failing chatterbox-turbo load on macOS); opt out with
    // `CRISPASR_GGUF_MMAP=0`.
    if (mmap_loader_enabled() && ggml_backend_is_cpu(backend)) {
        MappedFile mf(path, /*writable=*/true);
        if (mf.ok) {
            const size_t data_off = gguf_get_data_offset(gctx);
            const size_t mmap_size = mf.size;
            char* tensor_base = (char*)mf.base + data_off;
            const size_t buf_size = mmap_size > data_off ? (mmap_size - data_off) : 0;

            auto* mctx = new mmap_buffer_ctx{};
            mctx->mmap_base = mf.base;
            mctx->mmap_size = mmap_size;
            mctx->tensor_base = tensor_base;
            mf.release();
            // Hint kernel to start async readahead of the entire weight
            // region. Without this we hit a synchronous page fault on every
            // first access during prefill (~5-10 ms each on the 99%-full
            // external disk we hit during PLAN #51c F16 testing). Mirrors
            // llama.cpp's `llama_mmap` populate path.
#if !defined(_WIN32)
            ::posix_madvise(mctx->mmap_base, mctx->mmap_size, POSIX_MADV_WILLNEED);
#endif
            // PLAN #60c / #60f: optional preload + mlock, opt-in via env.
            if (preload_enabled())
                preload_pages(mctx->mmap_base, mctx->mmap_size);
            if (mlock_enabled())
                try_mlock(tag, mctx->mmap_base, mctx->mmap_size);

            out.buf = ggml_backend_buffer_init(ggml_backend_cpu_buffer_type(), mmap_buffer_iface, mctx, buf_size);
            if (!out.buf) {
                fprintf(stderr, "%s: failed to wrap mmap in backend buffer\n", tag);
#if defined(_WIN32)
                UnmapViewOfFile(mctx->mmap_base);
#else
                ::munmap(mctx->mmap_base, mctx->mmap_size);
#endif
                delete mctx;
                gguf_free(gctx);
                ggml_free(out.ctx);
                out.ctx = nullptr;
                return false;
            }

            // Pre-validate tensor bounds before calling ggml_backend_tensor_alloc,
            // which fires a hard GGML_ASSERT when any tensor's range exceeds
            // buf_size.  Seen on macOS 26.x Tahoe (issue #94) — likely a
            // mismatched or truncated GGUF file.  Detect it here and fall back
            // gracefully to the legacy alloc+copy path instead of crashing.
            bool bounds_ok = true;
            for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                if (tid < 0)
                    continue;
                const size_t off = gguf_get_tensor_offset(gctx, tid);
                if (off + ggml_nbytes(t) > buf_size) {
                    fprintf(stderr,
                            "%s: mmap bounds check failed for tensor '%s' "
                            "(off=%zu + nbytes=%zu > buf_size=%zu) — "
                            "falling back to legacy loader\n",
                            tag, ggml_get_name(t), off, ggml_nbytes(t), buf_size);
                    bounds_ok = false;
                    break;
                }
            }
            if (bounds_ok) {
                for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                    out.tensors[ggml_get_name(t)] = t;
                    const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                    if (tid < 0)
                        continue;
                    const size_t off = gguf_get_tensor_offset(gctx, tid);
                    ggml_backend_tensor_alloc(out.buf, t, tensor_base + off);
                }
                gguf_free(gctx);
                return true;
            }
            // Bounds check failed — release the mmap buffer and fall through to
            // the legacy alloc+copy path below.
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
        }
        // mmap failed or bounds check failed — fall through to the legacy
        // alloc + copy path. Functionally equivalent, just with more RSS.
    }

    // PLAN #51a (Metal variant): zero-copy GPU path via the device's
    // `buffer_from_host_ptr` capability. On Apple-Silicon Metal this maps
    // to `[MTLDevice newBufferWithBytesNoCopy:length:options:deallocator:]`
    // wrapping our mmap region in an MTLResourceStorageModeShared buffer
    // — same physical pages the CPU sees thanks to unified memory, no
    // device-side allocation, no copy. On discrete-Metal hosts (Intel +
    // eGPU) this lets the GPU page-fault from the file directly. The
    // device-cap probe means we silently fall through on backends that
    // don't advertise host-pointer support (CUDA without managed memory,
    // Vulkan, etc.).
    //
    // The inner buffer is returned as `out.buf` UNWRAPPED. We used to
    // wrap it to attach a munmap callback to free_buffer, but ggml-metal
    // pierces the iface abstraction and casts `buffer->context` straight
    // to its `ggml_metal_buffer_t` — wrapping made Metal read garbage
    // and emit "tensor 'X' buffer is nil" for every weight (kokoro
    // gibberish-audio regression). The mmap region is registered in
    // g_gpu_mmap and deliberately leaked when the buffer is freed: Metal
    // doesn't own the pages (deallocator=nil), and on macOS file-backed
    // pages can still be evicted under pressure. Process exit cleans up.
    if (mmap_loader_enabled() && !ggml_backend_is_cpu(backend)) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.buffer_from_host_ptr) {
            MappedFile mf(path, /*writable=*/true);
            if (mf.ok) {
                const size_t data_off = gguf_get_data_offset(gctx);
                char* tensor_base = (char*)mf.base + data_off;

                // Hand the entire mmap region (including GGUF header) to
                // the device. The backend's `buffer_from_host_ptr` uses
                // the size for its internal MTLBuffer view; tensor binds
                // below offset into this base.
                ggml_backend_buffer_t inner = ggml_backend_dev_buffer_from_host_ptr(dev, mf.base, mf.size,
                                                                                    /*max_tensor_size=*/0);
                if (inner) {
                    void* leaked_base = mf.base;
                    size_t leaked_size = mf.size;
                    mf.release();
                    // Hint kernel async readahead — same rationale as the
                    // CPU branch above. On Apple Silicon the unified-memory
                    // shared-storage MTLBuffer reads the same physical
                    // pages, so this readahead benefits both CPU and GPU
                    // accesses with one call.
#if !defined(_WIN32)
                    ::posix_madvise(leaked_base, leaked_size, POSIX_MADV_WILLNEED);
#endif
                    // PLAN #60c / #60f: optional preload + mlock, opt-in
                    // via env. mlock is particularly meaningful here —
                    // pinning prevents Metal's shared-storage reads from
                    // racing CPU page faults under memory pressure.
                    if (preload_enabled())
                        preload_pages(leaked_base, leaked_size);
                    if (mlock_enabled())
                        try_mlock(tag, leaked_base, leaked_size);

                    register_gpu_mmap(inner, leaked_base, leaked_size);
                    out.buf = inner;

                    // Pre-validate bounds (same rationale as CPU mmap path).
                    bool bounds_ok = true;
                    for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                        const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                        if (tid < 0)
                            continue;
                        const size_t off = gguf_get_tensor_offset(gctx, tid);
                        if (data_off + off + ggml_nbytes(t) > leaked_size) {
                            fprintf(stderr,
                                    "%s: GPU mmap bounds check failed for tensor '%s' "
                                    "(data_off=%zu + off=%zu + nbytes=%zu > file_size=%zu) — "
                                    "falling back to legacy loader\n",
                                    tag, ggml_get_name(t), data_off, off, ggml_nbytes(t), leaked_size);
                            bounds_ok = false;
                            break;
                        }
                    }
                    if (!bounds_ok) {
                        out.buf = nullptr;
                        // inner is registered in g_gpu_mmap and deliberately
                        // leaked (same as the normal teardown path).
                        // Fall through to legacy path.
                    } else {
                        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                            out.tensors[ggml_get_name(t)] = t;
                            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                            if (tid < 0)
                                continue;
                            const size_t off = gguf_get_tensor_offset(gctx, tid);
                            ggml_backend_tensor_alloc(out.buf, t, tensor_base + off);
                        }

                        gguf_free(gctx);
                        return true;
                    }
                }
                // buffer_from_host_ptr returned null — release mmap and fall
                // through to the legacy path. No warning; same rationale as
                // the CPU mmap branch.
            }
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", tag);
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    MappedFile mf(path);
    if (mf.ok) {
        // Issue #94: the legacy alloc+copy path took 30-60 s for the
        // 658 MB chatterbox-turbo T3 GGUF on slow disks, because each
        // ggml_backend_tensor_set hit a synchronous page fault on
        // its first access to the mmap source region. The zero-copy
        // mmap path already hints WILLNEED (line 526) for the same
        // reason; doing it here brings the legacy path's load time
        // back in line for users who opt out of the zero-copy path
        // with CRISPASR_GGUF_MMAP=0 (e.g. model files on network
        // mounts where mmap would SIGBUS on disconnect).
#if !defined(_WIN32)
        ::posix_madvise(mf.base, mf.size, POSIX_MADV_WILLNEED);
#endif
    }
    if (!mf.ok) {
        // Fallback: read via FILE* pread/fseek. This is the rare path —
        // most systems have working mmap. We implement it inline here so
        // models don't have to.
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            gguf_free(gctx);
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> tbuf;
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes)
                tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0)
                break;
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0)
                break;
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes)
                break;
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
    } else {
        const size_t data_off = gguf_get_data_offset(gctx);
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    gguf_free(gctx);
    return true;
}

void free_weights(WeightLoad& wl) {
    if (wl.buf) {
        ggml_backend_buffer_free(wl.buf);
        wl.buf = nullptr;
    }
    if (wl.buf_cpu) {
        ggml_backend_buffer_free(wl.buf_cpu);
        wl.buf_cpu = nullptr;
    }
    if (wl.ctx) {
        ggml_free(wl.ctx);
        wl.ctx = nullptr;
    }
    wl.tensors.clear();
}

int blk_layer_of_with_prefix(const char* tensor_name, const char* prefix) {
    if (!tensor_name || !prefix)
        return -1;
    const size_t plen = std::strlen(prefix);
    if (std::strncmp(tensor_name, prefix, plen) != 0)
        return -1;
    char* end = nullptr;
    long il = std::strtol(tensor_name + plen, &end, 10);
    if (!end || *end != '.' || il < 0)
        return -1;
    return (int)il;
}

int blk_layer_of(const char* tensor_name) {
    return blk_layer_of_with_prefix(tensor_name, "blk.");
}

bool is_gpu_tensor_with_prefix(const char* tensor_name, void* user) {
    const auto* cfg = static_cast<const LayerSplitConfig*>(user);
    const int il = blk_layer_of_with_prefix(tensor_name, cfg->prefix);
    if (il < 0)
        return true; // non-layered tensors stay on GPU
    return il < cfg->threshold;
}

bool is_gpu_tensor_blk(const char* tensor_name, void* user) {
    const int threshold = *static_cast<const int*>(user);
    const int il = blk_layer_of(tensor_name);
    if (il < 0)
        return true;
    return il < threshold;
}

bool load_weights_split(const char* path, ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, IsGpuTensor is_gpu,
                        void* user, const char* model_tag, WeightLoad& out) {
    const char* tag = model_tag ? model_tag : "core_gguf";

    if (!gpu_backend || !cpu_backend) {
        fprintf(stderr, "%s: load_weights_split requires both gpu and cpu backends\n", tag);
        return false;
    }
    if (!is_gpu) {
        fprintf(stderr, "%s: load_weights_split requires a non-null is_gpu predicate\n", tag);
        return false;
    }

    // Open metadata + create ggml_context with all tensor metadata (no_alloc).
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&out.ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
        return false;
    }

    // Pass 1: partition tensors by predicate, sum sizes per partition.
    std::vector<ggml_tensor*> gpu_tensors, cpu_tensors;
    size_t gpu_size = 0, cpu_size = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
        const char* tname = ggml_get_name(t);
        const bool to_gpu = is_gpu(tname, user);
        if (to_gpu) {
            gpu_tensors.push_back(t);
            gpu_size += ggml_nbytes(t);
        } else {
            cpu_tensors.push_back(t);
            cpu_size += ggml_nbytes(t);
        }
        out.tensors[tname] = t;
    }

    // Allocate per-partition backend buffers. Tensor alignment within the
    // buffer follows the backend buffer-type's alignment requirement;
    // pad each per-tensor offset up to that alignment.
    auto round_up = [](size_t n, size_t a) { return (n + a - 1) & ~(a - 1); };
    auto bind_partition = [&](ggml_backend_t be, const std::vector<ggml_tensor*>& tensors, size_t total,
                              ggml_backend_buffer_t* out_buf) -> bool {
        if (tensors.empty())
            return true;
        const size_t align = ggml_backend_get_alignment(be);
        // Compute final size with per-tensor alignment slack.
        size_t aligned_total = 0;
        for (ggml_tensor* t : tensors)
            aligned_total = round_up(aligned_total, align) + ggml_nbytes(t);
        (void)total;
        ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(be, aligned_total);
        if (!buf) {
            fprintf(stderr, "%s: failed to allocate %zu MiB backend buffer\n", tag, aligned_total / 1048576);
            return false;
        }
        char* base = (char*)ggml_backend_buffer_get_base(buf);
        size_t cursor = 0;
        for (ggml_tensor* t : tensors) {
            cursor = round_up(cursor, align);
            ggml_backend_tensor_alloc(buf, t, base + cursor);
            cursor += ggml_nbytes(t);
        }
        *out_buf = buf;
        return true;
    };

    if (!bind_partition(gpu_backend, gpu_tensors, gpu_size, &out.buf)) {
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }
    if (!bind_partition(cpu_backend, cpu_tensors, cpu_size, &out.buf_cpu)) {
        if (out.buf) {
            ggml_backend_buffer_free(out.buf);
            out.buf = nullptr;
        }
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // Copy tensor data from the file. Use mmap when available for zero-
    // copy where the kernel will demand-page; fall back to pread.
    MappedFile mf(path);
    if (!mf.ok) {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            free_weights(out);
            gguf_free(gctx);
            return false;
        }
        const size_t data_off = gguf_get_data_offset(gctx);
        std::vector<uint8_t> tbuf;
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes)
                tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0)
                break;
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0)
                break;
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes)
                break;
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
    } else {
        const size_t data_off = gguf_get_data_offset(gctx);
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    fprintf(stderr, "%s: weight residency: gpu=%zu MiB (%zu tensors), cpu=%zu MiB (%zu tensors)\n", tag,
            gpu_size / 1048576, gpu_tensors.size(), cpu_size / 1048576, cpu_tensors.size());

    gguf_free(gctx);
    return true;
}

// PLAN #60g: switch a previously-WILLNEED-hinted region to MADV_RANDOM.
// Used by callers (e.g., mimo-asr's transcribe loop) to tell the kernel
// "I'm done with sequential prefill access; my next reads will be
// random-order layer revisits during decode — please stop wasting IO
// on speculative readahead."
//
// We dispatch on the buffer's iface fields to detect which of our two
// mmap paths the buffer came from, since the buffer types themselves
// aren't exposed publicly. No-op if the buffer wasn't allocated through
// either path (incl. the legacy alloc+copy fallback when MMAP=0 or
// mmap failed).
void mmap_advise_random(ggml_backend_buffer_t buf) {
#if !defined(_WIN32)
    if (!buf)
        return;
    void* base = nullptr;
    size_t size = 0;
    if (buf->iface.free_buffer == mmap_buffer_free) {
        // CPU mmap path — context is mmap_buffer_ctx (our own iface).
        auto* mctx = (mmap_buffer_ctx*)buf->context;
        base = mctx->mmap_base;
        size = mctx->mmap_size;
    } else {
        // Non-CPU (Metal) mmap path — buffer is the inner backend buffer,
        // its context belongs to that backend; look the mmap region up
        // in our side-map instead.
        gpu_mmap_handle h = lookup_gpu_mmap(buf);
        base = h.base;
        size = h.size;
    }
    if (base && size > 0)
        ::posix_madvise(base, size, POSIX_MADV_RANDOM);
#else
    (void)buf;
#endif
}

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

ggml_tensor* try_get(const std::map<std::string, ggml_tensor*>& tensors, const char* name) {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

ggml_tensor* require(const std::map<std::string, ggml_tensor*>& tensors, const char* name, const char* model_tag) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "%s: required tensor '%s' not found in GGUF\n", model_tag ? model_tag : "core_gguf", name);
        return nullptr;
    }
    return it->second;
}

std::string format_layer_name(const char* fmt, int i) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i);
    return std::string(buf);
}

std::string format_layer_name(const char* fmt, int i, int j) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i, j);
    return std::string(buf);
}

} // namespace core_gguf

// ======================================================================
// BEGIN kokoro_engine.h
// ======================================================================

struct ggml_context;
struct gguf_context;

struct kokoro_context;

struct kokoro_context_params {
    int   n_threads;
    int   verbosity;
    bool  use_gpu;
    bool  flash_attn;
    float length_scale;
    bool  gen_force_metal;
};

// ======================================================================
// BEGIN kokoro_engine.cpp
// ======================================================================

// kokoro.cpp — runtime for hexgrad/Kokoro-82M and yl4579/StyleTTS2-
// LJSpeech (same architecture). The full forward pass at synthesis
// time:
//
//   phonemes (IPA, from espeak-ng or --tts-phonemes) → token IDs
//     ↓
//   text_enc (Embedding → 3× Conv1d k=5 + LN + GELU → bidir LSTM)
//                                                        → t_enc [L, 512]
//   bert (custom ALBERT-base, 12 parameter-shared layers) → bert_dur [L, 512]
//                                                        ↓
//   ref_s = voice_pack[L-1, 0, :]   split [pred 0:128 | dec 128:256]
//                                                        ↓
//   ProsodyPredictor (dur_enc 3× alternating LSTM/AdaLN, post-encoder
//   LSTM, dur_proj, shared LSTM, F0/N AdainResBlk1d stacks) →
//   durations + F0 + N
//                                                        ↓
//   align(t_enc, durations) → en [T_frames, 512]
//                                                        ↓
//   iSTFTNet decoder (encode + 4× decode AdainResBlk1d + asr_res +
//   F0_conv + N_conv → Generator with HnNSF source + 2× upsample +
//   noise injection + 6 resblocks averaged + conv_post → 22 channels)
//                                                        ↓
//   iSTFT on CPU (n_fft=20, hop=5, Hann) → 24 kHz audio
//
// This file is the M1 skeleton: GGUF two-pass load, hparams + vocab,
// voice-pack secondary loader, scheduler setup, and stub C ABI for the
// synth + stage extractors. Subsequent milestones fill in the forward
// pass.




namespace {

bool env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0;
}

// Hyperparameters read from `kokoro.*` and `kokoro.plbert.*` GGUF KV.
struct kokoro_hp {
    // Top-level
    uint32_t hidden_dim = 512;
    uint32_t style_dim = 128;
    uint32_t max_dur = 50;
    uint32_t n_token = 178;
    uint32_t n_mels = 80; // unused by the synthesis path; kept for completeness
    uint32_t n_layer = 3; // duration-encoder depth
    uint32_t text_enc_k = 5;
    uint32_t sample_rate = 24000;
    uint32_t vocab_size = 178;

    // PL-BERT (custom 12-layer parameter-shared ALBERT)
    uint32_t plbert_embd_size = 128;
    uint32_t plbert_hidden = 768;
    uint32_t plbert_n_layers = 12;
    uint32_t plbert_n_heads = 12;
    uint32_t plbert_ff = 2048;
    uint32_t plbert_max_pos = 512;
    uint32_t plbert_vocab_size = 178;

    // iSTFTNet
    uint32_t istft_init_ch = 512;
    uint32_t istft_n_fft = 20;
    uint32_t istft_hop = 5;
    uint32_t istft_n_dilations = 3;                    // entries per resblock
    std::vector<uint32_t> istft_upsample_rates;        // [10, 6]
    std::vector<uint32_t> istft_upsample_kernel_sizes; // [20, 12]
    std::vector<uint32_t> istft_resblock_kernel_sizes; // [3, 7, 11]
    std::vector<uint32_t> istft_resblock_dilations;    // flat, length 3*n_dilations
};

struct kokoro_vocab {
    std::vector<std::string> id_to_token;                 // size = vocab_size
    std::unordered_map<std::string, int32_t> token_to_id; // lookup for tokenizer
    int32_t pad_id = 0;                                   // index of "$" (Kokoro/StyleTTS2 pad)
};

struct kokoro_voice_pack {
    std::string name;
    uint32_t max_phonemes = 0;
    uint32_t style_dim = 0;
    ggml_tensor* pack = nullptr; // (max_phon, 1, 256) F32 — owned by vp_ctx_w/vp_buf_w
    ggml_context* vp_ctx_w = nullptr;
    ggml_backend_buffer_t vp_buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// Bounded LRU phoneme cache removed — RapidSpeech adapter only feeds the
// engine pre-tokenized phoneme IDs (no G2P, so nothing to cache).

} // namespace

struct kokoro_context {
    kokoro_context_params params{};
    int n_threads = 4;

    kokoro_hp hp;
    kokoro_vocab vocab;

    // Backends. `backend` is the user's choice (Metal/CUDA/CPU); `backend_cpu`
    // is always present. `gen_backend` defaults to the GPU backend now that
    // upstream ggml PR #1477 (commit a056a26f) eliminated the stride-10
    // ConvTranspose1d watchdog hang on Apple M1. Set KOKORO_GEN_CPU=1 to
    // force the legacy CPU path.
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t gen_backend = nullptr; // either backend or backend_cpu

    // Schedulers. `sched` runs everything except the iSTFTNet generator;
    // `gen_sched` runs the generator alone (GPU by default; CPU when
    // KOKORO_GEN_CPU=1). `bert_sched` aliases `sched` — kept as a named
    // alias so the BERT graph code can stay decoupled.
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_sched_t gen_sched = nullptr;

    // Primary weights (talker GGUF). Tensors stay resident here for the
    // life of the context.
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;

    // Voice pack (secondary GGUF).
    kokoro_voice_pack vp;
    bool vp_loaded = false;

    // Whether this context owns gguf_data/ctx_w and backends (the legacy
    // kokoro_init_from_file path) or borrows them from rs_context_t (the
    // new kokoro_init_from_rs_context path used by the adapter).
    bool owns_gguf = true;
    bool owns_backend = true;

#ifdef RS_USE_METAL_KOKORO
    // Custom Metal ConvTranspose1d kernel for the iSTFTNet generator's two
    // upsample ops (simdgroup-over-IC; 4-5x faster than ggml-metal's default).
    // When valid, the generator graph is split into 3 sub-graphs with custom
    // kernel calls between them. When null (Metal init failed, or CPU build),
    // the original single-graph path is used.
    std::unique_ptr<KokoroMetalConvT1D> metal_convt;
#endif

    // Per-node activation observer for importance-matrix collection. Set via
    // KokoroModel::set_imatrix_callback; consumed by kokoro_sched_compute.
    // Empty function ⇒ no instrumentation (production path).
    std::function<void(struct ggml_tensor *)> imatrix_cb;
};

// Run a graph through `sched`, optionally instrumented with the kokoro_context's
// imatrix callback. Mirrors cv3_sched_compute (cosyvoice3.h:22): arming the
// sched's per-node eval callback keeps src0/src1 live around each compute step,
// which is required for activation capture (the sched's buffer reuse may
// overwrite src1 with downstream output once compute returns). Production
// callers pay only one branch (`armed`) when imatrix_cb is empty.
static inline enum ggml_status kokoro_sched_compute(
    kokoro_context* c, ggml_backend_sched_t sched, struct ggml_cgraph* gf) {
    auto* cb = &c->imatrix_cb;
    const bool armed = cb && (bool)*cb;
    if (armed) {
        auto trampoline = [](struct ggml_tensor* t, bool ask, void* ud) -> bool {
            if (ask) return true;
            auto* fn = static_cast<std::function<void(struct ggml_tensor*)>*>(ud);
            (*fn)(t);
            return true;
        };
        ggml_backend_sched_set_eval_callback(sched, trampoline, cb);
    }
    enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
    if (armed) {
        ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    }
    return status;
}

namespace {

ggml_tensor* try_get(const kokoro_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return it == c->tensors.end() ? nullptr : it->second;
}

ggml_tensor* require(const kokoro_context* c, const char* name) {
    auto* t = try_get(c, name);
    if (!t) {
        fprintf(stderr, "kokoro: required tensor missing: %s\n", name);
    }
    return t;
}

// Stage profiler. Enable with RS_KOKORO_PROFILE=1.
// Accumulates per-stage time across the whole synth call (a single utterance
// re-runs predictor/text_enc/f0n/decoder_body up to 4× — the breakdown shows
// where deduplication would pay off).
struct kokoro_profile_bag {
    struct entry { std::string name; int64_t us; int calls; };
    std::vector<entry> stages;
    bool enabled = false;
    void init() {
        const char* s = std::getenv("RS_KOKORO_PROFILE");
        enabled = (s && *s && std::string(s) != "0");
        stages.clear();
    }
    void add(const char* name, int64_t us) {
        if (!enabled) return;
        for (auto& p : stages) {
            if (p.name == name) { p.us += us; p.calls += 1; return; }
        }
        stages.push_back({name, us, 1});
    }
    void dump() {
        if (!enabled) return;
        int64_t total = 0;
        for (auto& p : stages) total += p.us;
        fprintf(stderr, "[kokoro-prof] ==== stage breakdown ====\n");
        for (auto& p : stages) {
            double ms = p.us / 1000.0;
            double pct = total ? (100.0 * p.us / total) : 0.0;
            fprintf(stderr, "[kokoro-prof] %-22s %3d× %8.2f ms (%5.1f%%)\n", p.name.c_str(), p.calls, ms, pct);
        }
        fprintf(stderr, "[kokoro-prof] %-22s     %8.2f ms\n", "SUM", total / 1000.0);
    }
};
static kokoro_profile_bag g_kokoro_prof;

struct kokoro_profile_scope {
    const char* name;
    int64_t t0;
    kokoro_profile_scope(const char* n) : name(n), t0(g_kokoro_prof.enabled ? ggml_time_us() : 0) {}
    ~kokoro_profile_scope() {
        if (g_kokoro_prof.enabled) g_kokoro_prof.add(name, ggml_time_us() - t0);
    }
};
#define KOKORO_PROFILE(name) kokoro_profile_scope _kokoro_prof_scope(name)

// Utterance-level output cache. Many stage graphs produce multiple named
// outputs that the engine fetches via separate run_*(stage_name) calls — the
// naive code path recomputes the whole graph for each fetch. With this cache,
// the first call computes the graph once and stashes ALL named outputs; later
// calls (and duplicate calls of the same output across mag/phase passes) hit
// the cache. Cleared at the top of kokoro_run_audio.
struct kokoro_stage_cache {
    struct entry { std::string key; std::vector<float> data; };
    std::vector<entry> entries;
    void reset() { entries.clear(); }
    bool has(const std::string& key) const {
        for (auto& e : entries) if (e.key == key) return true;
        return false;
    }
    float* get_copy(const std::string& key, int* out_n) const {
        for (auto& e : entries) {
            if (e.key == key) {
                *out_n = (int)e.data.size();
                float* r = (float*)std::malloc(e.data.size() * sizeof(float));
                if (!r) return nullptr;
                std::memcpy(r, e.data.data(), e.data.size() * sizeof(float));
                return r;
            }
        }
        *out_n = 0;
        return nullptr;
    }
    void put(const std::string& key, const float* data, int n) {
        entries.push_back({key, std::vector<float>(data, data + n)});
    }
};
static kokoro_stage_cache g_kokoro_cache;

// Read a uint32 array from GGUF metadata into a std::vector. Empty result if
// the key is absent or the array element type is unexpected.
std::vector<uint32_t> kv_u32_array(gguf_context* g, const char* key) {
    std::vector<uint32_t> out;
    const int k = gguf_find_key(g, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY)
        return out;
    const gguf_type elem = gguf_get_arr_type(g, k);
    const int n = gguf_get_arr_n(g, k);
    out.reserve((size_t)n);
    if (elem == GGUF_TYPE_UINT32 || elem == GGUF_TYPE_INT32) {
        const auto* d = (const uint32_t*)gguf_get_arr_data(g, k);
        out.assign(d, d + n);
    } else if (elem == GGUF_TYPE_UINT64 || elem == GGUF_TYPE_INT64) {
        const auto* d = (const uint64_t*)gguf_get_arr_data(g, k);
        for (int i = 0; i < n; i++)
            out.push_back((uint32_t)d[i]);
    }
    return out;
}

// Sanity-check the loaded weights. Soft-validates a representative
// subset rather than enumerating every tensor (the full list is 459
// names; load_weights already provides the full map).
bool sanity_check_weights(const kokoro_context* c) {
    const char* must_have[] = {
        "bert.embd.tok.weight",       "bert.embd_proj.weight",
        "bert.attn_q.weight",         "bert.attn_q.bias",
        "bert.ffn_up.weight",         "bert.pooler.weight",
        "bert_proj.weight",           "bert_proj.bias",
        "text_enc.embd.weight",       "text_enc.cnn.0.conv.weight",
        "text_enc.lstm.weight_ih_l0", "text_enc.lstm.weight_ih_l0_reverse",
        "pred.F0_proj.weight",        "pred.N_proj.weight",
        "dec.gen.conv_post.weight",
    };
    bool ok = true;
    for (const char* name : must_have) {
        if (c->tensors.find(name) == c->tensors.end()) {
            fprintf(stderr, "kokoro: required tensor missing: %s\n", name);
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// M3 — BERT (PL-BERT, parameter-shared 12-layer ALBERT)
//
// Architecture (from kokoro/custom_albert.py + the AlbertConfig defaults
// inherited by PL-BERT):
//   * Embeddings: tok(178→128) + pos(512→128) + tt(2→128, idx 0)  → LN
//   * embd_proj: Linear 128 → 768
//   * 12× shared layers (post-norm):
//       y = LN(x + self_attn(x), attn_ln, eps=1e-12)
//       y = LN(y + GELU_new(ffn_up(y))→ffn_down, ffn_ln, eps=1e-12)
//   * (pooler exists in the GGUF but is unused by the synthesis path —
//     Kokoro reads `last_hidden_state`, not `pooler_output`.)
//   * bert_proj: Linear 768 → 512, applied per-token.
//
// The "bert_pooler_out" stage name in kokoro.h is legacy from the plan;
// the value at this stage is the per-token last_hidden_state of shape
// (768, L), not a pooled vector. Kept for ABI stability.
// ---------------------------------------------------------------------------

static const float kBertLayerNormEps = 1e-12f;

// Build the BERT graph for L tokens. The graph has two int32 inputs
// ("ids" and "positions", both length L) and exposes two outputs by
// name: "bert_pooler_out" (768, L) and "bert_proj_out" (512, L).
static ggml_cgraph* kokoro_build_graph_bert(kokoro_context* c, int L) {
    const auto& hp = c->hp;
    const int D_emb = (int)hp.plbert_embd_size; // 128
    const int D = (int)hp.plbert_hidden;        // 768
    const int n_h = (int)hp.plbert_n_heads;     // 12
    const int n_lay = (int)hp.plbert_n_layers;  // 12 shared
    const int hd = D / n_h;                     // 64
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // ~30 ops per layer × 12 + embd/proj/pooler = a few hundred. 1024 is plenty.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ---- Embeddings ----
    ggml_tensor* tok_w = require(c, "bert.embd.tok.weight");
    ggml_tensor* pos_w = require(c, "bert.embd.pos.weight");
    ggml_tensor* tt_w = require(c, "bert.embd.tt.weight");
    ggml_tensor* eln_w = require(c, "bert.embd.ln.weight");
    ggml_tensor* eln_b = require(c, "bert.embd.ln.bias");
    ggml_tensor* ep_w = require(c, "bert.embd_proj.weight");
    ggml_tensor* ep_b = require(c, "bert.embd_proj.bias");

    ggml_tensor* tok_emb = ggml_get_rows(ctx0, tok_w, ids);       // (D_emb, L) F32
    ggml_tensor* pos_emb = ggml_get_rows(ctx0, pos_w, positions); // (D_emb, L) F32
    // token_type_ids defaults to zero — slice tt_w[:, 0:1] and broadcast on L.
    ggml_tensor* tt_view = ggml_view_2d(ctx0, tt_w, D_emb, 1, tt_w->nb[1], 0); // (D_emb, 1) F16
    ggml_tensor* tt_emb = ggml_cast(ctx0, tt_view, GGML_TYPE_F32);             // (D_emb, 1) F32

    ggml_tensor* x = ggml_add(ctx0, tok_emb, pos_emb);
    x = ggml_add(ctx0, x, tt_emb);

    // Embedding LayerNorm (eps=1e-12).
    x = ggml_norm(ctx0, x, kBertLayerNormEps);
    x = ggml_mul(ctx0, x, eln_w);
    x = ggml_add(ctx0, x, eln_b);

    // embd_proj: Linear D_emb → D
    x = ggml_mul_mat(ctx0, ep_w, x);
    x = ggml_add(ctx0, x, ep_b); // (D, L)

    // ---- 12 shared ALBERT layers ----
    ggml_tensor* q_w = require(c, "bert.attn_q.weight");
    ggml_tensor* q_b = require(c, "bert.attn_q.bias");
    ggml_tensor* k_w = require(c, "bert.attn_k.weight");
    ggml_tensor* k_b = require(c, "bert.attn_k.bias");
    ggml_tensor* v_w = require(c, "bert.attn_v.weight");
    ggml_tensor* v_b = require(c, "bert.attn_v.bias");
    ggml_tensor* o_w = require(c, "bert.attn_o.weight");
    ggml_tensor* o_b = require(c, "bert.attn_o.bias");
    ggml_tensor* aln_w = require(c, "bert.attn_ln.weight");
    ggml_tensor* aln_b = require(c, "bert.attn_ln.bias");
    ggml_tensor* fu_w = require(c, "bert.ffn_up.weight");
    ggml_tensor* fu_b = require(c, "bert.ffn_up.bias");
    ggml_tensor* fd_w = require(c, "bert.ffn_down.weight");
    ggml_tensor* fd_b = require(c, "bert.ffn_down.bias");
    ggml_tensor* fln_w = require(c, "bert.ffn_ln.weight");
    ggml_tensor* fln_b = require(c, "bert.ffn_ln.bias");

    core_attn::EncoderSelfAttnParams eap{};
    eap.n_heads = n_h;
    eap.n_kv_heads = n_h; // MHA
    eap.head_dim = hd;
    eap.n_kv_grp = 1;
    eap.attn_scale = attn_scale;
    eap.n_ctx_orig = 0;
    eap.rope_theta = 0.0f;
    eap.permute_cont = true;

    for (int il = 0; il < n_lay; il++) {
        // Self-attention with biased Q/K/V/O, no RoPE, no mask.
        ggml_tensor* attn_out = core_attn::encoder_self_attn(ctx0, x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
                                                             /*positions*/ nullptr, /*mask*/ nullptr, eap);
        x = ggml_add(ctx0, x, attn_out);
        x = ggml_norm(ctx0, x, kBertLayerNormEps);
        x = ggml_mul(ctx0, x, aln_w);
        x = ggml_add(ctx0, x, aln_b);

        // FFN: up → GELU(tanh approx, "gelu_new") → down (with biases).
        ggml_tensor* ffn = ggml_mul_mat(ctx0, fu_w, x);
        ffn = ggml_add(ctx0, ffn, fu_b);
        ffn = ggml_gelu(ctx0, ffn); // tanh-approx == ALBERT "gelu_new"
        ffn = ggml_mul_mat(ctx0, fd_w, ffn);
        ffn = ggml_add(ctx0, ffn, fd_b);
        x = ggml_add(ctx0, x, ffn);
        x = ggml_norm(ctx0, x, kBertLayerNormEps);
        x = ggml_mul(ctx0, x, fln_w);
        x = ggml_add(ctx0, x, fln_b);
    }

    // Per-token last_hidden_state — exposed for diff-harness as the
    // "bert_pooler_out" stage (legacy name; not actually pooled).
    ggml_tensor* seq = x; // (D, L)
    seq = ggml_cont(ctx0, seq);
    ggml_set_name(seq, "bert_pooler_out");
    ggml_set_output(seq);
    ggml_build_forward_expand(gf, seq);

    // bert_proj: Linear 768 → 512, applied per-token.
    ggml_tensor* bp_w = require(c, "bert_proj.weight");
    ggml_tensor* bp_b = require(c, "bert_proj.bias");
    ggml_tensor* proj = ggml_mul_mat(ctx0, bp_w, seq); // (512, L)
    proj = ggml_add(ctx0, proj, bp_b);
    ggml_set_name(proj, "bert_proj_out");
    ggml_set_output(proj);
    ggml_build_forward_expand(gf, proj);

    ggml_free(ctx0);
    return gf;
}

// Wrap raw phoneme ids with the StyleTTS2 pad convention:
//   wrapped = [pad_id, raw_ids..., pad_id]
// The reference Python (KModel.forward) does this before feeding into
// BERT / text_enc / predictor. The pad_id is the StyleTTS2 "$" token
// (vocab index 0). All downstream stages see length L+2.
static std::vector<int32_t> kokoro_pad_wrap_ids(const kokoro_context* c, const int32_t* raw, int n_raw) {
    std::vector<int32_t> w;
    w.reserve((size_t)n_raw + 2);
    w.push_back(c->vocab.pad_id);
    w.insert(w.end(), raw, raw + n_raw);
    w.push_back(c->vocab.pad_id);
    return w;
}

// ---------------------------------------------------------------------------
// AdaLayerNorm (used by the predictor's DurationEncoder).
//
// Reference: kokoro/modules.py AdaLayerNorm (eps=1e-5).
//   gamma, beta = chunk(fc(s), 2)        # fc: Linear(style_dim, 2*C)
//   y = (1 + gamma) * LayerNorm(x) + beta
//
// We compute (1 + γ)·x + β as x + x·γ + β to avoid materialising a "1.0"
// tensor for the (1 + γ) term. Mathematically identical, one fewer
// constant input.
//
// Inputs:
//   x      ne = (C, T)  F32, channel-major (LN normalises over ne[0])
//   style  ne = (sty, 1) F32 — predictor reference vector
//   fc_w   ne = (sty, 2C) F16 — Linear weight
//   fc_b   ne = (2C,) F32 — Linear bias
//
// Output: (C, T) F32.
static const float kAdaLnEps = 1e-5f;

static inline ggml_tensor* kokoro_adaln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, ggml_tensor* fc_w,
                                        ggml_tensor* fc_b) {
    const int C = (int)x->ne[0];
    // h = fc(s) → (2C, 1)
    ggml_tensor* h = ggml_mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    // chunk along ne[0]: γ ∈ [0, C), β ∈ [C, 2C).
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)0 * C * ts);
    ggml_tensor* beta = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)1 * C * ts);

    ggml_tensor* normed = ggml_norm(ctx, x, kAdaLnEps);
    // x*γ + x → broadcast γ (C, 1) over T
    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    out = ggml_add(ctx, out, beta);
    return out;
}

// ---------------------------------------------------------------------------
// M4 — TextEncoder
//
// Reference: kokoro/modules.py TextEncoder.
//   Embedding(178, 512) → 3× Sequential(
//       Conv1d(512, 512, k=5, pad=2),
//       LayerNorm(512)        # over channel dim, ε=1e-5
//       LeakyReLU(0.2),
//       Dropout(0.2)          # no-op at inference
//   )
//   bidirectional LSTM(in=512, hidden=256) → output (B, D=512, T) after
//   the final transpose.
//
// Input: pad-wrapped phoneme ids (length L).
// Output stage `text_enc_out`: ne = (512, L)  F32.
// ---------------------------------------------------------------------------

static const float kTextEncLayerNormEps = 1e-5f;
static const float kTextEncLeakySlope = 0.2f;

static ggml_cgraph* kokoro_build_graph_text_enc(kokoro_context* c, int L) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_dim; // 512
    const int H_lstm = D / 2;         // 256 (bidir → 2*H = D)
    const int K = (int)hp.text_enc_k; // 5

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // 3 conv blocks + 1 bidir LSTM at modest L → a few thousand nodes max.
    // For L up to plbert_max_pos + 2 = 514, the LSTM dominates: 2 directions
    // * L * ~15 ops/step ≈ 15400 nodes. 32k headroom.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    // ---- Embedding: (178, 512) → (D, L) ----
    ggml_tensor* emb_w = require(c, "text_enc.embd.weight");
    ggml_tensor* x = ggml_get_rows(ctx0, emb_w, ids); // (D, L) F32

    // ---- 3× Conv1d(k=5, pad=2) + LN(channel) + LeakyReLU(0.2) ----
    // Layout flow per block:
    //   in:        (D, L)         ne[0]=D, ne[1]=L
    //   transpose: (L, D)         for ggml_conv_1d which expects (T, C_in)
    //   conv1d:    (L, D, 1)      output is 3D (OL, OC, B=1)
    //   add bias:  bias reshaped (1, D, 1) broadcasts on L and B
    //   reshape:   (L, D)         drop trailing 1
    //   transpose: (D, L)         channel-major for ggml_norm
    //   norm:      (D, L)         normalises over ne[0]=D
    //   * gamma + bias beta:      (D, L)
    //   leaky:     (D, L)
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    for (int il = 0; il < 3; il++) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.conv.weight", il);
        ggml_tensor* cw = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.conv.bias", il);
        ggml_tensor* cb = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.ln.gamma", il);
        ggml_tensor* lg = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.ln.beta", il);
        ggml_tensor* lb = require(c, nm);

        // (D, L) → (L, D) for conv input layout.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));                                          // (L, D)
        x = kokoro_conv_1d_2d(ctx0, cw, x, K, /*Cin*/ D, /*s=*/1, /*p=*/(K - 1) / 2, /*d=*/1); // (L, D, 1)
        x = ggml_add(ctx0, x, bias_1d(cb));
        // Drop the trailing batch dim → (L, D).
        x = ggml_reshape_2d(ctx0, x, L, D);

        // (L, D) → (D, L) for channel-major LN.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // (D, L)
        x = ggml_norm(ctx0, x, kTextEncLayerNormEps);
        x = ggml_mul(ctx0, x, lg);
        x = ggml_add(ctx0, x, lb);
        x = ggml_leaky_relu(ctx0, x, kTextEncLeakySlope, /*inplace=*/false);
    }

    // ---- Bidirectional LSTM ----
    // Input shape requirement: (in, T) = (D=512, L). x is already (D, L). ✓
    ggml_tensor* W_ih_f = require(c, "text_enc.lstm.weight_ih_l0");
    ggml_tensor* W_hh_f = require(c, "text_enc.lstm.weight_hh_l0");
    ggml_tensor* b_ih_f = require(c, "text_enc.lstm.bias_ih_l0");
    ggml_tensor* b_hh_f = require(c, "text_enc.lstm.bias_hh_l0");
    ggml_tensor* W_ih_r = require(c, "text_enc.lstm.weight_ih_l0_reverse");
    ggml_tensor* W_hh_r = require(c, "text_enc.lstm.weight_hh_l0_reverse");
    ggml_tensor* b_ih_r = require(c, "text_enc.lstm.bias_ih_l0_reverse");
    ggml_tensor* b_hh_r = require(c, "text_enc.lstm.bias_hh_l0_reverse");

    ggml_tensor* lstm_out =
        core_lstm::lstm_bidir(ctx0, gf, x, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                              H_lstm); // (2H = D, L)

    ggml_set_name(lstm_out, "text_enc_out");
    ggml_set_output(lstm_out);
    ggml_build_forward_expand(gf, lstm_out);

    ggml_free(ctx0);
    return gf;
}

// Run text_enc and copy the named stage back to a malloc'd float buffer.
// Pad-wraps the raw ids before computing.
static float* kokoro_run_text_enc(kokoro_context* c, const int32_t* raw_ids, int n_raw, const char* stage_name,
                                  int* out_n) {
    KOKORO_PROFILE("text_enc");
    if (out_n)
        *out_n = 0;
    if (n_raw <= 0)
        return nullptr;
    {
        std::string key = std::string("text_enc:") + stage_name;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }
    std::vector<int32_t> padded = kokoro_pad_wrap_ids(c, raw_ids, n_raw);
    const int L = (int)padded.size();
    if (L > (int)c->hp.plbert_max_pos) {
        fprintf(stderr, "kokoro: text_enc L_padded=%d exceeds max=%u\n", L, c->hp.plbert_max_pos);
        return nullptr;
    }

    ggml_cgraph* gf = kokoro_build_graph_text_enc(c, L);
    if (!gf)
        return nullptr;

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for text_enc\n");
        return nullptr;
    }

    ggml_tensor* in_ids = ggml_graph_get_tensor(gf, "ids");
    ggml_backend_tensor_set(in_ids, padded.data(), 0, (size_t)L * sizeof(int32_t));

    if (kokoro_sched_compute(c, c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: text_enc graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "kokoro: text_enc graph missing output '%s'\n", stage_name);
        return nullptr;
    }

    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    g_kokoro_cache.put(std::string("text_enc:") + stage_name, r, (int)n_floats);
    if (out_n)
        *out_n = (int)n_floats;
    return r;
}

// Run the BERT graph and copy the named stage tensor back to a malloc'd
// float buffer. Returns nullptr on failure. *out_n is set to the total
// number of float elements (D × L_padded) on success. The input `ids`
// are RAW phoneme ids; this function adds the StyleTTS2 pad-wrap before
// computing, so the output corresponds to L+2 tokens.
static float* kokoro_run_bert(kokoro_context* c, const int32_t* raw_ids, int n_raw, const char* stage_name,
                              int* out_n) {
    KOKORO_PROFILE("bert");
    if (out_n)
        *out_n = 0;
    if (n_raw <= 0) {
        fprintf(stderr, "kokoro: bert needs L > 0 (got %d)\n", n_raw);
        return nullptr;
    }
    {
        std::string key = std::string("bert:") + stage_name;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }
    std::vector<int32_t> padded = kokoro_pad_wrap_ids(c, raw_ids, n_raw);
    const int L = (int)padded.size();
    if (L > (int)c->hp.plbert_max_pos) {
        fprintf(stderr, "kokoro: bert L_padded=%d exceeds max_position_embeddings=%u\n", L, c->hp.plbert_max_pos);
        return nullptr;
    }

    ggml_cgraph* gf = kokoro_build_graph_bert(c, L);
    if (!gf)
        return nullptr;

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for bert\n");
        return nullptr;
    }

    // Fill inputs.
    ggml_tensor* in_ids = ggml_graph_get_tensor(gf, "ids");
    ggml_tensor* in_pos = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(in_ids, padded.data(), 0, (size_t)L * sizeof(int32_t));

    std::vector<int32_t> positions(L);
    for (int i = 0; i < L; i++)
        positions[i] = i;
    ggml_backend_tensor_set(in_pos, positions.data(), 0, (size_t)L * sizeof(int32_t));

    if (kokoro_sched_compute(c, c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: bert graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "kokoro: bert graph missing output '%s'\n", stage_name);
        return nullptr;
    }

    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    g_kokoro_cache.put(std::string("bert:") + stage_name, r, (int)n_floats);
    if (out_n)
        *out_n = (int)n_floats;
    return r;
}

// ---------------------------------------------------------------------------
// M6 — Alignment (CPU)
//
// Reference (model.py:110-114):
//   indices = repeat_interleave(arange(L), pred_dur)
//   aln[indices, arange(T_frames)] = 1
//   en = d.transpose(-1, -2) @ aln       # (640, T_frames)
//
// The matmul is just a column-repeat: en[:, j] = d[:, indices[j]]. We
// implement that directly to avoid materialising the (L, T_frames) one-hot
// alignment matrix.
//
// features:  (D, L) F32 input
// durations: (L,)   integer counts
// Returns malloc'd (D, T_frames) F32 buffer with T_frames = sum(durations).
// ---------------------------------------------------------------------------

static inline float* kokoro_align_repeat(const float* features, int D, int L, const int* durations, int* out_T_frames) {
    return core_align::repeat_interleave(features, D, L, durations, out_T_frames);
}

// ---------------------------------------------------------------------------
// AdaIN1d (used by AdainResBlk1d in M5b predictor F0/N stacks and M7
// decoder).
//
// Reference: kokoro/istftnet.py AdaIN1d.
//   InstanceNorm1d(C, affine=True) but the affine γ'/β' were *not* trained
//   meaningfully (kokoro author note), so the converter dropped them. We
//   only have fc.weight (sty, 2C) and fc.bias (2C,).
//
// Math:
//   gamma, beta = chunk(fc(s), 2)
//   y = (1 + gamma) * InstanceNorm1d(x) + beta
//
// InstanceNorm1d on (C, T) layout: per-channel mean/var across T. ggml_norm
// normalises along ne[0]; with (T, C) layout that's per-channel-along-T,
// which is exactly instance norm 1D. So we transpose, norm, transpose back.
// ---------------------------------------------------------------------------

static const float kAdaIn1dEps = 1e-5f; // PyTorch InstanceNorm1d default

static inline ggml_tensor* kokoro_adain1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style, ggml_tensor* fc_w,
                                          ggml_tensor* fc_b, const char* dbg_prefix = nullptr) {
    const int C = (int)x->ne[0];

    // Instance norm: transpose (C, T) → (T, C); ggml_norm normalises
    // along ne[0]=T per other dim ⇒ per-channel mean+var along T;
    // transpose back. NOTE: ggml-metal's kernel_norm_fuse_impl had a
    // cross-simdgroup reduction bug that produced wrong per-row mean
    // and variance for short T (specifically T values where the last
    // simdgroup ended up with ≤ a few active threads in the prior
    // parallel-sum loop, e.g. T=65 in the kokoro AdaIN1d). The bug
    // cascaded through AdaIN → conv → AdaIN into garbage audio for
    // short utterances ("hello world"). Fixed by the CrispASR patch
    // in ggml/src/ggml-metal/ggml-metal.metal (search for
    // "serial reduction by thread 0") — that patch MUST stay
    // co-versioned with this code; if you bump ggml without
    // re-applying it, kokoro short-input audio on Metal regresses.
    // See tests/test_metal_norm_repro.cpp for a standalone repro.
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_pre_norm_TC", dbg_prefix);
        ggml_set_name(xt, nm);
        ggml_set_output(xt);
    }
    xt = ggml_norm(ctx, xt, kAdaIn1dEps);
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_post_norm_TC", dbg_prefix);
        ggml_set_name(xt, nm);
        ggml_set_output(xt);
    }
    ggml_tensor* normed = ggml_cont(ctx, ggml_transpose(ctx, xt)); // (C, T)
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_normed", dbg_prefix);
        ggml_set_name(normed, nm);
        ggml_set_output(normed);
    }

    // gamma, beta from fc(s).
    ggml_tensor* h = ggml_mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_h", dbg_prefix);
        ggml_set_name(h, nm);
        ggml_set_output(h);
    }
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)0 * C * ts);
    ggml_tensor* beta = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)1 * C * ts);

    // (1 + γ) * normed + β  →  normed + normed*γ + β  (saves the "1" tensor).
    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_xgamma", dbg_prefix);
        ggml_set_name(x_gamma, nm);
        ggml_set_output(x_gamma);
    }
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_normed_plus_xgamma", dbg_prefix);
        ggml_set_name(out, nm);
        ggml_set_output(out);
    }
    out = ggml_add(ctx, out, beta);
    if (dbg_prefix) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "%s_out", dbg_prefix);
        ggml_set_name(out, nm);
        ggml_set_output(out);
    }
    return out;
}

// Thin alias for the depthwise ConvTranspose1d pool layer used by
// every Kokoro AdainResBlk1d with `upsample=True` (predictor F0[1] /
// N[1] and decoder.decode[3]). See core_convt::convt1d_depthwise_2x_k3
// for the (k=3, s=2, p=1, op=1) derivation, including the wrong-end
// trap that the M11 diff harness caught.
static inline ggml_tensor* kokoro_pool_2x_depthwise(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_kernel,
                                                    ggml_tensor* w_bias) {
    return core_convt::convt1d_depthwise_2x_k3(ctx, x, w_kernel, w_bias);
}

// ---------------------------------------------------------------------------
// AdainResBlk1d (predictor F0/N stacks).
//
// Reference: kokoro/istftnet.py AdainResBlk1d (note: M7 decoder uses a
// *different* class AdaINResBlock1 which has Snake-α; this one uses
// LeakyReLU(0.2)).
//
// Forward:
//   residual = x
//   x' = norm1(residual)              # AdaIN1d on dim_in
//   x' = LeakyReLU(0.2)(x')
//   x' = pool(x')                     # ConvTranspose1d depthwise 2× if upsample, else identity
//   x' = conv1(x')                    # Conv1d(dim_in→dim_out, k=3, pad=1)
//   x' = norm2(x')                    # AdaIN1d on dim_out
//   x' = LeakyReLU(0.2)(x')
//   x' = conv2(x')                    # Conv1d(dim_out→dim_out, k=3, pad=1)
//   shortcut = upsample(residual)     # F.interpolate(2x, nearest) if upsample
//   shortcut = conv1x1(shortcut)      # if learned_sc (dim_in != dim_out)
//   out = (x' + shortcut) / sqrt(2)
//
// Pass nullptr for `pool_w/pool_b` to skip pool (upsample=False blocks),
// and nullptr for `conv1x1_w` to skip the learned shortcut conv (when
// dim_in == dim_out).
// ---------------------------------------------------------------------------

static const float kAdainLeakySlope = 0.2f;

static inline ggml_tensor* kokoro_adain_resblk(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                               ggml_tensor* adain1_w, ggml_tensor* adain1_b, ggml_tensor* adain2_w,
                                               ggml_tensor* adain2_b, ggml_tensor* conv1_w, ggml_tensor* conv1_b,
                                               ggml_tensor* conv2_w, ggml_tensor* conv2_b, ggml_tensor* pool_w,
                                               ggml_tensor* pool_b, ggml_tensor* conv1x1_w,
                                               const char* dbg_prefix = nullptr) {
    const bool upsample = (pool_w != nullptr);
    const int dim_in = (int)x->ne[0];
    (void)dim_in;

    // ---- Residual path ----
    char nm[64];
    auto tag = [&](ggml_tensor* t, const char* suffix) {
        if (!dbg_prefix)
            return;
        std::snprintf(nm, sizeof(nm), "%s_%s", dbg_prefix, suffix);
        ggml_set_name(t, nm);
        ggml_set_output(t);
    };
    char ad1_prefix_buf[80];
    const char* ad1_prefix = nullptr;
    if (dbg_prefix) {
        std::snprintf(ad1_prefix_buf, sizeof(ad1_prefix_buf), "%s_adain1", dbg_prefix);
        ad1_prefix = ad1_prefix_buf;
    }
    ggml_tensor* xt = kokoro_adain1d(ctx, x, style, adain1_w, adain1_b, ad1_prefix); // (Cin, T)
    xt = ggml_leaky_relu(ctx, xt, kAdainLeakySlope, /*inplace=*/false);
    tag(xt, "after_lr1");

    if (upsample) {
        xt = kokoro_pool_2x_depthwise(ctx, xt, pool_w, pool_b); // (Cin, 2T)
        tag(xt, "after_pool");
    }

    // conv1: Conv1d(dim_in → dim_out, k=3, pad=1). Layout flow:
    //   (Cin, T') → transpose → (T', Cin) → ggml_conv_1d → (T', Cout, 1) → squeeze → (T', Cout) → transpose → (Cout, T')
    auto conv_k3 = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) -> ggml_tensor* {
        const int Tin = (int)in->ne[1];
        const int Cin = (int)in->ne[0];
        ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in));                  // (T, Cin)
        y = kokoro_conv_1d_2d(ctx, w, y, /*K*/ 3, Cin, /*s*/ 1, /*p*/ 1, /*d*/ 1); // (T, Cout, 1)
        // Add bias broadcast: bias ne=(Cout,) → reshape (1, Cout, 1)
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
        const int Cout = (int)w->ne[1];
        y = ggml_reshape_2d(ctx, y, Tin, Cout);        // (T, Cout)
        return ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T)
    };

    xt = conv_k3(xt, conv1_w, conv1_b); // (Cout, T')
    tag(xt, "after_conv1");
    char ad2_prefix_buf[80];
    const char* ad2_prefix = nullptr;
    if (dbg_prefix) {
        std::snprintf(ad2_prefix_buf, sizeof(ad2_prefix_buf), "%s_adain2", dbg_prefix);
        ad2_prefix = ad2_prefix_buf;
    }
    xt = kokoro_adain1d(ctx, xt, style, adain2_w, adain2_b, ad2_prefix); // (Cout, T')
    xt = ggml_leaky_relu(ctx, xt, kAdainLeakySlope, /*inplace=*/false);
    tag(xt, "after_lr2");
    xt = conv_k3(xt, conv2_w, conv2_b); // (Cout, T')
    tag(xt, "after_conv2");

    // ---- Shortcut path ----
    ggml_tensor* sc = x;
    if (upsample) {
        // F.interpolate(scale=2, mode='nearest'): each input column is duplicated.
        // Equivalent to (Cin, 1, T) → concat with itself on a new dim → (Cin, 2, T) → reshape (Cin, 2T).
        ggml_tensor* sc_3d = ggml_reshape_3d(ctx, sc, sc->ne[0], 1, sc->ne[1]);
        ggml_tensor* sc_dup = ggml_concat(ctx, sc_3d, sc_3d, /*dim=*/1); // (Cin, 2, T)
        sc = ggml_cont(ctx, ggml_reshape_2d(ctx, sc_dup, sc->ne[0], 2 * (int)sc->ne[1]));
    }
    if (conv1x1_w) {
        // Conv1d k=1, no bias. (Cin, T') → transpose → (T', Cin) → conv → (T', Cout, 1) → reshape → transpose
        const int Tin = (int)sc->ne[1];
        const int Cin = (int)sc->ne[0];
        ggml_tensor* sct = ggml_cont(ctx, ggml_transpose(ctx, sc));                          // (T', Cin)
        sct = kokoro_conv_1d_2d(ctx, conv1x1_w, sct, /*K*/ 1, Cin, /*s*/ 1, /*p*/ 0, /*d*/ 1); // (T', Cout, 1)
        const int Cout = (int)conv1x1_w->ne[1];
        sct = ggml_reshape_2d(ctx, sct, Tin, Cout);    // (T', Cout)
        sc = ggml_cont(ctx, ggml_transpose(ctx, sct)); // (Cout, T')
    }

    // out = (xt + sc) / sqrt(2)
    ggml_tensor* sum = ggml_add(ctx, xt, sc);
    return ggml_scale(ctx, sum, 1.0f / std::sqrt(2.0f));
}

// ---------------------------------------------------------------------------
// M5a — ProsodyPredictor (dur_enc + pred.lstm + dur_proj → durations)
//
// Reference: kokoro/modules.py ProsodyPredictor + DurationEncoder.
//
// Voice pack split (corrects the briefing — see model.py:104, 118):
//   ref_s[:, :128]   → DECODER style (used in M7)
//   ref_s[:, 128:]   → PREDICTOR style (used here)
//
// Forward at synth time:
//   bert_proj_out  (D=512, L)         from M3
//   style_pred     (sty=128, 1)        from voice.pack[idx, 0, 128:256]
//
//   x = cat([bert_proj_out, style_pred.broadcast(L)], axis=0)   (640, L)
//   for il in 0..2:
//       x = bidir_LSTM(x)                                        (512, L)
//       x = AdaLN(x, style_pred)                                 (512, L)
//       x = cat([x, style_pred.broadcast(L)], axis=0)            (640, L)
//   dur_enc_out = x                                              (640, L)
//
//   y = bidir_LSTM_pred(x)                                       (512, L)
//   z = Linear_dur_proj(y)                                       (50, L)
//   durations_pre = sum_rows(sigmoid(z))                         (1, L)
//   # ↑ runtime-side: round + clamp(min=1) → int durations[L]
// ---------------------------------------------------------------------------

// Pull the per-voice predictor style vector from the loaded voice pack.
// voice.pack ne = (256, 1, 510) per the converter; index along ne[2] by
// (clamp(L_raw, 1, max_phon) - 1) to get a (256, 1) slice, then take
// the back half (offset 128*sizeof(F32)) for the predictor side.
//
// We bake the index into the graph (graph is rebuilt per-utterance
// anyway), so this returns a graph-time view of the voice-pack tensor.
static ggml_tensor* kokoro_voice_style_slice(ggml_context* ctx, kokoro_context* c, int L_raw, int byte_offset) {
    ggml_tensor* pack = c->vp.pack;
    const uint32_t max_phon = (uint32_t)pack->ne[2];
    int idx = L_raw;
    if (idx < 1)
        idx = 1;
    if (idx > (int)max_phon) {
        fprintf(stderr, "kokoro: L_raw=%d clamped to max_phon=%u for voice pack\n", L_raw, max_phon);
        idx = (int)max_phon;
    }
    idx -= 1;
    // Single ggml_view_2d at the combined absolute offset. The previous
    // view-of-view (pack → idx slice → 128-channel half) hit a Metal-
    // specific bug where short utterances on GPU read the wrong slice
    // of the voice pack — the first divergent stages in the per-stage
    // diff (`f0_curve`, `n_curve`) showed amplitude ~15× the PyTorch
    // reference, cascading to dec_encode_out cos=0.27 and
    // dec_decode_3_out cos=-0.30. Folding the two offsets into one
    // view restores the cascade. Note ggml view tensors are named
    // "<parent> (view)" by default — the original logs flagged
    // "voice.pack (view) (view)" which is the symptom of this
    // double-view path.
    return ggml_view_2d(ctx, pack, /*ne0*/ 128, /*ne1*/ 1, pack->nb[1],
                        (size_t)idx * pack->nb[2] + (size_t)byte_offset);
}

// Predictor style: back half (offset 128*F32). model.py:104 → ref_s[:, 128:]
static ggml_tensor* kokoro_voice_style_pred_view(ggml_context* ctx, kokoro_context* c, int L_raw) {
    return kokoro_voice_style_slice(ctx, c, L_raw, 128 * (int)sizeof(float));
}

// Decoder style: front half (offset 0). model.py:118 → ref_s[:, :128]
static ggml_tensor* kokoro_voice_style_dec_view(ggml_context* ctx, kokoro_context* c, int L_raw) {
    return kokoro_voice_style_slice(ctx, c, L_raw, 0);
}

static ggml_cgraph* kokoro_build_graph_predictor(kokoro_context* c, int L, int L_raw) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_dim;   // 512
    const int Hsty = (int)hp.style_dim; // 128
    const int H_lstm = D / 2;           // 256

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // 4 bidir LSTMs at small L (≤ ~520) + ~10 cat/AdaLN ops.
    // Per LSTM: 2 dirs × L × ~14 ops ≈ 14600 nodes at L=520. 4 LSTMs ≈ 60k.
    // 65536 buffer is generous but well within the 262144 sched ceiling.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // ---- Inputs ----
    // bert_dur is the per-token output of `bert_proj` (M3 stage `bert_proj_out`),
    // re-supplied as a graph input. Layout (D, L_padded) F32.
    ggml_tensor* bert_dur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, L);
    ggml_set_name(bert_dur, "bert_dur");
    ggml_set_input(bert_dur);

    // ---- Style: bake voice-pack slice into the graph ----
    // s_pred is F16 (the voice pack is loaded F32 — wait, voice.pack is F32
    // per the GGUF dump — so style is F32 and ggml_mul_mat with the AdaLN
    // fc_w (F16) works fine).
    ggml_tensor* style_pred = kokoro_voice_style_pred_view(ctx0, c, L_raw); // (128, 1) F32

    // Broadcast style to (Hsty, L) for the cat operations.
    ggml_tensor* s_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Hsty, L);
    ggml_tensor* s_broad = ggml_repeat(ctx0, style_pred, s_template); // (128, L)

    // Initial cat([bert_dur, s_broad], dim=0) → (D + Hsty, L) = (640, L)
    ggml_tensor* x = ggml_concat(ctx0, bert_dur, s_broad, /*dim=*/0);

    // ---- 3× alternating bidir-LSTM + AdaLN + cat-with-style ----
    for (int il = 0; il < 3; il++) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_ih_l0", il);
        ggml_tensor* W_ih_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_hh_l0", il);
        ggml_tensor* W_hh_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_ih_l0", il);
        ggml_tensor* b_ih_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_hh_l0", il);
        ggml_tensor* b_hh_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_ih_l0_reverse", il);
        ggml_tensor* W_ih_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_hh_l0_reverse", il);
        ggml_tensor* W_hh_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_ih_l0_reverse", il);
        ggml_tensor* b_ih_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_hh_l0_reverse", il);
        ggml_tensor* b_hh_r = require(c, buf);

        x = core_lstm::lstm_bidir(ctx0, gf, x, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                                  H_lstm); // (512, L)

        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.adaln.weight", il);
        ggml_tensor* fc_w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.adaln.bias", il);
        ggml_tensor* fc_b = require(c, buf);
        x = kokoro_adaln(ctx0, x, style_pred, fc_w, fc_b); // (512, L)

        x = ggml_concat(ctx0, x, s_broad, /*dim=*/0); // (640, L)
    }

    // dur_enc_out — the (640, L) input to pred.lstm and to alignment.
    ggml_tensor* dur_enc_out = ggml_cont(ctx0, x);
    ggml_set_name(dur_enc_out, "dur_enc_out");
    ggml_set_output(dur_enc_out);
    ggml_build_forward_expand(gf, dur_enc_out);

    // ---- pred.lstm: (640, L) → (512, L) ----
    ggml_tensor* pl_W_ih_f = require(c, "pred.lstm.weight_ih_l0");
    ggml_tensor* pl_W_hh_f = require(c, "pred.lstm.weight_hh_l0");
    ggml_tensor* pl_b_ih_f = require(c, "pred.lstm.bias_ih_l0");
    ggml_tensor* pl_b_hh_f = require(c, "pred.lstm.bias_hh_l0");
    ggml_tensor* pl_W_ih_r = require(c, "pred.lstm.weight_ih_l0_reverse");
    ggml_tensor* pl_W_hh_r = require(c, "pred.lstm.weight_hh_l0_reverse");
    ggml_tensor* pl_b_ih_r = require(c, "pred.lstm.bias_ih_l0_reverse");
    ggml_tensor* pl_b_hh_r = require(c, "pred.lstm.bias_hh_l0_reverse");

    ggml_tensor* pl_out = core_lstm::lstm_bidir(ctx0, gf, dur_enc_out, pl_W_ih_f, pl_W_hh_f, pl_b_ih_f, pl_b_hh_f,
                                                pl_W_ih_r, pl_W_hh_r, pl_b_ih_r, pl_b_hh_r, H_lstm); // (512, L)
    ggml_set_name(pl_out, "pred_lstm_out");
    ggml_set_output(pl_out);

    // ---- pred.dur_proj: Linear(512 → 50) ----
    ggml_tensor* dp_w = require(c, "pred.dur_proj.weight"); // ne=(512, 50)
    ggml_tensor* dp_b = require(c, "pred.dur_proj.bias");   // ne=(50,)
    ggml_tensor* dp = ggml_mul_mat(ctx0, dp_w, pl_out);     // (50, L)
    dp = ggml_add(ctx0, dp, dp_b);
    dp = ggml_sigmoid(ctx0, dp);
    // sum over the 50 dim (ne[0]) → (1, L). Runtime does the round+clamp.
    ggml_tensor* dur_pre = ggml_sum_rows(ctx0, dp); // (1, L)
    ggml_set_name(dur_pre, "durations_pre");
    ggml_set_output(dur_pre);
    ggml_build_forward_expand(gf, dur_pre);

    ggml_free(ctx0);
    return gf;
}

// Run the predictor graph and return the named stage as malloc'd float[].
// Stage handling:
//   "dur_enc_out"  → (640, L) raw output
//   "durations"    → (L,) post-round, post-clamp(min=1), cast to float
static float* kokoro_run_predictor(kokoro_context* c, const int32_t* raw_ids, int n_raw, const char* stage_name,
                                   int* out_n) {
    KOKORO_PROFILE("predictor");
    if (out_n)
        *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: predictor needs voice pack — call kokoro_load_voice_pack first\n");
        return nullptr;
    }
    if (n_raw <= 0)
        return nullptr;

    {
        std::string key = std::string("predictor:") + stage_name;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }

    // 1. Run BERT to get bert_proj_out.
    int n_bp = 0;
    float* bp = kokoro_run_bert(c, raw_ids, n_raw, "bert_proj_out", &n_bp);
    if (!bp)
        return nullptr;
    const int D = (int)c->hp.hidden_dim;
    const int L = n_bp / D;
    if (L * D != n_bp) {
        fprintf(stderr, "kokoro: bert_proj_out size mismatch: %d not divisible by %d\n", n_bp, D);
        std::free(bp);
        return nullptr;
    }

    // 2. Build + run predictor graph.
    ggml_cgraph* gf = kokoro_build_graph_predictor(c, L, n_raw);
    if (!gf) {
        std::free(bp);
        return nullptr;
    }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for predictor\n");
        std::free(bp);
        return nullptr;
    }
    ggml_tensor* in_bd = ggml_graph_get_tensor(gf, "bert_dur");
    ggml_backend_tensor_set(in_bd, bp, 0, (size_t)n_bp * sizeof(float));
    std::free(bp);

    if (kokoro_sched_compute(c, c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: predictor graph compute failed\n");
        return nullptr;
    }

    // Fetch all three named outputs, cache them, then return the requested one.
    // This way the first call computes the graph once and subsequent calls
    // (durations after dur_enc_out, or repeats from mag→phase) hit the cache.
    auto cache_2d = [&](const char* graph_name, const char* cache_name) {
        ggml_tensor* out = ggml_graph_get_tensor(gf, graph_name);
        if (!out) return;
        const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
        std::vector<float> buf(n_floats);
        ggml_backend_tensor_get(out, buf.data(), 0, n_floats * sizeof(float));
        g_kokoro_cache.put(std::string("predictor:") + cache_name, buf.data(), (int)n_floats);
    };
    cache_2d("dur_enc_out", "dur_enc_out");
    cache_2d("pred_lstm_out", "pred_lstm_out");

    // durations needs post-processing (length_scale + banker's rounding).
    {
        ggml_tensor* out = ggml_graph_get_tensor(gf, "durations_pre");
        if (out) {
            std::vector<float> raw_dur((size_t)L);
            ggml_backend_tensor_get(out, raw_dur.data(), 0, (size_t)L * sizeof(float));
            // Banker's rounding to match PyTorch torch.round (R5). fesetround
            // is set globally for nearbyintf. length_scale is applied BEFORE
            // round so round-half-to-even semantics are preserved.
            const float length_scale = c->params.length_scale > 0.0f ? c->params.length_scale : 1.0f;
            std::vector<float> dur((size_t)L);
            for (int i = 0; i < L; i++) {
                float v = std::nearbyintf(raw_dur[i] * length_scale);
                if (v < 1.0f) v = 1.0f;
                dur[i] = v;
            }
            g_kokoro_cache.put("predictor:durations", dur.data(), L);
        }
    }

    std::string key = std::string("predictor:") + stage_name;
    if (!g_kokoro_cache.has(key)) {
        fprintf(stderr, "kokoro: unknown predictor stage '%s'\n", stage_name);
        return nullptr;
    }
    return g_kokoro_cache.get_copy(key, out_n);
}

// ---------------------------------------------------------------------------
// M5b — F0Ntrain (pred.shared LSTM + F0/N AdainResBlk1d × 3 + F0/N_proj)
//
// Reference: kokoro/modules.py ProsodyPredictor.F0Ntrain.
//   x, _ = self.shared(en.transpose(-1,-2))   # bidir LSTM 640 → 512
//   F0 = x.transpose(-1,-2)                   # (B, 512, T)
//   for block in self.F0:                     # 3 AdainResBlk1d:
//       F0 = block(F0, s)                     # F0[1] upsamples 2×
//   F0 = self.F0_proj(F0)                     # Conv1d 256 → 1, k=1
//   # N mirrors F0 with separate weights
//
// Output dims after the F0 stack:
//   F0[0]: (512, T) → (512, T)
//   F0[1]: (512, T) → (256, 2T)   [upsample=True, dim_in=512, dim_out=256]
//   F0[2]: (256, 2T) → (256, 2T)
//   F0_proj: (256, 2T) → (1, 2T)
//
// Final F0 / N curves are 1D length 2*T_frames.
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_f0n(kokoro_context* c, int T_frames, int L_raw) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_dim; // 512
    const int H_lstm = D / 2;         // 256

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // shared LSTM at T_frames (~60-1500) is the dominant cost: 2 dirs × T × ~14 ops.
    // 6 AdainResBlk1d (3 F0 + 3 N), each ~25 ops. 1024-node margin.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // ---- Input ----
    ggml_tensor* en = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, /*ne0=D+sty=*/640, T_frames);
    ggml_set_name(en, "en");
    ggml_set_input(en);

    // ---- Style: bake voice-pack predictor slice ----
    ggml_tensor* style_pred = kokoro_voice_style_pred_view(ctx0, c, L_raw); // (128, 1) F32

    // ---- pred.shared bidir LSTM (640 → 512) ----
    ggml_tensor* sh_W_ih_f = require(c, "pred.shared.weight_ih_l0");
    ggml_tensor* sh_W_hh_f = require(c, "pred.shared.weight_hh_l0");
    ggml_tensor* sh_b_ih_f = require(c, "pred.shared.bias_ih_l0");
    ggml_tensor* sh_b_hh_f = require(c, "pred.shared.bias_hh_l0");
    ggml_tensor* sh_W_ih_r = require(c, "pred.shared.weight_ih_l0_reverse");
    ggml_tensor* sh_W_hh_r = require(c, "pred.shared.weight_hh_l0_reverse");
    ggml_tensor* sh_b_ih_r = require(c, "pred.shared.bias_ih_l0_reverse");
    ggml_tensor* sh_b_hh_r = require(c, "pred.shared.bias_hh_l0_reverse");

    ggml_tensor* shared_out = core_lstm::lstm_bidir(ctx0, gf, en, sh_W_ih_f, sh_W_hh_f, sh_b_ih_f, sh_b_hh_f, sh_W_ih_r,
                                                    sh_W_hh_r, sh_b_ih_r, sh_b_hh_r,
                                                    H_lstm); // (512, T_frames)
    // Tag the shared LSTM output so kokoro_extract_stage can surface it
    // via "pred_shared_out". Used to bisect Metal-specific kokoro
    // regressions inside F0Ntrain (the LSTM is between pred_lstm_out and
    // f0_curve in the per-stage diff cascade).
    ggml_set_name(shared_out, "pred_shared_out");
    ggml_set_output(shared_out);
    ggml_build_forward_expand(gf, shared_out);

    // Helper to load AdainResBlk1d weights for prefix "pred.X.{idx}".
    auto load_resblk = [&](const char* prefix, int idx, bool has_pool) {
        struct W {
            ggml_tensor* a1w;
            ggml_tensor* a1b;
            ggml_tensor* a2w;
            ggml_tensor* a2b;
            ggml_tensor* c1w;
            ggml_tensor* c1b;
            ggml_tensor* c2w;
            ggml_tensor* c2b;
            ggml_tensor* poolw;
            ggml_tensor* poolb;
            ggml_tensor* sc;
        } w;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s.%d.adain1.weight", prefix, idx);
        w.a1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain1.bias", prefix, idx);
        w.a1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain2.weight", prefix, idx);
        w.a2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain2.bias", prefix, idx);
        w.a2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv1.weight", prefix, idx);
        w.c1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv1.bias", prefix, idx);
        w.c1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv2.weight", prefix, idx);
        w.c2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv2.bias", prefix, idx);
        w.c2b = require(c, buf);
        if (has_pool) {
            std::snprintf(buf, sizeof(buf), "%s.%d.pool.weight", prefix, idx);
            w.poolw = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.%d.pool.bias", prefix, idx);
            w.poolb = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.%d.conv1x1.weight", prefix, idx);
            w.sc = require(c, buf);
        } else {
            w.poolw = w.poolb = w.sc = nullptr;
        }
        return w;
    };

    // Opt-in op-level bisect for the F0[0] / N[0] AdainResBlk1d. When
    // KOKORO_DEBUG_INTERMEDIATES=1, every sub-op output inside the
    // first block (AdaIN1d → LeakyReLU → Conv1d → AdaIN1d → LeakyReLU
    // → Conv1d → residual) is named "dbg_pred_{f0,n}_0_…" and marked
    // as a graph output so KOKORO_DUMP_STAGES (see crispasr-diff) can
    // write each one to disk for GPU-vs-CPU comparison. Unset (the
    // default) and the block runs with no extra ops or outputs, so
    // production builds pay zero cost. Used to bisect the ggml_norm
    // Metal regression — keep available for the next per-op-level bug.
    static const bool s_dbg = []() {
        const char* v = std::getenv("KOKORO_DEBUG_INTERMEDIATES");
        return v && *v && *v != '0';
    }();
    auto run_stack = [&](const char* prefix, const char* stage_branch, ggml_tensor* in) -> ggml_tensor* {
        // F0/N stacks all share: idx 0 (no upsample, dim 512→512), idx 1 (upsample, 512→256), idx 2 (no upsample, 256→256).
        ggml_tensor* y = in;
        auto w0 = load_resblk(prefix, 0, /*has_pool=*/false);
        auto w1 = load_resblk(prefix, 1, /*has_pool=*/true);
        auto w2 = load_resblk(prefix, 2, /*has_pool=*/false);
        char dbg0_buf[64];
        const char* dbg0 = nullptr;
        if (s_dbg) {
            std::snprintf(dbg0_buf, sizeof(dbg0_buf), "dbg_pred_%s_0", stage_branch);
            dbg0 = dbg0_buf;
        }
        y = kokoro_adain_resblk(ctx0, y, style_pred, w0.a1w, w0.a1b, w0.a2w, w0.a2b, w0.c1w, w0.c1b, w0.c2w, w0.c2b,
                                /*pool*/ nullptr, nullptr, /*conv1x1*/ nullptr, dbg0);
        // Tag each AdainResBlk1d output as `pred_{f0,n}_{k}_out` so
        // crispasr-diff can compare them against the Python reference
        // and pinpoint the first stage that diverges on Metal.
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "pred_%s_0_out", stage_branch);
            y = ggml_cont(ctx0, y);
            ggml_set_name(y, nm);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        y = kokoro_adain_resblk(ctx0, y, style_pred, w1.a1w, w1.a1b, w1.a2w, w1.a2b, w1.c1w, w1.c1b, w1.c2w, w1.c2b,
                                w1.poolw, w1.poolb, w1.sc);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "pred_%s_1_out", stage_branch);
            y = ggml_cont(ctx0, y);
            ggml_set_name(y, nm);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        y = kokoro_adain_resblk(ctx0, y, style_pred, w2.a1w, w2.a1b, w2.a2w, w2.a2b, w2.c1w, w2.c1b, w2.c2w, w2.c2b,
                                /*pool*/ nullptr, nullptr, /*conv1x1*/ nullptr);
        {
            char nm[32];
            std::snprintf(nm, sizeof(nm), "pred_%s_2_out", stage_branch);
            y = ggml_cont(ctx0, y);
            ggml_set_name(y, nm);
            ggml_set_output(y);
            ggml_build_forward_expand(gf, y);
        }
        return y; // (256, 2*T_frames)
    };

    // F0 stack
    ggml_tensor* F0 = run_stack("pred.F0", "f0", shared_out);
    // F0_proj: Conv1d(256 → 1, k=1).
    ggml_tensor* fp_w = require(c, "pred.F0_proj.weight"); // ne=(1, 256, 1)
    ggml_tensor* fp_b = require(c, "pred.F0_proj.bias");   // ne=(1,)
    {
        const int Tf = (int)F0->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, F0));                          // (T, 256)
        y = kokoro_conv_1d_2d(ctx0, fp_w, y, /*K*/ 1, /*Cin*/ 256, /*s*/ 1, /*p*/ 0, /*d*/ 1); // (T, 1, 1)
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, fp_b, 1, 1, 1));                         // bias broadcast
        F0 = ggml_reshape_2d(ctx0, y, Tf, 1);                                                // (T, 1)
        // ggml_squeeze isn't available; just keep as (T, 1) and treat the 1 dim
        // as a no-op channel for the downstream extractor.
    }
    F0 = ggml_cont(ctx0, F0);
    ggml_set_name(F0, "f0_curve");
    ggml_set_output(F0);
    ggml_build_forward_expand(gf, F0);

    // N stack (mirror)
    ggml_tensor* N = run_stack("pred.N", "n", shared_out);
    ggml_tensor* np_w = require(c, "pred.N_proj.weight");
    ggml_tensor* np_b = require(c, "pred.N_proj.bias");
    {
        const int Tf = (int)N->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, N));
        y = kokoro_conv_1d_2d(ctx0, np_w, y, /*K*/ 1, /*Cin*/ 256, /*s*/ 1, /*p*/ 0, /*d*/ 1);
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, np_b, 1, 1, 1));
        N = ggml_reshape_2d(ctx0, y, Tf, 1);
    }
    N = ggml_cont(ctx0, N);
    ggml_set_name(N, "n_curve");
    ggml_set_output(N);
    ggml_build_forward_expand(gf, N);

    ggml_free(ctx0);
    return gf;
}

// Run the full predictor + alignment + F0Ntrain pipeline. Returns the named
// stage as malloc'd float[]. Stages handled:
//   "align_out" → (640, T_frames)
//   "f0_curve"  → (2*T_frames,)   (already squeezed from (1, 2*T_frames))
//   "n_curve"   → (2*T_frames,)
static float* kokoro_run_f0n(kokoro_context* c, const int32_t* raw_ids, int n_raw, const char* stage_name, int* out_n) {
    KOKORO_PROFILE("f0n");
    if (out_n)
        *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: F0Ntrain needs voice pack\n");
        return nullptr;
    }
    {
        std::string key = std::string("f0n:") + stage_name;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }
    // 1. Run predictor → dur_enc_out + durations.
    int n_de = 0, n_dr = 0;
    float* de = kokoro_run_predictor(c, raw_ids, n_raw, "dur_enc_out", &n_de);
    if (!de)
        return nullptr;
    float* dr = kokoro_run_predictor(c, raw_ids, n_raw, "durations", &n_dr);
    if (!dr) {
        std::free(de);
        return nullptr;
    }

    const int D = 640;
    const int L = n_dr;
    if (n_de != D * L) {
        fprintf(stderr, "kokoro: dur_enc/durations length mismatch %d vs %d*%d\n", n_de, D, L);
        std::free(de);
        std::free(dr);
        return nullptr;
    }

    // 2. CPU alignment (M6).
    std::vector<int> dur_int((size_t)L);
    for (int i = 0; i < L; i++)
        dur_int[i] = (int)dr[i];
    std::free(dr);
    int T_frames = 0;
    float* en = kokoro_align_repeat(de, D, L, dur_int.data(), &T_frames);
    std::free(de);
    if (!en || T_frames <= 0) {
        std::free(en);
        return nullptr;
    }

    if (std::strcmp(stage_name, "align_out") == 0) {
        g_kokoro_cache.put("f0n:align_out", en, D * T_frames);
        if (out_n)
            *out_n = D * T_frames;
        return en;
    }

    // 3. Run F0/N graph.
    ggml_cgraph* gf = kokoro_build_graph_f0n(c, T_frames, n_raw);
    if (!gf) {
        std::free(en);
        return nullptr;
    }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for F0Ntrain\n");
        std::free(en);
        return nullptr;
    }
    ggml_tensor* in_en = ggml_graph_get_tensor(gf, "en");
    ggml_backend_tensor_set(in_en, en, 0, (size_t)D * T_frames * sizeof(float));
    std::free(en);

    if (kokoro_sched_compute(c, c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: F0Ntrain graph compute failed\n");
        return nullptr;
    }

    // Fetch + cache both named outputs so the next call (f0_curve after
    // n_curve, or the duplicate f0_curve from run_generator) hits the cache.
    auto cache_2d = [&](const char* name) {
        ggml_tensor* out = ggml_graph_get_tensor(gf, name);
        if (!out) return;
        const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
        std::vector<float> buf(n_floats);
        ggml_backend_tensor_get(out, buf.data(), 0, n_floats * sizeof(float));
        g_kokoro_cache.put(std::string("f0n:") + name, buf.data(), (int)n_floats);
    };
    cache_2d("f0_curve");
    cache_2d("n_curve");

    const char* tname = stage_name; // "f0_curve" or "n_curve" — names match graph
    {
        std::string key = std::string("f0n:") + tname;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }
    // Fallback to direct fetch for any debug tname not cached above.
    ggml_tensor* out = ggml_graph_get_tensor(gf, tname);
    if (!out) {
        // dbg_* stages are opt-in (KOKORO_DEBUG_INTERMEDIATES=1); when
        // that's unset they're never tagged into the graph. Silently
        // return null so crispasr-diff can route them to SKIP rather
        // than flooding stderr in normal runs.
        if (std::strncmp(tname, "dbg_", 4) != 0)
            fprintf(stderr, "kokoro: F0Ntrain graph missing output '%s'\n", tname);
        return nullptr;
    }
    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    if (out_n)
        *out_n = (int)n_floats;
    return r;
}

// ---------------------------------------------------------------------------
// M7a/M7b — Decoder body (F0_conv + N_conv + asr_res + encode + 4× decode)
//
// Reference: kokoro/istftnet.py Decoder.forward.
//   F0 = self.F0_conv(F0_curve.unsqueeze(1))    # Conv1d 1→1, k=3, s=2, pad=1
//   N  = self.N_conv(N_curve.unsqueeze(1))      # same
//   x  = cat([asr, F0, N], axis=1)              # (514, T_frames)
//   x  = self.encode(x, s_dec)                  # AdainResBlk1d 514→1024
//   asr_res = self.asr_res(asr)                 # Conv1d 512→64, k=1
//   for i in 0..3:
//       x = cat([x, asr_res, F0, N], axis=1)    # → (1090, T)
//       x = self.decode[i](x, s_dec)            # AdainResBlk1d, last has upsample=True
//
// Output dims:
//   dec_encode_out:    (1024, T_frames)
//   dec.decode[0..2]:  (1024, T_frames)
//   dec.decode[3]:     (512, 2*T_frames)        [upsample=True]
//
// All AdainResBlk1d in the decoder use s_dec (voice pack [0:128], NOT
// the predictor's [128:256]).
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_decoder_body(kokoro_context* c, int T_frames, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // Decoder body has no LSTMs, just convs+AdaIN+LeakyReLU. ~50 ops/block × 5 blocks
    // + a few cat/F0_conv/asr_res steps. 16k node budget is generous.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // ---- Inputs (asr, F0_curve, N_curve from upstream stages) ----
    // asr ne=(512, T_frames) — duration-aligned text_enc_out
    ggml_tensor* asr_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, T_frames);
    ggml_set_name(asr_in, "asr");
    ggml_set_input(asr_in);
    // F0_curve ne=(2*T_frames,) → unsqueeze to (1, 2*T_frames) before F0_conv
    ggml_tensor* f0_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 2 * T_frames);
    ggml_set_name(f0_in, "f0");
    ggml_set_input(f0_in);
    ggml_tensor* n_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 2 * T_frames);
    ggml_set_name(n_in, "n");
    ggml_set_input(n_in);

    ggml_tensor* style_dec = kokoro_voice_style_dec_view(ctx0, c, L_raw); // (128, 1)

    // ---- F0_conv / N_conv: Conv1d(1, 1, k=3, s=2, pad=1) ----
    auto small_conv1d = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int s, int p) {
        // Input x ne=(1, 2T). Conv expects (T, C, 1) layout — we have (1, 2T) which IS (C=1, T=2T) in ggml,
        // but conv_1d wants (T, C). Transpose to (2T, 1).
        const int Tin = (int)x->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, x));                // (2T, 1)
        y = kokoro_conv_1d_2d(ctx0, w, y, /*K*/ 3, /*Cin*/ 1, s, p, /*d*/ 1);     // (Tout, 1, 1)
        const int Cout = (int)w->ne[1];
        ggml_tensor* b3 = ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1);
        y = ggml_add(ctx0, y, b3);
        const int Tout = (int)y->ne[0];
        y = ggml_reshape_2d(ctx0, y, Tout, Cout);
        (void)Tin;
        return ggml_cont(ctx0, ggml_transpose(ctx0, y)); // (Cout, Tout)
    };
    ggml_tensor* F0_conv_w = require(c, "dec.F0_conv.weight");
    ggml_tensor* F0_conv_b = require(c, "dec.F0_conv.bias");
    ggml_tensor* N_conv_w = require(c, "dec.N_conv.weight");
    ggml_tensor* N_conv_b = require(c, "dec.N_conv.bias");
    ggml_tensor* F0_d = small_conv1d(f0_in, F0_conv_w, F0_conv_b, /*s*/ 2, /*p*/ 1); // (1, T_frames)
    ggml_tensor* N_d = small_conv1d(n_in, N_conv_w, N_conv_b, /*s*/ 2, /*p*/ 1);     // (1, T_frames)

    // ---- cat([asr, F0_d, N_d], axis=0) → (514, T_frames) ----
    ggml_tensor* x = ggml_concat(ctx0, asr_in, F0_d, /*dim=*/0);
    x = ggml_concat(ctx0, x, N_d, /*dim=*/0); // (514, T)

    // ---- encode: AdainResBlk1d(514→1024) ----
    auto load_decoder_resblk = [&](const char* prefix, bool has_pool) {
        struct W {
            ggml_tensor *a1w, *a1b, *a2w, *a2b, *c1w, *c1b, *c2w, *c2b, *poolw, *poolb, *sc;
        } w;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s.adain1.weight", prefix);
        w.a1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain1.bias", prefix);
        w.a1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.weight", prefix);
        w.a2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.bias", prefix);
        w.a2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1.weight", prefix);
        w.c1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1.bias", prefix);
        w.c1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv2.weight", prefix);
        w.c2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv2.bias", prefix);
        w.c2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1x1.weight", prefix);
        w.sc = require(c, buf);
        if (has_pool) {
            std::snprintf(buf, sizeof(buf), "%s.pool.weight", prefix);
            w.poolw = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.pool.bias", prefix);
            w.poolb = require(c, buf);
        } else {
            w.poolw = w.poolb = nullptr;
        }
        return w;
    };

    {
        auto e = load_decoder_resblk("dec.encode", /*has_pool=*/false);
        x = kokoro_adain_resblk(ctx0, x, style_dec, e.a1w, e.a1b, e.a2w, e.a2b, e.c1w, e.c1b, e.c2w, e.c2b,
                                /*pool*/ nullptr, nullptr, e.sc); // (1024, T)
    }
    ggml_tensor* dec_encode_out = ggml_cont(ctx0, x);
    ggml_set_name(dec_encode_out, "dec_encode_out");
    ggml_set_output(dec_encode_out);
    ggml_build_forward_expand(gf, dec_encode_out);

    // ---- asr_res: Conv1d(512→64, k=1) ----
    ggml_tensor* asr_res_w = require(c, "dec.asr_res.weight");
    ggml_tensor* asr_res_b = require(c, "dec.asr_res.bias");
    ggml_tensor* asr_res_out;
    {
        const int Tin = (int)asr_in->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, asr_in));                            // (T, 512)
        y = kokoro_conv_1d_2d(ctx0, asr_res_w, y, /*K*/ 1, /*Cin*/ 512, /*s*/ 1, /*p*/ 0, /*d*/ 1); // (T, 64, 1)
        ggml_tensor* b3 = ggml_reshape_3d(ctx0, asr_res_b, 1, asr_res_b->ne[0], 1);
        y = ggml_add(ctx0, y, b3);
        y = ggml_reshape_2d(ctx0, y, Tin, 64);                  // (T, 64)
        asr_res_out = ggml_cont(ctx0, ggml_transpose(ctx0, y)); // (64, T)
    }

    // ---- 4× decode AdainResBlk1d, with cat([x, asr_res, F0_d, N_d]) before each ----
    x = dec_encode_out;
    for (int il = 0; il < 4; il++) {
        // cat → (1024 + 64 + 1 + 1, T) = (1090, T)
        ggml_tensor* xc = ggml_concat(ctx0, x, asr_res_out, /*dim=*/0); // (1088, T)
        xc = ggml_concat(ctx0, xc, F0_d, /*dim=*/0);
        xc = ggml_concat(ctx0, xc, N_d, /*dim=*/0); // (1090, T)

        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "dec.decode.%d", il);
        const bool has_pool = (il == 3);
        auto w = load_decoder_resblk(prefix, has_pool);
        x = kokoro_adain_resblk(ctx0, xc, style_dec, w.a1w, w.a1b, w.a2w, w.a2b, w.c1w, w.c1b, w.c2w, w.c2b, w.poolw,
                                w.poolb, w.sc);
    }
    // After block 3 (upsample=True): x is (512, 2*T_frames).
    ggml_tensor* dec_decode_3_out = ggml_cont(ctx0, x);
    ggml_set_name(dec_decode_3_out, "dec_decode_3_out");
    ggml_set_output(dec_decode_3_out);
    ggml_build_forward_expand(gf, dec_decode_3_out);

    ggml_free(ctx0);
    return gf;
}

// Run the full preamble (BERT, predictor, alignment, F0Ntrain, text_enc) +
// decoder body, returning the named stage as malloc'd float[].
static float* kokoro_run_decoder_body(kokoro_context* c, const int32_t* raw_ids, int n_raw, const char* stage_name,
                                      int* out_n) {
    KOKORO_PROFILE("decoder_body");
    if (out_n)
        *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: decoder needs voice pack\n");
        return nullptr;
    }
    {
        std::string key = std::string("decoder_body:") + stage_name;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }

    // 1. dur_enc_out + durations from predictor.
    int n_de = 0, n_dr = 0;
    float* de = kokoro_run_predictor(c, raw_ids, n_raw, "dur_enc_out", &n_de);
    if (!de)
        return nullptr;
    float* dr = kokoro_run_predictor(c, raw_ids, n_raw, "durations", &n_dr);
    if (!dr) {
        std::free(de);
        return nullptr;
    }
    const int L = n_dr;
    std::vector<int> dur_int((size_t)L);
    for (int i = 0; i < L; i++)
        dur_int[i] = (int)dr[i];
    std::free(dr);
    int T_frames = 0;
    float* en = kokoro_align_repeat(de, 640, L, dur_int.data(), &T_frames);
    std::free(de);
    if (!en)
        return nullptr;

    // 2. F0/N curves from F0Ntrain (uses en internally — recompute here for simplicity).
    int n_f = 0, n_nc = 0;
    float* f0 = kokoro_run_f0n(c, raw_ids, n_raw, "f0_curve", &n_f);
    if (!f0) {
        std::free(en);
        return nullptr;
    }
    float* nc = kokoro_run_f0n(c, raw_ids, n_raw, "n_curve", &n_nc);
    if (!nc) {
        std::free(en);
        std::free(f0);
        return nullptr;
    }

    // 3. text_enc_out, then duration-align to asr.
    int n_te = 0;
    float* te = kokoro_run_text_enc(c, raw_ids, n_raw, "text_enc_out", &n_te);
    if (!te) {
        std::free(en);
        std::free(f0);
        std::free(nc);
        return nullptr;
    }
    int T_frames_asr = 0;
    float* asr = kokoro_align_repeat(te, 512, L, dur_int.data(), &T_frames_asr);
    std::free(te);
    if (!asr || T_frames_asr != T_frames) {
        fprintf(stderr, "kokoro: T_frames mismatch (%d != %d)\n", T_frames_asr, T_frames);
        std::free(asr);
        std::free(en);
        std::free(f0);
        std::free(nc);
        return nullptr;
    }
    std::free(en); // not directly used by decoder body — F0Ntrain consumed it already

    // 4. Build + run decoder-body graph.
    ggml_cgraph* gf = kokoro_build_graph_decoder_body(c, T_frames, n_raw);
    if (!gf) {
        std::free(asr);
        std::free(f0);
        std::free(nc);
        return nullptr;
    }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for decoder body\n");
        std::free(asr);
        std::free(f0);
        std::free(nc);
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "asr"), asr, 0, (size_t)512 * T_frames * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "f0"), f0, 0, (size_t)2 * T_frames * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "n"), nc, 0, (size_t)2 * T_frames * sizeof(float));
    std::free(asr);
    std::free(f0);
    std::free(nc);

    if (kokoro_sched_compute(c, c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: decoder-body graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "kokoro: decoder-body graph missing output '%s'\n", stage_name);
        return nullptr;
    }
    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    g_kokoro_cache.put(std::string("decoder_body:") + stage_name, r, (int)n_floats);
    if (out_n)
        *out_n = (int)n_floats;
    return r;
}

// ---------------------------------------------------------------------------
// M7c — Generator (iSTFTNet)
//
// Reference: kokoro/istftnet.py Generator + AdaINResBlock1 + SineGen +
// SourceModuleHnNSF + TorchSTFT.
//
// Forward at synth time:
//   x = dec_decode_3_out                 (512, 2*T_frames)   from M7b
//   F0_curve = pred.F0_proj output        (1, 2*T_frames)    from M5b
//   s_dec  = voice.pack[:, :128]          (128, 1)
//
//   # CPU-side (pre-built before graph), uses random noise + STFT:
//   f0_up = repeat-300×(F0_curve)        (600*T_frames,)
//   har_source = m_source(SineGen(f0_up))(600*T_frames,)
//   har_spec, har_phase = STFT(har_source, n_fft=20, hop=5, hann periodic, center=True)
//   har = cat(har_spec, har_phase, dim=0) (22, T_har) where T_har = 120*T_frames + 1
//
//   for i in 0..1:
//       x = LeakyReLU(0.1)(x)
//       x_source = noise_convs[i](har)                    # k=12 s=6 (i=0); k=1 (i=1)
//       x_source = noise_res[i](x_source, s_dec)           # AdaINResBlock1 with Snake-α
//       x = ups[i](x)                                       # ConvTranspose1d
//       if i == 1: x = reflection_pad(x, (1, 0))           # left-pad 1
//       x = x + x_source
//       x = mean_j(resblocks[i*3+j](x, s_dec) for j in 0..2)
//   x = LeakyReLU(0.01)(x)                                  # default slope, NOT 0.1!
//   x = conv_post(x)                                        # Conv1d 128→22, k=7, p=3
//   spec  = exp(x[:11, :])                                  # NOT raw mag
//   phase = sin(x[11:, :])                                  # NOT raw phase
//
// All Generator AdaINResBlock1 (note capital B!) use Snake-α activation:
//   y = x + (1/α) * sin²(α*x)   with per-channel α (1, C, 1) F16, init=1.
// This is *different* from the predictor / decoder-body's `kokoro_adain_resblk`
// which uses LeakyReLU(0.2).
// ---------------------------------------------------------------------------

// Snake-α activation. Thin alias to core_act::snake_alpha — see that
// header for the (1, C, 1) layout convention and Metal-typing notes.
static inline ggml_tensor* kokoro_snake1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    return core_act::snake_alpha(ctx, x, alpha);
}

// AdaINResBlock1 (NOTE the capital "B" — this is the Generator's class, not
// the predictor/decoder-body's `kokoro_adain_resblk`).
// 3 sub-blocks (j=0..2) looping over `dilations` for the convs1; convs2
// always uses dilation=1. All convs use kernel_size=K with same-padding.
struct kokoro_resblock1_w {
    int K = 0;
    int dilations[3] = {1, 3, 5};
    ggml_tensor* a1w[3] = {};
    ggml_tensor* a1b[3] = {};
    ggml_tensor* a2w[3] = {};
    ggml_tensor* a2b[3] = {};
    ggml_tensor* alpha1[3] = {};
    ggml_tensor* alpha2[3] = {};
    ggml_tensor* c1w[3] = {};
    ggml_tensor* c1b[3] = {};
    ggml_tensor* c2w[3] = {};
    ggml_tensor* c2b[3] = {};
};

static void kokoro_load_resblock1(const kokoro_context* c, kokoro_resblock1_w& w, const char* prefix, int K,
                                  const int dilations[3]) {
    char buf[128];
    w.K = K;
    for (int j = 0; j < 3; j++)
        w.dilations[j] = dilations[j];
    for (int j = 0; j < 3; j++) {
        std::snprintf(buf, sizeof(buf), "%s.adain1.%d.weight", prefix, j);
        w.a1w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain1.%d.bias", prefix, j);
        w.a1b[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.%d.weight", prefix, j);
        w.a2w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.%d.bias", prefix, j);
        w.a2b[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.alpha1.%d", prefix, j);
        w.alpha1[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.alpha2.%d", prefix, j);
        w.alpha2[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs1.%d.weight", prefix, j);
        w.c1w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs1.%d.bias", prefix, j);
        w.c1b[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs2.%d.weight", prefix, j);
        w.c2w[j] = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.convs2.%d.bias", prefix, j);
        w.c2b[j] = require(c, buf);
    }
}

// Conv1d helper used by AdaINResBlock1: kernel size K, dilation d, same-padding.
// in: (Cin, T) F32. w: ne=(K*Cin, Cout) F16/quantized. b: ne=(Cout,) F32.
// Returns (Cout, T) F32.
static inline ggml_tensor* kokoro_conv1d_kd(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b, int K,
                                            int d) {
    const int Tin = (int)in->ne[1];
    const int Cin = (int)in->ne[0];
    const int Cout = (int)w->ne[1];
    const int p = d * (K - 1) / 2;                            // same-padding for stride=1
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in)); // (T, Cin)
    y = kokoro_conv_1d_2d(ctx, w, y, K, Cin, /*s*/ 1, p, d);  // (T, Cout, 1)
    if (b) {
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
    }
    y = ggml_reshape_2d(ctx, y, Tin, Cout);        // (T, Cout)
    return ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T)
}

static inline ggml_tensor* kokoro_resblock1_forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                                    const kokoro_resblock1_w& w) {
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xt = kokoro_adain1d(ctx, x, style, w.a1w[j], w.a1b[j]);
        xt = kokoro_snake1d(ctx, xt, w.alpha1[j]);
        xt = kokoro_conv1d_kd(ctx, xt, w.c1w[j], w.c1b[j], w.K, w.dilations[j]);
        xt = kokoro_adain1d(ctx, xt, style, w.a2w[j], w.a2b[j]);
        xt = kokoro_snake1d(ctx, xt, w.alpha2[j]);
        xt = kokoro_conv1d_kd(ctx, xt, w.c2w[j], w.c2b[j], w.K, /*d=*/1);
        x = ggml_add(ctx, xt, x);
    }
    return x;
}

// PyTorch ConvTranspose1d wrapper: ggml_conv_transpose_1d only supports
// padding=0, so we run with p=0 and crop `pad` samples from each side of the
// time axis afterwards. PyTorch formula (stride s, kernel k, padding p):
//   T_out = (T_in - 1) * s - 2*p + k
// ggml's p=0 output: T_unpad = (T_in - 1) * s + k. Crop p from each side.
//
// in: (Cin, T) F32. w: ne=(K, Cout, Cin) F16. b: ne=(Cout,) F32.
// Returns (Cout, T_out) F32.
static inline ggml_tensor* kokoro_convt1d_pad(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b,
                                              int stride, int pad) {
    const int Cout = (int)w->ne[1];
    ggml_tensor* xT = ggml_cont(ctx, ggml_transpose(ctx, in));         // (T, Cin)
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, xT, stride, 0, 1); // (T_unpad, Cout, 1, 1)
    const int T_unpad = (int)y->ne[0];
    const int T_out = T_unpad - 2 * pad;
    y = ggml_reshape_2d(ctx, y, T_unpad, Cout); // (T_unpad, Cout)
    if (pad > 0) {
        // Slice [pad : pad + T_out] along the time (ne[0]) axis.
        y = ggml_view_2d(ctx, y, T_out, Cout, (size_t)T_unpad * sizeof(float), (size_t)pad * sizeof(float));
        y = ggml_cont(ctx, y); // (T_out, Cout)
    }
    ggml_tensor* yT = ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
    if (b)
        yT = ggml_add(ctx, yT, b); // bias broadcasts on T
    return yT;
}

// Conv1d helper for noise_convs[i]. Different from kokoro_conv1d_kd in that K
// and stride may be non-trivial (k=12,s=6 for noise_convs[0]; k=1,s=1 for
// noise_convs[1]). Does PyTorch-style same-output handling via explicit
// padding param.
static inline ggml_tensor* kokoro_conv1d_ks(ggml_context* ctx, ggml_tensor* in, ggml_tensor* w, ggml_tensor* b, int K,
                                            int s, int p) {
    const int Cin = (int)in->ne[0];
    const int Cout = (int)w->ne[1];
    ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in));  // (T, Cin)
    y = kokoro_conv_1d_2d(ctx, w, y, K, Cin, s, p, /*d*/ 1);   // (T_out, Cout, 1)
    if (b) {
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
    }
    const int Tout = (int)y->ne[0];
    y = ggml_reshape_2d(ctx, y, Tout, Cout);
    return ggml_cont(ctx, ggml_transpose(ctx, y)); // (Cout, T_out)
}

// ---------------------------------------------------------------------------
// CPU-side: produce `har` (22, T_har) from f0_curve via SineGen + l_linear +
// tanh + STFT. Has random noise (rand_ini, randn_like) so MUST be CPU.
//
// Reference: istftnet.py SineGen._f02sine + SourceModuleHnNSF.forward +
// TorchSTFT.transform.
//
//   upsample_scale = 300 (= 10 * 6 * 5 from upsample_rates [10, 6] * hop 5)
//   harmonic_num = 8, dim = 9, sample_rate = 24000
//   sine_amp = 0.1, noise_std = 0.003, voiced_threshold = 10
//   n_fft = 20, hop = 5, hann periodic, center=True
//
// Output T_har = (L_high - n_fft + n_fft) / hop + 1 = L_high/hop + 1
//              = 600*T_frames/5 + 1 = 120*T_frames + 1.
// ---------------------------------------------------------------------------

static const float kKokoroTwoPi = 6.283185307179586f;

static float* kokoro_make_har(const kokoro_context* c, const float* f0_curve, int T_frames, int* out_T_har,
                              std::mt19937& rng) {
    const int upsample_scale = 300; // prod(upsample_rates) * hop
    const int harmonic_num = 8;
    const int dim = harmonic_num + 1; // 9
    const int sample_rate = 24000;
    const float sine_amp = 0.1f;
    const float noise_std = 0.003f;
    const float voiced_threshold = 10.0f;
    const int n_fft = 20;
    const int hop = 5;
    const int n_bins = n_fft / 2 + 1; // 11

    const int L_low = 2 * T_frames;
    const int L_high = upsample_scale * L_low;

    // ---- f0_upsamp: nn.Upsample(scale_factor=300) → default mode='nearest'.
    // Each f0 sample is repeated upsample_scale times.
    std::vector<float> f0_up((size_t)L_high);
    for (int i = 0; i < L_low; i++) {
        const float v = f0_curve[i];
        for (int j = 0; j < upsample_scale; j++)
            f0_up[(size_t)i * upsample_scale + j] = v;
    }

    // ---- SineGen._f02sine ----
    // 1. fn[t, k] = f0[t] * (k+1)
    // 2. rad[t, k] = (fn[t, k] / sr) mod 1
    // 3. rand_ini at t=0 for k>=1
    // 4. Downsample rad by 1/upsample_scale (linear interp, align_corners=False)
    // 5. cumsum * 2π → phase_low
    // 6. Upsample phase_low * upsample_scale by upsample_scale (linear)
    // 7. sin(phase) * sine_amp = sines
    std::vector<float> rad((size_t)L_high * dim);
    for (int t = 0; t < L_high; t++) {
        const float fv = f0_up[t];
        for (int k = 0; k < dim; k++) {
            float fn = fv * (float)(k + 1);
            float r = fn / (float)sample_rate;
            r = r - std::floor(r);
            rad[(size_t)t * dim + k] = r;
        }
    }
    // rand_ini: per-(B, dim) noise, B=1; entry at [0] zeroed; only first time step gets noise.
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
    std::vector<float> rand_ini((size_t)dim, 0.0f);
    for (int k = 1; k < dim; k++)
        rand_ini[k] = uni01(rng);
    for (int k = 0; k < dim; k++)
        rad[(size_t)0 * dim + k] += rand_ini[k];

    // Downsample rad (L_high → L_low). PyTorch F.interpolate(linear, align_corners=False):
    //   src_idx = (dst + 0.5) * scale - 0.5    (with scale = L_high / L_low = upsample_scale)
    std::vector<float> rad_low((size_t)L_low * dim);
    for (int j = 0; j < L_low; j++) {
        float src_idx = ((float)j + 0.5f) * (float)upsample_scale - 0.5f;
        float clamped = std::max(0.0f, std::min((float)(L_high - 1), src_idx));
        int i0 = (int)std::floor(clamped);
        int i1 = std::min(i0 + 1, L_high - 1);
        float frac = clamped - (float)i0;
        for (int k = 0; k < dim; k++) {
            float v0 = rad[(size_t)i0 * dim + k];
            float v1 = rad[(size_t)i1 * dim + k];
            rad_low[(size_t)j * dim + k] = v0 * (1.0f - frac) + v1 * frac;
        }
    }
    // cumsum * 2π
    std::vector<float> phase_low((size_t)L_low * dim);
    for (int k = 0; k < dim; k++) {
        float acc = 0.0f;
        for (int t = 0; t < L_low; t++) {
            acc += rad_low[(size_t)t * dim + k];
            phase_low[(size_t)t * dim + k] = acc * kKokoroTwoPi;
        }
    }
    // Upsample phase_low * upsample_scale (L_low → L_high), linear interp.
    std::vector<float> sines((size_t)L_high * dim);
    for (int t = 0; t < L_high; t++) {
        float src_idx = ((float)t + 0.5f) / (float)upsample_scale - 0.5f;
        float clamped = std::max(0.0f, std::min((float)(L_low - 1), src_idx));
        int i0 = (int)std::floor(clamped);
        int i1 = std::min(i0 + 1, L_low - 1);
        float frac = clamped - (float)i0;
        for (int k = 0; k < dim; k++) {
            float v0 = phase_low[(size_t)i0 * dim + k] * (float)upsample_scale;
            float v1 = phase_low[(size_t)i1 * dim + k] * (float)upsample_scale;
            float ph = v0 * (1.0f - frac) + v1 * frac;
            sines[(size_t)t * dim + k] = std::sin(ph) * sine_amp;
        }
    }

    // uv mask + noise + apply.
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::vector<float> sine_waves((size_t)L_high * dim);
    for (int t = 0; t < L_high; t++) {
        const float uv = (f0_up[t] > voiced_threshold) ? 1.0f : 0.0f;
        const float noise_a = uv * noise_std + (1.0f - uv) * sine_amp / 3.0f;
        for (int k = 0; k < dim; k++) {
            float n = noise_a * norm(rng);
            sine_waves[(size_t)t * dim + k] = sines[(size_t)t * dim + k] * uv + n;
        }
    }

    // ---- m_source: l_linear (9 → 1) + tanh ----
    // weight ne=(9, 1) F16; bias ne=(1,) F32.
    ggml_tensor* lw = require(c, "dec.gen.m_source.weight");
    ggml_tensor* lb = require(c, "dec.gen.m_source.bias");
    if (!lw || !lb)
        return nullptr;
    std::vector<uint16_t> w_f16((size_t)dim);
    std::vector<float> w_f32((size_t)dim);
    ggml_backend_tensor_get(lw, w_f16.data(), 0, (size_t)dim * sizeof(uint16_t));
    ggml_fp16_to_fp32_row((const ggml_fp16_t*)w_f16.data(), w_f32.data(), dim);
    float bias_v = 0.0f;
    ggml_backend_tensor_get(lb, &bias_v, 0, sizeof(float));

    std::vector<float> har_source((size_t)L_high);
    for (int t = 0; t < L_high; t++) {
        float s = bias_v;
        for (int k = 0; k < dim; k++)
            s += sine_waves[(size_t)t * dim + k] * w_f32[k];
        har_source[t] = std::tanh(s);
    }

    // ---- STFT (n_fft=20, hop=5, hann periodic, center=True, pad_mode=reflect) ----
    const int pad_n = n_fft / 2; // 10
    const int L_pad = L_high + 2 * pad_n;
    std::vector<float> padded((size_t)L_pad);
    for (int t = 0; t < L_high; t++)
        padded[t + pad_n] = har_source[t];
    // PyTorch reflect: 'b a | a b c ... y z | z y' actually non-replicated:
    //   padded[pad - 1 - i] = signal[i + 1]   for i in [0, pad-1]
    //   padded[pad + L + i] = signal[L - 2 - i]
    for (int i = 0; i < pad_n; i++) {
        if (i + 1 < L_high)
            padded[pad_n - 1 - i] = har_source[i + 1];
        if (L_high - 2 - i >= 0)
            padded[pad_n + L_high + i] = har_source[L_high - 2 - i];
    }
    // Hann periodic: w[n] = 0.5 - 0.5 cos(2π n / N).
    std::vector<float> hann((size_t)n_fft);
    for (int n = 0; n < n_fft; n++)
        hann[n] = 0.5f - 0.5f * std::cos(kKokoroTwoPi * (float)n / (float)n_fft);

    const int T_har = L_high / hop + 1;
    // Direct DFT (n_fft=20 makes 20² = 400 ops/frame trivial).
    // Pre-compute twiddles cos/sin for k in 0..n_bins-1, n in 0..n_fft-1.
    std::vector<float> tw_cos((size_t)n_bins * n_fft);
    std::vector<float> tw_sin((size_t)n_bins * n_fft);
    for (int k = 0; k < n_bins; k++) {
        for (int n = 0; n < n_fft; n++) {
            float ang = -kKokoroTwoPi * (float)k * (float)n / (float)n_fft;
            tw_cos[(size_t)k * n_fft + n] = std::cos(ang);
            tw_sin[(size_t)k * n_fft + n] = std::sin(ang);
        }
    }

    float* har = (float*)std::malloc((size_t)22 * T_har * sizeof(float));
    if (!har)
        return nullptr;
    for (int frame = 0; frame < T_har; frame++) {
        const int t0 = frame * hop;
        for (int k = 0; k < n_bins; k++) {
            float re = 0.0f, im = 0.0f;
            const float* tc = &tw_cos[(size_t)k * n_fft];
            const float* ts = &tw_sin[(size_t)k * n_fft];
            for (int n = 0; n < n_fft; n++) {
                const float xn = padded[t0 + n] * hann[n];
                re += xn * tc[n];
                im += xn * ts[n];
            }
            const float mag = std::sqrt(re * re + im * im);
            const float ph = std::atan2(im, re);
            // (22, T_har) channel-major: element (c, t) at offset c + t*22.
            har[(size_t)frame * 22 + k] = mag;
            har[(size_t)frame * 22 + n_bins + k] = ph;
        }
    }
    if (out_T_har)
        *out_T_har = T_har;
    return har;
}

// ---------------------------------------------------------------------------
// Generator graph builder.
//
// Inputs (graph-level):
//   "x_in"   ne=(512, 2*T_frames) F32  — dec_decode_3_out
//   "har"    ne=(22,  T_har)      F32  — pre-computed CPU-side
// Style is baked from the voice pack (decoder half).
//
// Outputs:
//   "gen_pre_post_out"  ne=(128, T_har)  — last leaky_relu output, before conv_post
//   "mag"               ne=(11, T_har)   — exp(x[:11]) after conv_post
//   "phase"             ne=(11, T_har)   — sin(x[11:]) after conv_post
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_generator(kokoro_context* c, int T_frames, int T_har, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // Each AdaINResBlock1 has 3 sub-blocks × ~14 ops (adain ×2 + snake ×2 + conv ×2 + add).
    // 8 such blocks (2 noise_res + 6 resblocks). Plus 2 ups, 2 noise_convs, conv_post.
    // Roughly 8*40 + 6*30 + 60 = 600 ops, comfortably within 32k.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T_frames);
    ggml_set_name(x, "x_in");
    ggml_set_input(x);

    ggml_tensor* har = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 22, T_har);
    ggml_set_name(har, "har");
    ggml_set_input(har);

    ggml_tensor* style = kokoro_voice_style_dec_view(ctx0, c, L_raw); // (128, 1)

    // Load all generator weights up-front.
    ggml_tensor* ups0_w = require(c, "dec.gen.ups.0.weight");
    ggml_tensor* ups0_b = require(c, "dec.gen.ups.0.bias");
    ggml_tensor* ups1_w = require(c, "dec.gen.ups.1.weight");
    ggml_tensor* ups1_b = require(c, "dec.gen.ups.1.bias");
    ggml_tensor* nc0_w = require(c, "dec.gen.noise_convs.0.weight");
    ggml_tensor* nc0_b = require(c, "dec.gen.noise_convs.0.bias");
    ggml_tensor* nc1_w = require(c, "dec.gen.noise_convs.1.weight");
    ggml_tensor* nc1_b = require(c, "dec.gen.noise_convs.1.bias");
    ggml_tensor* cp_w = require(c, "dec.gen.conv_post.weight");
    ggml_tensor* cp_b = require(c, "dec.gen.conv_post.bias");

    const int dilations[3] = {1, 3, 5};
    kokoro_resblock1_w noise_res0, noise_res1;
    kokoro_load_resblock1(c, noise_res0, "dec.gen.noise_res.0", /*K=*/7, dilations);
    kokoro_load_resblock1(c, noise_res1, "dec.gen.noise_res.1", /*K=*/11, dilations);
    kokoro_resblock1_w resblocks[6];
    const int rb_K[3] = {3, 7, 11};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            char prefix[64];
            std::snprintf(prefix, sizeof(prefix), "dec.gen.resblocks.%d", i * 3 + j);
            kokoro_load_resblock1(c, resblocks[i * 3 + j], prefix, rb_K[j], dilations);
        }
    }

    // ---- Upsample loop ----
    for (int i = 0; i < 2; i++) {
        // x = LeakyReLU(0.1)(x)
        x = ggml_leaky_relu(ctx0, x, /*slope=*/0.1f, /*inplace=*/false);

        // x_source = noise_convs[i](har); noise_res[i](x_source, s_dec)
        ggml_tensor* x_source;
        if (i == 0) {
            // k=12, s=6, p=(s+1)/2=3
            x_source = kokoro_conv1d_ks(ctx0, har, nc0_w, nc0_b, /*K*/ 12, /*s*/ 6, /*p*/ 3);
            x_source = kokoro_resblock1_forward(ctx0, x_source, style, noise_res0);
        } else {
            // k=1, s=1, p=0
            x_source = kokoro_conv1d_ks(ctx0, har, nc1_w, nc1_b, /*K*/ 1, /*s*/ 1, /*p*/ 0);
            x_source = kokoro_resblock1_forward(ctx0, x_source, style, noise_res1);
        }

        // x = ups[i](x)  with PyTorch padding handled via post-crop
        if (i == 0) {
            // k=20, s=10, p=(k-s)/2 = 5
            x = kokoro_convt1d_pad(ctx0, x, ups0_w, ups0_b, /*s*/ 10, /*p*/ 5);
        } else {
            // k=12, s=6, p=(k-s)/2 = 3
            x = kokoro_convt1d_pad(ctx0, x, ups1_w, ups1_b, /*s*/ 6, /*p*/ 3);
        }

        // Last upsample: reflection_pad((1, 0)) — 1 sample on the left, 0 on the right.
        // ggml_pad_reflect_1d works on the LAST dim (innermost), so transpose
        // (C, T) → (T, C), pad along T, transpose back.
        if (i == 1) {
            ggml_tensor* xT = ggml_cont(ctx0, ggml_transpose(ctx0, x));  // (T, C)
            xT = ggml_pad_reflect_1d(ctx0, xT, /*left=*/1, /*right=*/0); // (T+1, C)
            x = ggml_cont(ctx0, ggml_transpose(ctx0, xT));               // (C, T+1)
        }

        x = ggml_add(ctx0, x, x_source);

        // Average of 3 resblocks at indices i*3+0, i*3+1, i*3+2.
        ggml_tensor* xs = nullptr;
        for (int j = 0; j < 3; j++) {
            ggml_tensor* xj = kokoro_resblock1_forward(ctx0, x, style, resblocks[i * 3 + j]);
            xs = (xs == nullptr) ? xj : ggml_add(ctx0, xs, xj);
        }
        x = ggml_scale(ctx0, xs, 1.0f / 3.0f);
    }

    // x = LeakyReLU(0.01)(x)  — default PyTorch slope, NOT 0.1.
    x = ggml_leaky_relu(ctx0, x, /*slope=*/0.01f, /*inplace=*/false);
    ggml_tensor* gen_pre_post = ggml_cont(ctx0, x);
    ggml_set_name(gen_pre_post, "gen_pre_post_out");
    ggml_set_output(gen_pre_post);
    ggml_build_forward_expand(gf, gen_pre_post);

    // conv_post: Conv1d(128, 22, k=7, p=3). Same-padding output length = T_har.
    x = kokoro_conv1d_ks(ctx0, gen_pre_post, cp_w, cp_b, /*K*/ 7, /*s*/ 1, /*p*/ 3); // (22, T_har)

    // Split (22, T_har) along ne[0]: rows 0..10 = mag-side, 11..21 = phase-side.
    // View along ne[0] is non-contiguous (stride between rows mismatches ne[0]),
    // so cont after the view to give exp/sin a clean buffer.
    ggml_tensor* mag_view = ggml_view_2d(ctx0, x, 11, T_har, x->nb[1], 0);
    ggml_tensor* mag_in = ggml_cont(ctx0, mag_view);
    ggml_tensor* mag = ggml_exp(ctx0, mag_in);
    ggml_set_name(mag, "mag");
    ggml_set_output(mag);
    ggml_build_forward_expand(gf, mag);

    ggml_tensor* phase_view = ggml_view_2d(ctx0, x, 11, T_har, x->nb[1], (size_t)11 * sizeof(float));
    ggml_tensor* phase_in = ggml_cont(ctx0, phase_view);
    ggml_tensor* phase = ggml_sin(ctx0, phase_in);
    ggml_set_name(phase, "phase");
    ggml_set_output(phase);
    ggml_build_forward_expand(gf, phase);

    ggml_free(ctx0);
    return gf;
}

#ifdef RS_USE_METAL_KOKORO
// ---------------------------------------------------------------------------
// Generator graph, split into 3 sub-graphs around the two ConvTranspose1d
// ops so the custom Metal kernel can replace them (4-5x faster than
// ggml-metal's built-in kernel_conv_transpose_1d).
//
//   Phase A  (in: x_in, har; out: x_leaky_0, x_source_0)
//     ↓
//   Custom Metal ups0 (256, 20T)
//     ↓
//   Phase B  (in: ups0_out, x_source_0, har; out: x_leaky_1, x_source_1)
//     ↓
//   Custom Metal ups1 (128, 120T)
//     ↓
//   Phase C  (in: ups1_out, x_source_1; out: gen_pre_post, mag, phase)
//
// The single-graph builder (kokoro_build_graph_generator) remains the
// fallback for CPU/non-Metal builds and when KokoroMetalConvT1D::init() fails.
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_gen_phase_a(kokoro_context* c, int T_frames, int T_har, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    ggml_tensor* x_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T_frames);
    ggml_set_name(x_in, "x_in");
    ggml_set_input(x_in);
    ggml_tensor* har = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 22, T_har);
    ggml_set_name(har, "har");
    ggml_set_input(har);

    ggml_tensor* style = kokoro_voice_style_dec_view(ctx0, c, L_raw);

    ggml_tensor* nc0_w = require(c, "dec.gen.noise_convs.0.weight");
    ggml_tensor* nc0_b = require(c, "dec.gen.noise_convs.0.bias");
    const int dilations[3] = {1, 3, 5};
    kokoro_resblock1_w noise_res0;
    kokoro_load_resblock1(c, noise_res0, "dec.gen.noise_res.0", /*K=*/7, dilations);

    // x_leaky_0 needs to be in (T, C) layout for the Metal convt kernel
    // (kernel reads `in[ic*Ti+i]` — t-fastest in memory). leaky_relu preserves
    // (C, T) so transpose+cont before output.
    ggml_tensor* x_leaky_0 = ggml_leaky_relu(ctx0, x_in, /*slope=*/0.1f, /*inplace=*/false);
    x_leaky_0 = ggml_cont(ctx0, ggml_transpose(ctx0, x_leaky_0));
    ggml_set_name(x_leaky_0, "x_leaky_0");
    ggml_set_output(x_leaky_0);
    ggml_build_forward_expand(gf, x_leaky_0);

    ggml_tensor* x_source_0 = kokoro_conv1d_ks(ctx0, har, nc0_w, nc0_b, /*K*/ 12, /*s*/ 6, /*p*/ 3);
    x_source_0 = kokoro_resblock1_forward(ctx0, x_source_0, style, noise_res0);
    x_source_0 = ggml_cont(ctx0, x_source_0);
    ggml_set_name(x_source_0, "x_source_0");
    ggml_set_output(x_source_0);
    ggml_build_forward_expand(gf, x_source_0);

    ggml_free(ctx0);
    return gf;
}

static ggml_cgraph* kokoro_build_graph_gen_phase_b(kokoro_context* c, int T_frames, int T_har, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // ups0_out memory layout from Metal kernel: T_out fastest, OC slow —
    // declare matching that order, then transpose+cont inside the graph to
    // (C, T). x_source_0 was produced by Phase A in standard (C, T) layout.
    ggml_tensor* ups0_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 20 * T_frames, 256);
    ggml_set_name(ups0_in, "ups0_out");
    ggml_set_input(ups0_in);
    ggml_tensor* x_source_0 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 256, 20 * T_frames);
    ggml_set_name(x_source_0, "x_source_0");
    ggml_set_input(x_source_0);
    ggml_tensor* har = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 22, T_har);
    ggml_set_name(har, "har");
    ggml_set_input(har);

    ggml_tensor* ups0_out = ggml_cont(ctx0, ggml_transpose(ctx0, ups0_in));

    ggml_tensor* style = kokoro_voice_style_dec_view(ctx0, c, L_raw);

    ggml_tensor* nc1_w = require(c, "dec.gen.noise_convs.1.weight");
    ggml_tensor* nc1_b = require(c, "dec.gen.noise_convs.1.bias");
    const int dilations[3] = {1, 3, 5};
    kokoro_resblock1_w noise_res1;
    kokoro_load_resblock1(c, noise_res1, "dec.gen.noise_res.1", /*K=*/11, dilations);
    kokoro_resblock1_w resblocks[3];
    const int rb_K[3] = {3, 7, 11};
    for (int j = 0; j < 3; j++) {
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "dec.gen.resblocks.%d", j);
        kokoro_load_resblock1(c, resblocks[j], prefix, rb_K[j], dilations);
    }

    ggml_tensor* x = ggml_add(ctx0, ups0_out, x_source_0);

    ggml_tensor* xs = nullptr;
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xj = kokoro_resblock1_forward(ctx0, x, style, resblocks[j]);
        xs = (xs == nullptr) ? xj : ggml_add(ctx0, xs, xj);
    }
    x = ggml_scale(ctx0, xs, 1.0f / 3.0f);

    // x_leaky_1 needs to be in (T, C) layout for the Metal convt kernel.
    ggml_tensor* x_leaky_1 = ggml_leaky_relu(ctx0, x, /*slope=*/0.1f, /*inplace=*/false);
    x_leaky_1 = ggml_cont(ctx0, ggml_transpose(ctx0, x_leaky_1));
    ggml_set_name(x_leaky_1, "x_leaky_1");
    ggml_set_output(x_leaky_1);
    ggml_build_forward_expand(gf, x_leaky_1);

    // x_source for iteration 1: noise_convs[1](k=1, s=1, p=0) → noise_res[1]
    ggml_tensor* x_source_1 = kokoro_conv1d_ks(ctx0, har, nc1_w, nc1_b, /*K*/ 1, /*s*/ 1, /*p*/ 0);
    x_source_1 = kokoro_resblock1_forward(ctx0, x_source_1, style, noise_res1);
    x_source_1 = ggml_cont(ctx0, x_source_1);
    ggml_set_name(x_source_1, "x_source_1");
    ggml_set_output(x_source_1);
    ggml_build_forward_expand(gf, x_source_1);

    ggml_free(ctx0);
    return gf;
}

static ggml_cgraph* kokoro_build_graph_gen_phase_c(kokoro_context* c, int T_frames, int T_har, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // ups1_out memory layout from Metal kernel: T_out fastest, OC slow.
    // Declared ne[0]=T (matching the kernel's write order). x_source_1 from
    // Phase B is standard (C, T) ggml convention.
    const int T_post_ups1 = (20 * T_frames - 1) * 6 + 12 - 6; // = 120*T_frames
    ggml_tensor* ups1_out = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_post_ups1, 128);
    ggml_set_name(ups1_out, "ups1_out");
    ggml_set_input(ups1_out);
    ggml_tensor* x_source_1 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 128, T_har);
    ggml_set_name(x_source_1, "x_source_1");
    ggml_set_input(x_source_1);

    ggml_tensor* style = kokoro_voice_style_dec_view(ctx0, c, L_raw);

    ggml_tensor* cp_w = require(c, "dec.gen.conv_post.weight");
    ggml_tensor* cp_b = require(c, "dec.gen.conv_post.bias");
    const int dilations[3] = {1, 3, 5};
    kokoro_resblock1_w resblocks[3];
    const int rb_K[3] = {3, 7, 11};
    for (int j = 0; j < 3; j++) {
        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "dec.gen.resblocks.%d", 3 + j);
        kokoro_load_resblock1(c, resblocks[j], prefix, rb_K[j], dilations);
    }

    // ups1_out is already (T, C) — pad_reflect_1d pads along ne[0]=T directly.
    ggml_tensor* xT = ggml_pad_reflect_1d(ctx0, ups1_out, /*left=*/1, /*right=*/0); // (T+1, C)
    ggml_tensor* x = ggml_cont(ctx0, ggml_transpose(ctx0, xT));                     // (C, T+1)

    x = ggml_add(ctx0, x, x_source_1);

    ggml_tensor* xs = nullptr;
    for (int j = 0; j < 3; j++) {
        ggml_tensor* xj = kokoro_resblock1_forward(ctx0, x, style, resblocks[j]);
        xs = (xs == nullptr) ? xj : ggml_add(ctx0, xs, xj);
    }
    x = ggml_scale(ctx0, xs, 1.0f / 3.0f);

    x = ggml_leaky_relu(ctx0, x, /*slope=*/0.01f, /*inplace=*/false);
    ggml_tensor* gen_pre_post = ggml_cont(ctx0, x);
    ggml_set_name(gen_pre_post, "gen_pre_post_out");
    ggml_set_output(gen_pre_post);
    ggml_build_forward_expand(gf, gen_pre_post);

    x = kokoro_conv1d_ks(ctx0, gen_pre_post, cp_w, cp_b, /*K*/ 7, /*s*/ 1, /*p*/ 3);

    ggml_tensor* mag_view = ggml_view_2d(ctx0, x, 11, T_har, x->nb[1], 0);
    ggml_tensor* mag_in = ggml_cont(ctx0, mag_view);
    ggml_tensor* mag = ggml_exp(ctx0, mag_in);
    ggml_set_name(mag, "mag");
    ggml_set_output(mag);
    ggml_build_forward_expand(gf, mag);

    ggml_tensor* phase_view = ggml_view_2d(ctx0, x, 11, T_har, x->nb[1], (size_t)11 * sizeof(float));
    ggml_tensor* phase_in = ggml_cont(ctx0, phase_view);
    ggml_tensor* phase = ggml_sin(ctx0, phase_in);
    ggml_set_name(phase, "phase");
    ggml_set_output(phase);
    ggml_build_forward_expand(gf, phase);

    ggml_free(ctx0);
    return gf;
}

// Helper: run a sub-graph, copy out a named tensor (F32) to CPU.
// Returns malloc'd buffer of n_floats float elements; caller frees.
static bool kokoro_run_subgraph_get(kokoro_context* c, ggml_cgraph* gf, const char* out_name,
                                    std::vector<float>& out_buf) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, out_name);
    if (!t) { fprintf(stderr, "kokoro: missing sub-graph output '%s'\n", out_name); return false; }
    const size_t n = (size_t)t->ne[0] * (size_t)t->ne[1];
    out_buf.resize(n);
    ggml_backend_tensor_get(t, out_buf.data(), 0, n * sizeof(float));
    return true;
}

// Run the generator using the 3-sub-graph + custom Metal kernel path.
// On success, the cache contains "generator:mag" and "generator:phase".
static bool kokoro_run_generator_metal_convt(kokoro_context* c, int T_frames, int T_har, int n_raw,
                                             const float* x_in, const float* har) {
    auto set_inputs = [&](ggml_cgraph* gf, std::initializer_list<std::pair<const char*, const float*>> items,
                          std::initializer_list<size_t> sizes) {
        auto it = items.begin();
        auto sit = sizes.begin();
        for (; it != items.end(); ++it, ++sit) {
            ggml_tensor* t = ggml_graph_get_tensor(gf, it->first);
            if (!t) { fprintf(stderr, "kokoro: sub-graph input '%s' missing\n", it->first); return false; }
            ggml_backend_tensor_set(t, it->second, 0, *sit * sizeof(float));
        }
        return true;
    };

    // --- Phase A ---
    ggml_cgraph* gA = kokoro_build_graph_gen_phase_a(c, T_frames, T_har, n_raw);
    if (!gA) return false;
    ggml_backend_sched_reset(c->gen_sched);
    if (!ggml_backend_sched_alloc_graph(c->gen_sched, gA)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for phase A\n");
        return false;
    }
    if (!set_inputs(gA, {{"x_in", x_in}, {"har", har}},
                    {(size_t)512 * 2 * T_frames, (size_t)22 * T_har})) return false;
    if (kokoro_sched_compute(c, c->gen_sched, gA) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: phase A compute failed\n");
        return false;
    }
    std::vector<float> x_leaky_0, x_source_0;
    if (!kokoro_run_subgraph_get(c, gA, "x_leaky_0", x_leaky_0)) return false;
    if (!kokoro_run_subgraph_get(c, gA, "x_source_0", x_source_0)) return false;

    // --- Custom Metal convt0 ---
    const int T_ups0_out = c->metal_convt->output_T(0, 2 * T_frames); // = 20 * T_frames
    std::vector<float> ups0_out((size_t)256 * T_ups0_out);
    if (!c->metal_convt->run(0, x_leaky_0.data(), 2 * T_frames, ups0_out.data())) {
        fprintf(stderr, "kokoro: metal convt0 failed\n");
        return false;
    }

    // --- Phase B ---
    ggml_cgraph* gB = kokoro_build_graph_gen_phase_b(c, T_frames, T_har, n_raw);
    if (!gB) return false;
    ggml_backend_sched_reset(c->gen_sched);
    if (!ggml_backend_sched_alloc_graph(c->gen_sched, gB)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for phase B\n");
        return false;
    }
    if (!set_inputs(gB,
                    {{"ups0_out", ups0_out.data()}, {"x_source_0", x_source_0.data()}, {"har", har}},
                    {(size_t)256 * T_ups0_out, (size_t)256 * T_ups0_out, (size_t)22 * T_har}))
        return false;
    if (kokoro_sched_compute(c, c->gen_sched, gB) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: phase B compute failed\n");
        return false;
    }
    std::vector<float> x_leaky_1, x_source_1;
    if (!kokoro_run_subgraph_get(c, gB, "x_leaky_1", x_leaky_1)) return false;
    if (!kokoro_run_subgraph_get(c, gB, "x_source_1", x_source_1)) return false;

    // --- Custom Metal convt1 ---
    const int T_ups1_out = c->metal_convt->output_T(1, T_ups0_out); // = 120 * T_frames
    std::vector<float> ups1_out((size_t)128 * T_ups1_out);
    if (!c->metal_convt->run(1, x_leaky_1.data(), T_ups0_out, ups1_out.data())) {
        fprintf(stderr, "kokoro: metal convt1 failed\n");
        return false;
    }

    // --- Phase C ---
    ggml_cgraph* gC = kokoro_build_graph_gen_phase_c(c, T_frames, T_har, n_raw);
    if (!gC) return false;
    ggml_backend_sched_reset(c->gen_sched);
    if (!ggml_backend_sched_alloc_graph(c->gen_sched, gC)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for phase C\n");
        return false;
    }
    if (!set_inputs(gC,
                    {{"ups1_out", ups1_out.data()}, {"x_source_1", x_source_1.data()}},
                    {(size_t)128 * T_ups1_out, (size_t)128 * T_har}))
        return false;
    if (kokoro_sched_compute(c, c->gen_sched, gC) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: phase C compute failed\n");
        return false;
    }
    ggml_tensor* out_mag = ggml_graph_get_tensor(gC, "mag");
    ggml_tensor* out_phase = ggml_graph_get_tensor(gC, "phase");
    if (!out_mag || !out_phase) {
        fprintf(stderr, "kokoro: phase C missing mag/phase\n");
        return false;
    }
    auto cache_one = [&](ggml_tensor* t, const char* name) {
        const size_t n = (size_t)t->ne[0] * (size_t)t->ne[1];
        std::vector<float> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
        g_kokoro_cache.put(std::string("generator:") + name, buf.data(), (int)n);
    };
    cache_one(out_mag, "mag");
    cache_one(out_phase, "phase");
    return true;
}
#endif // RS_USE_METAL_KOKORO

// Run the full pre-generator chain (predictor → align → F0Ntrain → decoder
// body) plus the generator graph itself, and return the named stage as
// malloc'd float[]. Stages handled:
//   "gen_pre_post_out" → (128, T_har)
//   "mag"              → (11, T_har)
//   "phase"            → (11, T_har)
//
// rng is seeded from KOKORO_SEED env (default 0x12345) so the SineGen noise
// is reproducible run-to-run; the diff-harness M11 dumper should set the
// same seed on the Python side.
static float* kokoro_run_generator(kokoro_context* c, const int32_t* raw_ids, int n_raw, const char* stage_name,
                                   int* out_n) {
    KOKORO_PROFILE("generator_total");
    if (out_n)
        *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: generator needs voice pack\n");
        return nullptr;
    }

    // Cache hit fast-path: avoid the whole graph rebuild + compute.
    {
        std::string key = std::string("generator:") + stage_name;
        if (g_kokoro_cache.has(key)) return g_kokoro_cache.get_copy(key, out_n);
    }

    // 1. dec_decode_3_out — runs predictor + align + F0Ntrain + decoder body.
    int n_x = 0;
    float* x_in = kokoro_run_decoder_body(c, raw_ids, n_raw, "dec_decode_3_out", &n_x);
    if (!x_in)
        return nullptr;
    const int two_T = n_x / 512;
    if (two_T * 512 != n_x) {
        fprintf(stderr, "kokoro: dec_decode_3_out size %d not divisible by 512\n", n_x);
        std::free(x_in);
        return nullptr;
    }
    const int T_frames = two_T / 2;
    if (T_frames <= 0) {
        fprintf(stderr, "kokoro: T_frames=%d invalid\n", T_frames);
        std::free(x_in);
        return nullptr;
    }

    // 2. f0_curve — used by SineGen to build `har`.
    int n_f0 = 0;
    float* f0 = kokoro_run_f0n(c, raw_ids, n_raw, "f0_curve", &n_f0);
    if (!f0) {
        std::free(x_in);
        return nullptr;
    }
    if (n_f0 != 2 * T_frames) {
        fprintf(stderr, "kokoro: f0_curve length %d != 2*T_frames=%d\n", n_f0, 2 * T_frames);
        std::free(x_in);
        std::free(f0);
        return nullptr;
    }

    // 3. Build `har` (22, T_har) on CPU.
    const char* seed_env = std::getenv("KOKORO_SEED");
    uint32_t seed = seed_env ? (uint32_t)std::strtoul(seed_env, nullptr, 0) : 0x12345u;
    std::mt19937 rng(seed);
    int T_har = 0;
    float* har = kokoro_make_har(c, f0, T_frames, &T_har, rng);
    std::free(f0);
    if (!har) {
        std::free(x_in);
        return nullptr;
    }

    // 4. Build + run the generator graph on gen_sched.
#ifdef RS_USE_METAL_KOKORO
    if (c->metal_convt && c->metal_convt->is_valid()) {
        KOKORO_PROFILE("generator_metal_convt");
        bool ok = kokoro_run_generator_metal_convt(c, T_frames, T_har, n_raw, x_in, har);
        std::free(x_in);
        std::free(har);
        if (ok) {
            std::string key = std::string("generator:") + stage_name;
            if (!g_kokoro_cache.has(key)) {
                fprintf(stderr, "kokoro: unknown generator stage '%s'\n", stage_name);
                return nullptr;
            }
            return g_kokoro_cache.get_copy(key, out_n);
        }
        fprintf(stderr, "kokoro: metal_convt path failed, no fallback (x_in/har freed)\n");
        return nullptr;
    }
#endif
    ggml_cgraph* gf = kokoro_build_graph_generator(c, T_frames, T_har, n_raw);
    if (!gf) {
        std::free(x_in);
        std::free(har);
        return nullptr;
    }
    ggml_backend_sched_reset(c->gen_sched);
    if (!ggml_backend_sched_alloc_graph(c->gen_sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for generator\n");
        std::free(x_in);
        std::free(har);
        return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x_in"), x_in, 0, (size_t)512 * 2 * T_frames * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "har"), har, 0, (size_t)22 * T_har * sizeof(float));
    std::free(x_in);
    std::free(har);

    {
        KOKORO_PROFILE("generator_graph");
        if (kokoro_sched_compute(c, c->gen_sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "kokoro: generator graph compute failed\n");
            return nullptr;
        }
    }

    ggml_tensor* out_mag = ggml_graph_get_tensor(gf, "mag");
    ggml_tensor* out_phase = ggml_graph_get_tensor(gf, "phase");
    if (!out_mag || !out_phase) {
        fprintf(stderr, "kokoro: generator graph missing mag/phase outputs\n");
        return nullptr;
    }
    auto cache_one = [&](ggml_tensor* t, const char* name) {
        const size_t n_floats = (size_t)t->ne[0] * (size_t)t->ne[1];
        std::vector<float> buf(n_floats);
        ggml_backend_tensor_get(t, buf.data(), 0, n_floats * sizeof(float));
        g_kokoro_cache.put(std::string("generator:") + name, buf.data(), (int)n_floats);
    };
    cache_one(out_mag, "mag");
    cache_one(out_phase, "phase");

    std::string key = std::string("generator:") + stage_name;
    if (!g_kokoro_cache.has(key)) {
        fprintf(stderr, "kokoro: unknown generator stage '%s'\n", stage_name);
        return nullptr;
    }
    return g_kokoro_cache.get_copy(key, out_n);
}

// ---------------------------------------------------------------------------
// M8 — iSTFT (CPU)
//
// Reference: torch.istft(filter_length=20, hop=5, win_length=20,
//                        window=hann_window(20, periodic=True), center=True).
//
// For each frame m in 0..T_har-1:
//   X[k] = mag[k, m] * exp(j * phase[k, m])    for k in 0..10  (n_bins)
//   Mirror to full 20-bin: X[20-k] = conj(X[k]) for k in 1..9
//   y_frame[n] = (1/N) * Re(sum_k X[k] * exp(j*2π*k*n/N))      for n in 0..19
//   y_frame *= window
//   out[m*hop : m*hop + N] += y_frame
//   ola_norm[m*hop : m*hop + N] += window²
// out /= ola_norm  (where ola_norm > 0); strip pad samples.
// Output length = (T_har - 1) * hop = 600 * T_frames.
// ---------------------------------------------------------------------------

static float* kokoro_run_istft(const float* mag, const float* phase, int T_har, int* out_T_audio) {
    KOKORO_PROFILE("istft");
    const int n_fft = 20;
    const int hop = 5;
    const int n_bins = n_fft / 2 + 1;
    const int pad_n = n_fft / 2;

    // Hann periodic.
    std::vector<float> hann((size_t)n_fft);
    for (int n = 0; n < n_fft; n++)
        hann[n] = 0.5f - 0.5f * std::cos(kKokoroTwoPi * (float)n / (float)n_fft);

    // L_padded covers the OLA region; L_audio is the unpadded output (center=True
    // strips n_fft/2 samples from each end).
    const int L_padded = (T_har - 1) * hop + n_fft;
    const int L_audio = L_padded - 2 * pad_n;
    if (L_audio <= 0) {
        if (out_T_audio)
            *out_T_audio = 0;
        return nullptr;
    }

    std::vector<float> y((size_t)L_padded, 0.0f);
    std::vector<float> wsum((size_t)L_padded, 0.0f);

    // Pre-compute IDFT twiddles: cos/sin(2π k n / N) for k in 0..n_bins-1.
    // For k in [n_bins, n_fft-1] we exploit Hermitian symmetry (see below).
    std::vector<float> tw_cos((size_t)n_bins * n_fft);
    std::vector<float> tw_sin((size_t)n_bins * n_fft);
    for (int k = 0; k < n_bins; k++) {
        for (int n = 0; n < n_fft; n++) {
            float ang = kKokoroTwoPi * (float)k * (float)n / (float)n_fft;
            tw_cos[(size_t)k * n_fft + n] = std::cos(ang);
            tw_sin[(size_t)k * n_fft + n] = std::sin(ang);
        }
    }

    // Per frame: IDFT directly from half-spectrum using Hermitian symmetry.
    // For real-valued y[n]:
    //   y[n] = (1/N) * (Xr[0] + (-1)^n * Xr[N/2]
    //                   + 2 * sum_{k=1}^{N/2-1} (Xr[k]*cos(2πkn/N) - Xi[k]*sin(2πkn/N)))
    const float invN = 1.0f / (float)n_fft;
    for (int m = 0; m < T_har; m++) {
        // mag/phase have layout (n_bins, T_har) F32 contiguous: element (k, m) at k + m*n_bins.
        const int t0 = m * hop;
        for (int n = 0; n < n_fft; n++) {
            float acc = mag[(size_t)m * n_bins + 0]; // X[0] (real)
            // X[N/2]: real, sign alternates with n.
            const float xN2 =
                mag[(size_t)m * n_bins + (n_bins - 1)] * std::cos(phase[(size_t)m * n_bins + (n_bins - 1)]);
            acc += ((n & 1) ? -xN2 : xN2);
            // k = 1..N/2-1: doubled Hermitian pair.
            for (int k = 1; k < n_bins - 1; k++) {
                const float r = mag[(size_t)m * n_bins + k];
                const float ph = phase[(size_t)m * n_bins + k];
                const float xr = r * std::cos(ph);
                const float xi = r * std::sin(ph);
                acc += 2.0f * (xr * tw_cos[(size_t)k * n_fft + n] - xi * tw_sin[(size_t)k * n_fft + n]);
            }
            const float yval = acc * invN * hann[n];
            y[(size_t)t0 + n] += yval;
            wsum[(size_t)t0 + n] += hann[n] * hann[n];
        }
    }

    // Normalize by window-sum and strip the center=True padding.
    float* out = (float*)std::malloc((size_t)L_audio * sizeof(float));
    if (!out) {
        if (out_T_audio)
            *out_T_audio = 0;
        return nullptr;
    }
    const float wsum_eps = 1e-11f;
    for (int t = 0; t < L_audio; t++) {
        const float w = wsum[(size_t)pad_n + t];
        out[t] = (w > wsum_eps) ? y[(size_t)pad_n + t] / w : 0.0f;
    }
    if (out_T_audio)
        *out_T_audio = L_audio;
    return out;
}

// Stage extractor entry for the iSTFT'd audio. Runs the full chain
// (preamble → decoder body → generator → iSTFT) and returns the audio buffer.
static float* kokoro_run_audio(kokoro_context* c, const int32_t* raw_ids, int n_raw, int* out_n) {
    g_kokoro_prof.init();
    g_kokoro_cache.reset();
    KOKORO_PROFILE("AUDIO_TOTAL");
    if (out_n)
        *out_n = 0;
    int n_mag = 0;
    float* mag = kokoro_run_generator(c, raw_ids, n_raw, "mag", &n_mag);
    if (!mag)
        return nullptr;
    int n_phase = 0;
    float* phase = kokoro_run_generator(c, raw_ids, n_raw, "phase", &n_phase);
    if (!phase) {
        std::free(mag);
        return nullptr;
    }
    if (n_mag != n_phase || n_mag % 11 != 0) {
        fprintf(stderr, "kokoro: mag/phase length mismatch %d vs %d\n", n_mag, n_phase);
        std::free(mag);
        std::free(phase);
        return nullptr;
    }
    const int T_har = n_mag / 11;
    int T_audio = 0;
    float* audio = kokoro_run_istft(mag, phase, T_har, &T_audio);
    std::free(mag);
    std::free(phase);
    if (!audio)
        return nullptr;
    if (out_n)
        *out_n = T_audio;
    // Dump profile before returning. The AUDIO_TOTAL scope guard runs *after*
    // this so its sample includes only the synth path (not the dump itself).
    g_kokoro_prof.dump();
    return audio;
}

} // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

static struct kokoro_context_params kokoro_context_default_params(void) {
    kokoro_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    // The generator now always runs through gen_sched with the user's GPU
    // backend as primary and CPU as fallback. The old `gen_force_metal`
    // flag and KOKORO_GEN_FORCE_METAL / KOKORO_GEN_GPU / KOKORO_GEN_CPU
    // env vars are kept as inert no-ops for backwards compatibility — the
    // Metal-hang workaround they used to gate is gone (upstream ggml
    // PR #1477, commit a056a26f, eliminated the M1 ConvTranspose1d
    // watchdog hang). Weight buffers live on the GPU in the rs_context
    // borrow path, so a true CPU-only escape hatch isn't possible without
    // a duplicate CPU weight copy.
    p.gen_force_metal = true;
    p.flash_attn = true;
    p.length_scale = 1.0f;
    return p;
}


static int kokoro_load_voice_pack(struct kokoro_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    // Read voice metadata: kokoro_voice.{name,max_phonemes,style_dim}.
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            fprintf(stderr, "kokoro: failed to read voice pack '%s'\n", path);
            return -1;
        }

        std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "kokoro-voice") {
            fprintf(stderr, "kokoro: voice pack '%s' has architecture='%s' (want 'kokoro-voice')\n", path,
                    arch.c_str());
            gguf_free(g);
            return -1;
        }

        kokoro_voice_pack vp;
        vp.name = core_gguf::kv_str(g, "kokoro_voice.name", "");
        vp.max_phonemes = core_gguf::kv_u32(g, "kokoro_voice.max_phonemes", 0);
        vp.style_dim = core_gguf::kv_u32(g, "kokoro_voice.style_dim", 0);
        gguf_free(g);

        // Replace any previously-loaded pack.
        if (ctx->vp.vp_buf_w)
            ggml_backend_buffer_free(ctx->vp.vp_buf_w);
        if (ctx->vp.vp_ctx_w)
            ggml_free(ctx->vp.vp_ctx_w);
        ctx->vp = std::move(vp);
        ctx->vp_loaded = false;
    }

    // Load the single F32 tensor `voice.pack`. Use the main `backend` so
    // ggml_get_rows / ggml_view_2d access works without a backend hop
    // during graph build.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "kokoro.voice", wl)) {
        fprintf(stderr, "kokoro: failed to load voice pack tensors from '%s'\n", path);
        return -1;
    }

    auto it = wl.tensors.find("voice.pack");
    if (it == wl.tensors.end() || !it->second) {
        fprintf(stderr, "kokoro: voice pack '%s' missing 'voice.pack' tensor\n", path);
        ggml_backend_buffer_free(wl.buf);
        ggml_free(wl.ctx);
        return -1;
    }

    ctx->vp.tensors = std::move(wl.tensors);
    ctx->vp.pack = it->second;
    ctx->vp.vp_ctx_w = wl.ctx;
    ctx->vp.vp_buf_w = wl.buf;
    ctx->vp_loaded = true;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kokoro: voice '%s' (max_phonemes=%u style_dim=%u) loaded from '%s'\n", ctx->vp.name.c_str(),
                ctx->vp.max_phonemes, ctx->vp.style_dim, path);
    }
    return 0;
}

static int32_t* kokoro_phonemes_to_ids(struct kokoro_context* ctx, const char* phonemes, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !phonemes)
        return nullptr;

    // The vocab is 178 IPA symbols, each typically 1 codepoint (1-4 UTF-8
    // bytes). We greedy-tokenise: at each position, try the longest
    // matching token first (combining marks like ̃ or ː are 2 bytes and
    // attach to the previous letter, so they tokenise as their own ids
    // when they appear standalone). Unknown codepoints emit pad_id and a
    // stderr warning.
    std::vector<int32_t> ids;
    ids.reserve(std::strlen(phonemes));

    auto utf8_len = [](unsigned char c) -> int {
        if ((c & 0x80) == 0)
            return 1;
        if ((c & 0xE0) == 0xC0)
            return 2;
        if ((c & 0xF0) == 0xE0)
            return 3;
        if ((c & 0xF8) == 0xF0)
            return 4;
        return 1; // invalid leading byte — consume one byte and move on
    };

    const char* p = phonemes;
    while (*p) {
        // Try lookahead: 4-byte → 3-byte → 2-byte → 1-byte greedy.
        // Combining marks following a base letter can form 2-codepoint
        // tokens (e.g. "̃" follows a base) — we don't try to combine them
        // because StyleTTS2's vocab stores them as standalone entries.
        int best_len = 0;
        int best_id = -1;
        for (int try_len = 4; try_len >= 1; try_len--) {
            // Fast path: skip if the current byte's UTF-8 length forbids it.
            const int real_len = utf8_len((unsigned char)*p);
            if (try_len > real_len)
                continue;
            std::string tok(p, (size_t)try_len);
            auto it = ctx->vocab.token_to_id.find(tok);
            if (it != ctx->vocab.token_to_id.end()) {
                best_len = try_len;
                best_id = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            p += best_len;
        } else {
            // Match the reference KModel.forward() behaviour: unknown
            // phonemes are dropped (Python uses
            // `filter(lambda i: i is not None, map(vocab.get, phonemes))`).
            // Emitting pad here would diverge from the reference and
            // poison every downstream stage's diff.
            const int n = utf8_len((unsigned char)*p);
            if (ctx->params.verbosity >= 1) {
                std::string bad(p, (size_t)n);
                fprintf(stderr, "kokoro: unknown phoneme '%s' — skipped\n", bad.c_str());
            }
            p += n;
        }
    }

    int32_t* out = (int32_t*)std::malloc(ids.size() * sizeof(int32_t));
    if (!out)
        return nullptr;
    std::memcpy(out, ids.data(), ids.size() * sizeof(int32_t));
    if (out_n)
        *out_n = (int)ids.size();
    return out;
}

static float* kokoro_synthesize_phonemes(struct kokoro_context* ctx, const char* phonemes, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !phonemes)
        return nullptr;

    int n_ids = 0;
    int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
    if (!ids || n_ids == 0) {
        std::free(ids);
        fprintf(stderr, "kokoro: empty phoneme tokenisation for '%s'\n", phonemes);
        return nullptr;
    }
    int n_audio = 0;
    float* audio = kokoro_run_audio(ctx, ids, n_ids, &n_audio);
    std::free(ids);
    if (!audio || n_audio <= 0)
        return nullptr;
    if (out_n_samples)
        *out_n_samples = n_audio;
    return audio;
}


static void kokoro_pcm_free(float* pcm) {
    std::free(pcm);
}

static void kokoro_set_n_threads(struct kokoro_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

// Runtime length-scale setter (PLAN #88). The duration-predictor
// output gets multiplied by this scalar BEFORE the banker's round +
// clamp-min-1 in the "durations" stage extractor. Read on every
// synthesize call, so post-init mutation just changes the next
// call's pacing.
static void kokoro_set_length_scale(struct kokoro_context* ctx, float scale) {
    if (!ctx)
        return;
    if (scale < 0.25f)
        scale = 0.25f;
    if (scale > 4.0f)
        scale = 4.0f;
    ctx->params.length_scale = scale;
}

static void kokoro_free(struct kokoro_context* ctx) {
    if (!ctx)
        return;
    if (ctx->gen_sched)
        ggml_backend_sched_free(ctx->gen_sched);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->vp.vp_buf_w)
        ggml_backend_buffer_free(ctx->vp.vp_buf_w);
    if (ctx->vp.vp_ctx_w)
        ggml_free(ctx->vp.vp_ctx_w);
    if (ctx->owns_gguf) {
        if (ctx->buf_w)
            ggml_backend_buffer_free(ctx->buf_w);
        if (ctx->ctx_w)
            ggml_free(ctx->ctx_w);
    }
    if (ctx->owns_backend) {
        if (ctx->backend && ctx->backend != ctx->backend_cpu)
            ggml_backend_free(ctx->backend);
        if (ctx->backend_cpu)
            ggml_backend_free(ctx->backend_cpu);
    }
    delete ctx;
}

// ---------------------------------------------------------------------------
// Adapter-friendly init: re-use a pre-loaded GGUF (owned by rs_context_t).
// Skips fopen / gguf_init_from_file / backend init / weight upload. Tensor
// data is looked up via ggml_get_tensor(src_gguf_data, name) into our
// tensors map; the gguf itself is borrowed (not freed in kokoro_free).
// ---------------------------------------------------------------------------

static struct kokoro_context* kokoro_init_from_rs_context(
    struct ggml_context* src_gguf_data,
    struct gguf_context* src_gguf_ctx,
    ggml_backend_t user_backend,
    ggml_backend_t cpu_backend,
    struct kokoro_context_params params) {
    if (!src_gguf_data || !src_gguf_ctx || !cpu_backend) {
        fprintf(stderr, "kokoro: kokoro_init_from_rs_context: missing required argument\n");
        return nullptr;
    }
    std::fesetround(FE_TONEAREST);

    auto* c = new kokoro_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    c->owns_gguf = false;
    c->owns_backend = false;

    gguf_context* g = src_gguf_ctx;

    // ---- Metadata: hyperparameters + vocab ----
    {
        std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "kokoro") {
            fprintf(stderr, "kokoro: unexpected general.architecture='%s' (want 'kokoro')\n", arch.c_str());
            delete c;
            return nullptr;
        }

        auto& hp = c->hp;
        hp.hidden_dim = core_gguf::kv_u32(g, "kokoro.hidden_dim", hp.hidden_dim);
        hp.style_dim = core_gguf::kv_u32(g, "kokoro.style_dim", hp.style_dim);
        hp.max_dur = core_gguf::kv_u32(g, "kokoro.max_dur", hp.max_dur);
        hp.n_token = core_gguf::kv_u32(g, "kokoro.n_token", hp.n_token);
        hp.n_mels = core_gguf::kv_u32(g, "kokoro.n_mels", hp.n_mels);
        hp.n_layer = core_gguf::kv_u32(g, "kokoro.n_layer", hp.n_layer);
        hp.text_enc_k = core_gguf::kv_u32(g, "kokoro.text_encoder_kernel_size", hp.text_enc_k);
        hp.sample_rate = core_gguf::kv_u32(g, "kokoro.sample_rate", hp.sample_rate);
        hp.vocab_size = core_gguf::kv_u32(g, "kokoro.vocab_size", hp.vocab_size);

        hp.plbert_embd_size = core_gguf::kv_u32(g, "kokoro.plbert.embedding_size", hp.plbert_embd_size);
        hp.plbert_hidden = core_gguf::kv_u32(g, "kokoro.plbert.hidden_size", hp.plbert_hidden);
        hp.plbert_n_layers = core_gguf::kv_u32(g, "kokoro.plbert.num_hidden_layers", hp.plbert_n_layers);
        hp.plbert_n_heads = core_gguf::kv_u32(g, "kokoro.plbert.num_attention_heads", hp.plbert_n_heads);
        hp.plbert_ff = core_gguf::kv_u32(g, "kokoro.plbert.intermediate_size", hp.plbert_ff);
        hp.plbert_max_pos = core_gguf::kv_u32(g, "kokoro.plbert.max_position_embeddings", hp.plbert_max_pos);
        hp.plbert_vocab_size = core_gguf::kv_u32(g, "kokoro.plbert.vocab_size", hp.plbert_vocab_size);

        hp.istft_init_ch = core_gguf::kv_u32(g, "kokoro.istft.init_channel", hp.istft_init_ch);
        hp.istft_n_fft = core_gguf::kv_u32(g, "kokoro.istft.n_fft", hp.istft_n_fft);
        hp.istft_hop = core_gguf::kv_u32(g, "kokoro.istft.hop_size", hp.istft_hop);
        hp.istft_n_dilations = core_gguf::kv_u32(g, "kokoro.istft.resblock_n_dilations", hp.istft_n_dilations);

        hp.istft_upsample_rates = kv_u32_array(g, "kokoro.istft.upsample_rates");
        hp.istft_upsample_kernel_sizes = kv_u32_array(g, "kokoro.istft.upsample_kernel_sizes");
        hp.istft_resblock_kernel_sizes = kv_u32_array(g, "kokoro.istft.resblock_kernel_sizes");
        hp.istft_resblock_dilations = kv_u32_array(g, "kokoro.istft.resblock_dilation_sizes");

        if (hp.istft_upsample_rates.empty())
            hp.istft_upsample_rates = {10, 6};
        if (hp.istft_upsample_kernel_sizes.empty())
            hp.istft_upsample_kernel_sizes = {20, 12};
        if (hp.istft_resblock_kernel_sizes.empty())
            hp.istft_resblock_kernel_sizes = {3, 7, 11};
        if (hp.istft_resblock_dilations.empty())
            hp.istft_resblock_dilations = {1, 3, 5, 1, 3, 5, 1, 3, 5};

        c->vocab.id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
        for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++) {
            const auto& t = c->vocab.id_to_token[i];
            if (!t.empty())
                c->vocab.token_to_id[t] = i;
        }
        c->vocab.pad_id = 0;
    }

    if (params.verbosity >= 1) {
        const auto& hp = c->hp;
        fprintf(stderr,
                "kokoro: arch=kokoro  hidden=%u style=%u max_dur=%u n_token=%u "
                "vocab=%zu plbert=%uL/%uH/ff=%u  iSTFT n_fft=%u hop=%u sr=%u\n",
                hp.hidden_dim, hp.style_dim, hp.max_dur, hp.n_token, c->vocab.id_to_token.size(), hp.plbert_n_layers,
                hp.plbert_hidden, hp.plbert_ff, hp.istft_n_fft, hp.istft_hop, hp.sample_rate);
    }

    // ---- Backends (borrowed) ----
    c->backend_cpu = cpu_backend;
    c->backend = user_backend ? user_backend : cpu_backend;
    // Default-route generator to GPU (Metal-hang fix landed upstream as
    // ggml PR #1477). KOKORO_GEN_CPU=1 forces the legacy CPU path.
    c->gen_backend = params.gen_force_metal ? c->backend : c->backend_cpu;

    // ---- Tensor table: borrow tensors from src_gguf_data, no copy ----
    {
        const int n_tensors = gguf_get_n_tensors(g);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(g, i);
            ggml_tensor* t = ggml_get_tensor(src_gguf_data, name);
            if (!t)
                continue;
            c->tensors[name] = t;
        }
        c->ctx_w = src_gguf_data; // borrowed — not freed in kokoro_free
        c->buf_w = nullptr;
    }

    if (!sanity_check_weights(c)) {
        fprintf(stderr, "kokoro: weight sanity check failed (rs_context path)\n");
        kokoro_free(c);
        return nullptr;
    }

    // ---- Schedulers ----
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend)
            backends[n_be++] = c->backend_cpu;
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, /*graph_size=*/262144,
                                          /*parallel=*/false, /*op_offload=*/false);
        if (!c->sched) {
            fprintf(stderr, "kokoro: failed to allocate main scheduler\n");
            kokoro_free(c);
            return nullptr;
        }
        ggml_backend_t gen_backends[2];
        int n_gb = 0;
        // Generator scheduler holds both backends: the primary GPU and the
        // CPU fallback for any op the GPU can't handle. Weights live on
        // the GPU buffer (set up by rs_context_t), so a CPU-only sched
        // wouldn't be able to read them — the KOKORO_GEN_CPU env is now
        // purely advisory in this borrow path.
        gen_backends[n_gb++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend)
            gen_backends[n_gb++] = c->backend_cpu;
        c->gen_sched = ggml_backend_sched_new(gen_backends, nullptr, n_gb, /*graph_size=*/262144, false, false);
        if (!c->gen_sched) {
            fprintf(stderr, "kokoro: failed to allocate generator scheduler\n");
            kokoro_free(c);
            return nullptr;
        }
    }
    c->compute_meta.resize(ggml_tensor_overhead() * 262144 + ggml_graph_overhead_custom(262144, false));

#ifdef RS_USE_METAL_KOKORO
    // Initialize the custom Metal ConvTranspose1d kernel for the iSTFTNet
    // generator's two upsample ops. If the user's primary backend isn't Metal
    // (e.g. CPU build, CUDA, Vulkan), or if init fails, fall back to the
    // single-graph ggml path.
    const char *no_metal_convt = std::getenv("KOKORO_NO_METAL_CONVT");
    const bool disable_metal_convt = no_metal_convt && no_metal_convt[0] && no_metal_convt[0] != '0';
    auto is_metal_backend = [](ggml_backend_t b) {
        if (!b) return false;
        std::string n = ggml_backend_name(b);
        return n.find("Metal") != std::string::npos || n.find("MTL") != std::string::npos;
    };
    if (!disable_metal_convt && is_metal_backend(c->backend)) {
        ggml_tensor *ups0_w = try_get(c, "dec.gen.ups.0.weight");
        ggml_tensor *ups0_b = try_get(c, "dec.gen.ups.0.bias");
        ggml_tensor *ups1_w = try_get(c, "dec.gen.ups.1.weight");
        ggml_tensor *ups1_b = try_get(c, "dec.gen.ups.1.bias");
        if (ups0_w && ups0_b && ups1_w && ups1_b) {
            auto km = std::make_unique<KokoroMetalConvT1D>();
            if (km->init(ups0_w, ups0_b, /*s*/ 10, /*p*/ 5,
                         ups1_w, ups1_b, /*s*/ 6, /*p*/ 3)) {
                c->metal_convt = std::move(km);
            } else {
                fprintf(stderr, "kokoro: KokoroMetalConvT1D init failed; using ggml-metal fallback\n");
            }
        }
    }
#endif

    if (params.verbosity >= 1) {
        fprintf(stderr, "kokoro: loaded %zu tensors (rs_context borrow)  gen=%s\n", c->tensors.size(),
                c->gen_backend == c->backend_cpu ? "CPU (KOKORO_GEN_CPU=1)" : "GPU");
    }
    return c;
}

// ---------------------------------------------------------------------------
// Adapter-friendly synth: skip the IPA tokeniser and feed raw ids directly
// into the run loop. The adapter does its own tokenisation via
// kokoro_phonemes_to_ids before calling this (or supplies ids from a
// future misaki-zh G2P).
// ---------------------------------------------------------------------------

static float* kokoro_synthesize_from_ids(struct kokoro_context* ctx, const int32_t* token_ids, int n_ids,
                                             int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    if (!ctx || !token_ids || n_ids <= 0)
        return nullptr;
    int n_audio = 0;
    float* audio = kokoro_run_audio(ctx, token_ids, n_ids, &n_audio);
    if (!audio || n_audio <= 0)
        return nullptr;
    if (out_n_samples)
        *out_n_samples = n_audio;
    return audio;
}

// ======================================================================
// BEGIN kokoro.cpp (ISpeechModel adapter)
// ======================================================================

// kokoro.cpp — ISpeechModel adapter for the CrispASR-port Kokoro engine.
//
// This file is intentionally thin: all model logic lives in
// kokoro_engine.cpp (a near-verbatim port of CrispASR src/kokoro.cpp).
// The adapter:
//   * Borrows the pre-loaded GGUF and backends from rs_context_t.
//   * Routes ISpeechModel virtual calls into the engine's C ABI.
//   * Tokenises IPA input on PushText (G2P is left for Phase 3).
//   * Buffers synthesised PCM into KokoroState::audio_output for the
//     rs_context's GetAudioOutput drain pattern.





struct KokoroModel::Impl {
  kokoro_context *engine = nullptr;
  ~Impl() {
    if (engine) {
      kokoro_free(engine);
      engine = nullptr;
    }
  }
};

KokoroModel::KokoroModel() : impl_(std::make_unique<Impl>()) {
  meta_.arch_name = "kokoro";
  meta_.audio_sample_rate = 24000;
  // Kokoro provides phonemes directly; no shared fbank/mel frontend.
  meta_.use_external_frontend = true;
}

KokoroModel::~KokoroModel() = default;

// ---------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------

bool KokoroModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                       ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("KokoroModel::Load: rs_context missing gguf data");
    return false;
  }

  // rs_context_init_internal appends the CPU backend as the LAST entry
  // in ctx->backends (see rs_context.cpp). Use it as the engine's CPU
  // backend so the Metal-hang workaround keeps the generator on CPU.
  ggml_backend_t cpu_backend = nullptr;
  if (!ctx->backends.empty()) {
    cpu_backend = ctx->backends.back();
  }
  if (!cpu_backend) {
    cpu_backend = backend;
  }

  backend_ = backend;
  backend_cpu_ = cpu_backend;

  kokoro_context_params params = kokoro_context_default_params();
  params.n_threads = ctx->params.n_threads;
  params.use_gpu = ctx->params.use_gpu;
  params.length_scale = length_scale_;
  // Generator now runs on GPU by default — kokoro_context_default_params
  // already wires KOKORO_GEN_CPU into gen_force_metal=false as the
  // opt-out path.

  if (const char *ls = std::getenv("RS_KOKORO_LENGTH_SCALE")) {
    try {
      params.length_scale = std::stof(ls);
      length_scale_ = params.length_scale;
    } catch (...) {
      RS_LOG_WARN("KokoroModel: invalid RS_KOKORO_LENGTH_SCALE='%s' (ignored)",
                  ls);
    }
  }

  impl_->engine = kokoro_init_from_rs_context(
      ctx->gguf_data, ctx->ctx_gguf, backend, cpu_backend, params);
  if (!impl_->engine) {
    RS_LOG_ERR("KokoroModel::Load: kokoro_init_from_rs_context failed");
    return false;
  }

  // Optional voice pack via env var (Phase 1 default path before any C
  // API surface). Required for synthesis — without a voice pack the
  // predictor stage logs an error and returns nullptr.
  if (const char *vp = std::getenv("RS_KOKORO_VOICE_PATH")) {
    if (vp[0] != '\0') {
      if (kokoro_load_voice_pack(impl_->engine, vp) != 0) {
        RS_LOG_ERR("KokoroModel::Load: failed to load voice pack '%s'", vp);
        return false;
      }
      RS_LOG_INFO("KokoroModel: loaded voice pack '%s'", vp);
    }
  }

  return true;
}

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

std::shared_ptr<RSState> KokoroModel::CreateState() {
  return std::make_shared<KokoroState>();
}

// ---------------------------------------------------------------------
// PushText: IPA-only in Phase 1; misaki[zh] G2P arrives in Phase 3.
// ---------------------------------------------------------------------

bool KokoroModel::PushText(RSState &state, const char *text,
                           const char *language, const char *instruct) {
  (void)instruct;
  auto &st = static_cast<KokoroState &>(state);
  if (!impl_ || !impl_->engine) {
    RS_LOG_ERR("KokoroModel::PushText: engine not initialised");
    return false;
  }
  if (!text || !*text) {
    RS_LOG_ERR("KokoroModel::PushText: empty text");
    return false;
  }
  st.input_text = text;
  st.phoneme_ids.clear();
  st.phonemes_are_ipa = false;
  st.audio_output.clear();
  st.audio_read_cursor = 0;

  bool ipa = force_ipa_input_;
  if (language && std::strncmp(language, "ipa", 3) == 0) {
    ipa = true;
  }
  if (const char *env = std::getenv("RS_KOKORO_PHONEMES")) {
    if (env[0] == '1')
      ipa = true;
  }

  std::string g2p_buf;
  const char *phoneme_text = text;

  if (!ipa) {
    // Language detection: starts-with "zh" / "chinese" (case-insensitive).
    bool is_zh = false;
    if (language) {
      std::string lc;
      for (const char *p = language; *p; ++p)
        lc.push_back((char)std::tolower((unsigned char)*p));
      if (lc.rfind("zh", 0) == 0 || lc.rfind("chinese", 0) == 0) is_zh = true;
    }

    if (!is_zh) {
      RS_LOG_ERR(
          "KokoroModel: G2P only implemented for Chinese — pass --phonemes "
          "with IPA text (or set RS_KOKORO_PHONEMES=1) for other languages");
      return false;
    }

    if (!g2p_zh_) {
      g2p_zh_ = std::make_unique<rs::kokoro_zh::ZHG2P>();
    }
    if (!g2p_zh_->IsLoaded()) {
      std::vector<std::string> tried_jieba, tried_zh;
      std::string jieba_dir = rs::find_data_dir(
          "cppjieba/dict", "RS_KOKORO_JIEBA_DICT_DIR", &tried_jieba);
      std::string zh_dir = rs::find_data_dir(
          "kokoro_zh", "RS_KOKORO_ZH_DATA_DIR", &tried_zh);
      if (jieba_dir.empty() || zh_dir.empty()) {
        auto join = [](const std::vector<std::string> &v) {
          std::string s;
          for (auto &p : v) { s += "\n    "; s += p; }
          return s;
        };
        RS_LOG_ERR(
            "KokoroModel: ZH G2P data not found.\n"
            "  cppjieba dict (searched):%s\n"
            "  kokoro_zh data (searched):%s\n"
            "Place the data next to the binary at "
            "<exe_dir>/../share/rapidspeech/{cppjieba/dict,kokoro_zh}, or set "
            "RS_KOKORO_JIEBA_DICT_DIR / RS_KOKORO_ZH_DATA_DIR.\n"
            "Download: https://github.com/lovemefan/RapidSpeech.cpp/releases "
            "(see 'rapidspeech-data' asset).",
            join(tried_jieba).c_str(), join(tried_zh).c_str());
        return false;
      }
      if (!g2p_zh_->Load(jieba_dir, zh_dir)) {
        RS_LOG_ERR(
            "KokoroModel: failed to load ZH G2P (jieba='%s', data='%s')",
            jieba_dir.c_str(), zh_dir.c_str());
        return false;
      }
      RS_LOG_INFO("KokoroModel: ZH G2P initialised (jieba='%s', data='%s')",
                  jieba_dir.c_str(), zh_dir.c_str());
    }

    // Optional misaki[en]-style English fallback for Latin segments inside
    // ZH text. Default on; RS_KOKORO_EN=0 disables and Latin runs fall back
    // to the unk '❓' marker.
    {
      bool en_disabled = false;
      if (const char *e = std::getenv("RS_KOKORO_EN"))
        en_disabled = (e[0] == '0');
      if (!en_disabled && !g2p_en_) {
        auto en = std::make_unique<rs::kokoro_en::EnG2P>();
        std::vector<std::string> tried_en;
        std::string en_dir = rs::find_data_dir(
            "kokoro_en", "RS_KOKORO_EN_DATA_DIR", &tried_en);
        if (en_dir.empty()) {
          std::string joined;
          for (auto &p : tried_en) { joined += "\n    "; joined += p; }
          RS_LOG_WARN(
              "KokoroModel: EnG2P data not found (English in ZH text will "
              "fall back to '%s'). Searched:%s",
              g2p_zh_->Unk().c_str(), joined.c_str());
        } else if (en->Load(en_dir)) {
          g2p_en_ = std::move(en);
          rs::kokoro_en::EnG2P *raw = g2p_en_.get();
          g2p_zh_->SetEnCallable(
              [raw](const std::string &s) { return raw->Process(s); });
        } else {
          RS_LOG_WARN(
              "KokoroModel: EnG2P data not loaded from '%s'; English in "
              "ZH text will fall back to '%s'",
              en_dir.c_str(), g2p_zh_->Unk().c_str());
        }
      }
    }

    // Optional WeTextProcessing TN before G2P (default on, RS_WETEXT=0 to
    // disable). Pass-through if FST data is missing or fails to load.
    std::string tn_buf;
    const char *g2p_input = text;
    bool tn_disabled = false;
    if (const char *e = std::getenv("RS_WETEXT")) tn_disabled = (e[0] == '0');
    if (!tn_disabled) {
      if (!tn_) tn_ = std::make_unique<rs::WeTextNormalizer>();
      if (!tn_->IsLoaded()) {
        std::vector<std::string> tried_wt;
        std::string wt_dir = rs::find_data_dir(
            "wetext", "RS_WETEXT_DATA_DIR", &tried_wt);
        if (wt_dir.empty()) {
          std::string joined;
          for (auto &p : tried_wt) { joined += "\n    "; joined += p; }
          RS_LOG_WARN(
              "KokoroModel: WeText data not found (text normalisation "
              "disabled). Searched:%s", joined.c_str());
        } else {
          tn_->Load(wt_dir + "/zh_tn_tagger.fst",
                    wt_dir + "/zh_tn_verbalizer.fst");
          // Load failure is non-fatal; Normalize becomes a pass-through.
        }
      }
      tn_buf = tn_->Normalize(text);
      g2p_input = tn_buf.c_str();
      if (const char *e = std::getenv("RS_WETEXT_DUMP"))
        if (e[0] == '1')
          std::fprintf(stderr, "TN: %s -> %s\n", text, g2p_input);
    }

    g2p_buf = g2p_zh_->Process(g2p_input);
    if (g2p_buf.empty()) {
      RS_LOG_ERR("KokoroModel: ZH G2P produced empty phoneme string for "
                 "input '%s'",
                 text);
      return false;
    }
    if (const char *e = std::getenv("RS_KOKORO_DUMP_G2P")) {
      if (e[0] == '1') {
        std::fprintf(stderr, "CXX: %s\n", g2p_buf.c_str());
      }
    }
    phoneme_text = g2p_buf.c_str();
  }

  int n = 0;
  int32_t *ids = kokoro_phonemes_to_ids(impl_->engine, phoneme_text, &n);
  if (!ids || n <= 0) {
    if (ids)
      std::free(ids);
    RS_LOG_ERR("KokoroModel::PushText: tokenisation produced no IDs");
    return false;
  }
  st.phoneme_ids.assign(ids, ids + n);
  std::free(ids);
  st.phonemes_are_ipa = true;
  return true;
}

// ---------------------------------------------------------------------
// Encode: run the full engine forward pass; Decode is a no-op (audio
// is emitted in one shot).
// ---------------------------------------------------------------------

bool KokoroModel::Encode(const std::vector<float> &input_frames,
                         RSState &state, ggml_backend_sched_t sched) {
  (void)input_frames;
  (void)sched;
  auto &st = static_cast<KokoroState &>(state);
  if (!impl_ || !impl_->engine) {
    RS_LOG_ERR("KokoroModel::Encode: engine not initialised");
    return false;
  }
  if (st.phoneme_ids.empty()) {
    RS_LOG_ERR("KokoroModel::Encode: no phoneme IDs (call PushText first)");
    return false;
  }

  int n_samples = 0;
  float *pcm = kokoro_synthesize_from_ids(impl_->engine, st.phoneme_ids.data(),
                                          (int)st.phoneme_ids.size(),
                                          &n_samples);
  if (!pcm || n_samples <= 0) {
    if (pcm)
      kokoro_pcm_free(pcm);
    RS_LOG_ERR("KokoroModel::Encode: synthesis returned no audio");
    return false;
  }
  st.audio_output.assign(pcm, pcm + n_samples);
  st.audio_read_cursor = 0;
  kokoro_pcm_free(pcm);
  return true;
}

bool KokoroModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  (void)state;
  (void)sched;
  return true;
}

// ---------------------------------------------------------------------
// Audio drain — single-shot emission of the whole utterance.
// ---------------------------------------------------------------------

int KokoroModel::GetAudioOutput(RSState &state, float **out_data) {
  auto &st = static_cast<KokoroState &>(state);
  if (!out_data) {
    return 0;
  }
  int remaining = (int)st.audio_output.size() - st.audio_read_cursor;
  if (remaining <= 0) {
    *out_data = nullptr;
    return 0;
  }
  *out_data = st.audio_output.data() + st.audio_read_cursor;
  st.audio_read_cursor += remaining;
  return remaining;
}

// ---------------------------------------------------------------------
// Kokoro-specific knobs
// ---------------------------------------------------------------------

bool KokoroModel::LoadVoicePack(const char *path) {
  if (!impl_ || !impl_->engine || !path) {
    return false;
  }
  return kokoro_load_voice_pack(impl_->engine, path) == 0;
}

void KokoroModel::SetLengthScale(float s) {
  length_scale_ = s;
  if (impl_ && impl_->engine) {
    kokoro_set_length_scale(impl_->engine, s);
  }
}

void KokoroModel::set_imatrix_callback(
    std::function<void(struct ggml_tensor *)> cb) {
  if (impl_ && impl_->engine) {
    impl_->engine->imatrix_cb = std::move(cb);
  }
}

// Stubs for the rarely-used lookup helpers declared in the header.
// The engine owns the tensor table; these are kept available for future
// stage-extraction debug paths but currently delegate to nullptr.
struct ggml_tensor *KokoroModel::try_get(const char *name) const {
  auto it = tensors_.find(name ? name : "");
  return it == tensors_.end() ? nullptr : it->second;
}

struct ggml_tensor *KokoroModel::require(const char *name) const {
  auto *t = try_get(name);
  if (!t) {
    RS_LOG_ERR("KokoroModel::require: tensor missing '%s'", name ? name : "");
  }
  return t;
}

std::vector<int32_t>
KokoroModel::TokenizePhonemes(const char *phonemes) const {
  std::vector<int32_t> out;
  if (!impl_ || !impl_->engine || !phonemes) {
    return out;
  }
  int n = 0;
  int32_t *ids = kokoro_phonemes_to_ids(impl_->engine, phonemes, &n);
  if (ids && n > 0) {
    out.assign(ids, ids + n);
  }
  if (ids)
    std::free(ids);
  return out;
}

bool KokoroModel::ResolveRefS(KokoroState &state, int L_raw) const {
  (void)state;
  (void)L_raw;
  // Voice-pack slicing is handled inside the engine; this entry point
  // is kept for parity with the public header but is unused.
  return false;
}

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------

namespace {
struct KokoroRegistrar {
  KokoroRegistrar() {
    rs_register_model_arch(
        "kokoro", []() { return std::make_shared<KokoroModel>(); });
  }
} global_kokoro_reg;
} // namespace
