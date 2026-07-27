#include "arch/tenvad.h"
#include "core/rs_context.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

// Vendored TEN-VAD DSP frontend (third_party/ten-vad, Apache-2.0).
// aed_st.h here is the trimmed macro-only variant (no onnxruntime).
#include "aed_st.h"
#include "coeff.h"       // Hann-768 window + feature means/stds
#include "pitch_est.h"
#include "stft.h"

#include <cmath>
#include <cstring>
#include <algorithm>

#define TENVAD_MAX_NODES 1024

// ============================================================
// TEN-VAD — ggml graph computation + vendored DSP frontend
//
// Frontend per 256-sample hop (int16-scale floats), ported from
// ten-vad's aed.cc (AUP_Aed_proc / runOneFrm / aivad_proc):
//   pre-emphasis 0.97 -> STFT (fft 1024, win 768 Hann, hop 256)
//   -> bin power (513) -> pitch estimate (raw signal + bin power)
//   -> 40 log-mel (/32768^2, mean/std normalize) + pitch feature
//   -> 3-frame context stack [3, 41]
//
// Network (per frame, from ten-vad.onnx):
//   [1,3,41] -> dw conv 3x3 (valid) -> pw 1x1 -> 16ch -> ReLU
//   -> maxpool k3 s2 -> dw conv k3 s2 p(1,1) -> pw -> ReLU
//   -> dw conv k3 s2 p(0,1) -> pw -> ReLU -> flatten 5x16 -> 80
//   -> LSTM(80->64) -> LSTM(64->64)  (h/c passed via set/get)
//   -> concat(h2,h1)=128 -> fc 128->32 ReLU -> fc 32->1 sigmoid
//
// LSTM gates are pre-ordered i,f,g,o by the converter (from ONNX iofc),
// so the cell below matches SileroVadModel's.
// ============================================================

namespace {
constexpr int TENVAD_FFT   = 1024;
constexpr int TENVAD_WIN   = 768;
constexpr int TENVAD_NBINS = TENVAD_FFT / 2 + 1;  // 513
constexpr int TENVAD_NMEL  = AUP_AED_MEL_FILTER_BANK_NUM;  // 40
constexpr float TENVAD_EPS = 1e-20f;              // AUP_AED_EPS in aed.cc
constexpr float TENVAD_PREEMPH = 0.97f;
}

TenVadState::~TenVadState() {
    if (pitch_est) AUP_PE_destroy(&pitch_est);
    if (analyzer)  AUP_Analyzer_destroy(&analyzer);
}

TenVadModel::TenVadModel() {
    meta_.arch_name = "ten-vad";
    meta_.audio_sample_rate = 16000;
    meta_.n_mels = 0;
    meta_.vocab_size = 0;
}

TenVadModel::~TenVadModel() = default;

bool TenVadModel::MapTensors(ggml_context* gguf_data) {
    auto get = [&](const char* name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(gguf_data, name);
        if (!t) RS_LOG_WARN("TEN-VAD tensor not found: %s", name);
        return t;
    };

    weights_.conv1_dw_w = get("conv1.dw.weight");
    weights_.conv1_pw_w = get("conv1.pw.weight");
    weights_.conv1_pw_b = get("conv1.pw.bias");
    weights_.conv2_dw_w = get("conv2.dw.weight");
    weights_.conv2_pw_w = get("conv2.pw.weight");
    weights_.conv2_pw_b = get("conv2.pw.bias");
    weights_.conv3_dw_w = get("conv3.dw.weight");
    weights_.conv3_pw_w = get("conv3.pw.weight");
    weights_.conv3_pw_b = get("conv3.pw.bias");
    weights_.lstm1_weight_ih = get("lstm1.weight_ih");
    weights_.lstm1_weight_hh = get("lstm1.weight_hh");
    weights_.lstm1_bias_ih   = get("lstm1.bias_ih");
    weights_.lstm1_bias_hh   = get("lstm1.bias_hh");
    weights_.lstm2_weight_ih = get("lstm2.weight_ih");
    weights_.lstm2_weight_hh = get("lstm2.weight_hh");
    weights_.lstm2_bias_ih   = get("lstm2.bias_ih");
    weights_.lstm2_bias_hh   = get("lstm2.bias_hh");
    weights_.fc1_w = get("fc1.weight");
    weights_.fc1_b = get("fc1.bias");
    weights_.fc2_w = get("fc2.weight");
    weights_.fc2_b = get("fc2.bias");

    if (!weights_.conv1_dw_w || !weights_.lstm1_weight_ih ||
        !weights_.lstm2_weight_ih || !weights_.fc2_w) {
        RS_LOG_ERR("Critical TEN-VAD tensors missing. Check GGUF conversion.");
        return false;
    }
    return true;
}

bool TenVadModel::LoadHParams(gguf_context* ctx_gguf) {
    if (!ctx_gguf) return true;
    auto ki = [&](const char* k, int def) {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key >= 0 ? (int)gguf_get_val_i32(ctx_gguf, key) : def;
    };
    auto kf = [&](const char* k, float def) {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key >= 0 ? gguf_get_val_f32(ctx_gguf, key) : def;
    };
    hparams_.sample_rate    = ki("vad.sample_rate", 16000);
    hparams_.hop_size       = ki("vad.hop_size", 256);
    hparams_.context_frames = ki("vad.context_frames", 3);
    hparams_.feature_dim    = ki("vad.feature_dim", 41);
    hparams_.hidden_size    = ki("vad.hidden_size", 64);
    hparams_.reset_frames   = ki("vad.reset_frames", 1875);
    hparams_.threshold      = kf("vad.threshold", 0.5f);
    hparams_.min_silence_ms = ki("vad.min_silence_ms", 200);
    return true;
}

// Port of aed.cc AUP_Aed_resetVariables mel triangle generation:
// 40 triangles over 513 bins, mel points spanning 0..8000 Hz.
void TenVadModel::BuildMelFilterBank() {
    const int n_bins = TENVAD_NBINS;
    const int n_mel  = TENVAD_NMEL;
    mel_fb_.assign((size_t)n_mel * n_bins, 0.0f);

    size_t bin_buff[TENVAD_NMEL + 2];
    float low_mel  = 2595.0f * log10f(1.0f + 0.0f / 700.0f);
    float high_mel = 2595.0f * log10f(1.0f + 8000.0f / 700.0f);
    for (int i = 0; i < n_mel + 2; i++) {
        float mel_points = i * (high_mel - low_mel) / ((float)n_mel + 1.0f) + low_mel;
        float hz_points  = 700.0f * (powf(10.0f, mel_points / 2595.0f) - 1.0f);
        bin_buff[i] = (size_t)((TENVAD_FFT + 1.0f) * hz_points / (float)AUP_AED_FS);
    }
    for (int j = 0; j < n_mel; j++) {
        for (size_t i = bin_buff[j]; i < bin_buff[j + 1]; i++) {
            mel_fb_[(size_t)j * n_bins + i] =
                (float)(i - bin_buff[j]) / (float)(bin_buff[j + 1] - bin_buff[j]);
        }
        for (size_t i = bin_buff[j + 1]; i < bin_buff[j + 2]; i++) {
            mel_fb_[(size_t)j * n_bins + i] =
                (float)(bin_buff[j + 2] - i) / (float)(bin_buff[j + 2] - bin_buff[j + 1]);
        }
    }
}

bool TenVadModel::LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                             ggml_backend_t backend) {
    if (!gguf_data) return false;
    backend_ = backend;
    LoadHParams(ctx_gguf);
    meta_.audio_sample_rate = hparams_.sample_rate;
    if (!MapTensors(gguf_data)) return false;
    BuildMelFilterBank();
    RS_LOG_INFO("TEN-VAD loaded: sr=%d, hop=%d, threshold=%.2f",
                hparams_.sample_rate, hparams_.hop_size, hparams_.threshold);
    return true;
}

bool TenVadModel::Load(const std::unique_ptr<rs_context_t>& ctx,
                       ggml_backend_t backend) {
    if (!ctx || !ctx->gguf_data) return false;
    return LoadDirect(ctx->gguf_data, ctx->ctx_gguf, backend);
}

// Set up (or re-init) the per-stream DSP handles. Configuration mirrors
// aed.cc AUP_Aed_memAllocate: analyzer win 768 / hop 256 / fft 1024 with the
// Hann-768 window; pitch estimator with LPC prefiltering at 4 kHz.
bool TenVadModel::InitStateDSP(TenVadState& s) {
    if (!s.analyzer) {
        if (AUP_Analyzer_create(&s.analyzer) < 0) return false;
        Analyzer_StaticCfg acfg;
        AUP_Analyzer_getStaticCfg(s.analyzer, &acfg);
        acfg.win_len  = TENVAD_WIN;
        acfg.hop_size = hparams_.hop_size;
        acfg.fft_size = TENVAD_FFT;
        acfg.ana_win_coeff = AUP_AED_STFTWindow_Hann768;
        if (AUP_Analyzer_memAllocate(s.analyzer, &acfg) < 0) return false;
    }
    if (AUP_Analyzer_init(s.analyzer) < 0) return false;

    if (!s.pitch_est) {
        if (AUP_PE_create(&s.pitch_est) < 0) return false;
        PE_StaticCfg pcfg;
        AUP_PE_getStaticCfg(s.pitch_est, &pcfg);
        pcfg.fftSz = TENVAD_FFT;
        pcfg.anaWindowSz = TENVAD_WIN;
        pcfg.hopSz = (size_t)hparams_.hop_size;
        pcfg.useLPCPreFiltering = AUP_AED_PITCH_EST_USE_LPC;
        pcfg.procFs = AUP_AED_PITCH_EST_PROCFS;
        if (AUP_PE_memAllocate(s.pitch_est, &pcfg) < 0) return false;
    }
    PE_DynamCfg pdyn;
    pdyn.voicedThr = AUP_AED_PITCH_EST_DEFAULT_VOICEDTHR;
    AUP_PE_setDynamCfg(s.pitch_est, &pdyn);
    if (AUP_PE_init(s.pitch_est) < 0) return false;

    s.feat_stack.assign((size_t)hparams_.context_frames * hparams_.feature_dim, 0.0f);
    s.h1.assign(hparams_.hidden_size, 0.0f);
    s.c1.assign(hparams_.hidden_size, 0.0f);
    s.h2.assign(hparams_.hidden_size, 0.0f);
    s.c2.assign(hparams_.hidden_size, 0.0f);
    s.remainder.clear();
    s.time_signal_pre = 0.0f;
    s.frames_since_reset = 0;
    s.speech_prob = 0.0f;
    s.triggered = false;
    s.temp_end_sample = 0;
    s.current_sample = 0;
    return true;
}

std::shared_ptr<RSState> TenVadModel::CreateState() {
    auto s = std::make_shared<TenVadState>();
    if (!InitStateDSP(*s)) {
        RS_LOG_ERR("TEN-VAD: DSP state init failed");
        return nullptr;
    }
    return s;
}

bool TenVadModel::ResetState(RSState& state) {
    return InitStateDSP(dynamic_cast<TenVadState&>(state));
}

// ============================================================
// ggml graph for a single frame
// Input tensors:  feat [41,3], in_h1/in_c1/in_h2/in_c2 [64]
// Output tensors: speech_prob scalar, out_h1/out_c1/out_h2/out_c2 [64]
// ============================================================
static ggml_cgraph* tenvad_build_graph(
        const TenVadWeights& w,
        ggml_context* ctx0,
        ggml_tensor* feat,
        ggml_tensor* h1_in, ggml_tensor* c1_in,
        ggml_tensor* h2_in, ggml_tensor* c2_in)
{
    auto f16 = [&](ggml_tensor* t) { return ggml_cast(ctx0, t, GGML_TYPE_F16); };

    // Bias for conv output [T, OC, N]
    auto add_bias = [&](ggml_tensor* out, ggml_tensor* bias) -> ggml_tensor* {
        ggml_tensor* b = ggml_reshape_2d(ctx0, bias, 1, bias->ne[0]);
        return ggml_add(ctx0, out, b);
    };

    // ---- dw conv 3x3 over [W=41 features, H=3 context], valid ----
    ggml_tensor* cur = ggml_reshape_4d(ctx0, feat, feat->ne[0], feat->ne[1], 1, 1);
    cur = ggml_conv_2d(ctx0, f16(w.conv1_dw_w), cur, 1, 1, 0, 0, 1, 1);  // [39,1,1,1]

    // ---- pw 1x1 -> 16 channels, ReLU ----
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1);                      // [39,1]
    cur = ggml_conv_1d(ctx0, f16(w.conv1_pw_w), cur, 1, 0, 1);            // [39,16,1]
    cur = ggml_relu(ctx0, add_bias(cur, w.conv1_pw_b));

    // ---- maxpool k3 s2 over W ----
    cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_MAX, 3, 2, 0);             // [19,16,1]

    // ---- separable conv #1: dw k3 s2 pad(1,1) + pw, ReLU ----
    cur = ggml_conv_1d_dw(ctx0, f16(w.conv2_dw_w),
                          ggml_reshape_3d(ctx0, cur, cur->ne[0], cur->ne[1], 1),
                          2, 1, 1);                                       // [10,16,1]
    cur = ggml_conv_1d(ctx0, f16(w.conv2_pw_w), cur, 1, 0, 1);            // [10,16,1]
    cur = ggml_relu(ctx0, add_bias(cur, w.conv2_pw_b));

    // ---- separable conv #2: dw k3 s2 pad(0,1) + pw, ReLU ----
    // ONNX pads are asymmetric (right-only): zero-pad the tail manually.
    cur = ggml_pad(ctx0, cur, 1, 0, 0, 0);                                // [11,16,1]
    cur = ggml_conv_1d_dw(ctx0, f16(w.conv3_dw_w), cur, 2, 0, 1);         // [5,16,1]
    cur = ggml_conv_1d(ctx0, f16(w.conv3_pw_w), cur, 1, 0, 1);            // [5,16,1]
    cur = ggml_relu(ctx0, add_bias(cur, w.conv3_pw_b));

    // ---- flatten position-major (matches ONNX transpose [0,2,1] + reshape) ----
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1]);             // [5,16]
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));                     // [16,5]
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], 1);         // [80,1]

    // ---- LSTM cell (gates pre-ordered i,f,g,o by the converter) ----
    const int64_t H = h1_in->ne[0];
    auto lstm_cell = [&](ggml_tensor* x, ggml_tensor* h_in, ggml_tensor* c_in,
                         ggml_tensor* w_ih, ggml_tensor* w_hh,
                         ggml_tensor* b_ih, ggml_tensor* b_hh,
                         const char* h_name, const char* c_name)
            -> std::pair<ggml_tensor*, ggml_tensor*> {
        ggml_tensor* gates = ggml_add(ctx0, ggml_mul_mat(ctx0, w_ih, x), b_ih);
        ggml_tensor* hh = ggml_add(ctx0, ggml_mul_mat(ctx0, w_hh, h_in), b_hh);
        gates = ggml_add(ctx0, gates, hh);

        const size_t stride = ggml_row_size(gates->type, H);
        ggml_tensor* i_g = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, H, 0 * stride));
        ggml_tensor* f_g = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, H, 1 * stride));
        ggml_tensor* g_g = ggml_tanh(   ctx0, ggml_view_1d(ctx0, gates, H, 2 * stride));
        ggml_tensor* o_g = ggml_sigmoid(ctx0, ggml_view_1d(ctx0, gates, H, 3 * stride));

        ggml_tensor* c_new = ggml_add(ctx0,
            ggml_mul(ctx0, f_g, c_in),
            ggml_mul(ctx0, i_g, g_g));
        ggml_set_name(c_new, c_name);
        ggml_set_output(c_new);

        ggml_tensor* h_new = ggml_mul(ctx0, o_g, ggml_tanh(ctx0, c_new));
        ggml_set_name(h_new, h_name);
        ggml_set_output(h_new);
        return {h_new, c_new};
    };

    auto [h1_new, c1_new] = lstm_cell(cur, h1_in, c1_in,
                                      w.lstm1_weight_ih, w.lstm1_weight_hh,
                                      w.lstm1_bias_ih, w.lstm1_bias_hh,
                                      "out_h1", "out_c1");
    (void)c1_new;
    auto [h2_new, c2_new] = lstm_cell(h1_new, h2_in, c2_in,
                                      w.lstm2_weight_ih, w.lstm2_weight_hh,
                                      w.lstm2_bias_ih, w.lstm2_bias_hh,
                                      "out_h2", "out_c2");
    (void)c2_new;

    // ---- concat(h2, h1) -> fc 128->32 ReLU -> fc 32->1 sigmoid ----
    cur = ggml_concat(ctx0, h2_new, h1_new, 0);                           // [128,1]
    cur = ggml_relu(ctx0, ggml_add(ctx0, ggml_mul_mat(ctx0, w.fc1_w, cur), w.fc1_b));
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, w.fc2_w, cur), w.fc2_b);      // [1,1]
    cur = ggml_sigmoid(ctx0, cur);
    cur = ggml_reshape_1d(ctx0, cur, 1);
    ggml_set_name(cur, "speech_prob");
    ggml_set_output(cur);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, TENVAD_MAX_NODES, false);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

bool TenVadModel::Forward(TenVadState& s) {
    ggml_backend_t compute_backend = backend_;
    bool local_backend = false;
    if (!compute_backend) {
        compute_backend = ggml_backend_cpu_init();
        local_backend = true;
    }

    const int fea = hparams_.feature_dim;
    const int ctx_frames = hparams_.context_frames;
    const int H = hparams_.hidden_size;

    size_t meta_size = ggml_tensor_overhead() * TENVAD_MAX_NODES +
                       ggml_graph_overhead_custom(TENVAD_MAX_NODES, false);
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params params = { meta_size, meta.data(), true };
    ggml_context* meta_ctx = ggml_init(params);

    ggml_tensor* t_feat = ggml_new_tensor_2d(meta_ctx, GGML_TYPE_F32, fea, ctx_frames);
    ggml_set_name(t_feat, "feat");
    ggml_tensor* t_h1 = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, H);
    ggml_set_name(t_h1, "in_h1");
    ggml_tensor* t_c1 = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, H);
    ggml_set_name(t_c1, "in_c1");
    ggml_tensor* t_h2 = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, H);
    ggml_set_name(t_h2, "in_h2");
    ggml_tensor* t_c2 = ggml_new_tensor_1d(meta_ctx, GGML_TYPE_F32, H);
    ggml_set_name(t_c2, "in_c2");

    ggml_cgraph* gf = tenvad_build_graph(weights_, meta_ctx, t_feat,
                                         t_h1, t_c1, t_h2, t_c2);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(compute_backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);
    if (ok) {
        ggml_backend_tensor_set(t_feat, s.feat_stack.data(), 0,
                                (size_t)fea * ctx_frames * sizeof(float));
        ggml_backend_tensor_set(t_h1, s.h1.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(t_c1, s.c1.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(t_h2, s.h2.data(), 0, H * sizeof(float));
        ggml_backend_tensor_set(t_c2, s.c2.data(), 0, H * sizeof(float));

        ok = ggml_backend_graph_compute(compute_backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        float prob = 0.f;
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "speech_prob"),
                                &prob, 0, sizeof(float));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out_h1"),
                                s.h1.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out_c1"),
                                s.c1.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out_h2"),
                                s.h2.data(), 0, H * sizeof(float));
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out_c2"),
                                s.c2.data(), 0, H * sizeof(float));
        s.speech_prob = prob;
    } else {
        RS_LOG_ERR("TEN-VAD: graph alloc/compute failed");
    }

    ggml_gallocr_free(alloc);
    ggml_free(meta_ctx);
    if (local_backend) ggml_backend_free(compute_backend);
    return ok;
}

// One 256-sample hop (int16-scale floats), port of aed.cc runOneFrm.
bool TenVadModel::ProcessHop(const float* hop, TenVadState& s) {
    const int hop_sz = hparams_.hop_size;
    const int fea = hparams_.feature_dim;
    const int ctx_frames = hparams_.context_frames;

    // ---- pre-emphasis (stateful across hops) ----
    std::vector<float> emph(hop_sz);
    for (int i = 0; i < hop_sz; i++) {
        emph[i] = hop[i] - TENVAD_PREEMPH * s.time_signal_pre;
        s.time_signal_pre = hop[i];
    }

    // ---- STFT on the emphasized signal ----
    std::vector<float> spec(TENVAD_FFT, 0.0f);
    Analyzer_InputData ain;
    ain.input = emph.data();
    ain.iLength = hop_sz;
    Analyzer_OutputData aout;
    aout.output = spec.data();
    aout.oLength = TENVAD_FFT;
    if (AUP_Analyzer_proc(s.analyzer, &ain, &aout) < 0) {
        RS_LOG_ERR("TEN-VAD: analyzer proc failed");
        return false;
    }

    // ---- bin power (fftw rdft packing: re0 at [0], nyquist at [1]) ----
    float binpow[TENVAD_NBINS];
    binpow[0] = spec[0] * spec[0];
    binpow[TENVAD_NBINS - 1] = spec[1] * spec[1];
    for (int i = 1; i < TENVAD_NBINS - 1; i++) {
        binpow[i] = spec[2 * i] * spec[2 * i] + spec[2 * i + 1] * spec[2 * i + 1];
    }

    // ---- pitch feature (raw, non-emphasized signal) ----
    PE_InputData pin;
    pin.timeSignal = hop;
    pin.hopSz = hop_sz;
    pin.inBinPow = binpow;
    pin.nBins = TENVAD_NBINS;
    PE_OutputData pout;
    pout.pitchFreq = 0.0f;
    pout.voiced = -1;
    if (AUP_PE_proc(s.pitch_est, &pin, &pout) < 0) {
        RS_LOG_ERR("TEN-VAD: pitch estimator failed");
        return false;
    }

    // ---- feature stack: shift, then 40 log-mel + pitch, normalized ----
    std::memmove(s.feat_stack.data(), s.feat_stack.data() + fea,
                 sizeof(float) * (size_t)(ctx_frames - 1) * fea);
    float* f = s.feat_stack.data() + (size_t)(ctx_frames - 1) * fea;
    const float power_normal = 32768.0f * 32768.0f;
    for (int i = 0; i < TENVAD_NMEL; i++) {
        const float* coef = mel_fb_.data() + (size_t)i * TENVAD_NBINS;
        float v = 0.0f;
        for (int j = 0; j < TENVAD_NBINS; j++) v += binpow[j] * coef[j];
        v = logf(v / power_normal + TENVAD_EPS);
        f[i] = (v - AUP_AED_FEATURE_MEANS[i]) / (AUP_AED_FEATURE_STDS[i] + TENVAD_EPS);
    }
    for (int i = TENVAD_NMEL; i < fea; i++) {
        f[i] = (pout.pitchFreq - AUP_AED_FEATURE_MEANS[i]) /
               (AUP_AED_FEATURE_STDS[i] + TENVAD_EPS);
    }

    if (!Forward(s)) return false;

    // Periodic hidden-state reset (upstream aivadResetCnt behavior:
    // applies from the NEXT frame onward).
    if (++s.frames_since_reset >= hparams_.reset_frames) {
        std::fill(s.h1.begin(), s.h1.end(), 0.0f);
        std::fill(s.c1.begin(), s.c1.end(), 0.0f);
        std::fill(s.h2.begin(), s.h2.end(), 0.0f);
        std::fill(s.c2.begin(), s.c2.end(), 0.0f);
        s.frames_since_reset = 0;
    }

    // ---- VADIterator hysteresis (mirrors SileroVadModel) ----
    s.current_sample += hop_sz;
    float threshold = hparams_.threshold;
    float end_threshold = std::max(0.01f, threshold - 0.15f);
    int min_silence_samples = (hparams_.min_silence_ms * hparams_.sample_rate) / 1000;

    if (s.speech_prob >= threshold && s.temp_end_sample > 0) {
        s.temp_end_sample = 0;
    }
    if (s.speech_prob >= threshold && !s.triggered) {
        s.triggered = true;
    }
    if (s.speech_prob < end_threshold && s.triggered) {
        if (s.temp_end_sample == 0) {
            s.temp_end_sample = s.current_sample;
        }
        if (s.current_sample - s.temp_end_sample >= min_silence_samples) {
            s.triggered = false;
            s.temp_end_sample = 0;
        }
    }
    return true;
}

bool TenVadModel::Encode(const std::vector<float>& input_frames,
                         RSState& state,
                         ggml_backend_sched_t /*sched*/) {
    auto& s = dynamic_cast<TenVadState&>(state);
    if (!s.analyzer || !s.pitch_est) {
        RS_LOG_ERR("TEN-VAD: state not initialized");
        return false;
    }

    // Scale [-1,1] float PCM to the int16 range the frontend expects.
    s.remainder.reserve(s.remainder.size() + input_frames.size());
    for (float v : input_frames) s.remainder.push_back(v * 32768.0f);

    const size_t hop = (size_t)hparams_.hop_size;
    size_t off = 0;
    while (s.remainder.size() - off >= hop) {
        if (!ProcessHop(s.remainder.data() + off, s)) {
            s.remainder.erase(s.remainder.begin(), s.remainder.begin() + off);
            return false;
        }
        off += hop;
    }
    if (off > 0) {
        s.remainder.erase(s.remainder.begin(), s.remainder.begin() + off);
    }
    return true;
}

bool TenVadModel::Decode(RSState& /*state*/, ggml_backend_sched_t /*sched*/) {
    return true;
}

std::string TenVadModel::GetTranscription(RSState& state) {
    return dynamic_cast<TenVadState&>(state).triggered ? "SPEECH" : "SILENCE";
}

float TenVadModel::GetSpeechProbability(RSState& state) {
    return dynamic_cast<TenVadState&>(state).speech_prob;
}

static struct TenVadRegistrar {
    TenVadRegistrar() {
        rs_register_model_arch("ten-vad", []() -> std::shared_ptr<ISpeechModel> {
            return std::make_shared<TenVadModel>();
        });
    }
} s_ten_vad_registrar;
