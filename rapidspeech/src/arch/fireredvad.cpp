#include "arch/fireredvad.h"
#include "core/rs_context.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#define FIREREDVAD_MAX_NODES 8192

// ============================================================
// FireRedVAD — DFSMN graph computation.
//
// Reference: /Users/cenglingfan/code/python-project/FireRedVAD
//   core/detect_model.py  (DFSMN/FSMN, the actual nn.Module forward)
//   core/audio_feat.py    (kaldi_native_fbank + CMVN)
//   core/stream_vad_postprocessor.py  (4-state machine + segmentation)
// ============================================================

FireRedVadModel::FireRedVadModel() {
    meta_.arch_name = "firered-vad";
    meta_.audio_sample_rate = 16000;
    meta_.n_mels = 80;
    meta_.vocab_size = 0;
    // FireRedVAD was trained on kaldi_native_fbank features, which use the
    // Povey window. Declare it in meta so RSProcessor would also pick the
    // right window if the model is ever fed through that pipeline.
    meta_.window_type = WindowType::POVEY;
}

FireRedVadModel::~FireRedVadModel() = default;

bool FireRedVadModel::LoadHParams(gguf_context* ctx_gguf) {
    if (!ctx_gguf) return true;
    auto ki = [&](const char* k, int def) {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key >= 0 ? (int)gguf_get_val_i32(ctx_gguf, key) : def;
    };
    auto kf = [&](const char* k, float def) {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key >= 0 ? gguf_get_val_f32(ctx_gguf, key) : def;
    };
    hparams_.sample_rate       = ki("vad.sample_rate", 16000);
    hparams_.frame_length_ms   = ki("vad.frame_length_ms", 25);
    hparams_.frame_shift_ms    = ki("vad.frame_shift_ms", 10);
    hparams_.speech_threshold  = kf("vad.speech_threshold", 0.5f);
    hparams_.smooth_window_size= ki("vad.smooth_window_size", 5);
    hparams_.pad_start_frame   = ki("vad.pad_start_frame", 5);
    hparams_.min_speech_frame  = ki("vad.min_speech_frame", 8);
    hparams_.max_speech_frame  = ki("vad.max_speech_frame", 2000);
    hparams_.min_silence_frame = ki("vad.min_silence_frame", 20);
    hparams_.chunk_max_frame   = ki("vad.chunk_max_frame", 30000);

    hparams_.idim = ki("firered-vad.idim", 80);
    hparams_.odim = ki("firered-vad.odim", 1);
    hparams_.R    = ki("firered-vad.R", 8);
    hparams_.M    = ki("firered-vad.M", 1);
    hparams_.H    = ki("firered-vad.H", 256);
    hparams_.P    = ki("firered-vad.P", 128);
    hparams_.N1   = ki("firered-vad.N1", 20);
    hparams_.S1   = ki("firered-vad.S1", 1);
    hparams_.N2   = ki("firered-vad.N2", 0);
    hparams_.S2   = ki("firered-vad.S2", 0);
    hparams_.lookback_padding = ki("firered-vad.lookback_padding",
                                   (hparams_.N1 - 1) * hparams_.S1);
    return true;
}

bool FireRedVadModel::MapTensors(ggml_context* gguf_data) {
    auto get = [&](const char* name, bool required = true) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(gguf_data, name);
        if (!t && required) RS_LOG_WARN("FireRedVAD tensor missing: %s", name);
        return t;
    };

    weights_.cmvn_means   = get("cmvn.means");
    weights_.cmvn_inv_std = get("cmvn.inv_std");

    weights_.fc1_w = get("dfsmn.fc1.weight");
    weights_.fc1_b = get("dfsmn.fc1.bias");
    weights_.fc2_w = get("dfsmn.fc2.weight");
    weights_.fc2_b = get("dfsmn.fc2.bias");

    weights_.fsmn1_lookback_w = get("dfsmn.fsmn1.lookback.weight");
    if (hparams_.N2 > 0) {
        weights_.fsmn1_lookahead_w = get("dfsmn.fsmn1.lookahead.weight");
    }

    weights_.blocks.resize(hparams_.R - 1);
    for (int i = 0; i < hparams_.R - 1; ++i) {
        char buf[128];
        auto& blk = weights_.blocks[i];
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc1.weight", i);
        blk.fc1_w = get(buf);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc1.bias", i);
        blk.fc1_b = get(buf);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fc2.weight", i);
        blk.fc2_w = get(buf);
        snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fsmn.lookback.weight", i);
        blk.lookback_w = get(buf);
        if (hparams_.N2 > 0) {
            snprintf(buf, sizeof(buf), "dfsmn.fsmns.%d.fsmn.lookahead.weight", i);
            blk.lookahead_w = get(buf);
        }
    }

    weights_.dnns.resize(hparams_.M);
    for (int m = 0; m < hparams_.M; ++m) {
        char buf[64];
        snprintf(buf, sizeof(buf), "dnns.%d.weight", m);
        weights_.dnns[m].w = get(buf);
        snprintf(buf, sizeof(buf), "dnns.%d.bias", m);
        weights_.dnns[m].b = get(buf);
    }

    weights_.out_w = get("out.weight");
    weights_.out_b = get("out.bias");

    if (!weights_.fc1_w || !weights_.fc2_w || !weights_.fsmn1_lookback_w
        || !weights_.out_w || !weights_.cmvn_means || !weights_.cmvn_inv_std) {
        RS_LOG_ERR("FireRedVAD: critical tensors missing");
        return false;
    }
    return true;
}

void FireRedVadModel::InitAudioProcessor() {
    STFTConfig cfg;
    cfg.sample_rate = hparams_.sample_rate;
    cfg.frame_size = hparams_.sample_rate * hparams_.frame_length_ms / 1000;
    cfg.frame_step = hparams_.sample_rate * hparams_.frame_shift_ms / 1000;
    cfg.n_fft = 512;
    cfg.n_mels = hparams_.idim;
    cfg.window_type = WindowType::POVEY;  // Match kaldi_native_fbank training.
    cfg.use_lfr = false;
    cfg.use_cmvn = false;  // CMVN applied manually in ApplyCmvn() to match Python convention.
    audio_processor_ = std::make_unique<AudioProcessor>(cfg);
}

bool FireRedVadModel::LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                                 ggml_backend_t backend) {
    if (!gguf_data) return false;
    backend_ = backend;
    LoadHParams(ctx_gguf);
    meta_.audio_sample_rate = hparams_.sample_rate;
    meta_.n_mels = hparams_.idim;
    if (!MapTensors(gguf_data)) return false;
    InitAudioProcessor();

    RS_LOG_INFO("FireRedVAD loaded: idim=%d H=%d P=%d R=%d M=%d N1=%d LP=%d threshold=%.2f",
                hparams_.idim, hparams_.H, hparams_.P, hparams_.R, hparams_.M,
                hparams_.N1, hparams_.lookback_padding, hparams_.speech_threshold);
    return true;
}

bool FireRedVadModel::Load(const std::unique_ptr<rs_context_t>& ctx,
                           ggml_backend_t backend) {
    if (!ctx || !ctx->gguf_data) return false;
    return LoadDirect(ctx->gguf_data, ctx->ctx_gguf, backend);
}

std::shared_ptr<RSState> FireRedVadModel::CreateState() {
    auto s = std::make_shared<FireRedVadState>();
    const int total_fsmns = hparams_.R;       // 1 initial + (R-1) blocks
    s->caches.assign(total_fsmns,
                     std::vector<float>(
                         (size_t)hparams_.lookback_padding * hparams_.P, 0.0f));
    return s;
}

void FireRedVadModel::Reset(RSState& state) {
    auto& s = dynamic_cast<FireRedVadState&>(state);
    for (auto& c : s.caches) std::fill(c.begin(), c.end(), 0.0f);
    s.audio_remainder.clear();
    s.last_probs.clear();
    s.last_results.clear();
    s.frame_cnt = 0;
    s.smooth_window.clear();
    s.smooth_window_sum = 0.0f;
    s.fsm_state = VadFsmState::SILENCE;
    s.speech_cnt = 0;
    s.silence_cnt = 0;
    s.hit_max_speech = false;
    s.last_speech_start_frame = -1;
    s.last_speech_end_frame = -1;
    s.triggered = false;
    s.speech_prob = 0.0f;
}

void FireRedVadModel::ApplyCmvn(std::vector<float>& feat, int T) const {
    const int D = hparams_.idim;
    if ((int)feat.size() < T * D) return;
    const float* means = (const float*)weights_.cmvn_means->data;
    const float* inv_std = (const float*)weights_.cmvn_inv_std->data;
    for (int t = 0; t < T; ++t) {
        float* row = feat.data() + (size_t)t * D;
        for (int d = 0; d < D; ++d) {
            row[d] = (row[d] - means[d]) * inv_std[d];
        }
    }
}

// ============================================================
// FSMN sub-graph builder.
//
// Inputs:
//   p:        ne=[P, T]  current input (Linear convention)
//   cache_in: ne=[LP, P] previous lookback context
//   kernel:   ne=[N1, 1, P] depthwise conv kernel (cast to F16 internally)
//
// Algorithm (matches FSMN.forward in detect_model.py):
//   p_T   = transpose(p)               # ne=[T, P]
//   concat = ggml_concat(cache_in, p_T, dim=0)  # ne=[LP+T, P]
//   conv  = conv_1d_dw(kernel, concat, s=1, p=0, d=S1)  # ne=[T, P, 1]
//   mem_T = p_T + conv                 # ne=[T, P] residual on transposed view
//   mem   = transpose(mem_T)           # ne=[P, T]
//   cache_out = last LP rows of concat # ne=[LP, P]
//
// Returns (memory_out, cache_out). cache_out tensor is set as graph output.
// ============================================================
struct FsmnLayerResult {
    ggml_tensor* memory_out = nullptr;  // ne=[P, T]
    ggml_tensor* cache_out  = nullptr;  // ne=[LP, P]
};

static FsmnLayerResult fireredvad_fsmn_layer(
        ggml_context* ctx0,
        ggml_tensor* p,            // ne=[P, T]
        ggml_tensor* cache_in,     // ne=[LP, P]
        ggml_tensor* kernel,       // ne=[N1, 1, P]
        int S1,
        ggml_tensor* skip_residual /* optional: ne=[P, T], nullptr to skip */)
{
    const int64_t P  = p->ne[0];
    const int64_t T  = p->ne[1];
    const int64_t LP = cache_in->ne[0];

    // p_T = transpose(p) -> ne=[T, P]  (needs cont since transpose only swaps strides)
    ggml_tensor* p_T = ggml_cont(ctx0, ggml_transpose(ctx0, p));  // ne=[T, P]

    // concat along dim=0: ne=[LP+T, P]
    ggml_tensor* concat = ggml_concat(ctx0, cache_in, p_T, 0);

    // Reshape to 3D for conv1d_dw: ne=[L=LP+T, C=P, N=1]
    ggml_tensor* concat_3d = ggml_reshape_3d(ctx0, concat, LP + T, P, 1);

    // Depthwise conv expects F16 kernel. Cast once.
    ggml_tensor* k_f16 = ggml_cast(ctx0, kernel, GGML_TYPE_F16);

    // Stride 1, padding 0, dilation S1 -> output length = LP+T - S1*(N1-1) = T
    ggml_tensor* conv = ggml_conv_1d_dw(ctx0, k_f16, concat_3d, 1, 0, S1);
    // conv ne=[T, P, 1]

    ggml_tensor* conv_2d = ggml_reshape_2d(ctx0, conv, T, P);  // ne=[T, P]

    // memory_T = p_T + conv_2d  (both ne=[T, P])
    ggml_tensor* mem_T = ggml_add(ctx0, p_T, conv_2d);

    // Transpose back to ne=[P, T] for downstream Linear ops.
    ggml_tensor* mem = ggml_cont(ctx0, ggml_transpose(ctx0, mem_T));

    if (skip_residual) {
        mem = ggml_add(ctx0, mem, skip_residual);
    }

    // cache_out: last LP rows of concat (ne=[LP, P]).
    // Concat ne=[LP+T, P], nb[0]=4, nb[1]=4*(LP+T). View at offset T*4, ne0=LP.
    ggml_tensor* cache_out_view = ggml_view_2d(
        ctx0, concat,
        LP, P,
        concat->nb[1],
        (size_t)T * concat->nb[0]);
    ggml_tensor* cache_out = ggml_cont(ctx0, cache_out_view);

    FsmnLayerResult r;
    r.memory_out = mem;
    r.cache_out = cache_out;
    return r;
}

// ============================================================
// Build the full DFSMN forward graph for a chunk of T frames.
// Inputs (set via ggml_backend_tensor_set):
//   feat: ne=[idim, T]        (CMVN-applied fbank)
//   cache_in[r]: ne=[LP, P]   (one per FSMN layer; r in [0, R))
// Outputs (read back):
//   probs:        ne=[odim, T]  per-frame probability after sigmoid
//   cache_out[r]: ne=[LP, P]    updated lookback context
// ============================================================
static ggml_cgraph* fireredvad_build_graph(
        const FireRedVadWeights& w,
        const FireRedVadHParams& hp,
        ggml_context* ctx0,
        ggml_tensor* feat,
        const std::vector<ggml_tensor*>& cache_in /* size R */,
        std::vector<ggml_tensor*>& cache_out_named /* output, size R */)
{
    auto add_bias = [&](ggml_tensor* x, ggml_tensor* b) -> ggml_tensor* {
        // x: ne=[H, T], b: ne=[H]. ggml_add broadcasts ne=[H] -> ne=[H, T].
        return ggml_add(ctx0, x, b);
    };

    // Stage A: fc1 -> ReLU
    ggml_tensor* h = ggml_mul_mat(ctx0, w.fc1_w, feat);          // ne=[H, T]
    h = add_bias(h, w.fc1_b);
    h = ggml_relu(ctx0, h);

    // Stage B: fc2 -> ReLU
    ggml_tensor* p = ggml_mul_mat(ctx0, w.fc2_w, h);             // ne=[P, T]
    p = add_bias(p, w.fc2_b);
    p = ggml_relu(ctx0, p);

    cache_out_named.assign(hp.R, nullptr);

    // Stage C: initial FSMN (no skip)
    auto r0 = fireredvad_fsmn_layer(
        ctx0, p, cache_in[0], w.fsmn1_lookback_w, hp.S1, /*skip=*/nullptr);
    ggml_tensor* mem = r0.memory_out;                            // ne=[P, T]
    cache_out_named[0] = r0.cache_out;

    // Stage D: R-1 DFSMN blocks
    for (int i = 0; i < (int)w.blocks.size(); ++i) {
        const FireRedDFSMNBlock& blk = w.blocks[i];
        ggml_tensor* residual = mem;                             // skip carrier

        ggml_tensor* hb = ggml_mul_mat(ctx0, blk.fc1_w, mem);    // ne=[H, T]
        hb = add_bias(hb, blk.fc1_b);
        hb = ggml_relu(ctx0, hb);

        ggml_tensor* pb = ggml_mul_mat(ctx0, blk.fc2_w, hb);     // ne=[P, T]  (no bias)

        auto r = fireredvad_fsmn_layer(
            ctx0, pb, cache_in[i + 1], blk.lookback_w, hp.S1,
            /*skip=*/residual);
        mem = r.memory_out;
        cache_out_named[i + 1] = r.cache_out;
    }

    // Stage E: M DNN layers (Linear + ReLU)
    for (int m = 0; m < (int)w.dnns.size(); ++m) {
        ggml_tensor* o = ggml_mul_mat(ctx0, w.dnns[m].w, mem);   // ne=[H, T]
        o = add_bias(o, w.dnns[m].b);
        o = ggml_relu(ctx0, o);
        mem = o;
    }

    // Stage F: out Linear + sigmoid
    ggml_tensor* logits = ggml_mul_mat(ctx0, w.out_w, mem);      // ne=[odim, T]
    logits = ggml_add(ctx0, logits, w.out_b);
    ggml_tensor* probs = ggml_sigmoid(ctx0, logits);
    ggml_set_name(probs, "probs");
    ggml_set_output(probs);

    // Name + mark cache_out tensors as outputs
    for (int r = 0; r < hp.R; ++r) {
        char buf[32];
        snprintf(buf, sizeof(buf), "cache_out_%d", r);
        ggml_set_name(cache_out_named[r], buf);
        ggml_set_output(cache_out_named[r]);
    }

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, FIREREDVAD_MAX_NODES, false);
    ggml_build_forward_expand(gf, probs);
    for (int r = 0; r < hp.R; ++r) {
        ggml_build_forward_expand(gf, cache_out_named[r]);
    }
    return gf;
}

bool FireRedVadModel::Forward(const std::vector<float>& feat, int T,
                              FireRedVadState& state,
                              std::vector<float>& out_probs) {
    if (T <= 0) {
        out_probs.clear();
        return true;
    }
    const int D = hparams_.idim;
    const int P = hparams_.P;
    const int LP = hparams_.lookback_padding;
    const int R = hparams_.R;

    if ((int)feat.size() < T * D) {
        RS_LOG_ERR("FireRedVAD::Forward: feat size %zu < T*D = %d", feat.size(), T * D);
        return false;
    }

    ggml_backend_t compute_backend = backend_;
    bool local_backend = false;
    if (!compute_backend) {
        compute_backend = ggml_backend_cpu_init();
        local_backend = true;
    }

    const size_t meta_size = ggml_tensor_overhead() * FIREREDVAD_MAX_NODES
                           + ggml_graph_overhead_custom(FIREREDVAD_MAX_NODES, false);
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params params = { meta_size, meta.data(), true };
    ggml_context* meta_ctx = ggml_init(params);

    ggml_tensor* tensor_feat = ggml_new_tensor_2d(meta_ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(tensor_feat, "feat");
    ggml_set_input(tensor_feat);

    std::vector<ggml_tensor*> cache_in(R, nullptr);
    for (int r = 0; r < R; ++r) {
        ggml_tensor* t = ggml_new_tensor_2d(meta_ctx, GGML_TYPE_F32, LP, P);
        char buf[32];
        snprintf(buf, sizeof(buf), "cache_in_%d", r);
        ggml_set_name(t, buf);
        ggml_set_input(t);
        cache_in[r] = t;
    }

    std::vector<ggml_tensor*> cache_out;
    ggml_cgraph* gf = fireredvad_build_graph(weights_, hparams_, meta_ctx,
                                             tensor_feat, cache_in, cache_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(compute_backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        RS_LOG_ERR("FireRedVAD: gallocr alloc failed");
        ggml_gallocr_free(alloc);
        ggml_free(meta_ctx);
        if (local_backend) ggml_backend_free(compute_backend);
        return false;
    }

    // Upload feat + caches
    ggml_backend_tensor_set(tensor_feat, feat.data(), 0,
                            (size_t)T * D * sizeof(float));
    for (int r = 0; r < R; ++r) {
        ggml_backend_tensor_set(cache_in[r], state.caches[r].data(), 0,
                                state.caches[r].size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(compute_backend, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("FireRedVAD: graph compute failed");
        ggml_gallocr_free(alloc);
        ggml_free(meta_ctx);
        if (local_backend) ggml_backend_free(compute_backend);
        return false;
    }

    // Read outputs
    ggml_tensor* probs_t = ggml_graph_get_tensor(gf, "probs");
    out_probs.assign((size_t)T * hparams_.odim, 0.0f);
    ggml_backend_tensor_get(probs_t, out_probs.data(), 0,
                            out_probs.size() * sizeof(float));

    for (int r = 0; r < R; ++r) {
        char buf[32];
        snprintf(buf, sizeof(buf), "cache_out_%d", r);
        ggml_tensor* co = ggml_graph_get_tensor(gf, buf);
        ggml_backend_tensor_get(co, state.caches[r].data(), 0,
                                state.caches[r].size() * sizeof(float));
    }

    ggml_gallocr_free(alloc);
    ggml_free(meta_ctx);
    if (local_backend) ggml_backend_free(compute_backend);
    return true;
}

// ============================================================
// Postprocessor: 4-state machine identical to Python
// stream_vad_postprocessor.py:state_transition().
// ============================================================
StreamVadFrameResult FireRedVadModel::ProcessOneFrame(
        float raw_prob, FireRedVadState& s) const {
    if (raw_prob < 0.0f) raw_prob = 0.0f;
    if (raw_prob > 1.0f) raw_prob = 1.0f;
    s.frame_cnt += 1;

    // Smooth window mean
    float smoothed = raw_prob;
    const int W = std::max(1, hparams_.smooth_window_size);
    if (W > 1) {
        s.smooth_window.push_back(raw_prob);
        s.smooth_window_sum += raw_prob;
        if ((int)s.smooth_window.size() > W) {
            s.smooth_window_sum -= s.smooth_window.front();
            s.smooth_window.pop_front();
        }
        smoothed = s.smooth_window_sum / (float)s.smooth_window.size();
    }
    const bool is_speech = (smoothed >= hparams_.speech_threshold);

    StreamVadFrameResult r;
    r.frame_idx = s.frame_cnt;
    r.is_speech = is_speech;
    r.raw_prob = raw_prob;
    r.smoothed_prob = smoothed;

    const int min_speech = hparams_.min_speech_frame;
    const int max_speech = hparams_.max_speech_frame;
    const int min_silence = hparams_.min_silence_frame;
    const int pad_start = std::max(W, hparams_.pad_start_frame);

    if (s.hit_max_speech) {
        r.is_speech_start = true;
        r.speech_start_frame = s.frame_cnt;
        s.last_speech_start_frame = r.speech_start_frame;
        s.hit_max_speech = false;
    }

    switch (s.fsm_state) {
        case VadFsmState::SILENCE: {
            if (is_speech) {
                s.fsm_state = VadFsmState::POSSIBLE_SPEECH;
                s.speech_cnt += 1;
            } else {
                s.silence_cnt += 1;
                s.speech_cnt = 0;
            }
            break;
        }
        case VadFsmState::POSSIBLE_SPEECH: {
            if (is_speech) {
                s.speech_cnt += 1;
                if (s.speech_cnt >= min_speech) {
                    s.fsm_state = VadFsmState::SPEECH;
                    r.is_speech_start = true;
                    int sf = s.frame_cnt - s.speech_cnt + 1 - pad_start;
                    sf = std::max(1, sf);
                    sf = std::max(sf, s.last_speech_end_frame + 1);
                    r.speech_start_frame = sf;
                    s.last_speech_start_frame = sf;
                    s.silence_cnt = 0;
                }
            } else {
                s.fsm_state = VadFsmState::SILENCE;
                s.silence_cnt = 1;
                s.speech_cnt = 0;
            }
            break;
        }
        case VadFsmState::SPEECH: {
            s.speech_cnt += 1;
            if (is_speech) {
                s.silence_cnt = 0;
                if (s.speech_cnt >= max_speech) {
                    s.hit_max_speech = true;
                    s.speech_cnt = 0;
                    r.is_speech_end = true;
                    r.speech_end_frame = s.frame_cnt;
                    r.speech_start_frame = s.last_speech_start_frame;
                    s.last_speech_start_frame = -1;
                    s.last_speech_end_frame = r.speech_end_frame;
                }
            } else {
                s.fsm_state = VadFsmState::POSSIBLE_SILENCE;
                s.silence_cnt += 1;
            }
            break;
        }
        case VadFsmState::POSSIBLE_SILENCE: {
            s.speech_cnt += 1;
            if (is_speech) {
                s.fsm_state = VadFsmState::SPEECH;
                s.silence_cnt = 0;
                if (s.speech_cnt >= max_speech) {
                    s.hit_max_speech = true;
                    s.speech_cnt = 0;
                    r.is_speech_end = true;
                    r.speech_end_frame = s.frame_cnt;
                    r.speech_start_frame = s.last_speech_start_frame;
                    s.last_speech_start_frame = -1;
                    s.last_speech_end_frame = r.speech_end_frame;
                }
            } else {
                s.silence_cnt += 1;
                if (s.silence_cnt >= min_silence) {
                    s.fsm_state = VadFsmState::SILENCE;
                    r.is_speech_end = true;
                    r.speech_end_frame = s.frame_cnt;
                    r.speech_start_frame = s.last_speech_start_frame;
                    s.last_speech_end_frame = r.speech_end_frame;
                    s.last_speech_start_frame = -1;
                    s.speech_cnt = 0;
                }
            }
            break;
        }
    }
    return r;
}

std::vector<StreamVadFrameResult>
FireRedVadModel::DetectStreamingChunk(const std::vector<float>& pcm,
                                      RSState& state) {
    auto& s = dynamic_cast<FireRedVadState&>(state);
    std::vector<StreamVadFrameResult> out;
    if (pcm.empty() && s.audio_remainder.empty()) return out;

    // Stitch leftover samples from previous chunk.
    const int frame_size = hparams_.sample_rate * hparams_.frame_length_ms / 1000;
    const int frame_step = hparams_.sample_rate * hparams_.frame_shift_ms / 1000;
    std::vector<float> stitched;
    stitched.reserve(s.audio_remainder.size() + pcm.size());
    stitched.insert(stitched.end(), s.audio_remainder.begin(),
                    s.audio_remainder.end());
    stitched.insert(stitched.end(), pcm.begin(), pcm.end());

    if ((int)stitched.size() < frame_size) {
        // Not enough samples for a single frame yet — buffer and bail.
        s.audio_remainder = std::move(stitched);
        return out;
    }

    // Compute fbank for all complete frames in `stitched`.
    // Scale PCM from normalized [-1, 1] float to int16 range to match the
    // training-time feature distribution. Python reference reads wavs with
    // sf.read(..., dtype="int16") and feeds raw int16 magnitudes to
    // kaldi_native_fbank; without this rescale every log-mel is off by
    // ~log(32768^2) ≈ 20.79 and CMVN-applied features fall off-manifold,
    // crushing the model's speech probabilities.
    std::vector<float> stitched_scaled(stitched.size());
    for (size_t i = 0; i < stitched.size(); ++i) {
        stitched_scaled[i] = stitched[i] * 32768.0f;
    }
    std::vector<float> fbank;
    audio_processor_->Compute(stitched_scaled, fbank);
    const int D = hparams_.idim;
    const int T = (int)(fbank.size() / D);
    if (T <= 0) {
        // Save remainder for next chunk.
        int consumed = 0;  // nothing consumed (no frame produced)
        s.audio_remainder.assign(stitched.begin() + consumed, stitched.end());
        return out;
    }

    // Save remainder: ComputeFbank consumes (T-1)*step + frame_size samples;
    // everything past that is the unconsumed tail.
    const int consumed_samples = (T - 1) * frame_step + frame_size;
    if (consumed_samples < (int)stitched.size()) {
        // Keep enough overlap for the next chunk's first frame: last
        // (frame_size - frame_step) samples participate in the next frame.
        const int keep_start = (T - 1) * frame_step + frame_step;  // = T*step
        if (keep_start < (int)stitched.size()) {
            s.audio_remainder.assign(stitched.begin() + keep_start, stitched.end());
        } else {
            s.audio_remainder.clear();
        }
    } else {
        // Need to keep the last (frame_size - frame_step) samples so the next
        // chunk can produce its first frame at the right offset.
        const int keep_start = (int)stitched.size() - (frame_size - frame_step);
        if (keep_start > 0 && keep_start < (int)stitched.size()) {
            s.audio_remainder.assign(stitched.begin() + keep_start, stitched.end());
        } else {
            s.audio_remainder.clear();
        }
    }

    ApplyCmvn(fbank, T);

    std::vector<float> probs;
    if (!Forward(fbank, T, s, probs)) return out;

    out.reserve(T);
    for (int t = 0; t < T; ++t) {
        out.push_back(ProcessOneFrame(probs[t], s));
    }
    s.last_probs = std::move(probs);
    s.last_results = out;
    if (!out.empty()) {
        s.speech_prob = out.back().raw_prob;
        s.triggered = (s.fsm_state == VadFsmState::SPEECH
                       || s.fsm_state == VadFsmState::POSSIBLE_SILENCE);
    }
    return out;
}

FireRedVadModel::FullResult
FireRedVadModel::DetectFull(const std::vector<float>& pcm) {
    auto state_ptr = std::dynamic_pointer_cast<FireRedVadState>(CreateState());
    FireRedVadState& s = *state_ptr;
    FullResult result;
    if (pcm.empty()) return result;

    result.duration = (float)pcm.size() / (float)hparams_.sample_rate;

    std::vector<float> pcm_scaled(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        pcm_scaled[i] = pcm[i] * 32768.0f;
    }
    std::vector<float> fbank;
    audio_processor_->Compute(pcm_scaled, fbank);
    const int D = hparams_.idim;
    const int T = (int)(fbank.size() / D);
    if (T <= 0) return result;
    ApplyCmvn(fbank, T);

    // Chunk into chunk_max_frame slices to bound memory.
    std::vector<float> all_probs;
    all_probs.reserve(T);
    const int cmf = std::max(1, hparams_.chunk_max_frame);
    for (int start = 0; start < T; start += cmf) {
        const int cur_T = std::min(cmf, T - start);
        std::vector<float> chunk_feat(fbank.begin() + (size_t)start * D,
                                      fbank.begin() + (size_t)(start + cur_T) * D);
        std::vector<float> chunk_probs;
        if (!Forward(chunk_feat, cur_T, s, chunk_probs)) return result;
        all_probs.insert(all_probs.end(), chunk_probs.begin(), chunk_probs.end());
    }

    // Run postprocessor over the full sequence.
    result.frames.reserve(all_probs.size());
    for (float pr : all_probs) {
        result.frames.push_back(ProcessOneFrame(pr, s));
    }
    result.timestamps = ResultsToTimestamps(result.frames);
    return result;
}

std::vector<std::pair<float, float>>
FireRedVadModel::ResultsToTimestamps(
        const std::vector<StreamVadFrameResult>& results, int frame_per_second) {
    std::vector<std::pair<float, float>> out;
    if (results.empty()) return out;
    if (frame_per_second <= 0) frame_per_second = 100;

    int start = -1, end = -1;
    for (const auto& r : results) {
        if (r.is_speech_start) {
            start = std::max(0, r.speech_start_frame - 1);
            end = -1;
        } else if (r.is_speech_end) {
            end = std::max(0, r.speech_end_frame - 1);
            if (start >= 0) out.emplace_back(start, end);
            start = -1;
            end = -1;
        }
    }
    if (start != -1) {
        end = std::max(0, results.back().frame_idx - 1);
        out.emplace_back(start, end);
    }

    std::vector<std::pair<float, float>> secs;
    secs.reserve(out.size());
    const float inv = 1.0f / (float)frame_per_second;
    for (auto& p : out) {
        secs.emplace_back((float)p.first * inv, (float)p.second * inv);
    }
    return secs;
}

void FireRedVadModel::SetMode(int mode) {
    switch (mode) {
        case 0:  // VERY PERMISSIVE
            hparams_.speech_threshold = 0.3f;
            hparams_.min_speech_frame = 8;
            hparams_.min_silence_frame = 20;
            break;
        case 1:  // PERMISSIVE
            hparams_.speech_threshold = 0.5f;
            hparams_.min_speech_frame = 10;
            hparams_.min_silence_frame = 15;
            break;
        case 2:  // AGGRESSIVE
            hparams_.speech_threshold = 0.7f;
            hparams_.min_speech_frame = 15;
            hparams_.min_silence_frame = 10;
            break;
        case 3:  // VERY_AGGRESSIVE
            hparams_.speech_threshold = 0.9f;
            hparams_.min_speech_frame = 20;
            hparams_.min_silence_frame = 5;
            break;
        default:
            RS_LOG_WARN("FireRedVAD::SetMode: unknown mode %d", mode);
            break;
    }
}

// ============================================================
// ISpeechModel interface
// ============================================================
bool FireRedVadModel::Encode(const std::vector<float>& input_frames,
                             RSState& state,
                             ggml_backend_sched_t /*sched*/) {
    DetectStreamingChunk(input_frames, state);
    return true;
}

bool FireRedVadModel::Decode(RSState& /*state*/, ggml_backend_sched_t /*sched*/) {
    return true;
}

std::string FireRedVadModel::GetTranscription(RSState& state) {
    auto& s = dynamic_cast<FireRedVadState&>(state);
    return s.triggered ? "SPEECH" : "SILENCE";
}

static struct FireRedVadRegistrar {
    FireRedVadRegistrar() {
        rs_register_model_arch("firered-vad", []() -> std::shared_ptr<ISpeechModel> {
            return std::make_shared<FireRedVadModel>();
        });
    }
} s_firered_vad_registrar;
