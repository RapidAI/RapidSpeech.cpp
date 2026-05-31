// arch/campplus.cpp — see arch/campplus.h for the full pipeline contract.
//
// Implementation notes:
//   1. Weights live on `backend_` via the regular GGUF buffer (owned by the
//      C-API handle / rs_context_t). Folded BN (gamma, beta) tensors are
//      allocated in a *separate* synthetic ggml_context (`bn_ctx_`) backed by
//      a private `bn_buffer_` so they can be referenced from the graph the
//      same way as any other weight.
//   2. The graph reduces every BN to a per-channel `mul + add`. ReLU/sigmoid
//      come from ggml directly. CAM gate uses ggml_pool_1d for the seg_pool
//      and ggml_mean for the global pool. StatsPool is mean + std along T.
//   3. Fbank is a Kaldi-compatible Povey-window 80-bin filterbank with
//      preemphasis and per-utterance mean subtract. We inline a small
//      fbank routine here because the shared `AudioProcessor` doesn't expose
//      the per-utterance subtract step.

#include "arch/campplus.h"

#include "core/rs_context.h"
#include "frontend/audio_processor.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Hard cap on graph nodes. CAMPPlus has 12+24+16=52 dense layers × ~10 ops
// each + head/tail ~120 → ~640 nodes. 2048 leaves comfortable headroom.
#define CAMPPLUS_MAX_NODES 4096

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

CAMPPlusModel::CAMPPlusModel() {
    meta_.arch_name = "campplus";
    meta_.audio_sample_rate = 16000;
    meta_.n_mels = 80;
    meta_.vocab_size = 0;
    meta_.use_external_frontend = true;  // we own the fbank step
    meta_.window_type = WindowType::POVEY;
}

CAMPPlusModel::~CAMPPlusModel() {
    if (bn_buffer_) {
        ggml_backend_buffer_free(bn_buffer_);
        bn_buffer_ = nullptr;
    }
    if (bn_ctx_) {
        ggml_free(bn_ctx_);
        bn_ctx_ = nullptr;
    }
}

std::shared_ptr<RSState> CAMPPlusModel::CreateState() {
    return std::make_shared<CAMPPlusState>();
}

// ---------------------------------------------------------------------------
// HParams
// ---------------------------------------------------------------------------

bool CAMPPlusModel::LoadHParams(gguf_context* ctx_gguf) {
    if (!ctx_gguf) return true;  // accept defaults

    auto ki = [&](const char* k, int def) -> int {
        int64_t key = gguf_find_key(ctx_gguf, k);
        if (key < 0) return def;
        auto t = gguf_get_kv_type(ctx_gguf, key);
        if (t == GGUF_TYPE_UINT32) return (int)gguf_get_val_u32(ctx_gguf, key);
        return (int)gguf_get_val_i32(ctx_gguf, key);
    };
    auto kf = [&](const char* k, float def) -> float {
        int64_t key = gguf_find_key(ctx_gguf, k);
        return key < 0 ? def : gguf_get_val_f32(ctx_gguf, key);
    };

    hparams_.sample_rate      = ki("campplus.sample_rate", 16000);
    hparams_.n_mels           = ki("campplus.n_mels", 80);
    hparams_.fcm_in_channels  = ki("campplus.fcm_in_channels", 1);
    hparams_.fcm_out_channels = ki("campplus.fcm_out_channels", 32);
    hparams_.tdnn_in_channels = ki("campplus.tdnn_in_channels", 320);
    hparams_.tdnn_out_channels= ki("campplus.tdnn_out_channels", 128);
    hparams_.tdnn_kernel      = ki("campplus.tdnn_kernel", 5);
    hparams_.tdnn_stride      = ki("campplus.tdnn_stride", 2);
    hparams_.growth_rate      = ki("campplus.growth_rate", 32);
    hparams_.bn_channels      = ki("campplus.bn_channels", 128);
    hparams_.cam_kernel       = ki("campplus.cam_kernel", 3);
    hparams_.block1_layers    = ki("campplus.block1_layers", 12);
    hparams_.block1_dilation  = ki("campplus.block1_dilation", 1);
    hparams_.block2_layers    = ki("campplus.block2_layers", 24);
    hparams_.block2_dilation  = ki("campplus.block2_dilation", 2);
    hparams_.block3_layers    = ki("campplus.block3_layers", 16);
    hparams_.block3_dilation  = ki("campplus.block3_dilation", 2);
    hparams_.transit1_out     = ki("campplus.transit1_out", 256);
    hparams_.transit2_out     = ki("campplus.transit2_out", 512);
    hparams_.transit3_out     = ki("campplus.transit3_out", 512);
    hparams_.stats_pool_dim   = ki("campplus.stats_pool_dim", 1024);
    hparams_.embed_dim        = ki("campplus.embed_dim", 192);
    hparams_.bn_eps           = kf("campplus.bn_eps", 1e-5f);

    return true;
}

// ---------------------------------------------------------------------------
// MapTensors
// ---------------------------------------------------------------------------

bool CAMPPlusModel::MapTensors(ggml_context* gguf_data) {
    if (!gguf_data) return false;

    int n_missing = 0;
    auto get = [&](const std::string& name) -> ggml_tensor* {
        ggml_tensor* t = ggml_get_tensor(gguf_data, name.c_str());
        if (!t) {
            RS_LOG_DEBUG("campplus: tensor missing: %s", name.c_str());
            n_missing++;
        }
        return t;
    };

    auto bind_bn = [&](CAMPPlusBN& dst, const std::string& base) {
        // We keep raw pointers to the source mean/var/(weight)/(bias). The
        // folded gamma/beta are produced later in FoldAllBN; this only
        // records the channel count and stashes the source tensors via a
        // tiny lambda trick: store mean ptr in `gamma`, var in `beta`
        // temporarily — FoldAllBN swaps them for the real folded ones.
        // (The temp pointers carry channel info from ne[0].)
        ggml_tensor* m = get(base + ".running_mean");
        ggml_tensor* v = get(base + ".running_var");
        if (!m || !v) {
            return;
        }
        dst.gamma = m;       // temp — replaced in FoldAllBN
        dst.beta = v;        // temp — replaced in FoldAllBN
        dst.channels = (int)m->ne[0];
    };

    // --- FCM head ---
    weights_.head.conv1_w = get("head.conv1.weight");
    bind_bn(weights_.head.bn1, "head.bn1");
    weights_.head.conv2_w = get("head.conv2.weight");
    bind_bn(weights_.head.bn2, "head.bn2");

    auto bind_resblock = [&](CAMPPlusResBlock& b, const std::string& base, int stride) {
        b.stride = stride;
        b.conv1_w = get(base + ".conv1.weight");
        bind_bn(b.bn1, base + ".bn1");
        b.conv2_w = get(base + ".conv2.weight");
        bind_bn(b.bn2, base + ".bn2");
        // Shortcut only on stride!=1 blocks.
        ggml_tensor* sc_w = ggml_get_tensor(gguf_data, (base + ".shortcut.0.weight").c_str());
        if (sc_w) {
            b.has_shortcut = true;
            b.sc_w = sc_w;
            bind_bn(b.sc_bn, base + ".shortcut.1");
        } else {
            b.has_shortcut = false;
        }
        if (b.conv1_w) {
            b.in_channels = (int)b.conv1_w->ne[2];
            b.out_channels = (int)b.conv1_w->ne[3];
        }
    };

    weights_.head.layer1.assign(2, CAMPPlusResBlock{});
    bind_resblock(weights_.head.layer1[0], "head.layer1.0", 2);
    bind_resblock(weights_.head.layer1[1], "head.layer1.1", 1);
    weights_.head.layer2.assign(2, CAMPPlusResBlock{});
    bind_resblock(weights_.head.layer2[0], "head.layer2.0", 2);
    bind_resblock(weights_.head.layer2[1], "head.layer2.1", 1);

    // --- xvector units ---
    auto bind_unit = [&](CAMPPlusUnit& u, const std::string& base, int default_kw) {
        u.lin_w = get(base + ".linear.weight");
        bind_bn(u.bn, base + ".nl.bn");
        if (u.lin_w) {
            u.kernel_w = (int)u.lin_w->ne[0];
            u.in_dim   = (int)u.lin_w->ne[1];
            u.out_dim  = (int)u.lin_w->ne[2];
        } else {
            u.kernel_w = default_kw;
        }
    };

    bind_unit(weights_.tdnn,     "xv.tdnn",     hparams_.tdnn_kernel);
    bind_unit(weights_.transit1, "xv.transit1", 1);
    bind_unit(weights_.transit2, "xv.transit2", 1);
    bind_unit(weights_.transit3, "xv.transit3", 1);
    bind_unit(weights_.dense,    "xv.dense",    1);

    // out_nl: bare BN, no linear.
    bind_bn(weights_.out_nl_bn, "xv.out_nl.bn");

    // --- Dense blocks ---
    auto bind_dense_block = [&](CAMPPlusDenseBlock& blk, const std::string& base,
                                int n_layers, int dilation) {
        blk.num_layers = n_layers;
        blk.dilation = dilation;
        blk.layers.assign((size_t)n_layers, CAMPPlusDenseLayer{});
        for (int i = 0; i < n_layers; i++) {
            auto& L = blk.layers[(size_t)i];
            L.dilation = dilation;
            const std::string b = base + ".tdnnd" + std::to_string(i + 1);
            bind_bn(L.nonl1_bn, b + ".nonl1.bn");
            L.l1_w = get(b + ".l1.weight");
            bind_bn(L.nonl2_bn, b + ".nonl2.bn");
            L.cam_ll_w = get(b + ".cam.ll.weight");
            L.cam_l1_w = get(b + ".cam.l1.weight");
            L.cam_l1_b = get(b + ".cam.l1.bias");
            L.cam_l2_w = get(b + ".cam.l2.weight");
            L.cam_l2_b = get(b + ".cam.l2.bias");
            if (L.l1_w) {
                L.in_channels = (int)L.l1_w->ne[1];
            }
        }
    };

    bind_dense_block(weights_.block1, "xv.block1",
                     hparams_.block1_layers, hparams_.block1_dilation);
    bind_dense_block(weights_.block2, "xv.block2",
                     hparams_.block2_layers, hparams_.block2_dilation);
    bind_dense_block(weights_.block3, "xv.block3",
                     hparams_.block3_layers, hparams_.block3_dilation);

    // Sanity: must have anchors.
    bool ok = weights_.head.conv1_w && weights_.head.conv2_w &&
              weights_.tdnn.lin_w && weights_.dense.lin_w &&
              !weights_.block1.layers.empty() && weights_.block1.layers[0].l1_w &&
              !weights_.block2.layers.empty() && weights_.block2.layers[0].l1_w &&
              !weights_.block3.layers.empty() && weights_.block3.layers[0].l1_w &&
              weights_.out_nl_bn.channels > 0;
    if (!ok) {
        RS_LOG_ERR("campplus: required tensors missing (n_missing=%d)", n_missing);
        return false;
    }
    if (n_missing > 0) {
        RS_LOG_WARN("campplus: %d optional tensors missing (proceeding)", n_missing);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Read any tensor (F16/F32/quantized) into a host F32 vector.
// ---------------------------------------------------------------------------

static std::vector<float> rs_campplus_read_f32(ggml_tensor* t) {
    if (!t) return {};
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return out;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
        return out;
    }
    const size_t raw = ggml_nbytes(t);
    std::vector<uint8_t> tmp(raw);
    ggml_backend_tensor_get(t, tmp.data(), 0, raw);
    const ggml_type_traits* tt = ggml_get_type_traits(t->type);
    if (tt && tt->to_float) {
        tt->to_float(tmp.data(), out.data(), (int64_t)n);
    } else {
        std::fill(out.begin(), out.end(), 0.0f);
    }
    return out;
}

// ---------------------------------------------------------------------------
// FoldAllBN — read every BN (running_mean, running_var, weight, bias) and
// emit folded (gamma_eff, beta_eff) per-channel tensors into bn_ctx_/bn_buffer_.
//
//   gamma_eff = w / sqrt(var + eps)         (or 1/sqrt(var+eps) if affine=False)
//   beta_eff  = b - mean * gamma_eff        (or -mean*gamma_eff if affine=False)
// ---------------------------------------------------------------------------

bool CAMPPlusModel::FoldAllBN(ggml_context* gguf_data) {
    if (!backend_) {
        RS_LOG_ERR("campplus: FoldAllBN called without backend");
        return false;
    }

    // Collect every (BN, base_name) pair and pass.
    struct Entry {
        CAMPPlusBN* bn;
        std::string base;  // e.g. "head.bn1"
    };
    std::vector<Entry> entries;

    auto add = [&](CAMPPlusBN& bn, const std::string& base) {
        if (bn.channels > 0) entries.push_back({&bn, base});
    };

    add(weights_.head.bn1, "head.bn1");
    add(weights_.head.bn2, "head.bn2");
    auto add_resblocks = [&](std::vector<CAMPPlusResBlock>& blocks, const std::string& base) {
        for (size_t i = 0; i < blocks.size(); i++) {
            auto& b = blocks[i];
            const std::string p = base + "." + std::to_string(i);
            add(b.bn1, p + ".bn1");
            add(b.bn2, p + ".bn2");
            if (b.has_shortcut) add(b.sc_bn, p + ".shortcut.1");
        }
    };
    add_resblocks(weights_.head.layer1, "head.layer1");
    add_resblocks(weights_.head.layer2, "head.layer2");

    add(weights_.tdnn.bn,     "xv.tdnn.nl.bn");
    add(weights_.transit1.bn, "xv.transit1.nl.bn");
    add(weights_.transit2.bn, "xv.transit2.nl.bn");
    add(weights_.transit3.bn, "xv.transit3.nl.bn");
    add(weights_.dense.bn,    "xv.dense.nl.bn");
    add(weights_.out_nl_bn,   "xv.out_nl.bn");

    auto add_dense_block = [&](CAMPPlusDenseBlock& blk, const std::string& base) {
        for (size_t i = 0; i < blk.layers.size(); i++) {
            auto& L = blk.layers[i];
            const std::string p = base + ".tdnnd" + std::to_string((int)i + 1);
            add(L.nonl1_bn, p + ".nonl1.bn");
            add(L.nonl2_bn, p + ".nonl2.bn");
        }
    };
    add_dense_block(weights_.block1, "xv.block1");
    add_dense_block(weights_.block2, "xv.block2");
    add_dense_block(weights_.block3, "xv.block3");

    // Allocate bn_ctx_ for 2 tensors per BN entry (gamma + beta), + headroom.
    const size_t n_tensors = entries.size() * 2 + 16;
    const size_t mem_size = ggml_tensor_overhead() * n_tensors;
    ggml_init_params ip{ mem_size, nullptr, /*no_alloc=*/true };
    bn_ctx_ = ggml_init(ip);
    if (!bn_ctx_) { RS_LOG_ERR("campplus: ggml_init for bn_ctx_ failed"); return false; }

    // Create the per-channel (gamma, beta) tensors. We don't have backing
    // memory yet — `ggml_backend_alloc_ctx_tensors` will allocate them on
    // the backend.
    struct Pending {
        ggml_tensor* gamma_t;
        ggml_tensor* beta_t;
        std::vector<float> gamma;
        std::vector<float> beta;
    };
    std::vector<Pending> pendings(entries.size());
    const float eps = hparams_.bn_eps;

    for (size_t i = 0; i < entries.size(); i++) {
        auto& e = entries[i];
        const int C = e.bn->channels;

        // Source tensors: we stashed mean in `gamma`, var in `beta` from MapTensors.
        ggml_tensor* mean_t = e.bn->gamma;
        ggml_tensor* var_t  = e.bn->beta;
        // Optional weight/bias (affine=False cases lack these).
        ggml_tensor* w_t = ggml_get_tensor(gguf_data, (e.base + ".weight").c_str());
        ggml_tensor* b_t = ggml_get_tensor(gguf_data, (e.base + ".bias").c_str());

        std::vector<float> mean = rs_campplus_read_f32(mean_t);
        std::vector<float> var  = rs_campplus_read_f32(var_t);
        std::vector<float> wv = w_t ? rs_campplus_read_f32(w_t) : std::vector<float>{};
        std::vector<float> bv = b_t ? rs_campplus_read_f32(b_t) : std::vector<float>{};

        Pending& p = pendings[i];
        p.gamma.assign((size_t)C, 0.0f);
        p.beta.assign((size_t)C, 0.0f);
        for (int c = 0; c < C; c++) {
            const float inv_std = 1.0f / std::sqrt(var[(size_t)c] + eps);
            const float g = w_t ? wv[(size_t)c] * inv_std : inv_std;
            const float bb = (b_t ? bv[(size_t)c] : 0.0f) - mean[(size_t)c] * g;
            p.gamma[(size_t)c] = g;
            p.beta[(size_t)c]  = bb;
        }

        const std::string gname = e.base + ".gamma_eff";
        const std::string bname = e.base + ".beta_eff";
        p.gamma_t = ggml_new_tensor_1d(bn_ctx_, GGML_TYPE_F32, C);
        p.beta_t  = ggml_new_tensor_1d(bn_ctx_, GGML_TYPE_F32, C);
        ggml_set_name(p.gamma_t, gname.c_str());
        ggml_set_name(p.beta_t,  bname.c_str());
    }

    // Allocate backing storage on the backend.
    bn_buffer_ = ggml_backend_alloc_ctx_tensors(bn_ctx_, backend_);
    if (!bn_buffer_) { RS_LOG_ERR("campplus: failed to alloc bn buffer"); return false; }

    // Upload + swap the temp pointers in CAMPPlusBN to the real folded tensors.
    for (size_t i = 0; i < entries.size(); i++) {
        Pending& p = pendings[i];
        ggml_backend_tensor_set(p.gamma_t, p.gamma.data(), 0,
                                p.gamma.size() * sizeof(float));
        ggml_backend_tensor_set(p.beta_t,  p.beta.data(),  0,
                                p.beta.size()  * sizeof(float));
        entries[i].bn->gamma = p.gamma_t;
        entries[i].bn->beta  = p.beta_t;
    }

    RS_LOG_INFO("campplus: folded %zu BN layers into (gamma, beta)", entries.size());
    return true;
}

// ---------------------------------------------------------------------------
// Load entry points
// ---------------------------------------------------------------------------

bool CAMPPlusModel::LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                               ggml_backend_t backend) {
    if (!gguf_data || !backend) return false;
    backend_ = backend;
    if (!LoadHParams(ctx_gguf)) return false;
    meta_.audio_sample_rate = hparams_.sample_rate;
    meta_.n_mels = hparams_.n_mels;
    if (!MapTensors(gguf_data)) return false;
    if (!FoldAllBN(gguf_data)) return false;

    RS_LOG_INFO("campplus: loaded — sr=%d, n_mels=%d, embed_dim=%d, "
                "blocks=(%d,%d,%d) dil=(%d,%d,%d)",
                hparams_.sample_rate, hparams_.n_mels, hparams_.embed_dim,
                hparams_.block1_layers, hparams_.block2_layers, hparams_.block3_layers,
                hparams_.block1_dilation, hparams_.block2_dilation, hparams_.block3_dilation);
    return true;
}

bool CAMPPlusModel::Load(const std::unique_ptr<rs_context_t>& ctx,
                         ggml_backend_t backend) {
    if (!ctx || !ctx->gguf_data) return false;
    return LoadDirect(ctx->gguf_data, ctx->ctx_gguf, backend);
}

// ---------------------------------------------------------------------------
// Fbank — Kaldi-style 80-bin filterbank with Povey window, preemphasis,
// per-utterance mean subtract along time. Returns (T, n_mels) row-major.
// ---------------------------------------------------------------------------

std::vector<float> CAMPPlusModel::ComputeFbank(const float* pcm_16k, int n_samples,
                                               int& T_frames_out) const {
    T_frames_out = 0;
    if (!pcm_16k || n_samples <= 0) return {};

    STFTConfig cfg{};
    cfg.sample_rate   = hparams_.sample_rate;
    cfg.frame_size    = hparams_.frame_size;
    cfg.frame_step    = hparams_.frame_step;
    cfg.n_mels        = hparams_.n_mels;
    cfg.window_type   = WindowType::POVEY;
    cfg.use_lfr       = false;
    cfg.use_cmvn      = false;
    cfg.fbank_num_threads = 1;

    AudioProcessor proc(cfg);
    std::vector<float> input(pcm_16k, pcm_16k + n_samples);
    std::vector<float> feats;
    proc.Compute(input, feats);
    if (feats.empty()) return {};

    const int n_mels = hparams_.n_mels;
    const int T = (int)(feats.size() / (size_t)n_mels);
    if (T <= 0) return {};

    // Per-utterance mean subtract along time (CAMPPlus normalisation).
    std::vector<double> mean((size_t)n_mels, 0.0);
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            mean[(size_t)m] += (double)feats[(size_t)t * (size_t)n_mels + (size_t)m];
        }
    }
    const double inv_T = 1.0 / (double)T;
    for (int m = 0; m < n_mels; m++) mean[(size_t)m] *= inv_T;
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            feats[(size_t)t * (size_t)n_mels + (size_t)m] -= (float)mean[(size_t)m];
        }
    }

    T_frames_out = T;
    return feats;
}

// ---------------------------------------------------------------------------
// Graph builder — declared as static helpers below the class methods for
// readability. The forward order:
//
//   fbank (T, 80) → permute → (1, 80, T) → FCM head → (320, T)
//   → tdnn → (128, T') → block1 → transit1 → block2 → transit2
//   → block3 → transit3 → out_nl → StatsPool → dense → L2 norm → 192
// ---------------------------------------------------------------------------

namespace {

// Cast a weight tensor to F16 if not already (ggml_conv_* expects F16 kernels).
inline ggml_tensor* w_f16(ggml_context* ctx, ggml_tensor* w) {
    if (w->type == GGML_TYPE_F16) return w;
    return ggml_cast(ctx, w, GGML_TYPE_F16);
}

// Apply folded BN to a (... , C) ggml tensor whose channel axis is the slowest
// (ne[2] for 3D conv2d output [W, H, C], ne[1] for conv1d output [T, C]).
// We just multiply + add, broadcasting the (C,) gamma/beta to the larger shape.
//
// For 1D feature tensors with layout [T, C] (ne0=T, ne1=C), gamma/beta are
// (C,) which broadcasts naturally over ne0. For 2D head tensors with layout
// [W, H, C], (C,) also broadcasts over ne0*ne1 because ggml broadcasts along
// the matching axes.
inline ggml_tensor* apply_bn(ggml_context* ctx, ggml_tensor* x, const CAMPPlusBN& bn) {
    // Reshape gamma/beta so they align with the channel axis of x.
    // For x.ne = [..., C, ...] we let ggml broadcast against (C,) using
    // a 1D weight + ggml_mul.
    ggml_tensor* g = bn.gamma;
    ggml_tensor* b = bn.beta;
    // x layouts we use:
    //   conv2d output: [W, H, C, N=1]   → bn shape (C,) needs reshape to (1,1,C)
    //   conv1d output: [T, C, N=1]      → bn shape (C,) needs reshape to (1,C)
    // ggml_mul broadcasts when sizes are 1 along the broadcast dim.
    const int n_dim = ggml_n_dims(x);
    if (n_dim >= 3 && x->ne[2] == g->ne[0]) {
        // 2D-conv output: broadcast on ne[2].
        ggml_tensor* g3 = ggml_reshape_3d(ctx, g, 1, 1, g->ne[0]);
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]);
        x = ggml_mul(ctx, x, g3);
        x = ggml_add(ctx, x, b3);
        return x;
    }
    if (n_dim >= 2 && x->ne[1] == g->ne[0]) {
        // 1D-conv output: broadcast on ne[1].
        ggml_tensor* g2 = ggml_reshape_2d(ctx, g, 1, g->ne[0]);
        ggml_tensor* b2 = ggml_reshape_2d(ctx, b, 1, b->ne[0]);
        x = ggml_mul(ctx, x, g2);
        x = ggml_add(ctx, x, b2);
        return x;
    }
    // Fallback: assume channel axis is ne[0].
    x = ggml_mul(ctx, x, g);
    x = ggml_add(ctx, x, b);
    return x;
}

// Bias-less Conv1d wrapper. Input layout: [T, C_in, 1] with kernel
// stored ne=[kw, C_in, C_out]. Returns [T_out, C_out, 1].
inline ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                           int stride, int pad, int dilation) {
    return ggml_conv_1d(ctx, w_f16(ctx, w), x, stride, pad, dilation);
}

// Conv2d with stride (sH, sW) and pad (pH, pW). Input layout in ggml:
// ne = [W, H, C_in, N]. ggml_conv_2d takes (s0,s1,p0,p1,d0,d1) where
// 0=W axis (time), 1=H axis (mel). Returns [W_out, H_out, C_out, N].
inline ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                           int sH, int sW, int pH, int pW) {
    return ggml_conv_2d(ctx, w_f16(ctx, w), x, sW, sH, pW, pH, 1, 1);
}

// FCM BasicResBlock graph step. Input [W,H,C_in,1] → Output [W,H_out,C_out,1].
ggml_tensor* fcm_resblock_graph(ggml_context* ctx, ggml_tensor* x,
                                const CAMPPlusResBlock& b) {
    // conv1: stride (b.stride along H, 1 along W), pad (1,1), k=3×3
    ggml_tensor* y = conv2d(ctx, b.conv1_w, x, b.stride, 1, 1, 1);
    y = apply_bn(ctx, y, b.bn1);
    y = ggml_relu(ctx, y);
    // conv2: stride (1,1), pad (1,1), k=3×3
    y = conv2d(ctx, b.conv2_w, y, 1, 1, 1, 1);
    y = apply_bn(ctx, y, b.bn2);

    // Shortcut.
    ggml_tensor* sc;
    if (b.has_shortcut) {
        // 1×1 conv with stride (b.stride, 1).
        sc = conv2d(ctx, b.sc_w, x, b.stride, 1, 0, 0);
        sc = apply_bn(ctx, sc, b.sc_bn);
    } else {
        sc = x;
    }
    y = ggml_add(ctx, y, sc);
    y = ggml_relu(ctx, y);
    return y;
}

// CAMDenseTDNN layer. Input x: [T, C_in, 1]. Output: 32-channel feature
// concat-merged with x to produce [T, C_in+32, 1].
ggml_tensor* cam_dense_layer_graph(ggml_context* ctx, ggml_tensor* x,
                                   const CAMPPlusDenseLayer& L,
                                   int cam_kernel) {
    // Pre-bottleneck BN + ReLU (input channels = L.in_channels).
    ggml_tensor* y = apply_bn(ctx, x, L.nonl1_bn);
    y = ggml_relu(ctx, y);
    // Bottleneck: Conv1d (1, in→128).
    y = conv1d(ctx, L.l1_w, y, 1, 0, 1);
    // Post-bottleneck BN + ReLU.
    y = apply_bn(ctx, y, L.nonl2_bn);
    y = ggml_relu(ctx, y);
    // y: [T, 128, 1]
    const int64_t T = y->ne[0];
    const int64_t C = y->ne[1];

    // CAM gate.
    // Branch A: linear_local conv1d (k=cam_kernel, 128→32, dilation=L.dilation,
    //            padding = (k-1)/2 * dilation).
    const int pad = ((cam_kernel - 1) / 2) * L.dilation;
    ggml_tensor* ll = conv1d(ctx, L.cam_ll_w, y, 1, pad, L.dilation);
    // ll: [T, 32, 1]

    // Branch B: context = global_mean(x_along_T) + seg_pool(x, seg_len=100).
    //   global_mean: ggml_mean reduces along ne[0] → [1, 128, 1].
    ggml_tensor* gm = ggml_mean(ctx, y);    // [1, 128, 1]

    // Segment pool: avg_pool1d(y, k=100, s=100, ceil_mode=True). PyTorch's
    // ceil_mode divides the trailing partial window by the *real* element
    // count (not by k=100). So we split: pool the n_full full windows with
    // ggml_pool_1d, then take the mean of the partial tail and concat.
    const int64_t seg_len = 100;
    const int64_t n_full = T / seg_len;
    const int64_t partial_len = T - n_full * seg_len;
    const int64_t n_seg = n_full + (partial_len > 0 ? 1 : 0);

    ggml_tensor* seg = nullptr;
    if (n_full > 0) {
        ggml_tensor* y_full = ggml_view_3d(
            ctx, y,
            n_full * seg_len, C, 1,
            y->nb[1], y->nb[2], 0);
        y_full = ggml_cont(ctx, y_full);
        seg = ggml_pool_1d(ctx, y_full, GGML_OP_POOL_AVG,
                           (int)seg_len, (int)seg_len, 0);
        // seg: [n_full, 128, 1]
    }
    if (partial_len > 0) {
        ggml_tensor* y_tail = ggml_view_3d(
            ctx, y,
            partial_len, C, 1,
            y->nb[1], y->nb[2],
            n_full * seg_len * y->nb[0]);
        y_tail = ggml_cont(ctx, y_tail);
        ggml_tensor* tail_mean = ggml_mean(ctx, y_tail);  // [1, 128, 1]
        seg = (seg == nullptr) ? tail_mean
                               : ggml_concat(ctx, seg, tail_mean, 0);
    }
    // seg: [n_seg, 128, 1]

    // Repeat each element of seg `seg_len` times along ne[0].
    //   seg [n_seg, 128, 1] → reshape to [1, n_seg, 128, 1]
    //   → repeat to [seg_len, n_seg, 128, 1]
    //   → cont + reshape to [seg_len*n_seg, 128, 1]
    //   → view first T elements along ne[0].
    ggml_tensor* seg4 = ggml_reshape_4d(ctx, seg, 1, n_seg, C, 1);
    ggml_tensor* tmpl = ggml_new_tensor_4d(ctx, seg4->type,
                                           seg_len, n_seg, C, 1);
    ggml_tensor* seg_rep = ggml_repeat(ctx, seg4, tmpl);
    // Materialise contiguously, then collapse ne[0]·ne[1] → ne[0].
    seg_rep = ggml_cont(ctx, seg_rep);
    seg_rep = ggml_reshape_3d(ctx, seg_rep, seg_len * n_seg, C, 1);
    // Slice to length T.
    ggml_tensor* seg_T = ggml_view_3d(
        ctx, seg_rep,
        T, C, 1,
        seg_rep->nb[1], seg_rep->nb[2],
        0);
    seg_T = ggml_cont(ctx, seg_T);   // make sure subsequent ops see a
                                     // contiguous tensor with stride T
    // seg_T: [T, 128, 1]

    // context = gm + seg_T (gm broadcasts over T).
    ggml_tensor* ctxv = ggml_add(ctx, seg_T, gm);

    // linear1 (k=1, 128→64) + bias + ReLU.
    ggml_tensor* m = conv1d(ctx, L.cam_l1_w, ctxv, 1, 0, 1);
    if (L.cam_l1_b) {
        ggml_tensor* b1 = ggml_reshape_3d(ctx, L.cam_l1_b, 1, L.cam_l1_b->ne[0], 1);
        m = ggml_add(ctx, m, b1);
    }
    m = ggml_relu(ctx, m);
    // linear2 (k=1, 64→32) + bias + sigmoid.
    m = conv1d(ctx, L.cam_l2_w, m, 1, 0, 1);
    if (L.cam_l2_b) {
        ggml_tensor* b2 = ggml_reshape_3d(ctx, L.cam_l2_b, 1, L.cam_l2_b->ne[0], 1);
        m = ggml_add(ctx, m, b2);
    }
    m = ggml_sigmoid(ctx, m);
    // m: [T, 32, 1]

    ggml_tensor* gated = ggml_mul(ctx, ll, m);
    // gated: [T, 32, 1]

    // Concat along channel axis (ne[1]) with the layer input x.
    ggml_tensor* out = ggml_concat(ctx, x, gated, 1);
    return out;
}

// Transit layer. Order: BN → ReLU → Conv1d(k=1, in→out).
ggml_tensor* transit_graph(ggml_context* ctx, ggml_tensor* x, const CAMPPlusUnit& u) {
    ggml_tensor* y = apply_bn(ctx, x, u.bn);
    y = ggml_relu(ctx, y);
    y = conv1d(ctx, u.lin_w, y, 1, 0, 1);
    return y;
}

// CAMPPlus full graph builder.
//
// `feat` is the input fbank with ggml shape [T, n_mels, 1, 1] (ne0=T, ne1=80).
// Returns the output 192-d L2-normalized embedding tensor [192].
ggml_cgraph* campplus_build_graph(ggml_context* ctx, ggml_tensor* feat,
                                  const CAMPPlusWeights& w,
                                  const CAMPPlusHParams& hp) {
    // ---- Reshape fbank to 2D conv input ----
    // PyTorch CAMPPlus does: x.permute(2,1,0)  — i.e. (B=1, n_mels=80, T) → conv2d
    // unsqueezes channel to give (1, 1, 80, T).
    // In ggml conv2d, layout is [W, H, C_in, N] where W=time, H=mel, C_in=1, N=1.
    // Our input feat is [T, 80, 1, 1] = [W=T, H=80, C=1, N=1]. Already correct.
    ggml_tensor* x = feat;
    if (ggml_n_dims(x) < 3) {
        x = ggml_reshape_4d(ctx, x, x->ne[0], x->ne[1], 1, 1);
    }

    // ---- FCM head ----
    // conv1 (k=3, stride=1, pad=1): 1→32
    x = conv2d(ctx, w.head.conv1_w, x, 1, 1, 1, 1);
    x = apply_bn(ctx, x, w.head.bn1);
    x = ggml_relu(ctx, x);
    // layer1 (2 resblocks, 1st stride=2 along H)
    x = fcm_resblock_graph(ctx, x, w.head.layer1[0]);
    x = fcm_resblock_graph(ctx, x, w.head.layer1[1]);
    // layer2 (2 resblocks, 1st stride=2 along H)
    x = fcm_resblock_graph(ctx, x, w.head.layer2[0]);
    x = fcm_resblock_graph(ctx, x, w.head.layer2[1]);
    // conv2 (k=3, stride=(2,1), pad=(1,1)): 32→32
    x = conv2d(ctx, w.head.conv2_w, x, 2, 1, 1, 1);
    x = apply_bn(ctx, x, w.head.bn2);
    x = ggml_relu(ctx, x);
    // x: [W=T, H=10, C=32, N=1]

    // ---- Reshape (B=1, C=32, H=10, T) → (1, 320, T) ----
    // PyTorch FCM does `out.reshape(B, C*H, T)` which flattens (C, H) with
    // C slow, H fast. ggml's post-conv2d layout is [W=T, H=10, C=32, N=1]
    // with ne[0]=T innermost. For a fixed T-slot, iterating through the
    // 320 values lands on (h=k%10, c=k/10) → H fast, C slow — exactly the
    // same order as PyTorch's reshape. So we just merge ne[1]·ne[2] into
    // a single 320 channel axis without any permute.
    x = ggml_cont(ctx, x);  // ensure contiguous before reshape
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], 1);
    // x: [T, 320, 1]

    // ---- tdnn (Conv1d 320→128, k=5, s=2) + BN + ReLU ----
    {
        const int kw = (int)w.tdnn.lin_w->ne[0];
        const int pad = (kw - 1) / 2;
        x = conv1d(ctx, w.tdnn.lin_w, x, hp.tdnn_stride, pad, 1);
        x = apply_bn(ctx, x, w.tdnn.bn);
        x = ggml_relu(ctx, x);
    }
    // x: [T', 128, 1]

    // ---- block1 (12 layers, dil=1) → transit1 ----
    for (auto& L : w.block1.layers) {
        x = cam_dense_layer_graph(ctx, x, L, hp.cam_kernel);
    }
    x = transit_graph(ctx, x, w.transit1);

    // ---- block2 (24 layers, dil=2) → transit2 ----
    for (auto& L : w.block2.layers) {
        x = cam_dense_layer_graph(ctx, x, L, hp.cam_kernel);
    }
    x = transit_graph(ctx, x, w.transit2);

    // ---- block3 (16 layers, dil=2) → transit3 ----
    for (auto& L : w.block3.layers) {
        x = cam_dense_layer_graph(ctx, x, L, hp.cam_kernel);
    }
    x = transit_graph(ctx, x, w.transit3);

    // ---- out_nl: BN + ReLU ----
    x = apply_bn(ctx, x, w.out_nl_bn);
    x = ggml_relu(ctx, x);
    // x: [T', 512, 1]

    // ---- StatsPool: concat(mean(T), std(T)) along channel axis → (1024,) ----
    // ggml_mean reduces along ne[0] → [1, C, 1].
    ggml_tensor* mean_t = ggml_mean(ctx, x);
    // std(T) = sqrt(mean((x - mean)^2) * T / (T-1)) for unbiased.
    // Simplified: subtract mean (broadcast over T) then sqrt(mean(square)).
    ggml_tensor* x_centered = ggml_sub(ctx, x, mean_t);
    ggml_tensor* var_t = ggml_mean(ctx, ggml_sqr(ctx, x_centered));
    // Apply unbiased correction T/(T-1).
    const int T_pool = (int)x->ne[0];
    if (T_pool > 1) {
        const float corr = (float)T_pool / (float)(T_pool - 1);
        var_t = ggml_scale(ctx, var_t, corr);
    }
    ggml_tensor* std_t = ggml_sqrt(ctx, var_t);
    // mean_t / std_t: [1, C, 1]. Concat along ne[1] → [1, 2C, 1].
    ggml_tensor* stats = ggml_concat(ctx, mean_t, std_t, 1);
    // Reshape to [1, 1024, 1] then run dense (Conv1d, k=1, 1024→192).

    // ---- dense (Conv1d 1024→192, k=1) + BN(affine=False) ----
    ggml_tensor* y = conv1d(ctx, w.dense.lin_w, stats, 1, 0, 1);
    y = apply_bn(ctx, y, w.dense.bn);
    // y: [1, 192, 1]

    // ---- L2 normalize along the 192 channel axis ----
    // y is [1, 192, 1]; we want to L2-norm across the ne[1] axis. Squeeze
    // to [192] first.
    y = ggml_reshape_1d(ctx, y, y->ne[1]);
    // L2 norm: y / sqrt(sum(y^2)).
    ggml_tensor* sq = ggml_sqr(ctx, y);
    // Reshape to row vector [192, 1] for sum_rows.
    sq = ggml_reshape_2d(ctx, sq, sq->ne[0], 1);
    ggml_tensor* sum_sq = ggml_sum_rows(ctx, sq);   // [1, 1]
    ggml_tensor* nrm = ggml_sqrt(ctx, sum_sq);      // [1, 1]
    // Broadcast scalar division.
    ggml_tensor* y2 = ggml_reshape_2d(ctx, y, y->ne[0], 1);
    y2 = ggml_div(ctx, y2, nrm);
    y2 = ggml_reshape_1d(ctx, y2, y2->ne[0]);

    ggml_set_name(y2, "embedding");
    ggml_set_output(y2);

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, CAMPPLUS_MAX_NODES, false);
    ggml_build_forward_expand(gf, y2);
    return gf;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Embed / Encode
// ---------------------------------------------------------------------------

bool CAMPPlusModel::Embed(const float* pcm_16k, int n_samples, RSState& state,
                          ggml_backend_sched_t /*sched*/) {
    auto& cs = dynamic_cast<CAMPPlusState&>(state);
    cs.embedding.assign((size_t)hparams_.embed_dim, 0.0f);

    if (!pcm_16k || n_samples <= 0) {
        RS_LOG_WARN("campplus: empty input");
        return false;
    }
    if (!backend_) {
        RS_LOG_ERR("campplus: backend not set");
        return false;
    }

    // ---- 1. Compute fbank ----
    int T = 0;
    std::vector<float> feats = ComputeFbank(pcm_16k, n_samples, T);
    if (feats.empty() || T <= 0) {
        RS_LOG_ERR("campplus: fbank failed (n_samples=%d)", n_samples);
        return false;
    }
    const int n_mels = hparams_.n_mels;

    // Optional fbank dump for debugging numeric divergence vs ONNX.
    if (const char* dump_path = std::getenv("RS_CAMPPLUS_DUMP_FBANK")) {
        FILE* f = std::fopen(dump_path, "wb");
        if (f) {
            int32_t hdr[2] = {T, n_mels};
            std::fwrite(hdr, sizeof(int32_t), 2, f);
            std::fwrite(feats.data(), sizeof(float),
                        feats.size(), f);
            std::fclose(f);
            RS_LOG_INFO("campplus: dumped (T=%d, n_mels=%d) fbank to %s",
                        T, n_mels, dump_path);
        }
    }

    // PyTorch input layout for the FCM is (B=1, C=1, H=n_mels, W=T). In ggml
    // this is ne=[W=T, H=n_mels, C=1, N=1]. Our `feats` is row-major
    // (T, n_mels) — i.e. iterating m fastest, t slowest. ggml ne0 is the
    // fastest. So we need to transpose to layout (n_mels innermost? no —
    // we need ne0=T innermost, meaning along T row-major). Since ggml's
    // memory model is column-major in that sense (ne0 = innermost stride 1),
    // a tensor with ne=[T, n_mels] expects the data to be (n_mels rows × T
    // cols) row-major i.e. m slowest, t fastest. Convert (T, n_mels) →
    // (n_mels, T) i.e. transpose.
    std::vector<float> feats_TC((size_t)n_mels * (size_t)T);
    for (int t = 0; t < T; t++) {
        for (int m = 0; m < n_mels; m++) {
            feats_TC[(size_t)m * (size_t)T + (size_t)t] =
                feats[(size_t)t * (size_t)n_mels + (size_t)m];
        }
    }

    // ---- 2. Build a meta context, declare the input tensor, build graph. ----
    const size_t meta_size = ggml_tensor_overhead() * CAMPPLUS_MAX_NODES +
                             ggml_graph_overhead_custom(CAMPPLUS_MAX_NODES, false);
    std::vector<uint8_t> meta(meta_size);
    ggml_init_params ip{ meta_size, meta.data(), /*no_alloc=*/true };
    ggml_context* ctx0 = ggml_init(ip);
    if (!ctx0) { RS_LOG_ERR("campplus: ggml_init for meta_ctx failed"); return false; }

    ggml_tensor* feat_t = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T, n_mels, 1, 1);
    ggml_set_name(feat_t, "fbank");
    ggml_set_input(feat_t);

    ggml_cgraph* gf = campplus_build_graph(ctx0, feat_t, weights_, hparams_);
    if (!gf) {
        ggml_free(ctx0);
        return false;
    }

    // ---- 3. Allocate via gallocr & compute ----
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        RS_LOG_ERR("campplus: gallocr alloc failed");
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(feat_t, feats_TC.data(), 0, feats_TC.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("campplus: graph compute failed");
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return false;
    }

    // ---- 4. Read output ----
    ggml_tensor* emb_t = ggml_graph_get_tensor(gf, "embedding");
    if (!emb_t) {
        RS_LOG_ERR("campplus: embedding tensor not found in graph");
        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return false;
    }
    cs.embedding.resize((size_t)hparams_.embed_dim);
    ggml_backend_tensor_get(emb_t, cs.embedding.data(), 0,
                            cs.embedding.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx0);
    return true;
}

bool CAMPPlusModel::Encode(const std::vector<float>& input_frames, RSState& state,
                           ggml_backend_sched_t sched) {
    return Embed(input_frames.data(), (int)input_frames.size(), state, sched);
}

bool CAMPPlusModel::Decode(RSState& /*state*/, ggml_backend_sched_t /*sched*/) {
    return true;
}

std::string CAMPPlusModel::GetTranscription(RSState& /*state*/) {
    return {};
}

// ---------------------------------------------------------------------------
// Architecture registry
// ---------------------------------------------------------------------------

static struct CAMPPlusRegistrar {
    CAMPPlusRegistrar() {
        rs_register_model_arch("campplus", []() -> std::shared_ptr<ISpeechModel> {
            return std::make_shared<CAMPPlusModel>();
        });
    }
} s_campplus_registrar;
