#include "cosyvoice3_flow.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// =====================================================================
// File-local helpers — patterns from cosyvoice.cpp/src/cosyvoice-graph.cpp.
// =====================================================================

namespace {

ggml_backend_t pick_flow_backend(ggml_backend_sched_t sched) {
  const int n_backends = ggml_backend_sched_get_n_backends(sched);
  for (int i = 0; i < n_backends; ++i) {
    ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
      return b;
    }
  }
  return ggml_backend_sched_get_backend(sched, n_backends - 1);
}

ggml_tensor *flow_view_root(ggml_tensor *tensor) {
  while (tensor && tensor->view_src) {
    tensor = tensor->view_src;
  }
  return tensor;
}

void assign_flow_persistent_backend(ggml_backend_sched_t sched,
                                    ggml_backend_t backend,
                                    ggml_cgraph *gf,
                                    const std::vector<ggml_tensor *> &tensors) {
  if (!sched || !backend) {
    return;
  }

  for (ggml_tensor *tensor : tensors) {
    if (tensor) {
      ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
    }
  }

  if (!gf) {
    return;
  }
  const int n_nodes = ggml_graph_n_nodes(gf);
  for (int i = 0; i < n_nodes; ++i) {
    ggml_tensor *node = ggml_graph_node(gf, i);
    if (!node) {
      continue;
    }
    if (node->op != GGML_OP_CPY && node->op != GGML_OP_SET &&
        node->op != GGML_OP_SET_ROWS) {
      continue;
    }

    ggml_tensor *view_src = flow_view_root(node->view_src);
    if (!view_src && node->src[1]) {
      view_src = flow_view_root(node->src[1]);
    }
    if (!view_src && node->src[2]) {
      view_src = flow_view_root(node->src[2]);
    }
    if (!view_src) {
      continue;
    }

    for (ggml_tensor *tensor : tensors) {
      if (view_src == tensor) {
        ggml_backend_sched_set_tensor_backend(sched, node, backend);
        break;
      }
    }
  }
}

// concat([a, b], dim=0) for F32 tensors (matches ref1 ggml_concat call).
ggml_tensor *concat_f32(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b,
                        int dim) {
  return ggml_concat(ctx, a, b, dim);
}

// concat([prompt, gen], dim=0) for I32 token tensors. Stock ggml_concat is
// F32/F16 only — view as F32, concat, view back as I32 (zero-copy trick).
ggml_tensor *concat_i32(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b,
                        int dim) {
  ggml_tensor *av = ggml_view_tensor(ctx, a); av->type = GGML_TYPE_F32;
  ggml_tensor *bv = ggml_view_tensor(ctx, b); bv->type = GGML_TYPE_F32;
  ggml_tensor *r  = ggml_concat(ctx, av, bv, dim);
  r = ggml_view_tensor(ctx, r);
  r->type = GGML_TYPE_I32;
  return r;
}

ggml_tensor *unsqueeze1(ggml_context *ctx, ggml_tensor *x) {
  // Insert a new dim-1 axis at position 1 (matches ref1 `unsqueeze(x, 1)`).
  return ggml_view_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2],
                      x->nb[1], x->nb[1], x->nb[2], 0);
}

ggml_tensor *unsqueeze0(ggml_context *ctx, ggml_tensor *x) {
  return ggml_view_4d(ctx, x, 1, x->ne[0], x->ne[1], x->ne[2],
                      x->nb[0], x->nb[1], x->nb[2], 0);
}

ggml_tensor *mish(ggml_context *ctx, ggml_tensor *x) {
  // mish(x) = x * tanh(softplus(x))
  ggml_tensor *s = ggml_softplus(ctx, x);
  s = ggml_tanh(ctx, s);
  return ggml_mul(ctx, x, s);
}

// Linear(weight[in,out], bias[out]): y = W·x + b. Bias is optional.
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                    ggml_tensor *x) {
  ggml_tensor *y = ggml_mul_mat(ctx, w, x);
  if (b) y = ggml_add(ctx, y, b);
  return y;
}

// Standard LayerNorm with affine. Both weight and bias may be null (LN with
// no learned scale/shift — matches ref1 `LayerNorm` when fields are absent).
ggml_tensor *layer_norm(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                        ggml_tensor *x) {
  x = ggml_norm(ctx, x, 1e-6f);
  if (w) {
    x = ggml_mul(ctx, w, x);
    if (b) x = ggml_add(ctx, x, b);
  }
  return x;
}

// Plain Conv1d (group=1) using im2col + mul_mat. Layout: x is (T, C_in, B);
// weight is (K, C_in, C_out). Result is (T_out, C_out, B).
ggml_tensor *conv1d(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                    ggml_tensor *x, int stride, int pad, int dilation) {
  ggml_tensor *im = ggml_im2col(ctx, w, x, stride, 0, pad, 0, dilation, 0,
                                false, GGML_TYPE_F16);
  // im: (K*C_in, T_out, B, 1)
  ggml_tensor *wm = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
  // wm: (K*C_in, C_out, 1, 1).
  //
  // ggml_mul_mat broadcast rule requires `b.ne[2] % a.ne[2] == 0`; when im has
  // batch>1 and wm has batch=1, broadcast wm along its ne[2] to match. ref1
  // does the same (see `Conv1d::build_cgraph` L227-233 in cosyvoice-graph.cpp).
  if (im->ne[2] > 1) {
    wm = ggml_repeat_4d(ctx, wm, wm->ne[0], wm->ne[1], im->ne[2], wm->ne[3]);
  }
  ggml_tensor *y = ggml_mul_mat(ctx, im, wm);
  // y: (T_out, C_out, B, 1)
  if (b) y = ggml_add(ctx, y, unsqueeze0(ctx, b));
  return y;
}

// Grouped conv1d with explicit groups — used by CausalConvPositionEmbedding
// (groups=16). The reference implementation splits along the in-channel axis,
// runs g independent im2col+matmul calls, then concats. For the conv-pos-embed
// in DiT the per-group channel count is 64/16 = 4 (or 1024/16 = 64 for the
// 1024-channel side); both are small.
ggml_tensor *conv1d_grouped(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                            ggml_tensor *x, int stride, int pad, int dilation,
                            int groups) {
  GGML_ASSERT(groups >= 1);
  if (groups == 1) return conv1d(ctx, w, b, x, stride, pad, dilation);

  const int64_t c_in_per_g  = x->ne[1] / groups;
  const int64_t c_out_per_g = w->ne[2] / groups;
  const int64_t kernel      = w->ne[0];
  GGML_ASSERT(x->ne[1] % groups == 0);
  GGML_ASSERT(w->ne[2] % groups == 0);
  GGML_ASSERT(w->ne[1] == c_in_per_g);  // weight is [K, C_in/g, C_out]

  std::vector<ggml_tensor *> per_group(groups);
  for (int g = 0; g < groups; ++g) {
    ggml_tensor *xg = ggml_view_3d(
        ctx, x, x->ne[0], c_in_per_g, x->ne[2],
        x->nb[1], x->nb[2], g * c_in_per_g * x->nb[1]);
    xg = ggml_cont(ctx, xg);
    ggml_tensor *wg = ggml_view_3d(
        ctx, w, kernel, c_in_per_g, c_out_per_g,
        w->nb[1], w->nb[2], g * c_out_per_g * w->nb[2]);
    wg = ggml_cont(ctx, wg);
    ggml_tensor *im = ggml_im2col(ctx, wg, xg, stride, 0, pad, 0, dilation, 0,
                                  false, GGML_TYPE_F16);
    ggml_tensor *wm = ggml_reshape_2d(
        ctx, wg, wg->ne[0] * wg->ne[1], wg->ne[2]);
    if (im->ne[2] > 1) {
      wm = ggml_repeat_4d(ctx, wm, wm->ne[0], wm->ne[1], im->ne[2], wm->ne[3]);
    }
    ggml_tensor *y = ggml_mul_mat(ctx, im, wm);
    per_group[g] = y;
  }
  ggml_tensor *out = per_group[0];
  for (int g = 1; g < groups; ++g)
    out = ggml_concat(ctx, out, per_group[g], 1);
  if (b) out = ggml_add(ctx, out, unsqueeze0(ctx, b));
  return out;
}

} // namespace

// =====================================================================
// CosyVoice3FlowModel::Load and helpers
// =====================================================================

CosyVoice3FlowModel::~CosyVoice3FlowModel() {
  if (time_emb_buf_) ggml_backend_buffer_free(time_emb_buf_);
  if (cctx_owned_) ggml_free(cctx_owned_);
}

bool CosyVoice3FlowModel::Load(gguf_context *ctx_gguf, ggml_context *gguf_data,
                               ggml_backend_t backend) {
  if (!ctx_gguf || !gguf_data || !backend) {
    RS_LOG_ERR("Flow: invalid arguments to Load");
    return false;
  }

  if (!LoadHparams(ctx_gguf)) return false;

  // Collect all tensors into a name → ggml_tensor* map for cheap lookup.
  std::map<std::string, ggml_tensor *> tensors;
  const int n = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }
  if (!LoadTensors(tensors)) return false;
  if (!AllocSinusoidalEmbed(backend)) return false;

  // Cosine schedule scaled to euler_steps_: t[i] = 1 - cos((1/N) * pi/2 * i),
  // so t[0]=0, t[N]=1 regardless of N. Matches the upstream ref's t_span for
  // N=10 (the original constant 0.1 == 1/10).
  t_span_.resize((size_t)euler_steps_ + 1);
  const float c = 1.0f / (float)euler_steps_;
  for (int i = 0; i <= euler_steps_; ++i)
    t_span_[i] = 1.0f - std::cos(c * 0.5f * M_PI * (float)i);

  RS_LOG_INFO("Flow loaded: depth=%d dim=%d heads=%d head_dim=%d ff=%d "
              "mel=%d cfg=%.2f euler_steps=%d",
              depth_, dim_, n_heads_, head_dim_, ff_dim_, mel_dim_,
              (double)cfg_rate_, euler_steps_);
  return true;
}

bool CosyVoice3FlowModel::LoadHparams(gguf_context *ctx_gguf) {
  auto u32 = [&](const char *k, int dflt) {
    int id = gguf_find_key(ctx_gguf, k);
    return id >= 0 ? (int)gguf_get_val_u32(ctx_gguf, id) : dflt;
  };
  auto f32 = [&](const char *k, float dflt) {
    int id = gguf_find_key(ctx_gguf, k);
    return id >= 0 ? gguf_get_val_f32(ctx_gguf, id) : dflt;
  };
  depth_       = u32("cosyvoice3.flow.depth",       22);
  dim_         = u32("cosyvoice3.flow.dim",         1024);
  n_heads_     = u32("cosyvoice3.flow.n_heads",     16);
  head_dim_    = u32("cosyvoice3.flow.head_dim",    64);
  ff_dim_      = u32("cosyvoice3.flow.ff_dim",      2048);
  mel_dim_     = u32("cosyvoice3.flow.mel_dim",     80);
  spk_dim_in_  = u32("cosyvoice3.flow.spk_emb_dim", 192);
  pre_lookahead_len_ = u32("cosyvoice3.flow.pre_lookahead_len", 3);
  token_mel_ratio_   = u32("cosyvoice3.flow.token_mel_ratio",   2);
  cfg_rate_    = f32("cosyvoice3.flow.inference_cfg_rate", 0.7f);
  return true;
}

bool CosyVoice3FlowModel::LoadTensors(
    const std::map<std::string, ggml_tensor *> &tensors) {
  auto get = [&](const char *name, ggml_tensor **dst, bool required = true) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      if (required) RS_LOG_ERR("Flow: missing tensor %s", name);
      return !required;
    }
    *dst = it->second;
    return true;
  };

  bool ok = true;
  ok &= get("flow.input_embedding.weight",            &input_embedding_);
  ok &= get("flow.spk_embed_affine_layer.weight",     &spk_affine_w_);
  ok &= get("flow.spk_embed_affine_layer.bias",       &spk_affine_b_);
  ok &= get("flow.pre_lookahead_layer.conv1.weight",  &pre_look_c1_w_);
  ok &= get("flow.pre_lookahead_layer.conv1.bias",    &pre_look_c1_b_);
  ok &= get("flow.pre_lookahead_layer.conv2.weight",  &pre_look_c2_w_);
  ok &= get("flow.pre_lookahead_layer.conv2.bias",    &pre_look_c2_b_);

  ok &= get("flow.decoder.estimator.time_embed.time_mlp.0.weight",
            &time_mlp0_w_);
  ok &= get("flow.decoder.estimator.time_embed.time_mlp.0.bias",
            &time_mlp0_b_);
  ok &= get("flow.decoder.estimator.time_embed.time_mlp.2.weight",
            &time_mlp2_w_);
  ok &= get("flow.decoder.estimator.time_embed.time_mlp.2.bias",
            &time_mlp2_b_);

  ok &= get("flow.decoder.estimator.input_embed.proj.weight",
            &input_proj_w_);
  ok &= get("flow.decoder.estimator.input_embed.proj.bias",
            &input_proj_b_);
  ok &= get("flow.decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight",
            &cpe_c1_w_);
  ok &= get("flow.decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias",
            &cpe_c1_b_);
  ok &= get("flow.decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight",
            &cpe_c2_w_);
  ok &= get("flow.decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias",
            &cpe_c2_b_);

  ok &= get("flow.decoder.estimator.norm_out.linear.weight",
            &norm_out_lin_w_);
  ok &= get("flow.decoder.estimator.norm_out.linear.bias",
            &norm_out_lin_b_);
  ok &= get("flow.decoder.estimator.proj_out.weight", &proj_out_w_);
  ok &= get("flow.decoder.estimator.proj_out.bias",   &proj_out_b_);

  blocks_.resize(depth_);
  char buf[256];
  for (int i = 0; i < depth_; ++i) {
    auto &bl = blocks_[i];
    auto fmt = [&](const char *f) {
      snprintf(buf, sizeof(buf), f, i); return buf;
    };
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn_norm.linear.weight"), &bl.adaLN_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn_norm.linear.bias"),   &bl.adaLN_b);
    // attn_norm.norm (LayerNorm) has optional affine — make them optional.
    get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn_norm.norm.weight"),
        &bl.norm_w, /*required=*/false);
    get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn_norm.norm.bias"),
        &bl.norm_b, /*required=*/false);

    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_q.weight"),   &bl.to_q_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_q.bias"),     &bl.to_q_b);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_k.weight"),   &bl.to_k_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_k.bias"),     &bl.to_k_b);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_v.weight"),   &bl.to_v_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_v.bias"),     &bl.to_v_b);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_out.0.weight"), &bl.to_out_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.attn.to_out.0.bias"),   &bl.to_out_b);

    get(fmt("flow.decoder.estimator.transformer_blocks.%d.ff_norm.weight"),
        &bl.ff_norm_w, /*required=*/false);
    get(fmt("flow.decoder.estimator.transformer_blocks.%d.ff_norm.bias"),
        &bl.ff_norm_b, /*required=*/false);

    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.ff.ff.0.0.weight"), &bl.ff0_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.ff.ff.0.0.bias"),   &bl.ff0_b);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.ff.ff.2.weight"),   &bl.ff2_w);
    ok &= get(fmt("flow.decoder.estimator.transformer_blocks.%d.ff.ff.2.bias"),     &bl.ff2_b);
  }
  return ok;
}

bool CosyVoice3FlowModel::AllocSinusoidalEmbed(ggml_backend_t backend) {
  // Build the FunAudio-style sinusoidal-position-embedding table on the CPU
  // and upload to the backend. half_dim = dim/2 with dim=256, so 128 entries.
  constexpr int dim = 256;
  constexpr int half_dim = dim / 2;
  std::array<float, half_dim> buf{};
  const float emb = std::log(10000.f) / (half_dim - 1);
  for (int i = 0; i < half_dim; ++i)
    buf[i] = std::exp((float)i * -emb) * 1000.f;

  // Allocate a tiny ggml_context just to hold the tensor descriptor; the data
  // lives on the backend buffer.
  ggml_init_params ip = { ggml_tensor_overhead() + 256, nullptr, true };
  ggml_context *cctx = ggml_init(ip);
  time_emb_table_ = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, half_dim);
  ggml_set_name(time_emb_table_, "flow.sinus_emb_table");
  time_emb_buf_ = ggml_backend_alloc_ctx_tensors(cctx, backend);
  if (!time_emb_buf_) {
    ggml_free(cctx);
    RS_LOG_ERR("Flow: failed to allocate sinusoidal-embed table");
    return false;
  }
  ggml_backend_tensor_set(time_emb_table_, buf.data(), 0,
                          half_dim * sizeof(float));
  // NB: we leak `cctx` intentionally — it's a thin no_alloc context whose
  // sole tensor descriptor must outlive Load(). Freed in destructor below.
  // Actually: ggml_backend_alloc_ctx_tensors holds onto descriptors; we keep
  // the context for tensor metadata until the model is destroyed.
  // Store as opaque void* so we don't widen the header.
  cctx_owned_ = cctx;
  return true;
}

void CosyVoice3FlowModel::SetEulerSteps(int n) {
  if (n >= 2 && n <= 64) {
    euler_steps_ = n;
    t_span_.resize((size_t)euler_steps_ + 1);
    const float c = 1.0f / (float)euler_steps_;
    for (int i = 0; i <= euler_steps_; ++i)
      t_span_[i] = 1.0f - std::cos(c * 0.5f * M_PI * (float)i);
  }
}

// =====================================================================
// Build the encode graph: tokens → (mu, spks, cond).
//
// Mirrors cosyvoice-graph.cpp:546 CausalMaskedDiffWithDiT::build_cgraph_encode.
// =====================================================================

CosyVoice3FlowModel::EncodeOutputs CosyVoice3FlowModel::BuildEncodeGraph(
    ggml_context *ctx,
    ggml_tensor *prompt_token,
    ggml_tensor *gen_token,
    ggml_tensor *prompt_feat,
    ggml_tensor *embedding) const {
  // ----- Speaker embedding: l2-norm + Linear(192→80) ----------------------
  ggml_tensor *spks = ggml_l2_norm(ctx, embedding, 1e-6f);
  spks = linear(ctx, spk_affine_w_, spk_affine_b_, spks);
  // Result is [mel_dim,]. Broadcast happens later inside InputEmbedding.

  // ----- Token embedding lookup ------------------------------------------
  // The caller concatenates prompt+gen tokens on the host and passes them as
  // `gen_token` (length = T_pt + T_gen). The `prompt_token` arg is ignored
  // here — the legacy in-graph concat path was a stale plumbing artefact and
  // confused the scheduler when both halves needed different backends.
  (void)prompt_token;
  // ggml_get_rows: input_embedding is [mel_dim, codebook] (row-major in ggml,
  // so ne[0]=mel_dim, ne[1]=codebook). Rows are indexed along ne[1].
  ggml_tensor *embedded = ggml_get_rows(ctx, input_embedding_, gen_token);
  // embedded is [mel_dim, T_tok] in ggml memory order.

  // ----- PreLookaheadLayer (residual) ------------------------------------
  // PyTorch PreLookaheadLayer: in [B, T, C], permute to [B, C, T], pad right
  // by pre_lookahead_len=3, conv1d k=4, leaky_relu, pad right by k-1=2,
  // conv1d k=3, permute back, add residual.
  //
  // In ggml: `embedded` is [mel_dim=C, T_tok=T, B=1]; ref1 permutes (1,0,...)
  // to get [T, C, B] before padding along dim 0 (T), runs conv on T-axis,
  // then permutes back. The im2col convention in our `conv1d` helper treats
  // ne[0] as T and ne[1] as C, so we follow ref1 exactly.
  {
    ggml_tensor *h = ggml_permute(ctx, embedded, 1, 0, 2, 3);
    h = ggml_cont(ctx, h);
    // pad right by pre_lookahead_len along T (dim 0).
    h = ggml_pad_ext(ctx, h, 0, pre_lookahead_len_, 0, 0, 0, 0, 0, 0);
    h = conv1d(ctx, pre_look_c1_w_, pre_look_c1_b_, h, 1, 0, 1);
    h = ggml_leaky_relu(ctx, h, 0.01f, /*inplace=*/true);
    // pad right by (k2-1).
    const int k2 = (int)pre_look_c2_w_->ne[0];
    h = ggml_pad_ext(ctx, h, 0, k2 - 1, 0, 0, 0, 0, 0, 0);
    h = conv1d(ctx, pre_look_c2_w_, pre_look_c2_b_, h, 1, 0, 1);
    h = ggml_permute(ctx, h, 1, 0, 2, 3);
    h = ggml_cont(ctx, h);
    embedded = ggml_add(ctx, h, embedded);
  }

  // ----- Repeat-interleave by token_mel_ratio ----------------------------
  // PyTorch: x.repeat_interleave(token_mel_ratio, dim=1) over [B, T, C].
  // Ref1 expresses this as unsqueeze(1) then repeat_4d then reshape.
  // embedded is [mel_dim, T_tok, B=1]. We want [mel_dim, T_tok*ratio, B].
  {
    // Insert a length-`ratio` axis between dim 1 and dim 2:
    //   [mel_dim, T_tok, B] → view as [mel_dim, 1, T_tok, B]
    //   → repeat_4d to [mel_dim, ratio, T_tok, B]
    //   → reshape to [mel_dim, T_tok*ratio, B]
    ggml_tensor *h = ggml_view_4d(ctx, embedded,
                                  embedded->ne[0], 1,
                                  embedded->ne[1], embedded->ne[2],
                                  embedded->nb[0], embedded->nb[1],
                                  embedded->nb[2], 0);
    h = ggml_repeat_4d(ctx, h, h->ne[0], (int64_t)token_mel_ratio_,
                       h->ne[2], h->ne[3]);
    h = ggml_cont(ctx, h);
    embedded = ggml_reshape_3d(ctx, h,
                               h->ne[0],
                               h->ne[1] * h->ne[2],
                               h->ne[3]);
  }

  // ----- Conditional context = pad(prompt_feat, 0, T_gen_mel) ------------
  // prompt_feat is [mel_dim, T_pt_mel, 1]; pad along dim 1 (T) by T_gen_mel.
  const int64_t mel_len1 = prompt_feat ? prompt_feat->ne[1] : 0;
  const int64_t mel_len2 = embedded->ne[1] - mel_len1;
  ggml_tensor *cond;
  if (prompt_feat && mel_len2 > 0) {
    // pad_ext: pad right of dim 1 by mel_len2 zeros.
    cond = ggml_pad_ext(ctx, prompt_feat, 0, 0, 0, (int)mel_len2, 0, 0, 0, 0);
  } else if (prompt_feat) {
    cond = prompt_feat;
  } else {
    // No prompt → cond is all-zero, same shape as mu.
    cond = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                              embedded->ne[0], embedded->ne[1], 1);
    cond = ggml_scale(ctx, cond, 0.f);
  }

  return EncodeOutputs{embedded, spks, cond, mel_len1};
}

// =====================================================================
// One DiT block. Mirrors cosyvoice-graph.cpp:399 DiTBlock::build_cgraph.
//
//   x:            [dim, T, batch=2]
//   time_emb:     [dim, batch=2]      already silu-projected by TimestepEmbed
//   position_ids: [T_full, batch=2]   (we slice it for Q when cut_len > 0)
//   cut_len:      0 for all but the last block of the last Euler step
// =====================================================================

ggml_tensor *CosyVoice3FlowModel::BuildDiTBlock(ggml_context *ctx,
                                                const DiTBlock &bl,
                                                ggml_tensor *x,
                                                ggml_tensor *time_emb,
                                                ggml_tensor *position_ids,
                                                int64_t cut_len) const {
  const int64_t T_full = x->ne[1];
  const int64_t T_q    = T_full - cut_len;
  const int64_t batch  = x->ne[2];

  // ----- AdaLayerNormZero ------------------------------------------------
  //   emb = silu(time_emb); emb = linear(emb)   shape [6*dim, batch]
  //   split into 6 chunks along dim 0:
  //     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
  //   x_n = norm(x); x = x_n * (1+scale_msa) + shift_msa
  ggml_tensor *e = ggml_silu(ctx, time_emb);
  e = linear(ctx, bl.adaLN_w, bl.adaLN_b, e);  // [6*dim, batch]
  // Six chunks of size `dim` along dim 0.
  auto chunk = [&](int idx) {
    return ggml_view_2d(ctx, e, dim_, e->ne[1],
                        e->nb[1], (size_t)idx * dim_ * e->nb[0]);
  };
  ggml_tensor *shift_msa = chunk(0);
  ggml_tensor *scale_msa = chunk(1);
  ggml_tensor *gate_msa  = chunk(2);
  ggml_tensor *shift_mlp = chunk(3);
  ggml_tensor *scale_mlp = chunk(4);
  ggml_tensor *gate_mlp  = chunk(5);

  scale_msa = ggml_scale_bias(ctx, ggml_cont(ctx, scale_msa), 1.f, 1.f);
  scale_msa = unsqueeze1(ctx, scale_msa);   // [dim, 1, batch]
  shift_msa = unsqueeze1(ctx, ggml_cont(ctx, shift_msa));

  ggml_tensor *xn = layer_norm(ctx, bl.norm_w, bl.norm_b, x);
  xn = ggml_mul(ctx, xn, scale_msa);
  xn = ggml_add(ctx, xn, shift_msa);

  // ----- Attention -------------------------------------------------------
  // K and V always cover the full sequence; Q is sliced by cut_len.
  ggml_tensor *k = linear(ctx, bl.to_k_w, bl.to_k_b, xn);  // [dim, T_full, B]
  ggml_tensor *v = linear(ctx, bl.to_v_w, bl.to_v_b, xn);

  ggml_tensor *xn_q = xn;
  ggml_tensor *pos_q = position_ids;
  if (cut_len > 0) {
    xn_q = ggml_view_3d(ctx, xn, xn->ne[0], T_q, batch,
                        xn->nb[1], xn->nb[2],
                        (size_t)cut_len * xn->nb[1]);
    pos_q = ggml_view_2d(ctx, position_ids, T_q, batch,
                         position_ids->nb[1],
                         (size_t)cut_len * position_ids->nb[0]);
    pos_q = ggml_cont(ctx, pos_q);
  }
  ggml_tensor *q = linear(ctx, bl.to_q_w, bl.to_q_b, xn_q);

  // Apply RoPE before reshaping (matches ref1). RoPE op wants 3-D layout
  // [head_dim, 1, seq*B]; positions flattened to length seq*B.
  q = ggml_reshape_3d(ctx, q, q->ne[0], 1, T_q * batch);
  k = ggml_reshape_3d(ctx, k, k->ne[0], 1, T_full * batch);
  ggml_tensor *pos_q_flat = ggml_reshape_1d(ctx, pos_q, T_q * batch);
  ggml_tensor *pos_k_flat = ggml_reshape_1d(ctx, position_ids,
                                            T_full * batch);
  q = ggml_rope(ctx, q, pos_q_flat, head_dim_, GGML_ROPE_TYPE_NORMAL);
  k = ggml_rope(ctx, k, pos_k_flat, head_dim_, GGML_ROPE_TYPE_NORMAL);

  // Reshape back to [head_dim, n_heads, T, B] and permute to [head_dim, T,
  // n_heads, B] for attention.
  q = ggml_reshape_4d(ctx, q, head_dim_, n_heads_, T_q,    batch);
  k = ggml_reshape_4d(ctx, k, head_dim_, n_heads_, T_full, batch);
  v = ggml_reshape_4d(ctx, v, head_dim_, n_heads_, T_full, batch);
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
  v = ggml_permute(ctx, v, 0, 2, 1, 3);

  // QK^T softmax V — no mask (bidirectional attention).
  ggml_tensor *attn_scores = ggml_mul_mat(ctx, k, q);
  attn_scores = ggml_scale(ctx, attn_scores,
                           1.f / std::sqrt((float)head_dim_));
  ggml_tensor *attn_weights = ggml_soft_max(ctx, attn_scores);
  v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
  ggml_tensor *attn_out = ggml_mul_mat(ctx, v, attn_weights);
  attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
  attn_out = ggml_reshape_3d(ctx, attn_out,
                             attn_out->ne[0] * attn_out->ne[1], T_q, batch);
  attn_out = linear(ctx, bl.to_out_w, bl.to_out_b, attn_out);

  // gate_msa is [dim, batch]; broadcast across T.
  ggml_tensor *gate_msa_u = unsqueeze1(ctx, ggml_cont(ctx, gate_msa));
  attn_out = ggml_mul(ctx, attn_out, gate_msa_u);

  // Residual — if Q was sliced, slice the residual to match.
  ggml_tensor *x_res = x;
  if (cut_len > 0) {
    x_res = ggml_view_3d(ctx, x, x->ne[0], T_q, batch,
                         x->nb[1], x->nb[2],
                         (size_t)cut_len * x->nb[1]);
  }
  x = ggml_add(ctx, x_res, attn_out);

  // ----- FF: LayerNorm → (1+scale_mlp)*norm + shift_mlp → FF → gate_mlp -
  ggml_tensor *xn2 = layer_norm(ctx, bl.ff_norm_w, bl.ff_norm_b, x);
  scale_mlp = ggml_scale_bias(ctx, ggml_cont(ctx, scale_mlp), 1.f, 1.f);
  ggml_tensor *scale_mlp_u = unsqueeze1(ctx, scale_mlp);
  ggml_tensor *shift_mlp_u = unsqueeze1(ctx, ggml_cont(ctx, shift_mlp));
  xn2 = ggml_mul(ctx, xn2, scale_mlp_u);
  xn2 = ggml_add(ctx, xn2, shift_mlp_u);

  ggml_tensor *ff = linear(ctx, bl.ff0_w, bl.ff0_b, xn2);
  ff = ggml_gelu_erf(ctx, ff);
  ff = linear(ctx, bl.ff2_w, bl.ff2_b, ff);

  ggml_tensor *gate_mlp_u = unsqueeze1(ctx, ggml_cont(ctx, gate_mlp));
  ff = ggml_mul(ctx, ff, gate_mlp_u);
  x = ggml_add(ctx, x, ff);
  return x;
}

// =====================================================================
// Build a single CFM/Euler step graph.
//
// Inputs:
//   x_init      [mel_dim, T_mel, 1]  — current state (read on this step)
//   mu/cond     [mel_dim, T_mel, 1]
//   spks        [mel_dim, 1, 1]      — broadcasted inside DiT
//   cut_len     T_pt_mel (last step only; else 0)
//   t,dt        scalars (filled into t_in tensor)
//
// Produces x_next = x_init + dt · CFG(dphi).
// =====================================================================

ggml_tensor *CosyVoice3FlowModel::BuildCFMStepGraph(
    ggml_context *ctx,
    ggml_tensor *x_init,
    ggml_tensor *mu,
    ggml_tensor *spks,
    ggml_tensor *cond,
    int64_t cut_len,
    float t_value,
    float dt_value,
    ggml_tensor *&position_ids_out) const {
  const bool last_step = cut_len > 0;

  // CFG batch=2: pad mu/spks/cond with a zeroed second-batch slot.
  ggml_tensor *mu_in   = ggml_pad(ctx, mu,   0, 0, 1, 0);
  ggml_tensor *spks_in = ggml_pad(ctx, spks, 0, 0, 1, 0);
  ggml_tensor *cond_in = ggml_pad(ctx, cond, 0, 0, 1, 0);

  // Replicate x for batch=2.
  ggml_tensor *x_in = ggml_repeat_4d(ctx, x_init,
                                     x_init->ne[0], x_init->ne[1], 2,
                                     x_init->ne[3] == 0 ? 1 : x_init->ne[3]);

  // ----- Build t embedding (sinusoidal + 2-layer MLP) --------------------
  // t_in is a [2] tensor filled with the scalar `t_value` (same for both
  // CFG batch slots). We use ggml_fill so the value is baked into the graph
  // — no per-step input plumbing.
  ggml_tensor *t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
  t_in = ggml_fill_inplace(ctx, t_in, t_value);

  // Sinusoidal embed: t · emb_table → sin / cos, concat on dim 0.
  // t_in: [2]. Expand to [128, 2] = t * emb_table_broadcast.
  ggml_tensor *t_e = unsqueeze0(ctx, t_in);  // [1, 2, 1, 1]
  t_e = ggml_repeat_4d(ctx, t_e, time_emb_table_->ne[0],
                       t_e->ne[1], t_e->ne[2], t_e->ne[3]);
  ggml_tensor *m = ggml_mul(ctx, t_e, time_emb_table_);
  ggml_tensor *t_sin = ggml_sin(ctx, m);
  ggml_tensor *t_cos = ggml_cos(ctx, m);
  ggml_tensor *t_emb = ggml_concat(ctx, t_sin, t_cos, 0);  // [256, 2]
  t_emb = linear(ctx, time_mlp0_w_, time_mlp0_b_, t_emb);
  t_emb = ggml_silu(ctx, t_emb);
  t_emb = linear(ctx, time_mlp2_w_, time_mlp2_b_, t_emb);   // [dim=1024, 2]

  // ----- Build the DiT estimator graph ----------------------------------
  // Input layout: x_in / mu_in / cond_in are all (mel_dim, T_mel, B=2, 1) —
  // channels in ne[0], time in ne[1], CFG batch in ne[2]. The 4-way concat
  // happens along the channel axis (ne[0]), yielding (4·mel_dim=320, T_mel,
  // B, 1) ready for the input_embed proj.
  ggml_tensor *x = x_in;
  ggml_tensor *spks_bcast = ggml_repeat(ctx, spks_in, x);
  ggml_tensor *concat = ggml_concat(ctx, x,     cond_in,    0);  // 80+80
  concat = ggml_concat(ctx, concat, mu_in,      0);              // +80
  concat = ggml_concat(ctx, concat, spks_bcast, 0);              // +80 = 320
  ggml_tensor *h = linear(ctx, input_proj_w_, input_proj_b_, concat);

  // ConvPosEmbed: permute(1,0,...), pad-right (k-1), groupedConv g=16, mish,
  // pad-right (k-1), groupedConv g=16, mish, permute back, add residual.
  {
    ggml_tensor *p = ggml_permute(ctx, h, 1, 0, 2, 3);
    p = ggml_cont(ctx, p);
    const int k1 = (int)cpe_c1_w_->ne[0];
    p = ggml_pad_ext(ctx, p, k1 - 1, 0, 0, 0, 0, 0, 0, 0);
    p = conv1d_grouped(ctx, cpe_c1_w_, cpe_c1_b_, p, 1, 0, 1, 16);
    p = mish(ctx, p);
    const int k2 = (int)cpe_c2_w_->ne[0];
    p = ggml_pad_ext(ctx, p, k2 - 1, 0, 0, 0, 0, 0, 0, 0);
    p = conv1d_grouped(ctx, cpe_c2_w_, cpe_c2_b_, p, 1, 0, 1, 16);
    p = mish(ctx, p);
    p = ggml_permute(ctx, p, 1, 0, 2, 3);
    p = ggml_cont(ctx, p);
    h = ggml_add(ctx, h, p);
  }

  // Position ids: a [T_full, 2] input tensor. Ref1 plumbs this from the host
  // for every step; in our v1 we build it inline as 0..T-1 broadcast across
  // batch using a `fill`-like recipe: create as ggml_set_input then defer to
  // RunFlow to populate.
  ggml_tensor *position_ids =
      ggml_new_tensor_2d(ctx, GGML_TYPE_I32, h->ne[1], h->ne[2]);
  ggml_set_input(position_ids);
  ggml_set_name(position_ids, "flow.position_ids");
  position_ids_out = position_ids;
  ggml_tensor *pos_dup = ggml_dup(ctx, position_ids);

  // ----- 22 DiT blocks --------------------------------------------------
  for (int i = 0; i < depth_ - 1; ++i) {
    h = BuildDiTBlock(ctx, blocks_[i], h, t_emb, pos_dup, 0);
  }
  // Final block applies `cut_len` only on the last Euler step.
  h = BuildDiTBlock(ctx, blocks_[depth_ - 1], h, t_emb, pos_dup,
                    last_step ? cut_len : 0);

  // ----- Final norm + projection ----------------------------------------
  {
    ggml_tensor *e = ggml_silu(ctx, t_emb);
    e = linear(ctx, norm_out_lin_w_, norm_out_lin_b_, e);   // [2*dim, 2]
    ggml_tensor *scale = ggml_view_2d(
        ctx, e, dim_, e->ne[1], e->nb[1], 0);
    ggml_tensor *shift = ggml_view_2d(
        ctx, e, dim_, e->ne[1], e->nb[1], (size_t)dim_ * e->nb[0]);
    scale = ggml_scale_bias(ctx, ggml_cont(ctx, scale), 1.f, 1.f);
    ggml_tensor *xn = layer_norm(ctx, nullptr, nullptr, h);
    xn = ggml_mul(ctx, xn, unsqueeze1(ctx, scale));
    xn = ggml_add(ctx, xn, unsqueeze1(ctx, ggml_cont(ctx, shift)));
    h = xn;
  }
  ggml_tensor *out = linear(ctx, proj_out_w_, proj_out_b_, h);
  // out: [mel_dim, T_q, 2]  (T_q = T - cut_len on last step).

  // ----- Split CFG batch and apply guidance -----------------------------
  // out is (mel_dim, T_q, B=2, 1). Split along batch (ne[2]) into cond/null.
  const int64_t T_q = out->ne[1];
  ggml_tensor *dphi_cond = ggml_view_3d(
      ctx, out, out->ne[0], T_q, 1, out->nb[1], out->nb[2], 0);
  ggml_tensor *dphi_null = ggml_view_3d(
      ctx, out, out->ne[0], T_q, 1, out->nb[1], out->nb[2], out->nb[2]);
  dphi_cond = ggml_cont(ctx, dphi_cond);
  dphi_null = ggml_cont(ctx, dphi_null);

  // dphi = (1 + cfg)·dphi_cond − cfg·dphi_null
  dphi_cond = ggml_scale(ctx, dphi_cond, 1.f + cfg_rate_);
  dphi_null = ggml_scale(ctx, dphi_null, cfg_rate_);
  ggml_tensor *dphi = ggml_sub(ctx, dphi_cond, dphi_null);

  // Slice x_init if we trimmed (last step).
  ggml_tensor *x_cur = x_init;
  if (last_step) {
    x_cur = ggml_view_3d(ctx, x_init, x_init->ne[0], T_q, x_init->ne[2],
                         x_init->nb[1], x_init->nb[2],
                         (size_t)cut_len * x_init->nb[1]);
    x_cur = ggml_cont(ctx, x_cur);
  }

  ggml_tensor *x_next = ggml_add(ctx, x_cur,
                                 ggml_scale(ctx, dphi, dt_value));
  return x_next;
}

// =====================================================================
// RunFlow — orchestrates encode + N Euler steps + final trim.
//
// Strategy: for correctness first, we rebuild the cgraph for every Euler
// step. The reference caches the middle-step graph and patches scalars in
// place — we'll add that optimization in a follow-up once parity is proven.
// =====================================================================

bool CosyVoice3FlowModel::RunFlow(CosyVoice3State &state,
                                  ggml_backend_sched_t sched) {
  return RunFlowStreaming(state, sched,
                          (int)state.speech_token_ids.size(),
                          /*finalize=*/true);
}

bool CosyVoice3FlowModel::RunFlowStreaming(CosyVoice3State &state,
                                           ggml_backend_sched_t sched,
                                           int n_tokens_in,
                                           bool finalize,
                                           int n_steps_override) {
  // Offline-path skip: `state.flow_done` is only flipped by the finalize=true
  // exit, so this short-circuit never triggers during streaming.
  if (finalize && state.flow_done) return true;
  if (n_tokens_in <= 0 ||
      n_tokens_in > (int)state.speech_token_ids.size()) {
    RS_LOG_ERR("Flow: invalid n_tokens_in=%d (have %zu)",
               n_tokens_in, state.speech_token_ids.size());
    return false;
  }
  if (state.embedding.size() != (size_t)spk_dim_in_) {
    RS_LOG_ERR("Flow: embedding size %zu != %d", state.embedding.size(),
               spk_dim_in_);
    return false;
  }

  // Resolve Euler step count + cosine schedule. The default path reuses the
  // pre-baked `t_span_`; a positive override rebuilds the schedule locally so
  // streaming chunks can run fewer steps without disturbing offline state.
  const int local_steps = (n_steps_override > 0) ? n_steps_override
                                                 : euler_steps_;
  std::vector<float> local_t_span;
  const std::vector<float> *t_span_ptr = &t_span_;
  if (local_steps != euler_steps_) {
    local_t_span.resize((size_t)local_steps + 1);
    const float c = 1.0f / (float)local_steps;
    for (int i = 0; i <= local_steps; ++i)
      local_t_span[i] = 1.0f - std::cos(c * 0.5f * (float)M_PI * (float)i);
    t_span_ptr = &local_t_span;
  }
  const std::vector<float> &t_span_use = *t_span_ptr;

  const int T_gen     = n_tokens_in;
  const int T_pt      = (int)state.prompt_token.size();
  const int T_pt_mel  = (int)(state.prompt_feat.size() / mel_dim_);
  const int T_gen_mel = T_gen * token_mel_ratio_;
  const int T_mel     = T_pt_mel + T_gen_mel;
  GGML_ASSERT(T_pt_mel * mel_dim_ == (int)state.prompt_feat.size());

  // -----------------------------------------------------------------------
  // Allocate a single ggml_context big enough for both encode + 10 step graphs.
  // Each Euler step builds its own (small) cgraph against this context so we
  // pay graph-build allocation only once per step, not per node.
  // -----------------------------------------------------------------------
  // Budget: per step ~ (DiT_blocks=22) × O(20 nodes each) = 440 nodes; plus
  // input embed (~30), final norm (~10). Round generously to 1024 nodes per
  // step; total tensors ~ 4×nodes. We rebuild per-step in a fresh ctx so the
  // budget per call is comfortable.
  constexpr int FLOW_MAX_NODES = 2048;
  constexpr size_t FLOW_CTX_BYTES =
      FLOW_MAX_NODES * (sizeof(ggml_tensor) + 256);

  ggml_backend_t flow_backend = pick_flow_backend(sched);
  ggml_backend_buffer_t persistent_buf = nullptr;
  ggml_context *ctx_persist = nullptr;
  ggml_tensor *mu_dev = nullptr;
  ggml_tensor *cond_dev = nullptr;
  ggml_tensor *spks_dev = nullptr;
  ggml_tensor *x_dev[2] = {nullptr, nullptr};
  auto free_persistent = [&]() {
    if (persistent_buf) {
      ggml_backend_buffer_free(persistent_buf);
      persistent_buf = nullptr;
    }
    if (ctx_persist) {
      ggml_free(ctx_persist);
      ctx_persist = nullptr;
    }
  };

  {
    ggml_init_params ip = {
        (size_t)6 * ggml_tensor_overhead() + (1 << 16), nullptr, true};
    ctx_persist = ggml_init(ip);
    mu_dev = ggml_new_tensor_3d(ctx_persist, GGML_TYPE_F32,
                                mel_dim_, T_mel, 1);
    cond_dev = ggml_new_tensor_3d(ctx_persist, GGML_TYPE_F32,
                                  mel_dim_, T_mel, 1);
    spks_dev = ggml_new_tensor_3d(ctx_persist, GGML_TYPE_F32,
                                  mel_dim_, 1, 1);
    x_dev[0] = ggml_new_tensor_3d(ctx_persist, GGML_TYPE_F32,
                                  mel_dim_, T_mel, 1);
    x_dev[1] = ggml_new_tensor_3d(ctx_persist, GGML_TYPE_F32,
                                  mel_dim_, T_mel, 1);
    ggml_set_name(mu_dev, "flow.mu.persistent");
    ggml_set_name(cond_dev, "flow.cond.persistent");
    ggml_set_name(spks_dev, "flow.spks.persistent");
    ggml_set_name(x_dev[0], "flow.x0.persistent");
    ggml_set_name(x_dev[1], "flow.x1.persistent");

    persistent_buf = ggml_backend_alloc_ctx_tensors(ctx_persist, flow_backend);
    if (!persistent_buf) {
      RS_LOG_ERR("Flow: failed to allocate persistent device tensors");
      ggml_free(ctx_persist);
      return false;
    }
  }

  // -----------------------------------------------------------------------
  // Encode pass: build a graph that produces (mu, spks, cond) into outputs,
  // then copy them into persistent backend tensors that are re-used by every
  // Euler step without a host round-trip.
  // -----------------------------------------------------------------------
  {
    ggml_init_params ip = { FLOW_CTX_BYTES, nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, FLOW_MAX_NODES, false);

    // Single combined token vector (prompt + gen) — avoids an in-graph
    // concat that broke scheduler backend assignment with non-empty prompts.
    const int T_tok = T_pt + T_gen;
    std::vector<int32_t> tokens_host;
    tokens_host.reserve(T_tok);
    if (T_pt > 0) {
      tokens_host.insert(tokens_host.end(),
                         state.prompt_token.begin(),
                         state.prompt_token.end());
    }
    tokens_host.insert(tokens_host.end(),
                       state.speech_token_ids.begin(),
                       state.speech_token_ids.begin() + n_tokens_in);
    ggml_tensor *t_gen = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_tok);
    ggml_set_input(t_gen); ggml_set_name(t_gen, "tokens");
    ggml_tensor *t_prompt = nullptr;  // unused now; encode picks T_pt up from prompt_feat

    ggml_tensor *t_pf = nullptr;
    if (T_pt_mel > 0) {
      t_pf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mel_dim_, T_pt_mel, 1);
      ggml_set_input(t_pf); ggml_set_name(t_pf, "prompt_feat");
    }
    ggml_tensor *t_emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, spk_dim_in_);
    ggml_set_input(t_emb); ggml_set_name(t_emb, "spk_embedding");

    auto enc = BuildEncodeGraph(ctx, t_prompt, t_gen, t_pf, t_emb);
    ggml_tensor *mu_copy = ggml_cpy(ctx, enc.mu, mu_dev);
    ggml_tensor *cond_copy = ggml_cpy(ctx, enc.cond, cond_dev);
    ggml_tensor *spks_copy = ggml_cpy(ctx, enc.spks, spks_dev);
    ggml_build_forward_expand(gf, mu_copy);
    ggml_build_forward_expand(gf, cond_copy);
    ggml_build_forward_expand(gf, spks_copy);

    // Reset the scheduler in case a previous sub-model (V3 tokenizer /
    // CAMPPlus) left tensor backend assignments in flight.
    ggml_backend_sched_reset(sched);
    assign_flow_persistent_backend(
        sched, flow_backend, gf, {mu_dev, cond_dev, spks_dev});
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
      RS_LOG_ERR("Flow: failed to allocate encode graph");
      ggml_free(ctx);
      free_persistent();
      return false;
    }
    ggml_backend_tensor_set(t_gen, tokens_host.data(), 0,
                            tokens_host.size() * sizeof(int32_t));
    if (t_pf) {
      ggml_backend_tensor_set(t_pf, state.prompt_feat.data(), 0,
                              state.prompt_feat.size() * sizeof(float));
    }
    ggml_backend_tensor_set(t_emb, state.embedding.data(), 0,
                            spk_dim_in_ * sizeof(float));

    if (cv3_sched_compute(sched, gf, &imatrix_cb_) != GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("Flow: encode compute failed");
      ggml_free(ctx);
      free_persistent();
      return false;
    }

    // Debug: dump (mu, cond, spks) tensors for PT cross-check.
    // Format: int32 T_mel, int32 mel_dim, then f32 mu[mel_dim*T_mel],
    //         f32 cond[mel_dim*T_mel], f32 spks[mel_dim].
    if (const char *p = std::getenv("RS_CV3_DUMP_FLOW_ENC")) {
      if (FILE *f = std::fopen(p, "wb")) {
        std::vector<float> h_mu((size_t)mel_dim_ * T_mel);
        std::vector<float> h_cond((size_t)mel_dim_ * T_mel);
        std::vector<float> h_spks((size_t)mel_dim_);
        ggml_backend_tensor_get(mu_dev, h_mu.data(), 0,
                                h_mu.size() * sizeof(float));
        ggml_backend_tensor_get(cond_dev, h_cond.data(), 0,
                                h_cond.size() * sizeof(float));
        ggml_backend_tensor_get(spks_dev, h_spks.data(), 0,
                                h_spks.size() * sizeof(float));
        int32_t hdr[2] = {T_mel, mel_dim_};
        std::fwrite(hdr, sizeof(int32_t), 2, f);
        std::fwrite(h_mu.data(),   sizeof(float), h_mu.size(),   f);
        std::fwrite(h_cond.data(), sizeof(float), h_cond.size(), f);
        std::fwrite(h_spks.data(), sizeof(float), h_spks.size(), f);
        std::fclose(f);
        RS_LOG_INFO("Flow: dumped encode (mu,cond,spks) [%d, %d] -> %s",
                    T_mel, mel_dim_, p);
      }
    }
    ggml_free(ctx);
    ggml_backend_sched_reset(sched);
  }

  // -----------------------------------------------------------------------
  // Initial x: standard Gaussian noise [mel_dim, T_mel, 1]. Seeded so test
  // runs are deterministic.
  // -----------------------------------------------------------------------
  std::vector<float> h_x((size_t)mel_dim_ * T_mel);
  {
    std::mt19937 rng(state.rng());
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto &v : h_x) v = nd(rng);
  }
  ggml_backend_tensor_set(x_dev[0], h_x.data(), 0,
                          h_x.size() * sizeof(float));

  // -----------------------------------------------------------------------
  // Euler loop. Steps 1..N (N = local_steps), last step applies cut_len.
  // -----------------------------------------------------------------------
  // Precompute t[i] and dt[i] from the cosine schedule. ref1's get_t_and_dt
  // accumulates from index 0:  t[step] = sum_{k=1..step}(t_span[k]-t_span[k-1])
  //                                    = t_span[step] - t_span[0]
  //                                    + t_span[0]    (start value)
  // Simpler: t[step] = t_span[step-1]; dt[step] = t_span[step]-t_span[step-1].
  std::vector<float> ts(local_steps + 1), dts(local_steps + 1);
  for (int step = 1; step <= local_steps; ++step) {
    ts[step]  = t_span_use[step - 1];
    dts[step] = t_span_use[step] - t_span_use[step - 1];
  }

  for (int step = 1; step <= local_steps; ++step) {
    const bool last = (step == local_steps);
    // Streaming non-finalize chunks keep the prompt mel in-place so the
    // next chunk's encode gets a stable initial condition (matches upstream
    // `token2wav(stream=True, finalize=False)`).
    const int64_t cut_len = (last && finalize) ? (int64_t)T_pt_mel : 0;
    const int src_x = (step - 1) & 1;
    const int dst_x = step & 1;

    ggml_init_params ip = { FLOW_CTX_BYTES, nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, FLOW_MAX_NODES, false);

    ggml_tensor *pos_ids = nullptr;
    ggml_tensor *x_next = BuildCFMStepGraph(
        ctx, x_dev[src_x], mu_dev, spks_dev, cond_dev, cut_len,
        ts[step], dts[step], pos_ids);
    ggml_tensor *x_store = nullptr;
    if (last && finalize) {
      // Trimmed output: x_next is [mel_dim, T_gen_mel]; write into the
      // prefix of dst so the post-loop tensor_get reads contiguous rows.
      x_store = ggml_cpy(ctx, x_next, ggml_view_3d(
                                     ctx, x_dev[dst_x], mel_dim_, T_gen_mel, 1,
                                     x_dev[dst_x]->nb[1],
                                     x_dev[dst_x]->nb[2], 0));
    } else {
      x_store = ggml_cpy(ctx, x_next, x_dev[dst_x]);
    }
    ggml_build_forward_expand(gf, x_store);

    assign_flow_persistent_backend(
        sched, flow_backend, gf,
        {mu_dev, cond_dev, spks_dev, x_dev[0], x_dev[1]});
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
      RS_LOG_ERR("Flow: failed to allocate step graph (step=%d)", step);
      ggml_free(ctx);
      free_persistent();
      return false;
    }

    // Build position_ids on the host: [T_mel, 2] with each batch slot
    // identical = 0..T_mel-1.
    if (pos_ids) {
      std::vector<int32_t> pids((size_t)T_mel * 2);
      for (int b = 0; b < 2; ++b)
        for (int t = 0; t < T_mel; ++t)
          pids[(size_t)b * T_mel + t] = t;
      ggml_backend_tensor_set(pos_ids, pids.data(), 0,
                              pids.size() * sizeof(int32_t));
    }

    if (cv3_sched_compute(sched, gf, &imatrix_cb_) != GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("Flow: step compute failed (step=%d)", step);
      ggml_free(ctx);
      free_persistent();
      return false;
    }

    ggml_free(ctx);
    ggml_backend_sched_reset(sched);
  }

  // Streaming mode keeps the prompt mel attached so the caller can slice
  // by stream_token_offset; offline mode (finalize=true) emits only the
  // generated portion.
  const int T_out = finalize ? T_gen_mel : T_mel;

  // x_dev[local_steps & 1] is now [mel_dim, T_out] in its prefix.
  // Fetch once at the end and copy to row-major [T_out, 80] for HiFT.
  h_x.resize((size_t)mel_dim_ * T_out);
  ggml_backend_tensor_get(x_dev[local_steps & 1], h_x.data(), 0,
                          h_x.size() * sizeof(float));
  free_persistent();

  state.mel_output.assign((size_t)T_out * mel_dim_, 0.f);
  for (int t = 0; t < T_out; ++t) {
    for (int m = 0; m < mel_dim_; ++m) {
      state.mel_output[(size_t)t * mel_dim_ + m] =
          h_x[(size_t)t * mel_dim_ + m];
    }
  }
  if (finalize) {
    state.flow_done = true;
  }
  RS_LOG_INFO("Flow: produced mel [%d, %d]%s", T_out, mel_dim_,
              finalize ? "" : " (streaming, prompt prefix kept)");
  // Debug: dump the Flow mel output for offline inspection / PT-HiFT cross-check.
  if (const char *p = std::getenv("RS_CV3_DUMP_FLOW_MEL")) {
    FILE *f = std::fopen(p, "wb");
    if (f) {
      int32_t hdr[2] = {T_out, mel_dim_};
      std::fwrite(hdr, sizeof(int32_t), 2, f);
      std::fwrite(state.mel_output.data(), sizeof(float),
                  state.mel_output.size(), f);
      std::fclose(f);
      RS_LOG_INFO("Flow: dumped mel [%d, %d] f32 -> %s",
                  T_out, mel_dim_, p);
    }
  }
  return true;
}
