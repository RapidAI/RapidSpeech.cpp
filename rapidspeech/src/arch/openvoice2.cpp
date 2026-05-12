#include "openvoice2.h"
#include "core/rs_context.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>

#define OV2_MAX_NODES 8192

// Duration predictor outputs are calibrated for a different frame resolution
// than the hop_length in this model.  Scaling factor to map raw predictions to
// natural mel-frame durations.
#define OV2_DURATION_SCALE 12.0f

// =====================================================================
// Helper: Conv1D using ggml_conv_1d
// Input x is [in_channels, T], weight is [out_channels, in_channels, kw].
// ggml_conv_1d expects data in [OW, IC, N] layout (ne[1]==IC).
// =====================================================================
static struct ggml_tensor *conv1d_im2col(struct ggml_context *ctx,
                                          struct ggml_tensor *x,
                                          struct ggml_tensor *weight,
                                          struct ggml_tensor *bias,
                                          int kernel_size, int padding,
                                          int in_channels, int out_channels) {
  (void)kernel_size; (void)in_channels;  // kept for API compatibility
  struct ggml_tensor *x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
  struct ggml_tensor *out = ggml_conv_1d(ctx, weight, x_t, 1, padding, 1);

  out = ggml_cont(ctx, ggml_transpose(ctx, out));
  out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

  if (bias) {
    struct ggml_tensor *bias_2d = ggml_reshape_2d(ctx, bias, out_channels, 1);
    out = ggml_add(ctx, out, bias_2d);
  }

  return out;
}

// =====================================================================
// Helper: Weight Normalization
//
// Reconstructs the effective weight from weight_v and weight_g stored by
// torch.nn.utils.weight_norm.  The original weight is:
//     weight = g * v / ||v||
// where ||v|| is the L2 norm computed over all dims except the first
// (output-channel dim in PyTorch convention).  In GGUF the shapes are
// reversed, so weight_v is [ne0, ne1, ne2] and the norm is over ne0*ne1
// for each ne2 channel.
// =====================================================================
static struct ggml_tensor *apply_weight_norm(struct ggml_context *ctx,
                                              struct ggml_tensor *w_v,
                                              struct ggml_tensor *w_g) {
  int kw = (int)w_v->ne[0];  // kernel (or 1 for 1x1 conv)
  int md = (int)w_v->ne[1];  // middle dim
  int ch = (int)w_v->ne[2];  // per-channel count
  enum ggml_type orig_type = w_v->type;

  // Flatten norm dims: [kw * md, ch]
  int n = kw * md;
  struct ggml_tensor *w_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, w_v), n, ch);

  // Cast both to F32 for stable math (ggml_sum_rows requires F32)
  struct ggml_tensor *w_f32 = ggml_cast(ctx, w_2d, GGML_TYPE_F32);

  // ||v|| per channel: sqrt(sum(v^2, axis=0))
  struct ggml_tensor *sq    = ggml_sqr(ctx, w_f32);          // [n, ch]
  struct ggml_tensor *sum_sq = ggml_sum_rows(ctx, sq);      // [1, ch]
  struct ggml_tensor *norm  = ggml_sqrt(ctx, sum_sq);       // [1, ch]

  // Broadcast norm to match w_f32: [n, ch]
  struct ggml_tensor *norm_r = ggml_repeat(ctx, norm, w_f32);

  // v_norm = v / ||v||  →  [n, ch]
  struct ggml_tensor *v_norm = ggml_div(ctx, w_f32, norm_r);

  // Flatten weight_g from [1,1,ch] to [1,ch], cast to F32, broadcast
  struct ggml_tensor *g_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, w_g), 1, ch);
  struct ggml_tensor *g_f32 = ggml_cast(ctx, g_2d, GGML_TYPE_F32);
  struct ggml_tensor *g_r  = ggml_repeat(ctx, g_f32, v_norm);  // [n, ch]

  // weight = v_norm * g
  struct ggml_tensor *weight = ggml_mul(ctx, v_norm, g_r);    // [n, ch]

  // Cast back to original type
  weight = ggml_cast(ctx, weight, orig_type);

  // Reshape back to original 3-D shape
  weight = ggml_reshape_3d(ctx, weight, kw, md, ch);
  return weight;
}

// =====================================================================
// Helper: Layer Normalization
// =====================================================================
static struct ggml_tensor *layer_norm(struct ggml_context *ctx,
                                       struct ggml_tensor *x,
                                       struct ggml_tensor *w,
                                       struct ggml_tensor *b, float eps) {
  x = ggml_norm(ctx, x, eps);
  x = ggml_mul(ctx, x, w);
  if (b) x = ggml_add(ctx, x, b);
  return x;
}

// =====================================================================
// Pending-inputs registry for the no_alloc graph-building pattern.
//
// When building graphs with no_alloc=true (init_compute_ctx), input tensors
// have data=nullptr. We register the intended data here and upload it after
// ggml_backend_sched_alloc_graph via flush_pending_inputs().
// =====================================================================

struct PendingInput {
    struct ggml_tensor *tensor;
    std::vector<uint8_t> data;
};

static thread_local std::vector<PendingInput> g_pending_inputs;

static void register_pending_input(struct ggml_tensor *tensor,
                                    const void *data, size_t size) {
    PendingInput pi;
    pi.tensor = tensor;
    pi.data.resize(size);
    if (size > 0) std::memcpy(pi.data.data(), data, size);
    g_pending_inputs.push_back(std::move(pi));
}

static void flush_pending_inputs() {
    for (size_t i = 0; i < g_pending_inputs.size(); i++) {
        auto &pi = g_pending_inputs[i];
        ggml_backend_tensor_set(pi.tensor, pi.data.data(), 0, pi.data.size());
        if (pi.data.size() >= 4) {
            float first_val;
            ggml_backend_tensor_get(pi.tensor, &first_val, 0, sizeof(float));
        }
    }
    g_pending_inputs.clear();
}

// =====================================================================
// Helper: Multi-head Self-Attention with Relative Position (Shaw et al. 2018)
//
// emb_rel_k [head_dim, window_size]: Q-dependent relative position bias.
//   bias[i,j,h,b] = dot(Q[:,i,h,b], emb_rel_k[:, clip(i-j)])
//
// emb_rel_v [head_dim, window_size]: relative position value correction.
//   correction[i,d,h,b] = sum_j attn[i,j,h,b] * emb_rel_v[d, clip(i-j)]
//
// Context must be no_alloc=true (init_compute_ctx). Index tensors for
// get_rows are registered via register_pending_input and must be flushed
// by the caller after ggml_backend_sched_alloc_graph.
// =====================================================================
static struct ggml_tensor *multi_head_attention(
    struct ggml_context *ctx, struct ggml_tensor *x,
    struct ggml_tensor *q_w, struct ggml_tensor *q_b,
    struct ggml_tensor *k_w, struct ggml_tensor *k_b,
    struct ggml_tensor *v_w, struct ggml_tensor *v_b,
    struct ggml_tensor *o_w, struct ggml_tensor *o_b,
    struct ggml_tensor *emb_rel_k,
    struct ggml_tensor *emb_rel_v,
    int n_heads, int head_dim, int n_ctx) {

  int hidden = n_heads * head_dim;
  int B = x->ne[2] > 0 ? x->ne[2] : 1;


  // Squeeze kernel dim from 1×1 conv weights: [kw=1, in, out] → [in, out]
  // NOTE: weight tensors are from gguf_data context; ggml_cont copies them
  // into the compute ctx for safe reshaping
  if (q_w && q_w->ne[0] == 1 && q_w->ne[2] > 0)
    q_w = ggml_reshape_2d(ctx, ggml_cont(ctx, q_w), q_w->ne[1], q_w->ne[2]);
  if (k_w && k_w->ne[0] == 1 && k_w->ne[2] > 0)
    k_w = ggml_reshape_2d(ctx, ggml_cont(ctx, k_w), k_w->ne[1], k_w->ne[2]);
  if (v_w && v_w->ne[0] == 1 && v_w->ne[2] > 0)
    v_w = ggml_reshape_2d(ctx, ggml_cont(ctx, v_w), v_w->ne[1], v_w->ne[2]);
  if (o_w && o_w->ne[0] == 1 && o_w->ne[2] > 0)
    o_w = ggml_reshape_2d(ctx, ggml_cont(ctx, o_w), o_w->ne[1], o_w->ne[2]);


  // Linear projections
  struct ggml_tensor *Q = ggml_mul_mat(ctx, q_w, x);
  if (q_b) Q = ggml_add(ctx, Q, q_b);

  struct ggml_tensor *K = ggml_mul_mat(ctx, k_w, x);
  if (k_b) K = ggml_add(ctx, K, k_b);

  struct ggml_tensor *V = ggml_mul_mat(ctx, v_w, x);
  if (v_b) V = ggml_add(ctx, V, v_b);

  // Reshape to [head_dim, n_heads, n_ctx, B] → permute to [head_dim, n_ctx, n_heads, B]
  Q = ggml_permute(ctx,
      ggml_reshape_4d(ctx, Q, head_dim, n_heads, n_ctx, B), 0, 2, 1, 3);
  K = ggml_permute(ctx,
      ggml_reshape_4d(ctx, K, head_dim, n_heads, n_ctx, B), 0, 2, 1, 3);
  V = ggml_permute(ctx,
      ggml_reshape_4d(ctx, V, head_dim, n_heads, n_ctx, B), 0, 2, 1, 3);

  // Scaled dot-product attention
  float scale = 1.0f / sqrtf((float)head_dim);
  struct ggml_tensor *QK = ggml_mul_mat(ctx, Q, K);  // [n_ctx, n_ctx, n_heads, B]

  // --- Q-dependent relative position bias (emb_rel_k) ---
  if (emb_rel_k) {
    // Detect pre-computed bias matrix [n_ctx, n_ctx] vs weight tensor [head_dim, window_size]
    if ((int)emb_rel_k->ne[0] == n_ctx && (int)emb_rel_k->ne[1] == n_ctx) {
      // Pre-computed bias: broadcast [n_ctx, n_ctx, 1, 1] → [n_ctx, n_ctx, n_heads, B]
      struct ggml_tensor *bias_4d = ggml_reshape_4d(ctx, emb_rel_k, n_ctx, n_ctx, 1, 1);
      struct ggml_tensor *rel_bias = ggml_repeat(ctx, bias_4d, QK);
      QK = ggml_add(ctx, QK, rel_bias);
    } else {
      // Q-dependent relative position bias: bias = Q @ emb_rel_k, scattered to [n_ctx,n_ctx]
      int window_size = (int)emb_rel_k->ne[1];
      int max_rel_pos = (window_size - 1) / 2;

      // Compute Q_dot = Q^T @ emb_rel_k  →  [n_ctx*n_heads*B, window_size]
      // Q is permuted and not contiguous; make it contiguous before reshape
      struct ggml_tensor *Q_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, Q),
          head_dim, n_ctx * n_heads * B);
      struct ggml_tensor *erk = ggml_reshape_2d(ctx,
          ggml_cont(ctx, emb_rel_k), head_dim, window_size);
      struct ggml_tensor *Q_dot = ggml_mul_mat(ctx, Q_2d, erk);

      // Reshape to [window_size, n_ctx, n_heads, B] then permute to [n_ctx, window_size, n_heads, B]
      Q_dot = ggml_reshape_4d(ctx, Q_dot, window_size, n_ctx, n_heads, B);
      Q_dot = ggml_cont(ctx, ggml_permute(ctx, Q_dot, 1, 0, 2, 3));

      // Flatten, then reshape to [1, N] so each scalar is a 1-element row
      int64_t qdot_total = (int64_t)n_ctx * window_size * n_heads * B;
      struct ggml_tensor *Q_dot_rows = ggml_reshape_2d(ctx,
          ggml_reshape_1d(ctx, Q_dot, qdot_total), 1, qdot_total);

      // Precompute flat indices for gathering bias[i,j,h,b] = Q_dot[i, r, h, b]
      // where r = clip(i-j) + max_rel_pos
      int total_bias = n_ctx * n_ctx * n_heads * B;
      std::vector<int32_t> flat_indices(total_bias);
      int idx_pos = 0;
      for (int i = 0; i < n_ctx; i++) {
        for (int j = 0; j < n_ctx; j++) {
          int r = i - j;
          if (r < -max_rel_pos) r = -max_rel_pos;
          if (r > max_rel_pos) r = max_rel_pos;
          r += max_rel_pos;
          for (int h = 0; h < n_heads; h++) {
            for (int b = 0; b < B; b++) {
              flat_indices[idx_pos++] = (int32_t)(i +
                  n_ctx * ((int64_t)r + window_size * ((int64_t)h + n_heads * (int64_t)b)));
            }
          }
        }
      }

      struct ggml_tensor *idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_bias);
      ggml_set_name(idx, "rel_k_idx");
      ggml_set_input(idx);
      register_pending_input(idx, flat_indices.data(),
                             total_bias * sizeof(int32_t));

      // get_rows on [1, qdot_total] with idx gives [1, total_bias]
      struct ggml_tensor *bias_1d = ggml_get_rows(ctx, Q_dot_rows, idx);
      struct ggml_tensor *rel_bias = ggml_reshape_4d(ctx,
          ggml_reshape_1d(ctx, bias_1d, total_bias),
          n_ctx, n_ctx, n_heads, B);
      QK = ggml_add(ctx, QK, rel_bias);
    }
  }

  struct ggml_tensor *QK_soft = ggml_soft_max_ext(ctx, QK, nullptr, scale, 0.0f);

  // Attention output: attn @ V
  struct ggml_tensor *QKV = ggml_mul_mat(
      ctx, ggml_cont(ctx, ggml_transpose(ctx, V)), QK_soft);

  // --- Relative position VALUE correction (emb_rel_v) ---
  if (emb_rel_v) {
    int window_size = (int)emb_rel_v->ne[1];
    int max_rel_pos = (window_size - 1) / 2;

    // Build attn_by_rel[r, i, h, b] = attn[i, i-r, h, b] for each (i, r, h, b).
    // QK_soft: [n_ctx, n_ctx, n_heads, B], contiguous
    // Flat index of element [i, j, h, b]: i + n_ctx*(j + n_ctx*(h + n_heads*b))
    int64_t attn_total = (int64_t)n_ctx * n_ctx * n_heads * B;
    struct ggml_tensor *attn_rows = ggml_reshape_2d(ctx,
        ggml_reshape_1d(ctx, ggml_cont(ctx, QK_soft), attn_total),
        1, attn_total);

    int total_pairs = n_ctx * window_size * n_heads * B;
    std::vector<int32_t> rv_idx_data(total_pairs);
    std::vector<float>   rv_mask_data(total_pairs, 0.0f);
    int pi = 0;
    for (int i = 0; i < n_ctx; i++) {
      for (int r_idx = 0; r_idx < window_size; r_idx++) {
        int r = r_idx - max_rel_pos;
        int j = i - r;
        for (int h = 0; h < n_heads; h++) {
          for (int b = 0; b < B; b++) {
            if (j >= 0 && j < n_ctx) {
              rv_idx_data[pi] = (int32_t)(i +
                  n_ctx * ((int64_t)j + n_ctx * ((int64_t)h + n_heads * (int64_t)b)));
              rv_mask_data[pi] = 1.0f;
            } else {
              rv_idx_data[pi] = 0;  // out of bounds, will be masked
              rv_mask_data[pi] = 0.0f;
            }
            pi++;
          }
        }
      }
    }

    struct ggml_tensor *idx_v = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_pairs);
    ggml_set_name(idx_v, "rel_v_idx");
    ggml_set_input(idx_v);
    register_pending_input(idx_v, rv_idx_data.data(),
                           total_pairs * sizeof(int32_t));

    struct ggml_tensor *mask_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total_pairs);
    ggml_set_name(mask_v, "rel_v_mask");
    ggml_set_input(mask_v);
    register_pending_input(mask_v, rv_mask_data.data(),
                           total_pairs * sizeof(float));

    // get_rows on [1, attn_total] → [1, total_pairs] → apply mask → [window_size, n_ctx, n_heads, B]
    struct ggml_tensor *attn_gathered = ggml_get_rows(ctx, attn_rows, idx_v);
    struct ggml_tensor *attn_flat = ggml_reshape_1d(ctx, attn_gathered, total_pairs);
    attn_flat = ggml_mul(ctx, attn_flat,
        ggml_reshape_1d(ctx, mask_v, total_pairs));
    struct ggml_tensor *attn_by_rel = ggml_reshape_4d(ctx, attn_flat,
        n_ctx, window_size, n_heads, B);
    // Permute to [window_size, n_ctx, n_heads, B] for mul_mat
    attn_by_rel = ggml_permute(ctx, attn_by_rel, 1, 0, 2, 3);

    // correction = erv @ attn_by_rel: [head_dim, win] @ [win, n_ctx, nh, B]
    // → [n_ctx, head_dim, n_heads, B]
    // Copy emb_rel_v into compute context via ggml_cont
    struct ggml_tensor *erv = ggml_reshape_2d(ctx,
        ggml_cont(ctx, emb_rel_v), head_dim, window_size);
    struct ggml_tensor *rel_v_corr = ggml_mul_mat(ctx, erv, attn_by_rel);
    rel_v_corr = ggml_permute(ctx, rel_v_corr, 1, 0, 2, 3);  // → [head_dim, n_ctx, nh, B]
    QKV = ggml_add(ctx, QKV, rel_v_corr);
  }

  // Merge heads: permute & reshape back to [hidden, n_ctx, B]
  struct ggml_tensor *attn = ggml_reshape_3d(
      ctx,
      ggml_cont(ctx, ggml_permute(ctx, QKV, 0, 2, 1, 3)),
      hidden, n_ctx, B);

  // Output projection
  attn = ggml_mul_mat(ctx, o_w, attn);
  if (o_b) attn = ggml_add(ctx, attn, o_b);

  return attn;
}

// =====================================================================
// OpenVoice2Model implementation
// =====================================================================

OpenVoice2Model::OpenVoice2Model() {}
OpenVoice2Model::~OpenVoice2Model() {}

bool OpenVoice2Model::MapTensors(std::map<std::string, struct ggml_tensor*>& all) {
  for (auto& [name, tensor] : all) {
    if (name.find("text_encoder.") == 0) {
      weights_.text_encoder[name] = tensor;
    } else if (name.find("duration_predictor.") == 0) {
      weights_.duration_predictor[name] = tensor;
    } else if (name.find("flow_decoder.") == 0) {
      weights_.flow_decoder[name] = tensor;
    } else if (name.find("vocoder.") == 0) {
      weights_.vocoder[name] = tensor;
    } else if (name.find("posterior_encoder.") == 0) {
      weights_.posterior_encoder[name] = tensor;
    } else if (name.find("emb_") == 0) {
      weights_.embeddings[name] = tensor;
    } else {
      // Store in text_encoder as fallback (some weights may not have prefix)
      weights_.text_encoder[name] = tensor;
    }
  }
  RS_LOG_INFO("OpenVoice2: mapped %zu text_enc, %zu dur_pred, %zu flow, %zu vocoder tensors",
              weights_.text_encoder.size(), weights_.duration_predictor.size(),
              weights_.flow_decoder.size(), weights_.vocoder.size());
  return true;
}

bool OpenVoice2Model::Load(const std::unique_ptr<rs_context_t>& ctx,
                            ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Invalid context for OpenVoice2 Load");
    return false;
  }

  gguf_context* ctx_gguf = ctx->ctx_gguf;
  ggml_context* gguf_data = ctx->gguf_data;

  // Load hyperparameters from GGUF KV
  int64_t key;
  key = gguf_find_key(ctx_gguf, "openvoice2.hidden_channels");
  if (key != -1) hparams_.hidden_channels = gguf_get_val_i32(ctx_gguf, key);
  key = gguf_find_key(ctx_gguf, "openvoice2.sample_rate");
  if (key != -1) hparams_.sample_rate = gguf_get_val_i32(ctx_gguf, key);
  key = gguf_find_key(ctx_gguf, "openvoice2.hop_length");
  if (key != -1) hparams_.hop_length = gguf_get_val_i32(ctx_gguf, key);
  key = gguf_find_key(ctx_gguf, "openvoice2.n_fft");
  if (key != -1) hparams_.n_fft = gguf_get_val_i32(ctx_gguf, key);

  meta_.arch_name = "openvoice2";
  meta_.audio_sample_rate = hparams_.sample_rate;
  meta_.n_mels = hparams_.n_mels;
  meta_.vocab_size = hparams_.vocab_size;

  RS_LOG_INFO("OpenVoice2: hidden=%d, sr=%d, hop=%d",
              hparams_.hidden_channels, hparams_.sample_rate, hparams_.hop_length);

  // Init text frontend — try GGUF symbol table first, fall back to built-in
  {
    const int sym_key = gguf_find_key(ctx_gguf, "tokenizer.ggml.symbols");
    if (sym_key != -1) {
      std::string sym_json = gguf_get_val_str(ctx_gguf, sym_key);
      // Parse JSON array of strings (simple parser, no dependency needed)
      std::vector<std::string> symbols;
      size_t pos = sym_json.find('[');
      if (pos != std::string::npos) {
        pos++;  // skip '['
        while (pos < sym_json.size()) {
          if (sym_json[pos] == ']') break;
          if (sym_json[pos] == ',' || sym_json[pos] == ' ' || sym_json[pos] == '\n') { pos++; continue; }
          if (sym_json[pos] == '"') {
            pos++;  // skip opening '"'
            std::string sym;
            while (pos < sym_json.size() && sym_json[pos] != '"') {
              if (sym_json[pos] == '\\' && pos + 1 < sym_json.size()) pos++;
              sym += sym_json[pos++];
            }
            if (pos < sym_json.size()) pos++;  // skip closing '"'
            symbols.push_back(sym);
          } else {
            pos++;
          }
        }
      }
      if (!symbols.empty()) {
        text_frontend_.InitFromSymbols(symbols);
        RS_LOG_INFO("OpenVoice2: loaded %zu symbols from GGUF metadata", symbols.size());
      } else {
        text_frontend_.Init(nullptr);
        RS_LOG_WARN("OpenVoice2: failed to parse symbols, using built-in vocab");
      }
    } else {
      text_frontend_.Init(nullptr);
    }
  }

  // Map all tensors
  std::map<std::string, struct ggml_tensor*> tensors;
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; ++i) {
    const char* name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor* t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }

  if (!MapTensors(tensors)) return false;

  // Auto-detect n_mels from vocoder conv_pre input channels
  if (weights_.vocoder.count("vocoder.conv_pre.weight")) {
    auto* pre_w = weights_.vocoder["vocoder.conv_pre.weight"];
    hparams_.n_mels = static_cast<int32_t>(pre_w->ne[1]);  // in_channels
    RS_LOG_INFO("OpenVoice2: auto-detected n_mels=%d", hparams_.n_mels);
  }

  // Auto-detect n_flow_layers from flow_decoder weights (max index + 2 for flips)
  {
    int max_flow_idx = -1;
    for (auto& [name, t] : weights_.flow_decoder) {
      if (name.find("flow_decoder.flows.") == 0) {
        // name like "flow_decoder.flows.N.xxx"
        size_t pos = strlen("flow_decoder.flows.");
        int idx = 0;
        const char* p = name.c_str() + pos;
        while (*p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); p++; }
        if (idx > max_flow_idx) max_flow_idx = idx;
      }
    }
    if (max_flow_idx >= 0) {
      hparams_.n_flow_layers = max_flow_idx + 2;  // +1 for 0-index, +1 for final flip
      RS_LOG_INFO("OpenVoice2: auto-detected n_flow_layers=%d (max_idx=%d)",
                  hparams_.n_flow_layers, max_flow_idx);
    }
  }

  return true;
}

bool OpenVoice2Model::LoadConverter(const char* converter_path,
                                     ggml_backend_t backend) {
  RS_LOG_INFO("OpenVoice2: converter loading not yet implemented: %s", converter_path);
  converter_weights_.loaded = false;
  return true;  // Non-fatal: base TTS works without converter
}

std::shared_ptr<RSState> OpenVoice2Model::CreateState() {
  return std::make_shared<OpenVoice2State>();
}

// =====================================================================
// TTS-specific methods
// =====================================================================

bool OpenVoice2Model::PushText(RSState& state, const char* text,
                                const char* language) {
  auto& s = static_cast<OpenVoice2State&>(state);
  s.language = language ? language : "zh";

  // Set language_id based on language string
  if (s.language == "zh" || s.language == "ZH") s.language_id = 0;
  else if (s.language == "en" || s.language == "EN") s.language_id = 1;
  else if (s.language == "ja" || s.language == "JA") s.language_id = 2;
  else s.language_id = 3;  // "other"

  s.tone_ids.clear();
  s.phoneme_ids = text_frontend_.TextToPhonemeIds(text, s.language, &s.tone_ids);

  if (s.phoneme_ids.empty()) {
    RS_LOG_ERR("OpenVoice2: text frontend produced no phonemes");
    return false;
  }

  RS_LOG_INFO("OpenVoice2: text -> %zu phoneme IDs, %zu tone IDs",
              s.phoneme_ids.size(), s.tone_ids.size());
  return true;
}

bool OpenVoice2Model::PushReferenceAudio(RSState& state, const float* samples,
                                          int n_samples, int sample_rate,
                                          ggml_backend_sched_t sched) {
  auto& s = static_cast<OpenVoice2State&>(state);

  if (!converter_weights_.loaded) {
    RS_LOG_WARN("OpenVoice2: tone converter not loaded, ignoring reference audio");
    return true;  // Non-fatal
  }

  // TODO: compute mel spectrogram from reference audio
  // TODO: run tone color encoder to extract style embedding
  s.has_tone_embedding = false;
  return true;
}

// =====================================================================
// Encode: TextEncoder + DurationPredictor + FlowDecoder
// =====================================================================

bool OpenVoice2Model::Encode(const std::vector<float>& input_frames,
                              RSState& state, ggml_backend_sched_t sched) {
  auto& s = static_cast<OpenVoice2State&>(state);
  (void)input_frames;  // TTS doesn't use audio input for encoding

  if (s.phoneme_ids.empty()) {
    RS_LOG_ERR("OpenVoice2: no text pushed, call PushText first");
    return false;
  }

  // Step 1: Text Encoder
  if (!RunTextEncoder(s, sched)) return false;

  // Step 2: Duration Predictor
  if (!RunDurationPredictor(s, sched)) return false;

  // Step 3: Flow Decoder (generates full mel spectrogram)
  if (!RunFlowDecoder(s, sched)) return false;

  // Reset streaming cursor
  s.mel_chunk_cursor = 0;
  s.audio_output.clear();
  s.audio_read_cursor = 0;

  return true;
}

// =====================================================================
// Decode: Vocoder on next mel chunk (streaming)
// =====================================================================

bool OpenVoice2Model::Decode(RSState& state, ggml_backend_sched_t sched) {
  auto& s = static_cast<OpenVoice2State&>(state);

  if (s.mel_spectrogram.empty() || s.mel_chunk_cursor >= s.total_mel_frames) {
    return false;  // No more chunks
  }

  int chunk_size = hparams_.chunk_mel_frames;
  if (chunk_size <= 0) chunk_size = s.total_mel_frames;  // Non-streaming

  int mel_start = s.mel_chunk_cursor;
  int mel_len = std::min(chunk_size, s.total_mel_frames - mel_start);

  if (!RunVocoder(s, sched, mel_start, mel_len)) return false;

  s.mel_chunk_cursor += mel_len;
  return true;
}

int OpenVoice2Model::GetAudioOutput(RSState& state, float** out_data) {
  auto& s = static_cast<OpenVoice2State&>(state);
  if (s.audio_read_cursor >= static_cast<int>(s.audio_output.size())) {
    *out_data = nullptr;
    return 0;
  }
  *out_data = s.audio_output.data() + s.audio_read_cursor;
  int n = static_cast<int>(s.audio_output.size()) - s.audio_read_cursor;
  s.audio_read_cursor = static_cast<int>(s.audio_output.size());
  return n;
}

// =====================================================================
// Sub-graph: Text Encoder (Transformer with relative-position self-attention)
//
// Architecture (VITS/MeloTTS text encoder):
//   1. Phoneme Embedding lookup
//   2. Scale + add positional encoding
//   3. N × Transformer block:
//      a. LayerNorm → Multi-head Self-Attention → Residual
//      b. LayerNorm → Conv1D+GELU+Conv1D (FFN) → Residual
//   4. Final LayerNorm
// =====================================================================

bool OpenVoice2Model::RunTextEncoder(OpenVoice2State& state,
                                      ggml_backend_sched_t sched) {
  auto& w = weights_.text_encoder;
  if (w.empty()) {
    RS_LOG_ERR("OpenVoice2: no text_encoder weights loaded");
    return false;
  }

  int T = static_cast<int>(state.phoneme_ids.size());
  int C = hparams_.hidden_channels;
  int n_heads = hparams_.n_heads;
  int head_dim = C / n_heads;
  int n_layers = hparams_.n_layers;

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for TextEncoder");
    return false;
  }

  g_pending_inputs.clear();

  // Input: phoneme IDs
  struct ggml_tensor *phoneme_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
  ggml_set_name(phoneme_ids, "phoneme_ids");
  ggml_set_input(phoneme_ids);

  struct ggml_tensor *cur = nullptr;

  // --- Embedding Lookup ---
  // Try text_encoder.emb.weight first, then emb_g.weight
  struct ggml_tensor *emb_table = nullptr;
  for (auto& [name, t] : w) {
    if (name.find("emb.") != std::string::npos ||
        name.find("emb_") != std::string::npos) {
      emb_table = t;
      break;
    }
  }
  if (!emb_table) {
    // Fallback: look in embeddings map
    for (auto& [name, t] : weights_.embeddings) {
      emb_table = t;
      break;
    }
  }

  // Embedding table: [hidden, vocab] (ggml) or [vocab, hidden] (PyTorch)
  if (emb_table && (emb_table->ne[0] >= C || emb_table->ne[1] >= C)) {
    // PyTorch stores embeddings as [vocab, hidden] — transpose for ggml
    struct ggml_tensor *emb = emb_table;
    if (emb_table->ne[0] < C) {
      emb = ggml_transpose(ctx0, emb_table);
    }
    cur = ggml_get_rows(ctx0, emb, phoneme_ids);
    ggml_set_name(cur, "phoneme_emb");
  } else {
    // No embedding table available — create one-hot-like input directly
    // Use a zero tensor with identity-like values at phoneme positions
    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
    ggml_set_input(cur);
    RS_LOG_WARN("OpenVoice2: no embedding table found, using raw input");
  }

  // --- Tone Embedding ---
  // tone_emb.weight: [hidden_channels, n_tones] where n_tones=11 (tone 0-5 + extras)
  struct ggml_tensor *tone_emb_weight = nullptr;
  auto tone_it = w.find("text_encoder.tone_emb.weight");
  if (tone_it != w.end()) tone_emb_weight = tone_it->second;

  if (tone_emb_weight && !state.tone_ids.empty()) {
    int Ttone = (int)state.tone_ids.size();
    struct ggml_tensor *tone_ids_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, Ttone);
    ggml_set_name(tone_ids_tensor, "tone_ids");
    ggml_set_input(tone_ids_tensor);

    register_pending_input(tone_ids_tensor, state.tone_ids.data(),
                           Ttone * sizeof(int32_t));

    // Ensure embedding table has ne[0]=C for proper get_rows output
    struct ggml_tensor *tone_tbl = tone_emb_weight;
    if (tone_tbl->ne[0] < C) tone_tbl = ggml_transpose(ctx0, tone_tbl);
    struct ggml_tensor *tone_emb = ggml_get_rows(ctx0, tone_tbl, tone_ids_tensor);
    ggml_set_name(tone_emb, "tone_emb");
    cur = ggml_add(ctx0, cur, tone_emb);
  }

  // --- Language Embedding ---
  // language_emb.weight: [hidden_channels, n_langs] where n_langs=4
  struct ggml_tensor *lang_emb_weight = nullptr;
  auto lang_it = w.find("text_encoder.language_emb.weight");
  if (lang_it != w.end()) lang_emb_weight = lang_it->second;

  if (lang_emb_weight) {
    // Repeat lang_id T times so get_rows produces [C, T] directly
    std::vector<int32_t> lang_ids_all(T, state.language_id);
    struct ggml_tensor *lang_id_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(lang_id_tensor, "lang_ids");
    ggml_set_input(lang_id_tensor);

    register_pending_input(lang_id_tensor, lang_ids_all.data(),
                           T * sizeof(int32_t));

    struct ggml_tensor *lang_tbl = lang_emb_weight;
    if (lang_tbl->ne[0] < C) lang_tbl = ggml_transpose(ctx0, lang_tbl);
    struct ggml_tensor *lang_emb = ggml_get_rows(ctx0, lang_tbl, lang_id_tensor);
    ggml_set_name(lang_emb, "lang_emb");
    cur = ggml_add(ctx0, cur, lang_emb);
  }

  // --- Scale ---
  cur = ggml_scale(ctx0, cur, sqrtf((float)C));

  // --- Positional Encoding ---
  // Generate sinusoidal positional encoding
  int pos_len = T;
  struct ggml_tensor *pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, pos_len);
  ggml_set_name(pos_enc, "pos_enc");
  ggml_set_input(pos_enc);

  // Pre-compute sinusoidal positional encoding.
  // PE(pos, 2i)   = sin(pos / 10000^(2i/C))
  // PE(pos, 2i+1) = cos(pos / 10000^(2i/C))
  // Data layout: tensor is [C, pos_len] — ne0=C channels, ne1=pos_len positions.
  // In memory: channel c at position p is at offset p*C + c.
  // OR equivalently: row-major [C, T] where ne0 varies fastest.
  // ggml tensor layout: data[i0 + i1*ne0 + ...] where i0 varies fastest.
  // So data[c + p*C] is channel c at position p.
  std::vector<float> pos_data(C * pos_len, 0.0f);
  for (int p = 0; p < pos_len; p++) {
    for (int i = 0; i < C / 2; i++) {
      float freq = powf(10000.0f, -2.0f * (float)i / (float)C);
      pos_data[p * C + (2 * i)]     = sinf((float)p * freq);
      pos_data[p * C + (2 * i + 1)] = cosf((float)p * freq);
    }
  }
  // Data will be uploaded after graph allocation
  cur = ggml_add(ctx0, cur, pos_enc);

  // --- N Transformer Layers ---
  for (int layer = 0; layer < n_layers; layer++) {
    // Build weight name prefixes for this layer
    std::string prefix = "text_encoder.encoder.layers." + std::to_string(layer);
    std::string attn_pref = "text_encoder.encoder.attn_layers." + std::to_string(layer);

    struct ggml_tensor *residual = cur;

    // --- Self-Attention ---
    // Try attn_layers naming first (MeloTTS's relative position transformer),
    // fallback to standard transformer naming
    struct ggml_tensor *q_w = nullptr, *q_b = nullptr;
    struct ggml_tensor *k_w = nullptr, *k_b = nullptr;
    struct ggml_tensor *v_w = nullptr, *v_b = nullptr;
    struct ggml_tensor *o_w = nullptr, *o_b = nullptr;
    struct ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    struct ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    struct ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    struct ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    struct ggml_tensor *emb_rel_k = nullptr;
    struct ggml_tensor *emb_rel_v = nullptr;

    // Look up weights by common naming conventions
    // Chinese MeloTTS: norm_layers_1.<N>.gamma/beta, ffn_layers.<N>.conv_1.*
    std::string nl1 = "norm_layers_1." + std::to_string(layer);
    std::string nl2 = "norm_layers_2." + std::to_string(layer);
    std::string ffn  = "ffn_layers." + std::to_string(layer);
    for (auto& [name, t] : w) {
      if (name.find(attn_pref + ".conv_q.") != std::string::npos) { q_w = t;
      } else if (name.find(prefix + ".self_attn.q_proj.") != std::string::npos) { q_w = t;
      } else if (name.find(attn_pref + ".conv_k.") != std::string::npos) { k_w = t;
      } else if (name.find(prefix + ".self_attn.k_proj.") != std::string::npos) { k_w = t;
      } else if (name.find(attn_pref + ".conv_v.") != std::string::npos) { v_w = t;
      } else if (name.find(prefix + ".self_attn.v_proj.") != std::string::npos) { v_w = t;
      } else if (name.find(attn_pref + ".conv_o.") != std::string::npos) { o_w = t;
      } else if (name.find(prefix + ".self_attn.o_proj.") != std::string::npos) { o_w = t;
      } else if (name.find(attn_pref + ".emb_rel_k") != std::string::npos) {
        emb_rel_k = t;
      } else if (name.find(attn_pref + ".emb_rel_v") != std::string::npos) {
        emb_rel_v = t;
      } else if (name.find(prefix + ".norm1.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) norm1_w = t; else norm1_b = t;
      } else if (name.find(prefix + ".norm2.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) norm2_w = t; else norm2_b = t;
      } else if (name.find(prefix + ".conv1.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv1_w = t; else conv1_b = t;
      } else if (name.find(prefix + ".conv2.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv2_w = t; else conv2_b = t;
      } else if (name.find(nl1 + ".") != std::string::npos) {
        if (name.find("gamma") != std::string::npos) norm1_w = t; else norm1_b = t;
      } else if (name.find(nl2 + ".") != std::string::npos) {
        if (name.find("gamma") != std::string::npos) norm2_w = t; else norm2_b = t;
      } else if (name.find(ffn + ".conv_1.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv1_w = t; else conv1_b = t;
      } else if (name.find(ffn + ".conv_2.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv2_w = t; else conv2_b = t;
      }
    }

    // Self-attention block
    if (norm1_w && q_w && k_w && v_w && o_w) {
      cur = layer_norm(ctx0, cur, norm1_w, norm1_b, 1e-5f);
      struct ggml_tensor *attn_out = multi_head_attention(
          ctx0, cur, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
          emb_rel_k, nullptr, n_heads, head_dim, T);
      cur = ggml_add(ctx0, attn_out, residual);
    } else {
      RS_LOG_WARN("OpenVoice2: text_encoder layer %d missing attention weights, skipping", layer);
      cur = residual;
    }

    // FFN block (Conv1D)
    residual = cur;

    if (conv1_w && conv2_w) {
      if (norm2_w) {
        cur = layer_norm(ctx0, cur, norm2_w, norm2_b, 1e-5f);
      }

      // Conv1D 3×1 with same padding
      int k_size = conv1_w->ne[0] / C;  // infer kernel size from weight
      if (k_size <= 0) k_size = 3;
      if (conv1_w->ne[1] >= C) {
        // Weight: [kw, in, out] → ne[0]=kw, ne[1]=in, ne[2]=out
        cur = conv1d_im2col(ctx0, cur, conv1_w, conv1_b, k_size,
                            k_size / 2, C, conv1_w->ne[2]);
        cur = ggml_gelu(ctx0, cur);

        k_size = conv2_w->ne[0] / conv2_w->ne[1];
        if (k_size <= 0) k_size = 3;
        cur = conv1d_im2col(ctx0, cur, conv2_w, conv2_b, k_size,
                            k_size / 2, conv2_w->ne[1], conv2_w->ne[2]);
      } else {
        // Fallback: use mul_mat for linear layers
        cur = ggml_mul_mat(ctx0, conv1_w, cur);
        if (conv1_b) cur = ggml_add(ctx0, cur, conv1_b);
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, conv2_w, cur);
        if (conv2_b) cur = ggml_add(ctx0, cur, conv2_b);
      }

      cur = ggml_add(ctx0, cur, residual);
    } else {
      RS_LOG_WARN("OpenVoice2: text_encoder layer %d missing FFN weights, skipping", layer);
      cur = residual;
    }
  }

  // --- Final LayerNorm ---
  struct ggml_tensor *final_norm_w = nullptr, *final_norm_b = nullptr;
  for (auto& [name, t] : w) {
    if (name.find("after_norm.") != std::string::npos ||
        name.find("norm_post.") != std::string::npos) {
      if (name.find("weight") != std::string::npos || name.find("gamma") != std::string::npos)
        final_norm_w = t;
      else final_norm_b = t;
    }
  }
  if (final_norm_w) {
    cur = layer_norm(ctx0, cur, final_norm_w, final_norm_b, 1e-5f);
  }
  // No fallback RMS norm — let the model layers determine output scale

  ggml_set_name(cur, "text_encoder_out");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // --- Execute ---
  std::vector<int32_t> ids_data = state.phoneme_ids;

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: TextEncoder graph allocation failed");
    ggml_free(ctx0);
    return false;
  }

  // Upload input data after graph allocation (backend buffers are now assigned)
  ggml_backend_tensor_set(phoneme_ids, ids_data.data(), 0,
                          ids_data.size() * sizeof(int32_t));
  ggml_backend_tensor_set(pos_enc, pos_data.data(), 0,
                          pos_data.size() * sizeof(float));

  // Flush relative-position index/mask tensors registered by multi_head_attention
  flush_pending_inputs();

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read output ---
  state.encoder_hidden.resize(C * T);
  ggml_backend_tensor_get(cur, state.encoder_hidden.data(), 0,
                          state.encoder_hidden.size() * sizeof(float));
  state.encoder_T = T;

  {
    float hmin = 1e9, hmax = -1e9;
    double hsum = 0;
    for (int i = 0; i < C * T; i++) {
      float v = state.encoder_hidden[i];
      if (v < hmin) hmin = v;
      if (v > hmax) hmax = v;
      hsum += v;
    }
    RS_LOG_INFO("OpenVoice2: TextEncoder hidden values [%.4f..%.4f] mean=%.4f (%d values)",
                hmin, hmax, (float)(hsum / (C * T)), C * T);
  }

  ggml_free(ctx0);
  RS_LOG_INFO("OpenVoice2: TextEncoder -> [%d, %d]", C, T);
  return true;
}

// =====================================================================
// Sub-graph: Duration Predictor
//
// Architecture: Conv1D → LayerNorm → ReLU → Conv1D → LayerNorm → ReLU → Linear → exp
// =====================================================================

bool OpenVoice2Model::RunDurationPredictor(OpenVoice2State& state,
                                            ggml_backend_sched_t sched) {
  auto& w = weights_.duration_predictor;
  if (w.empty()) {
    RS_LOG_WARN("OpenVoice2: no duration_predictor weights, using default durations");
    // Fallback: assign default durations
    int T = state.encoder_T;
    state.durations.resize(T);
    state.total_mel_frames = 0;
    for (int t = 0; t < T; t++) {
      state.durations[t] = 5;
      state.total_mel_frames += 5;
    }
    return true;
  }

  int T = state.encoder_T;
  int C = hparams_.hidden_channels;

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for DurationPredictor");
    return false;
  }

  // Input: encoder hidden [C, T]
  struct ggml_tensor *hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
  ggml_set_name(hidden, "encoder_hidden");
  ggml_set_input(hidden);

  struct ggml_tensor *cur = hidden;

  // Find weights
  struct ggml_tensor *conv0_w = nullptr, *conv0_b = nullptr;
  struct ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
  struct ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
  struct ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
  struct ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
  struct ggml_tensor *cond_w = nullptr, *cond_b = nullptr;

  for (auto& [name, t] : w) {
    // Chinese MeloTTS: conv_1/conv_2, norm_1/norm_2
    // English MeloTTS: conv.0/conv.1, norm.0/norm.1
    if (name.find("conv_1.") != std::string::npos ||
        name.find("conv.0.") != std::string::npos) {
      if (name.find("weight") != std::string::npos || name.find("gamma") == std::string::npos)
        conv0_w = t; else conv0_b = t;
    } else if (name.find("conv_2.") != std::string::npos ||
               name.find("conv.1.") != std::string::npos) {
      if (name.find("weight") != std::string::npos || name.find("gamma") == std::string::npos)
        conv1_w = t; else conv1_b = t;
    } else if (name.find("norm_1.") != std::string::npos ||
               name.find("norm.0.") != std::string::npos) {
      if (name.find("gamma") != std::string::npos || name.find("weight") != std::string::npos)
        norm0_w = t; else norm0_b = t;
    } else if (name.find("norm_2.") != std::string::npos ||
               name.find("norm.1.") != std::string::npos) {
      if (name.find("gamma") != std::string::npos || name.find("weight") != std::string::npos)
        norm1_w = t; else norm1_b = t;
    } else if (name.find("proj.") != std::string::npos) {
      if (name.find("weight") != std::string::npos) proj_w = t; else proj_b = t;
    } else if (name.find("cond.") != std::string::npos) {
      if (name.find("weight") != std::string::npos) cond_w = t; else cond_b = t;
    }
  }

  if (!conv0_w || !proj_w) {
    RS_LOG_WARN("OpenVoice2: duration predictor missing key weights, using default durations");
    ggml_free(ctx0);
    int T2 = state.encoder_T;
    state.durations.resize(T2);
    state.total_mel_frames = 0;
    for (int t = 0; t < T2; t++) {
      state.durations[t] = 5;
      state.total_mel_frames += 5;
    }
    return true;
  }

  // Conv block 0
  int k_size = conv0_w->ne[0] / C;
  if (k_size <= 0) k_size = 3;
  int out_ch0 = static_cast<int>(conv0_w->ne[2]);
  if (out_ch0 <= 0) out_ch0 = static_cast<int>(conv0_w->ne[1]);
  if (conv0_w->ne[0] >= 3 && conv0_w->ne[1] >= C) {
    cur = conv1d_im2col(ctx0, cur, conv0_w, conv0_b, k_size, k_size / 2, C, out_ch0);
  } else {
    cur = ggml_mul_mat(ctx0, conv0_w, cur);
    if (conv0_b) cur = ggml_add(ctx0, cur, conv0_b);
  }

  // Cond skip connection: project encoder hidden and add to conv0 output.
  // cond weight: [1, 256, 192] — PW Conv1d(IC=256, OC=192) in PyTorch,
  // stored as [KW=1, ne1=256, ne2=192] in GGUF.
  if (cond_w) {
    // Squeeze kernel dim: [1, 256, 192] → [256, 192]
    struct ggml_tensor *cond_w2d = ggml_reshape_2d(ctx0, ggml_cont(ctx0, cond_w),
                                                     cond_w->ne[1], cond_w->ne[2]);
    // We need cond x hidden where cond_w2d is [256, 192] and hidden is [192, T].
    // ggml_mul_mat(A, B) = A^T @ B, so mul_mat(cond_w2d^T, hidden) = cond_w2d @ hidden.
    // cond_w2d^T is [192, 256], times hidden [192, T] gives [256, T].
    struct ggml_tensor *cond_out = ggml_mul_mat(ctx0,
        ggml_cont(ctx0, ggml_transpose(ctx0, cond_w2d)), hidden);
    // Result has shape [T, 256]; transpose to [256, T] to match cur
    cond_out = ggml_reshape_2d(ctx0,
        ggml_cont(ctx0, ggml_transpose(ctx0, cond_out)), out_ch0, T);
    cur = ggml_add(ctx0, cur, cond_out);
  }

  int cur_ch = cur->ne[0];
  if (norm0_w && norm0_w->ne[0] == cur_ch) {
    cur = layer_norm(ctx0, cur, norm0_w, norm0_b, 1e-5f);
  }

  cur = ggml_relu(ctx0, cur);

  // Conv block 1
  if (conv1_w) {
    int cur_ch1 = static_cast<int>(cur->ne[0]);
    int out_ch1 = static_cast<int>(conv1_w->ne[2]);
    if (out_ch1 <= 0) out_ch1 = static_cast<int>(conv1_w->ne[1]);
    if (conv1_w->ne[0] >= 3 && conv1_w->ne[1] >= cur_ch1) {
      k_size = 3;
      cur = conv1d_im2col(ctx0, cur, conv1_w, conv1_b, k_size, k_size / 2, cur_ch1, out_ch1);
    } else {
      cur = ggml_mul_mat(ctx0, conv1_w, cur);
      if (conv1_b) cur = ggml_add(ctx0, cur, conv1_b);
    }

    if (norm1_w && norm1_w->ne[0] == cur->ne[0]) {
      cur = layer_norm(ctx0, cur, norm1_w, norm1_b, 1e-5f);
    }

    cur = ggml_relu(ctx0, cur);
  }

  // Projection to 1 dimension
  // proj weight may be 3D [1, in, 1] (1x1 conv) → squeeze to 2D [in, 1]
  if (proj_w->ne[0] == 1 && proj_w->ne[2] > 0)
    proj_w = ggml_reshape_2d(ctx0, ggml_cont(ctx0, proj_w), proj_w->ne[1], proj_w->ne[2]);
  cur = ggml_mul_mat(ctx0, proj_w, cur);
  if (proj_b) cur = ggml_add(ctx0, cur, proj_b);

  // Squeeze channel dimension: [1, T] → [T]
  cur = ggml_reshape_1d(ctx0, cur, T);

  // exp() to get positive durations, add minimum 0.5
  cur = ggml_exp(ctx0, cur);
  // Add minimum duration to avoid zero-length
  float min_dur = 0.5f;
  struct ggml_tensor *min_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, T);
  ggml_set_input(min_t);
  std::vector<float> min_data(T, min_dur);

  cur = ggml_add(ctx0, cur, min_t);

  ggml_set_name(cur, "durations");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // --- Execute ---
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: DurationPredictor graph allocation failed");
    ggml_free(ctx0);
    return false;
  }

  ggml_backend_tensor_set(hidden, state.encoder_hidden.data(), 0,
                          state.encoder_hidden.size() * sizeof(float));
  ggml_backend_tensor_set(min_t, min_data.data(), 0, min_data.size() * sizeof(float));

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read output ---
  std::vector<float> dur_float(T);
  ggml_backend_tensor_get(cur, dur_float.data(), 0, T * sizeof(float));

  state.durations.resize(T);
  state.total_mel_frames = 0;
  float dur_min = 1e9, dur_max = -1e9;
  for (int t = 0; t < T; t++) {
    if (dur_float[t] < dur_min) dur_min = dur_float[t];
    if (dur_float[t] > dur_max) dur_max = dur_float[t];
    int dur = std::max(1, static_cast<int>(std::round(dur_float[t] * OV2_DURATION_SCALE)));
    state.durations[t] = dur;
    state.total_mel_frames += dur;
  }
  RS_LOG_INFO("OpenVoice2: raw durations [%.4f..%.4f] (%d values)", dur_min, dur_max, T);

  ggml_free(ctx0);
  RS_LOG_INFO("OpenVoice2: DurationPredictor -> %d total mel frames", state.total_mel_frames);
  return true;
}

// =====================================================================
// Sub-graph: Flow Decoder
//
// Architecture: Expand hidden by durations → N coupling blocks with
// WaveNet-style affine transforms
// =====================================================================

bool OpenVoice2Model::RunFlowDecoder(OpenVoice2State& state,
                                      ggml_backend_sched_t sched) {
  auto& w = weights_.flow_decoder;
  int n_mels = hparams_.n_mels;
  int T_mel = state.total_mel_frames;
  int C = hparams_.hidden_channels;
  int T_txt = state.encoder_T;


  if (T_mel <= 0) {
    RS_LOG_ERR("OpenVoice2: no mel frames to generate");
    return false;
  }

  if (w.empty()) {
    RS_LOG_WARN("OpenVoice2: no flow_decoder weights, generating placeholder mel");
    // Fallback: simple expansion of encoder hidden
    state.mel_spectrogram.resize(n_mels * T_mel, 0.0f);
    int mel_pos = 0;
    for (int t = 0; t < T_txt && mel_pos < T_mel; t++) {
      int dur = state.durations[t];
      for (int d = 0; d < dur && mel_pos < T_mel; d++) {
        for (int m = 0; m < std::min(n_mels, C); m++) {
          state.mel_spectrogram[m + mel_pos * n_mels] =
              state.encoder_hidden[m + t * C];
        }
        mel_pos++;
      }
    }
    return true;
  }

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for FlowDecoder");
    return false;
  }

  g_pending_inputs.clear();

  struct ggml_tensor *cur;

  // --- Expand hidden by durations ---
  // Pre-compute expanded hidden on CPU from state.encoder_hidden
  struct ggml_tensor *expanded = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T_mel);
  ggml_set_name(expanded, "expanded_hidden");

  std::vector<float> exp_data(C * T_mel, 0.0f);
  int mel_pos = 0;
  for (int t = 0; t < T_txt && mel_pos < T_mel; t++) {
    int dur = state.durations[t];
    for (int d = 0; d < dur && mel_pos < T_mel; d++) {
      for (int c = 0; c < C; c++) {
        exp_data[c + mel_pos * C] = state.encoder_hidden[c + t * C];
      }
      mel_pos++;
    }
  }
  ggml_set_input(expanded);
  cur = expanded;

  // --- Flow coupling blocks ---
  // Chinese MeloTTS: flows at even indices (0,2,4,6) have pre+enc+post transformer;
  // odd indices (1,3,5,7) are flip-only blocks.
  // English MeloTTS: flows at consecutive indices with in_layers WaveNet.
  int n_flows = hparams_.n_flow_layers;
  int C_flow = cur->ne[0];  // flow operates on this many channels (hidden_channels)
  int half = C_flow / 2;

  for (int flow = 0; flow < n_flows; flow++) {
    std::string flow_pref = "flow_decoder.flows." + std::to_string(flow);

    // Check if this flow has weights (coupling) or is flip-only
    bool has_weights = false;
    for (auto& [name, t] : w) {
      if (name.find(flow_pref + ".") == 0) { has_weights = true; break; }
    }

    if (!has_weights) {
      // Flip-only block: swap x_a and x_b
      struct ggml_tensor *x_a = ggml_view_2d(ctx0, cur, half, cur->ne[1],
                                              cur->nb[1], 0);
      struct ggml_tensor *x_b = ggml_view_2d(ctx0, cur, C_flow - half,
                                              cur->ne[1], cur->nb[1],
                                              half * sizeof(float));
      cur = ggml_concat(ctx0, x_b, x_a, 0);
      continue;
    }

    // --- Coupling block ---
    // Find weights: Chinese model uses pre/enc/post, English uses in_layers/WaveNet
    struct ggml_tensor *pre_w = nullptr, *pre_b = nullptr;   // 1x1 pre-conv
    struct ggml_tensor *post_w = nullptr, *post_b = nullptr;  // 1x1 post-conv
    // English model WaveNet weights:
    struct ggml_tensor *in_l0_w = nullptr, *in_l1_w = nullptr, *in_l2_w = nullptr;
    // Chinese model: transformer encoder weights per layer
    struct FlowEncWeights {
      struct ggml_tensor *norm1_w, *norm1_b;
      struct ggml_tensor *norm2_w, *norm2_b;
      struct ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
      struct ggml_tensor *conv1_w, *conv1_b, *conv2_w, *conv2_b;
      struct ggml_tensor *emb_rel_k = nullptr;
      struct ggml_tensor *emb_rel_v = nullptr;
    };
    std::vector<FlowEncWeights> enc_layers;
    int n_enc_layers = 0;

    for (auto& [name, t] : w) {
      if (name.find(flow_pref + ".") != 0) continue;
      // Chinese model naming
      if (name.find(".pre.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) pre_w = t; else pre_b = t;
      } else if (name.find(".post.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) post_w = t; else post_b = t;
      } else if (name.find(".enc.attn_layers.") != std::string::npos) {
        // Parse layer index: .enc.attn_layers.N.
        size_t p = name.find(".enc.attn_layers.") + 17;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("conv_q.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.q_w = t; else ew.q_b = t;
        } else if (name.find("conv_k.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.k_w = t; else ew.k_b = t;
        } else if (name.find("conv_v.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.v_w = t; else ew.v_b = t;
        } else if (name.find("conv_o.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.o_w = t; else ew.o_b = t;
        } else if (name.find("emb_rel_k") != std::string::npos) {
          ew.emb_rel_k = t;
        } else if (name.find("emb_rel_v") != std::string::npos) {
          ew.emb_rel_v = t;
        }
      } else if (name.find(".enc.norm_layers_1.") != std::string::npos) {
        size_t p = name.find(".enc.norm_layers_1.") + 20;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("gamma") != std::string::npos) ew.norm1_w = t; else ew.norm1_b = t;
      } else if (name.find(".enc.norm_layers_2.") != std::string::npos) {
        size_t p = name.find(".enc.norm_layers_2.") + 20;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("gamma") != std::string::npos) ew.norm2_w = t; else ew.norm2_b = t;
      } else if (name.find(".enc.ffn_layers.") != std::string::npos) {
        size_t p = name.find(".enc.ffn_layers.") + 17;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("conv_1.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.conv1_w = t; else ew.conv1_b = t;
        } else if (name.find("conv_2.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.conv2_w = t; else ew.conv2_b = t;
        }
      }
      // English model naming (WaveNet)
      else if (name.find(".in_layers.0.") != std::string::npos) in_l0_w = t;
      else if (name.find(".in_layers.1.") != std::string::npos) in_l1_w = t;
      else if (name.find(".in_layers.2.") != std::string::npos) in_l2_w = t;
    }
    n_enc_layers = static_cast<int>(enc_layers.size());

    // Check if this is a Chinese-style (pre+enc+post) or English-style (WaveNet) flow
    bool is_chinese_flow = (pre_w && post_w && n_enc_layers > 0);

    if (!is_chinese_flow && (!in_l0_w || !in_l2_w)) {
      RS_LOG_WARN("OpenVoice2: flow layer %d missing key weights, skipping", flow);
      continue;
    }

    // Split into x_a (first half) and x_b (second half)
    struct ggml_tensor *x_a = ggml_view_2d(ctx0, cur, half, cur->ne[1],
                                            cur->nb[1], 0);
    struct ggml_tensor *x_b = ggml_view_2d(ctx0, cur, C_flow - half,
                                            cur->ne[1], cur->nb[1],
                                            half * sizeof(float));

    if (is_chinese_flow) {
      // Chinese model: pre → transformer enc → post → additive coupling
      int n_heads = hparams_.n_heads;
      int head_dim = C_flow / n_heads;

      // Pre conv: [half, T] → [C_flow, T]
      struct ggml_tensor *h;
      if (pre_w->ne[1] >= half) {
        h = conv1d_im2col(ctx0, x_a, pre_w, pre_b, 1, 0, half, C_flow);
      } else {
        h = ggml_mul_mat(ctx0, pre_w, x_a);
        if (pre_b) h = ggml_add(ctx0, h, pre_b);
      }

      // Transformer encoder layers
      for (int l = 0; l < n_enc_layers; l++) {
        auto& ew = enc_layers[l];
        if (!ew.norm1_w || !ew.q_w) continue;
        struct ggml_tensor *res = h;

        // Self-attention
        h = layer_norm(ctx0, h, ew.norm1_w, ew.norm1_b, 1e-5f);
        struct ggml_tensor *attn = multi_head_attention(
            ctx0, h, ew.q_w, ew.q_b, ew.k_w, ew.k_b,
            ew.v_w, ew.v_b, ew.o_w, ew.o_b,
            nullptr, nullptr, n_heads, head_dim, h->ne[1]);  // temp disable
        h = ggml_add(ctx0, attn, res);

        // FFN
        if (ew.norm2_w && ew.conv1_w && ew.conv2_w) {
          res = h;
          h = layer_norm(ctx0, h, ew.norm2_w, ew.norm2_b, 1e-5f);
          int k_sz = ew.conv1_w->ne[0] / ew.conv1_w->ne[1];
          if (k_sz <= 0) k_sz = 5;
          h = conv1d_im2col(ctx0, h, ew.conv1_w, ew.conv1_b, k_sz, k_sz/2,
                            ew.conv1_w->ne[1], ew.conv1_w->ne[2]);
          h = ggml_gelu(ctx0, h);
          k_sz = ew.conv2_w->ne[0] / ew.conv2_w->ne[1];
          if (k_sz <= 0) k_sz = 5;
          h = conv1d_im2col(ctx0, h, ew.conv2_w, ew.conv2_b, k_sz, k_sz/2,
                            ew.conv2_w->ne[1], ew.conv2_w->ne[2]);
          h = ggml_add(ctx0, h, res);
        }
      }

      // Post conv: [C_flow, T] → [half, T]
      if (post_w->ne[1] >= C_flow) {
        h = conv1d_im2col(ctx0, h, post_w, post_b, 1, 0, C_flow, half);
      } else {
        h = ggml_mul_mat(ctx0, post_w, h);
        if (post_b) h = ggml_add(ctx0, h, post_b);
      }

      // Additive coupling: x_b' = x_b + h
      x_b = ggml_add(ctx0, x_b, h);

    } else {
      // English model: WaveNet-style coupling (original code)
      struct ggml_tensor *h = ggml_mul_mat(ctx0, in_l0_w, x_a);
      int h_half_w = h->ne[0] / 2;
      struct ggml_tensor *h1 = ggml_view_2d(ctx0, h, h_half_w, h->ne[1], h->nb[1], 0);
      struct ggml_tensor *h2 = ggml_view_2d(
          ctx0, h, h->ne[0] - h_half_w, h->ne[1], h->nb[1],
          h_half_w * sizeof(float));
      h = ggml_mul(ctx0, ggml_tanh(ctx0, h1), ggml_sigmoid(ctx0, h2));
      if (in_l1_w) h = ggml_mul_mat(ctx0, in_l1_w, h);
      h = ggml_mul_mat(ctx0, in_l2_w, h);

      int out_half_w = h->ne[0] / 2;
      struct ggml_tensor *mean_w = ggml_view_2d(ctx0, h, out_half_w, h->ne[1], h->nb[1], 0);
      struct ggml_tensor *log_scale_w = ggml_view_2d(
          ctx0, h, h->ne[0] - out_half_w, h->ne[1], h->nb[1],
          out_half_w * sizeof(float));

      x_b = ggml_sub(ctx0, x_b, mean_w);
      x_b = ggml_mul(ctx0, x_b, ggml_exp(ctx0, ggml_neg(ctx0, log_scale_w)));
    }

    // Concat x_a and transformed x_b
    cur = ggml_concat(ctx0, x_a, x_b, 0);
  }

  // Project to n_mels if output channels don't match
  if (cur->ne[0] != n_mels) {
    // Find projection layer
    struct ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    for (auto& [name, t] : w) {
      if (name.find("proj.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) proj_w = t; else proj_b = t;
      } else if (name.find("out_conv.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) proj_w = t; else proj_b = t;
      }
    }

    if (proj_w) {
      cur = ggml_mul_mat(ctx0, proj_w, cur);
      if (proj_b) cur = ggml_add(ctx0, cur, proj_b);
    }
    // else: just leave as-is
  }

  ggml_set_name(cur, "mel_spectrogram");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // --- Execute ---
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: FlowDecoder graph allocation failed");
    ggml_free(ctx0);
    return false;
  }


  // Upload input data after graph allocation
  ggml_backend_tensor_set(expanded, exp_data.data(), 0,
                          exp_data.size() * sizeof(float));

  // Flush relative-position index/mask tensors registered by multi_head_attention
  flush_pending_inputs();

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read output mel spectrogram ---
  int out_ch = cur->ne[0];
  int out_t = cur->ne[1];
  state.mel_spectrogram.resize(out_ch * out_t);
  ggml_backend_tensor_get(cur, state.mel_spectrogram.data(), 0,
                          state.mel_spectrogram.size() * sizeof(float));
  state.total_mel_frames = out_t;

  ggml_free(ctx0);
  RS_LOG_INFO("OpenVoice2: FlowDecoder -> mel [%d, %d]", out_ch, out_t);
  return true;
}

// =====================================================================
// Sub-graph: HiFi-GAN Vocoder
//
// Architecture:
//   1. Conv1D pre-conv (mel → hidden)
//   2. N upsampling blocks: LeakyReLU → TransposedConv → MRF(resblocks)
//   3. Conv1D post-conv (hidden → 1) → tanh
// =====================================================================

bool OpenVoice2Model::RunVocoder(OpenVoice2State& state,
                                  ggml_backend_sched_t sched,
                                  int mel_start, int mel_len) {
  auto& w = weights_.vocoder;
  int n_mels = hparams_.n_mels;
  int T_mel = state.total_mel_frames;
  int hop = hparams_.hop_length;

  if (w.empty()) {
    RS_LOG_WARN("OpenVoice2: no vocoder weights, generating placeholder audio");
    // Fallback: simple oscillator from mel values
    int n_samples = mel_len * hop;
    size_t prev = state.audio_output.size();
    state.audio_output.resize(prev + n_samples, 0.0f);
    for (int i = 0; i < n_samples; i++) {
      int mf = mel_start + i / hop;
      if (mf >= T_mel) break;
      float val = 0.0f;
      for (int m = 0; m < std::min(8, n_mels); m++) {
        val += state.mel_spectrogram[m + mf * n_mels];
      }
      state.audio_output[prev + i] = val * sinf(2.0f * 3.14159f * 440.0f *
          static_cast<float>(i % hop) / hop) * 0.01f;
    }
    return true;
  }

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for Vocoder");
    return false;
  }

  // Input: mel chunk [n_mels, mel_len]
  struct ggml_tensor *mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, mel_len);
  ggml_set_name(mel, "mel_chunk");
  ggml_set_input(mel);

  // Copy mel data
  std::vector<float> mel_data(n_mels * mel_len, 0.0f);
  for (int i = 0; i < mel_len; i++) {
    int frame = mel_start + i;
    if (frame < T_mel) {
      for (int m = 0; m < n_mels; m++) {
        mel_data[m + i * n_mels] = state.mel_spectrogram[m + frame * n_mels];
      }
    }
  }
  // Mel data will be uploaded after graph allocation
  struct ggml_tensor *cur = mel;

  // Find weights
  struct ggml_tensor *pre_w = nullptr, *pre_b = nullptr;
  struct ggml_tensor *post_w = nullptr, *post_b = nullptr;

  // Upsampler weights: up_N.weight, up_N.bias
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> up_weights;

  for (auto& [name, t] : w) {
    if (name.find("conv_pre.") != std::string::npos) {
      if (name.find("weight_v") != std::string::npos || name.find("weight") != std::string::npos)
        pre_w = t; else pre_b = t;
    } else if (name.find("conv_post.") != std::string::npos) {
      if (name.find("weight_v") != std::string::npos || name.find("weight") != std::string::npos)
        post_w = t; else post_b = t;
    }
  }

  // Pre-conv: mel → hidden
  if (pre_w) {
    int k_pre = static_cast<int>(pre_w->ne[0]);  // kernel size
    int in_pre = static_cast<int>(pre_w->ne[1]);
    int out_pre = static_cast<int>(pre_w->ne[2]);
    cur = conv1d_im2col(ctx0, cur, pre_w, pre_b, k_pre, k_pre / 2, in_pre, out_pre);
  }

  // --- Upsampling blocks ---
  // Count upsamplers
  int n_ups = 0;
  for (int i = 0; i < 8; i++) {
    std::string pref = "vocoder.ups." + std::to_string(i);
    bool found = false;
    for (auto& [name, t] : w) {
      if (name.find(pref + ".") == 0) { found = true; break; }
    }
    if (found) n_ups++;
    else break;
  }

  // Build sorted list of ALL residual blocks by global index.
  // HiFi-GAN resblocks are globally indexed (0, 1, 2, ...) and grouped by
  // channel count to match each upsampler stage — NOT per-stage suffixes.
  struct ResBlockWeights {
    struct ggml_tensor *c1_w, *c1_g, *c1_b;
    struct ggml_tensor *c2_w, *c2_g, *c2_b;
  };
  std::vector<ResBlockWeights> sorted_resblocks;

  for (int res = 0; res < 30; res++) {
    struct ggml_tensor *c1_w = nullptr, *c1_g = nullptr, *c1_b = nullptr;
    struct ggml_tensor *c2_w = nullptr, *c2_g = nullptr, *c2_b = nullptr;
    std::string res_prefix = "vocoder.resblocks." + std::to_string(res) + ".";

    bool has_any = false;
    for (auto& [name, t] : w) {
      if (name.find(res_prefix) != 0) continue;
      has_any = true;

      // Determine tensor type: weight_v, weight_g, plain weight, or bias
      bool is_wv = (name.find("weight_v") != std::string::npos);
      bool is_wg = (name.find("weight_g") != std::string::npos);
      bool is_plain_w = (!is_wv && !is_wg && name.find("weight") != std::string::npos);
      bool is_bias = (name.find("bias") != std::string::npos);

      if (name.find(".convs1.") != std::string::npos) {
        if (is_wv || is_plain_w) c1_w = t;
        else if (is_wg) c1_g = t;
        else if (is_bias) c1_b = t;
      } else if (name.find(".convs2.") != std::string::npos) {
        if (is_wv || is_plain_w) c2_w = t;
        else if (is_wg) c2_g = t;
        else if (is_bias) c2_b = t;
      }
    }

    if (!has_any) break;
    sorted_resblocks.push_back({c1_w, c1_g, c1_b, c2_w, c2_g, c2_b});
  }

  // Compute per-stage upsample rates whose product = hop_length
  std::vector<int> upsample_rates(n_ups, 2);
  {
    int remaining = hop;
    for (int i = 0; i < n_ups; i++) {
      if (i == n_ups - 1) {
        upsample_rates[i] = remaining;
      } else {
        int rate = 8;
        int min_remain = 1;
        for (int j = 0; j < (n_ups - i - 1); j++) min_remain *= 2;
        while (rate > 2) {
          if (remaining % rate == 0 && (remaining / rate) >= min_remain) break;
          rate /= 2;
        }
        upsample_rates[i] = rate;
        remaining /= rate;
      }
    }
  }

  size_t res_idx = 0;  // pointer into sorted_resblocks
  for (int up = 0; up < n_ups; up++) {
    int upsample_rate = upsample_rates[up];

    // Find upsampler weight (Chinese: weight_v + weight_g, English: plain weight)
    struct ggml_tensor *up_w = nullptr, *up_g = nullptr, *up_b = nullptr;
    std::string up_pref = "vocoder.ups." + std::to_string(up);
    for (auto& [name, t] : w) {
      if (name.find(up_pref + ".weight_v") != std::string::npos) up_w = t;
      else if (name.find(up_pref + ".weight_g") != std::string::npos) up_g = t;
      else if (name.find(up_pref + ".weight") != std::string::npos) up_w = t;
      else if (name.find(up_pref + ".bias") != std::string::npos) up_b = t;
    }

    if (up_w) {
      // Apply weight normalization if weight_g is present
      struct ggml_tensor *eff_w = up_w;
      if (up_g) {
        eff_w = apply_weight_norm(ctx0, up_w, up_g);
      }
      // ggml_conv_transpose_1d expects data [OW, IC, N] — transpose cur
      struct ggml_tensor *cur_t = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      struct ggml_tensor *up_out = ggml_conv_transpose_1d(
          ctx0, eff_w, cur_t,
          upsample_rate,  // stride
          0,               // padding (must be 0 for this ggml version)
          1);              // dilation

      // Result: [OL, OC, N] — transpose back to [OC, OL]
      up_out = ggml_cont(ctx0, ggml_transpose(ctx0, up_out));
      int out_ch = static_cast<int>(up_out->ne[0]);
      int out_len = static_cast<int>(up_out->ne[1]);
      up_out = ggml_reshape_2d(ctx0, up_out, out_ch, out_len);

      // Add bias after transposed conv
      if (up_b) {
        struct ggml_tensor *bias_2d = ggml_reshape_2d(ctx0, up_b, out_ch, 1);
        up_out = ggml_add(ctx0, up_out, bias_2d);
      }

      cur = up_out;
    } else {
      RS_LOG_WARN("OpenVoice2: upsampler %d not found, skipping", up);
      continue;
    }

    // MRF residual blocks: apply blocks with matching channel count
    int cur_ch = static_cast<int>(cur->ne[0]);
    while (res_idx < sorted_resblocks.size()) {
      auto& rb = sorted_resblocks[res_idx];
      if (!rb.c1_w || !rb.c2_w) { res_idx++; continue; }
      if (rb.c1_w->ne[1] != cur_ch) break;  // different stage group

      struct ggml_tensor *residual = cur;
      int in_ch = static_cast<int>(rb.c1_w->ne[1]);
      int out_ch_rb = static_cast<int>(rb.c1_w->ne[2]);
      int k_size = static_cast<int>(rb.c1_w->ne[0]);

      // Apply weight norm if weight_g present
      struct ggml_tensor *eff_c1 = rb.c1_w;
      struct ggml_tensor *eff_c2 = rb.c2_w;
      if (rb.c1_g) eff_c1 = apply_weight_norm(ctx0, rb.c1_w, rb.c1_g);
      if (rb.c2_g) eff_c2 = apply_weight_norm(ctx0, rb.c2_w, rb.c2_g);

      // First conv
      cur = conv1d_im2col(ctx0, cur, eff_c1, rb.c1_b,
                          k_size, k_size / 2, in_ch, out_ch_rb);

      cur = ggml_relu(ctx0, cur);  // LeakyReLU approximated

      // Second conv (maintains channel count for residual add)
      in_ch = static_cast<int>(eff_c2->ne[1]);
      out_ch_rb = static_cast<int>(eff_c2->ne[2]);
      k_size = static_cast<int>(eff_c2->ne[0]);
      cur = conv1d_im2col(ctx0, cur, eff_c2, rb.c2_b,
                          k_size, k_size / 2, in_ch, out_ch_rb);

      // Residual connection
      cur = ggml_add(ctx0, cur, residual);
      res_idx++;
    }
  }

  // Post-conv: hidden → 1 (audio)
  if (post_w) {
    cur = ggml_relu(ctx0, cur);  // LeakyReLU
    int k_post = static_cast<int>(post_w->ne[0]);
    int in_post = static_cast<int>(post_w->ne[1]);
    cur = conv1d_im2col(ctx0, cur, post_w, post_b, k_post, k_post / 2, in_post, 1);
    cur = ggml_tanh(ctx0, cur);
  }

  ggml_set_name(cur, "audio_output");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // --- Execute ---
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: Vocoder graph allocation failed");
    ggml_free(ctx0);
    return false;
  }

  // Upload mel data after graph allocation
  ggml_backend_tensor_set(mel, mel_data.data(), 0,
                          mel_data.size() * sizeof(float));

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read audio output ---
  int n_samples = cur->ne[0] * cur->ne[1];
  if (n_samples <= 0) {
    ggml_free(ctx0);
    return false;
  }

  std::vector<float> audio(n_samples);
  ggml_backend_tensor_get(cur, audio.data(), 0, n_samples * sizeof(float));

  // Append to state's audio buffer
  size_t prev_size = state.audio_output.size();
  state.audio_output.resize(prev_size + n_samples);
  std::memcpy(state.audio_output.data() + prev_size, audio.data(),
              n_samples * sizeof(float));

  ggml_free(ctx0);
  int n_audio = n_samples;
  RS_LOG_INFO("OpenVoice2: Vocoder chunk [%d..%d] -> %d samples",
              mel_start, mel_start + mel_len, n_audio);
  return true;
}

// =====================================================================
// Sub-graph: Tone Color Encoder (voice cloning reference encoder)
//
// Architecture: Mel → Conv2D blocks → Global Pooling → Style Embedding
// =====================================================================

bool OpenVoice2Model::RunToneColorEncoder(OpenVoice2State& state,
                                           const std::vector<float>& mel,
                                           ggml_backend_sched_t sched) {
  auto& w = converter_weights_.all_tensors;
  if (w.empty() || !converter_weights_.loaded) {
    RS_LOG_WARN("OpenVoice2: no tone color converter weights loaded");
    state.has_tone_embedding = false;
    return true;
  }

  RS_LOG_INFO("OpenVoice2: ToneColorEncoder not yet implemented (needs converter model)");
  state.has_tone_embedding = false;
  return true;
}

// =====================================================================
// Static registration
// =====================================================================
namespace {
struct OpenVoice2Registrar {
  OpenVoice2Registrar() {
    rs_register_model_arch("openvoice2", []() {
      return std::make_shared<OpenVoice2Model>();
    });
  }
} global_openvoice2_reg;
}  // namespace
