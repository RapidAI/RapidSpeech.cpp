#include "sensevoice.h"
#include "core/rs_context.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstring>
#include <functional>

#define SENSE_VOICE_ENCODER_MAX_NODES 8192
#define SENSE_VOICE_DECODER_MAX_NODES 16

/**
 * SenseVoice internal request state.
 * Manages persistent tensors and backend buffers that need to survive
 * between the Encode and Decode phases.
 */
struct SenseVoiceState : public RSState {
  // Persistent context for tensor metadata (lives as long as the state)
  struct ggml_context * ctx_persistent = nullptr;
  // Persistent buffer for actual tensor data on the backend (e.g., GPU memory)
  ggml_backend_buffer_t buffer_persistent = nullptr;

  // Pointer to the encoder output tensor (metadata in ctx_persistent, data in buffer_persistent)
  struct ggml_tensor * encoder_out = nullptr;

  std::vector<int32_t> ids;
  int language_id = 1; // Default: English (1)
  bool use_itn = false;

  SenseVoiceState() {
    // Initialize a small persistent context for model results metadata
    struct ggml_init_params params = { 128 * ggml_tensor_overhead(), nullptr, true };
    ctx_persistent = ggml_init(params);
  }

  ~SenseVoiceState() {
    if (buffer_persistent) {
      ggml_backend_buffer_free(buffer_persistent);
    }
    if (ctx_persistent) {
      ggml_free(ctx_persistent);
    }
  }
};

// --- Internal Mathematical Helpers ---

/**
 * Matrix multiplication with padding optimization for better performance on Apple Metal.
 */
static struct ggml_tensor * ggml_mul_mat_pad(struct ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y, int pad = 32) {
  const int n_pad_req = 8;
  if (x->ne[0] % pad == 0 || x->ne[0] / pad < n_pad_req) {
    return ggml_mul_mat(ctx, x, y);
  }

  struct ggml_tensor * x_0 = ggml_view_3d(ctx, x, (x->ne[0]/pad)*pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
  struct ggml_tensor * x_1 = ggml_view_3d(ctx, x,  x->ne[0]%pad,      x->ne[1], x->ne[2], x->nb[1], x->nb[2], x_0->ne[0]*x_0->nb[0]);

  struct ggml_tensor * y_0 = ggml_view_3d(ctx, y, (y->ne[0]/pad)*pad, y->ne[1], y->ne[2], y->nb[1], y->nb[2], 0);
  struct ggml_tensor * y_1 = ggml_view_3d(ctx, y,  y->ne[0]%pad,      y->ne[1], y->ne[2], y->nb[1], y->nb[2], y_0->ne[0]*y_0->nb[0]);

  return ggml_add(ctx, ggml_mul_mat(ctx, x_0, y_0), ggml_mul_mat(ctx, x_1, y_1));
}

/**
 * Forward pass for a single SANM (Self-Attention Network with Memory) layer.
 * Implements LayerNorm -> SelfAttention -> FSMN -> Residual -> LayerNorm -> MLP -> Residual.
 */
static struct ggml_tensor * encoder_layer_sanm_forward(
    const SenseVoiceHParams &hparams,
    struct ggml_context * ctx,
    struct ggml_tensor * cur,
    SenseVoiceLayerEncoder &layer) {

  const int n_state = hparams.n_encoder_hidden_state;
  const int n_head = hparams.n_encoder_attention_heads;
  const int n_ctx = cur->ne[1];
  const int n_batch = cur->ne[2];

  struct ggml_tensor * residual = cur;

  // 1. Layer Norm 1
  cur = ggml_norm(ctx, cur, hparams.eps);
  cur = ggml_add(ctx, ggml_mul(ctx, cur, layer.e_norm_w1), layer.e_norm_b1);

  // 2. Self Attention (Linear Projections)
  struct ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat_pad(ctx, ggml_cont(ctx, layer.e_attn_ln_q_w), cur), layer.e_attn_ln_q_b);
  struct ggml_tensor * K = ggml_add(ctx, ggml_mul_mat_pad(ctx, ggml_cont(ctx, layer.e_attn_ln_k_w), cur), layer.e_attn_ln_k_b);
  struct ggml_tensor * V = ggml_add(ctx, ggml_mul_mat_pad(ctx, ggml_cont(ctx, layer.e_attn_ln_v_w), cur), layer.e_attn_ln_v_b);

  // Reshape and Permute for multi-head attention
  struct ggml_tensor * Q_h = ggml_permute(ctx, ggml_reshape_4d(ctx, Q, n_state / n_head, n_head, n_ctx, n_batch), 0, 2, 1, 3);
  struct ggml_tensor * K_h = ggml_permute(ctx, ggml_reshape_4d(ctx, K, n_state / n_head, n_head, n_ctx, n_batch), 0, 2, 1, 3);
  struct ggml_tensor * V_h = ggml_permute(ctx, ggml_reshape_4d(ctx, V, n_state / n_head, n_head, n_ctx, n_batch), 0, 2, 1, 3);

  // Scaled Dot-Product Attention
  float scale = 1.0f / sqrtf(float(n_state) / n_head);
  struct ggml_tensor * KQ = ggml_mul_mat(ctx, K_h, Q_h);
  struct ggml_tensor * KQ_soft_max = ggml_soft_max_ext(ctx, KQ, nullptr, scale, 0.0f);
  struct ggml_tensor * KQV = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, V_h)), KQ_soft_max);

  // Merge heads
  struct ggml_tensor * attn_out = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3)), n_state, n_ctx, n_batch);
  attn_out = ggml_add(ctx, ggml_mul_mat(ctx, layer.e_attn_ln_out_w, attn_out), layer.e_attn_ln_out_b);

  // 3. FSMN Block (Memory Block) using depth-wise convolution logic
  int padding = (hparams.fsmn_kernel_size - 1) / 2; // Kernel size 31
  struct ggml_tensor * fsmn_in = ggml_cont(ctx, ggml_transpose(ctx, V));
  struct ggml_tensor * im2col = ggml_im2col(ctx,
                                           layer.e_attn_fsmn_w,
                                           ggml_reshape_3d(ctx, fsmn_in, fsmn_in->ne[0], fsmn_in->ne[2], fsmn_in->ne[1] * fsmn_in->ne[3]),
                                           1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
  struct ggml_tensor * fsmn_out = ggml_mul_mat(ctx, layer.e_attn_fsmn_w, im2col);
  fsmn_out = ggml_reshape_3d(ctx, fsmn_out, n_state, n_ctx, n_batch);

  attn_out = ggml_add(ctx, attn_out, fsmn_out);


  // 4. Layer Norm 2 + MLP
  cur = ggml_norm(ctx, attn_out, hparams.eps);
  cur = ggml_add(ctx, ggml_mul(ctx, cur, layer.e_norm_w2), layer.e_norm_b2);

  cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.e_mlp_w1, cur), layer.e_mlp_b1);
  cur = ggml_relu(ctx, cur);
  cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.e_mlp_w2, cur), layer.e_mlp_b2);

  return ggml_add(ctx, cur, attn_out);
}

// --- SenseVoiceModel Implementation ---

SenseVoiceModel::SenseVoiceModel() : ctx_weights_(nullptr) {
  encoder_ = std::make_unique<SenseVoiceEncoder>();
}

SenseVoiceModel::~SenseVoiceModel() {
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
    ctx_weights_ = nullptr;
  }
}

bool SenseVoiceModel::Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Invalid context provided to SenseVoiceModel::Load");
    return false;
  }

  gguf_context * ctx_gguf = ctx->ctx_gguf;
  ggml_context * gguf_data = ctx->gguf_data;

  // 1. Load Hyperparameters from GGUF KV
  hparams_.n_vocab = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "tokenizer.vocab_size"));
  hparams_.n_encoder_hidden_state = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.output_size"));
  hparams_.n_encoder_linear_units = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.linear_units"));
  hparams_.n_encoder_attention_heads = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.attention_heads"));
  hparams_.n_encoder_layers = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.num_blocks"));
  hparams_.n_tp_encoder_layers = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.tp_blocks"));
  hparams_.n_mels = 80;

  meta_.arch_name = "SenseVoiceSmall";
  meta_.audio_sample_rate = 16000;
  meta_.n_mels = hparams_.n_mels;
  meta_.vocab_size = hparams_.n_vocab;

  // 2. Load Vocabulary
  const int token_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
  if (token_idx != -1) {
    int n_vocab = gguf_get_arr_n(ctx_gguf, token_idx);
    for (int i = 0; i < n_vocab; i++) {
      vocab_.id_to_token[i] = gguf_get_arr_str(ctx_gguf, token_idx, i);
    }
  }

  // 3. Extract CMVN from GGUF metadata
  std::vector<float> cmvn_means, cmvn_vars;
  load_cmvn_params(ctx_gguf, cmvn_means, cmvn_vars);
  if (ctx->processor) {
    ctx->processor->SetCMVN(cmvn_means, cmvn_vars);
  }

  // 4. Map Tensors from ggml_data
  std::map<std::string, struct ggml_tensor*> tensors;
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; ++i) {
    const char * name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor * t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }

  return MapTensors(tensors);
}

bool SenseVoiceModel::MapTensors(std::map<std::string, struct ggml_tensor*>& tensors) {
  try {
    encoder_->embedding = tensors.at("embed.weight");

    std::vector<SenseVoiceLayerEncoder> tmp_encoder0(1);
    SetLayerWeights(tmp_encoder0, tensors, 1, "encoders0");
    encoder_->encoder0 = tmp_encoder0[0];

    SetLayerWeights(encoder_->encoders_layer, tensors, hparams_.n_encoder_layers - 1, "encoders");
    SetLayerWeights(encoder_->tp_encoders_layer, tensors, hparams_.n_tp_encoder_layers, "tp_encoders");

    encoder_->e_after_norm_w = tensors.at("encoder.after_norm.weight");
    encoder_->e_after_norm_b = tensors.at("encoder.after_norm.bias");
    encoder_->e_tp_norm_w    = tensors.at("encoder.tp_norm.weight");
    encoder_->e_tp_norm_b    = tensors.at("encoder.tp_norm.bias");

    encoder_->ctc_out_linear_weight = tensors.at("ctc.ctc_lo.weight");
    encoder_->ctc_out_linear_bias   = tensors.at("ctc.ctc_lo.bias");

    return true;
  } catch (...) {
    RS_LOG_ERR("Tensor mapping failed for SenseVoice.");
    return false;
  }
}

bool SenseVoiceModel::SetLayerWeights(std::vector<SenseVoiceLayerEncoder>& layers, std::map<std::string, struct ggml_tensor*>& tensors, int n_layers, const std::string& prefix) {
  layers.resize(n_layers);
  for (int i = 0; i < n_layers; ++i) {
    std::string p = "encoder." + prefix + "." + std::to_string(i);
    layers[i].e_attn_ln_out_w = tensors.at(p + ".self_attn.linear_out.weight");
    layers[i].e_attn_ln_out_b = tensors.at(p + ".self_attn.linear_out.bias");
    layers[i].e_attn_ln_q_w   = tensors.at(p + ".self_attn.linear_q.weight");
    layers[i].e_attn_ln_q_b   = tensors.at(p + ".self_attn.linear_q.bias");
    layers[i].e_attn_ln_k_w   = tensors.at(p + ".self_attn.linear_k.weight");
    layers[i].e_attn_ln_k_b   = tensors.at(p + ".self_attn.linear_k.bias");
    layers[i].e_attn_ln_v_w   = tensors.at(p + ".self_attn.linear_v.weight");
    layers[i].e_attn_ln_v_b   = tensors.at(p + ".self_attn.linear_v.bias");
    layers[i].e_attn_fsmn_w   = tensors.at(p + ".self_attn.fsmn_block.weight");
    layers[i].e_mlp_w1        = tensors.at(p + ".feed_forward.w_1.weight");
    layers[i].e_mlp_b1        = tensors.at(p + ".feed_forward.w_1.bias");
    layers[i].e_mlp_w2        = tensors.at(p + ".feed_forward.w_2.weight");
    layers[i].e_mlp_b2        = tensors.at(p + ".feed_forward.w_2.bias");
    layers[i].e_norm_w1       = tensors.at(p + ".norm1.weight");
    layers[i].e_norm_b1       = tensors.at(p + ".norm1.bias");
    layers[i].e_norm_w2       = tensors.at(p + ".norm2.weight");
    layers[i].e_norm_b2       = tensors.at(p + ".norm2.bias");
  }
  return true;
}

std::shared_ptr<RSState> SenseVoiceModel::CreateState() {
  return std::make_shared<SenseVoiceState>();
}

/**
 * Build and execute the Encoder computation graph.
 * 4 Prompt Tokens + Mel Features -> SANM Encoders -> Parallel Encoders -> Hidden States.
 */
bool SenseVoiceModel::Encode(const std::vector<float>& input_frames, RSState& state, ggml_backend_sched_t sched) {
  auto& sv_state = static_cast<SenseVoiceState&>(state);
  int n_len = input_frames.size() / hparams_.n_mels;
  int hidden = hparams_.n_encoder_hidden_state;

  // 1. Initialize transient context for graph nodes
  struct ggml_init_params params = { SENSE_VOICE_ENCODER_MAX_NODES * ggml_tensor_overhead(), nullptr, true };
  struct ggml_context * ctx0 = ggml_init(params);
  struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, SENSE_VOICE_ENCODER_MAX_NODES, false);

  // 2. Define Input Tensors
  struct ggml_tensor * feature = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.feats_dim, n_len);
  ggml_set_name(feature, "feats");
  ggml_set_input(feature);

  struct ggml_tensor * prompt_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 4, 1);
  ggml_set_name(prompt_ids, "embedding"); // Key from reference
  ggml_set_input(prompt_ids);

  struct ggml_tensor * position = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams_.feats_dim, n_len + 4, 1);
  ggml_set_name(position, "position");
  ggml_set_input(position);

  // 3. Process Prompt Tokens (Language, Task, ITN)
  struct ggml_tensor * emb = ggml_get_rows(ctx0, encoder_->embedding, prompt_ids);
  // Expand prompt embedding to match batch size (usually 1)
  emb = ggml_repeat(ctx0, emb, ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, emb->ne[0], emb->ne[1], 1));

  // 4. Concatenate Prompt and Mel features
  struct ggml_tensor * cur = ggml_concat(ctx0, emb, feature, 1);
  cur = ggml_scale(ctx0, cur, sqrtf(hidden));

  // 5. Add Sinusoidal Positional Encoding
  cur = ggml_add(ctx0, position, cur);

  // 6. Forward through SANM layers
  cur = encoder_layer_sanm_forward(hparams_, ctx0, cur, encoder_->encoder0);

  for (int i = 0; i < hparams_.n_encoder_layers - 1; ++i) {
    cur = encoder_layer_sanm_forward(hparams_, ctx0, cur, encoder_->encoders_layer[i]);
  }

  // Encoder Post-Norm
  cur = ggml_norm(ctx0, cur, hparams_.eps);
  cur = ggml_add(ctx0, ggml_mul(ctx0, cur, encoder_->e_after_norm_w), encoder_->e_after_norm_b);

  // Forward through Time-Parallel (TP) layers
  for (int i = 0; i < hparams_.n_tp_encoder_layers; ++i) {
    cur = encoder_layer_sanm_forward(hparams_, ctx0, cur, encoder_->tp_encoders_layer[i]);
  }

  // TP Final Norm
  cur = ggml_norm(ctx0, cur, hparams_.eps);
  cur = ggml_add(ctx0, ggml_mul(ctx0, cur, encoder_->e_tp_norm_w), encoder_->e_tp_norm_b);

  ggml_set_name(cur, "encoder_out");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // 7. Allocation Phase (CRITICAL for multi-backend safety)
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    ggml_free(ctx0);
    return false;
  }

  // 8. Upload data to backend buffers
  ggml_backend_tensor_set(feature, input_frames.data(), 0, input_frames.size() * sizeof(float));

  int p_tokens[4] = { sv_state.language_id, 1, 2, sv_state.use_itn ? 14 : 15 };
  ggml_backend_tensor_set(prompt_ids, p_tokens, 0, 4 * sizeof(int));

  // Dynamic generation of sinusoidal positional encoding
  std::vector<float> _pos((n_len + 4) * hidden);
  for (int k = 1; k <= (n_len + 4); ++k) {
    for (int i = 0; i < hidden / 2; ++i) {
      float val = k * powf(10000.0f, -2.0f * i / hidden);
      _pos[(k - 1) * hidden + i] = sinf(val);
      _pos[(k - 1) * hidden + i + hidden / 2] = cosf(val);
    }
  }
  ggml_backend_tensor_set(position, _pos.data(), 0, _pos.size() * sizeof(float));

  // 9. Execute computation
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    ggml_free(ctx0);
    return false;
  }

  // 10. Persist the results to state managed buffer before freeing graph context
  if (sv_state.encoder_out == nullptr || sv_state.encoder_out->ne[0] != cur->ne[0] || sv_state.encoder_out->ne[1] != cur->ne[1]) {
    if (sv_state.buffer_persistent) ggml_backend_buffer_free(sv_state.buffer_persistent);
    sv_state.encoder_out = ggml_new_tensor_2d(sv_state.ctx_persistent, cur->type, cur->ne[0], cur->ne[1]);
    ggml_backend_t primary_backend = ggml_backend_sched_get_backend(sched, 0);
    sv_state.buffer_persistent = ggml_backend_alloc_buffer(primary_backend, ggml_nbytes(sv_state.encoder_out));
    void * base_addr = ggml_backend_buffer_get_base(sv_state.buffer_persistent);
    ggml_backend_tensor_alloc(sv_state.buffer_persistent, sv_state.encoder_out, base_addr);
  }
  ggml_backend_tensor_copy(cur, sv_state.encoder_out);

  ggml_free(ctx0);
  return true;
}

/**
 * Build and execute the CTC Decoder computation graph.
 * Hidden States -> Linear Projection -> Softmax -> Argmax -> Token IDs.
 */
bool SenseVoiceModel::Decode(RSState& state, ggml_backend_sched_t sched) {
  auto& sv_state = static_cast<SenseVoiceState&>(state);
  if (!sv_state.encoder_out) return false;

  struct ggml_context * ctx0 = ggml_init({ 128 * ggml_tensor_overhead(), nullptr, true });
  struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, SENSE_VOICE_DECODER_MAX_NODES, false);

  // Map decoder input to the shape of the persistent encoder output
  struct ggml_tensor * encoder_in = ggml_new_tensor_2d(ctx0, sv_state.encoder_out->type,
                                                      sv_state.encoder_out->ne[0], sv_state.encoder_out->ne[1]);
  ggml_set_input(encoder_in);

  // Step: Linear projection to vocab size (ctc_lo)
  struct ggml_tensor * cur = ggml_mul_mat(ctx0, encoder_->ctc_out_linear_weight, encoder_in);
  cur = ggml_add(ctx0, cur, encoder_->ctc_out_linear_bias);

  // Step: Argmax over softmax probabilities
  struct ggml_tensor * probs = ggml_soft_max(ctx0, cur);
  struct ggml_tensor * argmax = ggml_argmax(ctx0, probs);

  ggml_set_output(argmax);
  ggml_build_forward_expand(gf, argmax);

  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    ggml_free(ctx0);
    return false;
  }

  // Transfer persistent result to decoder input
  ggml_backend_tensor_copy(sv_state.encoder_out, encoder_in);

  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    ggml_free(ctx0);
    return false;
  }

  // Extract predicted Token IDs from backend to host vector
  struct ggml_tensor * res_node = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
  sv_state.ids.resize(ggml_nelements(res_node));
  ggml_backend_tensor_get(res_node, sv_state.ids.data(), 0, sv_state.ids.size() * sizeof(int32_t));

  ggml_free(ctx0);
  return true;
}

// Registration logic
extern void rs_register_model_arch(const std::string& arch, std::function<std::shared_ptr<ISpeechModel>()> creator);
namespace {
struct SenseVoiceRegistrar {
  SenseVoiceRegistrar() {
    rs_register_model_arch("SenseVoiceSmall", [](){ return std::make_shared<SenseVoiceModel>(); });
  }
} global_sensevoice_reg;
}