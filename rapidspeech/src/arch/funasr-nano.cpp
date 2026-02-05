#include "funasr-nano.h"
#include "ctc_decoder.h"
#include "core/rs_context.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <functional>
#include "gguf.h"
#include "utils/debug_utils.h"
#include "utils/rs_wav.h"

// Increased node limit to handle deep FunASRNano graphs (50+ layers)
#define FUNASR_NANO_ENCODER_MAX_NODES 6144
#define FUNASR_NANO_DECODER_MAX_NODES 128

struct FunASRNanoState : public RSState {
  struct ggml_context * ctx_persistent = nullptr;
  ggml_backend_buffer_t buffer_persistent = nullptr;
  struct ggml_tensor * encoder_out = nullptr;

  std::vector<int32_t> ids;
  std::vector<std::string> tokens;
  int language_id = 0;
  bool use_itn = true;

  FunASRNanoState() {
    // Increase persistent context to ensure enough room for tensor metadata
    struct ggml_init_params params = { 512 * ggml_tensor_overhead(), nullptr, true };
    ctx_persistent = ggml_init(params);
  }

  ~FunASRNanoState() {
    if (buffer_persistent) ggml_backend_buffer_free(buffer_persistent);
    if (ctx_persistent) ggml_free(ctx_persistent);
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
    const FunASRNanoHParams &hparams,
    struct ggml_context * ctx,
    struct ggml_tensor * cur,
    FunASRNanoLayerEncoder &layer) {

  const int n_state = hparams.n_encoder_hidden_state;
  const int n_head = hparams.n_encoder_attention_heads;
  const int n_ctx = cur->ne[1];
  const int n_batch = cur->ne[2];

  struct ggml_tensor * residual = cur;
  if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
    residual = ggml_cpy(
            ctx, cur,
            ggml_new_tensor_3d(ctx, cur->type, cur->ne[0], cur->ne[1], cur->ne[2]));
  }
  // 1. Layer Norm 1
  cur = ggml_norm(ctx, cur, hparams.eps);
  cur = ggml_add(ctx, ggml_mul(ctx, cur, layer.e_norm_w1), layer.e_norm_b1);

  // 2. Self Attention (Linear Projections)
  struct ggml_tensor * Q = ggml_add(ctx, ggml_mul_mat_pad(ctx, ggml_cont(ctx, layer.e_attn_ln_q_w), cur), layer.e_attn_ln_q_b);
  struct ggml_tensor * K = ggml_add(ctx, ggml_mul_mat_pad(ctx, ggml_cont(ctx, layer.e_attn_ln_k_w), cur), layer.e_attn_ln_k_b);
  struct ggml_tensor * V = ggml_add(ctx, ggml_mul_mat_pad(ctx, layer.e_attn_ln_v_w, cur), layer.e_attn_ln_v_b);

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
  ggml_set_name(attn_out, "attn_out");

  // 3. FSMN Block (Memory Block) using depth-wise convolution logic
  int padding = (hparams.fsmn_kernel_size - 1) / 2; // Kernel size 31
  struct ggml_tensor * fsmn_in = ggml_cont(ctx, ggml_transpose(ctx, V));
  struct ggml_tensor * im2col = ggml_im2col(ctx,
                                           layer.e_attn_fsmn_w,
                                           ggml_reshape_4d(ctx, fsmn_in, fsmn_in->ne[0], 1,  fsmn_in->ne[1], fsmn_in->ne[2] * fsmn_in->ne[3]),
                                           1, 0, padding, 0, 1, 0, false, GGML_TYPE_F32);
  struct ggml_tensor * fsmn_out = ggml_mul_mat(ctx, layer.e_attn_fsmn_w, im2col);
  fsmn_out = ggml_reshape_3d(ctx, fsmn_out, im2col->ne[1], im2col->ne[2], im2col->ne[3]);
  fsmn_out = ggml_cont(ctx, ggml_transpose(ctx, fsmn_out));
  fsmn_out = ggml_add(ctx, fsmn_out, V);
  attn_out = ggml_add(ctx, fsmn_out, attn_out);

  if (layer.e_norm_w1->ne[0] == layer.e_norm_w2->ne[0]) {
    attn_out = ggml_add(ctx, attn_out, residual);
  }


  // 4. Layer Norm 2 + MLP
  cur = ggml_norm(ctx, attn_out, hparams.eps);
  cur = ggml_add(ctx, ggml_mul(ctx, cur, layer.e_norm_w2), layer.e_norm_b2);

  cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.e_mlp_w1, cur), layer.e_mlp_b1);
  cur = ggml_relu(ctx, cur);
  cur = ggml_add(ctx, ggml_mul_mat(ctx, layer.e_mlp_w2, cur), layer.e_mlp_b2);

  return ggml_add(ctx, cur, attn_out);
}

/**
 * Helper to safely initialize a ggml context and graph.
 * Prevents 0x0 crashes by checking allocation results.
 */
static bool init_compute_ctx(struct ggml_context ** ctx, struct ggml_cgraph ** gf, int n_nodes) {
  // We add 1MB of buffer to the tensor overhead to be safe
  size_t mem_size = n_nodes * ggml_tensor_overhead() + (1024 * 1024);
  struct ggml_init_params params = { mem_size, nullptr, true };
  *ctx = ggml_init(params);
  if (!(*ctx)) {
    RS_LOG_ERR("ggml_init failed: out of memory for context.");
    return false;
  }
  *gf = ggml_new_graph_custom(*ctx, n_nodes, false);
  if (!(*gf)) {
    RS_LOG_ERR("ggml_new_graph_custom failed: too many nodes or out of memory.");
    return false;
  }
  return true;
}

// --- FunASRNanoModel Implementation ---

FunASRNanoModel::FunASRNanoModel() : ctx_weights_(nullptr) {
  encoder_ = std::make_unique<FunASRNanoEncoder>();
}

FunASRNanoModel::~FunASRNanoModel() {
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
    ctx_weights_ = nullptr;
  }
}

bool FunASRNanoModel::Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Invalid context provided to FunASRNanoModel::Load");
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

  meta_.arch_name = "FunASRNanoSmall";
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

bool FunASRNanoModel::MapTensors(std::map<std::string, struct ggml_tensor*>& tensors) {
  try {
    encoder_->embedding = tensors.at("embed.weight");
    std::vector<FunASRNanoLayerEncoder> tmp_encoder0(1);
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
    RS_LOG_ERR("Tensor mapping failed for FunASRNano.");
    return false;
  }
}

bool FunASRNanoModel::SetLayerWeights(std::vector<FunASRNanoLayerEncoder>& layers, std::map<std::string, struct ggml_tensor*>& tensors, int n_layers, const std::string& prefix) {
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

std::shared_ptr<RSState> FunASRNanoModel::CreateState() {
  return std::make_shared<FunASRNanoState>();
}

/**
 * Build and execute the Encoder computation graph.
 * 4 Prompt Tokens + Mel Features -> SANM Encoders -> Parallel Encoders -> Hidden States.
 */
bool FunASRNanoModel::Encode(const std::vector<float>& input_frames, RSState& state, ggml_backend_sched_t sched) {
  auto& sv_state = static_cast<FunASRNanoState&>(state);
  int n_len = input_frames.size() / hparams_.feats_dim;
  int hidden = hparams_.n_encoder_hidden_state;

  struct ggml_context * ctx0 = nullptr;
  struct ggml_cgraph * gf = nullptr;

  if (!init_compute_ctx(&ctx0, &gf, FUNASR_NANO_ENCODER_MAX_NODES)) {
      return false;
  }

  // Define Input Tensors
  struct ggml_tensor * feature = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams_.feats_dim, n_len);
  struct ggml_tensor * prompt_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 4, 1);
  struct ggml_tensor * position = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hparams_.feats_dim, n_len + 4, 1);

  ggml_set_name(feature, "feats");
  ggml_set_name(prompt_ids, "prompt_ids");
  ggml_set_name(position, "position");

  ggml_set_input(feature);
  ggml_set_input(prompt_ids);
  ggml_set_input(position);

  // Graph construction logic...
  struct ggml_tensor * emb = ggml_get_rows(ctx0, encoder_->embedding, prompt_ids);
  ggml_set_name(emb, "embedding");
  emb = ggml_repeat(ctx0, emb, ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, emb->ne[0], emb->ne[1], 1));
  struct ggml_tensor * cur = ggml_concat(ctx0, emb, feature, 1);
  cur = ggml_scale(ctx0, cur, sqrtf(hidden));
  cur = ggml_add(ctx0, position, cur);

  cur = encoder_layer_sanm_forward(hparams_, ctx0, cur, encoder_->encoder0);
  for (int i = 0; i < hparams_.n_encoder_layers - 1; ++i) {
    cur = encoder_layer_sanm_forward(hparams_, ctx0, cur, encoder_->encoders_layer[i]);
  }

  cur = ggml_norm(ctx0, cur, hparams_.eps);
  cur = ggml_add(ctx0, ggml_mul(ctx0, cur, encoder_->e_after_norm_w), encoder_->e_after_norm_b);

  for (int i = 0; i < hparams_.n_tp_encoder_layers; ++i) {
    cur = encoder_layer_sanm_forward(hparams_, ctx0, cur, encoder_->tp_encoders_layer[i]);
  }

  cur = ggml_norm(ctx0, cur, hparams_.eps);
  cur = ggml_add(ctx0, ggml_mul(ctx0, cur, encoder_->e_tp_norm_w), encoder_->e_tp_norm_b);

  ggml_set_name(cur, "encoder_out");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // Allocate and execute
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    // If this is the first run, GGML might fail to allocate while it calculates reserves.
    // We check logs; if it still crashes here, the node count or memory is definitely the issue.
    ggml_free(ctx0);
    return false;
  }

  // Set data
  ggml_backend_tensor_set(feature, input_frames.data(), 0, input_frames.size() * sizeof(float));
  int p_tokens[4] = { sv_state.language_id, 1, 2, sv_state.use_itn ? 14 : 15 };
  ggml_backend_tensor_set(prompt_ids, p_tokens, 0, 4 * sizeof(int));

  // --- POSITIONAL ENCODING FIX ---
  // Construct sinusoidal position embedding based on FunASR reference
  auto p_n_len = position->ne[1];
  auto p_dim = position->ne[0];
  auto p_n_batch = position->ne[2];
  std::vector<float> _position(p_n_len * p_dim * p_n_batch);

  for (int b = 0; b < p_n_batch; b++) {
    for (int k = 1; k <= p_n_len; k++) {
      for (int i = 0; i < p_dim / 2; i++) {
        float freq = pow(10000, -2.0 * i / p_dim);
        _position[b * p_n_len * p_dim + (k - 1) * p_dim + i] = sinf(k * freq);
        _position[b * p_n_len * p_dim + (k - 1) * p_dim + i + p_dim / 2] = cosf(k * freq);
      }
    }
  }

  ggml_backend_tensor_set(position, _position.data(), 0, ggml_nelements(position) * sizeof(float));

  // ggml_backend_sched_set_eval_callback(sched, ggml_debug, new callback_data());
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    ggml_free(ctx0);
    return false;
  }

  // Persistence logic...
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
 * Enhanced Decode function supporting Greedy and Beam Search.
 */
bool FunASRNanoModel::Decode(RSState& state, ggml_backend_sched_t sched) {
    auto& sv_state = static_cast<FunASRNanoState&>(state);
    if (!sv_state.encoder_out) return false;

    int T = sv_state.encoder_out->ne[1];
    int V = hparams_.n_vocab;
    int beam_size = 1; // You can pull this from params later

    struct ggml_context * ctx0 = ggml_init({ 256 * ggml_tensor_overhead(), nullptr, true });
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, FUNASR_NANO_DECODER_MAX_NODES, false);

    struct ggml_tensor * encoder_in = ggml_new_tensor_2d(ctx0, sv_state.encoder_out->type,
                                                        sv_state.encoder_out->ne[0], sv_state.encoder_out->ne[1]);
    ggml_set_input(encoder_in);

    // 1. Linear projection to vocab size
    struct ggml_tensor * cur = ggml_mul_mat(ctx0, encoder_->ctc_out_linear_weight, encoder_in);
    cur = ggml_add(ctx0, cur, encoder_->ctc_out_linear_bias);

    // 2. Compute log-probabilities
    struct ggml_tensor * log_probs =  ggml_log(ctx0, ggml_soft_max(ctx0, cur));

    struct ggml_tensor * output_node = nullptr;

    if (beam_size <= 1) {
        // Greedy serach: Calculate argmax on backend
        output_node = ggml_argmax(ctx0, log_probs);
    } else {
        // Beam Search Mode: We need the full log-probs on host
        output_node = log_probs;
    }
    ggml_set_name(output_node, "output");

    ggml_set_output(output_node);
    ggml_build_forward_expand(gf, output_node);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_copy(sv_state.encoder_out, encoder_in);

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx0);
        return false;
    }


    // 3. Post-Processing on Host
    if (beam_size <= 1) {
        // --- Greedy Decoding ---
        std::vector<int32_t> raw_ids(T);
        ggml_backend_tensor_get(output_node, raw_ids.data(), 0, T * sizeof(int32_t));

        // Use CTCDecoder to collapse repeats and remove blanks
        sv_state.ids = CTCDecoder::GreedyDecode(raw_ids.data(), T);
    } else {
        // --- Beam Search Decoding ---
        std::vector<float> host_log_probs(T * V);
        ggml_backend_tensor_get(output_node, host_log_probs.data(), 0, T * V * sizeof(float));

        sv_state.ids = CTCDecoder::BeamSearchDecode(host_log_probs.data(), T, V, beam_size);
    }

    for (auto id: sv_state.ids){ sv_state.tokens.push_back(this->vocab_.id_to_token[id]);}

    ggml_free(ctx0);
    return true;
}

std:: string FunASRNanoModel::GetTranscription(RSState& state)
{
  auto& sv_state = static_cast<FunASRNanoState&>(state);
  std::string result;
  result.reserve(64);   // ⭐ 关键：避免反复 realloc

  for (const auto& s : sv_state.tokens) {
    result += s;
  }
  sv_state.tokens.clear();
  return result;
}
// Registration logic
extern void rs_register_model_arch(const std::string& arch, std::function<std::shared_ptr<ISpeechModel>()> creator);
namespace {
struct FunASRNanoRegistrar {
  FunASRNanoRegistrar() {
    rs_register_model_arch("FunASRNano", [](){ return std::make_shared<FunASRNanoModel>(); });
  }
} global_FunASRNano_reg;
}