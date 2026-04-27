#include "funasr-nano.h"
#include "core/rs_context.h"
#include "ctc_decoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "sensevoice.h"
#include "utils/debug_utils.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"
#include <functional>

#include "ggml-cpu.h"

// Increased node limit to handle deep FunASRNano graphs (50+ layers)
#define FUNASR_NANO_ENCODER_MAX_NODES 6144
#define FUNASR_NANO_DECODER_MAX_NODES 1024

struct FunASRNanoState : public RSState {
  struct ggml_context *ctx_persistent = nullptr;
  ggml_backend_buffer_t buffer_persistent = nullptr;
  struct ggml_tensor *encoder_out = nullptr;

  std::vector<int32_t> ids;
  std::vector<std::string> tokens;
  std::string user_input =
      "语音转写："; // User input prompt, default is "语音转写"
  int language_id = 0;

  FunASRNanoState() {
    // Increase persistent context to ensure enough room for tensor metadata
    struct ggml_init_params params = {512 * ggml_tensor_overhead(), nullptr,
                                      true};
    ctx_persistent = ggml_init(params);
  }

  ~FunASRNanoState() {
    if (buffer_persistent)
      ggml_backend_buffer_free(buffer_persistent);
    if (ctx_persistent)
      ggml_free(ctx_persistent);
  }
};

// --- FunASRNanoModel Implementation ---

FunASRNanoModel::FunASRNanoModel() : ctx_weights_(nullptr) {
  encoder_ = std::make_unique<SenseVoiceEncoderModel>();
  ctc_decoder_ = std::make_unique<FunASRNanoTransformerDecoder>();
  audio_adaptor_ = std::make_unique<FunASRNanoTransformerDecoder>();
}

FunASRNanoModel::~FunASRNanoModel() {
  if (ctx_weights_) {
    ggml_free(ctx_weights_);
    ctx_weights_ = nullptr;
  }
}

bool FunASRNanoModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                           ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Invalid context provided to FunASRNanoModel::Load");
    return false;
  }

  gguf_context *ctx_gguf = ctx->ctx_gguf;
  ggml_context *gguf_data = ctx->gguf_data;

  // 1. Load Hyperparameters from GGUF KV
  hparams_.n_vocab = gguf_get_val_i32(
      ctx_gguf, gguf_find_key(ctx_gguf, "tokenizer.vocab_size"));
  hparams_.n_encoder_hidden_state = gguf_get_val_i32(
      ctx_gguf, gguf_find_key(ctx_gguf, "encoder.output_size"));
  hparams_.n_encoder_linear_units = gguf_get_val_i32(
      ctx_gguf, gguf_find_key(ctx_gguf, "encoder.linear_units"));
  hparams_.n_encoder_attention_heads = gguf_get_val_i32(
      ctx_gguf, gguf_find_key(ctx_gguf, "encoder.attention_heads"));
  hparams_.n_encoder_layers =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.num_blocks"));
  hparams_.n_tp_encoder_layers =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "encoder.tp_blocks"));
  hparams_.n_ctc_layers =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ctc.n_layer"));
  hparams_.ctc_downsample_rate = gguf_get_val_i32(
      ctx_gguf, gguf_find_key(ctx_gguf, "ctc.downsample_rate"));
  hparams_.ctc_encoder_dim =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ctc.encoder_dim"));
  hparams_.ctc_llm_dim =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ctc.llm_dim"));
  hparams_.ctc_ffn_dim =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "ctc.ffn_dim"));
  hparams_.n_mels =
      gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf, "frontend.num_mels"));

  // Check if LLM is enabled - check for qwen3.block_count as indicator
  int qwen3_block_count_idx = gguf_find_key(ctx_gguf, "qwen3.block_count");
  if (qwen3_block_count_idx != -1) {
    hparams_.use_llm = true;
  }

  // Load adaptor layers if available
  int adaptor_n_layer_idx = gguf_find_key(ctx_gguf, "adaptor.n_layer");
  if (adaptor_n_layer_idx != -1) {
    hparams_.n_adaptor_layers = gguf_get_val_i32(ctx_gguf, adaptor_n_layer_idx);
  }

  if (hparams_.use_llm) {
    hparams_.n_llm_layer = gguf_get_val_i32(ctx_gguf, qwen3_block_count_idx);
    hparams_.n_llm_embd = gguf_get_val_i32(
        ctx_gguf, gguf_find_key(ctx_gguf, "qwen3.embedding_length"));
    hparams_.n_llm_head = gguf_get_val_i32(
        ctx_gguf, gguf_find_key(ctx_gguf, "qwen3.attention.head_count"));
    hparams_.head_dim = gguf_get_val_i32(
        ctx_gguf, gguf_find_key(ctx_gguf, "qwen3.attention.key_length"));
    // hparams_.n_llm_vocab = gguf_get_val_i32(ctx_gguf, gguf_find_key(ctx_gguf,
    // "llm.vocab_size")); hparams_.f_llm_rope_freq_base =
    // gguf_get_val_f32(ctx_gguf, gguf_find_key(ctx_gguf,
    // "qwen3.rope.freq_base"));

    RS_LOG_INFO("LLM enabled: layers=%d, embd=%d, heads=%d, vocab=%d",
                hparams_.n_llm_layer, hparams_.n_llm_embd, hparams_.n_llm_head,
                hparams_.n_llm_vocab);
  }

  meta_.arch_name = "FunASRNano";
  meta_.audio_sample_rate = gguf_get_val_i32(
      ctx_gguf, gguf_find_key(ctx_gguf, "frontend.sample_rate"));
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
  std::map<std::string, struct ggml_tensor *> tensors;
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t)
      tensors[name] = t;
  }

  bool success = MapTensors(tensors);

  // Load LLM if enabled
  if (success && hparams_.use_llm) {
    success = LoadLLM(ctx_gguf, tensors, backend);
  }

  return success;
}

bool FunASRNanoModel::LoadLLM(
    struct gguf_context *ctx_gguf,
    std::map<std::string, struct ggml_tensor *> &tensors,
    ggml_backend_t backend) {
  if (!hparams_.use_llm) {
    return false;
  }

  RS_LOG_INFO("Loading Qwen3 LLM from GGUF context");

  // 1. Create LLM model instance
  llm_model_ = std::make_shared<llm_model>();
  if (!llm_model_) {
    RS_LOG_ERR("Failed to create LLM model instance");
    return false;
  }

  // 2. Load metadata (hyperparameters and vocabulary)
  if (!llm_model_->load_metadata_from_gguf(ctx_gguf)) {
    RS_LOG_ERR("Failed to load LLM metadata");
    llm_model_.reset();
    return false;
  }

  // 3. Verify architecture
  if (llm_model_->arch() != LLM_ARCH_QWEN3) {
    RS_LOG_ERR("Loaded model is not Qwen3 architecture (got: %d)",
               static_cast<int>(llm_model_->arch()));
    llm_model_.reset();
    return false;
  }

  // 4. Map LLM tensors from the shared tensor map
  if (!llm_model_->map_tensors_qwen3(tensors)) {
    RS_LOG_ERR("Failed to map Qwen3 tensors");
    llm_model_.reset();
    return false;
  }

  // 5. Allocate backend buffers and copy tensor data
  // Create weight context for LLM
  struct ggml_init_params params = {/*.mem_size   =*/ggml_tensor_overhead() *
                                            gguf_get_n_tensors(ctx_gguf) +
                                        (1 << 20),
                                    /*.mem_buffer =*/nullptr,
                                    /*.no_alloc   =*/true};

  // Note: For combined models, tensors are already loaded in backend memory
  // We just need to reference them in the LLM model structure

  // Load Qwen3 special tokens
  auto &vocab = const_cast<llm_vocab &>(llm_model_->vocab());
  vocab.load_qwen3_special_tokens();

  RS_LOG_INFO("Qwen3 LLM loaded: layers=%d, embd=%d, heads=%d, vocab=%d",
              llm_model_->hparams().n_layer, llm_model_->hparams().n_embd,
              llm_model_->hparams().n_head, llm_model_->hparams().n_vocab);

  return true;
}

static void
MapTransformerDecoder(const std::string &prefix,
                      std::map<std::string, struct ggml_tensor *> &tensors,
                      std::unique_ptr<FunASRNanoTransformerDecoder> &decoder,
                      int n_layers) {
  // ctc decoder
  decoder->decoders_layer.resize(n_layers);

  decoder->linear1_weight = tensors.at(prefix + ".linear1.weight");
  decoder->linear1_bias = tensors.at(prefix + ".linear1.bias");
  decoder->linear2_weight = tensors.at(prefix + ".linear2.weight");
  decoder->linear2_bias = tensors.at(prefix + ".linear2.bias");

  for (int i = 0; i < n_layers; ++i) {
    // ctc_decoder.blocks.4.self_attn.linear_q.weight
    std::string p = prefix + ".blocks." + std::to_string(i);
    decoder->decoders_layer[i].self_attn_linear_q_weight =
        tensors.at(p + ".self_attn.linear_q.weight");
    decoder->decoders_layer[i].self_attn_linear_q_bias =
        tensors.at(p + ".self_attn.linear_q.bias");
    decoder->decoders_layer[i].self_attn_linear_k_weight =
        tensors.at(p + ".self_attn.linear_k.weight");
    decoder->decoders_layer[i].self_attn_linear_k_bias =
        tensors.at(p + ".self_attn.linear_k.bias");
    decoder->decoders_layer[i].self_attn_linear_v_weight =
        tensors.at(p + ".self_attn.linear_v.weight");
    decoder->decoders_layer[i].self_attn_linear_v_bias =
        tensors.at(p + ".self_attn.linear_v.bias");
    decoder->decoders_layer[i].self_attn_linear_out_weight =
        tensors.at(p + ".self_attn.linear_out.weight");
    decoder->decoders_layer[i].self_attn_linear_out_bias =
        tensors.at(p + ".self_attn.linear_out.bias");
    decoder->decoders_layer[i].feed_forward_w_1_weight =
        tensors.at(p + ".feed_forward.w_1.weight");
    decoder->decoders_layer[i].feed_forward_w_1_bias =
        tensors.at(p + ".feed_forward.w_1.bias");
    decoder->decoders_layer[i].feed_forward_w_2_weight =
        tensors.at(p + ".feed_forward.w_2.weight");
    decoder->decoders_layer[i].feed_forward_w_2_bias =
        tensors.at(p + ".feed_forward.w_2.bias");
    decoder->decoders_layer[i].norm1_weight = tensors.at(p + ".norm1.weight");
    decoder->decoders_layer[i].norm1_bias = tensors.at(p + ".norm1.bias");
    decoder->decoders_layer[i].norm2_weight = tensors.at(p + ".norm2.weight");
    decoder->decoders_layer[i].norm2_bias = tensors.at(p + ".norm2.bias");
  }
}

bool FunASRNanoModel::MapTensors(
    std::map<std::string, struct ggml_tensor *> &tensors) {
  try {
    encoder_->MapTensors(tensors);
    ctc_decoder_->ctc_out_linear_weight = tensors.at("ctc.ctc_lo.weight");
    ctc_decoder_->ctc_out_linear_bias = tensors.at("ctc.ctc_lo.bias");

    // ctc decoder
    MapTransformerDecoder("ctc_decoder", tensors, ctc_decoder_,
                          hparams_.n_ctc_layers);

    ctc_decoder_->ctc_out_linear_weight = tensors.at("ctc.ctc_lo.weight");
    ctc_decoder_->ctc_out_linear_bias = tensors.at("ctc.ctc_lo.bias");

    if (hparams_.use_llm) {
      MapTransformerDecoder("audio_adaptor", tensors, audio_adaptor_,
                            hparams_.n_adaptor_layers);
    }

    return true;
  } catch (...) {
    RS_LOG_ERR("Tensor mapping failed for FunASRNano.");
    return false;
  }
}

std::shared_ptr<RSState> FunASRNanoModel::CreateState() {
  return std::make_shared<FunASRNanoState>();
}

/**
 * Build and execute the Encoder computation graph.
 * 4 Prompt Tokens + Mel Features -> SANM Encoders -> Parallel Encoders ->
 * Hidden States.
 */
bool FunASRNanoModel::Encode(const std::vector<float> &input_frames,
                             RSState &state, ggml_backend_sched_t sched) {
  return encoder_->Encode(input_frames, state, sched);
}

static struct ggml_tensor *
decoder_forward(const FunASRNanoHParams &hparams, struct ggml_context *ctx,
                struct ggml_tensor *cur, FunASRNanoTransformerDecoder &layers) {
  // --- 1. Downsampling & Initial Projection ---
  // PyTorch: x.view(batch, chunk_num, dim * k)
  const int k = hparams.ctc_downsample_rate;
  const int encoder_dim = cur->ne[0];
  const int seq_len = cur->ne[1];
  const int batch_size = cur->ne[2];
  const int chunk_num = (seq_len + k - 1) / k;

  // Reshape to flatten the downsample dimension into the feature dimension
  // Result shape: [encoder_dim * k, chunk_num, batch_size]
  cur = ggml_reshape_3d(ctx, ggml_cont(ctx, cur), encoder_dim * k, chunk_num,
                        batch_size);

  // Linear 1 + ReLU
  cur = ggml_mul_mat(ctx, layers.linear1_weight, cur);
  cur = ggml_add(ctx, cur, layers.linear1_bias);
  cur = ggml_relu_inplace(ctx, cur);

  // Linear 2 -> Result shape: [llm_dim, chunk_num, batch_size]
  cur = ggml_mul_mat(ctx, layers.linear2_weight, cur);
  cur = ggml_add(ctx, cur, layers.linear2_bias);

  // --- 2. Transformer Encoder Blocks Loop ---
  const int llm_dim = cur->ne[0];
  const int n_head = hparams.ctc_attention_heads;
  const int d_k = llm_dim / n_head;
  const float scale = 1.0f / sqrtf((float)d_k);

  for (auto layer : layers.decoders_layer) {
    // --- Sub-layer 1: Multi-Head Attention ---
    struct ggml_tensor *block_input = cur;
    // Pre-Norm
    cur = ggml_norm(ctx, cur, hparams.eps);
    cur =
        ggml_add(ctx, ggml_mul(ctx, cur, layer.norm1_weight), layer.norm1_bias);

    // Q, K, V Projections
    struct ggml_tensor *q =
        ggml_add(ctx, ggml_mul_mat(ctx, layer.self_attn_linear_q_weight, cur),
                 layer.self_attn_linear_q_bias);
    struct ggml_tensor *k_vec =
        ggml_add(ctx, ggml_mul_mat(ctx, layer.self_attn_linear_k_weight, cur),
                 layer.self_attn_linear_k_bias);
    struct ggml_tensor *v =
        ggml_add(ctx, ggml_mul_mat(ctx, layer.self_attn_linear_v_weight, cur),
                 layer.self_attn_linear_v_bias);

    // Reshape for MHA: [d_k, chunk_num, head, batch]
    q = ggml_permute(
        ctx, ggml_reshape_4d(ctx, q, d_k, n_head, chunk_num, batch_size), 0, 2,
        1, 3);
    k_vec = ggml_permute(
        ctx, ggml_reshape_4d(ctx, k_vec, d_k, n_head, chunk_num, batch_size), 0,
        2, 1, 3);
    v = ggml_permute(
        ctx, ggml_reshape_4d(ctx, v, d_k, n_head, chunk_num, batch_size), 0, 2,
        1, 3);

    // Multi-head Attention Score: (Q * K^T) * scale
    // ggml_mul_mat(k, q) computes dot product over the first dimension (d_k)
    struct ggml_tensor *scores = ggml_mul_mat(ctx, k_vec, q);
    scores = ggml_scale_inplace(ctx, scores, scale);

    // Softmax
    struct ggml_tensor *probs = ggml_soft_max_inplace(ctx, scores);

    // 3. Calculate Context: [d_k, L, H, B] @ [L, L, H, B]
    // Problem: V's ne[0] is d_k, but Probs' ne[0] is L. They must match.
    // Solution: Transpose V so its ne[0] is L.
    struct ggml_tensor *v_T =
        ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [L, d_k, H, B]

    // Now multiply: [L, d_k, H, B] @ [L, L, H, B]
    // GGML reduces ne[0] (L).
    // Result ne[0] = Probs->ne[1] (L), ne[1] = v_T->ne[1] (d_k)
    struct ggml_tensor *context =
        ggml_mul_mat(ctx, v_T, probs); // [L, d_k, H, B]

    // 4. Final Transpose back to standard: [d_k, L, H, B]
    context = ggml_cont(ctx, ggml_permute(ctx, context, 0, 2, 1, 3));
    context = ggml_reshape_3d(ctx, context, llm_dim, chunk_num, batch_size);

    // Output Projection & Residual Connection
    cur = ggml_mul_mat(ctx, layer.self_attn_linear_out_weight, context);
    cur = ggml_add(ctx, cur, layer.self_attn_linear_out_bias);
    cur = ggml_add(ctx, cur, block_input);

    // --- Sub-layer 2: Feed Forward Network (FFN) ---
    struct ggml_tensor *ffn_input = cur;

    // Pre-Norm
    cur = ggml_norm(ctx, cur, hparams.eps);
    cur =
        ggml_add(ctx, ggml_mul(ctx, cur, layer.norm2_weight), layer.norm2_bias);

    // FFN: Linear 1 -> ReLU -> Linear 2
    cur = ggml_mul_mat(ctx, layer.feed_forward_w_1_weight, cur);
    cur = ggml_add(ctx, cur, layer.feed_forward_w_1_bias);
    cur = ggml_relu_inplace(ctx, cur);

    cur = ggml_mul_mat(ctx, layer.feed_forward_w_2_weight, cur);
    cur = ggml_add(ctx, cur, layer.feed_forward_w_2_bias);

    // Residual Connection
    cur = ggml_add(ctx, cur, ffn_input);
  }

  return cur;
}

/**
 * Project encoder output to LLM embedding space and decode with Qwen3
 *
 * Simplified version: Use the last position from prefill to decode
 * (Full autoregressive with KV cache requires more complex integration)
 */
bool FunASRNanoModel::DecodeWithLLM(RSState &state,
                                    ggml_backend_sched_t sched) {
  auto &sv_state = static_cast<FunASRNanoState &>(state);
  if (!sv_state.encoder_out || !llm_model_)
    return false;

  const int audio_T = sv_state.encoder_out->ne[1];     // Audio sequence length
  const int encoder_dim = sv_state.encoder_out->ne[0]; // Encoder dim (512)
  const int llm_dim = llm_model_->hparams().n_embd;    // Qwen3 hidden dim

  // Build prompt
  std::string user_input = "<|im_start|>system\nYou are a helpful "
                           "assistant.<|im_end|>\n<|im_start|>user\n" +
                           sv_state.user_input; // Default: "语音转写"
  std::string suffix_input = "<|im_end|>\n<|im_start|>assistant\n";
  // Tokenize user_input
  std::vector<int32_t> prefix_tokens;
  std::vector<int32_t> suffix_tokens;
  if (llm_model_) {
    prefix_tokens = llm_model_->vocab().tokenize(user_input, false);
  }

  if (llm_model_) {
    suffix_tokens = llm_model_->vocab().tokenize(suffix_input, false);
  }
  int audio_insert_idx = (int)prefix_tokens.size();
  int total_T = audio_insert_idx + audio_T + (int)suffix_tokens.size();

  RS_LOG_INFO("DecodeWithLLM: user_tokens=%d, audio_frames=%d, total_T=%d",
              audio_insert_idx, audio_T, total_T);

  // 2. Create graph builder context
  llm_cparams cparams;
  cparams.n_ctx = total_T + 100;
  cparams.n_batch = total_T;
  cparams.n_ubatch = total_T;
  cparams.n_threads = 4;
  cparams.n_threads_batch = 4;

  // Initialize Qwen3 graph builder if not already done
  if (!llm_graph_builder_) {
    llm_graph_builder_ =
        std::make_unique<llm_build_qwen3>(*llm_model_, cparams, sched);
  }

  // Initialize KV cache (not fully used yet, but created for compatibility)
  if (!llm_kv_cache_) {
    llm_kv_cache::config kv_config;
    kv_config.n_ctx = cparams.n_ctx;
    kv_config.type_k = GGML_TYPE_F16;
    kv_config.type_v = GGML_TYPE_F16;

    llm_kv_cache_ = std::make_unique<llm_kv_cache>(
        kv_config, llm_dim, llm_dim,
        llm_model_->hparams().n_head_kv > 0 ? llm_model_->hparams().n_head_kv
                                            : llm_model_->hparams().n_head,
        llm_model_->hparams().n_layer,
        ggml_backend_sched_get_backend(sched, 0));
  }

  // 3. Build projection graph for prefix + audio
  struct ggml_init_params proj_params = {512 * ggml_tensor_overhead(), nullptr,
                                         true};
  struct ggml_context *ctx_proj = ggml_init(proj_params);
  struct ggml_cgraph *gf_proj = ggml_new_graph(ctx_proj);

  // 3a. Token embeddings for user_input tokens
  ggml_tensor *inp_prefix_tokens =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, audio_insert_idx);
  ggml_set_name(inp_prefix_tokens, "input_prefix_tokens");
  ggml_tensor *inp_suffix_tokens =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, suffix_tokens.size());
  ggml_set_name(inp_suffix_tokens, "input_suffix_tokens");
  ggml_set_input(inp_prefix_tokens);
  ggml_tensor *prefix_embeds =
      ggml_get_rows(ctx_proj, llm_model_->tok_embd(), inp_prefix_tokens);
  ggml_tensor *suffix_embeds =
      ggml_get_rows(ctx_proj, llm_model_->tok_embd(), inp_suffix_tokens);
  // 3b. Audio encoder output -> projected to LLM space
  ggml_tensor *audio_encoder_out = ggml_new_tensor_2d(
      ctx_proj, sv_state.encoder_out->type, encoder_dim, audio_T);
  ggml_set_input(audio_encoder_out);
  ggml_tensor *audio_embeds =
      decoder_forward(hparams_, ctx_proj, audio_encoder_out, *audio_adaptor_);

  // 3c. Concatenate: prefix + audio
  ggml_tensor *llm_embeds =
      ggml_concat(ctx_proj, prefix_embeds, audio_embeds, 1);
  llm_embeds = ggml_concat(ctx_proj, llm_embeds, suffix_embeds, 1);
  ggml_set_output(llm_embeds);
  ggml_build_forward_expand(gf_proj, llm_embeds);

  // 4. Allocate and execute projection graph
  if (!ggml_backend_sched_alloc_graph(sched, gf_proj)) {
    ggml_free(ctx_proj);
    return false;
  }

  // Set input data
  ggml_backend_tensor_copy(sv_state.encoder_out, audio_encoder_out);
  ggml_backend_tensor_set(inp_prefix_tokens, prefix_tokens.data(), 0,
                          prefix_tokens.size() * sizeof(int32_t));
  ggml_backend_tensor_set(inp_suffix_tokens, suffix_tokens.data(), 0,
                          suffix_tokens.size() * sizeof(int32_t));

  if (ggml_backend_sched_graph_compute(sched, gf_proj) != GGML_STATUS_SUCCESS) {
    ggml_free(ctx_proj);
    return false;
  }

  // Get combined embeddings
  std::vector<float> llm_embeds_host(ggml_nelements(llm_embeds));
  ggml_backend_tensor_get(llm_embeds, llm_embeds_host.data(), 0,
                          ggml_nbytes(llm_embeds));

  ggml_free(ctx_proj);
  ggml_backend_sched_reset(sched);

  // 6. Build positions for LLM
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) {
    positions[i] = i + 2; // Position ids start from 2 (matches Python)
  }

  // 7. Create graph context and input tensor for LLM
  struct ggml_init_params llm_params = {512 * ggml_tensor_overhead(), nullptr,
                                        true};
  struct ggml_context *ctx_llm = ggml_init(llm_params);

  // Create input tensor for LLM graph
  ggml_tensor *llm_input =
      ggml_new_tensor_2d(ctx_llm, GGML_TYPE_F32, hparams_.n_llm_embd, total_T);
  ggml_set_name(llm_input, "llm_input");
  ggml_set_input(llm_input);

  // 8. Use Qwen3 to decode from embeddings
  llm_build_opts opts;
  opts.output_mode = llm_output_mode::OUTPUT_LOGITS;
  opts.skip_embeddings = true;
  opts.use_kv_cache = true; // Enable KV extraction for decode loop
  opts.causal_mask = true;

  auto result = llm_graph_builder_->build_graph_from_embeds(
      llm_input, total_T, nullptr, positions.data(), &opts);

  if (!result) {
    RS_LOG_ERR("Failed to build LLM graph");
    ggml_free(ctx_llm);
    return false;
  }

  // 9. Allocate and execute LLM graph
  if (!ggml_backend_sched_alloc_graph(sched, result->get_graph())) {
    ggml_free(ctx_llm);
    return false;
  }

  // Set input data after graph allocation
  ggml_backend_tensor_set(llm_input, llm_embeds_host.data(), 0,
                          llm_embeds_host.size() * sizeof(float));

  // Position ids were built during graph construction, now set the data
  ggml_tensor *positions_tensor = result->get_input_tensor("position_ids");

  if (!positions_tensor) {
    positions_tensor = result->get_input_tensor("position_ids_seq");
  }
  if (positions_tensor) {
    result->set_position_ids(positions_tensor, positions.data(), total_T);
    RS_LOG_INFO("Prefill: set position_ids, first=%d last=%d, n_tokens=%d",
                (int)positions[0], (int)positions[total_T - 1], total_T);
  } else {
    RS_LOG_ERR("Prefill: position_ids tensor not found!");
  }

  // Set causal mask data if needed
  ggml_tensor *causal_mask_tensor = result->get_input_tensor("causal_mask");
  if (causal_mask_tensor) {
    result->set_causal_mask(causal_mask_tensor, total_T, 0);
    RS_LOG_INFO("Prefill: set causal_mask, shape [%lld,%lld] type=%s",
                (long long)causal_mask_tensor->ne[0],
                (long long)causal_mask_tensor->ne[1],
                ggml_type_name(causal_mask_tensor->type));
  } else {
    RS_LOG_ERR("Prefill: causal_mask tensor not found!");
  }

  if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
      GGML_STATUS_SUCCESS) {
    ggml_free(ctx_llm);
    return false;
  }

  // 10. Get logits - now only the LAST token position is output (optimised)
  ggml_tensor *logits = result->get_logits();
  if (!logits) {
    RS_LOG_ERR("No logits from LLM");
    ggml_free(ctx_llm);
    return false;
  }

  // After the prefill optimisation logits->ne[1] == 1 (only last position).
  const int n_vocab = (int)logits->ne[0];
  std::vector<float> logits_host(n_vocab); // single position
  ggml_backend_tensor_get(logits, logits_host.data(), 0,
                          logits_host.size() * sizeof(float));

  // Diagnostics: check logits range at last position
  {
    float max_logit = -1e30f, min_logit = 1e30f;
    int max_idx = 0;
    int nan_count = 0, inf_count = 0;
    for (int v = 0; v < n_vocab; ++v) {
      float val = logits_host[v];
      if (std::isnan(val)) {
        nan_count++;
        continue;
      }
      if (std::isinf(val)) {
        inf_count++;
        continue;
      }
      if (val > max_logit) {
        max_logit = val;
        max_idx = v;
      }
      if (val < min_logit)
        min_logit = val;
    }
    RS_LOG_INFO("Prefill logits at last pos: min=%.4f max=%.4f max_idx=%d "
                "nan=%d inf=%d (n_vocab=%d)",
                min_logit, max_logit, max_idx, nan_count, inf_count, n_vocab);
  }

  std::vector<int32_t> token_ids;

  RS_LOG_INFO("Sampling from last position (prefill output = 1 token)");

  // Greedy sample from the single logits row
  int32_t best_token = 0;
  float best_prob = logits_host[0];
  for (int v = 1; v < n_vocab; ++v) {
    if (logits_host[v] > best_prob) {
      best_prob = logits_host[v];
      best_token = v;
    }
  }
  token_ids.push_back(best_token);

  RS_LOG_INFO("First token: %d (%s)", best_token,
              llm_model_->vocab().decode(best_token).c_str());

  // Extract K/V per layer from output tensors and store in host cache
  const auto &lp = llm_model_->hparams();
  const int n_layer = lp.n_layer;
  const int n_head_kv = lp.n_head_kv > 0 ? lp.n_head_kv : lp.n_head;
  const int head_dim = lp.head_dim;
  const int kv_dim = head_dim * n_head_kv;

  host_kv_cache_k_.assign(n_layer, std::vector<float>());
  host_kv_cache_v_.assign(n_layer, std::vector<float>());
  n_cached_tokens_ = total_T;

  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor *k_out = result->get_kv_output_k(il);
    ggml_tensor *v_out = result->get_kv_output_v(il);
    if (k_out && v_out) {
      // KV output tensors are 3D [head_dim, n_head_kv, n_tokens] (contiguous)
      // Memory layout is equivalent to 2D [kv_dim, n_tokens] for contiguous
      // tensors
      size_t kv_bytes = ggml_nbytes(k_out);
      size_t kv_size = kv_bytes / sizeof(float);
      host_kv_cache_k_[il].resize(kv_size);
      host_kv_cache_v_[il].resize(kv_size);
      ggml_backend_tensor_get(k_out, host_kv_cache_k_[il].data(), 0, kv_bytes);
      ggml_backend_tensor_get(v_out, host_kv_cache_v_[il].data(), 0, kv_bytes);
    } else {
      RS_LOG_ERR("Prefill KV output null for layer %d! k=%p v=%p", il, k_out,
                 v_out);
    }
  }

  // Verify KV cache data integrity after prefill
  {
    float k_sum = 0.0f, v_sum = 0.0f;
    if (!host_kv_cache_k_.empty() && !host_kv_cache_k_[0].empty()) {
      size_t check_n = 16;
      if (check_n > host_kv_cache_k_[0].size())
        check_n = host_kv_cache_k_[0].size();
      for (size_t i = 0; i < check_n; ++i) {
        k_sum += host_kv_cache_k_[0][i];
        v_sum += host_kv_cache_v_[0][i];
      }
    }
    RS_LOG_INFO(
        "Prefill KV cache: layer0 K_sum(16)=%.4f V_sum(16)=%.4f, n_cached=%d",
        k_sum, v_sum, n_cached_tokens_);
  }

  ggml_free(ctx_llm);
  ggml_backend_sched_reset(sched);

  // ==========================================
  // Autoregressive decode loop
  // ==========================================
  const int32_t eos_token_id = llm_model_->vocab().token_eos();

  // ==========================================
  // O3: Allocate persistent GPU-side KV cache buffers
  // Instead of uploading the entire KV cache every step, we allocate
  // max-capacity GPU buffers ONCE using ggml_backend_alloc_ctx_tensors
  // (independent of the scheduler, so they survive sched_reset cycles).
  // Each step only writes the new 1-column KV via ggml_cpy inside the
  // compute graph, then uses a view for attention.
  // ==========================================
  const int32_t n_kv_max = n_cached_tokens_ + MAX_DECODE_TOKENS;
  std::vector<ggml_tensor *> gpu_kv_k_vec(n_layer, nullptr);
  std::vector<ggml_tensor *> gpu_kv_v_vec(n_layer, nullptr);
  ggml_backend_buffer_t kv_gpu_buf = nullptr;

  {
    struct ggml_init_params kv_buf_params = {
        (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16), nullptr,
        true};
    struct ggml_context *ctx_kv = ggml_init(kv_buf_params);

    for (int il = 0; il < n_layer; ++il) {
      gpu_kv_k_vec[il] =
          ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
      gpu_kv_v_vec[il] =
          ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
      ggml_set_name(gpu_kv_k_vec[il],
                    ("gpu_kv_k_" + std::to_string(il)).c_str());
      ggml_set_name(gpu_kv_v_vec[il],
                    ("gpu_kv_v_" + std::to_string(il)).c_str());
    }

    // Allocate on the GPU backend (Metal), independent of scheduler
    // Use n_backends to safely check if a GPU backend is available
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    ggml_backend_t gpu_backend =
        (n_backends > 1)
            ? ggml_backend_sched_get_backend(sched, 1) // GPU backend at index 1
            : ggml_backend_sched_get_backend(sched, 0); // CPU-only fallback
    if (n_backends <= 1) {
      RS_LOG_INFO("GPU backend not available, using CPU for KV cache");
    }

    kv_gpu_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, gpu_backend);
    if (!kv_gpu_buf) {
      RS_LOG_ERR("Failed to allocate GPU KV cache buffers");
    }

    // Upload initial KV data from host (prefill results)
    for (int il = 0; il < n_layer; ++il) {
      size_t kv_bytes = (size_t)kv_dim * n_cached_tokens_ * sizeof(float);
      ggml_backend_tensor_set(gpu_kv_k_vec[il], host_kv_cache_k_[il].data(), 0,
                              kv_bytes);
      ggml_backend_tensor_set(gpu_kv_v_vec[il], host_kv_cache_v_[il].data(), 0,
                              kv_bytes);
    }

    RS_LOG_INFO("GPU KV cache allocated: %d layers, kv_dim=%d, n_kv_max=%d "
                "(%.1f MB/layer, total %.1f MB)",
                n_layer, kv_dim, n_kv_max,
                (double)kv_dim * n_kv_max * 2 * sizeof(float) / (1024 * 1024),
                (double)kv_dim * n_kv_max * 2 * n_layer * sizeof(float) /
                    (1024 * 1024));
    // Note: ctx_kv must NOT be freed — the tensor descriptors must persist
    // for the lifetime of the GPU buffer. We'll free both at the end of decode.
  }

  // ==========================================
  // Autoregressive decode loop (with GPU-persistent KV cache)
  // ==========================================

  for (int step = 0; step < MAX_DECODE_TOKENS; ++step) {
    llm_pos decode_pos = (llm_pos)(n_cached_tokens_ + 2);
    RS_LOG_INFO("Decode step %d: pos=%d, n_cached=%d, input_token=%d (%s)",
                step, (int)decode_pos, n_cached_tokens_, best_token,
                llm_model_->vocab().decode(best_token).c_str());

    llm_build_opts dec_opts;
    dec_opts.output_mode = llm_output_mode::OUTPUT_LOGITS;
    dec_opts.skip_embeddings = false;
    dec_opts.use_kv_cache = true;
    dec_opts.is_decode_step = true;
    dec_opts.n_kv_cache = n_cached_tokens_;
    dec_opts.n_kv_max = n_kv_max;
    dec_opts.causal_mask = true;
    // Pass GPU-persistent KV buffers — build_kv_cache_concat will use
    // views into these buffers instead of creating kv_cached input tensors
    dec_opts.gpu_kv_k = gpu_kv_k_vec.data();
    dec_opts.gpu_kv_v = gpu_kv_v_vec.data();

    auto dec_result = llm_graph_builder_->build_graph(&best_token, 1, nullptr,
                                                      &decode_pos, &dec_opts);

    if (!dec_result) {
      RS_LOG_ERR("Failed to build decode graph at step %d", step);
      break;
    }

    if (!ggml_backend_sched_alloc_graph(sched, dec_result->get_graph())) {
      RS_LOG_ERR("Failed to allocate decode graph at step %d", step);
      break;
    }

    // Set input token ID
    ggml_tensor *inp_tok = dec_result->get_input_tensor("inp_tokens");
    if (inp_tok) {
      ggml_backend_tensor_set(inp_tok, &best_token, 0, sizeof(int32_t));
    }

    // Set position IDs
    ggml_tensor *dec_pos_tensor = dec_result->get_input_tensor("position_ids");
    if (!dec_pos_tensor)
      dec_pos_tensor = dec_result->get_input_tensor("position_ids_seq");
    if (dec_pos_tensor) {
      dec_result->set_position_ids(dec_pos_tensor, &decode_pos, 1);
    }

    // Set causal mask
    ggml_tensor *dec_mask_tensor = dec_result->get_input_tensor("causal_mask");
    if (dec_mask_tensor) {
      dec_result->set_causal_mask(dec_mask_tensor, 1, n_cached_tokens_);
    }

    // O3: GPU KV buffer already contains the first n_cached columns.
    // build_kv_cache_concat uses views into the GPU buffer for the
    // cached portion — no need to upload full KV cache each step!

    // Execute decode step
    if (ggml_backend_sched_graph_compute(sched, dec_result->get_graph()) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("Decode graph compute failed at step %d", step);
      break;
    }

    // Extract logits and sample next token
    ggml_tensor *dec_logits = dec_result->get_logits();
    if (!dec_logits) {
      RS_LOG_ERR("No logits from decode step %d", step);
      break;
    }

    std::vector<float> dec_logits_host(n_vocab);
    ggml_backend_tensor_get(dec_logits, dec_logits_host.data(), 0,
                            n_vocab * sizeof(float));

    int32_t next_token = 0;
    float next_prob = dec_logits_host[0];
    for (int v = 1; v < n_vocab; ++v) {
      if (dec_logits_host[v] > next_prob) {
        next_prob = dec_logits_host[v];
        next_token = v;
      }
    }

    // Update host KV cache + GPU KV buffer: append new column
    for (int il = 0; il < n_layer; ++il) {
      ggml_tensor *k_out = dec_result->get_kv_output_k(il);
      ggml_tensor *v_out = dec_result->get_kv_output_v(il);
      if (k_out && v_out) {
        const size_t new_col_bytes = (size_t)kv_dim * sizeof(float);
        const size_t col_offset =
            (size_t)n_cached_tokens_ * kv_dim * sizeof(float);
        host_kv_cache_k_[il].resize((n_cached_tokens_ + 1) * kv_dim);
        host_kv_cache_v_[il].resize((n_cached_tokens_ + 1) * kv_dim);
        ggml_backend_tensor_get(
            k_out, host_kv_cache_k_[il].data() + n_cached_tokens_ * kv_dim,
            col_offset, new_col_bytes);
        ggml_backend_tensor_get(
            v_out, host_kv_cache_v_[il].data() + n_cached_tokens_ * kv_dim,
            col_offset, new_col_bytes);

        // Write new column to GPU-persistent buffer for next step's view
        ggml_backend_tensor_set(gpu_kv_k_vec[il],
                                host_kv_cache_k_[il].data() +
                                    n_cached_tokens_ * kv_dim,
                                col_offset, new_col_bytes);
        ggml_backend_tensor_set(gpu_kv_v_vec[il],
                                host_kv_cache_v_[il].data() +
                                    n_cached_tokens_ * kv_dim,
                                col_offset, new_col_bytes);
      }
    }

    n_cached_tokens_++;

    if (next_token == eos_token_id) {
      RS_LOG_INFO("EOS token reached at step %d", step);
      ggml_backend_sched_reset(sched);
      break;
    }

    token_ids.push_back(next_token);
    best_token = next_token;

    ggml_backend_sched_reset(sched);
  }

  // Free GPU KV cache buffer
  if (kv_gpu_buf) {
    ggml_backend_buffer_free(kv_gpu_buf);
    kv_gpu_buf = nullptr;
  }

  // Convert token IDs to text
  auto &funasr_state = static_cast<FunASRNanoState &>(state);
  if (llm_model_) {
    std::string result_text = llm_model_->vocab().detokenize(token_ids);
    funasr_state.tokens.push_back(result_text);
    RS_LOG_INFO("DecodeWithLLM: %s", result_text.c_str());
  }

  RS_LOG_INFO("DecodeWithLLM: decoded %zu tokens", token_ids.size());
  return true;
}

bool FunASRNanoModel::DecodeWithoutLLM(RSState &state,
                                       ggml_backend_sched_t sched) {
  auto &sv_state = static_cast<SenseVoiceState &>(state);
  if (!sv_state.encoder_out)
    return false;

  int T = sv_state.encoder_out->ne[1];
  int V = hparams_.n_vocab;

  struct ggml_context *ctx0 =
      ggml_init({2 * 1024 * ggml_tensor_overhead(), nullptr, true});
  struct ggml_cgraph *gf =
      ggml_new_graph_custom(ctx0, FUNASR_NANO_DECODER_MAX_NODES, false);

  struct ggml_tensor *encoder_in = ggml_new_tensor_2d(
      ctx0, sv_state.encoder_out->type, sv_state.encoder_out->ne[0],
      sv_state.encoder_out->ne[1]);
  ggml_set_input(encoder_in);

  // transformer

  struct ggml_tensor *cur =
      decoder_forward(hparams_, ctx0, encoder_in, *ctc_decoder_);

  // 1. Linear projection to vocab size
  cur = ggml_mul_mat(ctx0, ctc_decoder_->ctc_out_linear_weight, cur);
  cur = ggml_add(ctx0, cur, ctc_decoder_->ctc_out_linear_bias);

  // 2. Compute log-probabilities
  struct ggml_tensor *log_probs = ggml_log(ctx0, ggml_soft_max(ctx0, cur));

  struct ggml_tensor *output_node = nullptr;

  if (beam_size <= 1)
    // Greedy serach: Calculate argmax on backend
    output_node = ggml_argmax(ctx0, log_probs);

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
  // print_tensor(output_node);
  if (beam_size <= 1) {
    raw_ids.resize(T);
    ggml_backend_tensor_get(output_node, raw_ids.data(), 0,
                            T * sizeof(int32_t));
  } else {
    host_log_probs.resize(T * V);
    ggml_backend_tensor_get(output_node, host_log_probs.data(), 0,
                            T * V * sizeof(float));
  }
  sv_state.ids = CTCDecoder::GreedyDecode(raw_ids.data(), T);
  ggml_free(ctx0);
  return true;
};

/**
 * Enhanced Decode function supporting Greedy and Beam Search.
 */
bool FunASRNanoModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  auto &sv_state = static_cast<SenseVoiceState &>(state);

  // Choose decode path based on LLM availability
  bool success = false;
  if (hparams_.use_llm && llm_model_) {
    RS_LOG_INFO("Decoding with Qwen3 LLM");
    success = DecodeWithLLM(state, sched);
  } else {
    success = DecodeWithoutLLM(state, sched);
  }

  if (!success) {
    return false;
  }

  // Convert token IDs to text (for non-LLM path)
  if (!hparams_.use_llm || !llm_model_) {
    for (auto id : sv_state.ids) {
      sv_state.tokens.push_back(this->vocab_.id_to_token[id]);
    }
  }

  return true;
}

std::string FunASRNanoModel::GetTranscription(RSState &state) {
  auto &sv_state = static_cast<FunASRNanoState &>(state);
  std::string result;
  result.reserve(64); // ⭐ 关键：避免反复 realloc

  for (const auto &s : sv_state.tokens) {
    result += s;
  }
  sv_state.tokens.clear();
  return result;
}

// Registration logic
extern void
rs_register_model_arch(const std::string &arch,
                       std::function<std::shared_ptr<ISpeechModel>()> creator);
namespace {
struct FunASRNanoRegistrar {
  FunASRNanoRegistrar() {
    rs_register_model_arch(
        "FunASRNano", []() { return std::make_shared<FunASRNanoModel>(); });
  }
} global_FunASRNano_reg;
} // namespace