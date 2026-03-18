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
      "语音转写"; // User input prompt, default is "语音转写"
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

  // Check if LLM is enabled
  int use_llm_idx = gguf_find_key(ctx_gguf, "llm.use");
  if (use_llm_idx != -1) {
    hparams_.use_llm = gguf_get_val_i32(ctx_gguf, use_llm_idx) != 0;
  }

  if (hparams_.use_llm) {
    hparams_.n_llm_layer = gguf_get_val_i32(
        ctx_gguf, gguf_find_key(ctx_gguf, "qwen3.block_count"));
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
 * Chat format prompt:
 *   [{"role": "system", "content": "You are a helpful assistant."},
 *    {"role": "user", "content":
 * "{user_input}<|startofspeech|>!<|endofspeech|>"},
 *    {"role": "assistant", "content": ""}]
 *
 * Flow:
 * 1. Build and tokenize chat format prompt
 * 2. Find <|startofspeech|> and <|endofspeech|> positions
 * 3. Embed text tokens and project audio encoder output to LLM embedding space
 * 4. Insert audio_embeds at <|startofspeech|> to <|endofspeech|> range
 * 5. Feed combined embeddings to Qwen3 for decoding
 */
bool FunASRNanoModel::DecodeWithLLM(RSState &state,
                                    ggml_backend_sched_t sched) {
  auto &sv_state = static_cast<FunASRNanoState &>(state);
  if (!sv_state.encoder_out || !llm_model_)
    return false;

  const int audio_T = sv_state.encoder_out->ne[1];     // Audio sequence length
  const int encoder_dim = sv_state.encoder_out->ne[0]; // Encoder dim (512)
  const int llm_dim = llm_model_->hparams().n_embd;    // Qwen3 hidden dim

  // Build chat format prompt with placeholders for audio insertion
  // Format: {user_input}<|startofspeech|>!<|endofspeech|>
  // The "<|startofspeech|>!<|endofspeech|>" part will be REPLACED by audio
  // embeddings (no token embeddings)
  std::string user_input = sv_state.user_input; // Default: "语音转写"

  // Tokenize user_input only (audio will be inserted after this)
  std::vector<int32_t> prefix_tokens;
  if (llm_model_) {
    prefix_tokens = llm_model_->vocab().tokenize(user_input, false);
  }

  // audio_insert_idx: position where audio embeddings start (after user_input
  // tokens)
  int audio_insert_idx = (int)prefix_tokens.size();

  // Total sequence length: user_input tokens + audio frames
  int total_T = audio_insert_idx + audio_T;

  RS_LOG_INFO(
      "DecodeWithLLM: user_tokens=%d, audio_frames=%d, total=%d, insert_idx=%d",
      audio_insert_idx, audio_T, total_T, audio_insert_idx);

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

  // Initialize KV cache for LLM if not already done
  if (!llm_kv_cache_) {
    llm_kv_cache::config kv_config;
    kv_config.n_ctx = cparams.n_ctx;
    kv_config.type_k = GGML_TYPE_F16;
    kv_config.type_v = GGML_TYPE_F16;

    llm_kv_cache_ = std::make_unique<llm_kv_cache>(
        kv_config, llm_dim, llm_dim,
        llm_model_->hparams().n_head_kv > 0 ? llm_model_->hparams().n_head_kv
                                            : llm_model_->hparams().n_head,
        ggml_backend_sched_get_backend(sched, 0));
  }

  // 3. Build projection graph
  struct ggml_init_params proj_params = {512 * ggml_tensor_overhead(), nullptr,
                                         true};
  struct ggml_context *ctx_proj = ggml_init(proj_params);
  struct ggml_cgraph *gf_proj = ggml_new_graph(ctx_proj);

  // 3a. Token embeddings for user_input tokens
  ggml_tensor *inp_prefix_tokens =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, audio_insert_idx);
  ggml_set_name(inp_prefix_tokens, "input_prefix_tokens");
  ggml_set_input(inp_prefix_tokens);
  ggml_tensor *prefix_embeds =
      ggml_get_rows(ctx_proj, llm_model_->tok_embd(), inp_prefix_tokens);

  // 3b. Audio encoder output: [encoder_dim, audio_T] -> projected to [llm_dim,
  // audio_T]
  ggml_tensor *audio_encoder_out = ggml_new_tensor_2d(
      ctx_proj, sv_state.encoder_out->type, encoder_dim, audio_T);
  ggml_set_input(audio_encoder_out);
  ggml_tensor *audio_embeds =
      decoder_forward(hparams_, ctx_proj, audio_encoder_out, *audio_adaptor_);

  // 3c. Build combined embeddings: [user_input_tokens] + [audio_embeds]
  // Final shape: [llm_dim, total_T] where total_T = audio_insert_idx + audio_T
  // Note: <|startofspeech|> and <|endofspeech|> are NOT converted to embeddings
  ggml_tensor *llm_embeds = nullptr;

  // Concatenate: prefix (user_input) + audio
  llm_embeds = ggml_concat(ctx_proj, prefix_embeds, audio_embeds, 1);

  ggml_set_output(llm_embeds);
  ggml_build_forward_expand(gf_proj, llm_embeds);

  // 4. Allocate and execute projection graph
  if (!ggml_backend_sched_alloc_graph(sched, gf_proj)) {
    ggml_free(ctx_proj);
    return false;
  }

  // Set input data: audio encoder output and token IDs
  ggml_backend_tensor_copy(sv_state.encoder_out, audio_encoder_out);
  ggml_backend_tensor_set(inp_prefix_tokens, prefix_tokens.data(), 0,
                          prefix_tokens.size() * sizeof(int32_t));

  // Execute projection graph to get combined embeddings
  if (ggml_backend_sched_graph_compute(sched, gf_proj) != GGML_STATUS_SUCCESS) {
    ggml_free(ctx_proj);
    return false;
  }

  // 5. Get projected embeddings before reset
  std::vector<float> llm_embeds_host(ggml_nelements(llm_embeds));
  ggml_backend_tensor_get(llm_embeds, llm_embeds_host.data(), 0,
                          ggml_nbytes(llm_embeds));

  // Free projection graph context
  ggml_free(ctx_proj);

  // Reset scheduler for LLM graph allocation
  ggml_backend_sched_reset(sched);

  // 6. Build positions for LLM
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) {
    positions[i] = i;
  }

  // 7. Create new graph context and input tensor for LLM
  struct ggml_init_params llm_params = {512 * ggml_tensor_overhead(), nullptr,
                                        true};
  struct ggml_context *ctx_llm = ggml_init(llm_params);
  struct ggml_cgraph *gf_llm = ggml_new_graph(ctx_llm);

  // Create input tensor for LLM graph
  ggml_tensor *llm_input =
      ggml_new_tensor_2d(ctx_llm, GGML_TYPE_F32, hparams_.n_llm_embd, total_T);
  ggml_set_name(llm_input, "llm_input");
  ggml_set_input(llm_input);

  // 8. Use Qwen3 to decode from embeddings
  llm_build_opts opts;
  opts.output_mode = llm_output_mode::OUTPUT_LOGITS;
  opts.skip_embeddings = true;
  opts.use_kv_cache = true;
  opts.causal_mask = true;

  auto result = llm_graph_builder_->build_graph_from_embeds(
      llm_input, total_T, llm_kv_cache_.get(), positions.data(), &opts);

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
  }

  if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
      GGML_STATUS_SUCCESS) {
    ggml_free(ctx_llm);
    return false;
  }

  // 10. Get logits and perform greedy decoding (only for audio part)
  ggml_tensor *logits = result->get_logits();
  if (!logits) {
    RS_LOG_ERR("No logits from LLM");
    ggml_free(ctx_llm);
    return false;
  }

  const int n_vocab = llm_model_->hparams().n_vocab;
  std::vector<float> logits_host(n_vocab * total_T);
  ggml_backend_tensor_get(logits, logits_host.data(), 0,
                          logits_host.size() * sizeof(float));

  // Greedy decoding: only decode audio part (skip prefix tokens)
  // Start decoding from audio_insert_idx (where audio embeddings begin)
  std::vector<int32_t> token_ids;
  token_ids.reserve(audio_T);

  for (int t = audio_insert_idx; t < total_T; ++t) {
    const float *logits_t = logits_host.data() + t * n_vocab;
    int32_t best_token = 0;
    float best_prob = logits_t[0];

    for (int v = 1; v < n_vocab; ++v) {
      if (logits_t[v] > best_prob) {
        best_prob = logits_t[v];
        best_token = v;
      }
    }

    if (llm_model_->vocab().is_eos(best_token)) {
      break;
    }

    token_ids.push_back(best_token);
  }

  // 11. Convert token IDs to text using LLM's tokenizer
  auto &funasr_state = static_cast<FunASRNanoState &>(state);
  if (llm_model_) {
    // Use Qwen3's detokenizer for proper BPE decoding
    std::string result_text = llm_model_->vocab().detokenize(token_ids);
    funasr_state.tokens.push_back(result_text);
    RS_LOG_INFO("DecodeWithLLM: %s", result_text.c_str());
  } else {
    // Fallback: use local vocab (may not be correct for LLM tokens)
    for (int32_t token_id : token_ids) {
      auto it = vocab_.id_to_token.find(token_id);
      if (it != vocab_.id_to_token.end()) {
        funasr_state.tokens.push_back(it->second);
      }
    }
  }

  RS_LOG_INFO("DecodeWithLLM: decoded %zu tokens", token_ids.size());
  ggml_free(ctx_llm);
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
  // ggml_backend_sched_set_eval_callback(sched, ggml_debug, new
  // callback_data());
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