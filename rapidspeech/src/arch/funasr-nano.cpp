#include "funasr-nano.h"
#include "core/rs_context.h"
#include "ctc_decoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "hotword_retriever.h"
#include "sensevoice.h"
#include "utils/debug_utils.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <unordered_set>

#include "ggml-cpu.h"

// Increased node limit to handle deep FunASRNano graphs (50+ layers)
#define FUNASR_NANO_ENCODER_MAX_NODES 6144
#define FUNASR_NANO_DECODER_MAX_NODES 1024

struct FunASRNanoState : public SenseVoiceState {};

static int funasr_utf8_decode(const char *p, size_t avail, uint32_t *cp) {
  if (avail == 0)
    return 0;
  unsigned char c0 = (unsigned char)p[0];
  if (c0 < 0x80) {
    *cp = c0;
    return 1;
  }
  if ((c0 >> 5) == 0x6 && avail >= 2) {
    *cp = ((uint32_t)(c0 & 0x1F) << 6) | ((uint32_t)(p[1] & 0x3F));
    return 2;
  }
  if ((c0 >> 4) == 0xE && avail >= 3) {
    *cp = ((uint32_t)(c0 & 0x0F) << 12) |
          ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
    return 3;
  }
  if ((c0 >> 3) == 0x1E && avail >= 4) {
    *cp = ((uint32_t)(c0 & 0x07) << 18) |
          ((uint32_t)(p[1] & 0x3F) << 12) |
          ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
    return 4;
  }
  *cp = c0;
  return 1;
}

static std::string decode_ctc_token_bytes(const std::string &token) {
  if (token.empty() || token[0] == '<')
    return "";

  std::string bytes;
  bytes.reserve(token.size());
  for (size_t i = 0; i < token.size();) {
    uint32_t cp = 0;
    int n = funasr_utf8_decode(token.data() + i, token.size() - i, &cp);
    if (n <= 0)
      break;
    if (cp <= 0xFF) {
      bytes.push_back((char)cp);
    } else {
      bytes.append(token.data() + i, n);
    }
    i += n;
  }
  return bytes;
}

// --- FunASRNanoModel Implementation ---

FunASRNanoModel::FunASRNanoModel()
    : ctx_weights_(nullptr), user_input_prompt_("语音转写："),
      active_prompt_("语音转写：") {
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
  int ctc_odim_idx = gguf_find_key(ctx_gguf, "ctc.odim");
  if (ctc_odim_idx != -1) {
    hparams_.n_ctc_vocab = gguf_get_val_i32(ctx_gguf, ctc_odim_idx);
  }
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
  runtime_use_llm_ = hparams_.use_llm;

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

  // 2. Load CTC vocabulary.  Combined FunASRNano+Qwen GGUFs keep the Qwen
  // tokenizer under tokenizer.ggml.tokens; CTC needs its own token table.
  int token_idx = gguf_find_key(ctx_gguf, "ctc.tokenizer.ggml.tokens");
  if (token_idx == -1) {
    int generic_token_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
    if (generic_token_idx != -1 &&
        gguf_get_arr_n(ctx_gguf, generic_token_idx) == hparams_.n_ctc_vocab) {
      token_idx = generic_token_idx;
    }
  }
  if (token_idx != -1) {
    int n_vocab = gguf_get_arr_n(ctx_gguf, token_idx);
    for (int i = 0; i < n_vocab; i++) {
      vocab_.id_to_token[i] = gguf_get_arr_str(ctx_gguf, token_idx, i);
    }
    vocab_.n_vocab = n_vocab;
    vocab_.has_ctc_tokens = true;
  } else {
    RS_LOG_WARN("FunASRNano: CTC token table missing; CTC/partial text will "
                "be suppressed. Reconvert GGUF to include "
                "ctc.tokenizer.ggml.tokens.");
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
                                            static_cast<size_t>(gguf_get_n_tensors(ctx_gguf)) +
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

bool FunASRNanoModel::EncodeBatch(
    const std::vector<const std::vector<float> *> &inputs,
    const std::vector<RSState *> &states, ggml_backend_sched_t sched) {
  return encoder_->EncodeBatch(inputs, states, sched);
}

void FunASRNanoModel::set_imatrix_callback(
    std::function<void(struct ggml_tensor *)> cb) {
  imatrix_cb_ = cb;
  if (encoder_) {
    encoder_->set_imatrix_callback(cb);
  }
}

namespace {
// Trampoline shared by every FunASR compute call: installed as the sched's
// per-node eval callback so that the imatrix collector sees each MUL_MAT's
// src1 while it is still live (before any downstream op that shares its
// buffer slot has executed).
bool imatrix_eval_trampoline(struct ggml_tensor *t, bool ask, void *ud) {
  if (ask) return true;
  auto *cb = static_cast<std::function<void(struct ggml_tensor *)> *>(ud);
  (*cb)(t);
  return true;
}

// RAII install/teardown for the per-node observer around a single compute
// call.  Keeping the eval callback armed past compute would let it fire on
// unrelated downstream graphs (e.g. LLM forward after CTC) and double-count
// or even crash if those graphs use a different sched.
struct ImatrixHookGuard {
  ggml_backend_sched_t sched;
  bool armed = false;
  ImatrixHookGuard(ggml_backend_sched_t s,
                   std::function<void(struct ggml_tensor *)> *cb)
      : sched(s) {
    if (cb && *cb) {
      ggml_backend_sched_set_eval_callback(sched, imatrix_eval_trampoline, cb);
      armed = true;
    }
  }
  ~ImatrixHookGuard() {
    if (armed) {
      ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    }
  }
};
}  // namespace

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
    // F32 accumulator on CUDA: F16 default overflows for reduction-heavy
    // attention scores, collapsing softmax into degenerate output.
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
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
  const bool debug_decode = std::getenv("RS_DEBUG_ASR_DECODE") != nullptr;

  const int audio_T = sv_state.encoder_out->ne[1];     // Audio sequence length
  const int encoder_dim = sv_state.encoder_out->ne[0]; // Encoder dim (512)
  const int llm_dim = llm_model_->hparams().n_embd;    // Qwen3 hidden dim

  if (cached_prefix_tokens_.empty() ||
      cached_user_input_prompt_ != active_prompt_) {
    std::string user_input = "<|im_start|>system\nYou are a helpful "
                             "assistant.<|im_end|>\n<|im_start|>user\n" +
                             active_prompt_;
    std::string suffix_input = "<|im_end|>\n<|im_start|>assistant\n";
    cached_prefix_tokens_ = llm_model_->vocab().tokenize(user_input, false);
    cached_suffix_tokens_ = llm_model_->vocab().tokenize(suffix_input, false);
    cached_user_input_prompt_ = active_prompt_;
    if (std::getenv("RS_HOTWORD_DEBUG"))
      RS_LOG_INFO("[hotword] LLM user prompt = |%s| (%zu prefix tokens)",
                  active_prompt_.c_str(), cached_prefix_tokens_.size());
  }
  const std::vector<int32_t> &prefix_tokens = cached_prefix_tokens_;
  const std::vector<int32_t> &suffix_tokens = cached_suffix_tokens_;

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

  // Always create a fresh graph builder for each segment.
  // The builder stores a reference to cparams (on our stack) and must not
  // outlive this call; reusing a stale builder across segments would dangle.
  llm_graph_builder_ =
      std::make_unique<llm_build_qwen3>(*llm_model_, cparams, sched);

  // Clear any stale KV cache state from previous segments
  host_kv_cache_k_.clear();
  host_kv_cache_v_.clear();
  n_cached_tokens_ = 0;

  const auto &lp = llm_model_->hparams();
  const int n_layer = lp.n_layer;
  const int n_head_kv = lp.n_head_kv > 0 ? lp.n_head_kv : lp.n_head;
  const int head_dim = lp.head_dim;
  const int kv_dim = head_dim * n_head_kv;
  const int32_t n_kv_max = total_T + MAX_DECODE_TOKENS;

  std::vector<ggml_tensor *> gpu_kv_k_vec(n_layer, nullptr);
  std::vector<ggml_tensor *> gpu_kv_v_vec(n_layer, nullptr);
  ggml_backend_buffer_t kv_gpu_buf = nullptr;
  ggml_backend_t kv_backend = nullptr;
  struct ggml_context *ctx_kv = nullptr;

  {
    struct ggml_init_params kv_buf_params = {
        (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16), nullptr,
        true};
    ctx_kv = ggml_init(kv_buf_params);

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

    int n_backends = ggml_backend_sched_get_n_backends(sched);
    int gpu_idx = n_backends - 1;
    for (int bi = 0; bi < n_backends; ++bi) {
      ggml_backend_t b = ggml_backend_sched_get_backend(sched, bi);
      ggml_backend_dev_t dev = ggml_backend_get_device(b);
      if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        gpu_idx = bi;
        break;
      }
    }
    ggml_backend_t gpu_backend =
        ggml_backend_sched_get_backend(sched, gpu_idx);
    kv_backend = gpu_backend;
    RS_LOG_INFO("KV cache allocated on backend %d/%d: %s", gpu_idx,
                n_backends, ggml_backend_name(gpu_backend));

    kv_gpu_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, gpu_backend);
    if (!kv_gpu_buf) {
      RS_LOG_ERR("DecodeWithLLM: Failed to allocate GPU KV cache buffers");
      ggml_free(ctx_kv);
      return false;
    }

    std::vector<float> zero_kv((size_t)kv_dim * n_kv_max, 0.0f);
    for (int il = 0; il < n_layer; ++il) {
      ggml_backend_tensor_set(gpu_kv_k_vec[il], zero_kv.data(), 0,
                              zero_kv.size() * sizeof(float));
      ggml_backend_tensor_set(gpu_kv_v_vec[il], zero_kv.data(), 0,
                              zero_kv.size() * sizeof(float));
    }

    RS_LOG_INFO("GPU KV cache allocated: %d layers, kv_dim=%d, n_kv_max=%d "
                "(%.1f MB/layer, total %.1f MB)",
                n_layer, kv_dim, n_kv_max,
                (double)kv_dim * n_kv_max * 2 * sizeof(float) / (1024 * 1024),
                (double)kv_dim * n_kv_max * 2 * n_layer * sizeof(float) /
                    (1024 * 1024));
  }

  // 3. Build projection graph for prefix + audio.  The resulting
  // llm_embeds tensor is passed directly into the LLM prefill graph, so the
  // projection output never round-trips through host memory.
  struct ggml_init_params proj_params = {512 * ggml_tensor_overhead(), nullptr,
                                         true};
  struct ggml_context *ctx_proj = ggml_init(proj_params);

  // 3a. Token embeddings for user_input tokens
  ggml_tensor *inp_prefix_tokens =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, audio_insert_idx);
  ggml_set_name(inp_prefix_tokens, "input_prefix_tokens");
  ggml_tensor *inp_suffix_tokens =
      ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, suffix_tokens.size());
  ggml_set_name(inp_suffix_tokens, "input_suffix_tokens");
  ggml_set_input(inp_prefix_tokens);
  ggml_set_input(inp_suffix_tokens);
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

  // 6. Build positions for LLM
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) {
    positions[i] = i + 2; // Position ids start from 2 (matches Python)
  }

  // 8. Use Qwen3 to decode from embeddings
  llm_build_opts opts;
  opts.output_mode = llm_output_mode::OUTPUT_LOGITS;
  opts.skip_embeddings = true;
  opts.use_kv_cache = true;
  opts.n_kv_max = n_kv_max;
  opts.gpu_kv_k = gpu_kv_k_vec.data();
  opts.gpu_kv_v = gpu_kv_v_vec.data();
  opts.causal_mask = true;

  auto result = llm_graph_builder_->build_graph_from_embeds(
      llm_embeds, total_T, nullptr, positions.data(), &opts);

  if (!result) {
    RS_LOG_ERR("Failed to build LLM graph");
    ggml_free(ctx_proj);
    ggml_backend_buffer_free(kv_gpu_buf);
    ggml_free(ctx_kv);
    return false;
  }

  // 9. Allocate and execute LLM graph
  result->assign_kv_backend(sched, kv_backend, n_layer);
  if (!ggml_backend_sched_alloc_graph(sched, result->get_graph())) {
    RS_LOG_ERR("DecodeWithLLM: failed to allocate LLM prefill graph");
    ggml_free(ctx_proj);
    ggml_backend_buffer_free(kv_gpu_buf);
    ggml_free(ctx_kv);
    return false;
  }

  ggml_backend_tensor_copy(sv_state.encoder_out, audio_encoder_out);
  ggml_backend_tensor_set(inp_prefix_tokens, prefix_tokens.data(), 0,
                          prefix_tokens.size() * sizeof(int32_t));
  ggml_backend_tensor_set(inp_suffix_tokens, suffix_tokens.data(), 0,
                          suffix_tokens.size() * sizeof(int32_t));

  // Position ids were built during graph construction, now set the data
  ggml_tensor *positions_tensor = result->get_input_tensor("position_ids");

  if (!positions_tensor) {
    positions_tensor = result->get_input_tensor("position_ids_seq");
  }
  if (positions_tensor) {
    result->set_position_ids(positions_tensor, positions.data(), total_T);
    if (debug_decode) {
      RS_LOG_DEBUG("Prefill: set position_ids, first=%d last=%d, n_tokens=%d",
                   (int)positions[0], (int)positions[total_T - 1], total_T);
    }
  } else {
    RS_LOG_ERR("Prefill: position_ids tensor not found!");
  }

  // Set causal mask data if needed
  ggml_tensor *causal_mask_tensor = result->get_input_tensor("causal_mask");
  if (causal_mask_tensor) {
    result->set_causal_mask(causal_mask_tensor, total_T, 0);
    if (debug_decode) {
      RS_LOG_DEBUG("Prefill: set causal_mask, shape [%lld,%lld] type=%s",
                   (long long)causal_mask_tensor->ne[0],
                   (long long)causal_mask_tensor->ne[1],
                   ggml_type_name(causal_mask_tensor->type));
    }
  } else {
    RS_LOG_ERR("Prefill: causal_mask tensor not found!");
  }

  {
    ImatrixHookGuard hook(sched, &imatrix_cb_);
    if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("DecodeWithLLM: LLM prefill compute failed");
      ggml_backend_sched_reset(sched);
      ggml_free(ctx_proj);
      ggml_backend_buffer_free(kv_gpu_buf);
      ggml_free(ctx_kv);
      return false;
    }
  }

  // 10. Get logits - now only the LAST token position is output (optimised)
  ggml_tensor *logits = result->get_logits();
  if (!logits) {
    RS_LOG_ERR("DecodeWithLLM: no logits tensor from LLM prefill");
    ggml_backend_sched_reset(sched);
    ggml_free(ctx_proj);
    ggml_backend_buffer_free(kv_gpu_buf);
    ggml_free(ctx_kv);
    return false;
  }

  // After the prefill optimisation logits->ne[1] == 1 (only last position).
  const int n_vocab = (int)logits->ne[0];
  std::vector<float> logits_host(n_vocab); // single position
  ggml_backend_tensor_get(logits, logits_host.data(), 0,
                          logits_host.size() * sizeof(float));

  // Diagnostics: check logits range at last position
  if (debug_decode) {
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
    RS_LOG_DEBUG("Prefill logits at last pos: min=%.4f max=%.4f max_idx=%d "
                 "nan=%d inf=%d (n_vocab=%d)",
                 min_logit, max_logit, max_idx, nan_count, inf_count, n_vocab);
  }

  std::vector<int32_t> token_ids;

  if (debug_decode) {
    RS_LOG_DEBUG("Sampling from last position (prefill output = 1 token)");
  }

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

  // Log EOS probability for diagnostics (compare across segments)
  const int32_t eos_token_id = llm_model_->vocab().token_eos();
  float eos_logit = (eos_token_id >= 0 && eos_token_id < n_vocab)
                        ? logits_host[eos_token_id]
                        : -INFINITY;
  if (debug_decode) {
    RS_LOG_DEBUG(
        "First token: %d (%s) | EOS=%d logit=%.4f | top_logit=%.4f id=%d",
        best_token, llm_model_->vocab().decode(best_token).c_str(),
        eos_token_id, eos_logit, best_prob, best_token);
  }

  n_cached_tokens_ = total_T;

  // Verify KV cache data integrity after prefill
  if (debug_decode) {
    RS_LOG_DEBUG("Prefill KV cache is device-resident, n_cached=%d",
                 n_cached_tokens_);
  }

  ggml_free(ctx_proj);
  ggml_backend_sched_reset(sched);

  // ==========================================
  // Autoregressive decode loop
  // ==========================================

  // Repetition detection: break on degenerate decode loops
  // Two patterns: consecutive (A→A→A...) and alternating (A→B→A→B...)
  int consecutive_same_token = 0;
  int alternating_count = 0;
  static constexpr int MAX_CONSECUTIVE_REPEAT = 10;
  static constexpr int MAX_ALTERNATING_REPEAT = 10;

  // Repetition penalty: reduce logits of already-generated tokens to
  // prevent the model from getting stuck in repetition loops. This is
  // the standard approach used by llama.cpp / vLLM / etc.
  static constexpr float REPETITION_PENALTY = 1.10f;
  static constexpr int PENALTY_WINDOW = 64; // only penalize recent tokens

  // ==========================================
  // Pre-warm gallocr: reserve GPU compute buffer at maximum size.
  //
  // Decode graph shape is fixed at n_kv_max, so one reservation covers all
  // token steps and ACL graph caching sees stable tensor shapes.
  {
    llm_build_opts warmup_opts;
    warmup_opts.output_mode   = llm_output_mode::OUTPUT_LOGITS;
    warmup_opts.skip_embeddings = false;
    warmup_opts.use_kv_cache  = true;
    warmup_opts.is_decode_step = true;
    warmup_opts.n_kv_cache    = n_kv_max - 1;
    warmup_opts.n_kv_max      = n_kv_max;
    warmup_opts.fixed_kv_cache_shape = true;
    warmup_opts.causal_mask   = true;
    warmup_opts.gpu_kv_k      = gpu_kv_k_vec.data();
    warmup_opts.gpu_kv_v      = gpu_kv_v_vec.data();

    llm_pos warmup_pos  = (llm_pos)(n_kv_max - 1 + 2);
    int32_t dummy_token = 0;
    auto warmup_result  = llm_graph_builder_->build_graph(
        &dummy_token, 1, nullptr, &warmup_pos, &warmup_opts);
    if (warmup_result) {
      // sched_reserve calls sched_reset internally; gallocr size_max persists.
      warmup_result->assign_kv_backend(sched, kv_backend, n_layer);
      ggml_backend_sched_reserve(sched, warmup_result->get_graph());
    }
  }

  // ==========================================
  // Autoregressive decode loop (with GPU-persistent KV cache)
  // ==========================================
  for (int step = 0; step < MAX_DECODE_TOKENS; ++step) {
    llm_pos decode_pos = (llm_pos)(n_cached_tokens_ + 2);
    if (debug_decode) {
      RS_LOG_DEBUG("Decode step %d: pos=%d, n_cached=%d, input_token=%d (%s)",
                   step, (int)decode_pos, n_cached_tokens_, best_token,
                   llm_model_->vocab().decode(best_token).c_str());
    }

    llm_build_opts dec_opts;
    dec_opts.output_mode = llm_output_mode::OUTPUT_LOGITS;
    dec_opts.skip_embeddings = false;
    dec_opts.use_kv_cache = true;
    dec_opts.is_decode_step = true;
    dec_opts.n_kv_cache = n_cached_tokens_;
    dec_opts.n_kv_max = n_kv_max;
    dec_opts.fixed_kv_cache_shape = true;
    dec_opts.causal_mask = true;
    dec_opts.gpu_kv_k = gpu_kv_k_vec.data();
    dec_opts.gpu_kv_v = gpu_kv_v_vec.data();

    auto dec_result = llm_graph_builder_->build_graph(&best_token, 1, nullptr,
                                                      &decode_pos, &dec_opts);

    if (!dec_result) {
      RS_LOG_ERR("Failed to build decode graph at step %d", step);
      break;
    }

    dec_result->assign_kv_backend(sched, kv_backend, n_layer);
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

    dec_result->set_kv_write_indices(n_cached_tokens_, n_layer);

    // Execute decode step
    {
      ImatrixHookGuard hook(sched, &imatrix_cb_);
      if (ggml_backend_sched_graph_compute(sched, dec_result->get_graph()) !=
          GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("Decode graph compute failed at step %d", step);
        break;
      }
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

    // Apply repetition penalty to recently generated tokens.
    // For each token that appeared in the last PENALTY_WINDOW steps,
    // divide its logit by REPETITION_PENALTY if it's positive,
    // multiply by REPETITION_PENALTY if it's negative (discourages
    // the token further).
    {
      int penalty_start =
          std::max(0, (int)token_ids.size() - PENALTY_WINDOW);
      std::unordered_set<int32_t> recent_tokens;
      for (int p = penalty_start; p < (int)token_ids.size(); ++p) {
        recent_tokens.insert(token_ids[p]);
      }
      for (int32_t tid : recent_tokens) {
        if (tid >= 0 && tid < n_vocab) {
          float &logit = dec_logits_host[tid];
          if (logit > 0.0f) {
            logit /= REPETITION_PENALTY;
          } else {
            logit *= REPETITION_PENALTY;
          }
        }
      }
    }

    int32_t next_token = 0;
    float next_prob = dec_logits_host[0];
    for (int v = 1; v < n_vocab; ++v) {
      if (dec_logits_host[v] > next_prob) {
        next_prob = dec_logits_host[v];
        next_token = v;
      }
    }

    // Track consecutive duplicate tokens (A→A→A...)
    if (next_token == best_token) {
      consecutive_same_token++;
    } else {
      consecutive_same_token = 0;
    }

    // Track alternating pattern (A→B→A→B...) by checking if this
    // token matches the token 2 steps back but differs from the
    // immediately preceding one.
    if (token_ids.size() >= 2 && next_token == token_ids[token_ids.size() - 2] &&
        next_token != token_ids.back()) {
      alternating_count++;
    } else {
      alternating_count = 0;
    }

    n_cached_tokens_++;

    if (next_token == eos_token_id) {
      RS_LOG_INFO("EOS token reached at step %d", step);
      ggml_backend_sched_reset(sched);
      break;
    }

    if (consecutive_same_token >= MAX_CONSECUTIVE_REPEAT) {
      RS_LOG_INFO("WARNING: token %d (%s) repeated %d times consecutively — "
                  "forcing stop",
                  next_token, llm_model_->vocab().decode(next_token).c_str(),
                  consecutive_same_token);
      ggml_backend_sched_reset(sched);
      break;
    }

    if (alternating_count >= MAX_ALTERNATING_REPEAT) {
      RS_LOG_INFO("WARNING: alternating pattern %d→%d→%d repeated %d times — "
                  "forcing stop",
                  next_token,
                  token_ids.size() >= 1 ? token_ids.back() : -1,
                  next_token,
                  alternating_count);
      ggml_backend_sched_reset(sched);
      break;
    }

    token_ids.push_back(next_token);
    best_token = next_token;

    ggml_backend_sched_reset(sched);
  }

  // Free GPU KV cache buffer and tensor context
  if (kv_gpu_buf) {
    ggml_backend_buffer_free(kv_gpu_buf);
    kv_gpu_buf = nullptr;
  }
  if (ctx_kv) {
    ggml_free(ctx_kv);
    ctx_kv = nullptr;
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
  {
    ImatrixHookGuard hook(sched, &imatrix_cb_);
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
      ggml_free(ctx0);
      return false;
    }
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

void FunASRNanoModel::SetUseLLM(bool use) {
  runtime_use_llm_ = use && hparams_.use_llm && llm_model_;
}

void FunASRNanoModel::SetCTCPrecheck(bool enable) {
  ctc_precheck_ = enable;
}

void FunASRNanoModel::SetUserInputPrompt(const std::string &prompt) {
  user_input_prompt_ = prompt;
  active_prompt_ = prompt;
  cached_user_input_prompt_.clear();
  cached_prefix_tokens_.clear();
  cached_suffix_tokens_.clear();
}

void FunASRNanoModel::SetHotwords(const std::vector<std::string> &hotwords) {
  hotwords_ = hotwords;
  if (hotwords_.empty()) {
    hotword_retriever_.reset();
    return;
  }
  // Optional runtime tuning without a rebuild.
  if (const char *e = std::getenv("RS_HOTWORD_TAU"))
    hotword_tau_ = std::atof(e);
  if (const char *e = std::getenv("RS_HOTWORD_TOPK"))
    hotword_topk_ = std::atoi(e);
  if (const char *e = std::getenv("RS_HOTWORD_FORCE"))
    hotword_force_ = std::atoi(e) != 0;
  if (!hotword_retriever_) {
    hotword_retriever_ = std::make_unique<rs::HotwordRetriever>();
    hotword_retriever_->Load(); // warns + degrades if pinyin data missing
  }
  hotword_retriever_->SetHotwords(hotwords_);
  RS_LOG_INFO("Hotword config: tau=%.3f topk=%d", hotword_tau_, hotword_topk_);
}

bool FunASRNanoModel::DecodeBatch(const std::vector<RSState *> &states,
                                  ggml_backend_sched_t sched) {
  if (states.empty()) return true;
  if (states.size() == 1) return Decode(*states[0], sched);

  // LLM/hotword/precheck paths have per-request prompt/KV decisions. Batched
  // CTC covers --two-pass first pass and no-LLM serving.
  const bool want_hotwords = hotword_retriever_ &&
                             !hotword_retriever_->Empty() &&
                             runtime_use_llm_ && llm_model_;
  if (runtime_use_llm_ || want_hotwords) {
    return ISpeechModel::DecodeBatch(states, sched);
  }

  const int n_batch = (int)states.size();
  auto &first = static_cast<SenseVoiceState &>(*states[0]);
  if (!first.encoder_out) return false;
  const int C = first.encoder_out->ne[0];
  const int T = first.encoder_out->ne[1];
  const int V = hparams_.n_vocab;
  for (auto *state : states) {
    if (!state) return false;
    auto &sv_state = static_cast<SenseVoiceState &>(*state);
    if (!sv_state.encoder_out ||
        sv_state.encoder_out->ne[0] != C ||
        sv_state.encoder_out->ne[1] != T) {
      return false;
    }
  }

  struct ggml_context *ctx0 =
      ggml_init({4 * 1024 * ggml_tensor_overhead(), nullptr, true});
  struct ggml_cgraph *gf =
      ggml_new_graph_custom(ctx0, FUNASR_NANO_DECODER_MAX_NODES, false);

  struct ggml_tensor *encoder_in =
      ggml_new_tensor_3d(ctx0, first.encoder_out->type, C, T, n_batch);
  ggml_set_input(encoder_in);

  struct ggml_tensor *cur =
      decoder_forward(hparams_, ctx0, encoder_in, *ctc_decoder_);
  cur = ggml_mul_mat(ctx0, ctc_decoder_->ctc_out_linear_weight, cur);
  cur = ggml_add(ctx0, cur, ctc_decoder_->ctc_out_linear_bias);
  ggml_set_name(cur, "funasr_ctc_logits_batch");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    ggml_free(ctx0);
    return false;
  }

  const size_t bytes_per_state =
      ggml_row_size(first.encoder_out->type, C) * (size_t)T;
  std::vector<uint8_t> enc_tmp(bytes_per_state);
  for (int b = 0; b < n_batch; ++b) {
    auto &sv_state = static_cast<SenseVoiceState &>(*states[b]);
    ggml_backend_tensor_get(sv_state.encoder_out, enc_tmp.data(), 0,
                            bytes_per_state);
    ggml_backend_tensor_set(encoder_in, enc_tmp.data(),
                            (size_t)b * encoder_in->nb[2], bytes_per_state);
  }

  ggml_status status;
  {
    ImatrixHookGuard hook(sched, &imatrix_cb_);
    status = ggml_backend_sched_graph_compute(sched, gf);
  }
  if (status != GGML_STATUS_SUCCESS) {
    ggml_free(ctx0);
    return false;
  }

  const int T_out = (int)cur->ne[1];
  const size_t logits_bytes_per_state = (size_t)V * T_out * sizeof(float);
  std::vector<float> logits_host((size_t)V * T_out);
  std::vector<int32_t> raw_ids(T_out);
  for (int b = 0; b < n_batch; ++b) {
    auto &sv_state = static_cast<SenseVoiceState &>(*states[b]);
    ggml_backend_tensor_get(cur, logits_host.data(), (size_t)b * cur->nb[2],
                            logits_bytes_per_state);
    for (int t = 0; t < T_out; ++t) {
      const float *row = logits_host.data() + (size_t)t * V;
      int best = 0;
      float best_v = row[0];
      for (int v = 1; v < V; ++v) {
        if (row[v] > best_v) {
          best_v = row[v];
          best = v;
        }
      }
      raw_ids[t] = best;
    }
    sv_state.ids = CTCDecoder::GreedyDecode(raw_ids.data(), T_out);
    sv_state.tokens.clear();
    if (vocab_.has_ctc_tokens) {
      for (auto id : sv_state.ids) {
        auto it = vocab_.id_to_token.find(id);
        if (it != vocab_.id_to_token.end()) {
          sv_state.tokens.push_back(decode_ctc_token_bytes(it->second));
        }
      }
    }
  }

  ggml_free(ctx0);
  return true;
}

std::string FunASRNanoModel::BuildHotwordPrompt(
    const std::vector<std::string> &hotwords) const {
  std::string joined;
  for (size_t i = 0; i < hotwords.size(); ++i) {
    if (i)
      joined += ", ";
    joined += hotwords[i];
  }
  // Default matches Fun-ASR model.py get_prompt() byte-for-byte. Set
  // RS_HOTWORD_PROMPT_STYLE=strong to use a more forceful instruction that
  // tells the model to emit the exact hotword form (incl. rare characters).
  const char *style = std::getenv("RS_HOTWORD_PROMPT_STYLE");
  std::string p;
  if (style && std::string(style) == "strong") {
    p = "请把下面这段语音转写为文本。如果语音中出现了与热词列表读音相同或相近的词，"
        "必须严格按照热词列表中的写法输出（即使是生僻字也要照写，不要替换成其他同音字）。"
        "\n\n热词列表：[" + joined + "]\n" + user_input_prompt_;
  } else {
    p = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，"
        "我们会留空。\n\n\n**上下文信息：**\n\n\n";
    p += "热词列表：[" + joined + "]\n";
    p += user_input_prompt_;
  }
  return p;
}

/**
 * Enhanced Decode function.
 *
 * When LLM is enabled, a fast CTC pre-check runs first to detect silence.
 * If CTC produces only blank tokens the audio is noise/silence and the
 * expensive LLM decode is skipped, preventing hallucinated output.
 */
bool FunASRNanoModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  auto &sv_state = static_cast<SenseVoiceState &>(state);

  if (!sv_state.encoder_out)
    return false;

  // ── CTC pass: silence pre-check and/or hotword retrieval ──
  //   * hotword RAG   — needs a real CTC hypothesis, so it runs the FULL CTC
  //                     decoder (DecodeWithoutLLM) and retrieves from it.
  //   * ctc_precheck_ — only needs "is there any non-blank token", so when no
  //                     hotwords are active it uses a cheap ctc_lo-only pass.
  bool has_speech = true;
  std::vector<std::string> retrieved;

  const bool want_hotwords = hotword_retriever_ &&
                             !hotword_retriever_->Empty() &&
                             runtime_use_llm_ && llm_model_;

  if (want_hotwords) {
    // Full CTC decode → collapsed hypothesis ids in sv_state.ids.
    if (DecodeWithoutLLM(state, sched)) {
      std::string ctc_text;
      for (int32_t id : sv_state.ids) {
        auto it = vocab_.id_to_token.find(id);
        if (it != vocab_.id_to_token.end())
          ctc_text += decode_ctc_token_bytes(it->second);
      }
      if (ctc_precheck_)
        has_speech = !sv_state.ids.empty();
      if (has_speech) {
        retrieved =
            hotword_retriever_->Retrieve(ctc_text, hotword_tau_, hotword_topk_);
        RS_LOG_INFO("Hotword: CTC='%s' retrieved %zu/%zu hotwords",
                    ctc_text.c_str(), retrieved.size(),
                    hotword_retriever_->Size());
      }
    }
    // Clear the CTC tokens so they don't leak into the final transcription and
    // reset the scheduler for a fresh LLM graph.
    sv_state.tokens.clear();
    ggml_backend_sched_reset(sched);
  } else if (ctc_precheck_ && runtime_use_llm_ && llm_model_) {
    has_speech = false;

    struct ggml_init_params ctc_params = {2 * 1024 * ggml_tensor_overhead(),
                                          nullptr, true};
    struct ggml_context *ctx_ctc = ggml_init(ctc_params);
    struct ggml_cgraph *gf_ctc =
        ggml_new_graph_custom(ctx_ctc, 256, false);

    struct ggml_tensor *enc_in = ggml_new_tensor_2d(
        ctx_ctc, sv_state.encoder_out->type, sv_state.encoder_out->ne[0],
        sv_state.encoder_out->ne[1]);
    ggml_set_input(enc_in);

    struct ggml_tensor *cur = ggml_mul_mat(ctx_ctc,
        ctc_decoder_->ctc_out_linear_weight, enc_in);
    cur = ggml_add(ctx_ctc, cur, ctc_decoder_->ctc_out_linear_bias);
    cur = ggml_argmax(ctx_ctc, cur);
    ggml_set_output(cur);
    ggml_build_forward_expand(gf_ctc, cur);

    if (ggml_backend_sched_alloc_graph(sched, gf_ctc)) {
      ggml_backend_tensor_copy(sv_state.encoder_out, enc_in);
      ggml_status ctc_status;
      {
        ImatrixHookGuard hook(sched, &imatrix_cb_);
        ctc_status = ggml_backend_sched_graph_compute(sched, gf_ctc);
      }
      if (ctc_status == GGML_STATUS_SUCCESS) {
        int T = sv_state.encoder_out->ne[1];
        raw_ids.resize(T);
        ggml_backend_tensor_get(cur, raw_ids.data(), 0,
                                T * sizeof(int32_t));
        for (int i = 0; i < T; ++i) {
          if (raw_ids[i] != 0) { // non-blank token → speech
            has_speech = true;
            break;
          }
        }
      }
    }
    ggml_free(ctx_ctc);

    // Reset scheduler so the main decode path can allocate a fresh graph.
    ggml_backend_sched_reset(sched);
  }

  if (!has_speech) {
    sv_state.tokens.clear();
    return true;
  }

  // Compose the prompt for this utterance: hotword-injected when we retrieved
  // anything, otherwise the plain user prompt (zero regression when unused).
  active_prompt_ =
      retrieved.empty() ? user_input_prompt_ : BuildHotwordPrompt(retrieved);

  // Choose decode path based on LLM availability AND runtime toggle
  bool success = false;
  if (runtime_use_llm_ && llm_model_) {
    RS_LOG_INFO("Decoding with Qwen3 LLM");
    success = DecodeWithLLM(state, sched);
  } else {
    success = DecodeWithoutLLM(state, sched);
  }

  if (!success) {
    return false;
  }

  // Convert token IDs to text (for non-LLM path)
  if (!runtime_use_llm_ || !llm_model_) {
    if (vocab_.has_ctc_tokens) {
      for (auto id : sv_state.ids) {
        auto it = vocab_.id_to_token.find(id);
        if (it != vocab_.id_to_token.end()) {
          sv_state.tokens.push_back(decode_ctc_token_bytes(it->second));
        }
      }
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

  // Deterministic hotword forcing: rewrite homophone spans to the exact hotword
  // spelling the LLM refused to emit .
  if (hotword_force_ && hotword_retriever_ && !hotword_retriever_->Empty()) {
    std::string forced = hotword_retriever_->ApplySubstitutions(result);
    if (forced != result) {
      RS_LOG_INFO("Hotword force: '%s' -> '%s'", result.c_str(),
                  forced.c_str());
      result = std::move(forced);
    }
  }
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
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
global_FunASRNano_reg;
} // namespace
