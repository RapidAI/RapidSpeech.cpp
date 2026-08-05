#include "moss_transcribe.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>

// Encoder graph node budget — 24 encoder layers + conv stem + adaptor.
#ifndef MOSS_TRANSCRIBE_ENC_MAX_NODES
#define MOSS_TRANSCRIBE_ENC_MAX_NODES 8192
#endif

// ============================================================================
// Construction
// ============================================================================

namespace {
// Default prompt from moss_transcribe_diarize.inference_utils.DEFAULT_PROMPT.
const char *kDefaultPrompt =
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。";
} // namespace

MossTranscribeModel::MossTranscribeModel() : user_input_prompt_(kDefaultPrompt) {}
MossTranscribeModel::~MossTranscribeModel() = default;

void MossTranscribeModel::SetUserInputPrompt(const std::string &prompt) {
  user_input_prompt_ = prompt;
}

std::shared_ptr<RSState> MossTranscribeModel::CreateState() {
  return std::make_shared<MossTranscribeState>();
}

// ============================================================================
// Encoder tensor mapping. Names match scripts/convert_moss_transcribe_to_gguf.py
// ============================================================================

namespace {

inline struct ggml_tensor *
find_opt(std::map<std::string, struct ggml_tensor *> &tensors,
         const std::string &name) {
  auto it = tensors.find(name);
  return it != tensors.end() ? it->second : nullptr;
}

inline struct ggml_tensor *
find_req(std::map<std::string, struct ggml_tensor *> &tensors,
         const std::string &name) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    RS_LOG_ERR("MossTranscribe: missing required tensor %s", name.c_str());
    return nullptr;
  }
  return it->second;
}

} // namespace

bool MossTranscribeModel::MapEncoderTensors(
    std::map<std::string, struct ggml_tensor *> &tensors) {
  for (int i = 0; i < 2; ++i) {
    const std::string base = "a.conv1d." + std::to_string(i);
    weights_.conv1d_w[i] = find_req(tensors, base + ".weight");
    weights_.conv1d_b[i] = find_req(tensors, base + ".bias");
    if (!weights_.conv1d_w[i] || !weights_.conv1d_b[i]) return false;
  }

  weights_.position_embd = find_req(tensors, "a.position_embd.weight");
  if (!weights_.position_embd) return false;

  weights_.post_norm_w = find_req(tensors, "a.post_norm.weight");
  weights_.post_norm_b = find_req(tensors, "a.post_norm.bias");
  if (!weights_.post_norm_w || !weights_.post_norm_b) return false;

  weights_.mm_1_w = find_req(tensors, "mm.a.mlp.0.weight");
  weights_.mm_1_b = find_req(tensors, "mm.a.mlp.0.bias");
  weights_.mm_2_w = find_req(tensors, "mm.a.mlp.1.weight");
  weights_.mm_2_b = find_req(tensors, "mm.a.mlp.1.bias");
  weights_.mm_norm_w = find_req(tensors, "mm.a.norm.weight");
  weights_.mm_norm_b = find_req(tensors, "mm.a.norm.bias");
  if (!weights_.mm_1_w || !weights_.mm_1_b || !weights_.mm_2_w ||
      !weights_.mm_2_b || !weights_.mm_norm_w || !weights_.mm_norm_b) {
    return false;
  }

  weights_.layers.resize(hparams_.enc_n_layer);
  for (int il = 0; il < hparams_.enc_n_layer; ++il) {
    const std::string p = "a.blk." + std::to_string(il) + ".";
    auto &L = weights_.layers[il];
    L.attn_norm_w = find_req(tensors, p + "attn_norm.weight");
    L.attn_norm_b = find_req(tensors, p + "attn_norm.bias");
    L.q_w = find_req(tensors, p + "attn_q.weight");
    L.q_b = find_opt(tensors, p + "attn_q.bias");
    L.k_w = find_req(tensors, p + "attn_k.weight"); // Whisper: no k bias
    L.v_w = find_req(tensors, p + "attn_v.weight");
    L.v_b = find_opt(tensors, p + "attn_v.bias");
    L.o_w = find_req(tensors, p + "attn_o.weight");
    L.o_b = find_opt(tensors, p + "attn_o.bias");
    L.ffn_norm_w = find_req(tensors, p + "ffn_norm.weight");
    L.ffn_norm_b = find_req(tensors, p + "ffn_norm.bias");
    L.ffn_up_w   = find_req(tensors, p + "ffn_up.weight");
    L.ffn_up_b   = find_opt(tensors, p + "ffn_up.bias");
    L.ffn_dn_w   = find_req(tensors, p + "ffn_down.weight");
    L.ffn_dn_b   = find_opt(tensors, p + "ffn_down.bias");

    if (!L.attn_norm_w || !L.attn_norm_b || !L.q_w || !L.k_w || !L.v_w ||
        !L.o_w || !L.ffn_norm_w || !L.ffn_norm_b || !L.ffn_up_w ||
        !L.ffn_dn_w) {
      return false;
    }
  }
  return true;
}

// ============================================================================
// LLM loading — identical to Qwen3ASR's path (llm_model with tied lm_head).
// ============================================================================

bool MossTranscribeModel::LoadLLM(
    struct gguf_context *ctx_gguf,
    std::map<std::string, struct ggml_tensor *> &tensors,
    ggml_backend_t backend) {
  (void)backend;
  llm_model_ = std::make_shared<llm_model>();
  if (!llm_model_->load_metadata_from_gguf(ctx_gguf)) {
    RS_LOG_ERR("MossTranscribe: failed to load LLM metadata");
    llm_model_.reset();
    return false;
  }
  if (llm_model_->arch() != LLM_ARCH_QWEN3) {
    RS_LOG_ERR("MossTranscribe: LLM is not Qwen3 (got %d)",
               (int)llm_model_->arch());
    llm_model_.reset();
    return false;
  }
  if (!llm_model_->map_tensors_qwen3(tensors)) {
    RS_LOG_ERR("MossTranscribe: failed to map Qwen3 LLM tensors");
    llm_model_.reset();
    return false;
  }
  auto &vocab = const_cast<llm_vocab &>(llm_model_->vocab());
  vocab.load_qwen3_special_tokens();
  // MOSS audio placeholder tokens (override the FunASRNano speech-token
  // aliases at 151669/151670).
  vocab.add_special_token("<|audio_start|>", hparams_.audio_start_id);
  vocab.add_special_token("<|audio_end|>", hparams_.audio_end_id);
  vocab.add_special_token("<|audio_pad|>", hparams_.audio_pad_id);

  // Digit token ids for the time anchors inside the audio span.
  for (int d = 0; d < 10; ++d) {
    const std::string s(1, (char)('0' + d));
    auto ids = vocab.tokenize(s, false);
    if (ids.size() != 1) {
      RS_LOG_ERR("MossTranscribe: digit '%d' is not a single token", d);
      return false;
    }
    digit_token_ids_[d] = ids[0];
  }

  RS_LOG_INFO("MossTranscribe LLM loaded: layers=%d, embd=%d, heads=%d, "
              "vocab=%d, eos=%d",
              llm_model_->hparams().n_layer, llm_model_->hparams().n_embd,
              llm_model_->hparams().n_head, llm_model_->hparams().n_vocab,
              llm_model_->vocab().token_eos());
  return true;
}

// ============================================================================
// Load
// ============================================================================

bool MossTranscribeModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                               ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("MossTranscribe::Load: invalid context");
    return false;
  }
  gguf_context *ctx_gguf  = ctx->ctx_gguf;
  ggml_context *gguf_data = ctx->gguf_data;

  auto get_i32 = [&](const char *key, int32_t def) -> int32_t {
    int idx = gguf_find_key(ctx_gguf, key);
    if (idx == -1) return def;
    auto type = gguf_get_kv_type(ctx_gguf, idx);
    if (type == GGUF_TYPE_UINT32) return (int32_t)gguf_get_val_u32(ctx_gguf, idx);
    return gguf_get_val_i32(ctx_gguf, idx);
  };
  auto get_f32 = [&](const char *key, float def) -> float {
    int idx = gguf_find_key(ctx_gguf, key);
    if (idx == -1) return def;
    return gguf_get_val_f32(ctx_gguf, idx);
  };

  hparams_.n_mels        = get_i32("mosstd.encoder.n_mels", 80);
  hparams_.n_fft         = get_i32("mosstd.encoder.n_fft", 400);
  hparams_.hop_length    = get_i32("mosstd.encoder.hop_length", 160);
  hparams_.sample_rate   = get_i32("mosstd.encoder.sample_rate", 16000);
  hparams_.chunk_frames  = get_i32("mosstd.encoder.chunk_frames", 3000);
  hparams_.enc_d_model   = get_i32("mosstd.encoder.d_model", 1024);
  hparams_.enc_n_head    = get_i32("mosstd.encoder.n_head", 16);
  hparams_.enc_n_layer   = get_i32("mosstd.encoder.n_layer", 24);
  hparams_.enc_ffn_dim   = get_i32("mosstd.encoder.ffn_dim", 4096);
  hparams_.enc_max_pos   = get_i32("mosstd.encoder.max_source_positions", 1500);
  hparams_.enc_norm_eps  = get_f32("mosstd.encoder.layer_norm_eps", 1e-5f);

  hparams_.merge_size       = get_i32("mosstd.adaptor.merge_size", 4);
  hparams_.adaptor_in       = get_i32("mosstd.adaptor.in_dim", 4096);
  hparams_.adaptor_norm_eps = get_f32("mosstd.adaptor.norm_eps", 1e-6f);

  hparams_.audio_pad_id   = get_i32("mosstd.audio.pad_token_id", 151671);
  hparams_.audio_start_id = get_i32("mosstd.audio.start_token_id", 151669);
  hparams_.audio_end_id   = get_i32("mosstd.audio.end_token_id", 151670);
  hparams_.time_marker_enable =
      get_i32("mosstd.time_marker.enable", 1) != 0;
  hparams_.time_marker_every_seconds =
      get_i32("mosstd.time_marker.every_seconds", 5);
  hparams_.audio_tokens_per_second =
      get_f32("mosstd.audio.tokens_per_second", 12.5f);

  int qwen3_block_count_idx = gguf_find_key(ctx_gguf, "qwen3.block_count");
  if (qwen3_block_count_idx == -1) {
    RS_LOG_ERR("MossTranscribe: LLM section missing in GGUF");
    return false;
  }
  hparams_.n_llm_layer = gguf_get_val_i32(ctx_gguf, qwen3_block_count_idx);
  hparams_.n_llm_embd  = get_i32("qwen3.embedding_length", 0);

  meta_.arch_name             = "MossTranscribeDiarize";
  meta_.audio_sample_rate     = hparams_.sample_rate;
  meta_.n_mels                = hparams_.n_mels;
  meta_.vocab_size            = get_i32("tokenizer.vocab_size", 0);
  meta_.use_external_frontend = true;

  RS_LOG_INFO("MossTranscribe hparams: enc d=%d h=%d L=%d ff=%d mels=%d  "
              "merge=%d adaptor_in=%d  LLM L=%d d=%d",
              hparams_.enc_d_model, hparams_.enc_n_head, hparams_.enc_n_layer,
              hparams_.enc_ffn_dim, hparams_.n_mels, hparams_.merge_size,
              hparams_.adaptor_in, hparams_.n_llm_layer, hparams_.n_llm_embd);

  // Mel extractor: HF WhisperFeatureExtractor semantics — per-30s chunk,
  // periodic Hann window, drop trailing STFT frame (3001 -> 3000).
  WhisperMelConfig mel_cfg;
  mel_cfg.sample_rate     = hparams_.sample_rate;
  mel_cfg.n_fft           = hparams_.n_fft;
  mel_cfg.hop_length      = hparams_.hop_length;
  mel_cfg.n_mels          = hparams_.n_mels;
  mel_cfg.f_min           = 0.0f;
  mel_cfg.f_max           = (float)hparams_.sample_rate / 2.0f;
  mel_cfg.chunk_size      = hparams_.chunk_frames;
  mel_cfg.periodic_window = true;
  mel_cfg.max_frames      = hparams_.chunk_frames;
  mel_ = std::make_unique<WhisperMelExtractor>(mel_cfg);

  std::map<std::string, struct ggml_tensor *> tensors;
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }

  if (!MapEncoderTensors(tensors)) {
    RS_LOG_ERR("MossTranscribe: encoder tensor mapping failed");
    return false;
  }
  return LoadLLM(ctx_gguf, tensors, backend);
}

// ============================================================================
// Encoder graph (one 30 s chunk) + execute
// ============================================================================

namespace {

struct MossEncGraphHelper {
  ggml_context *ctx;
  const MossTranscribeHParams &hp;
  const MossTranscribeWeights &w;

  // Pre-LN bidirectional transformer block (Whisper encoder layer).
  ggml_tensor *build_layer(ggml_tensor *cur, int il, int n_tokens) const {
    const auto &L = w.layers[il];
    const int n_head   = hp.enc_n_head;
    const int d_model  = hp.enc_d_model;
    const int head_dim = d_model / n_head;

    // attention residual branch
    ggml_tensor *residual = cur;
    {
      cur = ggml_norm(ctx, cur, hp.enc_norm_eps);
      cur = ggml_add(ctx, ggml_mul(ctx, cur, L.attn_norm_w), L.attn_norm_b);

      ggml_tensor *Q = ggml_mul_mat(ctx, L.q_w, cur);
      if (L.q_b) Q = ggml_add(ctx, Q, L.q_b);
      ggml_tensor *K = ggml_mul_mat(ctx, L.k_w, cur); // no k bias
      ggml_tensor *V = ggml_mul_mat(ctx, L.v_w, cur);
      if (L.v_b) V = ggml_add(ctx, V, L.v_b);

      Q = ggml_reshape_3d(ctx, Q, head_dim, n_head, n_tokens);
      K = ggml_reshape_3d(ctx, K, head_dim, n_head, n_tokens);
      V = ggml_reshape_3d(ctx, V, head_dim, n_head, n_tokens);

      Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
      K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
      V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
      // shape: [head_dim, n_tokens, n_head]

      const float scale = 1.0f / std::sqrt((float)head_dim);
      ggml_tensor *scores = ggml_mul_mat(ctx, K, Q);
      scores = ggml_scale_inplace(ctx, scores, scale);
      ggml_tensor *probs = ggml_soft_max_inplace(ctx, scores);

      ggml_tensor *Vt = ggml_cont(ctx, ggml_permute(ctx, V, 1, 0, 2, 3));
      ggml_tensor *ctxv = ggml_mul_mat(ctx, Vt, probs);
      ctxv = ggml_cont(ctx, ggml_permute(ctx, ctxv, 0, 2, 1, 3));
      ctxv = ggml_reshape_2d(ctx, ctxv, d_model, n_tokens);

      cur = ggml_mul_mat(ctx, L.o_w, ctxv);
      if (L.o_b) cur = ggml_add(ctx, cur, L.o_b);
      cur = ggml_add(ctx, cur, residual);
    }

    // FFN residual branch (Pre-LN, GELU-erf, gateless).
    {
      ggml_tensor *r2 = cur;
      cur = ggml_norm(ctx, cur, hp.enc_norm_eps);
      cur = ggml_add(ctx, ggml_mul(ctx, cur, L.ffn_norm_w), L.ffn_norm_b);

      cur = ggml_mul_mat(ctx, L.ffn_up_w, cur);
      if (L.ffn_up_b) cur = ggml_add(ctx, cur, L.ffn_up_b);
      cur = ggml_gelu_erf(ctx, cur);
      cur = ggml_mul_mat(ctx, L.ffn_dn_w, cur);
      if (L.ffn_dn_b) cur = ggml_add(ctx, cur, L.ffn_dn_b);

      cur = ggml_add(ctx, cur, r2);
    }
    return cur;
  }
};

} // namespace

bool MossTranscribeModel::RunEncoderChunk(const std::vector<float> &mel_features,
                                          int token_len, RSState &state,
                                          ggml_backend_sched_t sched) {
  auto &st = static_cast<MossTranscribeState &>(state);
  const int n_mel    = hparams_.n_mels;
  const int n_frames = hparams_.chunk_frames;          // 3000
  const int enc_T    = n_frames / 2;                   // 1500 after conv stride 2
  const int merge    = hparams_.merge_size;
  const int valid_T  = token_len * merge;              // frames used downstream
  if (valid_T <= 0 || valid_T > enc_T) {
    RS_LOG_ERR("MossTranscribe: invalid token_len=%d (enc_T=%d)", token_len,
               enc_T);
    return false;
  }

  struct ggml_init_params params = {
      MOSS_TRANSCRIBE_ENC_MAX_NODES * ggml_tensor_overhead() + (1u << 20),
      nullptr,
      true};
  struct ggml_context *ctx0 = ggml_init(params);
  struct ggml_cgraph *gf =
      ggml_new_graph_custom(ctx0, MOSS_TRANSCRIBE_ENC_MAX_NODES, false);

  // 1. Input mel [n_frames, n_mel] (frames fast — matches WhisperMelExtractor).
  ggml_tensor *inp_mel =
      ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_frames, n_mel);
  ggml_set_name(inp_mel, "inp_mel");
  ggml_set_input(inp_mel);

  // 2. Conv1d stem: conv0 k3 s1 p1 (n_mels -> d), conv1 k3 s2 p1 (d -> d).
  ggml_tensor *cur = ggml_conv_1d(ctx0, weights_.conv1d_w[0], inp_mel, 1, 1, 1);
  cur = ggml_add(ctx0, cur,
                 ggml_reshape_2d(ctx0, weights_.conv1d_b[0], 1,
                                 weights_.conv1d_b[0]->ne[0]));
  cur = ggml_gelu_erf(ctx0, cur);
  cur = ggml_conv_1d(ctx0, weights_.conv1d_w[1], cur, 2, 1, 1);
  cur = ggml_add(ctx0, cur,
                 ggml_reshape_2d(ctx0, weights_.conv1d_b[1], 1,
                                 weights_.conv1d_b[1]->ne[0]));
  cur = ggml_gelu_erf(ctx0, cur);
  // cur: [enc_T, d_model] — transpose to hidden-fast [d_model, enc_T].
  cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

  // 3. Learned absolute position embeddings (full 1500-frame table).
  cur = ggml_add(ctx0, cur, weights_.position_embd);

  // 4. Transformer encoder.
  MossEncGraphHelper helper{ctx0, hparams_, weights_};
  for (int il = 0; il < hparams_.enc_n_layer; ++il) {
    cur = helper.build_layer(cur, il, enc_T);
  }

  // 5. Final LayerNorm.
  cur = ggml_norm(ctx0, cur, hparams_.enc_norm_eps);
  cur = ggml_add(ctx0, ggml_mul(ctx0, cur, weights_.post_norm_w),
                 weights_.post_norm_b);

  // 6. Slice valid frames, 4x temporal merge, VQAdaptor.
  cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, cur->ne[0], valid_T,
                                     cur->nb[1], 0));
  cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * merge, token_len);

  cur = ggml_mul_mat(ctx0, weights_.mm_1_w, cur);
  cur = ggml_add(ctx0, cur, weights_.mm_1_b);
  cur = ggml_silu(ctx0, cur);
  cur = ggml_mul_mat(ctx0, weights_.mm_2_w, cur);
  cur = ggml_add(ctx0, cur, weights_.mm_2_b);
  cur = ggml_norm(ctx0, cur, hparams_.adaptor_norm_eps);
  cur = ggml_add(ctx0, ggml_mul(ctx0, cur, weights_.mm_norm_w),
                 weights_.mm_norm_b);
  // cur: [n_llm_embd, token_len]
  ggml_set_name(cur, "audio_embeds_out");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("MossTranscribe: failed to allocate encoder graph");
    ggml_free(ctx0);
    return false;
  }
  ggml_backend_tensor_set(inp_mel, mel_features.data(), 0,
                          mel_features.size() * sizeof(float));
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("MossTranscribe: encoder graph compute failed");
    ggml_free(ctx0);
    return false;
  }

  const int n_embd = (int)cur->ne[0];
  const size_t old = st.audio_embeds.size();
  st.audio_embeds.resize(old + (size_t)n_embd * token_len);
  ggml_backend_tensor_get(cur, st.audio_embeds.data() + old, 0,
                          (size_t)n_embd * token_len * sizeof(float));
  st.T_audio += token_len;
  st.n_embd = n_embd;

  if (std::getenv("RS_MOSS_DEBUG")) {
    const float *e = st.audio_embeds.data() + old;
    double s0 = 0, s2 = 0;
    for (int i = 0; i < n_embd; ++i) { s0 += e[i]; s2 += (double)e[i] * e[i]; }
    RS_LOG_INFO("MossTranscribe dbg: chunk embed[0] mean=%.4f rms=%.4f "
                "first={%.4f,%.4f,%.4f,%.4f}",
                s0 / n_embd, std::sqrt(s2 / n_embd), e[0], e[1], e[2], e[3]);
  }

  ggml_free(ctx0);
  ggml_backend_sched_reset(sched);
  return true;
}

bool MossTranscribeModel::Encode(const std::vector<float> &input_frames,
                                 RSState &state, ggml_backend_sched_t sched) {
  auto &st = static_cast<MossTranscribeState &>(state);
  if (!mel_) {
    RS_LOG_ERR("MossTranscribe::Encode: mel extractor uninitialised");
    return false;
  }
  if (input_frames.empty()) return false;

  st.audio_embeds.clear();
  st.T_audio = 0;

  // 30 s chunking exactly like MossTranscribeDiarizeProcessor._chunk_audio:
  // each chunk is padded (zeros) to n_samples before the mel transform, and
  // contributes (n-1) // (hop * 2 * merge) + 1 audio tokens.
  const int64_t n_samples_chunk =
      (int64_t)hparams_.chunk_frames * hparams_.hop_length; // 480000
  const int64_t stride =
      (int64_t)hparams_.hop_length * 2 * hparams_.merge_size; // 1280

  const int64_t total = (int64_t)input_frames.size();
  for (int64_t start = 0; start < total; start += n_samples_chunk) {
    const int64_t n_raw = std::min(n_samples_chunk, total - start);
    const int token_len = (int)((n_raw - 1) / stride + 1);

    std::vector<float> chunk((size_t)n_samples_chunk, 0.0f);
    std::memcpy(chunk.data(), input_frames.data() + start,
                (size_t)n_raw * sizeof(float));

    std::vector<float> mel;
    int n_frames = mel_->Compute(chunk, mel);
    if (n_frames != hparams_.chunk_frames) {
      RS_LOG_ERR("MossTranscribe: unexpected mel frame count %d (want %d)",
                 n_frames, hparams_.chunk_frames);
      return false;
    }
    if (std::getenv("RS_MOSS_DEBUG")) {
      double s = 0;
      for (float v : mel) s += v;
      RS_LOG_INFO("MossTranscribe dbg: mel[0,:4]={%.4f,%.4f,%.4f,%.4f} "
                  "mean=%.4f",
                  mel[0], mel[1], mel[2], mel[3], s / mel.size());
    }
    if (!RunEncoderChunk(mel, token_len, state, sched)) return false;
  }

  RS_LOG_INFO("MossTranscribe encoder: T_audio=%d n_embd=%d (%.1f s audio)",
              st.T_audio, st.n_embd,
              (float)total / hparams_.sample_rate);
  return st.T_audio > 0;
}

// ============================================================================
// Prompt construction
// ============================================================================

std::vector<int32_t>
MossTranscribeModel::BuildAudioSpanIds(int n_audio_tokens) const {
  std::vector<int32_t> out;
  out.reserve(n_audio_tokens + 32);

  const int every = hparams_.time_marker_every_seconds;
  if (!hparams_.time_marker_enable || n_audio_tokens <= 0 || every <= 0) {
    out.assign(std::max(n_audio_tokens, 0), hparams_.audio_pad_id);
    return out;
  }
  const int tokens_per_marker =
      (int)(hparams_.audio_tokens_per_second * every);
  if (tokens_per_marker <= 0) {
    out.assign(n_audio_tokens, hparams_.audio_pad_id);
    return out;
  }

  const float duration =
      (float)n_audio_tokens / hparams_.audio_tokens_per_second;
  int consumed = 0;
  for (int sec = every; sec <= (int)duration; sec += every) {
    const int pos = (sec / every) * tokens_per_marker;
    const int segment_len = pos - consumed;
    if (segment_len > 0) {
      out.insert(out.end(), segment_len, hparams_.audio_pad_id);
      consumed += segment_len;
    }
    for (char c : std::to_string(sec)) {
      out.push_back(digit_token_ids_[c - '0']);
    }
  }
  const int remainder = n_audio_tokens - consumed;
  if (remainder > 0) {
    out.insert(out.end(), remainder, hparams_.audio_pad_id);
  }
  return out;
}

// ============================================================================
// LLM decode — mirror of Qwen3ASR::DecodeWithLLM, with the audio embeddings
// scattered into the <|audio_pad|> positions of the ChatML prompt
// (masked_scatter equivalent) and pure greedy decoding (HF do_sample=False).
// ============================================================================

bool MossTranscribeModel::DecodeWithLLM(RSState &state,
                                        ggml_backend_sched_t sched) {
  auto &st = static_cast<MossTranscribeState &>(state);
  if (st.audio_embeds.empty() || !llm_model_) return false;

  const int audio_T = st.T_audio;
  const int n_embd  = st.n_embd;
  if (n_embd != (int)llm_model_->hparams().n_embd) {
    RS_LOG_ERR("MossTranscribe: encoder output dim (%d) != LLM n_embd (%d)",
               n_embd, (int)llm_model_->hparams().n_embd);
    return false;
  }

  // ---- Token stream: ChatML prompt with the audio span in the middle ----
  const auto &vocab = llm_model_->vocab();
  const std::string prefix_text =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n<|audio_start|>";
  const std::string suffix_text =
      std::string("<|audio_end|>\n") + user_input_prompt_ +
      "<|im_end|>\n<|im_start|>assistant\n";

  std::vector<int32_t> input_ids = vocab.tokenize(prefix_text, false);
  const std::vector<int32_t> span_ids = BuildAudioSpanIds(audio_T);
  input_ids.insert(input_ids.end(), span_ids.begin(), span_ids.end());
  const std::vector<int32_t> suffix_ids = vocab.tokenize(suffix_text, false);
  input_ids.insert(input_ids.end(), suffix_ids.begin(), suffix_ids.end());

  const int total_T = (int)input_ids.size();
  RS_LOG_INFO("MossTranscribe DecodeWithLLM: prompt=%d tokens (audio=%d)",
              total_T, audio_T);
  if (std::getenv("RS_MOSS_DEBUG")) {
    std::string head, tail;
    for (int i = 0; i < std::min(20, total_T); ++i)
      head += std::to_string(input_ids[i]) + " ";
    for (int i = std::max(0, total_T - 25); i < total_T; ++i)
      tail += std::to_string(input_ids[i]) + " ";
    RS_LOG_INFO("MossTranscribe dbg prompt head: %s", head.c_str());
    RS_LOG_INFO("MossTranscribe dbg prompt tail: %s", tail.c_str());
  }

  // Adaptive decode budget: transcripts run ~<=2x the audio token count.
  max_decode_tokens_ = std::min(8192, std::max(1024, audio_T * 2 + 512));

  llm_cparams cparams;
  cparams.n_ctx           = total_T + max_decode_tokens_;
  cparams.n_batch         = total_T;
  cparams.n_ubatch        = total_T;
  cparams.n_threads       = 4;
  cparams.n_threads_batch = 4;
  llm_graph_builder_ =
      std::make_unique<llm_build_qwen3>(*llm_model_, cparams, sched);

  host_kv_cache_k_.clear();
  host_kv_cache_v_.clear();
  n_cached_tokens_ = 0;

  // -------- (a) embedding lookup graph + audio scatter --------
  struct ggml_init_params proj_params = {
      1024 * ggml_tensor_overhead() + (1u << 16), nullptr, true};
  struct ggml_context *ctx_proj = ggml_init(proj_params);
  struct ggml_cgraph *gf_proj   = ggml_new_graph(ctx_proj);

  ggml_tensor *inp_ids = ggml_new_tensor_1d(ctx_proj, GGML_TYPE_I32, total_T);
  ggml_set_input(inp_ids);
  ggml_set_name(inp_ids, "prompt_tokens");
  ggml_tensor *tok_embd =
      ggml_get_rows(ctx_proj, llm_model_->tok_embd(), inp_ids);
  ggml_set_output(tok_embd);
  ggml_build_forward_expand(gf_proj, tok_embd);

  if (!ggml_backend_sched_alloc_graph(sched, gf_proj)) {
    RS_LOG_ERR("MossTranscribe: failed to allocate embedding graph");
    ggml_free(ctx_proj);
    return false;
  }
  ggml_backend_tensor_set(inp_ids, input_ids.data(), 0,
                          total_T * sizeof(int32_t));
  if (ggml_backend_sched_graph_compute(sched, gf_proj) !=
      GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("MossTranscribe: embedding graph compute failed");
    ggml_free(ctx_proj);
    return false;
  }

  std::vector<float> llm_in_host((size_t)n_embd * total_T);
  ggml_backend_tensor_get(tok_embd, llm_in_host.data(), 0,
                          llm_in_host.size() * sizeof(float));
  ggml_free(ctx_proj);
  ggml_backend_sched_reset(sched);

  // masked_scatter: audio embedding columns into the audio_pad positions.
  {
    int scattered = 0;
    for (int i = 0; i < total_T && scattered < audio_T; ++i) {
      if (input_ids[i] == hparams_.audio_pad_id) {
        std::memcpy(llm_in_host.data() + (size_t)i * n_embd,
                    st.audio_embeds.data() + (size_t)scattered * n_embd,
                    (size_t)n_embd * sizeof(float));
        scattered++;
      }
    }
    if (scattered != audio_T) {
      RS_LOG_ERR("MossTranscribe: audio token mismatch (pads=%d embeds=%d)",
                 scattered, audio_T);
      return false;
    }
  }

  // -------- (b) prefill --------
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) positions[i] = i;

  struct ggml_init_params llm_params = {
      512 * ggml_tensor_overhead() + (1u << 16), nullptr, true};
  struct ggml_context *ctx_llm = ggml_init(llm_params);
  ggml_tensor *llm_input =
      ggml_new_tensor_2d(ctx_llm, GGML_TYPE_F32, n_embd, total_T);
  ggml_set_input(llm_input);
  ggml_set_name(llm_input, "llm_input");

  llm_build_opts opts;
  opts.output_mode     = llm_output_mode::OUTPUT_LOGITS;
  opts.skip_embeddings = true;
  opts.use_kv_cache    = true;
  opts.causal_mask     = true;

  auto result = llm_graph_builder_->build_graph_from_embeds(
      llm_input, total_T, nullptr, positions.data(), &opts);
  if (!result) {
    RS_LOG_ERR("MossTranscribe: failed to build LLM prefill graph");
    ggml_free(ctx_llm);
    return false;
  }
  if (!ggml_backend_sched_alloc_graph(sched, result->get_graph())) {
    RS_LOG_ERR("MossTranscribe: failed to alloc LLM prefill graph");
    ggml_free(ctx_llm);
    return false;
  }
  ggml_backend_tensor_set(llm_input, llm_in_host.data(), 0,
                          llm_in_host.size() * sizeof(float));
  if (auto t = result->get_input_tensor("position_ids")) {
    result->set_position_ids(t, positions.data(), total_T);
  } else if (auto t2 = result->get_input_tensor("position_ids_seq")) {
    result->set_position_ids(t2, positions.data(), total_T);
  }
  if (auto t = result->get_input_tensor("causal_mask")) {
    result->set_causal_mask(t, total_T, 0);
  }
  if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
      GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("MossTranscribe: LLM prefill compute failed");
    ggml_free(ctx_llm);
    return false;
  }

  ggml_tensor *logits = result->get_logits();
  if (!logits) {
    RS_LOG_ERR("MossTranscribe: no logits from prefill");
    ggml_free(ctx_llm);
    return false;
  }
  const int n_vocab = (int)logits->ne[0];
  std::vector<float> logits_host(n_vocab);
  ggml_backend_tensor_get(logits, logits_host.data(), 0,
                          (size_t)n_vocab * sizeof(float));
  int32_t best_token = 0;
  {
    float best = logits_host[0];
    for (int v = 1; v < n_vocab; ++v) {
      if (logits_host[v] > best) { best = logits_host[v]; best_token = v; }
    }
  }
  st.token_ids.clear();
  st.token_ids.push_back(best_token);

  // Snapshot prefill KV cache.
  const auto &lp = llm_model_->hparams();
  const int n_layer   = lp.n_layer;
  const int n_head_kv = lp.n_head_kv > 0 ? lp.n_head_kv : lp.n_head;
  const int head_dim  = lp.head_dim;
  const int kv_dim    = head_dim * n_head_kv;
  host_kv_cache_k_.assign(n_layer, {});
  host_kv_cache_v_.assign(n_layer, {});
  n_cached_tokens_ = total_T;
  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor *k_out = result->get_kv_output_k(il);
    ggml_tensor *v_out = result->get_kv_output_v(il);
    if (!k_out || !v_out) {
      RS_LOG_ERR("MossTranscribe: prefill KV missing for layer %d", il);
      continue;
    }
    size_t kv_bytes = ggml_nbytes(k_out);
    host_kv_cache_k_[il].resize(kv_bytes / sizeof(float));
    host_kv_cache_v_[il].resize(kv_bytes / sizeof(float));
    ggml_backend_tensor_get(k_out, host_kv_cache_k_[il].data(), 0, kv_bytes);
    ggml_backend_tensor_get(v_out, host_kv_cache_v_[il].data(), 0, kv_bytes);
  }

  ggml_free(ctx_llm);
  ggml_backend_sched_reset(sched);

  // -------- (c) GPU-persistent KV buffers & gallocr pre-warm --------
  const int32_t n_kv_max = n_cached_tokens_ + max_decode_tokens_;
  std::vector<ggml_tensor *> gpu_kv_k_vec(n_layer, nullptr);
  std::vector<ggml_tensor *> gpu_kv_v_vec(n_layer, nullptr);
  ggml_backend_buffer_t kv_gpu_buf = nullptr;
  struct ggml_context *ctx_kv = nullptr;
  {
    struct ggml_init_params kv_buf_params = {
        (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16),
        nullptr, true};
    ctx_kv = ggml_init(kv_buf_params);
    for (int il = 0; il < n_layer; ++il) {
      gpu_kv_k_vec[il] =
          ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
      gpu_kv_v_vec[il] =
          ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
      ggml_set_name(gpu_kv_k_vec[il],
                    ("mosstd_kv_k_" + std::to_string(il)).c_str());
      ggml_set_name(gpu_kv_v_vec[il],
                    ("mosstd_kv_v_" + std::to_string(il)).c_str());
    }
    int n_backends = ggml_backend_sched_get_n_backends(sched);
    ggml_backend_t gpu_backend =
        ggml_backend_sched_get_backend(sched, n_backends - 1);
    kv_gpu_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, gpu_backend);
    if (!kv_gpu_buf) {
      RS_LOG_ERR("MossTranscribe: failed to allocate GPU KV buffers");
    }
    for (int il = 0; il < n_layer; ++il) {
      size_t bytes = (size_t)kv_dim * n_cached_tokens_ * sizeof(float);
      ggml_backend_tensor_set(gpu_kv_k_vec[il],
                              host_kv_cache_k_[il].data(), 0, bytes);
      ggml_backend_tensor_set(gpu_kv_v_vec[il],
                              host_kv_cache_v_[il].data(), 0, bytes);
    }
  }
  {
    llm_build_opts warmup_opts;
    warmup_opts.output_mode     = llm_output_mode::OUTPUT_LOGITS;
    warmup_opts.skip_embeddings = false;
    warmup_opts.use_kv_cache    = true;
    warmup_opts.is_decode_step  = true;
    warmup_opts.n_kv_cache      = n_kv_max - 1;
    warmup_opts.n_kv_max        = n_kv_max;
    warmup_opts.causal_mask     = true;
    warmup_opts.gpu_kv_k        = gpu_kv_k_vec.data();
    warmup_opts.gpu_kv_v        = gpu_kv_v_vec.data();
    llm_pos warmup_pos = (llm_pos)(n_kv_max - 1);
    int32_t dummy = 0;
    auto warmup = llm_graph_builder_->build_graph(
        &dummy, 1, nullptr, &warmup_pos, &warmup_opts);
    if (warmup) ggml_backend_sched_reserve(sched, warmup->get_graph());
  }

  // -------- (d) autoregressive greedy decode --------
  const int32_t eos_token_id = llm_model_->vocab().token_eos();
  std::vector<float> kv_stage_k(kv_dim), kv_stage_v(kv_dim);
  int consecutive_same_token = 0;
  constexpr int MAX_CONSECUTIVE_REPEAT = 30;

  for (int step = 0; step < max_decode_tokens_; ++step) {
    llm_pos decode_pos = (llm_pos)n_cached_tokens_;
    llm_build_opts dec_opts;
    dec_opts.output_mode     = llm_output_mode::OUTPUT_LOGITS;
    dec_opts.skip_embeddings = false;
    dec_opts.use_kv_cache    = true;
    dec_opts.is_decode_step  = true;
    dec_opts.n_kv_cache      = n_cached_tokens_;
    dec_opts.n_kv_max        = n_kv_max;
    dec_opts.causal_mask     = true;
    dec_opts.gpu_kv_k        = gpu_kv_k_vec.data();
    dec_opts.gpu_kv_v        = gpu_kv_v_vec.data();

    auto dec = llm_graph_builder_->build_graph(&best_token, 1, nullptr,
                                               &decode_pos, &dec_opts);
    if (!dec) {
      RS_LOG_ERR("MossTranscribe: decode build failed at step %d", step);
      break;
    }
    if (!ggml_backend_sched_alloc_graph(sched, dec->get_graph())) {
      RS_LOG_ERR("MossTranscribe: decode alloc failed at step %d", step);
      break;
    }
    if (auto t = dec->get_input_tensor("inp_tokens")) {
      ggml_backend_tensor_set(t, &best_token, 0, sizeof(int32_t));
    }
    if (auto t = dec->get_input_tensor("position_ids")) {
      dec->set_position_ids(t, &decode_pos, 1);
    } else if (auto t2 = dec->get_input_tensor("position_ids_seq")) {
      dec->set_position_ids(t2, &decode_pos, 1);
    }
    if (auto t = dec->get_input_tensor("causal_mask")) {
      dec->set_causal_mask(t, 1, n_cached_tokens_);
    }
    if (ggml_backend_sched_graph_compute(sched, dec->get_graph()) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("MossTranscribe: decode compute failed at step %d", step);
      break;
    }
    ggml_tensor *dec_logits = dec->get_logits();
    if (!dec_logits) {
      RS_LOG_ERR("MossTranscribe: no logits at step %d", step);
      break;
    }
    std::vector<float> dec_logits_host(n_vocab);
    ggml_backend_tensor_get(dec_logits, dec_logits_host.data(), 0,
                            (size_t)n_vocab * sizeof(float));

    int32_t next_token = 0;
    {
      float best = dec_logits_host[0];
      for (int v = 1; v < n_vocab; ++v) {
        if (dec_logits_host[v] > best) {
          best = dec_logits_host[v];
          next_token = v;
        }
      }
    }

    if (next_token == best_token) consecutive_same_token++;
    else                          consecutive_same_token = 0;

    // Stage new KV column into the persistent GPU buffers.
    for (int il = 0; il < n_layer; ++il) {
      ggml_tensor *k_out = dec->get_kv_output_k(il);
      ggml_tensor *v_out = dec->get_kv_output_v(il);
      if (!k_out || !v_out) continue;
      const size_t new_bytes = (size_t)kv_dim * sizeof(float);
      const size_t off = (size_t)n_cached_tokens_ * kv_dim * sizeof(float);
      ggml_backend_tensor_get(k_out, kv_stage_k.data(), off, new_bytes);
      ggml_backend_tensor_get(v_out, kv_stage_v.data(), off, new_bytes);
      ggml_backend_tensor_set(gpu_kv_k_vec[il], kv_stage_k.data(), off,
                              new_bytes);
      ggml_backend_tensor_set(gpu_kv_v_vec[il], kv_stage_v.data(), off,
                              new_bytes);
    }
    n_cached_tokens_++;

    if (next_token == eos_token_id) {
      ggml_backend_sched_reset(sched);
      break;
    }
    if (consecutive_same_token >= MAX_CONSECUTIVE_REPEAT) {
      RS_LOG_WARN("MossTranscribe: token %d repeated %d times — stopping",
                  next_token, consecutive_same_token);
      ggml_backend_sched_reset(sched);
      break;
    }
    st.token_ids.push_back(next_token);
    best_token = next_token;
    ggml_backend_sched_reset(sched);
  }

  if (kv_gpu_buf) ggml_backend_buffer_free(kv_gpu_buf);
  if (ctx_kv) ggml_free(ctx_kv);

  std::string text = llm_model_->vocab().detokenize(st.token_ids);
  st.tokens.clear();
  st.tokens.push_back(text);
  RS_LOG_INFO("MossTranscribe: %s", text.c_str());
  return true;
}

bool MossTranscribeModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  return DecodeWithLLM(state, sched);
}

std::string MossTranscribeModel::GetTranscription(RSState &state) {
  auto &st = static_cast<MossTranscribeState &>(state);
  std::string result;
  result.reserve(64);
  for (const auto &s : st.tokens) result += s;
  st.tokens.clear();
  return result;
}

// ============================================================================
// Registration
// ============================================================================

extern void
rs_register_model_arch(const std::string &arch,
                       std::function<std::shared_ptr<ISpeechModel>()> creator);

namespace {
struct MossTranscribeRegistrar {
  MossTranscribeRegistrar() {
    rs_register_model_arch("MossTranscribeDiarize", []() {
      return std::make_shared<MossTranscribeModel>();
    });
  }
} g_moss_transcribe_registrar;
} // namespace
