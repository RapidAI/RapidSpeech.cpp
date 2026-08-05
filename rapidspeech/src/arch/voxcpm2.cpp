#include "voxcpm2.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>

// =====================================================================
// VoxCPM2Model Implementation
// =====================================================================

VoxCPM2Model::VoxCPM2Model()
{
    meta_.arch_name = "voxcpm2";
    meta_.audio_sample_rate = 48000;  // Output is 48kHz
    meta_.n_mels = 0;  // Not used in this architecture
    meta_.use_external_frontend = true;
}

VoxCPM2Model::~VoxCPM2Model()
{
    base_lm_.reset();
    if (dit_fused_buffer_)
    {
        ggml_backend_buffer_free(dit_fused_buffer_);
        dit_fused_buffer_ = nullptr;
    }
    if (dit_fused_ctx_)
    {
        ggml_free(dit_fused_ctx_);
        dit_fused_ctx_ = nullptr;
    }
    if (ctx_data_)
    {
        // Note: don't free ctx_data_, it's owned by rs_context_t
        ctx_data_ = nullptr;
    }
}

bool VoxCPM2Model::Load(const std::unique_ptr<rs_context_t> &ctx,
                        ggml_backend_t backend)
{
    if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data)
    {
        RS_LOG_ERR("VoxCPM2: Invalid context");
        return false;
    }

    backend_ = backend;
    ctx_data_ = ctx->gguf_data;

    // Load hyperparameters
    if (!load_hparams_from_gguf(ctx->ctx_gguf))
    {
        RS_LOG_ERR("VoxCPM2: Failed to load hyperparameters");
        return false;
    }

    // Build tensor map
    const int n_tensors = gguf_get_n_tensors(ctx->ctx_gguf);
    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx->ctx_gguf, i);
        ggml_tensor *tensor = ggml_get_tensor(ctx_data_, name);
        if (tensor)
        {
            tensors_[name] = tensor;
        }
    }

    RS_LOG_INFO("VoxCPM2: Loaded %zu tensors", tensors_.size());

    has_tokenizer_ = load_tokenizer_from_gguf(ctx->ctx_gguf);
    if (!has_tokenizer_)
    {
        RS_LOG_WARN("VoxCPM2: tokenizer data not found in GGUF");
    }

    // Load Base LM (reuse llm_model for Qwen2-like architecture)
    has_base_lm_ = LoadBaseLM(ctx->ctx_gguf);
    if (!has_base_lm_)
    {
        RS_LOG_ERR("VoxCPM2: Failed to load Base LM");
        return false;
    }

    // Load other components
    has_text_encoder_ = LoadTextEncoder(ctx->ctx_gguf);
    if (!has_text_encoder_)
    {
        RS_LOG_WARN("VoxCPM2: Text encoder not loaded (will use placeholder)");
    }

    has_fsq_ = LoadScalarQuantization(ctx->ctx_gguf);
    if (!has_fsq_)
    {
        RS_LOG_WARN("VoxCPM2: Scalar quantization not loaded");
    }

    has_dit_ = LoadDiT(ctx->ctx_gguf);
    if (!has_dit_)
    {
        RS_LOG_WARN("VoxCPM2: DiT not loaded");
    }

    has_audio_vae_ = LoadAudioVAE(ctx->ctx_gguf);
    if (!has_audio_vae_)
    {
        RS_LOG_WARN("VoxCPM2: Audio VAE not loaded");
    }
    else
    {
        auto read_f32 = [](struct ggml_tensor *tensor) {
            const int64_t count = ggml_nelements(tensor);
            std::vector<float> values(count);
            if (tensor->type == GGML_TYPE_F32)
            {
                ggml_backend_tensor_get(tensor, values.data(), 0,
                                       count * sizeof(float));
            }
            else if (tensor->type == GGML_TYPE_F16)
            {
                std::vector<ggml_fp16_t> packed(count);
                ggml_backend_tensor_get(tensor, packed.data(), 0,
                                       count * sizeof(ggml_fp16_t));
                ggml_fp16_to_fp32_row(packed.data(), values.data(), count);
            }
            else if (tensor->type == GGML_TYPE_BF16)
            {
                std::vector<ggml_bf16_t> packed(count);
                ggml_backend_tensor_get(tensor, packed.data(), 0,
                                       count * sizeof(ggml_bf16_t));
                for (int64_t i = 0; i < count; ++i)
                {
                    values[i] = ggml_bf16_to_fp32(packed[i]);
                }
            }
            else
            {
                std::vector<uint8_t> packed(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor, packed.data(), 0,
                                       packed.size());
                const ggml_type_traits *traits =
                    ggml_get_type_traits(tensor->type);
                if (traits && traits->to_float)
                {
                    traits->to_float(packed.data(), values.data(), count);
                }
                else
                {
                    values.clear();
                }
            }
            return values;
        };

        static const std::string vae_prefix = "audio_vae.decoder.";
        static const std::string weight_v_suffix = ".weight_v";
        for (const auto &entry : tensors_)
        {
            const std::string &name = entry.first;
            if (name.compare(0, vae_prefix.size(), vae_prefix) != 0 ||
                name.size() <= weight_v_suffix.size() ||
                name.compare(name.size() - weight_v_suffix.size(),
                             weight_v_suffix.size(), weight_v_suffix) != 0)
            {
                continue;
            }
            const std::string prefix =
                name.substr(0, name.size() - weight_v_suffix.size());
            auto g_it = tensors_.find(prefix + ".weight_g");
            if (g_it == tensors_.end())
            {
                continue;
            }
            const int64_t channels = ggml_nelements(g_it->second);
            const int64_t value_count = ggml_nelements(entry.second);
            if (channels <= 0 || value_count % channels != 0)
            {
                continue;
            }
            const int64_t rows = value_count / channels;
            const std::vector<float> g = read_f32(g_it->second);
            const std::vector<float> v = read_f32(entry.second);
            if (g.size() != static_cast<size_t>(channels) ||
                v.size() != static_cast<size_t>(value_count))
            {
                continue;
            }
            std::vector<float> scales(channels);
            const bool packed_deconv =
                ggml_is_quantized(entry.second->type) &&
                entry.second->ne[0] == channels;
            for (int64_t channel = 0; channel < channels; ++channel)
            {
                double sum_sq = 0.0;
                for (int64_t row = 0; row < rows; ++row)
                {
                    const size_t offset = packed_deconv
                        ? static_cast<size_t>(row * channels + channel)
                        : static_cast<size_t>(channel * rows + row);
                    const float value = v[offset];
                    sum_sq += static_cast<double>(value) * value;
                }
                scales[channel] = g[channel] /
                    std::sqrt(static_cast<float>(sum_sq) + 1e-12f);
            }
            audio_vae_weight_scales_.emplace(prefix, std::move(scales));
        }
        RS_LOG_INFO("VoxCPM2: cached %zu AudioVAE weight-norm scales",
                    audio_vae_weight_scales_.size());
    }

    RS_LOG_INFO("VoxCPM2: Model loaded successfully");
    return true;
}

bool VoxCPM2Model::load_hparams_from_gguf(struct gguf_context *ctx_gguf)
{
    int kv_idx;

    // Base LM parameters
    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.lm.num_hidden_layers");
    if (kv_idx >= 0) hparams_.lm_num_hidden_layers = gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.lm.hidden_size");
    if (kv_idx >= 0) hparams_.lm_hidden_size = gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.lm.vocab_size");
    if (kv_idx >= 0) hparams_.lm_vocab_size = gguf_get_val_u32(ctx_gguf, kv_idx);

    // Audio VAE parameters
    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.audio_vae.out_sample_rate");
    if (kv_idx >= 0)
    {
        hparams_.vae_out_sample_rate = gguf_get_val_u32(ctx_gguf, kv_idx);
        meta_.audio_sample_rate = hparams_.vae_out_sample_rate;
    }

    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.audio_start_token");
    if (kv_idx >= 0) token_audio_start_ = gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.audio_end_token");
    if (kv_idx >= 0) token_audio_end_ = gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.ref_audio_start_token");
    if (kv_idx >= 0) token_ref_audio_start_ = gguf_get_val_u32(ctx_gguf, kv_idx);

    kv_idx = gguf_find_key(ctx_gguf, "voxcpm2.ref_audio_end_token");
    if (kv_idx >= 0) token_ref_audio_end_ = gguf_get_val_u32(ctx_gguf, kv_idx);

    RS_LOG_INFO("VoxCPM2 HParams: lm_layers=%d, hidden=%d, vocab=%d, sr=%d",
                hparams_.lm_num_hidden_layers, hparams_.lm_hidden_size,
                hparams_.lm_vocab_size, hparams_.vae_out_sample_rate);

    return true;
}

ggml_tensor *VoxCPM2Model::get_tensor(const std::string &name)
{
    auto it = tensors_.find(name);
    if (it == tensors_.end())
    {
        return nullptr;
    }
    return it->second;
}

bool VoxCPM2Model::load_tokenizer_from_gguf(struct gguf_context *ctx_gguf)
{
    id_to_token_.clear();
    token_to_id_.clear();
    merge_rank_.clear();

    const int tok_key = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
    if (tok_key < 0 || gguf_get_kv_type(ctx_gguf, tok_key) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx_gguf, tok_key) != GGUF_TYPE_STRING)
    {
        return false;
    }

    const int n_tokens = static_cast<int>(gguf_get_arr_n(ctx_gguf, tok_key));
    id_to_token_.resize(n_tokens);
    for (int i = 0; i < n_tokens; ++i)
    {
        id_to_token_[i] = gguf_get_arr_str(ctx_gguf, tok_key, i);
        token_to_id_[id_to_token_[i]] = i;
    }

    const int merge_key = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
    if (merge_key >= 0 && gguf_get_kv_type(ctx_gguf, merge_key) == GGUF_TYPE_ARRAY &&
        gguf_get_arr_type(ctx_gguf, merge_key) == GGUF_TYPE_STRING)
    {
        const int n_merges = static_cast<int>(gguf_get_arr_n(ctx_gguf, merge_key));
        for (int i = 0; i < n_merges; ++i)
        {
            merge_rank_[gguf_get_arr_str(ctx_gguf, merge_key, i)] = i;
        }
    }

    auto find_id = [&](const char *tok, int32_t fallback) {
        auto it = token_to_id_.find(tok);
        return it == token_to_id_.end() ? fallback : it->second;
    };
    token_unk_ = find_id("<unk>", 0);
    token_bos_ = find_id("<s>", 1);
    token_eos_ = find_id("</s>", 2);

    RS_LOG_INFO("VoxCPM2: tokenizer loaded (%d tokens, %zu merges)",
                n_tokens, merge_rank_.size());
    return n_tokens > 0;
}

bool VoxCPM2Model::LoadTextEncoder(struct gguf_context *ctx_gguf)
{
    // Check for text encoder tensors
    // Pattern: feat_encoder.encoder.layers.K.{...}
    if (!get_tensor("feat_encoder.encoder.layers.0.input_layernorm.weight"))
    {
        RS_LOG_WARN("VoxCPM2: Text encoder tensors not found");
        return false;
    }

    RS_LOG_INFO("VoxCPM2: Text encoder ready");
    return true;
}

bool VoxCPM2Model::LoadBaseLM(struct gguf_context *ctx_gguf)
{
    // Check for Base LM tensors (base_lm.*)
    if (!get_tensor("base_lm.embed_tokens.weight"))
    {
        RS_LOG_WARN("VoxCPM2: Base LM tensors not found");
        return false;
    }

    RS_LOG_INFO("VoxCPM2: Base LM tensors ready");
    return true;
}

bool VoxCPM2Model::LoadScalarQuantization(struct gguf_context *ctx_gguf)
{
    // Pattern in the current VoxCPM2 GGUF: fsq_layer.{in,out}_proj.*
    if (!get_tensor("fsq_layer.in_proj.weight") ||
        !get_tensor("fsq_layer.out_proj.weight"))
    {
        RS_LOG_WARN("VoxCPM2: FSQ tensors not found");
        return false;
    }

    RS_LOG_INFO("VoxCPM2: FSQ ready");
    return true;
}

bool VoxCPM2Model::LoadDiT(struct gguf_context *ctx_gguf)
{
    // Pattern in the current VoxCPM2 GGUF: feat_decoder.estimator.*
    if (!get_tensor("feat_decoder.estimator.in_proj.weight") ||
        !get_tensor("feat_decoder.estimator.decoder.layers.0.self_attn.q_proj.weight") ||
        !get_tensor("feat_decoder.estimator.out_proj.weight"))
    {
        RS_LOG_WARN("VoxCPM2: DiT tensors not found");
        return false;
    }

    size_t metadata_size = 0;
    for (int layer = 0; layer < hparams_.dit_num_layers; ++layer)
    {
        metadata_size += ggml_tensor_overhead();
    }
    metadata_size += 64 * 1024;
    struct ggml_init_params params = {
        .mem_size   = metadata_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    dit_fused_ctx_ = ggml_init(params);
    if (dit_fused_ctx_)
    {
        dit_qkv_weights_.reserve(hparams_.dit_num_layers);
        for (int layer = 0; layer < hparams_.dit_num_layers; ++layer)
        {
            const std::string prefix =
                "feat_decoder.estimator.decoder.layers." +
                std::to_string(layer) + ".self_attn.";
            struct ggml_tensor *q = get_tensor(prefix + "q_proj.weight");
            struct ggml_tensor *k = get_tensor(prefix + "k_proj.weight");
            struct ggml_tensor *v = get_tensor(prefix + "v_proj.weight");
            if (!q || !k || !v || q->type != k->type || q->type != v->type ||
                q->ne[0] != k->ne[0] || q->ne[0] != v->ne[0])
            {
                dit_qkv_weights_.clear();
                break;
            }
            struct ggml_tensor *qkv = ggml_new_tensor_2d(
                dit_fused_ctx_, q->type, q->ne[0],
                q->ne[1] + k->ne[1] + v->ne[1]);
            ggml_set_name(qkv, (prefix + "qkv_proj.weight").c_str());
            dit_qkv_weights_.push_back(qkv);
        }
        if (dit_qkv_weights_.size() ==
            static_cast<size_t>(hparams_.dit_num_layers))
        {
            dit_fused_buffer_ =
                ggml_backend_alloc_ctx_tensors(dit_fused_ctx_, backend_);
        }
        if (dit_fused_buffer_)
        {
            for (int layer = 0; layer < hparams_.dit_num_layers; ++layer)
            {
                const std::string prefix =
                    "feat_decoder.estimator.decoder.layers." +
                    std::to_string(layer) + ".self_attn.";
                struct ggml_tensor *q = get_tensor(prefix + "q_proj.weight");
                struct ggml_tensor *k = get_tensor(prefix + "k_proj.weight");
                struct ggml_tensor *v = get_tensor(prefix + "v_proj.weight");
                struct ggml_tensor *qkv = dit_qkv_weights_[layer];
                const size_t q_bytes = ggml_nbytes(q);
                const size_t k_bytes = ggml_nbytes(k);
                std::vector<uint8_t> packed(
                    q_bytes + k_bytes + ggml_nbytes(v));
                ggml_backend_tensor_get(q, packed.data(), 0, q_bytes);
                ggml_backend_tensor_get(
                    k, packed.data() + q_bytes, 0, k_bytes);
                ggml_backend_tensor_get(
                    v, packed.data() + q_bytes + k_bytes, 0, ggml_nbytes(v));
                ggml_backend_tensor_set(qkv, packed.data(), 0, packed.size());
            }
            RS_LOG_INFO("VoxCPM2: fused %zu DiT QKV projections",
                        dit_qkv_weights_.size());
        }
        else
        {
            dit_qkv_weights_.clear();
        }
    }

    RS_LOG_INFO("VoxCPM2: DiT ready");
    return true;
}

bool VoxCPM2Model::LoadAudioVAE(struct gguf_context *ctx_gguf)
{
    // Full synthesis requires the AudioVAE decoder weights. Some incomplete
    // GGUFs carry only AudioVAE metadata, but no decoder tensors.
    if (!get_tensor("audio_vae.decoder.model.0.weight_v") &&
        !get_tensor("vae.dec.layer.0.weight_v") &&
        !get_tensor("audio_vae.decoder.conv_in.weight"))
    {
        RS_LOG_WARN("VoxCPM2: Audio VAE decoder tensors not found");
        return false;
    }

    RS_LOG_INFO("VoxCPM2: Audio VAE ready");
    return true;
}

std::shared_ptr<RSState> VoxCPM2Model::CreateState()
{
    auto state = std::make_shared<VoxCPM2State>();
    state->generation_seed = seed_;
    state->diffusion_steps = diffusion_steps_;
    return state;
}

bool VoxCPM2Model::Encode(const std::vector<float> &input_frames, RSState &state,
                          ggml_backend_sched_t sched)
{
    // For TTS, encoding happens in PushText
    return true;
}

bool VoxCPM2Model::Decode(RSState &state, ggml_backend_sched_t sched)
{
    return DecodeImpl(state, sched, {});
}

bool VoxCPM2Model::DecodeStream(RSState &state, ggml_backend_sched_t sched,
                                AudioChunkCallback emit)
{
    if (!emit)
    {
        RS_LOG_ERR("VoxCPM2: DecodeStream needs a non-null emit callback");
        return false;
    }
    return DecodeImpl(state, sched, emit);
}

bool VoxCPM2Model::DecodeImpl(RSState &state, ggml_backend_sched_t sched,
                              const AudioChunkCallback &emit)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return false;
    }

    RS_LOG_INFO("VoxCPM2: Starting synthesis pipeline");

    if (!has_tokenizer_)
    {
        RS_LOG_ERR("VoxCPM2: this GGUF does not contain tokenizer data; "
                   "normal TTS requires tokenizer.ggml.tokens/merges or tokenizer.vocab_tensor");
        return false;
    }

    if (!has_audio_vae_)
    {
        RS_LOG_ERR("VoxCPM2: this GGUF does not contain AudioVAE decoder tensors; "
                   "normal TTS cannot decode acoustic latents to waveform");
        return false;
    }

    if (!has_text_encoder_ || !has_base_lm_ || !has_fsq_ || !has_dit_)
    {
        RS_LOG_ERR("VoxCPM2: incomplete model tensors "
                   "(text_encoder=%d base_lm=%d fsq=%d dit=%d audio_vae=%d)",
                   (int)has_text_encoder_, (int)has_base_lm_, (int)has_fsq_,
                   (int)has_dit_, (int)has_audio_vae_);
        return false;
    }

    // Stage 1: Text Encoder
    if (!s->text_encoded)
    {
        RS_LOG_INFO("VoxCPM2: [1/5] Running text encoder");
        const auto stage_begin = std::chrono::steady_clock::now();
        if (!RunTextEncoder(*s, sched))
        {
            RS_LOG_ERR("VoxCPM2: Text encoder failed");
            return false;
        }
        RS_LOG_INFO("VoxCPM2: text encoder took %.1f ms",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - stage_begin).count());
        s->text_encoded = true;
    }

    // Stage 2: Base LM (autoregressive generation)
    if (!s->tokens_generated)
    {
        RS_LOG_INFO("VoxCPM2: [2/5] Running Base LM (autoregressive)");
        const auto stage_begin = std::chrono::steady_clock::now();
        if (!RunBaseLM(*s, sched))
        {
            RS_LOG_ERR("VoxCPM2: Base LM failed");
            return false;
        }
        RS_LOG_INFO("VoxCPM2: Base LM prefill took %.1f ms",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - stage_begin).count());
        s->tokens_generated = true;
    }

    // Stage 3: Scalar Quantization
    if (!s->latents_quantized)
    {
        RS_LOG_INFO("VoxCPM2: [3/5] Running scalar quantization");
        const auto stage_begin = std::chrono::steady_clock::now();
        if (!RunScalarQuantization(*s, sched))
        {
            RS_LOG_ERR("VoxCPM2: Scalar quantization failed");
            return false;
        }
        RS_LOG_INFO("VoxCPM2: FSQ/RALM prefill took %.1f ms",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - stage_begin).count());
        s->latents_quantized = true;
    }

    // Stage 4: DiT (diffusion)
    if (!s->latents_denoised)
    {
        RS_LOG_INFO("VoxCPM2: [4/5] Running DiT (diffusion)");
        const auto stage_begin = std::chrono::steady_clock::now();
        if (!RunDiT(*s, sched, emit))
        {
            RS_LOG_ERR("VoxCPM2: DiT failed");
            return false;
        }
        RS_LOG_INFO("VoxCPM2: DiT/AR stage took %.1f ms",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - stage_begin).count());
        s->latents_denoised = true;
    }

    // Stage 5: Audio VAE
    if (!s->audio_generated)
    {
        RS_LOG_INFO("VoxCPM2: [5/5] Running Audio VAE");
        const auto stage_begin = std::chrono::steady_clock::now();
        if (!RunAudioVAE(*s, sched))
        {
            RS_LOG_ERR("VoxCPM2: Audio VAE failed");
            return false;
        }
        RS_LOG_INFO("VoxCPM2: AudioVAE took %.1f ms",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - stage_begin).count());
        s->audio_generated = true;
    }

    RS_LOG_INFO("VoxCPM2: Synthesis completed");
    return true;
}

std::string VoxCPM2Model::GetTranscription(RSState &state)
{
    // Not applicable for TTS model
    return "";
}

bool VoxCPM2Model::PushText(RSState &state, const char *text,
                            const char *language,
                            const char *instruct)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return false;
    }

    if (!text || !*text)
    {
        RS_LOG_ERR("VoxCPM2: Empty text input");
        return false;
    }

    s->input_text = text;
    s->token_ids = tokenize(text);
    s->generation_seed = seed_;
    s->diffusion_steps = diffusion_steps_;

    RS_LOG_INFO("VoxCPM2: Pushed text '%s' (%zu tokens)", text, s->token_ids.size());

    // Reset pipeline flags
    s->text_encoded = false;
    s->tokens_generated = false;
    s->latents_quantized = false;
    s->latents_denoised = false;
    s->audio_generated = false;

    return true;
}

int VoxCPM2Model::GetAudioOutput(RSState &state, float **out_data)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return -1;
    }

    if (!s->audio_generated || s->audio_output.empty())
    {
        return 0;
    }

    *out_data = s->audio_output.data();
    return static_cast<int>(s->audio_output.size());
}

bool VoxCPM2Model::PushReferenceAudio(RSState &state, const float *samples,
                                      int n_samples, int sample_rate,
                                      ggml_backend_sched_t sched)
{
    (void)sched;
    return SetReferenceAudio(state, samples, n_samples, sample_rate);
}

void VoxCPM2Model::SetDiffusionSteps(int n_steps)
{
    if (n_steps > 0)
    {
        diffusion_steps_ = n_steps;
    }
}

void VoxCPM2Model::SetSeed(uint64_t seed)
{
    seed_ = seed;
}

void VoxCPM2Model::ConfigureState(VoxCPM2State &state, uint64_t seed,
                                  int diffusion_steps) const
{
    state.generation_seed = seed;
    state.diffusion_steps = diffusion_steps > 0 ? diffusion_steps : 8;
}

// =====================================================================
// Voice Cloning Interface
// =====================================================================

bool VoxCPM2Model::SetReferenceAudio(RSState &state, const float *audio,
                                     int n_samples, int sample_rate)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return false;
    }

    if (!audio || n_samples <= 0)
    {
        RS_LOG_ERR("VoxCPM2: Invalid reference audio");
        return false;
    }

    // Store reference audio
    s->reference_audio.assign(audio, audio + n_samples);
    s->reference_sample_rate = sample_rate;
    s->has_reference = true;
    s->reference_encoded = false;

    // Determine cloning mode
    if (s->has_prompt)
    {
        s->clone_mode = VoxCPM2State::ULTIMATE;
        RS_LOG_INFO("VoxCPM2: Cloning mode set to ULTIMATE (prompt + reference)");
    }
    else if (s->has_style)
    {
        s->clone_mode = VoxCPM2State::CONTROLLABLE;
        RS_LOG_INFO("VoxCPM2: Cloning mode set to CONTROLLABLE (reference + style)");
    }
    else
    {
        s->clone_mode = VoxCPM2State::BASIC;
        RS_LOG_INFO("VoxCPM2: Cloning mode set to BASIC (reference only)");
    }

    RS_LOG_INFO("VoxCPM2: Reference audio set (%d samples, %d Hz)",
                n_samples, sample_rate);
    return true;
}

bool VoxCPM2Model::SetPromptAudio(RSState &state, const float *audio,
                                  int n_samples, int sample_rate,
                                  const char *transcript)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return false;
    }

    if (!audio || n_samples <= 0 || !transcript)
    {
        RS_LOG_ERR("VoxCPM2: Invalid prompt audio or transcript");
        return false;
    }

    // Store prompt audio and transcript
    s->prompt_audio.assign(audio, audio + n_samples);
    s->prompt_text = transcript;
    s->has_prompt = true;

    // Ultimate cloning requires both prompt and reference
    if (s->has_reference)
    {
        s->clone_mode = VoxCPM2State::ULTIMATE;
        RS_LOG_INFO("VoxCPM2: Cloning mode set to ULTIMATE");
    }

    RS_LOG_INFO("VoxCPM2: Prompt audio set (%d samples, %d Hz, text: '%s')",
                n_samples, sample_rate, transcript);
    return true;
}

bool VoxCPM2Model::SetStyleDescription(RSState &state, const char *style)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return false;
    }

    if (!style || !*style)
    {
        RS_LOG_ERR("VoxCPM2: Empty style description");
        return false;
    }

    s->style_description = style;
    s->has_style = true;

    // Update cloning mode
    if (s->has_reference)
    {
        s->clone_mode = VoxCPM2State::CONTROLLABLE;
        RS_LOG_INFO("VoxCPM2: Cloning mode set to CONTROLLABLE");
    }

    RS_LOG_INFO("VoxCPM2: Style description set: '%s'", style);
    return true;
}

bool VoxCPM2Model::SetVoiceDesign(RSState &state, const char *voice_desc)
{
    auto *s = dynamic_cast<VoxCPM2State *>(&state);
    if (!s)
    {
        RS_LOG_ERR("VoxCPM2: Invalid state type");
        return false;
    }

    if (!voice_desc || !*voice_desc)
    {
        RS_LOG_ERR("VoxCPM2: Empty voice description");
        return false;
    }

    s->voice_description = voice_desc;
    s->has_voice_design = true;
    s->clone_mode = VoxCPM2State::VOICE_DESIGN;

    RS_LOG_INFO("VoxCPM2: Voice design mode: '%s'", voice_desc);
    return true;
}

std::vector<int32_t> VoxCPM2Model::tokenize(const std::string &text)
{
    auto utf8_len_at = [](const std::string &s, size_t pos) -> size_t {
        const unsigned char c = static_cast<unsigned char>(s[pos]);
        if ((c & 0x80) == 0) return 1;
        if ((c & 0xE0) == 0xC0) return std::min<size_t>(2, s.size() - pos);
        if ((c & 0xF0) == 0xE0) return std::min<size_t>(3, s.size() - pos);
        if ((c & 0xF8) == 0xF0) return std::min<size_t>(4, s.size() - pos);
        return 1;
    };

    auto utf8_codepoint = [](const std::string &s) -> uint32_t {
        if (s.empty()) return 0;
        const unsigned char c0 = static_cast<unsigned char>(s[0]);
        if (c0 < 0x80) return c0;
        if ((c0 & 0xE0) == 0xC0 && s.size() >= 2)
        {
            return ((c0 & 0x1F) << 6) |
                   (static_cast<unsigned char>(s[1]) & 0x3F);
        }
        if ((c0 & 0xF0) == 0xE0 && s.size() >= 3)
        {
            return ((c0 & 0x0F) << 12) |
                   ((static_cast<unsigned char>(s[1]) & 0x3F) << 6) |
                   (static_cast<unsigned char>(s[2]) & 0x3F);
        }
        if ((c0 & 0xF8) == 0xF0 && s.size() >= 4)
        {
            return ((c0 & 0x07) << 18) |
                   ((static_cast<unsigned char>(s[1]) & 0x3F) << 12) |
                   ((static_cast<unsigned char>(s[2]) & 0x3F) << 6) |
                   (static_cast<unsigned char>(s[3]) & 0x3F);
        }
        return 0xFFFD;
    };

    auto is_cjk = [&](const std::string &piece) -> bool {
        const uint32_t cp = utf8_codepoint(piece);
        return (cp >= 0x4E00 && cp <= 0x9FFF) ||
               (cp >= 0x3400 && cp <= 0x4DBF) ||
               (cp >= 0xF900 && cp <= 0xFAFF) ||
               (cp >= 0x20000 && cp <= 0x2A6DF);
    };

    auto split_utf8 = [&](const std::string &s) {
        std::vector<std::string> pieces;
        for (size_t pos = 0; pos < s.size();)
        {
            const size_t len = utf8_len_at(s, pos);
            pieces.emplace_back(s, pos, len);
            pos += len;
        }
        return pieces;
    };

    std::vector<int32_t> tokens;

    std::string normalized = "\xE2\x96\x81";
    for (char c : text)
    {
        if (c == ' ')
        {
            normalized += "\xE2\x96\x81";
        }
        else
        {
            normalized.push_back(c);
        }
    }

    std::vector<std::string> pieces = split_utf8(normalized);
    while (pieces.size() > 1)
    {
        int best_rank = INT_MAX;
        int best_pos = -1;
        for (int i = 0; i + 1 < static_cast<int>(pieces.size()); ++i)
        {
            auto it = merge_rank_.find(pieces[i] + " " + pieces[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank)
            {
                best_rank = it->second;
                best_pos = i;
            }
        }
        if (best_pos < 0)
        {
            break;
        }
        pieces[best_pos] += pieces[best_pos + 1];
        pieces.erase(pieces.begin() + best_pos + 1);
    }

    for (const std::string &piece : pieces)
    {
        auto it = token_to_id_.find(piece);
        if (it == token_to_id_.end())
        {
            tokens.push_back(token_unk_);
            continue;
        }

        const std::string clean =
            piece.rfind("\xE2\x96\x81", 0) == 0 ? piece.substr(3) : piece;
        const std::vector<std::string> chars = split_utf8(clean);
        bool split_multi_cjk = chars.size() >= 2;
        for (const std::string &ch : chars)
        {
            if (!is_cjk(ch))
            {
                split_multi_cjk = false;
                break;
            }
        }

        if (split_multi_cjk)
        {
            for (const std::string &ch : chars)
            {
                auto jt = token_to_id_.find(ch);
                tokens.push_back(jt == token_to_id_.end() ? token_unk_ : jt->second);
            }
        }
        else
        {
            tokens.push_back(it->second);
        }
    }

    if (token_audio_start_ >= 0)
    {
        tokens.push_back(token_audio_start_);
    }

    return tokens;
}

// =====================================================================
// Forward Pass Implementations
// =====================================================================

bool VoxCPM2Model::RunReferenceEncoder(VoxCPM2State &state, ggml_backend_sched_t sched)
{
    if (!state.has_reference && !state.has_prompt)
    {
        // No reference audio, use default speaker embedding
        return true;
    }

    RS_LOG_INFO("VoxCPM2: Running reference encoder");

    // TODO: Implement reference encoder
    // For now, create dummy speaker embeddings
    int emb_dim = 256;  // Speaker embedding dimension
    state.reference_embeddings.resize(emb_dim, 0.01f);

    state.reference_encoded = true;
    return true;
}

bool VoxCPM2Model::RunTextEncoder(VoxCPM2State &state, ggml_backend_sched_t sched)
{
    RS_LOG_INFO("VoxCPM2: Building TSLM token embeddings");

    const int n_text_tokens = state.token_ids.size();
    const int d_model = hparams_.lm_hidden_size;

    struct ggml_tensor *base_lm_embed = get_tensor("base_lm.embed_tokens.weight");
    if (!base_lm_embed)
    {
        RS_LOG_ERR("VoxCPM2: base_lm.embed_tokens.weight not found");
        return false;
    }

    if (base_lm_embed->ne[0] != d_model)
    {
        RS_LOG_ERR("VoxCPM2: unexpected token embedding dim %lld (expected %d)",
                   (long long)base_lm_embed->ne[0], d_model);
        return false;
    }

    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context *ctx0 = ggml_init(params);
    if (!ctx0)
    {
        RS_LOG_ERR("VoxCPM2: Failed to create compute context");
        return false;
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);
    struct ggml_tensor *token_ids_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(token_ids_tensor, "token_ids");
    ggml_set_input(token_ids_tensor);

    struct ggml_tensor *output = ggml_cont(ctx0, ggml_get_rows(ctx0, base_lm_embed, token_ids_tensor));
    ggml_set_name(output, "encoder_output");
    ggml_build_forward_expand(gf, output);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
    {
        RS_LOG_ERR("VoxCPM2: Failed to allocate graph");
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(token_ids_tensor, state.token_ids.data(), 0,
                           n_text_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
    {
        RS_LOG_ERR("VoxCPM2: token embedding compute failed");
        ggml_free(ctx0);
        return false;
    }

    state.text_seq_len = n_text_tokens;
    state.text_embeddings.resize(n_text_tokens * d_model);
    ggml_backend_tensor_get(output, state.text_embeddings.data(), 0,
                           n_text_tokens * d_model * sizeof(float));

    ggml_free(ctx0);

    // 输出一些统计信息
    float sum = 0.0f, sum_sq = 0.0f;
    for (float v : state.text_embeddings)
    {
        sum += v;
        sum_sq += v * v;
    }
    float mean = sum / state.text_embeddings.size();
    float std = sqrtf(sum_sq / state.text_embeddings.size() - mean * mean);

    RS_LOG_INFO("VoxCPM2: TSLM embeddings: [%d, %d], mean=%.4f, std=%.4f",
                n_text_tokens, d_model, mean, std);
    return true;
}

bool VoxCPM2Model::RunBaseLM(VoxCPM2State &state, ggml_backend_sched_t sched)
{
    RS_LOG_INFO("VoxCPM2: Running Base LM prefill");

    const int n_text_tokens = state.text_seq_len;
    const int hidden_dim = hparams_.lm_hidden_size;
    const int n_layers = hparams_.lm_num_hidden_layers;
    const int n_heads = hparams_.lm_num_attention_heads;
    const int n_kv_heads = hparams_.lm_num_key_value_heads;
    const int head_dim = hidden_dim / n_heads;

    struct ggml_init_params params = {
        .mem_size   = 1024 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context *ctx0 = ggml_init(params);
    if (!ctx0)
    {
        RS_LOG_ERR("VoxCPM2: Failed to create compute context for Base LM");
        return false;
    }

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);
    auto to_f32 = [&](struct ggml_tensor *tensor) {
        return tensor && tensor->type != GGML_TYPE_F32 ?
            ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
    };

    struct ggml_tensor *text_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, n_text_tokens);
    ggml_set_name(text_emb, "text_embeddings");
    ggml_set_input(text_emb);

    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_text_tokens);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    struct ggml_tensor *attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_text_tokens, n_text_tokens);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    struct ggml_tensor *hidden = text_emb;
    ggml_set_name(hidden, "lm_input");
    std::vector<struct ggml_tensor *> k_cache_tensors;
    std::vector<struct ggml_tensor *> v_cache_tensors;
    k_cache_tensors.reserve(n_layers);
    v_cache_tensors.reserve(n_layers);

    RS_LOG_INFO("VoxCPM2: Base LM prefill input: [%d, %d], layers=%d",
                n_text_tokens, hidden_dim, n_layers);

    for (int layer = 0; layer < n_layers; ++layer)
    {
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "base_lm.layers.%d", layer);

        // Layer norm
        struct ggml_tensor *ln_weight = get_tensor(std::string(prefix) + ".input_layernorm.weight");
        if (!ln_weight)
        {
            RS_LOG_WARN("VoxCPM2: Layer %d weights not found, stopping", layer);
            break;
        }

        struct ggml_tensor *ln_hidden = ggml_rms_norm(ctx0, hidden, 1e-6f);
        ln_hidden = ggml_mul(ctx0, ln_hidden, to_f32(ln_weight));

        // Self-attention (GQA)
        struct ggml_tensor *q_weight = get_tensor(std::string(prefix) + ".self_attn.q_proj.weight");
        struct ggml_tensor *k_weight = get_tensor(std::string(prefix) + ".self_attn.k_proj.weight");
        struct ggml_tensor *v_weight = get_tensor(std::string(prefix) + ".self_attn.v_proj.weight");
        struct ggml_tensor *o_weight = get_tensor(std::string(prefix) + ".self_attn.o_proj.weight");

        if (!q_weight || !k_weight || !v_weight || !o_weight)
        {
            RS_LOG_WARN("VoxCPM2: Attention weights not found for layer %d", layer);
            break;
        }

        // Q, K, V projections
        struct ggml_tensor *Q = ggml_mul_mat(ctx0, q_weight, ln_hidden);
        struct ggml_tensor *K = ggml_mul_mat(ctx0, k_weight, ln_hidden);
        struct ggml_tensor *V = ggml_mul_mat(ctx0, v_weight, ln_hidden);

        Q = ggml_reshape_4d(ctx0, Q, head_dim, n_heads, n_text_tokens, 1);
        K = ggml_reshape_4d(ctx0, K, head_dim, n_kv_heads, n_text_tokens, 1);
        V = ggml_reshape_4d(ctx0, V, head_dim, n_kv_heads, n_text_tokens, 1);

        Q = ggml_rope_ext_inplace(ctx0, Q, positions,
                                  get_tensor("base_lm.rope_short_factors"),
                                  head_dim, GGML_ROPE_TYPE_NEOX,
                                  hparams_.lm_max_position_embeddings,
                                  hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f);
        K = ggml_rope_ext_inplace(ctx0, K, positions,
                                  get_tensor("base_lm.rope_short_factors"),
                                  head_dim, GGML_ROPE_TYPE_NEOX,
                                  hparams_.lm_max_position_embeddings,
                                  hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f);

        // Permute to ggml_flash_attn_ext layout: [head_dim, seq, heads].
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
        ggml_set_output(K);
        ggml_set_output(V);
        k_cache_tensors.push_back(K);
        v_cache_tensors.push_back(V);

        const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        struct ggml_tensor *attn_out = ggml_flash_attn_ext(
            ctx0, Q, K, V, attn_mask, attn_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);

        attn_out = ggml_reshape_2d(ctx0, attn_out, hidden_dim, n_text_tokens);

        // Output projection
        attn_out = ggml_mul_mat(ctx0, o_weight, attn_out);

        // Residual connection
        hidden = ggml_add(ctx0, hidden, attn_out);

        // FFN
        struct ggml_tensor *ffn_ln_weight = get_tensor(std::string(prefix) + ".post_attention_layernorm.weight");
        struct ggml_tensor *gate_weight = get_tensor(std::string(prefix) + ".mlp.gate_proj.weight");
        struct ggml_tensor *up_weight = get_tensor(std::string(prefix) + ".mlp.up_proj.weight");
        struct ggml_tensor *down_weight = get_tensor(std::string(prefix) + ".mlp.down_proj.weight");

        if (ffn_ln_weight && gate_weight && up_weight && down_weight)
        {
            struct ggml_tensor *ffn_hidden = ggml_rms_norm(ctx0, hidden, 1e-6f);
            ffn_hidden = ggml_mul(ctx0, ffn_hidden, to_f32(ffn_ln_weight));

            // SwiGLU
            struct ggml_tensor *gate_out = ggml_mul_mat(ctx0, gate_weight, ffn_hidden);
            struct ggml_tensor *up_out = ggml_mul_mat(ctx0, up_weight, ffn_hidden);
            struct ggml_tensor *ffn_out = ggml_mul(ctx0, ggml_silu(ctx0, gate_out), up_out);

            // Down projection
            ffn_out = ggml_mul_mat(ctx0, down_weight, ffn_out);

            // Residual
            hidden = ggml_add(ctx0, hidden, ffn_out);
        }
    }

    // Final layer norm
    struct ggml_tensor *norm_weight = get_tensor("base_lm.norm.weight");
    if (norm_weight)
    {
        hidden = ggml_rms_norm(ctx0, hidden, 1e-6f);
        hidden = ggml_mul(ctx0, hidden, to_f32(norm_weight));
    }

    ggml_build_forward_expand(gf, hidden);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
    {
        RS_LOG_ERR("VoxCPM2: Failed to allocate Base LM graph");
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(text_emb, state.text_embeddings.data(), 0,
                           n_text_tokens * hidden_dim * sizeof(float));

    std::vector<int32_t> pos_data(n_text_tokens);
    for (int i = 0; i < n_text_tokens; ++i)
    {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(positions, pos_data.data(), 0,
                           n_text_tokens * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask_data(static_cast<size_t>(n_text_tokens) * n_text_tokens);
    for (int q = 0; q < n_text_tokens; ++q)
    {
        for (int k = 0; k < n_text_tokens; ++k)
        {
            const float val = k <= q ? 0.0f : -INFINITY;
            mask_data[static_cast<size_t>(q) * n_text_tokens + k] = ggml_fp32_to_fp16(val);
        }
    }
    ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                           mask_data.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
    {
        RS_LOG_ERR("VoxCPM2: Base LM compute failed");
        ggml_free(ctx0);
        return false;
    }

    std::vector<float> lm_output(n_text_tokens * hidden_dim);
    ggml_backend_tensor_get(hidden, lm_output.data(), 0,
                           n_text_tokens * hidden_dim * sizeof(float));

    const size_t cache_values_per_layer =
        static_cast<size_t>(head_dim) * n_text_tokens * n_kv_heads;
    state.base_lm_k_cache.resize(
        cache_values_per_layer * k_cache_tensors.size());
    state.base_lm_v_cache.resize(
        cache_values_per_layer * v_cache_tensors.size());
    for (size_t layer = 0; layer < k_cache_tensors.size(); ++layer)
    {
        ggml_backend_tensor_get(
            k_cache_tensors[layer],
            state.base_lm_k_cache.data() + layer * cache_values_per_layer,
            0, cache_values_per_layer * sizeof(float));
        ggml_backend_tensor_get(
            v_cache_tensors[layer],
            state.base_lm_v_cache.data() + layer * cache_values_per_layer,
            0, cache_values_per_layer * sizeof(float));
    }
    state.ar_cache_length = n_text_tokens;

    ggml_free(ctx0);

    state.base_lm_hidden.assign(lm_output.end() - hidden_dim, lm_output.end());
    state.base_lm_outputs = std::move(lm_output);

    const int tokens_per_text = 6;
    state.audio_token_len = std::max(4, n_text_tokens * tokens_per_text);
    state.audio_tokens.resize(state.audio_token_len);

    for (int i = 0; i < state.audio_token_len; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < 64; ++j)
        {
            sum += state.base_lm_hidden[(i * 64 + j) % hidden_dim];
        }

        int token_id = 100 + (int)(fabs(sum) * 50.0f + i) % 900;
        state.audio_tokens[i] = token_id;
    }

    RS_LOG_INFO("VoxCPM2: Base LM prefill completed, provisional patch count=%d",
                state.audio_token_len);
    return true;
}

bool VoxCPM2Model::RunScalarQuantization(VoxCPM2State &state, ggml_backend_sched_t sched)
{
    RS_LOG_INFO("VoxCPM2: Running scalar quantization");

    if (state.base_lm_hidden.size() != static_cast<size_t>(hparams_.lm_hidden_size))
    {
        RS_LOG_ERR("VoxCPM2: Base LM hidden state missing for FSQ");
        return false;
    }

    struct ggml_tensor *in_w = get_tensor("fsq_layer.in_proj.weight");
    struct ggml_tensor *in_b = get_tensor("fsq_layer.in_proj.bias");
    struct ggml_tensor *out_w = get_tensor("fsq_layer.out_proj.weight");
    struct ggml_tensor *out_b = get_tensor("fsq_layer.out_proj.bias");
    struct ggml_tensor *lm_to_dit_w = get_tensor("lm_to_dit_proj.weight");
    struct ggml_tensor *lm_to_dit_b = get_tensor("lm_to_dit_proj.bias");
    struct ggml_tensor *fusion_w = get_tensor("fusion_concat_proj.weight");
    struct ggml_tensor *fusion_b = get_tensor("fusion_concat_proj.bias");
    struct ggml_tensor *res_to_dit_w = get_tensor("res_to_dit_proj.weight");
    struct ggml_tensor *res_to_dit_b = get_tensor("res_to_dit_proj.bias");
    struct ggml_tensor *ralm_norm_w = get_tensor("residual_lm.norm.weight");

    if (!in_w || !in_b || !out_w || !out_b)
    {
        RS_LOG_ERR("VoxCPM2: FSQ tensors are missing");
        return false;
    }

    const int hidden_dim = hparams_.lm_hidden_size;
    const int fsq_dim = hparams_.scalar_quantization_latent_dim;
    const int dit_dim = hparams_.dit_hidden_dim;
    const int n_pos = static_cast<int>(state.base_lm_outputs.size()) / hidden_dim;
    const bool has_ralm =
        n_pos > 0 &&
        fusion_w && fusion_b && res_to_dit_w && res_to_dit_b && ralm_norm_w &&
        get_tensor("residual_lm.layers.0.input_layernorm.weight");

    struct ggml_init_params params = {
        .mem_size   = 512 * 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    struct ggml_context *ctx0 = ggml_init(params);
    if (!ctx0)
    {
        RS_LOG_ERR("VoxCPM2: Failed to create FSQ context");
        return false;
    }

    auto to_f32 = [&](struct ggml_tensor *tensor) {
        return tensor && tensor->type != GGML_TYPE_F32 ?
            ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
    };
    auto add_bias = [&](struct ggml_tensor *tensor, struct ggml_tensor *bias,
                        int out_dim) {
        if (!bias)
        {
            return tensor;
        }
        struct ggml_tensor *bias_view =
            ggml_reshape_2d(ctx0, to_f32(bias), out_dim, 1);
        if (tensor->ne[1] != 1)
        {
            bias_view = ggml_repeat(ctx0, bias_view, tensor);
        }
        return ggml_add(ctx0, tensor, bias_view);
    };

    struct ggml_cgraph *gf = ggml_new_graph(ctx0);
    struct ggml_tensor *hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, 1);
    ggml_set_name(hidden, "base_lm_hidden");
    ggml_set_input(hidden);

    struct ggml_tensor *x = ggml_mul_mat(ctx0, in_w, hidden);
    x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, to_f32(in_b), fsq_dim, 1));
    x = ggml_tanh(ctx0, x);
    x = ggml_scale(ctx0, x, static_cast<float>(hparams_.scalar_quantization_scale));
    x = ggml_round(ctx0, x);
    x = ggml_scale(ctx0, x, 1.0f / static_cast<float>(hparams_.scalar_quantization_scale));

    struct ggml_tensor *fsq_out = ggml_mul_mat(ctx0, out_w, x);
    fsq_out = add_bias(fsq_out, out_b, hidden_dim);

    struct ggml_tensor *lm_cond = hidden;
    if (lm_to_dit_w && lm_to_dit_b)
    {
        lm_cond = ggml_mul_mat(ctx0, lm_to_dit_w, hidden);
        lm_cond = add_bias(lm_cond, lm_to_dit_b, dit_dim);
    }

    struct ggml_tensor *res_cond = nullptr;
    struct ggml_tensor *all_hidden = nullptr;
    struct ggml_tensor *attn_mask = nullptr;
    std::vector<struct ggml_tensor *> ralm_k_cache_tensors;
    std::vector<struct ggml_tensor *> ralm_v_cache_tensors;
    if (has_ralm)
    {
        all_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, n_pos);
        ggml_set_name(all_hidden, "base_lm_outputs");
        ggml_set_input(all_hidden);

        attn_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pos, n_pos);
        ggml_set_name(attn_mask, "ralm_attn_mask");
        ggml_set_input(attn_mask);

        struct ggml_tensor *zero_feat = ggml_scale(ctx0, all_hidden, 0.0f);
        struct ggml_tensor *fusion_in = ggml_concat(ctx0, all_hidden, zero_feat, 0);
        struct ggml_tensor *ralm_hidden = ggml_mul_mat(ctx0, fusion_w, fusion_in);
        ralm_hidden = add_bias(ralm_hidden, fusion_b, hidden_dim);

        const int n_heads = hparams_.lm_num_attention_heads;
        const int n_kv_heads = hparams_.lm_num_key_value_heads;
        const int head_dim = hidden_dim / n_heads;
        const int n_ralm_layers = 8;

        for (int layer = 0; layer < n_ralm_layers; ++layer)
        {
            const std::string prefix =
                "residual_lm.layers." + std::to_string(layer);

            struct ggml_tensor *ln_w = get_tensor(prefix + ".input_layernorm.weight");
            struct ggml_tensor *q_w = get_tensor(prefix + ".self_attn.q_proj.weight");
            struct ggml_tensor *k_w = get_tensor(prefix + ".self_attn.k_proj.weight");
            struct ggml_tensor *v_w = get_tensor(prefix + ".self_attn.v_proj.weight");
            struct ggml_tensor *o_w = get_tensor(prefix + ".self_attn.o_proj.weight");
            struct ggml_tensor *ffn_ln_w = get_tensor(prefix + ".post_attention_layernorm.weight");
            struct ggml_tensor *gate_w = get_tensor(prefix + ".mlp.gate_proj.weight");
            struct ggml_tensor *up_w = get_tensor(prefix + ".mlp.up_proj.weight");
            struct ggml_tensor *down_w = get_tensor(prefix + ".mlp.down_proj.weight");
            if (!ln_w || !q_w || !k_w || !v_w || !o_w ||
                !ffn_ln_w || !gate_w || !up_w || !down_w)
            {
                RS_LOG_WARN("VoxCPM2: RALM layer %d tensors missing; using partial RALM", layer);
                break;
            }

            struct ggml_tensor *residual = ralm_hidden;
            struct ggml_tensor *cur = ggml_rms_norm(ctx0, ralm_hidden, hparams_.lm_rms_norm_eps);
            cur = ggml_mul(ctx0, cur, to_f32(ln_w));

            struct ggml_tensor *Q = ggml_mul_mat(ctx0, q_w, cur);
            struct ggml_tensor *K = ggml_mul_mat(ctx0, k_w, cur);
            struct ggml_tensor *V = ggml_mul_mat(ctx0, v_w, cur);
            Q = ggml_reshape_4d(ctx0, Q, head_dim, n_heads, n_pos, 1);
            K = ggml_reshape_4d(ctx0, K, head_dim, n_kv_heads, n_pos, 1);
            V = ggml_reshape_4d(ctx0, V, head_dim, n_kv_heads, n_pos, 1);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
            ggml_set_output(K);
            ggml_set_output(V);
            ralm_k_cache_tensors.push_back(K);
            ralm_v_cache_tensors.push_back(V);

            const float attn_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
            struct ggml_tensor *attn = ggml_flash_attn_ext(
                ctx0, Q, K, V, attn_mask, attn_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(ctx0, attn, hidden_dim, n_pos);
            attn = ggml_mul_mat(ctx0, o_w, attn);
            ralm_hidden = ggml_add(ctx0, residual, attn);

            residual = ralm_hidden;
            cur = ggml_rms_norm(ctx0, ralm_hidden, hparams_.lm_rms_norm_eps);
            cur = ggml_mul(ctx0, cur, to_f32(ffn_ln_w));
            struct ggml_tensor *gate = ggml_mul_mat(ctx0, gate_w, cur);
            struct ggml_tensor *up = ggml_mul_mat(ctx0, up_w, cur);
            struct ggml_tensor *ffn = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
            ffn = ggml_mul_mat(ctx0, down_w, ffn);
            ralm_hidden = ggml_add(ctx0, residual, ffn);
        }

        ralm_hidden = ggml_cont(ctx0, ralm_hidden);
        struct ggml_tensor *ralm_last = ggml_view_2d(
            ctx0, ralm_hidden, hidden_dim, 1, ralm_hidden->nb[1],
            static_cast<size_t>(n_pos - 1) * ralm_hidden->nb[1]);
        ralm_last = ggml_rms_norm(ctx0, ralm_last, hparams_.lm_rms_norm_eps);
        ralm_last = ggml_mul(ctx0, ralm_last, to_f32(ralm_norm_w));
        res_cond = ggml_mul_mat(ctx0, res_to_dit_w, ralm_last);
        res_cond = add_bias(res_cond, res_to_dit_b, dit_dim);
    }
    else
    {
        RS_LOG_WARN("VoxCPM2: RALM tensors unavailable; DiT mu second half remains zero");
    }

    ggml_build_forward_expand(gf, fsq_out);
    ggml_build_forward_expand(gf, lm_cond);
    if (res_cond)
    {
        ggml_build_forward_expand(gf, res_cond);
    }

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf))
    {
        RS_LOG_ERR("VoxCPM2: Failed to allocate FSQ/RALM graph");
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(hidden, state.base_lm_hidden.data(), 0,
                           state.base_lm_hidden.size() * sizeof(float));
    if (all_hidden)
    {
        ggml_backend_tensor_set(all_hidden, state.base_lm_outputs.data(), 0,
                               state.base_lm_outputs.size() * sizeof(float));
    }
    if (attn_mask)
    {
        std::vector<ggml_fp16_t> mask_data(static_cast<size_t>(n_pos) * n_pos);
        for (int q = 0; q < n_pos; ++q)
        {
            for (int k = 0; k < n_pos; ++k)
            {
                const float val = k <= q ? 0.0f : -INFINITY;
                mask_data[static_cast<size_t>(q) * n_pos + k] = ggml_fp32_to_fp16(val);
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                               mask_data.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
    {
        RS_LOG_ERR("VoxCPM2: FSQ/RALM compute failed");
        ggml_free(ctx0);
        return false;
    }

    std::vector<float> fsq_vec(static_cast<size_t>(fsq_out->ne[0]) * fsq_out->ne[1]);
    std::vector<float> lm_vec(static_cast<size_t>(lm_cond->ne[0]) * lm_cond->ne[1]);
    std::vector<float> res_vec(static_cast<size_t>(dit_dim), 0.0f);
    ggml_backend_tensor_get(fsq_out, fsq_vec.data(), 0, fsq_vec.size() * sizeof(float));
    ggml_backend_tensor_get(lm_cond, lm_vec.data(), 0, lm_vec.size() * sizeof(float));
    if (res_cond)
    {
        ggml_backend_tensor_get(res_cond, res_vec.data(), 0, res_vec.size() * sizeof(float));
    }
    if (!ralm_k_cache_tensors.empty())
    {
        const int head_dim = hidden_dim / hparams_.lm_num_attention_heads;
        const size_t cache_values_per_layer =
            static_cast<size_t>(head_dim) * n_pos *
            hparams_.lm_num_key_value_heads;
        state.residual_lm_k_cache.resize(
            cache_values_per_layer * ralm_k_cache_tensors.size());
        state.residual_lm_v_cache.resize(
            cache_values_per_layer * ralm_v_cache_tensors.size());
        for (size_t layer = 0; layer < ralm_k_cache_tensors.size(); ++layer)
        {
            ggml_backend_tensor_get(
                ralm_k_cache_tensors[layer],
                state.residual_lm_k_cache.data() +
                    layer * cache_values_per_layer,
                0, cache_values_per_layer * sizeof(float));
            ggml_backend_tensor_get(
                ralm_v_cache_tensors[layer],
                state.residual_lm_v_cache.data() +
                    layer * cache_values_per_layer,
                0, cache_values_per_layer * sizeof(float));
        }
    }
    ggml_free(ctx0);

    state.dit_mu.assign(static_cast<size_t>(2) * dit_dim, 0.0f);
    const int lm_copy_dim = std::min<int>(dit_dim, lm_vec.size());
    const int res_copy_dim = std::min<int>(dit_dim, res_vec.size());
    std::copy(lm_vec.begin(), lm_vec.begin() + lm_copy_dim, state.dit_mu.begin());
    std::copy(res_vec.begin(), res_vec.begin() + res_copy_dim,
              state.dit_mu.begin() + dit_dim);

    const int latent_dim = 64;
    int latent_frames = std::max(hparams_.patch_size, state.audio_token_len);

    state.quantized_latents.resize(latent_frames * latent_dim);

    for (int i = 0; i < latent_frames; ++i)
    {
        for (int d = 0; d < latent_dim; ++d)
        {
            const size_t idx = (static_cast<size_t>(i) * latent_dim + d) % fsq_vec.size();
            state.quantized_latents[static_cast<size_t>(i) * latent_dim + d] = fsq_vec[idx] * 0.02f;
        }
    }

    RS_LOG_INFO("VoxCPM2: FSQ/RALM built DiT mu [%d, %d] and provisional [%d, %d] latents",
                2, dit_dim, latent_frames, latent_dim);
    return true;
}

bool VoxCPM2Model::RunDiT(VoxCPM2State &state, ggml_backend_sched_t sched,
                          const AudioChunkCallback &emit)
{
    RS_LOG_INFO("VoxCPM2: Running DiT");

    if (state.dit_mu.size() != static_cast<size_t>(2 * hparams_.dit_hidden_dim))
    {
        RS_LOG_ERR("VoxCPM2: DiT conditioning mu is missing");
        return false;
    }

    const int feat_dim = 64;
    const int patch_size = hparams_.patch_size;
    const int d_model = hparams_.dit_hidden_dim;
    const int n_layers = hparams_.dit_num_layers;
    const int head_dim = hparams_.dit_kv_channels;
    const int single_seq_len = 2 + 1 + patch_size + patch_size;
    const int cfg_batch = hparams_.dit_inference_cfg_rate > 1.0f ? 2 : 1;
    const int seq_len = single_seq_len * cfg_batch;
    const int x_offset = 2 + 1 + patch_size;
    double dit_forward_ms = 0.0;
    double cfm_ms = 0.0;
    double locenc_ms = 0.0;
    double ar_update_ms = 0.0;
    size_t streamed_samples = 0;
    int stream_first_patches = 4;
    if (const char *value = std::getenv("RS_VOXCPM2_STREAM_FIRST_PATCHES"))
    {
        stream_first_patches = std::max(1, std::atoi(value));
    }
    int stream_chunk_patches = 2;
    if (const char *value = std::getenv("RS_VOXCPM2_STREAM_CHUNK_PATCHES"))
    {
        stream_chunk_patches = std::max(1, std::atoi(value));
    }
    int stream_holdback_ms = 80;
    if (const char *value = std::getenv("RS_VOXCPM2_STREAM_HOLDBACK_MS"))
    {
        stream_holdback_ms = std::max(0, std::min(500, std::atoi(value)));
    }

    int latent_frames = static_cast<int>(state.quantized_latents.size()) / feat_dim;
    if (latent_frames <= 0)
    {
        RS_LOG_ERR("VoxCPM2: no quantized latents for DiT");
        return false;
    }
    const int provisional_patches = (latent_frames + patch_size - 1) / patch_size;

    auto bf16_round = [](float value) {
        return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
    };
    auto sinusoidal = [&](float t) {
        std::vector<float> emb(d_model, 0.0f);
        const int half = d_model / 2;
        const float log_base =
            std::log(10000.0f) / std::max(1, half - 1);
        const float scaled_t = bf16_round(
            bf16_round(1000.0f) * bf16_round(t));
        for (int i = 0; i < half; ++i)
        {
            const float index = bf16_round(static_cast<float>(i));
            const float exponent = bf16_round(-index * log_base);
            const float inv_freq = bf16_round(std::exp(exponent));
            const float arg = bf16_round(scaled_t * inv_freq);
            emb[i] = bf16_round(std::sin(arg));
            emb[i + half] = bf16_round(std::cos(arg));
        }
        return emb;
    };

    struct ggml_context *patch_ctx = nullptr;
    struct ggml_cgraph *patch_graph = nullptr;
    struct ggml_tensor *patch_x_in = nullptr;
    struct ggml_tensor *patch_cond_in = nullptr;
    struct ggml_tensor *patch_mu_in = nullptr;
    struct ggml_tensor *patch_t_sin = nullptr;
    struct ggml_tensor *patch_dt_sin = nullptr;
    struct ggml_tensor *patch_positions = nullptr;
    struct ggml_tensor *patch_attn_mask = nullptr;
    struct ggml_tensor *patch_velocity = nullptr;
    auto reset_patch_graph = [&]() {
        if (patch_ctx)
        {
            ggml_backend_sched_reset(sched);
            ggml_free(patch_ctx);
            patch_ctx = nullptr;
            patch_graph = nullptr;
            patch_x_in = nullptr;
            patch_cond_in = nullptr;
            patch_mu_in = nullptr;
            patch_t_sin = nullptr;
            patch_dt_sin = nullptr;
            patch_positions = nullptr;
            patch_attn_mask = nullptr;
            patch_velocity = nullptr;
        }
    };

    auto run_patch = [&](const std::vector<float> &x_patch,
                         const std::vector<float> &cond_patch,
                         const std::vector<float> &mu_data,
                         float t_scalar,
                         float dt_scalar) {
        if (!patch_ctx)
        {
            struct ggml_init_params params = {
                .mem_size   = 512 * 1024 * 1024,
                .mem_buffer = nullptr,
                .no_alloc   = true,
            };
            patch_ctx = ggml_init(params);
            if (!patch_ctx)
            {
                return std::vector<float>(patch_size * feat_dim, 0.0f);
            }
            struct ggml_context *ctx0 = patch_ctx;

            auto to_f32 = [&](struct ggml_tensor *tensor) {
                return tensor && tensor->type != GGML_TYPE_F32 ?
                    ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
            };

            patch_graph = ggml_new_graph(ctx0);
            patch_x_in = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, feat_dim, patch_size * cfg_batch);
            ggml_set_name(patch_x_in, "dit_x");
            ggml_set_input(patch_x_in);
            patch_cond_in = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, feat_dim, patch_size * cfg_batch);
            ggml_set_name(patch_cond_in, "dit_cond");
            ggml_set_input(patch_cond_in);
            patch_mu_in = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, d_model, 2 * cfg_batch);
            ggml_set_name(patch_mu_in, "dit_mu");
            ggml_set_input(patch_mu_in);
            patch_t_sin = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, d_model, cfg_batch);
            ggml_set_name(patch_t_sin, "dit_time");
            ggml_set_input(patch_t_sin);
            patch_dt_sin = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, d_model, cfg_batch);
            ggml_set_name(patch_dt_sin, "dit_dt");
            ggml_set_input(patch_dt_sin);
            patch_positions = ggml_new_tensor_1d(
                ctx0, GGML_TYPE_I32, seq_len);
            ggml_set_name(patch_positions, "dit_positions");
            ggml_set_input(patch_positions);
            if (cfg_batch > 1)
            {
                patch_attn_mask = ggml_new_tensor_2d(
                    ctx0, GGML_TYPE_F16, seq_len, seq_len);
                ggml_set_name(patch_attn_mask, "dit_cfg_mask");
                ggml_set_input(patch_attn_mask);
            }
        auto linear = [&](const std::string &w_name, const std::string &b_name,
                          struct ggml_tensor *input, int out_dim) {
            struct ggml_tensor *w = get_tensor(w_name);
            struct ggml_tensor *b = get_tensor(b_name);
            struct ggml_tensor *y = ggml_mul_mat(ctx0, w, input);
            if (b)
            {
                y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, to_f32(b), out_dim, 1));
            }
            return y;
        };
        auto mlp_time = [&](const std::string &prefix, struct ggml_tensor *input) {
            struct ggml_tensor *y = linear(prefix + ".linear_1.weight",
                                           prefix + ".linear_1.bias", input, d_model);
            y = ggml_silu(ctx0, y);
            y = linear(prefix + ".linear_2.weight",
                       prefix + ".linear_2.bias", y, d_model);
            return y;
        };

        struct ggml_tensor *time_emb = mlp_time("feat_decoder.estimator.time_mlp", patch_t_sin);
        struct ggml_tensor *dt_emb = mlp_time("feat_decoder.estimator.delta_time_mlp", patch_dt_sin);
        time_emb = ggml_add(ctx0, time_emb, dt_emb);

        struct ggml_tensor *x_proj = linear("feat_decoder.estimator.in_proj.weight",
                                            "feat_decoder.estimator.in_proj.bias", patch_x_in, d_model);
        struct ggml_tensor *cond_proj = linear("feat_decoder.estimator.cond_proj.weight",
                                               "feat_decoder.estimator.cond_proj.bias", patch_cond_in, d_model);

        struct ggml_tensor *cur = nullptr;
        for (int batch = 0; batch < cfg_batch; ++batch)
        {
            struct ggml_tensor *mu_part = ggml_view_2d(
                ctx0, patch_mu_in, d_model, 2, patch_mu_in->nb[1],
                static_cast<size_t>(batch * 2) * patch_mu_in->nb[1]);
            struct ggml_tensor *time_part = ggml_view_2d(
                ctx0, time_emb, d_model, 1, time_emb->nb[1],
                static_cast<size_t>(batch) * time_emb->nb[1]);
            struct ggml_tensor *cond_part = ggml_view_2d(
                ctx0, cond_proj, d_model, patch_size, cond_proj->nb[1],
                static_cast<size_t>(batch * patch_size) * cond_proj->nb[1]);
            struct ggml_tensor *x_part = ggml_view_2d(
                ctx0, x_proj, d_model, patch_size, x_proj->nb[1],
                static_cast<size_t>(batch * patch_size) * x_proj->nb[1]);
            struct ggml_tensor *sequence = ggml_concat(
                ctx0, mu_part, time_part, 1);
            sequence = ggml_concat(ctx0, sequence, cond_part, 1);
            sequence = ggml_concat(ctx0, sequence, x_part, 1);
            cur = cur ? ggml_concat(ctx0, cur, sequence, 1) : sequence;
        }

        for (int layer = 0; layer < n_layers; ++layer)
        {
            const std::string prefix =
                "feat_decoder.estimator.decoder.layers." + std::to_string(layer);

            struct ggml_tensor *residual = cur;
            cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
            cur = ggml_mul(ctx0, cur, to_f32(get_tensor(prefix + ".input_layernorm.weight")));

            struct ggml_tensor *q_w = get_tensor(prefix + ".self_attn.q_proj.weight");
            struct ggml_tensor *k_w = get_tensor(prefix + ".self_attn.k_proj.weight");
            struct ggml_tensor *v_w = get_tensor(prefix + ".self_attn.v_proj.weight");
            struct ggml_tensor *o_w = get_tensor(prefix + ".self_attn.o_proj.weight");

            const int q_dim = q_w->ne[1];
            const int kv_dim = k_w->ne[1];
            const int n_heads = q_dim / head_dim;
            const int n_kv_heads = kv_dim / head_dim;

            struct ggml_tensor *Q = nullptr;
            struct ggml_tensor *K = nullptr;
            struct ggml_tensor *V = nullptr;
            if (layer < static_cast<int>(dit_qkv_weights_.size()))
            {
                struct ggml_tensor *qkv = ggml_mul_mat(
                    ctx0, dit_qkv_weights_[layer], cur);
                Q = ggml_view_2d(
                    ctx0, qkv, q_dim, seq_len, qkv->nb[1], 0);
                K = ggml_view_2d(
                    ctx0, qkv, kv_dim, seq_len, qkv->nb[1],
                    static_cast<size_t>(q_dim) * qkv->nb[0]);
                V = ggml_view_2d(
                    ctx0, qkv, kv_dim, seq_len, qkv->nb[1],
                    static_cast<size_t>(q_dim + kv_dim) * qkv->nb[0]);
                Q = ggml_cont(ctx0, Q);
                K = ggml_cont(ctx0, K);
                V = ggml_cont(ctx0, V);
            }
            else
            {
                Q = ggml_mul_mat(ctx0, q_w, cur);
                K = ggml_mul_mat(ctx0, k_w, cur);
                V = ggml_mul_mat(ctx0, v_w, cur);
            }
            Q = ggml_reshape_4d(ctx0, Q, head_dim, n_heads, seq_len, 1);
            K = ggml_reshape_4d(ctx0, K, head_dim, n_kv_heads, seq_len, 1);
            V = ggml_reshape_4d(ctx0, V, head_dim, n_kv_heads, seq_len, 1);
            Q = ggml_rope_ext_inplace(ctx0, Q, patch_positions,
                                      get_tensor("base_lm.rope_short_factors"),
                                      head_dim, GGML_ROPE_TYPE_NEOX,
                                      hparams_.lm_max_position_embeddings,
                                      hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                                      0.0f, 0.0f);
            K = ggml_rope_ext_inplace(ctx0, K, patch_positions,
                                      get_tensor("base_lm.rope_short_factors"),
                                      head_dim, GGML_ROPE_TYPE_NEOX,
                                      hparams_.lm_max_position_embeddings,
                                      hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                                      0.0f, 0.0f);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));

            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            struct ggml_tensor *attn = ggml_flash_attn_ext(
                ctx0, Q, K, V, patch_attn_mask, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(ctx0, attn, q_dim, seq_len);
            attn = ggml_mul_mat(ctx0, o_w, attn);
            cur = ggml_add(ctx0, residual, attn);

            residual = cur;
            cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
            cur = ggml_mul(ctx0, cur, to_f32(get_tensor(prefix + ".post_attention_layernorm.weight")));
            struct ggml_tensor *gate =
                ggml_mul_mat(ctx0, get_tensor(prefix + ".mlp.gate_proj.weight"), cur);
            struct ggml_tensor *up =
                ggml_mul_mat(ctx0, get_tensor(prefix + ".mlp.up_proj.weight"), cur);
            struct ggml_tensor *ffn = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
            ffn = ggml_mul_mat(ctx0, get_tensor(prefix + ".mlp.down_proj.weight"), ffn);
            cur = ggml_add(ctx0, residual, ffn);
        }

        struct ggml_tensor *tail = nullptr;
        for (int batch = 0; batch < cfg_batch; ++batch)
        {
            struct ggml_tensor *part = ggml_view_2d(
                ctx0, cur, d_model, patch_size, cur->nb[1],
                static_cast<size_t>(batch * single_seq_len + x_offset) *
                    cur->nb[1]);
            tail = tail ? ggml_concat(ctx0, tail, part, 1) : part;
        }
        tail = ggml_rms_norm(ctx0, tail, hparams_.lm_rms_norm_eps);
        tail = ggml_mul(ctx0, tail, to_f32(get_tensor("feat_decoder.estimator.decoder.norm.weight")));
        patch_velocity = ggml_mul_mat(ctx0, get_tensor("feat_decoder.estimator.out_proj.weight"), tail);
        patch_velocity = ggml_add(ctx0, patch_velocity, ggml_reshape_2d(ctx0,
            to_f32(get_tensor("feat_decoder.estimator.out_proj.bias")), feat_dim, 1));
        ggml_set_name(patch_velocity, "dit_velocity");
        ggml_build_forward_expand(patch_graph, patch_velocity);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, patch_graph))
        {
            reset_patch_graph();
            return std::vector<float>(patch_size * feat_dim, 0.0f);
        }

            std::vector<int32_t> pos_data(seq_len);
            for (int i = 0; i < seq_len; ++i)
            {
                pos_data[i] = i % single_seq_len;
            }
            ggml_backend_tensor_set(patch_positions, pos_data.data(), 0,
                                   pos_data.size() * sizeof(int32_t));
            if (patch_attn_mask)
            {
                std::vector<ggml_fp16_t> mask_data(
                    static_cast<size_t>(seq_len) * seq_len);
                for (int query = 0; query < seq_len; ++query)
                {
                    for (int key = 0; key < seq_len; ++key)
                    {
                        const bool same_batch =
                            query / single_seq_len == key / single_seq_len;
                        mask_data[static_cast<size_t>(query) * seq_len + key] =
                            ggml_fp32_to_fp16(same_batch ? 0.0f : -INFINITY);
                    }
                }
                ggml_backend_tensor_set(
                    patch_attn_mask, mask_data.data(), 0,
                    mask_data.size() * sizeof(ggml_fp16_t));
            }
        }

        std::vector<float> x_data;
        std::vector<float> cond_data;
        std::vector<float> mu_batch;
        std::vector<float> t_data;
        std::vector<float> dt_data;
        const std::vector<float> t_one = sinusoidal(t_scalar);
        const std::vector<float> dt_one = sinusoidal(dt_scalar);
        for (int batch = 0; batch < cfg_batch; ++batch)
        {
            x_data.insert(x_data.end(), x_patch.begin(), x_patch.end());
            cond_data.insert(cond_data.end(), cond_patch.begin(), cond_patch.end());
            if (batch == 0)
            {
                mu_batch.insert(mu_batch.end(), mu_data.begin(), mu_data.end());
            }
            else
            {
                mu_batch.insert(mu_batch.end(), mu_data.size(), 0.0f);
            }
            t_data.insert(t_data.end(), t_one.begin(), t_one.end());
            dt_data.insert(dt_data.end(), dt_one.begin(), dt_one.end());
        }

        ggml_backend_tensor_set(patch_x_in, x_data.data(), 0, x_data.size() * sizeof(float));
        ggml_backend_tensor_set(patch_cond_in, cond_data.data(), 0, cond_data.size() * sizeof(float));
        ggml_backend_tensor_set(patch_mu_in, mu_batch.data(), 0, mu_batch.size() * sizeof(float));
        ggml_backend_tensor_set(patch_t_sin, t_data.data(), 0, t_data.size() * sizeof(float));
        ggml_backend_tensor_set(patch_dt_sin, dt_data.data(), 0, dt_data.size() * sizeof(float));

        std::vector<float> velocity(
            static_cast<size_t>(cfg_batch) * patch_size * feat_dim);
        if (ggml_backend_sched_graph_compute(sched, patch_graph) == GGML_STATUS_SUCCESS)
        {
            ggml_backend_tensor_get(patch_velocity, velocity.data(), 0,
                                   velocity.size() * sizeof(float));
        }

        return velocity;
    };

    auto transpose_cf_to_tf = [&](const std::vector<float> &cf) {
        std::vector<float> tf(patch_size * feat_dim, 0.0f);
        for (int t = 0; t < patch_size; ++t)
        {
            for (int c = 0; c < feat_dim; ++c)
            {
                tf[static_cast<size_t>(t) * feat_dim + c] =
                    cf[static_cast<size_t>(c) * patch_size + t];
            }
        }
        return tf;
    };
    auto transpose_tf_to_cf = [&](const std::vector<float> &tf) {
        std::vector<float> cf(patch_size * feat_dim, 0.0f);
        for (int t = 0; t < patch_size; ++t)
        {
            for (int c = 0; c < feat_dim; ++c)
            {
                cf[static_cast<size_t>(c) * patch_size + t] =
                    tf[static_cast<size_t>(t) * feat_dim + c];
            }
        }
        return cf;
    };
    auto build_t_span = [&]() {
        const int steps = std::max(1, state.diffusion_steps);
        std::vector<float> t_span(steps + 1);
        for (int i = 0; i <= steps; ++i)
        {
            const float t = bf16_round(
                1.0f - static_cast<float>(i) / steps);
            const float angle = bf16_round(
                bf16_round(static_cast<float>(M_PI) * 0.5f) * t);
            const float cos_value = bf16_round(std::cos(angle));
            const float delta = bf16_round(
                bf16_round(cos_value - 1.0f) + t);
            t_span[i] = bf16_round(t + delta);
        }
        return t_span;
    };
    auto solve_cfm_patch = [&](const std::vector<float> &cond_patch,
                               std::mt19937 &rng) {
        const int state_size = patch_size * feat_dim;
        const int steps = std::max(1, state.diffusion_steps);
        const int zero_init_steps = std::max(1, static_cast<int>(steps * 0.04f));
        const std::vector<float> t_span = build_t_span();
        std::vector<float> x_cf(state_size, 0.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (float &v : x_cf)
        {
            v = bf16_round(normal(rng));
        }

        std::vector<float> zero_mu(state.dit_mu.size(), 0.0f);
        for (int step = 1; step <= steps; ++step)
        {
            const float t_cur = t_span[step - 1];
            const float dt_val = t_span[step - 1] - t_span[step];
            if (step <= zero_init_steps)
            {
                continue;
            }

            const std::vector<float> x_tf = transpose_cf_to_tf(x_cf);
            const auto dit_begin = std::chrono::steady_clock::now();
            std::vector<float> batched_velocity = run_patch(
                x_tf, cond_patch, state.dit_mu, t_cur, 0.0f);
            std::vector<float> velocity_tf(
                batched_velocity.begin(),
                batched_velocity.begin() + state_size);

            if (hparams_.dit_inference_cfg_rate > 1.0f)
            {
                std::vector<float> uncond_tf(
                    batched_velocity.begin() + state_size,
                    batched_velocity.begin() + 2 * state_size);
                double dot = 0.0;
                double norm_sq = 0.0;
                for (int i = 0; i < state_size; ++i)
                {
                    dot += static_cast<double>(velocity_tf[i]) * uncond_tf[i];
                    norm_sq += static_cast<double>(uncond_tf[i]) * uncond_tf[i];
                }
                const float st_star =
                    static_cast<float>(dot / (norm_sq + 1e-8));
                for (int i = 0; i < state_size; ++i)
                {
                    const float neg_scaled = uncond_tf[i] * st_star;
                    velocity_tf[i] = neg_scaled +
                        hparams_.dit_inference_cfg_rate * (velocity_tf[i] - neg_scaled);
                }
            }
            dit_forward_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - dit_begin).count();

            const std::vector<float> velocity_cf = transpose_tf_to_cf(velocity_tf);
            for (int i = 0; i < state_size; ++i)
            {
                x_cf[i] -= dt_val * velocity_cf[i];
            }
        }

        return transpose_cf_to_tf(x_cf);
    };

    auto run_locenc = [&](const std::vector<float> &patch) {
        const int enc_dim = hparams_.encoder_hidden_dim;
        const int seq = patch_size + 1;
        struct ggml_tensor *in_w = get_tensor("feat_encoder.in_proj.weight");
        struct ggml_tensor *in_b = get_tensor("feat_encoder.in_proj.bias");
        struct ggml_tensor *special = get_tensor("feat_encoder.special_token");
        struct ggml_tensor *norm_w = get_tensor("feat_encoder.encoder.norm.weight");
        struct ggml_tensor *enc_to_lm_w = get_tensor("enc_to_lm_proj.weight");
        struct ggml_tensor *enc_to_lm_b = get_tensor("enc_to_lm_proj.bias");
        if (!in_w || !in_b || !special || !norm_w || !enc_to_lm_w || !enc_to_lm_b)
        {
            RS_LOG_ERR("VoxCPM2: LocEnc projection tensors are missing");
            return std::vector<float>();
        }

        struct ggml_init_params params = {
            .mem_size   = 384 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context *ctx0 = ggml_init(params);
        if (!ctx0)
        {
            return std::vector<float>();
        }
        auto to_f32 = [&](struct ggml_tensor *tensor) {
            return tensor && tensor->type != GGML_TYPE_F32 ?
                ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
        };

        struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 4096, false);
        struct ggml_tensor *patch_in =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, feat_dim, patch_size);
        ggml_set_name(patch_in, "locenc_patch");
        ggml_set_input(patch_in);
        struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq);
        ggml_set_name(positions, "locenc_positions");
        ggml_set_input(positions);

        struct ggml_tensor *projected = ggml_mul_mat(ctx0, in_w, patch_in);
        projected = ggml_add(ctx0, projected,
                             ggml_reshape_2d(ctx0, to_f32(in_b), enc_dim, 1));
        struct ggml_tensor *cls =
            ggml_reshape_2d(ctx0, to_f32(special), enc_dim, 1);
        struct ggml_tensor *cur = ggml_concat(ctx0, cls, projected, 1);

        for (int layer = 0; layer < hparams_.encoder_num_layers; ++layer)
        {
            const std::string prefix =
                "feat_encoder.encoder.layers." + std::to_string(layer);
            struct ggml_tensor *ln_w = get_tensor(prefix + ".input_layernorm.weight");
            struct ggml_tensor *q_w = get_tensor(prefix + ".self_attn.q_proj.weight");
            struct ggml_tensor *k_w = get_tensor(prefix + ".self_attn.k_proj.weight");
            struct ggml_tensor *v_w = get_tensor(prefix + ".self_attn.v_proj.weight");
            struct ggml_tensor *o_w = get_tensor(prefix + ".self_attn.o_proj.weight");
            struct ggml_tensor *ffn_ln_w =
                get_tensor(prefix + ".post_attention_layernorm.weight");
            struct ggml_tensor *gate_w = get_tensor(prefix + ".mlp.gate_proj.weight");
            struct ggml_tensor *up_w = get_tensor(prefix + ".mlp.up_proj.weight");
            struct ggml_tensor *down_w = get_tensor(prefix + ".mlp.down_proj.weight");
            if (!ln_w || !q_w || !k_w || !v_w || !o_w ||
                !ffn_ln_w || !gate_w || !up_w || !down_w)
            {
                RS_LOG_ERR("VoxCPM2: LocEnc layer %d tensors are missing", layer);
                ggml_free(ctx0);
                return std::vector<float>();
            }

            struct ggml_tensor *residual = cur;
            cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
            cur = ggml_mul(ctx0, cur, to_f32(ln_w));

            const int q_dim = q_w->ne[1];
            const int kv_dim = k_w->ne[1];
            const int enc_head_dim = hparams_.encoder_kv_channels;
            const int n_heads = q_dim / enc_head_dim;
            const int n_kv_heads = kv_dim / enc_head_dim;
            struct ggml_tensor *Q = ggml_mul_mat(ctx0, q_w, cur);
            struct ggml_tensor *K = ggml_mul_mat(ctx0, k_w, cur);
            struct ggml_tensor *V = ggml_mul_mat(ctx0, v_w, cur);
            Q = ggml_reshape_4d(ctx0, Q, enc_head_dim, n_heads, seq, 1);
            K = ggml_reshape_4d(ctx0, K, enc_head_dim, n_kv_heads, seq, 1);
            V = ggml_reshape_4d(ctx0, V, enc_head_dim, n_kv_heads, seq, 1);
            Q = ggml_rope_ext_inplace(ctx0, Q, positions,
                                      get_tensor("base_lm.rope_short_factors"),
                                      enc_head_dim, GGML_ROPE_TYPE_NEOX,
                                      hparams_.lm_max_position_embeddings,
                                      hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                                      0.0f, 0.0f);
            K = ggml_rope_ext_inplace(ctx0, K, positions,
                                      get_tensor("base_lm.rope_short_factors"),
                                      enc_head_dim, GGML_ROPE_TYPE_NEOX,
                                      hparams_.lm_max_position_embeddings,
                                      hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                                      0.0f, 0.0f);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
            const float scale = 1.0f / std::sqrt(static_cast<float>(enc_head_dim));
            struct ggml_tensor *attn =
                ggml_flash_attn_ext(ctx0, Q, K, V, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(ctx0, attn, q_dim, seq);
            attn = ggml_mul_mat(ctx0, o_w, attn);
            cur = ggml_add(ctx0, residual, attn);

            residual = cur;
            cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
            cur = ggml_mul(ctx0, cur, to_f32(ffn_ln_w));
            struct ggml_tensor *gate = ggml_mul_mat(ctx0, gate_w, cur);
            struct ggml_tensor *up = ggml_mul_mat(ctx0, up_w, cur);
            struct ggml_tensor *ffn = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
            ffn = ggml_mul_mat(ctx0, down_w, ffn);
            cur = ggml_add(ctx0, residual, ffn);
        }

        cur = ggml_cont(ctx0, cur);
        struct ggml_tensor *cls_out = ggml_view_2d(ctx0, cur, enc_dim, 1,
                                                  cur->nb[1], 0);
        cls_out = ggml_rms_norm(ctx0, cls_out, hparams_.lm_rms_norm_eps);
        cls_out = ggml_mul(ctx0, cls_out, to_f32(norm_w));
        struct ggml_tensor *enc_lm = ggml_mul_mat(ctx0, enc_to_lm_w, cls_out);
        enc_lm = ggml_add(ctx0, enc_lm,
                          ggml_reshape_2d(ctx0, to_f32(enc_to_lm_b),
                                         hparams_.lm_hidden_size, 1));
        ggml_build_forward_expand(gf, enc_lm);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
        {
            ggml_free(ctx0);
            return std::vector<float>();
        }
        std::vector<int32_t> pos_data(seq);
        for (int i = 0; i < seq; ++i)
        {
            pos_data[i] = i;
        }
        ggml_backend_tensor_set(patch_in, patch.data(), 0,
                               patch.size() * sizeof(float));
        ggml_backend_tensor_set(positions, pos_data.data(), 0,
                               pos_data.size() * sizeof(int32_t));
        std::vector<float> result(hparams_.lm_hidden_size);
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        {
            result.clear();
        }
        else
        {
            ggml_backend_tensor_get(enc_lm, result.data(), 0,
                                   result.size() * sizeof(float));
        }
        ggml_free(ctx0);
        return result;
    };

    auto run_stop_score = [&](const std::vector<float> &hidden) {
        struct ggml_tensor *proj_w = get_tensor("stop_proj.weight");
        struct ggml_tensor *proj_b = get_tensor("stop_proj.bias");
        struct ggml_tensor *head_w = get_tensor("stop_head.weight");
        if (!proj_w || !proj_b || !head_w)
        {
            return 0.0f;
        }
        const int hidden_dim = hparams_.lm_hidden_size;
        struct ggml_init_params params = {
            .mem_size   = 16 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context *ctx0 = ggml_init(params);
        if (!ctx0)
        {
            return 0.0f;
        }
        auto to_f32 = [&](struct ggml_tensor *tensor) {
            return tensor && tensor->type != GGML_TYPE_F32 ?
                ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
        };
        struct ggml_cgraph *gf = ggml_new_graph(ctx0);
        struct ggml_tensor *input =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, 1);
        ggml_set_input(input);
        struct ggml_tensor *projected = ggml_mul_mat(ctx0, proj_w, input);
        projected = ggml_add(ctx0, projected,
                             ggml_reshape_2d(ctx0, to_f32(proj_b), hidden_dim, 1));
        projected = ggml_silu(ctx0, projected);
        struct ggml_tensor *logits = ggml_mul_mat(ctx0, head_w, projected);
        ggml_build_forward_expand(gf, logits);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
        {
            ggml_free(ctx0);
            return 0.0f;
        }
        ggml_backend_tensor_set(input, hidden.data(), 0,
                               hidden.size() * sizeof(float));
        float values[2] = {0.0f, 0.0f};
        if (ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS)
        {
            ggml_backend_tensor_get(logits, values, 0, sizeof(values));
        }
        ggml_free(ctx0);
        const float max_value = std::max(values[0], values[1]);
        const float e0 = std::exp(values[0] - max_value);
        const float e1 = std::exp(values[1] - max_value);
        return e1 / (e0 + e1);
    };

    auto run_ar_update_full = [&](const std::vector<float> &lm_inputs,
                                  const std::vector<float> &feat_inputs,
                                  int n_text_positions,
                                  std::vector<float> &next_hidden,
                                  std::vector<float> &next_mu) {
        const int hidden_dim = hparams_.lm_hidden_size;
        const int dit_dim = hparams_.dit_hidden_dim;
        const int n_pos = static_cast<int>(lm_inputs.size()) / hidden_dim;
        if (n_pos <= n_text_positions ||
            feat_inputs.size() != lm_inputs.size())
        {
            return false;
        }

        struct ggml_init_params params = {
            .mem_size   = 1024 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context *ctx0 = ggml_init(params);
        if (!ctx0)
        {
            return false;
        }
        auto to_f32 = [&](struct ggml_tensor *tensor) {
            return tensor && tensor->type != GGML_TYPE_F32 ?
                ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
        };
        auto add_bias = [&](struct ggml_tensor *tensor,
                            struct ggml_tensor *bias, int out_dim) {
            return bias ? ggml_add(ctx0, tensor,
                ggml_reshape_2d(ctx0, to_f32(bias), out_dim, 1)) : tensor;
        };

        struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 12288, false);
        struct ggml_tensor *input =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, n_pos);
        ggml_set_name(input, "ar_lm_inputs");
        ggml_set_input(input);
        struct ggml_tensor *feat =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, n_pos);
        ggml_set_name(feat, "ar_feat_inputs");
        ggml_set_input(feat);
        struct ggml_tensor *audio_mask =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, n_pos);
        ggml_set_name(audio_mask, "ar_audio_mask");
        ggml_set_input(audio_mask);
        struct ggml_tensor *text_mask =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, n_pos);
        ggml_set_name(text_mask, "ar_text_mask");
        ggml_set_input(text_mask);
        struct ggml_tensor *positions =
            ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
        ggml_set_name(positions, "ar_positions");
        ggml_set_input(positions);
        struct ggml_tensor *causal_mask =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_pos, n_pos);
        ggml_set_name(causal_mask, "ar_causal_mask");
        ggml_set_input(causal_mask);

        auto transformer = [&](struct ggml_tensor *cur,
                               const std::string &root, int layer_count,
                               bool use_rope) {
            for (int layer = 0; layer < layer_count; ++layer)
            {
                const std::string prefix =
                    root + ".layers." + std::to_string(layer);
                struct ggml_tensor *residual = cur;
                cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
                cur = ggml_mul(ctx0, cur,
                               to_f32(get_tensor(prefix + ".input_layernorm.weight")));
                struct ggml_tensor *q_w =
                    get_tensor(prefix + ".self_attn.q_proj.weight");
                struct ggml_tensor *k_w =
                    get_tensor(prefix + ".self_attn.k_proj.weight");
                struct ggml_tensor *v_w =
                    get_tensor(prefix + ".self_attn.v_proj.weight");
                struct ggml_tensor *o_w =
                    get_tensor(prefix + ".self_attn.o_proj.weight");
                const int q_dim = q_w->ne[1];
                const int kv_dim = k_w->ne[1];
                const int lm_head_dim = hparams_.lm_kv_channels;
                const int n_heads = q_dim / lm_head_dim;
                const int n_kv_heads = kv_dim / lm_head_dim;
                struct ggml_tensor *Q = ggml_mul_mat(ctx0, q_w, cur);
                struct ggml_tensor *K = ggml_mul_mat(ctx0, k_w, cur);
                struct ggml_tensor *V = ggml_mul_mat(ctx0, v_w, cur);
                Q = ggml_reshape_4d(ctx0, Q, lm_head_dim, n_heads, n_pos, 1);
                K = ggml_reshape_4d(ctx0, K, lm_head_dim, n_kv_heads, n_pos, 1);
                V = ggml_reshape_4d(ctx0, V, lm_head_dim, n_kv_heads, n_pos, 1);
                if (use_rope)
                {
                    Q = ggml_rope_ext_inplace(
                        ctx0, Q, positions,
                        get_tensor("base_lm.rope_short_factors"),
                        lm_head_dim, GGML_ROPE_TYPE_NEOX,
                        hparams_.lm_max_position_embeddings,
                                              hparams_.lm_rope_theta, 1.0f,
                                              0.0f, 1.0f, 0.0f, 0.0f);
                    K = ggml_rope_ext_inplace(
                        ctx0, K, positions,
                        get_tensor("base_lm.rope_short_factors"),
                        lm_head_dim, GGML_ROPE_TYPE_NEOX,
                        hparams_.lm_max_position_embeddings,
                                              hparams_.lm_rope_theta, 1.0f,
                                              0.0f, 1.0f, 0.0f, 0.0f);
                }
                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
                V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
                const float scale = 1.0f / std::sqrt(static_cast<float>(lm_head_dim));
                struct ggml_tensor *attn = ggml_flash_attn_ext(
                    ctx0, Q, K, V, causal_mask, scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
                attn = ggml_reshape_2d(ctx0, attn, q_dim, n_pos);
                attn = ggml_mul_mat(ctx0, o_w, attn);
                cur = ggml_add(ctx0, residual, attn);

                residual = cur;
                cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
                cur = ggml_mul(ctx0, cur, to_f32(get_tensor(
                    prefix + ".post_attention_layernorm.weight")));
                struct ggml_tensor *gate = ggml_mul_mat(ctx0,
                    get_tensor(prefix + ".mlp.gate_proj.weight"), cur);
                struct ggml_tensor *up = ggml_mul_mat(ctx0,
                    get_tensor(prefix + ".mlp.up_proj.weight"), cur);
                struct ggml_tensor *ffn =
                    ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
                ffn = ggml_mul_mat(ctx0,
                    get_tensor(prefix + ".mlp.down_proj.weight"), ffn);
                cur = ggml_add(ctx0, residual, ffn);
            }
            return cur;
        };

        struct ggml_tensor *tslm = transformer(
            input, "base_lm", hparams_.lm_num_hidden_layers, true);
        tslm = ggml_rms_norm(ctx0, tslm, hparams_.lm_rms_norm_eps);
        tslm = ggml_mul(ctx0, tslm, to_f32(get_tensor("base_lm.norm.weight")));

        struct ggml_tensor *fsq = ggml_mul_mat(
            ctx0, get_tensor("fsq_layer.in_proj.weight"), tslm);
        fsq = add_bias(fsq, get_tensor("fsq_layer.in_proj.bias"),
                       hparams_.scalar_quantization_latent_dim);
        fsq = ggml_tanh(ctx0, fsq);
        fsq = ggml_scale(ctx0, fsq,
                         static_cast<float>(hparams_.scalar_quantization_scale));
        fsq = ggml_round(ctx0, fsq);
        fsq = ggml_scale(ctx0, fsq,
                         1.0f / hparams_.scalar_quantization_scale);
        fsq = ggml_mul_mat(ctx0, get_tensor("fsq_layer.out_proj.weight"), fsq);
        fsq = add_bias(fsq, get_tensor("fsq_layer.out_proj.bias"), hidden_dim);

        struct ggml_tensor *selected = ggml_add(
            ctx0, ggml_mul(ctx0, fsq, audio_mask),
            ggml_mul(ctx0, tslm, text_mask));
        struct ggml_tensor *fusion_input = ggml_concat(ctx0, selected, feat, 0);
        struct ggml_tensor *ralm = ggml_mul_mat(
            ctx0, get_tensor("fusion_concat_proj.weight"), fusion_input);
        ralm = add_bias(ralm, get_tensor("fusion_concat_proj.bias"), hidden_dim);
        ralm = transformer(ralm, "residual_lm", 8, false);
        ralm = ggml_cont(ctx0, ralm);
        struct ggml_tensor *ralm_last = ggml_view_2d(
            ctx0, ralm, hidden_dim, 1, ralm->nb[1],
            static_cast<size_t>(n_pos - 1) * ralm->nb[1]);
        ralm_last = ggml_rms_norm(ctx0, ralm_last, hparams_.lm_rms_norm_eps);
        ralm_last = ggml_mul(ctx0, ralm_last,
                             to_f32(get_tensor("residual_lm.norm.weight")));

        fsq = ggml_cont(ctx0, fsq);
        struct ggml_tensor *fsq_last = ggml_view_2d(
            ctx0, fsq, hidden_dim, 1, fsq->nb[1],
            static_cast<size_t>(n_pos - 1) * fsq->nb[1]);
        struct ggml_tensor *lm_cond = ggml_mul_mat(
            ctx0, get_tensor("lm_to_dit_proj.weight"), fsq_last);
        lm_cond = add_bias(lm_cond, get_tensor("lm_to_dit_proj.bias"), dit_dim);
        struct ggml_tensor *res_cond = ggml_mul_mat(
            ctx0, get_tensor("res_to_dit_proj.weight"), ralm_last);
        res_cond = add_bias(res_cond, get_tensor("res_to_dit_proj.bias"), dit_dim);
        ggml_build_forward_expand(gf, fsq_last);
        ggml_build_forward_expand(gf, lm_cond);
        ggml_build_forward_expand(gf, res_cond);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
        {
            ggml_free(ctx0);
            return false;
        }
        std::vector<float> audio_mask_data(lm_inputs.size(), 0.0f);
        std::vector<float> text_mask_data(lm_inputs.size(), 1.0f);
        for (int pos = n_text_positions; pos < n_pos; ++pos)
        {
            const size_t offset = static_cast<size_t>(pos) * hidden_dim;
            std::fill(audio_mask_data.begin() + offset,
                      audio_mask_data.begin() + offset + hidden_dim, 1.0f);
            std::fill(text_mask_data.begin() + offset,
                      text_mask_data.begin() + offset + hidden_dim, 0.0f);
        }
        std::vector<int32_t> pos_data(n_pos);
        for (int i = 0; i < n_pos; ++i)
        {
            pos_data[i] = i;
        }
        std::vector<ggml_fp16_t> mask_data(static_cast<size_t>(n_pos) * n_pos);
        for (int q = 0; q < n_pos; ++q)
        {
            for (int k = 0; k < n_pos; ++k)
            {
                mask_data[static_cast<size_t>(q) * n_pos + k] =
                    ggml_fp32_to_fp16(k <= q ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(input, lm_inputs.data(), 0,
                               lm_inputs.size() * sizeof(float));
        ggml_backend_tensor_set(feat, feat_inputs.data(), 0,
                               feat_inputs.size() * sizeof(float));
        ggml_backend_tensor_set(audio_mask, audio_mask_data.data(), 0,
                               audio_mask_data.size() * sizeof(float));
        ggml_backend_tensor_set(text_mask, text_mask_data.data(), 0,
                               text_mask_data.size() * sizeof(float));
        ggml_backend_tensor_set(positions, pos_data.data(), 0,
                               pos_data.size() * sizeof(int32_t));
        ggml_backend_tensor_set(causal_mask, mask_data.data(), 0,
                               mask_data.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        {
            ggml_free(ctx0);
            return false;
        }
        next_hidden.resize(hidden_dim);
        next_mu.resize(static_cast<size_t>(2) * dit_dim);
        ggml_backend_tensor_get(fsq_last, next_hidden.data(), 0,
                               next_hidden.size() * sizeof(float));
        ggml_backend_tensor_get(lm_cond, next_mu.data(), 0,
                               dit_dim * sizeof(float));
        ggml_backend_tensor_get(res_cond, next_mu.data() + dit_dim, 0,
                               dit_dim * sizeof(float));
        ggml_free(ctx0);
        return true;
    };

    auto run_ar_update_incremental = [&](const std::vector<float> &audio_embed,
                                         std::vector<float> &next_hidden,
                                         std::vector<float> &next_mu) {
        const int hidden_dim = hparams_.lm_hidden_size;
        const int dit_dim = hparams_.dit_hidden_dim;
        const int n_heads = hparams_.lm_num_attention_heads;
        const int n_kv_heads = hparams_.lm_num_key_value_heads;
        const int head_dim = hidden_dim / n_heads;
        const int n_base_layers = hparams_.lm_num_hidden_layers;
        const int n_ralm_layers = 8;
        const int n_past = state.ar_cache_length;
        const size_t base_cache_size = static_cast<size_t>(n_base_layers) *
            n_kv_heads * n_past * head_dim;
        const size_t ralm_cache_size = static_cast<size_t>(n_ralm_layers) *
            n_kv_heads * n_past * head_dim;
        if (audio_embed.size() != static_cast<size_t>(hidden_dim) ||
            n_past <= 0 || state.base_lm_k_cache.size() != base_cache_size ||
            state.base_lm_v_cache.size() != base_cache_size ||
            state.residual_lm_k_cache.size() != ralm_cache_size ||
            state.residual_lm_v_cache.size() != ralm_cache_size)
        {
            return false;
        }

        struct ggml_init_params params = {
            .mem_size   = 768 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context *ctx0 = ggml_init(params);
        if (!ctx0)
        {
            return false;
        }
        auto to_f32 = [&](struct ggml_tensor *tensor) {
            return tensor && tensor->type != GGML_TYPE_F32 ?
                ggml_cast(ctx0, tensor, GGML_TYPE_F32) : tensor;
        };
        auto add_bias = [&](struct ggml_tensor *tensor,
                            struct ggml_tensor *bias, int out_dim) {
            return bias ? ggml_add(ctx0, tensor,
                ggml_reshape_2d(ctx0, to_f32(bias), out_dim, 1)) : tensor;
        };

        struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 12288, false);
        struct ggml_tensor *input =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_dim, 1);
        ggml_set_name(input, "ar_incremental_input");
        ggml_set_input(input);
        struct ggml_tensor *position =
            ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(position, "ar_incremental_position");
        ggml_set_input(position);
        struct ggml_tensor *base_k_cache = ggml_new_tensor_4d(
            ctx0, GGML_TYPE_F32, head_dim, n_past, n_kv_heads, n_base_layers);
        struct ggml_tensor *base_v_cache = ggml_new_tensor_4d(
            ctx0, GGML_TYPE_F32, head_dim, n_past, n_kv_heads, n_base_layers);
        struct ggml_tensor *ralm_k_cache = ggml_new_tensor_4d(
            ctx0, GGML_TYPE_F32, head_dim, n_past, n_kv_heads, n_ralm_layers);
        struct ggml_tensor *ralm_v_cache = ggml_new_tensor_4d(
            ctx0, GGML_TYPE_F32, head_dim, n_past, n_kv_heads, n_ralm_layers);
        ggml_set_name(base_k_cache, "ar_base_k_cache");
        ggml_set_name(base_v_cache, "ar_base_v_cache");
        ggml_set_name(ralm_k_cache, "ar_ralm_k_cache");
        ggml_set_name(ralm_v_cache, "ar_ralm_v_cache");
        ggml_set_input(base_k_cache);
        ggml_set_input(base_v_cache);
        ggml_set_input(ralm_k_cache);
        ggml_set_input(ralm_v_cache);

        std::vector<struct ggml_tensor *> base_k_new;
        std::vector<struct ggml_tensor *> base_v_new;
        std::vector<struct ggml_tensor *> ralm_k_new;
        std::vector<struct ggml_tensor *> ralm_v_new;

        auto transformer_step = [&](struct ggml_tensor *cur,
                                    const std::string &model_prefix,
                                    int n_layers, bool with_rope,
                                    struct ggml_tensor *k_cache,
                                    struct ggml_tensor *v_cache,
                                    std::vector<struct ggml_tensor *> &k_new,
                                    std::vector<struct ggml_tensor *> &v_new) {
            for (int layer = 0; layer < n_layers; ++layer)
            {
                const std::string prefix = model_prefix + ".layers." +
                    std::to_string(layer);
                struct ggml_tensor *residual = cur;
                cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
                cur = ggml_mul(ctx0, cur, to_f32(get_tensor(
                    prefix + ".input_layernorm.weight")));

                struct ggml_tensor *q_w =
                    get_tensor(prefix + ".self_attn.q_proj.weight");
                struct ggml_tensor *k_w =
                    get_tensor(prefix + ".self_attn.k_proj.weight");
                struct ggml_tensor *v_w =
                    get_tensor(prefix + ".self_attn.v_proj.weight");
                struct ggml_tensor *o_w =
                    get_tensor(prefix + ".self_attn.o_proj.weight");
                struct ggml_tensor *Q = ggml_mul_mat(ctx0, q_w, cur);
                struct ggml_tensor *K = ggml_mul_mat(ctx0, k_w, cur);
                struct ggml_tensor *V = ggml_mul_mat(ctx0, v_w, cur);
                Q = ggml_reshape_4d(ctx0, Q, head_dim, n_heads, 1, 1);
                K = ggml_reshape_4d(ctx0, K, head_dim, n_kv_heads, 1, 1);
                V = ggml_reshape_4d(ctx0, V, head_dim, n_kv_heads, 1, 1);
                if (with_rope)
                {
                    Q = ggml_rope_ext_inplace(
                        ctx0, Q, position,
                        get_tensor("base_lm.rope_short_factors"),
                        head_dim, GGML_ROPE_TYPE_NEOX,
                        hparams_.lm_max_position_embeddings,
                        hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                        0.0f, 0.0f);
                    K = ggml_rope_ext_inplace(
                        ctx0, K, position,
                        get_tensor("base_lm.rope_short_factors"),
                        head_dim, GGML_ROPE_TYPE_NEOX,
                        hparams_.lm_max_position_embeddings,
                        hparams_.lm_rope_theta, 1.0f, 0.0f, 1.0f,
                        0.0f, 0.0f);
                }
                Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
                K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
                V = ggml_cont(ctx0, ggml_permute(ctx0, V, 0, 2, 1, 3));
                k_new.push_back(K);
                v_new.push_back(V);

                struct ggml_tensor *K_past = ggml_view_3d(
                    ctx0, k_cache, head_dim, n_past, n_kv_heads,
                    k_cache->nb[1], k_cache->nb[2],
                    static_cast<size_t>(layer) * k_cache->nb[3]);
                struct ggml_tensor *V_past = ggml_view_3d(
                    ctx0, v_cache, head_dim, n_past, n_kv_heads,
                    v_cache->nb[1], v_cache->nb[2],
                    static_cast<size_t>(layer) * v_cache->nb[3]);
                struct ggml_tensor *K_all = ggml_concat(ctx0, K_past, K, 1);
                struct ggml_tensor *V_all = ggml_concat(ctx0, V_past, V, 1);
                const float scale = 1.0f /
                    std::sqrt(static_cast<float>(head_dim));
                struct ggml_tensor *attn = ggml_flash_attn_ext(
                    ctx0, Q, K_all, V_all, nullptr, scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
                attn = ggml_reshape_2d(ctx0, attn, hidden_dim, 1);
                attn = ggml_mul_mat(ctx0, o_w, attn);
                cur = ggml_add(ctx0, residual, attn);

                residual = cur;
                cur = ggml_rms_norm(ctx0, cur, hparams_.lm_rms_norm_eps);
                cur = ggml_mul(ctx0, cur, to_f32(get_tensor(
                    prefix + ".post_attention_layernorm.weight")));
                struct ggml_tensor *gate = ggml_mul_mat(
                    ctx0, get_tensor(prefix + ".mlp.gate_proj.weight"), cur);
                struct ggml_tensor *up = ggml_mul_mat(
                    ctx0, get_tensor(prefix + ".mlp.up_proj.weight"), cur);
                struct ggml_tensor *ffn =
                    ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
                ffn = ggml_mul_mat(
                    ctx0, get_tensor(prefix + ".mlp.down_proj.weight"), ffn);
                cur = ggml_add(ctx0, residual, ffn);
            }
            return cur;
        };

        struct ggml_tensor *tslm = transformer_step(
            input, "base_lm", n_base_layers, true,
            base_k_cache, base_v_cache, base_k_new, base_v_new);
        tslm = ggml_rms_norm(ctx0, tslm, hparams_.lm_rms_norm_eps);
        tslm = ggml_mul(
            ctx0, tslm, to_f32(get_tensor("base_lm.norm.weight")));

        struct ggml_tensor *fsq = ggml_mul_mat(
            ctx0, get_tensor("fsq_layer.in_proj.weight"), tslm);
        fsq = add_bias(fsq, get_tensor("fsq_layer.in_proj.bias"),
                       hparams_.scalar_quantization_latent_dim);
        fsq = ggml_tanh(ctx0, fsq);
        fsq = ggml_scale(ctx0, fsq,
                         static_cast<float>(hparams_.scalar_quantization_scale));
        fsq = ggml_round(ctx0, fsq);
        fsq = ggml_scale(ctx0, fsq,
                         1.0f / hparams_.scalar_quantization_scale);
        fsq = ggml_mul_mat(ctx0, get_tensor("fsq_layer.out_proj.weight"), fsq);
        fsq = add_bias(fsq, get_tensor("fsq_layer.out_proj.bias"), hidden_dim);

        struct ggml_tensor *lm_cond = ggml_mul_mat(
            ctx0, get_tensor("lm_to_dit_proj.weight"), fsq);
        lm_cond = add_bias(
            lm_cond, get_tensor("lm_to_dit_proj.bias"), dit_dim);
        struct ggml_tensor *fusion_input = ggml_concat(ctx0, fsq, input, 0);
        struct ggml_tensor *ralm = ggml_mul_mat(
            ctx0, get_tensor("fusion_concat_proj.weight"), fusion_input);
        ralm = add_bias(
            ralm, get_tensor("fusion_concat_proj.bias"), hidden_dim);
        ralm = transformer_step(
            ralm, "residual_lm", n_ralm_layers, false,
            ralm_k_cache, ralm_v_cache, ralm_k_new, ralm_v_new);
        ralm = ggml_rms_norm(ctx0, ralm, hparams_.lm_rms_norm_eps);
        ralm = ggml_mul(
            ctx0, ralm, to_f32(get_tensor("residual_lm.norm.weight")));
        struct ggml_tensor *res_cond = ggml_mul_mat(
            ctx0, get_tensor("res_to_dit_proj.weight"), ralm);
        res_cond = add_bias(
            res_cond, get_tensor("res_to_dit_proj.bias"), dit_dim);

        auto concat_new_cache = [&](const std::vector<struct ggml_tensor *> &items) {
            struct ggml_tensor *combined = nullptr;
            for (struct ggml_tensor *item : items)
            {
                struct ggml_tensor *flat = ggml_reshape_1d(
                    ctx0, item, ggml_nelements(item));
                combined = combined ? ggml_concat(ctx0, combined, flat, 0) : flat;
            }
            return combined;
        };
        struct ggml_tensor *base_k_out = concat_new_cache(base_k_new);
        struct ggml_tensor *base_v_out = concat_new_cache(base_v_new);
        struct ggml_tensor *ralm_k_out = concat_new_cache(ralm_k_new);
        struct ggml_tensor *ralm_v_out = concat_new_cache(ralm_v_new);
        ggml_build_forward_expand(gf, fsq);
        ggml_build_forward_expand(gf, lm_cond);
        ggml_build_forward_expand(gf, res_cond);
        ggml_build_forward_expand(gf, base_k_out);
        ggml_build_forward_expand(gf, base_v_out);
        ggml_build_forward_expand(gf, ralm_k_out);
        ggml_build_forward_expand(gf, ralm_v_out);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
        {
            ggml_free(ctx0);
            return false;
        }
        const int32_t position_value = n_past;
        ggml_backend_tensor_set(
            input, audio_embed.data(), 0, audio_embed.size() * sizeof(float));
        ggml_backend_tensor_set(position, &position_value, 0, sizeof(position_value));
        ggml_backend_tensor_set(
            base_k_cache, state.base_lm_k_cache.data(), 0,
            state.base_lm_k_cache.size() * sizeof(float));
        ggml_backend_tensor_set(
            base_v_cache, state.base_lm_v_cache.data(), 0,
            state.base_lm_v_cache.size() * sizeof(float));
        ggml_backend_tensor_set(
            ralm_k_cache, state.residual_lm_k_cache.data(), 0,
            state.residual_lm_k_cache.size() * sizeof(float));
        ggml_backend_tensor_set(
            ralm_v_cache, state.residual_lm_v_cache.data(), 0,
            state.residual_lm_v_cache.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        {
            ggml_free(ctx0);
            return false;
        }

        next_hidden.resize(hidden_dim);
        next_mu.resize(static_cast<size_t>(2) * dit_dim);
        ggml_backend_tensor_get(
            fsq, next_hidden.data(), 0, next_hidden.size() * sizeof(float));
        ggml_backend_tensor_get(
            lm_cond, next_mu.data(), 0, dit_dim * sizeof(float));
        ggml_backend_tensor_get(
            res_cond, next_mu.data() + dit_dim, 0, dit_dim * sizeof(float));
        const size_t base_new_size =
            static_cast<size_t>(n_base_layers) * n_kv_heads * head_dim;
        const size_t ralm_new_size =
            static_cast<size_t>(n_ralm_layers) * n_kv_heads * head_dim;
        std::vector<float> base_k_values(base_new_size);
        std::vector<float> base_v_values(base_new_size);
        std::vector<float> ralm_k_values(ralm_new_size);
        std::vector<float> ralm_v_values(ralm_new_size);
        ggml_backend_tensor_get(base_k_out, base_k_values.data(), 0,
                                base_new_size * sizeof(float));
        ggml_backend_tensor_get(base_v_out, base_v_values.data(), 0,
                                base_new_size * sizeof(float));
        ggml_backend_tensor_get(ralm_k_out, ralm_k_values.data(), 0,
                                ralm_new_size * sizeof(float));
        ggml_backend_tensor_get(ralm_v_out, ralm_v_values.data(), 0,
                                ralm_new_size * sizeof(float));
        ggml_free(ctx0);

        auto append_cache = [&](std::vector<float> &cache,
                                const std::vector<float> &new_values,
                                int n_layers) {
            std::vector<float> expanded(
                static_cast<size_t>(n_layers) * n_kv_heads *
                (n_past + 1) * head_dim);
            for (int layer = 0; layer < n_layers; ++layer)
            {
                for (int head = 0; head < n_kv_heads; ++head)
                {
                    const size_t old_offset =
                        (static_cast<size_t>(layer) * n_kv_heads + head) *
                        n_past * head_dim;
                    const size_t new_offset =
                        (static_cast<size_t>(layer) * n_kv_heads + head) *
                        head_dim;
                    const size_t dst_offset =
                        (static_cast<size_t>(layer) * n_kv_heads + head) *
                        (n_past + 1) * head_dim;
                    std::copy_n(cache.data() + old_offset,
                                static_cast<size_t>(n_past) * head_dim,
                                expanded.data() + dst_offset);
                    std::copy_n(new_values.data() + new_offset, head_dim,
                                expanded.data() + dst_offset +
                                    static_cast<size_t>(n_past) * head_dim);
                }
            }
            cache = std::move(expanded);
        };
        append_cache(state.base_lm_k_cache, base_k_values, n_base_layers);
        append_cache(state.base_lm_v_cache, base_v_values, n_base_layers);
        append_cache(state.residual_lm_k_cache, ralm_k_values, n_ralm_layers);
        append_cache(state.residual_lm_v_cache, ralm_v_values, n_ralm_layers);
        state.ar_cache_length = n_past + 1;
        return true;
    };

    const int n_text_positions = state.text_seq_len;
    const int max_patches = std::max(20, n_text_positions * 6 + 10);
    const int min_patches = std::max(4, provisional_patches);
    std::vector<float> lm_inputs = state.text_embeddings;
    std::vector<float> feat_inputs(lm_inputs.size(), 0.0f);
    std::vector<float> lm_hidden = state.base_lm_hidden;
    std::mt19937 rng(static_cast<uint32_t>(state.generation_seed));
    std::vector<float> cond_patch(patch_size * feat_dim, 0.0f);
    state.denoised_latents.clear();
    state.denoised_latents.reserve(
        static_cast<size_t>(max_patches) * patch_size * feat_dim);
    const char *full_ar_env = std::getenv("RS_VOXCPM2_FULL_AR");
    bool use_incremental_ar =
        !(full_ar_env && std::strcmp(full_ar_env, "1") == 0);
    const char *validate_ar_env = std::getenv("RS_VOXCPM2_VALIDATE_AR");
    const bool validate_ar =
        validate_ar_env && std::strcmp(validate_ar_env, "1") == 0;

    int generated_patches = 0;
    int last_stream_patch = 0;
    const size_t stream_holdback_samples = static_cast<size_t>(
        hparams_.vae_out_sample_rate * stream_holdback_ms / 1000);
    auto trim_stream_tail = [&](size_t begin, size_t end) {
        if (end <= begin || begin >= state.audio_output.size())
        {
            return end;
        }
        end = std::min(end, state.audio_output.size());
        const float threshold = 0.012f;
        size_t last_active = begin;
        bool found = false;
        for (size_t i = begin; i < end; ++i)
        {
            if (std::fabs(state.audio_output[i]) > threshold)
            {
                last_active = i;
                found = true;
            }
        }
        if (!found)
        {
            return begin;
        }
        const size_t keep_samples = static_cast<size_t>(
            hparams_.vae_out_sample_rate * 60 / 1000);
        return std::min(end, last_active + 1 + keep_samples);
    };
    auto emit_prefix = [&](bool final_chunk) {
        if (!emit || state.audio_output.size() <= streamed_samples)
        {
            return;
        }
        size_t end = state.audio_output.size();
        if (final_chunk)
        {
            end = trim_stream_tail(streamed_samples, end);
        }
        else if (end > streamed_samples + stream_holdback_samples)
        {
            end -= stream_holdback_samples;
        }
        if (end > streamed_samples)
        {
            emit(state.audio_output.data() + streamed_samples,
                 end - streamed_samples);
            streamed_samples = end;
        }
    };
    for (int p = 0; p < max_patches; ++p)
    {
        const auto cfm_begin = std::chrono::steady_clock::now();
        std::vector<float> out_patch = solve_cfm_patch(cond_patch, rng);
        cfm_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cfm_begin).count();
        reset_patch_graph();
        const auto locenc_begin = std::chrono::steady_clock::now();
        std::vector<float> enc_lm = run_locenc(out_patch);
        locenc_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - locenc_begin).count();
        if (enc_lm.size() != static_cast<size_t>(hparams_.lm_hidden_size))
        {
            RS_LOG_ERR("VoxCPM2: LocEnc failed at patch %d", p);
            return false;
        }
        state.denoised_latents.insert(state.denoised_latents.end(),
                                      out_patch.begin(), out_patch.end());
        cond_patch = out_patch;
        ++generated_patches;

        if (emit && generated_patches >= stream_first_patches &&
            (last_stream_patch == 0 ||
             generated_patches - last_stream_patch >= stream_chunk_patches))
        {
            state.audio_token_len = generated_patches * patch_size;
            const auto vae_begin = std::chrono::steady_clock::now();
            if (!RunAudioVAE(state, sched))
            {
                RS_LOG_ERR("VoxCPM2: streaming AudioVAE prefix failed");
                return false;
            }
            const size_t before = streamed_samples;
            emit_prefix(false);
            last_stream_patch = generated_patches;
            RS_LOG_INFO(
                "VoxCPM2: stream chunk %zu samples after %d patches "
                "(VAE %.1f ms)",
                streamed_samples - before, generated_patches,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - vae_begin).count());
        }

        const float stop = run_stop_score(lm_hidden);
        if (p < 3 || stop > 0.5f)
        {
            RS_LOG_INFO("VoxCPM2: AR patch %d stop=%.4f", p + 1, stop);
        }
        if (stop > 0.5f && generated_patches >= min_patches)
        {
            break;
        }

        lm_inputs.insert(lm_inputs.end(), enc_lm.begin(), enc_lm.end());
        feat_inputs.insert(feat_inputs.end(), enc_lm.begin(), enc_lm.end());
        std::vector<float> next_mu;
        const auto ar_begin = std::chrono::steady_clock::now();
        bool ar_ok = false;
        if (use_incremental_ar)
        {
            std::vector<float> incremental_hidden;
            std::vector<float> incremental_mu;
            ar_ok = run_ar_update_incremental(
                enc_lm, incremental_hidden, incremental_mu);
            if (ar_ok && validate_ar)
            {
                std::vector<float> full_hidden;
                std::vector<float> full_mu;
                if (run_ar_update_full(lm_inputs, feat_inputs,
                                       n_text_positions, full_hidden, full_mu))
                {
                    float hidden_diff = 0.0f;
                    float mu_diff = 0.0f;
                    for (size_t i = 0; i < full_hidden.size(); ++i)
                    {
                        hidden_diff = std::max(
                            hidden_diff,
                            std::fabs(full_hidden[i] - incremental_hidden[i]));
                    }
                    for (size_t i = 0; i < full_mu.size(); ++i)
                    {
                        mu_diff = std::max(
                            mu_diff,
                            std::fabs(full_mu[i] - incremental_mu[i]));
                    }
                    RS_LOG_INFO(
                        "VoxCPM2: AR cache validation patch=%d hidden_diff=%.6g mu_diff=%.6g",
                        p + 1, hidden_diff, mu_diff);
                }
            }
            if (ar_ok)
            {
                lm_hidden = std::move(incremental_hidden);
                next_mu = std::move(incremental_mu);
            }
            else
            {
                RS_LOG_WARN("VoxCPM2: incremental AR unavailable; using full AR");
                use_incremental_ar = false;
            }
        }
        if (!use_incremental_ar)
        {
            ar_ok = run_ar_update_full(lm_inputs, feat_inputs,
                                       n_text_positions, lm_hidden, next_mu);
        }
        if (!ar_ok)
        {
            RS_LOG_ERR("VoxCPM2: autoregressive conditioning failed at patch %d", p);
            return false;
        }
        ar_update_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ar_begin).count();
        state.dit_mu = std::move(next_mu);
    }

    state.base_lm_hidden = lm_hidden;
    state.audio_token_len = generated_patches * patch_size;
    if (generated_patches >= max_patches)
    {
        RS_LOG_WARN("VoxCPM2: AR loop reached safety ceiling (%d patches; provisional=%d)",
                    max_patches, provisional_patches);
    }
    RS_LOG_INFO("VoxCPM2: AR CFM output: [%d, %d] latents, %d patches (%d steps)",
                generated_patches * patch_size, feat_dim, generated_patches,
                std::max(1, state.diffusion_steps));
    RS_LOG_INFO("VoxCPM2: DiT timing cfm=%.1f ms (forward=%.1f ms), "
                "locenc=%.1f ms, ar_update=%.1f ms",
                cfm_ms, dit_forward_ms, locenc_ms, ar_update_ms);
    if (emit)
    {
        if (!RunAudioVAE(state, sched))
        {
            RS_LOG_ERR("VoxCPM2: streaming final AudioVAE failed");
            return false;
        }
        emit_prefix(true);
        state.audio_generated = true;
    }
    return true;
}

namespace
{
std::vector<float> tensor_to_f32(struct ggml_tensor *tensor)
{
    if (!tensor)
    {
        return {};
    }

    const int64_t n = ggml_nelements(tensor);
    std::vector<float> out(n);
    if (tensor->type == GGML_TYPE_F32)
    {
        ggml_backend_tensor_get(tensor, out.data(), 0, n * sizeof(float));
    }
    else if (tensor->type == GGML_TYPE_F16)
    {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(tensor, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
    }
    else if (tensor->type == GGML_TYPE_I32)
    {
        std::vector<int32_t> tmp(n);
        ggml_backend_tensor_get(tensor, tmp.data(), 0, n * sizeof(int32_t));
        for (int64_t i = 0; i < n; ++i)
        {
            out[i] = static_cast<float>(tmp[i]);
        }
    }
    return out;
}

std::vector<float> reconstruct_weight_norm_conv(const std::vector<float> &g,
                                                const std::vector<float> &v,
                                                int out_ch, int in_ch, int ksize)
{
    std::vector<float> w(static_cast<size_t>(out_ch) * in_ch * ksize, 0.0f);
    for (int oc = 0; oc < out_ch; ++oc)
    {
        double sum_sq = 0.0;
        const size_t base = static_cast<size_t>(oc) * in_ch * ksize;
        for (int ic = 0; ic < in_ch; ++ic)
        {
            for (int k = 0; k < ksize; ++k)
            {
                const float val = v[base + static_cast<size_t>(ic) * ksize + k];
                sum_sq += static_cast<double>(val) * val;
            }
        }
        const float scale = g.empty() ? 1.0f :
            g[std::min<int>(oc, static_cast<int>(g.size()) - 1)] /
            std::sqrt(static_cast<float>(sum_sq) + 1e-12f);
        for (int ic = 0; ic < in_ch; ++ic)
        {
            for (int k = 0; k < ksize; ++k)
            {
                w[base + static_cast<size_t>(ic) * ksize + k] =
                    v[base + static_cast<size_t>(ic) * ksize + k] * scale;
            }
        }
    }
    return w;
}

std::vector<float> reconstruct_weight_norm_deconv(const std::vector<float> &g,
                                                  const std::vector<float> &v,
                                                  int in_ch, int out_ch, int ksize)
{
    std::vector<float> w(static_cast<size_t>(in_ch) * out_ch * ksize, 0.0f);
    for (int ic = 0; ic < in_ch; ++ic)
    {
        double sum_sq = 0.0;
        const size_t base = static_cast<size_t>(ic) * out_ch * ksize;
        for (int oc = 0; oc < out_ch; ++oc)
        {
            for (int k = 0; k < ksize; ++k)
            {
                const float val = v[base + static_cast<size_t>(oc) * ksize + k];
                sum_sq += static_cast<double>(val) * val;
            }
        }
        const float scale = g.empty() ? 1.0f :
            g[std::min<int>(ic, static_cast<int>(g.size()) - 1)] /
            std::sqrt(static_cast<float>(sum_sq) + 1e-12f);
        for (int oc = 0; oc < out_ch; ++oc)
        {
            for (int k = 0; k < ksize; ++k)
            {
                w[base + static_cast<size_t>(oc) * ksize + k] =
                    v[base + static_cast<size_t>(oc) * ksize + k] * scale;
            }
        }
    }
    return w;
}

void snake1d_inplace(std::vector<float> &x, const std::vector<float> &alpha,
                     int channels, int frames)
{
    if (alpha.empty())
    {
        return;
    }
    for (int c = 0; c < channels; ++c)
    {
        const float a = std::max(std::fabs(alpha[std::min<int>(c, alpha.size() - 1)]), 1e-6f);
        float *row = x.data() + static_cast<size_t>(c) * frames;
        for (int t = 0; t < frames; ++t)
        {
            const float s = std::sin(a * row[t]);
            row[t] += (s * s) / a;
        }
    }
}

std::vector<float> causal_conv1d_cpu(const std::vector<float> &x,
                                     const std::vector<float> &w,
                                     const std::vector<float> &bias,
                                     int in_ch, int out_ch, int ksize,
                                     int frames, int dilation, int groups)
{
    std::vector<float> y(static_cast<size_t>(out_ch) * frames, 0.0f);
    const int in_per_group = in_ch / groups;
    const int out_per_group = out_ch / groups;
    for (int oc = 0; oc < out_ch; ++oc)
    {
        const int group = oc / out_per_group;
        const int ic_begin = group * in_per_group;
        const int ic_end = ic_begin + in_per_group;
        for (int t = 0; t < frames; ++t)
        {
            float acc = bias.empty() ? 0.0f : bias[std::min<int>(oc, bias.size() - 1)];
            for (int ic = ic_begin; ic < ic_end; ++ic)
            {
                const int local_ic = ic - ic_begin;
                for (int k = 0; k < ksize; ++k)
                {
                    const int it = t - (ksize - 1 - k) * dilation;
                    if (it < 0 || it >= frames)
                    {
                        continue;
                    }
                    const size_t widx = static_cast<size_t>(oc) * in_per_group * ksize +
                                        static_cast<size_t>(local_ic) * ksize + k;
                    acc += x[static_cast<size_t>(ic) * frames + it] * w[widx];
                }
            }
            y[static_cast<size_t>(oc) * frames + t] = acc;
        }
    }
    return y;
}

std::vector<float> causal_deconv1d_cpu(const std::vector<float> &x,
                                       const std::vector<float> &w,
                                       const std::vector<float> &bias,
                                       int in_ch, int out_ch, int ksize,
                                       int frames, int stride)
{
    const int out_frames = frames * stride;
    std::vector<float> y(static_cast<size_t>(out_ch) * out_frames, 0.0f);
    for (int ic = 0; ic < in_ch; ++ic)
    {
        for (int t = 0; t < frames; ++t)
        {
            const float xv = x[static_cast<size_t>(ic) * frames + t];
            for (int oc = 0; oc < out_ch; ++oc)
            {
                for (int k = 0; k < ksize; ++k)
                {
                    const int ot = t * stride + k;
                    if (ot >= out_frames)
                    {
                        continue;
                    }
                    const size_t widx = static_cast<size_t>(ic) * out_ch * ksize +
                                        static_cast<size_t>(oc) * ksize + k;
                    y[static_cast<size_t>(oc) * out_frames + ot] += xv * w[widx];
                }
            }
        }
    }
    if (!bias.empty())
    {
        for (int oc = 0; oc < out_ch; ++oc)
        {
            float *row = y.data() + static_cast<size_t>(oc) * out_frames;
            const float b = bias[std::min<int>(oc, bias.size() - 1)];
            for (int t = 0; t < out_frames; ++t)
            {
                row[t] += b;
            }
        }
    }
    return y;
}

struct ggml_tensor *conv_1d_f32(struct ggml_context *ctx,
                                struct ggml_tensor *weight,
                                struct ggml_tensor *input,
                                int stride, int padding, int dilation)
{
    if (weight->type != GGML_TYPE_F32)
    {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    }
    struct ggml_tensor *im2col = ggml_im2col(
        ctx, weight, input, stride, 0, padding, 0, dilation, 0,
        false, GGML_TYPE_F32);
    struct ggml_tensor *result = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                        im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, weight,
                        weight->ne[0] * weight->ne[1], weight->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1],
                           weight->ne[2], im2col->ne[2]);
}

struct ggml_tensor *conv_1d_dw_f32(struct ggml_context *ctx,
                                   struct ggml_tensor *weight,
                                   struct ggml_tensor *input,
                                   int stride, int padding, int dilation)
{
    if (weight->type != GGML_TYPE_F32)
    {
        weight = ggml_cast(ctx, weight, GGML_TYPE_F32);
    }
    struct ggml_tensor *grouped_input = ggml_reshape_4d(
        ctx, input, input->ne[0], 1, input->ne[1], input->ne[2]);
    struct ggml_tensor *im2col = ggml_im2col(
        ctx, weight, grouped_input, stride, 0, padding, 0, dilation, 0,
        false, GGML_TYPE_F32);
    struct ggml_tensor *result = ggml_mul_mat(ctx, im2col, weight);
    return ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);
}

} // namespace

bool VoxCPM2Model::RunAudioVAE(VoxCPM2State &state, ggml_backend_sched_t sched)
{
    (void)sched;
    RS_LOG_INFO("VoxCPM2: Running Audio VAE");
    const char *vae_debug_env = std::getenv("RS_VOXCPM2_VAE_DEBUG");
    const bool debug_vae = vae_debug_env && std::strcmp(vae_debug_env, "1") == 0;
    auto log_vae_stats = [&](const char *stage, const std::vector<float> &values) {
        if (!debug_vae || values.empty())
        {
            return;
        }
        double sum_sq = 0.0;
        float peak = 0.0f;
        for (float value : values)
        {
            sum_sq += static_cast<double>(value) * value;
            peak = std::max(peak, std::fabs(value));
        }
        RS_LOG_INFO("VoxCPM2: VAE debug %s rms=%.6f peak=%.6f n=%zu",
                    stage, std::sqrt(sum_sq / values.size()), peak, values.size());
    };

    const int latent_dim = 64;
    const int latent_frames = static_cast<int>(state.denoised_latents.size()) / latent_dim;
    if (latent_frames <= 0)
    {
        RS_LOG_ERR("VoxCPM2: no denoised latents for AudioVAE");
        return false;
    }

    auto run_ggml_decoder = [&]() {
        struct ggml_init_params params = {
            .mem_size   = 32 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        struct ggml_context *ctx0 = ggml_init(params);
        if (!ctx0)
        {
            return false;
        }
        struct ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 8192, false);
        bool graph_ok = true;
        std::vector<std::pair<std::string, struct ggml_tensor *>> debug_outputs;
        std::vector<std::pair<struct ggml_tensor *, const std::vector<float> *>>
            scale_inputs;
        auto capture_debug = [&](const std::string &name,
                                 struct ggml_tensor *value) {
            if (debug_vae)
            {
                ggml_set_output(value);
                debug_outputs.emplace_back(name, value);
            }
        };

        auto to_f32 = [&](struct ggml_tensor *value) {
            return value && value->type != GGML_TYPE_F32 ?
                ggml_cast(ctx0, value, GGML_TYPE_F32) : value;
        };
        auto channel_param = [&](struct ggml_tensor *value, int channels) {
            if (!value || ggml_nelements(value) < channels)
            {
                graph_ok = false;
                return static_cast<struct ggml_tensor *>(nullptr);
            }
            return ggml_reshape_2d(ctx0, to_f32(value), 1, channels);
        };
        auto add_bias = [&](struct ggml_tensor *value,
                            struct ggml_tensor *bias, int channels) {
            struct ggml_tensor *param = channel_param(bias, channels);
            return param ? ggml_add(ctx0, value, param) : value;
        };
        auto weight_norm_scale = [&](const std::string &prefix,
                                     int rows, int channels) {
            auto scale_it = audio_vae_weight_scales_.find(prefix);
            if (scale_it == audio_vae_weight_scales_.end() ||
                scale_it->second.size() != static_cast<size_t>(channels) ||
                !get_tensor(prefix + ".weight_v") || rows <= 0)
            {
                graph_ok = false;
                return static_cast<struct ggml_tensor *>(nullptr);
            }
            struct ggml_tensor *scale = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, 1, channels);
            ggml_set_input(scale);
            scale_inputs.emplace_back(scale, &scale_it->second);
            return scale;
        };
        auto snake = [&](struct ggml_tensor *value,
                         const std::string &name, int channels) {
            struct ggml_tensor *alpha = channel_param(get_tensor(name), channels);
            if (!alpha)
            {
                return value;
            }
            struct ggml_tensor *phase = ggml_mul(ctx0, value, alpha);
            struct ggml_tensor *wave = ggml_sin(ctx0, phase);
            wave = ggml_sqr(ctx0, wave);
            return ggml_add(ctx0, value, ggml_div(ctx0, wave, alpha));
        };
        auto causal_conv = [&](struct ggml_tensor *value,
                               const std::string &prefix,
                               int in_channels, int out_channels,
                               int kernel, int dilation, bool depthwise) {
            struct ggml_tensor *weight = get_tensor(prefix + ".weight_v");
            if (!weight)
            {
                graph_ok = false;
                return value;
            }
            if (ggml_is_quantized(weight->type))
            {
                if (depthwise || kernel != 1 ||
                    weight->ne[0] != in_channels ||
                    weight->ne[1] != out_channels)
                {
                    graph_ok = false;
                    return value;
                }
                struct ggml_tensor *input_mat = ggml_cont(
                    ctx0, ggml_transpose(ctx0, value));
                struct ggml_tensor *output_mat = ggml_mul_mat(
                    ctx0, weight, input_mat);
                struct ggml_tensor *output = ggml_cont(
                    ctx0, ggml_transpose(ctx0, output_mat));
                struct ggml_tensor *scale = weight_norm_scale(
                    prefix, in_channels, out_channels);
                if (scale)
                {
                    output = ggml_mul(ctx0, output, scale);
                }
                return ggml_cont(ctx0, add_bias(
                    output, get_tensor(prefix + ".bias"), out_channels));
            }
            struct ggml_tensor *input = ggml_reshape_3d(
                ctx0, value, value->ne[0], in_channels, 1);
            const int padding = (kernel - 1) * dilation;
            struct ggml_tensor *output = depthwise ?
                ggml_conv_1d_dw(ctx0, weight, input, 1, padding, dilation) :
                ggml_conv_1d(ctx0, weight, input, 1, padding, dilation);
            output = ggml_reshape_2d(ctx0, output, output->ne[0], out_channels);
            const int frames = static_cast<int>(value->ne[0]);
            output = ggml_view_2d(ctx0, output, frames, out_channels,
                                  output->nb[1], 0);
            struct ggml_tensor *scale = weight_norm_scale(
                prefix, kernel * (depthwise ? 1 : in_channels), out_channels);
            if (scale)
            {
                output = ggml_mul(ctx0, output, scale);
            }
            return ggml_cont(ctx0, add_bias(
                output, get_tensor(prefix + ".bias"), out_channels));
        };
        auto causal_deconv = [&](struct ggml_tensor *value,
                                 const std::string &prefix,
                                 int in_channels, int out_channels,
                                 int kernel, int stride) {
            struct ggml_tensor *weight = get_tensor(prefix + ".weight_v");
            if (!weight)
            {
                graph_ok = false;
                return value;
            }
            struct ggml_tensor *scale = weight_norm_scale(
                prefix, kernel * out_channels, in_channels);
            if (scale)
            {
                value = ggml_mul(ctx0, value, scale);
            }
            const int input_frames = static_cast<int>(value->ne[0]);
            const int frames = input_frames * stride;

            // For K=2*stride, each output sample receives one contribution
            // from the current input frame and one from the previous frame.
            // Expressing the projection as GEMM lets Metal use its optimized
            // matrix kernels instead of the scalar ConvTranspose kernel.
            struct ggml_tensor *weight_mat = weight;
            if (!ggml_is_quantized(weight->type))
            {
                struct ggml_tensor *weight_rows = ggml_reshape_2d(
                    ctx0, weight, kernel * out_channels, in_channels);
                weight_mat = ggml_cont(
                    ctx0, ggml_transpose(ctx0, weight_rows));
            }
            else if (weight->ne[0] != in_channels ||
                     weight->ne[1] != kernel * out_channels)
            {
                graph_ok = false;
                return value;
            }
            struct ggml_tensor *input_mat = ggml_cont(
                ctx0, ggml_transpose(ctx0, value));
            struct ggml_tensor *projected = ggml_mul_mat(
                ctx0, weight_mat, input_mat);
            projected = ggml_reshape_3d(
                ctx0, projected, kernel, out_channels, input_frames);

            struct ggml_tensor *current = ggml_view_3d(
                ctx0, projected, stride, out_channels, input_frames,
                projected->nb[1], projected->nb[2], 0);
            current = ggml_cont(
                ctx0, ggml_permute(ctx0, current, 0, 2, 1, 3));
            current = ggml_reshape_2d(
                ctx0, current, frames, out_channels);

            struct ggml_tensor *previous = ggml_view_3d(
                ctx0, projected, stride, out_channels, input_frames - 1,
                projected->nb[1], projected->nb[2],
                static_cast<size_t>(stride) * projected->nb[0]);
            previous = ggml_cont(
                ctx0, ggml_permute(ctx0, previous, 0, 2, 1, 3));
            previous = ggml_reshape_2d(
                ctx0, previous, frames - stride, out_channels);
            struct ggml_tensor *leading_zeros = ggml_scale(
                ctx0, ggml_view_2d(ctx0, current, stride, out_channels,
                                   current->nb[1], 0), 0.0f);
            previous = ggml_concat(ctx0, leading_zeros, previous, 0);
            struct ggml_tensor *output = ggml_add(ctx0, current, previous);
            return ggml_cont(ctx0, add_bias(
                output, get_tensor(prefix + ".bias"), out_channels));
        };
        auto sr_cond = [&](struct ggml_tensor *value, int channels,
                           int layer_idx) {
            const std::string prefix =
                "audio_vae.decoder.sr_cond_model." + std::to_string(layer_idx);
            struct ggml_tensor *scale = get_tensor(prefix + ".scale_embed.weight");
            struct ggml_tensor *bias = get_tensor(prefix + ".bias_embed.weight");
            if (!scale || scale->ne[0] != channels || scale->ne[1] <= 3)
            {
                graph_ok = false;
                return value;
            }
            struct ggml_tensor *scale_row = ggml_view_1d(
                ctx0, scale, channels, static_cast<size_t>(3) * scale->nb[1]);
            scale_row = ggml_reshape_2d(ctx0, to_f32(scale_row), 1, channels);
            value = ggml_mul(ctx0, value, scale_row);
            if (bias)
            {
                struct ggml_tensor *bias_row = ggml_view_1d(
                    ctx0, bias, channels, static_cast<size_t>(3) * bias->nb[1]);
                bias_row = ggml_reshape_2d(ctx0, to_f32(bias_row), 1, channels);
                value = ggml_add(ctx0, value, bias_row);
            }
            return value;
        };
        auto residual_unit = [&](struct ggml_tensor *value,
                                 const std::string &prefix,
                                 int channels, int dilation) {
            struct ggml_tensor *skip = value;
            value = snake(value, prefix + ".block.0.alpha", channels);
            value = causal_conv(value, prefix + ".block.1", channels,
                                channels, 7, dilation, true);
            value = snake(value, prefix + ".block.2.alpha", channels);
            value = causal_conv(value, prefix + ".block.3", channels,
                                channels, 1, 1, false);
            return ggml_add(ctx0, skip, value);
        };

        struct ggml_tensor *latent = ggml_new_tensor_2d(
            ctx0, GGML_TYPE_F32, latent_frames, latent_dim);
        ggml_set_name(latent, "voxcpm2_vae_latent");
        ggml_set_input(latent);
        struct ggml_tensor *cur = causal_conv(
            latent, "audio_vae.decoder.model.0", 64, 64, 7, 1, true);
        capture_debug("graph.model0", cur);
        cur = causal_conv(cur, "audio_vae.decoder.model.1", 64, 2048,
                          1, 1, false);
        capture_debug("graph.model1", cur);

        static const int up_rates[] = {8, 6, 5, 2, 2, 2};
        static const int out_channels[] = {1024, 512, 256, 128, 64, 32};
        int channels = 2048;
        for (int block = 0; block < 6; ++block)
        {
            const int layer_idx = block + 2;
            const std::string prefix =
                "audio_vae.decoder.model." + std::to_string(layer_idx);
            cur = sr_cond(cur, channels, layer_idx);
            cur = snake(cur, prefix + ".block.0.alpha", channels);
            cur = causal_deconv(cur, prefix + ".block.1", channels,
                                out_channels[block], 2 * up_rates[block],
                                up_rates[block]);
            channels = out_channels[block];
            cur = residual_unit(cur, prefix + ".block.2", channels, 1);
            cur = residual_unit(cur, prefix + ".block.3", channels, 3);
            cur = residual_unit(cur, prefix + ".block.4", channels, 9);
            capture_debug("graph.block" + std::to_string(block), cur);
        }
        cur = snake(cur, "audio_vae.decoder.model.8.alpha", channels);
        cur = causal_conv(cur, "audio_vae.decoder.model.9", channels,
                          1, 7, 1, false);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_cont(ctx0, ggml_view_1d(
            ctx0, cur, ggml_nelements(cur), 0));
        ggml_set_name(cur, "voxcpm2_vae_audio");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        if (!graph_ok)
        {
            ggml_free(ctx0);
            return false;
        }
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf))
        {
            ggml_free(ctx0);
            return false;
        }
        for (const auto &entry : scale_inputs)
        {
            ggml_backend_tensor_set(entry.first, entry.second->data(), 0,
                                   entry.second->size() * sizeof(float));
        }
        std::vector<float> latent_cf(state.denoised_latents.size());
        for (int t = 0; t < latent_frames; ++t)
        {
            for (int c = 0; c < latent_dim; ++c)
            {
                latent_cf[static_cast<size_t>(c) * latent_frames + t] =
                    state.denoised_latents[static_cast<size_t>(t) * latent_dim + c];
            }
        }
        ggml_backend_tensor_set(latent, latent_cf.data(), 0,
                               latent_cf.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS)
        {
            ggml_free(ctx0);
            return false;
        }
        if (debug_vae)
        {
            for (const auto &entry : debug_outputs)
            {
                std::vector<float> values(ggml_nelements(entry.second));
                ggml_backend_tensor_get(entry.second, values.data(), 0,
                                       values.size() * sizeof(float));
                log_vae_stats(entry.first.c_str(), values);
            }
        }
        state.audio_output.resize(ggml_nelements(cur));
        ggml_backend_tensor_get(cur, state.audio_output.data(), 0,
                               state.audio_output.size() * sizeof(float));
        ggml_free(ctx0);
        RS_LOG_INFO("VoxCPM2: AudioVAE ggml graph generated %zu samples (%.2fs)",
                    state.audio_output.size(),
                    static_cast<float>(state.audio_output.size()) /
                        hparams_.vae_out_sample_rate);
        return true;
    };

    const char *vae_backend = std::getenv("RS_VOXCPM2_VAE_CPU");
    const bool force_cpu_vae = vae_backend && std::strcmp(vae_backend, "1") == 0;
    if (!force_cpu_vae && run_ggml_decoder())
    {
        return true;
    }
    RS_LOG_WARN("VoxCPM2: AudioVAE graph unavailable; falling back to CPU decoder");

    auto tensor = [&](const std::string &name) {
        return tensor_to_f32(get_tensor(name));
    };
    auto conv_weight = [&](const std::string &prefix, int out_ch, int in_ch, int ksize) {
        return reconstruct_weight_norm_conv(
            tensor(prefix + ".weight_g"), tensor(prefix + ".weight_v"),
            out_ch, in_ch, ksize);
    };
    auto deconv_weight = [&](const std::string &prefix, int in_ch, int out_ch, int ksize) {
        return reconstruct_weight_norm_deconv(
            tensor(prefix + ".weight_g"), tensor(prefix + ".weight_v"),
            in_ch, out_ch, ksize);
    };
    auto apply_sr_cond = [&](std::vector<float> &x, int channels, int frames, int layer_idx) {
        const std::vector<float> scale =
            tensor("audio_vae.decoder.sr_cond_model." + std::to_string(layer_idx) + ".scale_embed.weight");
        const std::vector<float> bias =
            tensor("audio_vae.decoder.sr_cond_model." + std::to_string(layer_idx) + ".bias_embed.weight");
        if (scale.empty())
        {
            return;
        }
        const int bucket = 3;
        for (int c = 0; c < channels; ++c)
        {
            const float s = scale[static_cast<size_t>(bucket) * channels + c];
            const float b = bias.empty() ? 0.0f : bias[static_cast<size_t>(bucket) * channels + c];
            float *row = x.data() + static_cast<size_t>(c) * frames;
            for (int t = 0; t < frames; ++t)
            {
                row[t] = row[t] * s + b;
            }
        }
    };
    auto residual_unit = [&](const std::string &prefix,
                             const std::vector<float> &input,
                             int channels, int frames, int dilation) {
        std::vector<float> x = input;
        snake1d_inplace(x, tensor(prefix + ".block.0.alpha"), channels, frames);
        std::vector<float> w1 = conv_weight(prefix + ".block.1", channels, 1, 7);
        x = causal_conv1d_cpu(x, w1, tensor(prefix + ".block.1.bias"),
                              channels, channels, 7, frames, dilation, channels);
        snake1d_inplace(x, tensor(prefix + ".block.2.alpha"), channels, frames);
        std::vector<float> w2 = conv_weight(prefix + ".block.3", channels, channels, 1);
        x = causal_conv1d_cpu(x, w2, tensor(prefix + ".block.3.bias"),
                              channels, channels, 1, frames, 1, 1);
        for (size_t i = 0; i < x.size() && i < input.size(); ++i)
        {
            x[i] += input[i];
        }
        return x;
    };

    RS_LOG_INFO("VoxCPM2: AudioVAE decoding %d latent frames with real decoder weights",
                latent_frames);

    std::vector<float> x(static_cast<size_t>(latent_dim) * latent_frames, 0.0f);
    for (int t = 0; t < latent_frames; ++t)
    {
        for (int c = 0; c < latent_dim; ++c)
        {
            x[static_cast<size_t>(c) * latent_frames + t] =
                state.denoised_latents[static_cast<size_t>(t) * latent_dim + c];
        }
    }

    int channels = latent_dim;
    int frames = latent_frames;

    x = causal_conv1d_cpu(x, conv_weight("audio_vae.decoder.model.0", 64, 1, 7),
                          tensor("audio_vae.decoder.model.0.bias"),
                          64, 64, 7, frames, 1, 64);
    log_vae_stats("cpu.model0", x);
    x = causal_conv1d_cpu(x, conv_weight("audio_vae.decoder.model.1", 2048, 64, 1),
                          tensor("audio_vae.decoder.model.1.bias"),
                          64, 2048, 1, frames, 1, 1);
    log_vae_stats("cpu.model1", x);
    channels = 2048;

    static const int up_rates[] = {8, 6, 5, 2, 2, 2};
    static const int out_channels[] = {1024, 512, 256, 128, 64, 32};
    for (int b = 0; b < 6; ++b)
    {
        const int layer_idx = b + 2;
        const std::string prefix = "audio_vae.decoder.model." + std::to_string(layer_idx);
        apply_sr_cond(x, channels, frames, layer_idx);
        snake1d_inplace(x, tensor(prefix + ".block.0.alpha"), channels, frames);

        const int next_channels = out_channels[b];
        const int ksize = 2 * up_rates[b];
        x = causal_deconv1d_cpu(
            x, deconv_weight(prefix + ".block.1", channels, next_channels, ksize),
            tensor(prefix + ".block.1.bias"), channels, next_channels,
            ksize, frames, up_rates[b]);
        channels = next_channels;
        frames *= up_rates[b];

        x = residual_unit(prefix + ".block.2", x, channels, frames, 1);
        x = residual_unit(prefix + ".block.3", x, channels, frames, 3);
        x = residual_unit(prefix + ".block.4", x, channels, frames, 9);
        const std::string debug_stage = "cpu.block" + std::to_string(b);
        log_vae_stats(debug_stage.c_str(), x);

        RS_LOG_INFO("VoxCPM2: AudioVAE block %d -> [%d, %d]", b, channels, frames);
    }

    snake1d_inplace(x, tensor("audio_vae.decoder.model.8.alpha"), channels, frames);
    std::vector<float> pcm = causal_conv1d_cpu(
        x, conv_weight("audio_vae.decoder.model.9", 1, channels, 7),
        tensor("audio_vae.decoder.model.9.bias"), channels, 1, 7, frames, 1, 1);

    state.audio_output.resize(frames);
    for (int i = 0; i < frames; ++i)
    {
        state.audio_output[i] = std::tanh(pcm[i]);
    }

    RS_LOG_INFO("VoxCPM2: AudioVAE generated %d samples (%.2fs @ %d Hz)",
                frames, static_cast<float>(frames) / hparams_.vae_out_sample_rate,
                hparams_.vae_out_sample_rate);
    return true;
}

// =====================================================================
// Model Registration
// =====================================================================

namespace
{
struct VoxCPM2Registrar
{
    VoxCPM2Registrar()
    {
        rs_register_model_arch("voxcpm2", []() {
            return std::make_shared<VoxCPM2Model>();
        });
    }
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
static VoxCPM2Registrar global_voxcpm2_reg;
} // namespace
