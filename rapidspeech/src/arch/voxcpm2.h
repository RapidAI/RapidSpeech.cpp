#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "arch/llm_model.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * VoxCPM2 TTS Model — 2B parameter diffusion autoregressive TTS
 *
 * Architecture:
 *   Text → Text Encoder (12L) → Base LM (28L, Qwen2-like)
 *       → Scalar Quantization → DiT (12L, CFM) → Audio VAE → 48kHz Audio
 *
 * Components:
 *   1. Text Encoder: 12-layer transformer (1024-dim)
 *   2. Base LM: 28-layer autoregressive LM (2048-dim, GQA)
 *   3. Scalar Quantization: 512-dim latent space
 *   4. DiT: 12-layer diffusion transformer with CFM
 *   5. Audio VAE: Convolutional decoder (64-dim → 48kHz)
 *
 * Reference: OpenBMB/VoxCPM2
 */

// =====================================================================
// Hyperparameters
// =====================================================================

struct VoxCPM2HParams
{
    // Base LM (Qwen2-like)
    int32_t lm_num_hidden_layers = 28;
    int32_t lm_hidden_size = 2048;
    int32_t lm_num_attention_heads = 16;
    int32_t lm_num_key_value_heads = 2;  // GQA
    int32_t lm_intermediate_size = 6144;
    int32_t lm_vocab_size = 73448;
    int32_t lm_max_position_embeddings = 32768;
    float lm_rms_norm_eps = 1e-5f;
    float lm_rope_theta = 10000.0f;
    int32_t lm_kv_channels = 128;

    // Text Encoder
    int32_t encoder_hidden_dim = 1024;
    int32_t encoder_ffn_dim = 4096;
    int32_t encoder_num_heads = 16;
    int32_t encoder_num_layers = 12;
    int32_t encoder_kv_channels = 128;

    // DiT (Diffusion Transformer)
    int32_t dit_hidden_dim = 1024;
    int32_t dit_ffn_dim = 4096;
    int32_t dit_num_heads = 16;
    int32_t dit_num_layers = 12;
    int32_t dit_kv_channels = 128;
    float dit_inference_cfg_rate = 2.0f;

    // Scalar Quantization
    int32_t scalar_quantization_latent_dim = 512;
    int32_t scalar_quantization_scale = 9;

    // Audio VAE
    int32_t vae_encoder_dim = 128;
    std::vector<int32_t> vae_encoder_rates = {2, 5, 8, 8};
    int32_t vae_latent_dim = 64;
    int32_t vae_decoder_dim = 2048;
    std::vector<int32_t> vae_decoder_rates = {8, 6, 5, 2, 2, 2};
    int32_t vae_sample_rate = 16000;
    int32_t vae_out_sample_rate = 48000;

    // General
    int32_t patch_size = 4;
    int32_t feat_dim = 64;
    int32_t max_length = 8192;
};

// =====================================================================
// VoxCPM2 State
// =====================================================================

struct VoxCPM2State : public RSState
{
    // Input
    std::string input_text;
    std::vector<int32_t> token_ids;

    // Voice cloning (reference audio)
    std::vector<float> reference_audio;  // Reference speaker audio
    int32_t reference_sample_rate = 0;
    std::vector<float> reference_embeddings;  // Speaker embeddings

    // Ultimate cloning (prompt audio + transcript)
    std::vector<float> prompt_audio;     // Prompt audio (for continuation)
    std::string prompt_text;              // Transcript of prompt audio
    std::vector<float> prompt_embeddings;

    // Style control (optional)
    std::string style_description;        // e.g., "(slightly faster, cheerful tone)"

    // Voice design (no reference needed)
    std::string voice_description;        // e.g., "(young woman, gentle voice)"

    // Text Encoder output
    std::vector<float> text_embeddings;  // [seq_len, 1024]
    int32_t text_seq_len = 0;

    // Base LM output (audio tokens)
    std::vector<float> base_lm_hidden;  // Last TSLM hidden state [2048]
    std::vector<float> base_lm_outputs; // TSLM hidden states [T, 2048]
    std::vector<float> base_lm_k_cache; // [layer, kv_head, position, head_dim]
    std::vector<float> base_lm_v_cache;
    std::vector<float> residual_lm_k_cache;
    std::vector<float> residual_lm_v_cache;
    int32_t ar_cache_length = 0;
    std::vector<int32_t> audio_tokens;
    int32_t audio_token_len = 0;

    // Scalar Quantization output
    std::vector<float> quantized_latents;  // [T, 512]
    std::vector<float> dit_mu;             // [2, 1024] flattened conditioning

    // DiT output (denoised latents)
    std::vector<float> denoised_latents;  // [T, 64]

    // Audio VAE output (final audio)
    std::vector<float> audio_output;  // 48kHz

    // Status flags
    bool has_reference = false;
    bool has_prompt = false;
    bool has_style = false;
    bool has_voice_design = false;

    bool reference_encoded = false;
    bool text_encoded = false;
    bool tokens_generated = false;
    bool latents_quantized = false;
    bool latents_denoised = false;
    bool audio_generated = false;
    uint64_t generation_seed = 42;
    int32_t diffusion_steps = 8;

    // Cloning mode
    enum CloneMode {
        NONE = 0,           // No cloning, default voice
        BASIC = 1,          // reference_audio only
        CONTROLLABLE = 2,   // reference_audio + style_description
        ULTIMATE = 3,       // prompt_audio + prompt_text + reference_audio
        VOICE_DESIGN = 4    // voice_description only (no reference)
    };
    CloneMode clone_mode = NONE;
};

// =====================================================================
// VoxCPM2 Model
// =====================================================================

// rs-server drives per-request VoxCPM2 states through this C++ interface.
// The core shared library uses hidden visibility by default, so export the
// class just like the CosyVoice3 server scheduler interface.
#if defined(_WIN32)
#define RS_VOXCPM2_EXPORT __declspec(dllexport)
#else
#define RS_VOXCPM2_EXPORT __attribute__((visibility("default")))
#endif

class RS_VOXCPM2_EXPORT VoxCPM2Model : public ISpeechModel
{
public:
    VoxCPM2Model();
    ~VoxCPM2Model() override;

    // ISpeechModel interface
    bool Load(const std::unique_ptr<rs_context_t> &ctx,
              ggml_backend_t backend) override;

    std::shared_ptr<RSState> CreateState() override;

    bool Encode(const std::vector<float> &input_frames, RSState &state,
                ggml_backend_sched_t sched) override;

    bool Decode(RSState &state, ggml_backend_sched_t sched) override;

    using AudioChunkCallback = ISpeechModel::AudioChunkCallback;
    bool SupportsTTSStreaming() const override { return true; }
    bool DecodeStream(RSState &state, ggml_backend_sched_t sched,
                      AudioChunkCallback emit) override;

    std::string GetTranscription(RSState &state) override;

    // TTS interface
    bool PushText(RSState &state, const char *text,
                  const char *language = nullptr,
                  const char *instruct = nullptr) override;

    int GetAudioOutput(RSState &state, float **out_data) override;

    bool PushReferenceAudio(RSState &state, const float *samples,
                            int n_samples, int sample_rate,
                            ggml_backend_sched_t sched) override;

    void SetDiffusionSteps(int n_steps) override;
    void SetSeed(uint64_t seed) override;
    void ConfigureState(VoxCPM2State &state, uint64_t seed,
                        int diffusion_steps) const;

    // Voice cloning interface
    bool SetReferenceAudio(RSState &state, const float *audio, int n_samples,
                           int sample_rate);

    bool SetPromptAudio(RSState &state, const float *audio, int n_samples,
                        int sample_rate, const char *transcript);

    bool SetStyleDescription(RSState &state, const char *style);

    bool SetVoiceDesign(RSState &state, const char *voice_desc);

    const RSModelMeta &GetMeta() const override { return meta_; }

private:
    // Component loaders
    bool LoadTextEncoder(struct gguf_context *ctx_gguf);
    bool LoadBaseLM(struct gguf_context *ctx_gguf);
    bool LoadScalarQuantization(struct gguf_context *ctx_gguf);
    bool LoadDiT(struct gguf_context *ctx_gguf);
    bool LoadAudioVAE(struct gguf_context *ctx_gguf);

    // Forward passes
    bool RunReferenceEncoder(VoxCPM2State &state, ggml_backend_sched_t sched);
    bool RunTextEncoder(VoxCPM2State &state, ggml_backend_sched_t sched);
    bool RunBaseLM(VoxCPM2State &state, ggml_backend_sched_t sched);
    bool RunScalarQuantization(VoxCPM2State &state, ggml_backend_sched_t sched);
    bool RunDiT(VoxCPM2State &state, ggml_backend_sched_t sched,
                const AudioChunkCallback &emit = {});
    bool RunAudioVAE(VoxCPM2State &state, ggml_backend_sched_t sched);
    bool DecodeImpl(RSState &state, ggml_backend_sched_t sched,
                    const AudioChunkCallback &emit);

    // Utilities
    ggml_tensor *get_tensor(const std::string &name);
    bool load_hparams_from_gguf(struct gguf_context *ctx_gguf);
    bool load_tokenizer_from_gguf(struct gguf_context *ctx_gguf);

    // Tokenizer
    std::vector<int32_t> tokenize(const std::string &text);

    // Audio resampling
    std::vector<float> resample_audio(const std::vector<float> &audio,
                                     int src_rate, int dst_rate);

    // Members
    RSModelMeta meta_;
    VoxCPM2HParams hparams_;

    // Backend and tensors
    ggml_backend_t backend_ = nullptr;
    struct ggml_context *ctx_data_ = nullptr;
    std::unordered_map<std::string, struct ggml_tensor *> tensors_;
    std::unordered_map<std::string, std::vector<float>> audio_vae_weight_scales_;
    struct ggml_context *dit_fused_ctx_ = nullptr;
    ggml_backend_buffer_t dit_fused_buffer_ = nullptr;
    std::vector<struct ggml_tensor *> dit_qkv_weights_;

    // Base LM (reuse existing llm_model)
    std::shared_ptr<llm_model> base_lm_;

    bool has_tokenizer_ = false;
    bool has_text_encoder_ = false;
    bool has_base_lm_ = false;
    bool has_fsq_ = false;
    bool has_dit_ = false;
    bool has_audio_vae_ = false;
    uint64_t seed_ = 42;
    int32_t diffusion_steps_ = 8;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    int32_t token_bos_ = 1;
    int32_t token_eos_ = 2;
    int32_t token_unk_ = 0;
    int32_t token_audio_start_ = 101;
    int32_t token_audio_end_ = 102;
    int32_t token_ref_audio_start_ = 103;
    int32_t token_ref_audio_end_ = 104;
};
