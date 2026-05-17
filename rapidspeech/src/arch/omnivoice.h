#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "llm_graph.h"
#include "llm_model.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * OmniVoice TTS Model — single-stage non-autoregressive diffusion TTS.
 *
 * Architecture:
 *   Text Tokenizer (Qwen3 BPE) ──────┐
 *                                     ├→ Bidirectional Qwen3-0.6B → 8× Codebook Heads
 *   Acoustic Tokens (8 codebooks) ────┘        ↓ 32-step MaskGIT + CFG
 *                                     Audio Tokenizer Decoder (RVQ + DAC)
 *                                              ↓
 *                                     24 kHz waveform
 *
 * Reference: "OmniVoice: Towards Omnilingual Zero-Shot TTS with Diffusion
 *             Language Models" (arXiv:2604.00688)
 */

#define OMNIVOICE_N_COEBOOKS 8
#define OMNIVOICE_DEFAULT_N_STEPS 32
#define OMNIVOICE_MAX_NODES 16384

#ifdef RS_USE_METAL_DAC
class DACMetalDecoder;
#endif

// =====================================================================
// GPT-2 Byte-Level BPE Tokenizer (for OmniVoice)
// =====================================================================

struct OmniVoiceBPETokenizer {
    std::unordered_map<std::string, int> vocab;       // token_str -> id
    std::unordered_map<std::string, int> merges;      // "a b" -> rank
    std::string                          byte2str[256];
    int                                  eos_id = -1;
    int                                  n_vocab = 0;
    std::vector<std::string>             id_to_str;
    std::vector<std::pair<std::string, int>> specials; // registered special tokens

    bool load_from_gguf(const char *gguf_path);
    bool load_omnivoice_specials(const char *gguf_path);
    std::vector<int> encode(const std::string &text, bool add_eos = false) const;
    int get_special_id(const std::string &name) const;
};

// =====================================================================
// Hyperparameters
// =====================================================================

struct OmniVoiceHParams {
    // Transformer backbone
    int32_t n_layer = 28;
    int32_t n_embd = 1024;
    int32_t n_head = 16;
    int32_t n_head_kv = 8;
    int32_t head_dim = 128;
    int32_t n_ff = 3072;

    // Text tokenizer
    int32_t text_vocab_size = 151936;

    // Acoustic codebooks
    int32_t n_codebooks = OMNIVOICE_N_COEBOOKS;
    int32_t codebook_sizes[OMNIVOICE_N_COEBOOKS] = {};

    // Audio
    int32_t audio_sample_rate = 24000;
    int32_t hop_length = 960;  // cumulative DAC upsample stride

    // Diffusion
    int32_t n_diff_steps = OMNIVOICE_DEFAULT_N_STEPS;
    float   diffusion_tau = 0.1f;

    // RoPE
    float rope_theta = 1000000.0f;
    float eps = 1e-6f;
};

// =====================================================================
// OmniVoice State
// =====================================================================

struct OmniVoiceState : public RSState {
    // Text input
    std::string text_original;  // original text before tokenization
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> prompt_tokens;
    int n_prompt_frames = 0;

    // Acoustic token buffer [n_total_frames * n_codebooks]
    std::vector<int32_t> acoustic_tokens;
    int n_total_frames = 0;
    int n_target_frames = 0;

    // Audio output
    std::vector<float> audio_output;

    // Transformer hidden state cache
    struct ggml_tensor *encoder_out = nullptr;
    ggml_backend_buffer_t buffer_persistent = nullptr;
    struct ggml_context *ctx_persistent = nullptr;

    // Diffusion progress
    int diff_step = 0;

    // Language and instruct for prompt
    std::string language = "English";
    std::string instruct = "male";
    std::string ref_text;  // transcript of reference audio for voice cloning

    // Prevent re-decoding after first successful vocoder run
    bool vocoder_done = false;

    OmniVoiceState() {
        struct ggml_init_params params = {512 * ggml_tensor_overhead(), nullptr, true};
        ctx_persistent = ggml_init(params);
    }

    ~OmniVoiceState() override {
        if (buffer_persistent) ggml_backend_buffer_free(buffer_persistent);
        if (ctx_persistent) ggml_free(ctx_persistent);
    }
};

// =====================================================================
// RVQ Codec (8 codebooks, decode + encode paths)
// =====================================================================

struct RVQCodebookWeights {
    struct ggml_tensor *embed = nullptr;          // [64, 1025]
    struct ggml_tensor *embed_sq = nullptr;       // [1025] precomputed ||embed[:,j]||^2
    struct ggml_tensor *project_in_w = nullptr;   // [1024, 64] encode path
    struct ggml_tensor *project_in_b = nullptr;   // [64]
    struct ggml_tensor *project_out_w = nullptr;  // [64, 1024] decode path
    struct ggml_tensor *project_out_b = nullptr;  // [1024]
    std::vector<float>  embed_sq_cpu;             // CPU copy of embed_sq
};

struct RVQCodec {
    int num_codebooks = 8;
    int codebook_size = 1025;
    int codebook_dim = 64;
    int hidden = 1024;
    RVQCodebookWeights cb[OMNIVOICE_N_COEBOOKS];
};

// =====================================================================
// DAC Decoder (5 upsampling blocks)
// =====================================================================

#define DAC_NUM_BLOCKS 5
#define DAC_RES_UNITS 3

struct DACSnakeWeights {
    struct ggml_tensor *a = nullptr;  // [1, C]
    std::vector<float>  inv_b;        // precomputed 1/(alpha+eps)
};

struct DACResUnitWeights {
    DACSnakeWeights     s1, s2;
    struct ggml_tensor *c1w = nullptr; // [7, C, C]
    struct ggml_tensor *c1b = nullptr; // [C]
    struct ggml_tensor *c2w = nullptr; // [1, C, C]
    struct ggml_tensor *c2b = nullptr; // [C]
    int                 dilation = 1;
};

struct DACBlockWeights {
    DACSnakeWeights     s1;
    struct ggml_tensor *ctw = nullptr; // [K, OC, IC] for conv_transpose_1d
    struct ggml_tensor *ctb = nullptr; // [OC]
    DACResUnitWeights   ru[DAC_RES_UNITS];
    int                 in_ch = 0;
    int                 out_ch = 0;
    int                 stride = 1;
    int                 kernel = 2;
    int                 pad = 1;
    int                 output_pad = 0;
};

struct DACDecoder {
    struct ggml_tensor *c1w = nullptr; // [7, 256, 1024]
    struct ggml_tensor *c1b = nullptr; // [1024]
    DACBlockWeights     blk[DAC_NUM_BLOCKS];
    DACSnakeWeights     s_final;
    struct ggml_tensor *c2w = nullptr; // [7, 32, 1]
    struct ggml_tensor *c2b = nullptr; // [1]

    struct ggml_context *weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
};

// =====================================================================
// DAC Encoder (5 downsampling blocks, mirror of decoder)
// =====================================================================

struct DACEncBlockWeights {
    DACSnakeWeights     s1;
    struct ggml_tensor *cw = nullptr;  // [K, OC, IC] for conv_1d
    struct ggml_tensor *cb = nullptr;  // [OC]
    DACResUnitWeights   ru[DAC_RES_UNITS];
    int                 in_ch = 0, out_ch = 0, stride = 1, kernel = 2, pad = 1;
};

struct DACEncoder {
    struct ggml_tensor *c1w = nullptr; // conv1 weight
    struct ggml_tensor *c1b = nullptr; // conv1 bias
    DACEncBlockWeights  blk[DAC_NUM_BLOCKS];
    DACSnakeWeights     s_final;
    struct ggml_tensor *c2w = nullptr; // conv2 weight
    struct ggml_tensor *c2b = nullptr; // conv2 bias
};

// =====================================================================
// HuBERT / SemanticSpeech feature extractor structures
// =====================================================================

#define HUBERT_NUM_LAYERS 12
#define HUBERT_HIDDEN 768
#define HUBERT_NUM_HEADS 12
#define HUBERT_FFN_INNER 3072

struct HubertConvLayerWeights {
    struct ggml_tensor *conv_w = nullptr;
    struct ggml_tensor *ln_w = nullptr;   // GroupNorm weight (layer 0 only)
    struct ggml_tensor *ln_b = nullptr;   // GroupNorm bias (layer 0 only)
};

struct HubertFeatExtractor {
    HubertConvLayerWeights conv[7];
};

struct HubertFeatProjection {
    struct ggml_tensor *ln_w = nullptr;    // LayerNorm weight
    struct ggml_tensor *ln_b = nullptr;    // LayerNorm bias
    struct ggml_tensor *proj_w = nullptr;  // Linear weight [512, 768]
    struct ggml_tensor *proj_b = nullptr;  // Linear bias [768]
};

struct HubertAttentionWeights {
    struct ggml_tensor *q_w = nullptr, *q_b = nullptr;
    struct ggml_tensor *k_w = nullptr, *k_b = nullptr;
    struct ggml_tensor *v_w = nullptr, *v_b = nullptr;
    struct ggml_tensor *o_w = nullptr, *o_b = nullptr;
};

struct HubertFFNWeights {
    struct ggml_tensor *w1_w = nullptr, *w1_b = nullptr;  // intermediate
    struct ggml_tensor *w2_w = nullptr, *w2_b = nullptr;  // output
};

struct HubertLayerWeights {
    struct ggml_tensor *ln_attn_w = nullptr, *ln_attn_b = nullptr;
    HubertAttentionWeights attn;
    struct ggml_tensor *ln_ffn_w = nullptr, *ln_ffn_b = nullptr;
    HubertFFNWeights ffn;
};

struct HubertEncInit {
    struct ggml_tensor *pos_conv_w = nullptr, *pos_conv_b = nullptr;
    struct ggml_tensor *ln_w = nullptr, *ln_b = nullptr;
};

// SemanticEncoder (refines HuBERT features)
struct SemanticEncBlockWeights {
    struct ggml_tensor *cw = nullptr, *cb = nullptr;  // conv
    struct { struct ggml_tensor *c1w, *c2w; } ru[2];  // res_units (no bias, no snake)
};

struct SemanticEncoderWeights {
    struct ggml_tensor *c1w = nullptr;  // encoder_semantic.conv
    SemanticEncBlockWeights blk[2];     // encoder_semantic.conv_blocks.0/1
};

// =====================================================================
// MaskGIT Config
// =====================================================================

struct MaskgitConfig {
    int      num_step = 32;
    float    guidance_scale = 2.0f;
    float    t_shift = 0.1f;
    float    layer_penalty_factor = 5.0f;
    float    position_temperature = 5.0f;
    float    class_temperature = 0.0f;
    uint64_t seed = 42;
};

// =====================================================================
// Prompt
// =====================================================================

struct OmniVoicePrompt {
    int B = 1;
    int B_prime = 2;  // cond + uncond
    int K = 8;
    int S_max = 0;
    int c_len = 0;  // cond effective length
    int u_len = 0;  // uncond effective length

    std::vector<int32_t> input_ids;       // [B_prime, K, S_max]
    std::vector<int32_t> audio_mask;      // [B_prime, S_max]
    std::vector<int32_t> attention_mask;  // [B_prime, S_max, S_max]
};

// =====================================================================
// OmniVoice Model
// =====================================================================

class OmniVoiceModel : public ISpeechModel {
public:
    OmniVoiceModel();
    ~OmniVoiceModel() override;

    // ISpeechModel interface
    bool Load(const std::unique_ptr<rs_context_t> &ctx, ggml_backend_t backend) override;
    std::shared_ptr<RSState> CreateState() override;

    bool Encode(const std::vector<float> &input_frames, RSState &state,
                ggml_backend_sched_t sched) override;

    bool Decode(RSState &state, ggml_backend_sched_t sched) override;

    std::string GetTranscription(RSState &state) override { (void)state; return ""; }
    const RSModelMeta &GetMeta() const override { return meta_; }

    // TTS-specific methods
    bool PushText(RSState &state, const char *text, const char *language = nullptr,
                  const char *instruct = nullptr) override;
    bool PushReferenceAudio(RSState &state, const float *samples, int n_samples,
                            int sample_rate, ggml_backend_sched_t sched) override;
    bool PushReferenceText(RSState &state, const char *ref_text) override;
    int GetAudioOutput(RSState &state, float **out_data) override;
    void SetDiffusionSteps(int n_steps) override { n_diff_steps_ = n_steps; }

private:
    RSModelMeta meta_;
    OmniVoiceHParams hparams_;
    int n_diff_steps_ = OMNIVOICE_DEFAULT_N_STEPS;

    // BPE tokenizer
    OmniVoiceBPETokenizer bpe_;

    // Qwen3 backbone
    llm_model_ptr llm_model_;

    // Audio codebook heads [n_codebooks]
    struct ggml_tensor *codebook_head_weight_[OMNIVOICE_N_COEBOOKS] = {};
    struct ggml_tensor *codebook_head_bias_[OMNIVOICE_N_COEBOOKS] = {};

    // Acoustic embeddings [n_codebooks]
    struct ggml_tensor *acoustic_embeddings_[OMNIVOICE_N_COEBOOKS] = {};

    // Combined audio tensors (OmniVoice convention: [H, K*V] per tensor)
    struct ggml_tensor *combined_audio_embeddings_ = nullptr;
    struct ggml_tensor *combined_audio_heads_ = nullptr;
    // Single combined [H, 2*K*V] tensor (omnivoice.audio_codebook_weights)
    struct ggml_tensor *combined_codebook_weights_ = nullptr;

    // Output norm
    struct ggml_tensor *output_norm_ = nullptr;

    // Text embedding
    struct ggml_tensor *text_embd_ = nullptr;

    // Audio codec
    RVQCodec rvq_;
    DACDecoder dac_;
    DACEncoder dac_enc_;
    HubertFeatExtractor hubert_feat_;
    HubertFeatProjection hubert_proj_;
    HubertEncInit hubert_enc_init_;
    HubertLayerWeights hubert_layers_[HUBERT_NUM_LAYERS];
    SemanticEncoderWeights sem_enc_;
    struct ggml_tensor *fc_w_ = nullptr; // [1024, 1024] encode path
    struct ggml_tensor *fc_b_ = nullptr; // [1024]
    struct ggml_tensor *fc2_w_ = nullptr; // [1024, 256]
    struct ggml_tensor *fc2_b_ = nullptr; // [256]

    bool codec_loaded_ = false;
    std::string codec_path_;

    // Audio config
    int audio_vocab_size_ = 1025;
    int audio_mask_id_ = 1024;

    // Backend for codec weights (always CPU — DAC vocoder has many small conv
    // ops where GPU kernel launch overhead dominates, making CPU ~300x faster)
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;

#ifdef RS_USE_METAL_DAC
    // Optional Metal GPU backend for DAC vocoder (fused kernels, faster than CPU)
    std::unique_ptr<DACMetalDecoder> dac_metal_;
#endif

    // Internal methods
    bool LoadLM(const std::unique_ptr<rs_context_t> &ctx);
    bool LoadCodec(struct ggml_context *gguf_data);
    bool LoadEncoderWeights();
    bool MapTensors(std::map<std::string, struct ggml_tensor *> &tensors);

    bool BuildPrompt(OmniVoiceState &state, OmniVoicePrompt &prompt);
    bool RunDiffusionDecode(OmniVoiceState &state, ggml_backend_sched_t sched);
    bool RunVocoder(OmniVoiceState &state, ggml_backend_sched_t sched);

    // LLM forward (batched cond+uncond)
    std::vector<float> RunLLMForwardBatched(const OmniVoicePrompt &prompt,
                                            ggml_backend_sched_t sched,
                                            int T_audio);

    // Graph reuse: build LLM graph once, reuse across diffusion steps
    struct DiffusionGraphState;
    bool BuildDiffusionGraph(DiffusionGraphState &gs, const OmniVoicePrompt &prompt,
                             ggml_backend_sched_t sched, int T_audio);
    std::vector<float> RunDiffusionGraph(DiffusionGraphState &gs,
                                          const OmniVoicePrompt &prompt,
                                          ggml_backend_sched_t sched);
    void FreeDiffusionGraph(DiffusionGraphState &gs, ggml_backend_sched_t sched);

    // Audio encoding for voice cloning
    std::vector<int32_t> EncodeReferenceAudio(const float *audio_24k, int n_samples);

    // MaskGIT helpers
    void MaskGITLogSoftmax(float *x, int V);
    void MaskGITTopKFilter(float *x, int V, float ratio);
    void MaskGITGumbel(float *x, int n, float temperature, uint64_t seed, uint32_t &ctr_lo);
};
