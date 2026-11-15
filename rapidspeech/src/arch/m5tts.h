#pragma once

// M5-TTS (ASLP-lab/M5-TTS, 50M voice-clone LM + Qwen3-TTS 12Hz codec).
//
// Pipeline (mirrors the upstream ONNX reference implementation:
// onnx/graphs.py + infer_onnx.py from the M5-TTS repo):
//
//   text (+ref text) → wetext TN → llama3 byte-level BPE
//     → conditioning prefix <|caption_*|>...<|lang_begin|>lang<|lang_end|>
//     → backbone (llama-30M, 8L, KV-cached) prefill over
//         [text frames ‖ audio-prompt frames]
//     → frame loop:
//         backbone step → last_hidden → codebook0_head → sample c0
//         → MTP depth decoder (llama-20M, 5L, per-frame KV) ×15 codebooks
//         → 16 codec codes per frame (EOS = all-zero frame)
//     → Qwen 12Hz codec decoder (8L sliding-window transformer + causal
//         conv upsampler, stateful streaming) → 24 kHz PCM
//
// Reference audio is encoded with the codec encoder (SEANet + transformer +
// RVQ argmin quantization) bundled in the same GGUF.
//
// GGUF: general.architecture = "m5-tts"; produced by
// scripts/convert_m5tts_to_gguf.py.

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "frontend/wetext_normalizer.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace m5tts {

// ---------------------------------------------------------------------------
// Hyperparameters (populated from GGUF metadata in Load)
// ---------------------------------------------------------------------------
struct HParams {
    // ---- LM: temporal backbone (llama-30M flavor) ----
    int bb_layers   = 8;
    int bb_heads    = 8;
    int bb_kv_heads = 2;
    int bb_head_dim = 64;
    int bb_ff       = 2048;
    // ---- LM: local MTP depth decoder (llama-20M flavor) ----
    int dc_layers   = 5;
    int dc_heads    = 8;
    int dc_kv_heads = 2;
    int dc_head_dim = 64;
    int dc_ff       = 2048;
    // ---- shared LM ----
    int hidden      = 512;      // embed dim of both LM halves
    int text_vocab  = 129992;
    int audio_vocab = 2051;     // 2048 codebook + 3 specials
    int codebooks   = 16;
    int max_seq_len = 2048;
    float norm_eps  = 1e-5f;
    float temperature = 0.9f;
    int topk        = 10;
    int min_frames  = 6;        // EOS disallowed while frame index <= this
    int max_frames  = 500;
    int sample_rate = 24000;
    int hop_length  = 1920;     // codec: 24 kHz / 12.5 fps
    int bos_id      = 128000;
    int eos_id      = 128001;
    int special_ids[8] = {};    // caption/lang/accent/speaker begin+end

    // ---- codec decoder (Qwen3TTSTokenizerV2Decoder) ----
    int cd_layers    = 8;
    int cd_heads     = 16;
    int cd_kv_heads  = 16;
    int cd_head_dim  = 64;
    int cd_hidden    = 512;
    int cd_ff        = 1024;
    int cd_latent    = 1024;
    int cd_dec_dim   = 1536;
    int cd_cb_dim    = 512;
    int cd_cb_size   = 2048;
    int cd_sem_size  = 4096;
    int cd_window    = 72;
    float cd_eps     = 1e-5f;
    float cd_theta   = 10000.0f;
    std::vector<int> upsample_rates    = {8, 5, 4, 3};
    std::vector<int> upsampling_ratios = {2, 2};

    // ---- codec encoder (Qwen3TTSTokenizerV2Encoder / Mimi) ----
    int ce_layers   = 8;
    int ce_heads    = 8;
    int ce_kv_heads = 8;
    int ce_head_dim = 64;
    int ce_hidden   = 512;
    int ce_ff       = 2048;
    int ce_window   = 250;
    int ce_filters  = 64;
    int ce_compress = 2;
    int ce_valid_q  = 16;       // semantic + 15 acoustic quantizers
    float ce_eps    = 1e-5f;
    float ce_theta  = 10000.0f;
    std::vector<int> enc_ratios = {4, 5, 6, 8};  // forward-order downsample
};

// ---------------------------------------------------------------------------
// llama block (LlamaDecoderLayer): RMSNorm pre-norm, no-bias projections,
// SwiGLU MLP. Shared shape for backbone / MTP decoder / codec transformers
// (codec adds per-block LayerScale gammas).
// ---------------------------------------------------------------------------
struct LlamaBlock {
    ggml_tensor *ln1_w = nullptr;    // input_layernorm
    ggml_tensor *q_w = nullptr, *k_w = nullptr, *v_w = nullptr, *o_w = nullptr;
    ggml_tensor *ln2_w = nullptr;    // post_attention_layernorm
    ggml_tensor *gate_w = nullptr, *up_w = nullptr, *down_w = nullptr;
    // codec transformer extras (null for the LM blocks)
    ggml_tensor *attn_gamma = nullptr, *mlp_gamma = nullptr;
};

// Mimi-style encoder transformer layer: LayerNorm (with bias), fc1/fc2 MLP.
struct MimiBlock {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *q_w = nullptr, *k_w = nullptr, *v_w = nullptr, *o_w = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc2_w = nullptr;
    ggml_tensor *attn_gamma = nullptr, *mlp_gamma = nullptr;
};

// Causal Conv1d (channels-first weight ne=[k, in_c, out_c]; depthwise
// kernels are ne=[k, 1, C] with dw=true).
struct Conv1D {
    ggml_tensor *w = nullptr;
    ggml_tensor *b = nullptr;
    int k = 0, in_c = 0, out_c = 0, stride = 1, dilation = 1;
    bool dw = false;
};

// SnakeBeta activation params.
struct SnakeAct {
    ggml_tensor *alpha = nullptr;
    ggml_tensor *beta = nullptr;
};

// Codec decoder weights.
struct CodecDecoderWeights {
    Conv1D pre_conv;                       // 512 → 1024, k3
    ggml_tensor *tf_input_proj_w = nullptr, *tf_input_proj_b = nullptr;
    ggml_tensor *tf_output_proj_w = nullptr, *tf_output_proj_b = nullptr;
    ggml_tensor *tf_norm_w = nullptr;      // final RMSNorm
    std::vector<LlamaBlock> tf_layers;
    // upsample: 2 × [transconv (k=s=factor) + ConvNeXt]
    struct Upstage {
        Conv1D transconv;
        Conv1D dwconv;                     // depthwise k7 (groups=dim)
        ggml_tensor *norm_w = nullptr, *norm_b = nullptr;  // LayerNorm
        ggml_tensor *pw1_w = nullptr, *pw1_b = nullptr;
        ggml_tensor *pw2_w = nullptr, *pw2_b = nullptr;
        ggml_tensor *gamma = nullptr;
    };
    std::vector<Upstage> upstage;
    Conv1D decoder_in;                     // 1024 → 1536, k7
    struct DecoderBlock {
        SnakeAct act0;                     // SnakeBeta(in_dim)
        Conv1D transconv;                  // in→out, k=2*rate, s=rate
        struct RU {
            SnakeAct act1, act2;
            Conv1D conv1;                  // k7, dilation d
            Conv1D conv2;                  // k1
        };
        RU ru[3];                          // dilations 1, 3, 9
    };
    std::vector<DecoderBlock> blocks;
    SnakeAct act_final;
    Conv1D decoder_out;                    // 96 → 1, k7
    // RVQ (embedding tables normalized by cluster_usage at Load)
    ggml_tensor *sem_embed = nullptr;      // [256, sem_size]
    ggml_tensor *sem_out_proj_w = nullptr, *sem_out_proj_b = nullptr;
    std::vector<ggml_tensor *> acou_embed; // 15 × [256, 2048]
    ggml_tensor *acou_out_proj_w = nullptr, *acou_out_proj_b = nullptr;
};

// Codec encoder (Mimi): SEANet + transformer + downsample + quantizer.
struct CodecEncoderWeights {
    Conv1D stem;                           // 64 ← 1, k7
    struct Stage {
        Conv1D res1;                       // C → C/2, k3
        Conv1D res2;                       // C/2 → C, k1
        Conv1D down;                       // C → 2C, k=2*stride, s=stride
    };
    std::vector<Stage> stages;             // strides 4,5,6,8 (64→1024)
    Conv1D final_conv;                     // 1024 → 512, k3
    Conv1D downsample;                     // 512 → 512, k4 s2 (post-transformer)
    std::vector<MimiBlock> tf_layers;
    ggml_tensor *tf_norm_w = nullptr, *tf_norm_b = nullptr;  // final LayerNorm
    // quantizer
    ggml_tensor *sem_in_proj_w = nullptr, *sem_out_proj_w = nullptr;
    ggml_tensor *sem_embed = nullptr;      // [256, sem_size]
    ggml_tensor *acou_in_proj_w = nullptr, *acou_out_proj_w = nullptr;
    std::vector<ggml_tensor *> acou_embed; // 15 × [256, 2048]
};

// ---------------------------------------------------------------------------
// llama3 byte-level BPE (vocab + merges shipped in the GGUF; mirrors the
// HF `tokenizers` llama3 template: regex pre-tokenization + byte-level BPE
// with special-token splitting).
// ---------------------------------------------------------------------------
struct BPETokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merges;  // "a\x01b" -> rank
    std::string byte2str[256];                        // GPT-2 byte encoder
    int bos_id = 128000;
    int eos_id = 128001;
    bool loaded = false;

    void init_byte_encoder();
    bool load_from_gguf(struct gguf_context *gguf);
    // Encode with BOS ... EOS template (matches Tokenizer.encode().ids).
    std::vector<int32_t> encode(const std::string &text) const;
    // Raw BPE without the template (special tokens split verbatim).
    std::vector<int32_t> encode_raw(const std::string &text) const;
};

// ---------------------------------------------------------------------------
// Per-request state
// ---------------------------------------------------------------------------
struct State : public RSState {
    // Inputs.
    std::string text;
    std::string language = "Chinese";
    std::string ref_text;
    std::vector<float> ref_audio;   // any rate; resampled to 24k
    int ref_sr = 24000;
    bool ref_ready = false;

    // Intermediates.
    std::vector<int32_t> text_ids;
    std::vector<int32_t> prompt_codes;  // [T_prompt × 16] interleaved frames
    int T_prompt = 0;
    std::vector<int32_t> gen_codes;     // [T_gen × 16] interleaved frames
    int T_gen = 0;
    bool generated = false;

    // Outputs.
    std::vector<float> audio_output;    // 24 kHz f32 (prompt cut, tail trimmed)
};

// ---------------------------------------------------------------------------
// Public model class
// ---------------------------------------------------------------------------
class Model : public ISpeechModel {
public:
    Model();
    ~Model() override;

    bool Load(const std::unique_ptr<rs_context_t> &ctx,
              ggml_backend_t backend) override;
    std::shared_ptr<RSState> CreateState() override;

    // Encode: reference-audio codec encode + LM prefill + frame loop.
    bool Encode(const std::vector<float> &input_frames, RSState &state,
                ggml_backend_sched_t sched) override;
    // Decode: codec decode of prompt+generated codes → PCM.
    bool Decode(RSState &state, ggml_backend_sched_t sched) override;

    std::string GetTranscription(RSState &state) override { (void)state; return ""; }
    const RSModelMeta &GetMeta() const override { return meta_; }

    bool PushText(RSState &state, const char *text, const char *language,
                  const char *instruct) override;
    bool PushReferenceAudio(RSState &state, const float *samples, int n_samples,
                            int sample_rate, ggml_backend_sched_t sched) override;
    bool PushReferenceText(RSState &state, const char *ref_text) override;
    int GetAudioOutput(RSState &state, float **out_data) override;
    void SetSeed(uint64_t seed) override { seed_ = seed; }

private:
    // graph input/output plumbing (shared with m5tts_lm.cpp / m5tts_codec.cpp)
    static bool SetInputF32(ggml_cgraph *gf, const char *name, const float *d,
                            size_t n);
    static bool SetInputI32(ggml_cgraph *gf, const char *name, const int32_t *d,
                            size_t n);
    static bool AllocGraph(ggml_backend_sched_t sched, ggml_cgraph *gf,
                           const char *what);
    static bool ComputeGraph(ggml_backend_sched_t sched, ggml_cgraph *gf,
                             const char *what);

    bool LoadHParams(struct gguf_context *g);
    bool MapTensors(struct ggml_context *gguf_data, gguf_context *g);
    bool BuildCaches(ggml_backend_t backend);
    // normalize RVQ embedding tables: E = embedding_sum / max(usage, eps)
    void BuildNormalizedCodebooks(ggml_backend_sched_t sched);

    // Reference audio preprocessing (24 kHz, peak-normalized, trimmed).
    static std::vector<float> PrepareReference(const std::vector<float> &pcm,
                                               int sr, int target_sr);
    // Conditioning prefix + prompt transcript + target text.
    std::string ConditionText(const std::string &ref_text,
                              const std::string &text,
                              const std::string &lang) const;
    // Sampling (mirrors onnx/graphs.py DepthFrame.sample).
    int SampleLogits(const float *logits, int n, bool allow_eos,
                     std::mt19937 &rng) const;

    // ---- LM graphs (m5tts_lm.cpp) ----
    bool Prefill(ggml_backend_sched_t sched, const int32_t *text_ids, int Tt,
                 const int32_t *prompt_codes, int Tp,
                 std::vector<float> &last_hidden,
                 std::vector<float> &c0_logits, int &n_past);
    bool BackboneStep(ggml_backend_sched_t sched, const float *embed,
                      int pos, std::vector<float> &last_hidden,
                      std::vector<float> &c0_logits);
    // One codebook of the MTP depth decoder.
    //   step 0: two tokens [projection(last_hidden) ‖ projection(emb(c0))]
    //   step k: one token projection(emb(c_{k-1} + (k-1)*2051))
    // Inputs: codes = the codebook token(s) already sampled (c0 for step 0,
    // c_{k-1} for step k); first_proj = projection(last_hidden) for step 0.
    bool DepthStep(ggml_backend_sched_t sched, int cb_idx, int n_in,
                   const float *first_proj, const int32_t *codes,
                   int pos_base, std::vector<float> &logits);
    bool NextHidden(ggml_backend_sched_t sched, const int32_t *codes,
                    std::vector<float> &out);

    // ---- codec graphs (m5tts_codec.cpp) ----
    bool EncodeReference(ggml_backend_sched_t sched, const float *pcm, int n,
                         std::vector<int32_t> &codes, int &T);
    bool DecodeCodes(ggml_backend_sched_t sched, const int32_t *codes, int T,
                     std::vector<float> &pcm);
    bool DecodeChunk(ggml_backend_sched_t sched, const int32_t *codes, int n,
                     std::vector<float> &pcm_out);
    void ResetCodecState();

private:
    RSModelMeta meta_;
    HParams hp_;
    uint64_t seed_ = 1234;
    std::mt19937 rng_{1234};

    BPETokenizer bpe_;
    std::unique_ptr<rs::WeTextNormalizer> tn_;
    bool tn_tried_ = false;

    // ---- LM weights ----
    ggml_tensor *text_emb_ = nullptr;
    ggml_tensor *audio_emb_ = nullptr;
    ggml_tensor *projection_w_ = nullptr;
    ggml_tensor *codebook0_head_ = nullptr;
    ggml_tensor *audio_head_ = nullptr;    // ne=[2051, 512, 15]
    ggml_tensor *bb_norm_w_ = nullptr;
    ggml_tensor *bb_inv_freq_ = nullptr;   // [32] llama3-scaled
    std::vector<LlamaBlock> bb_layers_;
    ggml_tensor *dc_norm_w_ = nullptr;
    ggml_tensor *dc_inv_freq_ = nullptr;
    std::vector<LlamaBlock> dc_layers_;

    // ---- codec weights ----
    ggml_tensor *cd_inv_freq_ = nullptr;
    ggml_tensor *ce_inv_freq_ = nullptr;
    CodecDecoderWeights cdw_;
    CodecEncoderWeights cew_;

    // ---- persistent caches (allocated at Load) ----
    ggml_context *ctx_kv_ = nullptr;
    ggml_backend_buffer_t buf_kv_ = nullptr;
    ggml_tensor *bb_kv_k_ = nullptr, *bb_kv_v_ = nullptr;
    ggml_tensor *dc_kv_k_ = nullptr, *dc_kv_v_ = nullptr;
    ggml_tensor *cd_kv_k_ = nullptr, *cd_kv_v_ = nullptr;
    struct ConvState {
        std::string name;
        ggml_tensor *t = nullptr;
        int C = 0, S = 0;  // channels, history samples
    };
    std::vector<ConvState> conv_states_;
    int cd_offset_ = 0;                    // total frames fed to codec decoder
    ggml_tensor *sc_eps5_ = nullptr;       // persistent scalar 1e-5
    ggml_tensor *sc_eps9_ = nullptr;       // persistent scalar 1e-9

    // RVQ embedding tables (raw) + usage stats → normalized copies
    std::vector<ggml_tensor *> rvq_tables_;   // raw embedding_sum
    std::vector<ggml_tensor *> rvq_usage_;    // cluster_usage
    std::vector<ggml_tensor *> rvq_norm_;     // normalized (built at Load)

    ggml_backend_t backend_ = nullptr;
    std::vector<ggml_backend_t> backends_;
};

}  // namespace m5tts
