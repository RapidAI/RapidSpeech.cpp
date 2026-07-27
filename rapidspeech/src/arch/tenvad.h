#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "rapidspeech.h"   // RS_API
#include <vector>
#include <string>

/**
 * TEN-VAD hyperparameters (loaded from GGUF metadata).
 *
 * Frame rate: 62.5 fps (hop 256 samples @ 16 kHz = 16 ms).
 */
struct TenVadHParams {
    int sample_rate = 16000;
    int hop_size = 256;          // 16 ms at 16 kHz
    int context_frames = 3;      // model input = 3-frame feature stack
    int feature_dim = 41;        // 40 log-mel + 1 pitch frequency
    int hidden_size = 64;        // LSTM hidden (x2 layers, h+c each)
    int reset_frames = 1875;     // zero LSTM states every ~30 s (upstream behavior)
    float threshold = 0.5f;
    int min_silence_ms = 200;
};

/**
 * Runtime state for TEN-VAD inference.
 *
 * Owns the per-stream DSP handles from the vendored ten-vad frontend
 * (third_party/ten-vad): the STFT analyzer keeps a 768-sample window FIFO
 * across 256-sample hops, and the pitch estimator carries LPC/tracking state.
 * The 4 LSTM state vectors (h/c for both layers) live in host memory and are
 * uploaded/read back per frame like SileroVadState.
 */
struct RS_API TenVadState : public RSState {
    // DSP handles (AUP_Analyzer / AUP_PE from third_party/ten-vad)
    void* analyzer = nullptr;
    void* pitch_est = nullptr;

    // Pre-emphasis carry-over: previous raw sample (int16 scale)
    float time_signal_pre = 0.0f;

    // Feature context stack [context_frames x feature_dim], oldest first
    std::vector<float> feat_stack;

    // LSTM states: h1, c1, h2, c2 (hidden_size each)
    std::vector<float> h1, c1, h2, c2;

    // Pending samples (int16 scale) not yet forming a full hop
    std::vector<float> remainder;

    // Frames since last periodic LSTM reset
    int frames_since_reset = 0;

    // Last computed speech probability
    float speech_prob = 0.0f;

    // VADIterator state machine (mirrors SileroVadState)
    bool triggered = false;
    int temp_end_sample = 0;
    int current_sample = 0;

    TenVadState() = default;
    ~TenVadState() override;

    TenVadState(const TenVadState&) = delete;
    TenVadState& operator=(const TenVadState&) = delete;
};

/**
 * TEN-VAD weights mapped from GGUF tensors.
 *
 * GGUF data layout (matching scripts/convert_tenvad_to_gguf.py):
 *   conv1.dw.weight [1,1,3,3]  -> ggml ne=[3,3,1,1]  (ggml_conv_2d kernel)
 *   conv1.pw.weight [16,1,1]   -> ggml ne=[1,1,16]   (ggml_conv_1d, K=1)
 *   conv{2,3}.dw.weight [16,1,3] -> ggml ne=[3,1,16] (ggml_conv_1d_dw)
 *   conv{2,3}.pw.weight [16,16,1] -> ggml ne=[1,16,16]
 *   lstm{1,2}.weight_ih [4H,in] -> ggml ne=[in,4H]; gates pre-ordered i,f,g,o
 *   fc{1,2}.weight [out,in]    -> ggml ne=[in,out]; mul_mat(W,x)==F.linear
 */
struct TenVadWeights {
    struct ggml_tensor* conv1_dw_w = nullptr;
    struct ggml_tensor* conv1_pw_w = nullptr;
    struct ggml_tensor* conv1_pw_b = nullptr;

    struct ggml_tensor* conv2_dw_w = nullptr;
    struct ggml_tensor* conv2_pw_w = nullptr;
    struct ggml_tensor* conv2_pw_b = nullptr;

    struct ggml_tensor* conv3_dw_w = nullptr;
    struct ggml_tensor* conv3_pw_w = nullptr;
    struct ggml_tensor* conv3_pw_b = nullptr;

    struct ggml_tensor* lstm1_weight_ih = nullptr;
    struct ggml_tensor* lstm1_weight_hh = nullptr;
    struct ggml_tensor* lstm1_bias_ih = nullptr;
    struct ggml_tensor* lstm1_bias_hh = nullptr;

    struct ggml_tensor* lstm2_weight_ih = nullptr;
    struct ggml_tensor* lstm2_weight_hh = nullptr;
    struct ggml_tensor* lstm2_bias_ih = nullptr;
    struct ggml_tensor* lstm2_bias_hh = nullptr;

    struct ggml_tensor* fc1_w = nullptr;
    struct ggml_tensor* fc1_b = nullptr;
    struct ggml_tensor* fc2_w = nullptr;
    struct ggml_tensor* fc2_b = nullptr;
};

/**
 * TEN-VAD model (https://github.com/ten-framework/ten-vad).
 *
 * Frontend (host, ported from ten-vad's aed.cc, DSP modules vendored under
 * third_party/ten-vad): input PCM scaled to int16 range -> pre-emphasis 0.97
 * -> STFT (fft 1024 / win 768 Hann / hop 256) -> power spectrum -> 40 mel
 * (log + mean/std normalize) + pitch-frequency feature -> 3-frame stack.
 *
 * Network (ggml graph, one frame per Encode like SileroVadModel):
 * separable convs -> 2x LSTM(64) with external h/c -> dense -> sigmoid.
 */
class RS_API TenVadModel : public ISpeechModel {
public:
    TenVadModel();
    ~TenVadModel() override;

    bool Load(const std::unique_ptr<rs_context_t>& ctx,
              ggml_backend_t backend) override;

    /** Direct load from bare gguf/ggml contexts (CLI / C-API path). */
    bool LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                    ggml_backend_t backend);

    std::shared_ptr<RSState> CreateState() override;

    /**
     * Encode consumes 16 kHz mono PCM in [-1, 1]. Samples are buffered in the
     * state; every complete 256-sample hop runs the DSP frontend + network.
     * speech_prob reflects the most recent completed hop.
     */
    bool Encode(const std::vector<float>& input_frames, RSState& state,
                ggml_backend_sched_t sched) override;
    bool Decode(RSState& state, ggml_backend_sched_t sched) override;
    std::string GetTranscription(RSState& state) override;
    const RSModelMeta& GetMeta() const override { return meta_; }

    // ---- VAD helpers (same surface as SileroVadModel) ----
    float GetSpeechProbability(RSState& state);

    bool IsSpeechTriggered(RSState& state) {
        return dynamic_cast<TenVadState&>(state).triggered;
    }

    void SetThreshold(float threshold) { hparams_.threshold = threshold; }
    float GetThreshold() const { return hparams_.threshold; }
    int GetHopSize() const { return hparams_.hop_size; }

    /** Reset all streaming state in-place (DSP handles + LSTM + stack). */
    bool ResetState(RSState& state);

    ggml_backend_t GetBackend() { return backend_; }

private:
    RSModelMeta meta_;
    TenVadHParams hparams_;
    TenVadWeights weights_;
    ggml_backend_t backend_ = nullptr;

    // Mel filterbank coefficients [40 x 513], generated at Load
    // (port of aed.cc AUP_Aed_resetVariables mel triangle generation).
    std::vector<float> mel_fb_;

    bool MapTensors(ggml_context* gguf_data);
    bool LoadHParams(gguf_context* ctx_gguf);
    void BuildMelFilterBank();

    // Set up (or re-init) the DSP handles on a state.
    bool InitStateDSP(TenVadState& s);

    // Run frontend + network for exactly one hop (int16-scale samples).
    bool ProcessHop(const float* hop, TenVadState& s);

    // Run the ggml network on the current feature stack; updates h/c + prob.
    bool Forward(TenVadState& s);
};
