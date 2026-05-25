#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "frontend/audio_processor.h"
#include "rapidspeech.h"   // RS_API
#include <deque>
#include <memory>
#include <string>
#include <vector>

/**
 * FireRedVAD (DFSMN) — SOTA streaming/offline VAD.
 *
 * Architecture: 1 initial FSMN + (R-1) DFSMN blocks (with skip) + M DNN layers
 *               + Linear output -> sigmoid.
 * Frame rate: 100 Hz (10 ms shift), 25 ms frame, 80-mel fbank + CMVN.
 *
 * GGUF tensor layout assumptions (matching scripts/convert_fireredvad_to_gguf.py):
 *   Linear W: PT [out, in] stored C-order; GGML ne=[in, out]; mul_mat(W, x) == F.linear(x, W).
 *   Depthwise Conv1d W: PT [P, 1, N1]; GGML ne=[N1, 1, P]; matches ggml_conv_1d_dw.
 */

struct FireRedVadHParams {
    int sample_rate = 16000;
    int frame_length_ms = 25;
    int frame_shift_ms = 10;

    int idim = 80;
    int odim = 1;
    int R = 8;
    int M = 1;
    int H = 256;
    int P = 128;
    int N1 = 20;
    int S1 = 1;
    int N2 = 0;
    int S2 = 0;
    int lookback_padding = 19;  // (N1-1) * S1

    // Postprocessor defaults (mirrors FireRedStreamVadConfig in stream_vad.py)
    float speech_threshold = 0.5f;
    int smooth_window_size = 5;
    int pad_start_frame = 5;
    int min_speech_frame = 8;
    int max_speech_frame = 2000;
    int min_silence_frame = 20;
    int chunk_max_frame = 30000;
};

/**
 * One DFSMN block weights (the R-1 blocks after the initial FSMN).
 *   fc1: Linear(P -> H) + ReLU
 *   fc2: Linear(H -> P) (no bias)
 *   fsmn: depthwise Conv1d lookback (+ optional lookahead)
 *   skip connection adds inputs to output.
 */
struct FireRedDFSMNBlock {
    struct ggml_tensor* fc1_w = nullptr;
    struct ggml_tensor* fc1_b = nullptr;
    struct ggml_tensor* fc2_w = nullptr;
    struct ggml_tensor* lookback_w = nullptr;
    struct ggml_tensor* lookahead_w = nullptr;  // optional (N2 > 0)
};

/** One DNN layer (Linear + ReLU + Dropout in PT; Dropout no-op at inference). */
struct FireRedDNNLayer {
    struct ggml_tensor* w = nullptr;
    struct ggml_tensor* b = nullptr;
};

struct FireRedVadWeights {
    // CMVN
    struct ggml_tensor* cmvn_means = nullptr;
    struct ggml_tensor* cmvn_inv_std = nullptr;

    // Initial fc1, fc2 (the "fsmn1" prelude before R-1 blocks)
    struct ggml_tensor* fc1_w = nullptr;
    struct ggml_tensor* fc1_b = nullptr;
    struct ggml_tensor* fc2_w = nullptr;
    struct ggml_tensor* fc2_b = nullptr;

    // Initial FSMN (no skip)
    struct ggml_tensor* fsmn1_lookback_w = nullptr;
    struct ggml_tensor* fsmn1_lookahead_w = nullptr;  // optional

    // R-1 DFSMN blocks (with skip)
    std::vector<FireRedDFSMNBlock> blocks;

    // M DNN layers
    std::vector<FireRedDNNLayer> dnns;

    // Output linear
    struct ggml_tensor* out_w = nullptr;
    struct ggml_tensor* out_b = nullptr;
};

/** Per-frame VAD postprocessor output (mirrors Python StreamVadFrameResult). */
struct StreamVadFrameResult {
    int frame_idx = 0;            // 1-based
    bool is_speech = false;
    float raw_prob = 0.0f;
    float smoothed_prob = 0.0f;
    bool is_speech_start = false;
    bool is_speech_end = false;
    int speech_start_frame = -1;  // 1-based
    int speech_end_frame = -1;    // 1-based
};

enum class VadFsmState : int {
    SILENCE = 0,
    POSSIBLE_SPEECH = 1,
    SPEECH = 2,
    POSSIBLE_SILENCE = 3,
};

/**
 * FireRedVAD runtime state.
 *   caches[r] holds the last `lookback_padding * P` floats of FSMN input,
 *   stored as ne=[LP, P] row-major (GGML view). Length R (1 initial + R-1 blocks).
 *   audio_remainder: leftover PCM samples from previous streaming chunk
 *     (< frame_size), to be prepended in the next call.
 *   Postprocessor state mirrors StreamVadPostprocessor (Python).
 */
struct FireRedVadState : public RSState {
    // FSMN caches (one per FSMN layer)
    std::vector<std::vector<float>> caches;

    // Streaming fbank remainder (samples not yet consumed by a full window)
    std::vector<float> audio_remainder;

    // Last forward's per-frame probabilities
    std::vector<float> last_probs;

    // Per-frame postprocessor results from latest Encode/DetectStreamingChunk
    std::vector<StreamVadFrameResult> last_results;

    // ---- Postprocessor state (mirrors StreamVadPostprocessor) ----
    int frame_cnt = 0;
    std::deque<float> smooth_window;
    float smooth_window_sum = 0.0f;
    VadFsmState fsm_state = VadFsmState::SILENCE;
    int speech_cnt = 0;
    int silence_cnt = 0;
    bool hit_max_speech = false;
    int last_speech_start_frame = -1;
    int last_speech_end_frame = -1;

    // VAD output flag (for ISpeechModel::GetTranscription)
    bool triggered = false;
    float speech_prob = 0.0f;
};

class RS_API FireRedVadModel : public ISpeechModel {
public:
    FireRedVadModel();
    ~FireRedVadModel() override;

    bool Load(const std::unique_ptr<rs_context_t>& ctx,
              ggml_backend_t backend) override;

    bool LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                    ggml_backend_t backend);

    std::shared_ptr<RSState> CreateState() override;

    // Encode treats `input_frames` as raw 16 kHz mono PCM and runs the full
    // pipeline (fbank+CMVN -> DFSMN -> postprocessor). The latest per-frame
    // results are stashed on the state for retrieval via the helpers below.
    bool Encode(const std::vector<float>& input_frames, RSState& state,
                ggml_backend_sched_t sched) override;
    bool Decode(RSState& state, ggml_backend_sched_t sched) override;
    std::string GetTranscription(RSState& state) override;
    const RSModelMeta& GetMeta() const override { return meta_; }

    // ---- VAD-specific helpers ----

    /** Reset all streaming state (caches + postprocessor + remainder). */
    void Reset(RSState& state);

    /** Process a streaming PCM chunk and return the per-frame results. */
    std::vector<StreamVadFrameResult>
    DetectStreamingChunk(const std::vector<float>& pcm, RSState& state);

    /** One-shot detection over a whole utterance. Returns frame results +
     *  collapsed (start, end) timestamps in seconds. */
    struct FullResult {
        std::vector<StreamVadFrameResult> frames;
        std::vector<std::pair<float, float>> timestamps;  // (start_s, end_s)
        float duration = 0.0f;
    };
    FullResult DetectFull(const std::vector<float>& pcm);

    /** Switch between built-in permissiveness modes (0..3), like Python set_mode(). */
    void SetMode(int mode);

    void SetThreshold(float t) { hparams_.speech_threshold = t; }
    float GetThreshold() const { return hparams_.speech_threshold; }

    bool IsSpeechTriggered(RSState& state) {
        return dynamic_cast<FireRedVadState&>(state).triggered;
    }

    float GetSpeechProbability(RSState& state) {
        return dynamic_cast<FireRedVadState&>(state).speech_prob;
    }

    /** Convert a frame-result sequence into (start_s, end_s) seconds segments
     *  (mirrors Python results_to_timestamps). */
    static std::vector<std::pair<float, float>>
    ResultsToTimestamps(const std::vector<StreamVadFrameResult>& results,
                        int frame_per_second = 100);

    ggml_backend_t GetBackend() { return backend_; }

private:
    RSModelMeta meta_;
    FireRedVadHParams hparams_;
    FireRedVadWeights weights_;
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<AudioProcessor> audio_processor_;

    bool MapTensors(ggml_context* gguf_data);
    bool LoadHParams(gguf_context* ctx_gguf);
    void InitAudioProcessor();

    // Run DFSMN forward on a [T, idim] fbank+CMVN feature tensor. Writes
    // per-frame probabilities into `out_probs` (size T) and updates the
    // state caches.
    bool Forward(const std::vector<float>& feat, int T,
                 FireRedVadState& state, std::vector<float>& out_probs);

    // Apply CMVN in-place: out = (in - means) * inv_std.
    void ApplyCmvn(std::vector<float>& feat, int T) const;

    // Postprocessor: process one raw probability and update state machine.
    StreamVadFrameResult ProcessOneFrame(float raw_prob, FireRedVadState& s) const;
};
