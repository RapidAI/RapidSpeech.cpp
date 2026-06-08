#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "rapidspeech.h" // RS_API
#include "sensevoice_encoder.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * SenseVoice vocabulary structure
 */
struct SenseVoiceVocab {
  std::unordered_map<int, std::string> id_to_token;
  int n_vocab = 0;
};

struct SenseVoiceDecoder {
  struct ggml_tensor *ctc_out_linear_weight, *ctc_out_linear_bias;
};

class RS_API SenseVoiceModel : public ISpeechModel {
public:
  SenseVoiceModel();
  virtual ~SenseVoiceModel();

  // Implement base class interface
  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;

  // Fixed: added ggml_backend_sched_t to match ISpeechModel interface
  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override;
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;
  std::string GetTranscription(RSState &state) override;

  const RSModelMeta &GetMeta() const override { return meta_; }

  // KWS hook: when enabled, Decode() copies full log-prob matrix into the
  // state's kws_log_probs (row-major [T, V]) instead of producing tokens.
  // Returned pointer is owned by the state and valid until the next Decode().
  void SetKWSMode(RSState &state, bool enable) const;
  const float *GetCTCLogits(const RSState &state, int *T, int *V) const;
  const std::unordered_map<int, std::string> &GetIdToToken() const {
    return vocab_.id_to_token;
  }
  int GetBlankId() const { return 0; }

  void set_imatrix_callback(std::function<void(struct ggml_tensor *)> cb) {
    if (encoder_) encoder_->set_imatrix_callback(std::move(cb));
  }

private:
  RSModelMeta meta_;
  SenseVoiceHParams hparams_;
  SenseVoiceVocab vocab_;
  std::unique_ptr<SenseVoiceEncoderModel> encoder_;
  std::unique_ptr<SenseVoiceDecoder> decoder_;
  struct ggml_context *ctx_weights_;

  bool MapTensors(std::map<std::string, struct ggml_tensor *> &tensors);
  bool SetLayerWeights(std::vector<SenseVoiceLayerEncoder> &layers,
                       std::map<std::string, struct ggml_tensor *> &tensors,
                       int n_layers, const std::string &prefix);
};