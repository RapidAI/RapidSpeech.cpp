#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include "utils/rs_wav.h"
/**
 * SenseVoice hyperparameters structure
 */
struct SenseVoiceHParams {
  int32_t n_vocab = 0;
  int32_t n_encoder_hidden_state = 512;
  int32_t n_encoder_linear_units = 2048;
  int32_t n_encoder_attention_heads = 8;
  int32_t n_encoder_layers = 50;
  int32_t n_tp_encoder_layers = 10;
  int32_t n_mels = 80;
  int32_t feats_dim = 560;
  int32_t fsmn_kernel_size = 11;
  float eps = 1e-5f;
};

/**
 * SenseVoice vocabulary structure
 */
struct SenseVoiceVocab {
  std::unordered_map<int, std::string> id_to_token;
  int n_vocab = 0;
};

/**
 * SANM encoder layer structure
 */
struct SenseVoiceLayerEncoder {
  struct ggml_tensor * e_attn_ln_q_w, * e_attn_ln_q_b;
  struct ggml_tensor * e_attn_ln_k_w, * e_attn_ln_k_b;
  struct ggml_tensor * e_attn_ln_v_w, * e_attn_ln_v_b;
  struct ggml_tensor * e_attn_fsmn_w;
  struct ggml_tensor * e_attn_ln_out_w, * e_attn_ln_out_b;
  struct ggml_tensor * e_mlp_w1, * e_mlp_b1;
  struct ggml_tensor * e_mlp_w2, * e_mlp_b2;
  struct ggml_tensor * e_norm_w1, * e_norm_b1;
  struct ggml_tensor * e_norm_w2, * e_norm_b2;
};

/**
 * Collection of SenseVoice model tensors
 */
struct SenseVoiceEncoder {
  struct ggml_tensor * embedding;
  SenseVoiceLayerEncoder encoder0;
  std::vector<SenseVoiceLayerEncoder> encoders_layer;
  std::vector<SenseVoiceLayerEncoder> tp_encoders_layer;
  struct ggml_tensor * e_after_norm_w, * e_after_norm_b;
  struct ggml_tensor * e_tp_norm_w, * e_tp_norm_b;
  struct ggml_tensor * ctc_out_linear_weight, * ctc_out_linear_bias;
};

class SenseVoiceModel : public ISpeechModel {
public:
  SenseVoiceModel();
  virtual ~SenseVoiceModel();

  // Implement base class interface
  bool Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;

  // Fixed: added ggml_backend_sched_t to match ISpeechModel interface
  bool Encode(const std::vector<float>& input_frames, RSState& state, ggml_backend_sched_t sched) override;
  bool Decode(RSState& state, ggml_backend_sched_t sched) override;
  std:: string GetTranscription(RSState& state) override;

  const RSModelMeta& GetMeta() const override { return meta_; }

private:
  RSModelMeta meta_;
  SenseVoiceHParams hparams_;
  SenseVoiceVocab vocab_;
  std::unique_ptr<SenseVoiceEncoder> encoder_;
  struct ggml_context * ctx_weights_;

  bool MapTensors(std::map<std::string, struct ggml_tensor*>& tensors);
  bool SetLayerWeights(std::vector<SenseVoiceLayerEncoder>& layers,
                       std::map<std::string, struct ggml_tensor*>& tensors,
                       int n_layers, const std::string& prefix);
};