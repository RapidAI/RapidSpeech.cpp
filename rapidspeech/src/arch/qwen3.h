#pragma once

#include "llm_graph.h"
#include "llm_model.h"
#include <memory>

/**
 * Qwen3 Graph Builder
 *
 * Implements computation graph for Qwen3 architecture:
 * - RMSNorm
 * - SwiGLU FFN
 * - RoPE embedding
 * - GQA/MQA attention
 *
 * Reusable for TTS/ASR scenarios via llm_build_opts
 */
class llm_build_qwen3 : public llm_graph_builder {
public:
  llm_build_qwen3(const llm_model &model, const llm_cparams &cparams,
                  ggml_backend_sched_t sched);

  ~llm_build_qwen3() override = default;

  /**
   * Build computation graph for Qwen3
   *
   * @param tokens Input token IDs
   * @param n_tokens Number of tokens
   * @param kv_cache KV cache (optional)
   * @param pos Positions (optional)
   * @param opts Build options (optional, controls output mode)
   * @return Graph result
   */
  llm_graph_result_ptr
  build_graph(const int32_t *tokens, uint32_t n_tokens,
              llm_kv_cache *kv_cache = nullptr, const llm_pos *pos = nullptr,
              const llm_build_opts *opts = nullptr) override;

  /**
   * Build computation graph from embeddings (for TTS/ASR)
   *
   * This allows Qwen3 to be reused as a decoder in TTS systems
   * where the input is acoustic embeddings rather than tokens.
   *
   * @param embeds Input embeddings [n_embd, n_tokens]
   * @param n_tokens Number of tokens
   * @param kv_cache KV cache (optional)
   * @param pos Positions (optional)
   * @param opts Build options (optional)
   * @return Graph result
   */
  llm_graph_result_ptr
  build_graph_from_embeds(ggml_tensor *embeds, uint32_t n_tokens,
                          llm_kv_cache *kv_cache = nullptr,
                          const llm_pos *pos = nullptr,
                          const llm_build_opts *opts = nullptr) override;

private:
  const llm_model &model_;

  // Graph building helpers for Qwen3
  ggml_tensor *build_embedding_layer(ggml_context *ctx, const int32_t *tokens,
                                     uint32_t n_tokens);
  ggml_tensor *build_transformer_layer(ggml_context *ctx, ggml_tensor *cur,
                                       int32_t il, llm_kv_cache *kv_cache,
                                       const llm_pos *pos, uint32_t n_tokens);
  ggml_tensor *build_attention_layer(ggml_context *ctx, ggml_tensor *cur,
                                     int32_t il, llm_kv_cache *kv_cache,
                                     const llm_pos *pos, uint32_t n_tokens);
  ggml_tensor *build_ffn_layer(ggml_context *ctx, ggml_tensor *cur, int32_t il);

  // Position management
  ggml_tensor *build_position_tensor(ggml_context *ctx, const llm_pos *pos,
                                     uint32_t n_tokens);

  // Attention mask
  ggml_tensor *build_causal_mask_tensor(ggml_context *ctx, uint32_t n_tokens,
                                        uint32_t n_kv_cache = 0);

  // Helper to check if we should extract intermediate output
  bool should_extract_layer(int32_t il) const;
};

/**
 * Register Qwen3 model loader and graph builder
 */
void llm_register_qwen3();
