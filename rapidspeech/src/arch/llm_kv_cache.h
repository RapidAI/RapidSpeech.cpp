#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "llm_types.h"
#include <map>
#include <memory>
#include <vector>

// ============================================
// Simplified KV Cache for RapidSpeech
// ============================================

/**
 * Simplified KV cache management for decoder-only LLMs
 *
 * Features:
 * - Continuous batching
 * - Sequence management (copy, remove, keep)
 * - Optional RoPE shift for long context
 *
 * Usage:
 * 1. Create llm_kv_cache with model parameters
 * 2. For each batch:
 *    a. Call prepare() to find slots for new tokens
 *    b. Build compute graph using get_k/get_v
 *    c. Execute graph
 *    d. Call update() to commit new KV states
 */
class llm_kv_cache {
public:
  // Configuration
  struct config {
    uint32_t n_ctx = 512;   // Total context size
    uint32_t n_seq_max = 1; // Max concurrent sequences
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    bool offload = true;  // Offload to GPU
    bool unified = false; // Use unified memory
    uint32_t n_pad = 1;   // Padding for alignment
  };

  // Slot information for batch processing
  struct slot_info {
    std::vector<llm_seq_id> seq_ids;  // Sequence ID for each token
    std::vector<uint32_t> positions;  // Position in sequence
    std::vector<uint32_t> kv_indices; // KV cache indices

    uint32_t n_tokens() const { return seq_ids.size(); }
    bool empty() const { return seq_ids.empty(); }
    void clear() {
      seq_ids.clear();
      positions.clear();
      kv_indices.clear();
    }
  };

  llm_kv_cache(const config &cfg,
               uint32_t n_embd_k_gqa, // K dimension per head * n_head_kv
               uint32_t n_embd_v_gqa, // V dimension per head * n_head_kv
               uint32_t n_head_kv,    // Number of KV heads
               ggml_backend_t backend // Backend for allocation
  );

  ~llm_kv_cache();

  // ========================================
  // Sequence Management
  // ========================================

  // Remove tokens from a sequence
  bool seq_rm(llm_seq_id seq_id, llm_pos p0, llm_pos p1);

  // Copy sequence (for beam search)
  void seq_cp(llm_seq_id seq_id_src, llm_seq_id seq_id_dst, llm_pos p0,
              llm_pos p1);

  // Keep sequence (prevent eviction)
  void seq_keep(llm_seq_id seq_id);

  // Clear all sequences
  void clear(bool clear_data = true);

  // ========================================
  // Batch Preparation
  // ========================================

  /**
   * Prepare KV cache for a new batch
   *
   * @param seq_ids Sequence ID for each token in batch
   * @param positions Position in sequence for each token
   * @return slot_info with KV indices, or empty on failure
   */
  slot_info prepare(const std::vector<llm_seq_id> &seq_ids,
                    const std::vector<llm_pos> &positions);

  /**
   * Update cache after batch execution
   * Commits new KV states from the prepare phase
   */
  bool update();

  // ========================================
  // Graph Building Helpers
  // ========================================

  /**
   * Get K tensor view for attention
   *
   * @param ctx GGML context for creating tensor
   * @param il Layer index
   * @param n_kv Number of KV pairs to attend to
   * @param sinfo Slot info from prepare()
   * @return K tensor [n_embd_k_gqa, n_kv]
   */
  ggml_tensor *get_k(ggml_context *ctx, int32_t il, uint32_t n_kv,
                     const slot_info &sinfo) const;

  /**
   * Get V tensor view for attention
   */
  ggml_tensor *get_v(ggml_context *ctx, int32_t il, uint32_t n_kv,
                     const slot_info &sinfo) const;

  /**
   * Copy new K values to cache
   *
   * @param ctx GGML context
   * @param k_cur Current K values [n_embd_k_gqa, n_tokens]
   * @param k_idxs Destination indices in cache
   * @param il Layer index
   * @return Copy operation tensor
   */
  ggml_tensor *cpy_k(ggml_context *ctx, ggml_tensor *k_cur, ggml_tensor *k_idxs,
                     int32_t il) const;

  /**
   * Copy new V values to cache
   */
  ggml_tensor *cpy_v(ggml_context *ctx, ggml_tensor *v_cur, ggml_tensor *v_idxs,
                     int32_t il) const;

  // ========================================
  // State Management
  // ========================================

  // Get current cache size (number of used KV cells)
  uint32_t size() const { return used_cells_; }

  // Get total capacity
  uint32_t capacity() const { return config_.n_ctx; }

  // Check if cache can hold more tokens
  bool has_space(uint32_t n_tokens) const {
    return used_cells_ + n_tokens <= config_.n_ctx;
  }

  // Get memory usage in bytes
  size_t memory_size() const;

private:
  config config_;

  // KV cache tensors per layer
  // Shape: [n_embd_k_gqa, n_ctx] for K
  // Shape: [n_embd_v_gqa, n_ctx] for V (if not transposed)
  struct layer_cache {
    ggml_tensor *k = nullptr;
    ggml_tensor *v = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer *buffer = nullptr;
  };
  std::vector<layer_cache> layers_;

  // Cell state tracking
  std::vector<llm_seq_id> cell_seq_ids_; // Sequence ID for each cell
  std::vector<llm_pos> cell_positions_;  // Position for each cell

  // Per-sequence state
  struct sequence_state {
    llm_pos pos_min = 0;  // Minimum position in sequence
    llm_pos pos_max = -1; // Maximum position (invalid if -1)
    bool is_kept = false; // Prevent eviction
  };
  std::map<llm_seq_id, sequence_state> seq_states_;

  // Current batch state
  slot_info current_slot_;
  uint32_t used_cells_ = 0;
  uint32_t search_start_ = 0; // For ring buffer search

  // Backend for allocation
  ggml_backend_t backend_ = nullptr;

  // Helper to find contiguous slot
  bool find_contiguous_slot(uint32_t n_tokens, uint32_t &out_start);
};

// Smart pointer for KV cache
using llm_kv_cache_ptr = std::unique_ptr<llm_kv_cache>;
