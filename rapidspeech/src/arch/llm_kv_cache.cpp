#include "llm_kv_cache.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cstring>
#include <iostream>

#define KV_LOG_INFO(fmt, ...) std::printf("[KV] " fmt "\n", ##__VA_ARGS__)
#define KV_LOG_ERROR(fmt, ...)                                                 \
  std::fprintf(stderr, "[KV] Error: " fmt "\n", ##__VA_ARGS__)

// ============================================
// llm_kv_cache Implementation
// ============================================

llm_kv_cache::llm_kv_cache(const config &cfg, uint32_t n_embd_k_gqa,
                           uint32_t n_embd_v_gqa, uint32_t n_head_kv,
                           ggml_backend_t backend)
    : config_(cfg), backend_(backend) {

  // Initialize cell state tracking
  cell_seq_ids_.resize(config_.n_ctx, -1);
  cell_positions_.resize(config_.n_ctx, 0);

  // Allocate KV tensors per layer
  // For now, we allocate a single layer; actual models will have multiple
  // layers
  layers_.resize(1); // Will be expanded based on model

  for (auto &layer : layers_) {
    // Create context for KV tensors
    struct ggml_init_params params = {
        /*.mem_size   =*/2 * (size_t)config_.n_ctx *
                (n_embd_k_gqa + n_embd_v_gqa) * ggml_type_size(cfg.type_k) +
            (1 << 20),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true};

    layer.ctx = ggml_init(params);
    if (!layer.ctx) {
      throw std::runtime_error("Failed to initialize KV cache context");
    }

    // Create K tensor: [n_embd_k_gqa, n_ctx]
    layer.k =
        ggml_new_tensor_2d(layer.ctx, cfg.type_k, n_embd_k_gqa, config_.n_ctx);

    // Create V tensor: [n_embd_v_gqa, n_ctx]
    layer.v =
        ggml_new_tensor_2d(layer.ctx, cfg.type_v, n_embd_v_gqa, config_.n_ctx);

    // Allocate buffer
    layer.buffer = ggml_backend_alloc_ctx_tensors(layer.ctx, backend);
    if (!layer.buffer) {
      ggml_free(layer.ctx);
      throw std::runtime_error("Failed to allocate KV cache buffer");
    }

    KV_LOG_INFO("Allocated KV cache: K=%zu MB, V=%zu MB",
                ggml_nbytes(layer.k) / (1 << 20),
                ggml_nbytes(layer.v) / (1 << 20));
  }

  KV_LOG_INFO("KV cache initialized: n_ctx=%d, type_k=%d, type_v=%d",
              config_.n_ctx, cfg.type_k, cfg.type_v);
}

llm_kv_cache::~llm_kv_cache() {
  for (auto &layer : layers_) {
    if (layer.buffer) {
      ggml_backend_buffer_free(layer.buffer);
    }
    if (layer.ctx) {
      ggml_free(layer.ctx);
    }
  }
}

void llm_kv_cache::clear(bool clear_data) {
  // Reset cell state
  std::fill(cell_seq_ids_.begin(), cell_seq_ids_.end(), -1);
  std::fill(cell_positions_.begin(), cell_positions_.end(), 0);

  // Clear sequence states
  seq_states_.clear();

  // Reset counters
  used_cells_ = 0;
  search_start_ = 0;
  current_slot_.clear();

  // Zero out data if requested
  if (clear_data && backend_) {
    for (auto &layer : layers_) {
      if (layer.buffer) {
        ggml_backend_buffer_clear(layer.buffer, 0);
      }
    }
  }
}

bool llm_kv_cache::seq_rm(llm_seq_id seq_id, llm_pos p0, llm_pos p1) {
  // Remove tokens in range [p0, p1) from sequence
  bool found = false;

  for (uint32_t i = 0; i < config_.n_ctx; ++i) {
    if (cell_seq_ids_[i] == seq_id && cell_positions_[i] >= p0 &&
        cell_positions_[i] < p1) {
      cell_seq_ids_[i] = -1;
      cell_positions_[i] = 0;
      found = true;
    }
  }

  // Update sequence state
  auto it = seq_states_.find(seq_id);
  if (it != seq_states_.end()) {
    if (p1 > it->second.pos_min) {
      it->second.pos_min = p1;
    }
    if (it->second.pos_max < p0) {
      it->second.pos_max = -1; // Invalid
    }
  }

  return found;
}

void llm_kv_cache::seq_cp(llm_seq_id seq_id_src, llm_seq_id seq_id_dst,
                          llm_pos p0, llm_pos p1) {
  // Copy tokens from source sequence to destination
  std::vector<std::pair<uint32_t, uint32_t>> copies; // (src_idx, dst_idx)

  // Find source cells and allocate destination cells
  for (uint32_t i = 0; i < config_.n_ctx; ++i) {
    if (cell_seq_ids_[i] == seq_id_src && cell_positions_[i] >= p0 &&
        cell_positions_[i] < p1) {

      // Find free cell for destination
      uint32_t dst_idx = config_.n_ctx;
      for (uint32_t j = search_start_; j < config_.n_ctx; ++j) {
        if (cell_seq_ids_[j] == -1) {
          dst_idx = j;
          break;
        }
      }

      if (dst_idx < config_.n_ctx) {
        copies.push_back({i, dst_idx});
      }
    }
  }

  // Perform copies
  for (const auto &[src_idx, dst_idx] : copies) {
    cell_seq_ids_[dst_idx] = seq_id_dst;
    cell_positions_[dst_idx] = cell_positions_[src_idx];

    // Copy tensor data
    for (const auto &layer : layers_) {
      // Copy K
      ggml_backend_tensor_copy(ggml_view_1d(layer.ctx, layer.k, layer.k->ne[0],
                                            src_idx * layer.k->nb[1]),
                               ggml_view_1d(layer.ctx, layer.k, layer.k->ne[0],
                                            dst_idx * layer.k->nb[1]));

      // Copy V
      ggml_backend_tensor_copy(ggml_view_1d(layer.ctx, layer.v, layer.v->ne[0],
                                            src_idx * layer.v->nb[1]),
                               ggml_view_1d(layer.ctx, layer.v, layer.v->ne[0],
                                            dst_idx * layer.v->nb[1]));
    }
  }

  // Update sequence state
  seq_states_[seq_id_dst] = seq_states_[seq_id_src];
}

void llm_kv_cache::seq_keep(llm_seq_id seq_id) {
  auto it = seq_states_.find(seq_id);
  if (it != seq_states_.end()) {
    it->second.is_kept = true;
  }
}

llm_kv_cache::slot_info
llm_kv_cache::prepare(const std::vector<llm_seq_id> &seq_ids,
                      const std::vector<llm_pos> &positions) {
  slot_info info;

  if (seq_ids.size() != positions.size()) {
    KV_LOG_ERROR("seq_ids and positions size mismatch");
    return info;
  }

  uint32_t n_tokens = seq_ids.size();

  // Find contiguous slot
  uint32_t start_idx;
  if (!find_contiguous_slot(n_tokens, start_idx)) {
    KV_LOG_ERROR("No contiguous slot for %d tokens", n_tokens);
    return info;
  }

  // Fill slot info
  info.seq_ids = seq_ids;
  info.positions.resize(positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    info.positions[i] = static_cast<uint32_t>(positions[i]);
  }
  info.kv_indices.resize(n_tokens);

  for (uint32_t i = 0; i < n_tokens; ++i) {
    info.kv_indices[i] = start_idx + i;

    // Update cell state
    cell_seq_ids_[start_idx + i] = seq_ids[i];
    cell_positions_[start_idx + i] = positions[i];

    // Update sequence state
    auto &state = seq_states_[seq_ids[i]];
    state.pos_min = std::min(state.pos_min, positions[i]);
    state.pos_max = std::max(state.pos_max, static_cast<llm_pos>(positions[i]));
  }

  used_cells_ = std::max(used_cells_, start_idx + n_tokens);
  search_start_ = start_idx + n_tokens;

  current_slot_ = info;
  return info;
}

bool llm_kv_cache::update() {
  // Commit current slot
  // For simple implementation, this is a no-op since we already updated in
  // prepare() In more complex implementations, this would handle async copies

  current_slot_.clear();
  return true;
}

ggml_tensor *llm_kv_cache::get_k(ggml_context *ctx, int32_t il, uint32_t n_kv,
                                 const slot_info &sinfo) const {
  if (il >= static_cast<int32_t>(layers_.size())) {
    return nullptr;
  }

  const auto &layer = layers_[il];

  // Create view into KV cache
  // For contiguous slot, we can use a simple 2D view
  if (sinfo.kv_indices.size() == n_kv) {
    uint32_t start = sinfo.kv_indices[0];
    return ggml_view_2d(ctx, layer.k, layer.k->ne[0], n_kv, layer.k->nb[1],
                        start * layer.k->nb[1]);
  }

  // For non-contiguous, we need gather operation (TODO)
  return nullptr;
}

ggml_tensor *llm_kv_cache::get_v(ggml_context *ctx, int32_t il, uint32_t n_kv,
                                 const slot_info &sinfo) const {
  if (il >= static_cast<int32_t>(layers_.size())) {
    return nullptr;
  }

  const auto &layer = layers_[il];

  if (sinfo.kv_indices.size() == n_kv) {
    uint32_t start = sinfo.kv_indices[0];
    return ggml_view_2d(ctx, layer.v, layer.v->ne[0], n_kv, layer.v->nb[1],
                        start * layer.v->nb[1]);
  }

  return nullptr;
}

ggml_tensor *llm_kv_cache::cpy_k(ggml_context *ctx, ggml_tensor *k_cur,
                                 ggml_tensor *k_idxs, int32_t il) const {
  if (il >= static_cast<int32_t>(layers_.size())) {
    return nullptr;
  }

  const auto &layer = layers_[il];

  // Use ggml_set_rows to copy k_cur to cache at indices specified by k_idxs
  return ggml_set_rows(ctx, layer.k, k_idxs, k_cur);
}

ggml_tensor *llm_kv_cache::cpy_v(ggml_context *ctx, ggml_tensor *v_cur,
                                 ggml_tensor *v_idxs, int32_t il) const {
  if (il >= static_cast<int32_t>(layers_.size())) {
    return nullptr;
  }

  const auto &layer = layers_[il];

  return ggml_set_rows(ctx, layer.v, v_idxs, v_cur);
}

size_t llm_kv_cache::memory_size() const {
  size_t total = 0;
  for (const auto &layer : layers_) {
    if (layer.buffer) {
      total += ggml_backend_buffer_get_size(layer.buffer);
    }
  }
  return total;
}

bool llm_kv_cache::find_contiguous_slot(uint32_t n_tokens,
                                        uint32_t &out_start) {
  // Simple ring buffer search
  for (uint32_t start = search_start_; start < config_.n_ctx; ++start) {
    bool found = true;

    for (uint32_t i = 0; i < n_tokens && start + i < config_.n_ctx; ++i) {
      if (cell_seq_ids_[start + i] != -1 &&
          !seq_states_[cell_seq_ids_[start + i]].is_kept) {
        found = false;
        break;
      }
    }

    if (found && start + n_tokens <= config_.n_ctx) {
      out_start = start;
      return true;
    }
  }

  // Wrap around search
  for (uint32_t start = 0; start < search_start_ && start < config_.n_ctx;
       ++start) {
    bool found = true;

    for (uint32_t i = 0; i < n_tokens && start + i < config_.n_ctx; ++i) {
      if (cell_seq_ids_[start + i] != -1 &&
          !seq_states_[cell_seq_ids_[start + i]].is_kept) {
        found = false;
        break;
      }
    }

    if (found && start + n_tokens <= config_.n_ctx) {
      out_start = start;
      return true;
    }
  }

  return false;
}
