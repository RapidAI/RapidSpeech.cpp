#pragma once

#include <cstdint>
#include <string>

// ============================================
// Architecture Types
// ============================================

/**
 * Model architecture
 */
enum llm_arch {
  LLM_ARCH_UNKNOWN = 0,
  LLM_ARCH_LLAMA,
  LLM_ARCH_QWEN2,
  LLM_ARCH_QWEN3,
  LLM_ARCH_COUNT
};

inline std::string llm_arch_to_string(llm_arch arch) {
  switch (arch) {
  case LLM_ARCH_LLAMA:
    return "llama";
  case LLM_ARCH_QWEN2:
    return "qwen2";
  case LLM_ARCH_QWEN3:
    return "qwen3";
  default:
    return "unknown";
  }
}

// ============================================
// Vocabulary Types
// ============================================

/**
 * Vocabulary type
 */
enum llm_vocab_type {
  LLM_VOCAB_TYPE_BPE,    // Byte-pair encoding
  LLM_VOCAB_TYPE_WPM,    // Word-piece model
  LLM_VOCAB_TYPE_UNIGRAM // Unigram
};

/**
 * Vocabulary preprocessing type
 */
enum llm_vocab_pre_type {
  LLM_VOCAB_PRE_TYPE_DEFAULT,
  LLM_VOCAB_PRE_TYPE_LLAMA,
  LLM_VOCAB_PRE_TYPE_QWEN
};

// ============================================
// Sequence and Position Types
// ============================================

/**
 * Sequence ID type
 */
using llm_seq_id = int32_t;

/**
 * Position type (can be int32 or int64 depending on context length)
 */
using llm_pos = int32_t;

// ============================================
// Invalid/Special Values
// ============================================

constexpr llm_seq_id LLM_SEQ_ID_NONE = -1;
constexpr llm_pos LLM_POS_NONE = -1;
