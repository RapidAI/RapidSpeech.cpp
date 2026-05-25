#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "rapidspeech.h"  // RS_API

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rapidspeech {

// =====================================================================
// BERT hyperparameters
// =====================================================================

struct BertHParams {
    int hidden_size       = 0;
    int num_layers        = 0;
    int num_heads         = 0;
    int intermediate_size = 0;
    int max_position      = 0;
    int vocab_size        = 0;
    int type_vocab_size   = 2;
    float layer_norm_eps  = 1e-12f;
    // MeloTTS averages the last N hidden states across layers.
    int feature_layers    = 4;
    int head_dim() const { return num_heads ? hidden_size / num_heads : 0; }
};

// =====================================================================
// BERT weights (one entry per encoder layer)
// =====================================================================

struct BertAttentionWeights {
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
};

struct BertFFNWeights {
    ggml_tensor *w1_w = nullptr, *w1_b = nullptr;  // intermediate
    ggml_tensor *w2_w = nullptr, *w2_b = nullptr;  // output
};

struct BertLayerWeights {
    BertAttentionWeights attn;
    ggml_tensor *ln_attn_w = nullptr, *ln_attn_b = nullptr;
    BertFFNWeights       ffn;
    ggml_tensor *ln_ffn_w  = nullptr, *ln_ffn_b  = nullptr;
};

struct BertEmbeddingWeights {
    ggml_tensor *word_emb       = nullptr;  // [H, V]
    ggml_tensor *position_emb   = nullptr;  // [H, max_pos]
    ggml_tensor *token_type_emb = nullptr;  // [H, type_vocab_size] (optional)
    ggml_tensor *ln_w           = nullptr;
    ggml_tensor *ln_b           = nullptr;
};

// =====================================================================
// WordPiece tokenizer (BERT-style)
// =====================================================================

class RS_API WordPieceTokenizer {
public:
    bool LoadFromGGUF(struct gguf_context *gguf);

    // Tokenize a UTF-8 string into BERT subword IDs, automatically inserting
    // [CLS] at the start and [SEP] at the end. Truncates to max_subwords
    // (excluding the two specials) if non-positive uses 510.
    std::vector<int> Tokenize(const std::string &text,
                              int max_subwords = -1,
                              std::vector<int> *out_word_boundaries = nullptr) const;

    int unk_id() const { return unk_id_; }
    int cls_id() const { return cls_id_; }
    int sep_id() const { return sep_id_; }
    int pad_id() const { return pad_id_; }
    int mask_id() const { return mask_id_; }
    bool do_lower_case() const { return do_lower_case_; }

    const std::string &id_to_token(int id) const;
    int token_to_id(const std::string &t) const;
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    int unk_id_  = 100;
    int cls_id_  = 101;
    int sep_id_  = 102;
    int pad_id_  = 0;
    int mask_id_ = 103;
    bool do_lower_case_ = true;

    // Splits the raw text into pre-tokens following HF BasicTokenizer:
    // - lowercase + NFD-strip-accents (if do_lower_case_)
    // - split on whitespace
    // - split off Chinese / CJK code-point characters into their own tokens
    // - split on ASCII punctuation
    // Returns a vector of UTF-8 substrings; out_word_starts (parallel) marks
    // whether each substring starts a new logical "word" (true) or is a
    // continuation/punctuation that nevertheless gets its own pre-token.
    std::vector<std::string> BasicTokenize(const std::string &text) const;

    // Greedy longest-match WordPiece. Returns subword IDs for one pre-token.
    // Unknown chunks emit [UNK]. Continuation pieces are stored as "##xxx".
    void WordPiece(const std::string &pre_token, std::vector<int> &out) const;
};

// =====================================================================
// BertModel
// =====================================================================

class RS_API BertModel {
public:
    BertModel();
    ~BertModel();

    BertModel(const BertModel &) = delete;
    BertModel &operator=(const BertModel &) = delete;

    // Loads a BERT GGUF produced by scripts/convert_bert_to_gguf.py.
    bool LoadFromGGUF(const char *path, bool use_gpu = false);

    // Encodes `text` and returns per-subword features as a flat row-major
    // [n_subwords, hidden] vector (subword fast index). Mean of last
    // `feature_layers` hidden states is taken to match MeloTTS.
    //
    // If out_subword_ids != nullptr it is filled with the same IDs the
    // tokenizer produced (including [CLS] / [SEP]).
    // If out_word_boundaries != nullptr it is filled with a parallel vector
    // marking the first subword of each logical input "word" (true / false).
    std::vector<float> Encode(const std::string &text,
                              std::vector<int> *out_subword_ids = nullptr,
                              std::vector<int> *out_word_boundaries = nullptr);

    int hidden() const { return hp_.hidden_size; }
    int feature_layers() const { return hp_.feature_layers; }
    const BertHParams &params() const { return hp_; }
    const WordPieceTokenizer &tokenizer() const { return tok_; }

private:
    BertHParams hp_;
    BertEmbeddingWeights emb_;
    std::vector<BertLayerWeights> layers_;
    WordPieceTokenizer tok_;

    // GGUF + ggml ownership
    struct gguf_context *gguf_ctx_ = nullptr;
    struct ggml_context *weight_ctx_ = nullptr;   // weights live here
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t weight_buf_ = nullptr;

    // For graph computation
    ggml_backend_sched_t sched_ = nullptr;

    bool MapTensors();
};

}  // namespace rapidspeech
