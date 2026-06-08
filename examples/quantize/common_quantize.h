#pragma once

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Custom mixed-precision ftypes — NOT in ggml.h.
// Use values outside ggml's current enumerators but inside the underlying-type
// range (clang infers [-32, 31] from the existing enumerators, so values must
// stay within that). 5/6 sit in the 4→7 gap; 28/29 are above the current
// top (Q1_0 = 27).  Only 30 and 31 remain free — keep new variants minimal.
#define GGML_FTYPE_MOSTLY_Q4_K_M     ((enum ggml_ftype)5)
#define GGML_FTYPE_MOSTLY_Q5_K_M     ((enum ggml_ftype)6)
#define GGML_FTYPE_MOSTLY_Q3_K_M     ((enum ggml_ftype)28)
#define GGML_FTYPE_MOSTLY_Q2_K_M     ((enum ggml_ftype)29)
#define GGML_FTYPE_MOSTLY_IQ2_XXS_M  ((enum ggml_ftype)30)
#define GGML_FTYPE_MOSTLY_IQ1_S_M    ((enum ggml_ftype)31)

// Parse quantization type string (e.g. "q4_k") to ggml_ftype enum
enum ggml_ftype ggml_parse_ftype(const char *str);

// Print supported quantization types to file pointer
void ggml_print_ftypes(FILE *fp = stderr);

// Options for rapid_speech_ggml_quantize.  Allows opting out of the default
// mixed-precision policy:
//   pure                  — skip all per-tensor "bump" logic; every quantized
//                           weight uses the ftype's underlying qtype.  When
//                           combined with a non-K_M ftype, the caller must
//                           also drop embed/lm_head/ctc from to_skip.
//   token_embedding_type  — force this type on tensors matching TOKEN_EMBD
//                           (embed_tokens.weight, embed.weight).  Overrides
//                           both --pure and the K_M bump.  GGML_TYPE_COUNT
//                           means "unset".
//   output_tensor_type    — same, but for OUTPUT (lm_head.weight, ctc.ctc_lo,
//                           ctc_out_linear).
struct rs_quantize_options {
  enum ggml_ftype ftype = GGML_FTYPE_ALL_F32;
  int nthread = 4;
  std::vector<std::string> to_quant = {".*"};
  std::vector<std::string> to_skip;
  std::string imatrix_file;
  bool pure = false;
  enum ggml_type token_embedding_type = GGML_TYPE_COUNT;
  enum ggml_type output_tensor_type   = GGML_TYPE_COUNT;
};

// Quantize a GGUF model: read fname_inp, quantize weight tensors, write
// fname_out.  Uses opts to control strategy and per-tensor overrides.
bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const rs_quantize_options &opts);

// Backward-compatible overload; internally builds an rs_quantize_options
// with pure=false and no per-tensor overrides.
bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const ggml_ftype ftype, const int nthread,
                                const std::vector<std::string> &to_quant,
                                const std::vector<std::string> &to_skip,
                                const std::string &imatrix_file = "");
