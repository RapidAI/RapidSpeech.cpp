#pragma once

#include <ggml.h>
#include <gguf.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Mixed-precision ftypes following GGUF spec conventions
// (not in ggml.h enum — these are higher-level quantization strategies)
#define GGML_FTYPE_MOSTLY_Q4_K_M ((enum ggml_ftype)15)
#define GGML_FTYPE_MOSTLY_Q5_K_M ((enum ggml_ftype)17)

// Parse quantization type string (e.g. "q4_k") to ggml_ftype enum
enum ggml_ftype ggml_parse_ftype(const char *str);

// Print supported quantization types to file pointer
void ggml_print_ftypes(FILE *fp = stderr);

// Quantize a GGUF model: read fname_inp, quantize weight tensors, write
// fname_out
bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const ggml_ftype ftype, const int nthread,
                                const std::vector<std::string> &to_quant,
                                const std::vector<std::string> &to_skip);
