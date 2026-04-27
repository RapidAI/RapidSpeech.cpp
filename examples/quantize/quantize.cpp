#include "common_quantize.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

#define RS_QUANTIZE_LOG_INFO(fmt, ...)                                         \
  std::printf("[RapidSpeech] " fmt "\n", ##__VA_ARGS__)
#define RS_QUANTIZE_LOG_ERROR(fmt, ...)                                        \
  std::fprintf(stderr, "[RapidSpeech] ERROR: " fmt "\n", ##__VA_ARGS__)

// quantize a model
static bool rs_model_quantize(const std::string &fname_inp,
                              const std::string &fname_out, ggml_ftype ftype) {

  RS_QUANTIZE_LOG_INFO("Loading model from '%s'", fname_inp.c_str());

  struct ggml_context *ctx = NULL;
  struct gguf_init_params gguf_params = {
      /*.no_alloc = */ false,
      /*.ctx      = */ &ctx,
  };

  struct gguf_context *gguf_ctx =
      gguf_init_from_file(fname_inp.c_str(), gguf_params);
  if (!gguf_ctx) {
    RS_QUANTIZE_LOG_ERROR("Failed to open GGUF file: %s", fname_inp.c_str());
    return false;
  }

  // Print GGUF metadata
  {
    RS_QUANTIZE_LOG_INFO("GGUF version:      %d", gguf_get_version(gguf_ctx));
    RS_QUANTIZE_LOG_INFO("GGUF alignment:   %zu", gguf_get_alignment(gguf_ctx));
    RS_QUANTIZE_LOG_INFO("GGUF data offset: %zu",
                         gguf_get_data_offset(gguf_ctx));

    const int n_kv = gguf_get_n_kv(gguf_ctx);
    RS_QUANTIZE_LOG_INFO("GGUF n_kv: %d", n_kv);

    for (int i = 0; i < n_kv; ++i) {
      const char *key = gguf_get_key(gguf_ctx, i);
      RS_QUANTIZE_LOG_INFO("  kv[%d]: %s", i, key);
    }
  }

  const int n_tensors = gguf_get_n_tensors(gguf_ctx);
  RS_QUANTIZE_LOG_INFO("Number of tensors: %d", n_tensors);

  // Regexes of tensor names to NOT quantize (keep original precision)
  // For Q4_K_M / Q5_K_M: embed/output/ctc_lo are quantized to Q6_K by
  // rs_get_qtype_for_tensor, so they must NOT be in the skip list.
  bool is_k_m =
      (ftype == GGML_FTYPE_MOSTLY_Q4_K_M || ftype == GGML_FTYPE_MOSTLY_Q5_K_M);

  std::vector<std::string> to_skip = {
      // FSMN block weights — 1D convolution, quantization hurts accuracy
      "encoder.*.fsmn_block.weight",
      // Norm weights — small tensors, quantization not beneficial
      ".*norm.*weight",
      // VAD parameters (if present in model)
      "_model.stft.forward_basis_buffer.weight",
      "_model.encoder.*.reparam_conv.weight",
      "_model.encoder.*.reparam_conv.bias", "_model.decoder.rnn.weight_ih",
      "_model.decoder.rnn.weight_hh", "_model.decoder.decoder.2.weight",
      "_model.decoder.decoder.2.bias"};

  if (!is_k_m) {
    // For non-mixed strategies: keep embed/lm_head/ctc at F16/F32
    to_skip.insert(to_skip.begin(), {"embed_tokens.weight", "lm_head.weight",
                                     "ctc.ctc_lo.weight"});
  }

  // Quantize all weight tensors except those in to_skip
  const std::vector<std::string> to_quant = {".*"};

  if (!rapid_speech_ggml_quantize(ctx, gguf_ctx, fname_inp, fname_out, ftype, 4,
                                  to_quant, to_skip)) {
    RS_QUANTIZE_LOG_ERROR("Failed to quantize model '%s'", fname_inp.c_str());
    return false;
  }

  return true;
}

static void rs_print_usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <model-input.gguf> <model-output.gguf> <type>\n\n",
               argv0);
  std::fprintf(stderr, "Quantization types:\n");
  ggml_print_ftypes(stderr);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    rs_print_usage(argv[0]);
    return 1;
  }

  // needed to initialize f16 tables
  {
    struct ggml_init_params params = {0, NULL, false};
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
  }

  const std::string fname_inp = argv[1];
  const std::string fname_out = argv[2];

  const ggml_ftype ftype = ggml_parse_ftype(argv[3]);
  if (ftype == GGML_FTYPE_UNKNOWN) {
    RS_QUANTIZE_LOG_ERROR("Unknown quantization type: %s", argv[3]);
    ggml_print_ftypes(stderr);
    return 1;
  }

  RS_QUANTIZE_LOG_INFO("Input:  %s", fname_inp.c_str());
  RS_QUANTIZE_LOG_INFO("Output: %s", fname_out.c_str());
  RS_QUANTIZE_LOG_INFO("Type:   %d (%s)", ftype,
                       ftype == GGML_FTYPE_MOSTLY_Q4_K_M ? "Q4_K_M"
                       : ftype == GGML_FTYPE_MOSTLY_Q5_K_M
                           ? "Q5_K_M"
                           : ggml_type_name(ggml_ftype_to_ggml_type(ftype)));

  const int64_t t_main_start_us = ggml_time_us();

  int64_t t_quantize_us = 0;
  {
    const int64_t t_start_us = ggml_time_us();

    if (!rs_model_quantize(fname_inp, fname_out, ftype)) {
      RS_QUANTIZE_LOG_ERROR("Failed to quantize model from '%s'",
                            fname_inp.c_str());
      return 1;
    }

    t_quantize_us = ggml_time_us() - t_start_us;
  }

  // report timing
  {
    const int64_t t_main_end_us = ggml_time_us();

    std::printf("\n");
    std::printf("%s: quantize time = %8.2f ms\n", __func__,
                t_quantize_us / 1000.0f);
    std::printf("%s:    total time = %8.2f ms\n", __func__,
                (t_main_end_us - t_main_start_us) / 1000.0f);
  }

  return 0;
}
