#include "common_quantize.h"
#include <climits>
#include <inttypes.h>
#include <map>
#include <mutex>
#include <regex>
#include <stdarg.h>
#include <thread>

// ============================================
// Quantization type mapping
// ============================================

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},     {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},     {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},     {"q2_k", GGML_FTYPE_MOSTLY_Q2_K},
    {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},     {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    {"q4_k_m", GGML_FTYPE_MOSTLY_Q4_K_M}, {"q5_k", GGML_FTYPE_MOSTLY_Q5_K},
    {"q5_k_m", GGML_FTYPE_MOSTLY_Q5_K_M}, {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
};

void ggml_print_ftypes(FILE *fp) {
  for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
    fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
  }
}

enum ggml_ftype ggml_parse_ftype(const char *str) {
  enum ggml_ftype ftype;
  if (str[0] == 'q') {
    const auto it = GGML_FTYPE_MAP.find(str);
    if (it == GGML_FTYPE_MAP.end()) {
      fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
      return GGML_FTYPE_UNKNOWN;
    }
    ftype = it->second;
  } else {
    ftype = (enum ggml_ftype)atoi(str);
  }
  return ftype;
}

// ============================================
// Internal helpers
// ============================================

static void zeros(std::ofstream &file, size_t n) {
  char zero = 0;
  for (size_t i = 0; i < n; ++i) {
    file.write(&zero, 1);
  }
}

static std::string format_tensor_shape(const struct ggml_tensor *t) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
  for (int i = 1; i < GGML_MAX_DIMS; i++) {
    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64,
             t->ne[i]);
  }
  return buf;
}

static std::string format(const char *fmt, ...) {
  va_list ap;
  va_list ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  GGML_ASSERT(size >= 0 && size < INT_MAX);
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  GGML_ASSERT(size2 == size);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

// ============================================
// Multi-threaded quantization
// ============================================

static size_t rs_tensor_quantize_internal(enum ggml_type new_type,
                                          const float *f32_data, void *new_data,
                                          const int64_t chunk_size,
                                          int64_t nrows, int64_t n_per_row,
                                          const float *imatrix,
                                          std::vector<std::thread> &workers,
                                          const int nthread) {

  if (nthread < 2) {
    // single-thread
    size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0,
                                          nrows, n_per_row, imatrix);
    if (!ggml_validate_row_data(new_type, new_data, new_size)) {
      throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
  }

  std::mutex mutex;
  int64_t counter = 0;
  size_t new_size = 0;
  bool valid = true;

  auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data,
                  new_data, chunk_size, nrows, n_per_row, imatrix]() {
    const int64_t nrows_per_chunk = chunk_size / n_per_row;
    size_t local_size = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      int64_t first_row = counter;
      counter += nrows_per_chunk;
      if (first_row >= nrows) {
        if (local_size > 0) {
          new_size += local_size;
        }
        break;
      }
      lock.unlock();

      const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
      size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data,
                                             first_row * n_per_row, this_nrow,
                                             n_per_row, imatrix);
      local_size += this_size;

      // validate the quantized data
      const size_t row_size = ggml_row_size(new_type, n_per_row);
      void *this_data = (char *)new_data + first_row * row_size;
      if (!ggml_validate_row_data(new_type, this_data, this_size)) {
        std::unique_lock<std::mutex> lock(mutex);
        valid = false;
        break;
      }
    }
  };

  for (int it = 0; it < nthread - 1; ++it) {
    workers.emplace_back(compute);
  }
  compute();
  for (auto &w : workers) {
    w.join();
  }
  workers.clear();

  if (!valid) {
    throw std::runtime_error("quantized data validation failed");
  }
  return new_size;
}

// ============================================
// Per-tensor type selection for mixed-precision strategies
// ============================================

static ggml_type
rs_get_qtype_for_tensor(ggml_ftype ftype, const std::string &name,
                        const struct ggml_tensor *tensor,
                        const std::vector<std::string> &to_skip) {
  // Check if this tensor should be quantized at all
  bool quantize = name.rfind("weight") == name.size() - 6;
  for (const auto &s : to_skip) {
    if (std::regex_match(name, std::regex(s))) {
      quantize = false;
      break;
    }
  }
  quantize &= (ggml_n_dims(tensor) >= 2);
  quantize &= name.find("_norm.weight") == std::string::npos;
  quantize &= name.find(".norm.weight") == std::string::npos;

  if (!quantize) {
    return tensor->type;
  }

  // Select per-tensor type based on the overall strategy
  switch (ftype) {
  case GGML_FTYPE_MOSTLY_Q4_K_M:
    // Q4_K_M: embed + lm_head + ctc_lo → Q6_K, everything else → Q4_K
    if (name.find("embed_tokens.weight") != std::string::npos ||
        name.find("lm_head.weight") != std::string::npos ||
        name.find("ctc.ctc_lo.weight") != std::string::npos) {
      return GGML_TYPE_Q6_K;
    }
    // Check ne[0] alignment for Q4_K (blck_size=256)
    if (tensor->ne[0] % ggml_blck_size(GGML_TYPE_Q4_K) != 0) {
      return tensor->type;
    }
    return GGML_TYPE_Q4_K;

  case GGML_FTYPE_MOSTLY_Q5_K_M:
    // Q5_K_M: embed + lm_head + ctc_lo → Q6_K, everything else → Q5_K
    if (name.find("embed_tokens.weight") != std::string::npos ||
        name.find("lm_head.weight") != std::string::npos ||
        name.find("ctc.ctc_lo.weight") != std::string::npos) {
      return GGML_TYPE_Q6_K;
    }
    if (tensor->ne[0] % ggml_blck_size(GGML_TYPE_Q5_K) != 0) {
      return tensor->type;
    }
    return GGML_TYPE_Q5_K;

  default: {
    // Uniform quantization — resolve ftype → qtype
    ggml_type qtype = ggml_ftype_to_ggml_type(ftype);
    // Check ne[0] alignment
    if (tensor->ne[0] % ggml_blck_size(qtype) != 0) {
      return tensor->type;
    }
    return qtype;
  }
  }
}

// ============================================
// Main quantize function
// ============================================

bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const ggml_ftype ftype, const int nthread,
                                const std::vector<std::string> &to_quant,
                                const std::vector<std::string> &to_skip) {

  // Validate that the ftype is supported
  {
    bool supported = false;
    switch (ftype) {
    case GGML_FTYPE_MOSTLY_Q4_0:
    case GGML_FTYPE_MOSTLY_Q4_1:
    case GGML_FTYPE_MOSTLY_Q5_0:
    case GGML_FTYPE_MOSTLY_Q5_1:
    case GGML_FTYPE_MOSTLY_Q8_0:
    case GGML_FTYPE_MOSTLY_Q2_K:
    case GGML_FTYPE_MOSTLY_Q3_K:
    case GGML_FTYPE_MOSTLY_Q4_K:
    case GGML_FTYPE_MOSTLY_Q5_K:
    case GGML_FTYPE_MOSTLY_Q6_K:
    case GGML_FTYPE_MOSTLY_Q4_K_M:
    case GGML_FTYPE_MOSTLY_Q5_K_M:
      supported = true;
      break;
    default:
      break;
    }
    if (!supported) {
      fprintf(stderr, "%s: unsupported model type %d\n", __func__, ftype);
      return false;
    }
  }

  printf("%s: quantizing with strategy ftype=%d ..\n", __func__, ftype);

  size_t total_size_org = 0;
  size_t total_size_new = 0;

  std::vector<std::thread> workers;
  workers.reserve(nthread);

  int idx = 0;
  std::vector<uint8_t> work;

  struct gguf_context *ctx_out = gguf_init_empty();

  // copy the KV pairs from the input file
  gguf_set_kv(ctx_out, gguf_input);
  gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
  gguf_set_val_u32(ctx_out, "general.file_type", ftype);

  const int n_tensors = gguf_get_n_tensors(gguf_input);

  // open model gguf file for reading tensor data
  auto fin = std::ifstream(fname_inp, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: cannot open model file for loading tensors\n",
            __func__);
    return false;
  }

  // Add tensor descriptors to output context (types will be updated during
  // quantization)
  for (int i = 0; i < n_tensors; ++i) {
    const std::string name = gguf_get_tensor_name(gguf_input, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx, name.c_str());
    gguf_add_tensor(ctx_out, tensor);
  }

  std::ofstream fout;

  auto close_ofstream = [&]() {
    if (fout.is_open()) {
      fout.seekp(0);
      std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
      gguf_get_meta_data(ctx_out, data.data());
      fout.write((const char *)data.data(), data.size());
      fout.close();
    }
  };

  auto new_ofstream = [&]() {
    GGML_ASSERT(ctx_out && "Found uninitialized gguf_context");
    fout = std::ofstream(fname_out, std::ios::binary);
    fout.exceptions(std::ofstream::failbit);
    const size_t meta_size = gguf_get_meta_size(ctx_out);
    ::zeros(fout, meta_size);
  };

  new_ofstream();

  for (int i = 0; i < n_tensors; ++i) {
    const std::string name = gguf_get_tensor_name(gguf_input, i);
    const size_t offset = gguf_get_tensor_offset(gguf_input, i) +
                          gguf_get_data_offset(gguf_input);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx, name.c_str());

    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
      fprintf(
          stderr,
          "%s: unsupported tensor type %d (%s) for quantization (tensor: %s)\n",
          __func__, tensor->type, ggml_type_name((ggml_type)tensor->type),
          name.c_str());
      close_ofstream();
      return false;
    }

    fin.seekg(offset, std::ios::beg);
    if (!fin) {
      fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__,
              name.c_str());
      close_ofstream();
      return false;
    }

    // read tensor data
    int num_bytes = ggml_nbytes(tensor);
    fin.read(reinterpret_cast<char *>(tensor->data), num_bytes);

    printf("[%4d/%4d] %48s - [%s], type = %6s, ", ++idx, n_tensors,
           name.c_str(), format_tensor_shape(tensor).c_str(),
           ggml_type_name(tensor->type));

    // Determine per-tensor target type (handles mixed-precision strategies)
    enum ggml_type new_type =
        rs_get_qtype_for_tensor(ftype, name, tensor, to_skip);
    bool quantize = (new_type != tensor->type) && ggml_is_quantized(new_type);

    if (!quantize && new_type == tensor->type) {
      // Check if rs_get_qtype_for_tensor indicated a skip due to alignment
      // by returning the original type for a weight tensor that would normally
      // be quantized
      bool is_weight = name.rfind("weight") == name.size() - 6;
      if (is_weight && ggml_n_dims(tensor) >= 2 &&
          name.find("_norm.weight") == std::string::npos &&
          name.find(".norm.weight") == std::string::npos) {
        bool in_skip = false;
        for (const auto &s : to_skip) {
          if (std::regex_match(name, std::regex(s))) {
            in_skip = true;
            break;
          }
        }
        if (!in_skip) {
          // Skipped due to alignment — note it
          printf("(skipped: alignment) ");
        }
      }
    }

    void *new_data;
    size_t new_size;

    if (!quantize) {
      new_type = tensor->type;
      new_data = tensor->data;
      new_size = ggml_nbytes(tensor);
      printf("size = %8.3f MB\n", ggml_nbytes(tensor) / 1024.0 / 1024.0);
    } else {
      const int64_t nelements = ggml_nelements(tensor);
      const float *imatrix = nullptr;

      float *f32_data;
      std::vector<float> f32_converted; // only used when input is F16

      if (tensor->type == GGML_TYPE_F32) {
        f32_data = (float *)tensor->data;
      } else {
        // F16 → F32: need separate buffer since sizes differ
        f32_converted.resize(nelements);
        auto data_f16 = (const ggml_fp16_t *)tensor->data;
        for (int64_t j = 0; j < nelements; ++j) {
          f32_converted[j] = ggml_fp16_to_fp32(data_f16[j]);
        }
        f32_data = f32_converted.data();
      }

      printf("converting to %s .. ", ggml_type_name(new_type));
      fflush(stdout);

      if (work.size() < (size_t)nelements * 4) {
        work.resize(nelements * 4); // upper bound on size
      }
      new_data = work.data();

      const int64_t n_per_row = tensor->ne[0];
      const int64_t nrows = tensor->ne[1];

      static const int64_t min_chunk_size = 32 * 512;
      const int64_t chunk_size =
          (n_per_row >= min_chunk_size
               ? n_per_row
               : n_per_row * ((min_chunk_size + n_per_row - 1) / n_per_row));

      const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
      const int64_t nchunk = (nelements_matrix + chunk_size - 1) / chunk_size;
      const int64_t nthread_use =
          nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk))
                      : 1;

      // quantize each expert/slice separately
      new_size = 0;
      for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
        const float *f32_data_03 = f32_data + i03 * nelements_matrix;
        void *new_data_03 =
            (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
        const float *imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

        new_size += rs_tensor_quantize_internal(
            new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row,
            imatrix_03, workers, nthread_use);
      }
      printf("size = %8.2f MiB -> %8.2f MiB\n",
             ggml_nbytes(tensor) / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    total_size_org += ggml_nbytes(tensor);
    total_size_new += new_size;

    // update the gguf meta data
    gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
    gguf_set_tensor_data(ctx_out, name.c_str(), new_data);

    // write tensor data + padding
    fout.write((const char *)new_data, new_size);
    zeros(fout, GGML_PAD(new_size, GGUF_DEFAULT_ALIGNMENT) - new_size);
  }

  close_ofstream();
  gguf_free(ctx_out);

  printf("%s: model size  = %8.2f MB\n", __func__,
         total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__,
         total_size_new / 1024.0 / 1024.0, ftype,
         ftype == GGML_FTYPE_MOSTLY_Q4_K_M ? "Q4_K_M"
         : ftype == GGML_FTYPE_MOSTLY_Q5_K_M
             ? "Q5_K_M"
             : ggml_type_name(ggml_ftype_to_ggml_type(ftype)));

  return true;
}
