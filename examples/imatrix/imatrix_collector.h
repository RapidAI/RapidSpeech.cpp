#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// Activation-aware quantization: collect input activation squared statistics
// for each weight tensor during inference, then use them as importance weights
// during quantization (AWQ technique).
//
// Usage:
//   1. Create IMatrixCollector
//   2. After each ggml_graph_compute, call collect_from_graph(gf)
//   3. Call save(fname) to write .dat file
//   4. Pass .dat to quantize tool via --imatrix

struct IMatrixCollector {
    struct Entry {
        std::vector<double> values;  // sum of squared activations per column
        int64_t count = 0;           // number of samples accumulated
    };
    std::unordered_map<std::string, Entry> stats;

    // Collect activation statistics for a single MUL_MAT node.
    //
    // MUST be called from a per-node sched eval callback (see
    // ggml_backend_sched_set_eval_callback), not after the whole graph
    // finishes computing.  After compute, ggml_backend_sched_alloc_graph's
    // buffer reuse may have overwritten src1's data with the output of a
    // later, lifetime-disjoint op — so a post-compute walk reads garbage
    // (often NaN) and silently corrupts the imatrix.  Per-node firing
    // happens right after the node's matmul, while src1 is still the data
    // the matmul actually consumed.
    void collect_node(struct ggml_tensor *node) {
        if (!node || node->op != GGML_OP_MUL_MAT) return;

        struct ggml_tensor *weight = node->src[0];
        struct ggml_tensor *act = node->src[1];

        // Only collect for named weight tensors (loaded from GGUF)
        if (!weight->name[0]) return;
        // Only F32 activations (full precision)
        if (act->type != GGML_TYPE_F32) return;
        // Skip trivially small batches
        if (act->ne[1] < 4) return;
        // Skip non-contiguous activations (permuted/reshaped views).
        // Their nb[1] stride and ggml_nbytes do not describe a flat row-major
        // layout, so both host indexing and ggml_backend_tensor_get would
        // over-read; on GPU backends the underlying device memory is also
        // owned by view_src, not act->buffer.
        if (!ggml_is_contiguous(act)) return;
        // Skip view tensors entirely. View tensors have act->buffer == NULL
        // (the real buffer lives on view_src), so ggml_backend_tensor_get
        // would null-deref the backend's get_tensor vtable on GPU. The
        // underlying activation is almost always also fed as src1 of some
        // other named MUL_MAT, so dropping the view does not lose stats.
        if (act->view_src) return;
        // Genuine GGUF weights have op == GGML_OP_NONE; CONT-of-weight and
        // similar would carry uninitialised name bytes in some ggml versions.
        if (weight->op != GGML_OP_NONE) return;

        // Skip uninitialized activations (e.g. not yet after compute)
        if (!act->buffer && !act->data) return;

        int64_t ncol = act->ne[0];
        int64_t nrows = act->ne[1] * act->ne[2] * act->ne[3];
        if (ncol <= 0 || nrows <= 0) return;

        std::string name(weight->name);
        auto &e = stats[name];
        if (e.values.empty()) {
            e.values.resize(ncol, 0.0);
        } else if ((int64_t)e.values.size() != ncol) {
            fprintf(stderr,
                    "[imatrix] WARN: ncol mismatch for '%s' (have %zu, got "
                    "%lld) — skipping\n",
                    name.c_str(), e.values.size(), (long long)ncol);
            return;
        }

        // Read activation data (may need GPU→CPU copy)
        std::vector<float> act_data;
        const float *data;
        bool is_host = !act->buffer || ggml_backend_buffer_is_host(act->buffer);
        if (!is_host) {
            size_t nbytes = ggml_nbytes(act);
            act_data.resize(nbytes / sizeof(float));
            ggml_backend_tensor_get(act, act_data.data(), 0, nbytes);
            data = act_data.data();
        } else {
            data = (const float *)act->data;
            if (!data) return;
        }

        // Skip if activation contains NaN/Inf (would poison the imatrix
        // and crash downstream IQ grid lookups).  This shouldn't happen
        // at per-node timing on a healthy model, but defend anyway.
        bool any_bad = false;
        size_t span = (size_t)nrows * (size_t)(act->nb[1] / sizeof(float));
        for (size_t k = 0; k < span && !any_bad; k += std::max<size_t>(1, span / 16)) {
            if (!std::isfinite(data[k])) any_bad = true;
        }
        if (any_bad) {
            fprintf(stderr,
                    "[imatrix] WARN: non-finite activations in '%s' — skipping node\n",
                    name.c_str());
            return;
        }

        // Accumulate squared activations per column (contiguous: nb[1] == ncol*4)
        for (int64_t r = 0; r < nrows; r++) {
            const float *row = data + r * (act->nb[1] / sizeof(float));
            for (int64_t j = 0; j < ncol; j++) {
                double v = (double)row[j];
                e.values[j] += v * v;
            }
        }
        e.count += nrows;
    }

    // Save as legacy .dat format (compatible with llama.cpp tools).
    void save(const std::string &fname) const {
        int64_t max_count = 0;
        for (auto &kv : stats) {
            if (kv.second.count > max_count) max_count = kv.second.count;
        }
        if (max_count <= 0) {
            fprintf(stderr, "IMatrixCollector: no data collected\n");
            return;
        }

        int32_t chunk_size = 256;
        int32_t ncall = (int32_t)((max_count + chunk_size - 1) / chunk_size);

        // Count valid entries
        std::vector<std::string> names;
        for (auto &kv : stats) {
            if (kv.second.count > 0 && !kv.second.values.empty()) {
                names.push_back(kv.first);
            }
        }
        std::sort(names.begin(), names.end());

        int32_t n_entries = (int32_t)names.size();
        std::ofstream out(fname, std::ios::binary);
        if (!out) {
            fprintf(stderr, "IMatrixCollector: failed to open %s\n", fname.c_str());
            return;
        }

        out.write((const char *)&n_entries, sizeof(n_entries));

        for (auto &name : names) {
            auto &e = stats.at(name);
            int32_t nval = (int32_t)e.values.size();

            // Normalize: average activation² × ncall
            std::vector<float> tmp(nval);
            double inv_count = 1.0 / (double)e.count;
            for (int32_t j = 0; j < nval; j++) {
                tmp[j] = (float)(e.values[j] * inv_count * (double)ncall);
            }

            int32_t name_len = (int32_t)name.size();
            out.write((const char *)&name_len, sizeof(name_len));
            out.write(name.data(), name_len);
            out.write((const char *)&ncall, sizeof(ncall));
            out.write((const char *)&nval, sizeof(nval));
            out.write((const char *)tmp.data(), nval * sizeof(float));
        }

        int32_t last_chunk = (int32_t)(max_count / chunk_size);
        out.write((const char *)&last_chunk, sizeof(last_chunk));

        const char *dataset = "tts-calibration";
        int32_t dlen = (int32_t)strlen(dataset);
        out.write((const char *)&dlen, sizeof(dlen));
        out.write(dataset, dlen);

        printf("IMatrixCollector: saved %d entries to %s\n", n_entries, fname.c_str());
    }

    // Load legacy .dat format.
    bool load(const std::string &fname) {
        std::ifstream in(fname, std::ios::binary);
        if (!in) {
            fprintf(stderr, "IMatrixCollector: failed to open %s\n", fname.c_str());
            return false;
        }

        int32_t n_entries;
        in.read((char *)&n_entries, sizeof(n_entries));
        if (in.fail() || n_entries < 1) {
            fprintf(stderr, "IMatrixCollector: no data in %s\n", fname.c_str());
            return false;
        }

        stats.clear();
        int32_t chunk_size = 256;

        for (int i = 0; i < n_entries; ++i) {
            int32_t len = 0;
            in.read((char *)&len, sizeof(len));
            std::vector<char> name_buf(len + 1, 0);
            in.read(name_buf.data(), len);
            std::string name(name_buf.data());

            int32_t ncall = 0;
            in.read((char *)&ncall, sizeof(ncall));
            int32_t nval = 0;
            in.read((char *)&nval, sizeof(nval));
            if (in.fail() || nval < 1) {
                fprintf(stderr, "IMatrixCollector: bad entry %d in %s\n", i, fname.c_str());
                stats.clear();
                return false;
            }

            auto &e = stats[name];
            e.values.resize(nval, 0.0);

            std::vector<float> tmp(nval);
            in.read((char *)tmp.data(), nval * sizeof(float));
            if (in.fail()) {
                fprintf(stderr, "IMatrixCollector: failed reading data for %s\n", name.c_str());
                stats.clear();
                return false;
            }

            // Reverse the normalization: multiply by chunk_size to get
            // accumulated sum-of-squares approximating the original scale
            for (int32_t j = 0; j < nval; j++) {
                e.values[j] = (double)tmp[j] * (double)chunk_size;
            }
            e.count = (int64_t)ncall * chunk_size;
        }

        // Read tail: last_chunk + dataset name (optional, may not exist)
        int32_t last_chunk = 0;
        if (in.peek() != EOF) {
            in.read((char *)&last_chunk, sizeof(last_chunk));
        }

        printf("IMatrixCollector: loaded %d entries from %s\n", n_entries, fname.c_str());
        return true;
    }

    // Get imatrix data for a tensor, ready to pass to ggml_quantize_chunk.
    // Returns nullptr if no data for this tensor.
    // The returned pointer is valid as long as the collector is alive.
    const float *get_imatrix(const std::string &name, int64_t ncol) const {
        auto it = stats.find(name);
        if (it == stats.end()) return nullptr;
        if ((int64_t)it->second.values.size() != ncol) return nullptr;
        if (it->second.count <= 0) return nullptr;

        // Cache normalized float copy
        auto &cache = m_imatrix_cache[name];
        if (cache.empty()) {
            cache.resize(ncol);
            double inv_count = 1.0 / (double)it->second.count;
            for (int64_t j = 0; j < ncol; j++) {
                cache[j] = (float)(it->second.values[j] * inv_count);
            }
        }
        return cache.data();
    }

private:
    mutable std::unordered_map<std::string, std::vector<float>> m_imatrix_cache;
};
