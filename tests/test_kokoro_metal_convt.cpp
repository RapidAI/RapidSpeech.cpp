// Validate KokoroMetalConvT1D against a CPU reference (deterministic random
// weights/input). Verifies the kernel matches PyTorch ConvTranspose1d semantics
// for the two Kokoro generator layers (ups0: K=20, s=10, p=5, IC=512, OC=256;
// ups1: K=12, s=6, p=3, IC=256, OC=128).
//
// Run:
//   ./build/bin/test_kokoro_metal_convt
//
// Exits non-zero if max|diff| exceeds 5e-3 (F16 weight quantization tolerance).
//
// On non-Apple builds this test is skipped (returns 0).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef __APPLE__
#include "ggml-metal.h"
#endif

#ifdef __APPLE__
#include "arch/kokoro_metal.h"

// ---------------------------------------------------------------------------
// ggml-metal benchmark of the same convt op (the kernel we are replacing)
// ---------------------------------------------------------------------------
struct GgmlMetalConvT {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t w_buf = nullptr;
    ggml_context *w_ctx = nullptr;
    ggml_tensor *w = nullptr;   // F16 (K, OC, IC)
    ggml_tensor *b = nullptr;   // F32 (OC)
    int K = 0, IC = 0, OC = 0, stride = 0, pad = 0;

    ~GgmlMetalConvT() {
        if (w_buf) ggml_backend_buffer_free(w_buf);
        if (w_ctx) ggml_free(w_ctx);
        if (backend) ggml_backend_free(backend);
    }

    bool init(const ggml_fp16_t *w_data, const float *b_data,
              int K_, int IC_, int OC_, int stride_, int pad_) {
        K = K_; IC = IC_; OC = OC_; stride = stride_; pad = pad_;
        backend = ggml_backend_metal_init();
        if (!backend) return false;

        ggml_init_params ip = {2 * ggml_tensor_overhead(), nullptr, true};
        w_ctx = ggml_init(ip);
        w = ggml_new_tensor_3d(w_ctx, GGML_TYPE_F16, K, OC, IC);
        b = ggml_new_tensor_1d(w_ctx, GGML_TYPE_F32, OC);
        ggml_set_name(w, "w");
        ggml_set_name(b, "b");
        w_buf = ggml_backend_alloc_ctx_tensors(w_ctx, backend);
        if (!w_buf) return false;
        ggml_backend_tensor_set(w, w_data, 0, ggml_nbytes(w));
        ggml_backend_tensor_set(b, b_data, 0, ggml_nbytes(b));
        return true;
    }

    // Run via a fresh graph each call (matches kokoro.cpp pattern).
    // Input: F32 (IC, T_in) packed T-fast (matches what we feed kernel).
    // Returns cropped F32 (OC, T_out) packed T-fast.
    bool run(const float *input, int T_in, float *output) {
        std::vector<uint8_t> meta(ggml_tensor_overhead() * 16 + ggml_graph_overhead());
        ggml_init_params ip = {meta.size(), meta.data(), true};
        ggml_context *ctx0 = ggml_init(ip);

        // ggml_conv_transpose_1d takes src1 (input) with layout (T_in, IC).
        // We get input in (IC, T_in). Re-interpret as (T_in, IC) for the call.
        ggml_tensor *in_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_in, IC);
        ggml_set_name(in_t, "in");

        ggml_tensor *y = ggml_conv_transpose_1d(ctx0, w, in_t, stride, 0, 1);
        // y shape: (T_unpad, OC, 1, 1)
        const int T_unpad = (T_in - 1) * stride + K;
        const int T_out   = T_unpad - 2 * pad;
        y = ggml_reshape_2d(ctx0, y, T_unpad, OC);
        if (pad > 0) {
            y = ggml_view_2d(ctx0, y, T_out, OC, (size_t)T_unpad * sizeof(float),
                             (size_t)pad * sizeof(float));
            y = ggml_cont(ctx0, y);
        }
        // (T_out, OC) — bias broadcasts on T after transpose
        ggml_tensor *yT = ggml_cont(ctx0, ggml_transpose(ctx0, y)); // (OC, T_out)
        yT = ggml_add(ctx0, yT, b);
        ggml_set_name(yT, "out");

        ggml_cgraph *gf = ggml_new_graph(ctx0);
        ggml_build_forward_expand(gf, yT);

        ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, gf)) {
            ggml_gallocr_free(alloc);
            ggml_free(ctx0);
            return false;
        }
        ggml_tensor *in_g = ggml_graph_get_tensor(gf, "in");
        ggml_backend_tensor_set(in_g, input, 0, ggml_nbytes(in_g));
        ggml_backend_graph_compute(backend, gf);

        ggml_tensor *out_g = ggml_graph_get_tensor(gf, "out");
        ggml_backend_tensor_get(out_g, output, 0, ggml_nbytes(out_g));

        ggml_gallocr_free(alloc);
        ggml_free(ctx0);
        return true;
    }
};
#endif

#ifdef __APPLE__

// CPU reference: matches PyTorch ConvTranspose1d(in, w, b, stride, padding=p)
//   Output T_out = (T_in - 1) * s + K - 2p
//   For output position t (post-crop), j = t + p; out[oc, t] = sum over IC, i, k
//   where i*s + k == j, i in [0, T_in), k in [0, K), of w[k, oc, ic] * in[ic, i].
//
// w_ggml shape ne=(K, OC, IC), K fastest, F16
// in shape (IC, T_in), T_in fastest, F32
// out shape (OC, T_out), T_out fastest, F32
static void convt1d_ref_f16(const ggml_fp16_t *w, const float *bias,
                            const float *in, int T_in,
                            int K, int IC, int OC, int s, int p,
                            float *out) {
    const int T_unpad = (T_in - 1) * s + K;
    const int T_out   = T_unpad - 2 * p;
    for (int oc = 0; oc < OC; oc++) {
        for (int t = 0; t < T_out; t++) {
            const int j = t + p;
            int a = j - K + 1;
            int i_min = (a <= 0) ? 0 : (a + s - 1) / s;
            int i_max = j / s;
            if (i_max > T_in - 1) i_max = T_in - 1;
            float acc = 0.0f;
            for (int ic = 0; ic < IC; ic++) {
                for (int i = i_min; i <= i_max; i++) {
                    const int k = j - i * s;
                    const float wv = ggml_fp16_to_fp32(
                        w[(size_t)k + (size_t)oc * K + (size_t)ic * K * OC]);
                    acc += wv * in[(size_t)i + (size_t)ic * T_in];
                }
            }
            out[(size_t)t + (size_t)oc * T_out] = acc + bias[oc];
        }
    }
}

struct FakeTensor {
    std::vector<uint8_t> buf;
    ggml_tensor t{};
    void make_w_f16(int K, int OC, int IC, std::mt19937 &rng) {
        const size_t n = (size_t)K * OC * IC;
        buf.resize(n * sizeof(ggml_fp16_t));
        ggml_fp16_t *p = (ggml_fp16_t *)buf.data();
        std::normal_distribution<float> nd(0.0f, 0.1f);
        for (size_t i = 0; i < n; i++) p[i] = ggml_fp32_to_fp16(nd(rng));
        std::memset(&t, 0, sizeof(t));
        t.type = GGML_TYPE_F16;
        t.ne[0] = K; t.ne[1] = OC; t.ne[2] = IC; t.ne[3] = 1;
        t.data = buf.data();
    }
    void make_b_f32(int OC, std::mt19937 &rng) {
        buf.resize((size_t)OC * sizeof(float));
        float *p = (float *)buf.data();
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (int i = 0; i < OC; i++) p[i] = nd(rng);
        std::memset(&t, 0, sizeof(t));
        t.type = GGML_TYPE_F32;
        t.ne[0] = OC; t.ne[1] = 1; t.ne[2] = 1; t.ne[3] = 1;
        t.data = buf.data();
    }
};

static int run_layer_check(int layer, int K, int IC, int OC, int stride, int pad, int T_in,
                           const ggml_fp16_t *w, const float *b, KokoroMetalConvT1D &kernel) {
    const int T_out = (T_in - 1) * stride + K - 2 * pad;
    std::mt19937 rng(0xC0FFEE + layer);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> in((size_t)IC * T_in);
    for (auto &v : in) v = nd(rng);

    std::vector<float> ref((size_t)OC * T_out);
    auto t0 = std::chrono::steady_clock::now();
    convt1d_ref_f16(w, b, in.data(), T_in, K, IC, OC, stride, pad, ref.data());
    auto t1 = std::chrono::steady_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<float> got((size_t)OC * T_out, 0.0f);
    // Warm-up
    if (!kernel.run(layer, in.data(), T_in, got.data())) {
        std::fprintf(stderr, "[layer %d] kernel.run failed\n", layer);
        return 1;
    }
    const int n_iter = 5;
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < n_iter; it++) {
        kernel.run(layer, in.data(), T_in, got.data());
    }
    auto t3 = std::chrono::steady_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / n_iter;

    // Benchmark ggml-metal kernel (the one we are replacing)
    GgmlMetalConvT gm;
    std::vector<float> gm_out((size_t)OC * T_out, 0.0f);
    double ggml_ms = -1.0;
    if (gm.init(w, b, K, IC, OC, stride, pad)) {
        gm.run(in.data(), T_in, gm_out.data()); // warm-up
        auto u0 = std::chrono::steady_clock::now();
        for (int it = 0; it < n_iter; it++) {
            gm.run(in.data(), T_in, gm_out.data());
        }
        auto u1 = std::chrono::steady_clock::now();
        ggml_ms = std::chrono::duration<double, std::milli>(u1 - u0).count() / n_iter;
    }

    float max_abs = 0.0f, sum_sq_diff = 0.0f, sum_sq_ref = 0.0f;
    int bad_idx = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        float d = std::fabs(got[i] - ref[i]);
        if (d > max_abs) { max_abs = d; bad_idx = (int)i; }
        sum_sq_diff += d * d;
        sum_sq_ref  += ref[i] * ref[i];
    }
    float snr_db = 10.0f * std::log10(sum_sq_ref / (sum_sq_diff + 1e-30f));

    std::printf("[layer %d] K=%d IC=%d OC=%d s=%d p=%d T_in=%d T_out=%d\n",
                layer, K, IC, OC, stride, pad, T_in, T_out);
    std::printf("           max|Δ|=%.3e  SNR=%.1f dB\n", max_abs, snr_db);
    std::printf("           ref(cpu)=%.1f ms   ggml-metal=%.2f ms   ours=%.2f ms",
                ref_ms, ggml_ms, gpu_ms);
    if (ggml_ms > 0) {
        std::printf("   speedup vs ggml-metal: %.2fx", ggml_ms / gpu_ms);
    }
    std::printf("\n");

    const float tol = 5e-3f;
    if (max_abs > tol) {
        std::printf("  FAIL: max|Δ|=%.3e > %.1e (worst idx %d: got %.6f vs ref %.6f)\n",
                    max_abs, tol, bad_idx, got[bad_idx], ref[bad_idx]);
        return 1;
    }
    return 0;
}

int main() {
    std::mt19937 rng(0xDEADBEEF);

    // Build synthetic weights for both layers
    FakeTensor w0, b0, w1, b1;
    w0.make_w_f16(20, 256, 512, rng);
    b0.make_b_f32(256, rng);
    w1.make_w_f16(12, 128, 256, rng);
    b1.make_b_f32(128, rng);

    KokoroMetalConvT1D kernel;
    if (!kernel.init(&w0.t, &b0.t, /*s*/ 10, /*p*/ 5,
                     &w1.t, &b1.t, /*s*/ 6,  /*p*/ 3)) {
        std::fprintf(stderr, "kernel.init failed\n");
        return 1;
    }

    int fail = 0;
    // Layer 0: ups0. Realistic T_in ranges (T_frames = 2*phon_len ~ 200 for ~3s clip).
    fail += run_layer_check(0, 20, 512, 256, 10, 5, /*T_in*/ 256,
                            (const ggml_fp16_t *)w0.buf.data(),
                            (const float *)b0.buf.data(), kernel);
    // Layer 1: ups1.
    fail += run_layer_check(1, 12, 256, 128, 6, 3, /*T_in*/ 2560,
                            (const ggml_fp16_t *)w1.buf.data(),
                            (const float *)b1.buf.data(), kernel);

    if (fail) {
        std::printf("\nFAILED\n");
        return 1;
    }
    std::printf("\nPASSED\n");
    return 0;
}

#else // __APPLE__

int main() {
    std::printf("[skip] not Apple/Metal — test_kokoro_metal_convt has nothing to do.\n");
    return 0;
}

#endif
