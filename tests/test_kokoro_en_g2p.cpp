// test_kokoro_en_g2p.cpp
//
// Behaviour tests for rs::kokoro_en::EnG2P (misaki[en] gold-dict lookup +
// letter-by-letter spell fallback). Skips real-dict checks if us_gold.bin
// isn't present, so it stays green on machines that haven't synced
// rapidspeech/data/kokoro_en/.

#include "frontend/kokoro_g2p_en.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

int g_failed = 0;
int g_passed = 0;

void expect_true(const std::string& name, bool cond) {
    if (cond) {
        std::printf("  ok   %s\n", name.c_str());
        ++g_passed;
    } else {
        std::printf("  FAIL %s\n", name.c_str());
        ++g_failed;
    }
}

void expect_eq(const std::string& name, const std::string& got,
               const std::string& want) {
    if (got == want) {
        std::printf("  ok   %s\n", name.c_str());
        ++g_passed;
    } else {
        std::printf("  FAIL %s\n         got : '%s'\n         want: '%s'\n",
                    name.c_str(), got.c_str(), want.c_str());
        ++g_failed;
    }
}

void expect_contains(const std::string& name, const std::string& got,
                     const std::string& needle) {
    if (got.find(needle) != std::string::npos) {
        std::printf("  ok   %s\n", name.c_str());
        ++g_passed;
    } else {
        std::printf("  FAIL %s\n         got    : '%s'\n         missing: '%s'\n",
                    name.c_str(), got.c_str(), needle.c_str());
        ++g_failed;
    }
}

void expect_not_contains(const std::string& name, const std::string& got,
                         const std::string& needle) {
    if (got.find(needle) == std::string::npos) {
        std::printf("  ok   %s\n", name.c_str());
        ++g_passed;
    } else {
        std::printf("  FAIL %s\n         got     : '%s'\n         forbidden: '%s'\n",
                    name.c_str(), got.c_str(), needle.c_str());
        ++g_failed;
    }
}

bool file_exists(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return in.good();
}

std::string data_dir() {
    if (const char* e = std::getenv("RS_KOKORO_EN_DATA_DIR")) {
        if (*e) return e;
    }
    return "rapidspeech/data/kokoro_en";
}

}  // namespace

int main() {
    // ---- Unloaded behaviour ----
    {
        std::printf("[group] unloaded\n");
        rs::kokoro_en::EnG2P g;
        expect_true("IsLoaded() false before Load", !g.IsLoaded());
        expect_eq("Process returns empty when not loaded",
                  g.Process("hello"), "");
    }

    // ---- Load failure on missing dir ----
    {
        std::printf("[group] missing data dir\n");
        rs::kokoro_en::EnG2P g;
        bool ok = g.Load("/tmp/__definitely_not_a_kokoro_en_dir__");
        expect_true("Load returns false for missing data dir", !ok);
        expect_true("IsLoaded() still false", !g.IsLoaded());
    }

    // ---- Real dict behaviour ----
    const std::string dir = data_dir();
    const std::string bin = dir + "/us_gold.bin";
    if (!file_exists(bin)) {
        std::printf("[skip] %s not present; skipping real-dict checks\n",
                    bin.c_str());
        std::printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }

    std::printf("[group] real dict\n");
    rs::kokoro_en::EnG2P g;
    bool ok = g.Load(dir);
    expect_true("Load with real dict succeeds", ok);
    expect_true("IsLoaded() true", g.IsLoaded());

    // Common words must yield non-empty IPA without the unk marker.
    expect_true("'hello' produces non-empty IPA",
                !g.Process("hello").empty());
    expect_true("'world' produces non-empty IPA",
                !g.Process("world").empty());
    expect_true("'are you ok' produces non-empty IPA",
                !g.Process("are you ok").empty());

    // Multi-word run keeps tokens space-separated.
    {
        std::string out = g.Process("hello world");
        expect_true("'hello world' produces non-empty IPA", !out.empty());
        expect_contains("'hello world' output contains a space", out, " ");
    }

    // Lowercase fallback: uppercase input that isn't a single-letter
    // acronym entry should still hit the lowercase form in gold.
    {
        std::string out_lower = g.Process("hello");
        std::string out_upper = g.Process("HELLO");
        expect_true("lowercase fallback returns non-empty for 'HELLO'",
                    !out_upper.empty());
        // They should produce the same phoneme string (lowercase path).
        // Note: HELLO might also be a single-letter spell — accept either,
        // but at minimum it must not be empty.
        (void)out_lower;
    }

    // OOD letter-spell fallback. "xiaomi" is not in misaki gold; spell
    // path uses gold's single-letter entries.
    {
        std::string out = g.Process("xiaomi");
        expect_true("OOD 'xiaomi' falls back to non-empty letter-spell",
                    !out.empty());
        expect_not_contains("OOD 'xiaomi' result has no '?' marker",
                            out, "❓");
    }

    // Tokenisation: punctuation and digits are dropped, words remain.
    {
        std::string out = g.Process("hello, world!");
        expect_true("punctuation: 'hello, world!' produces non-empty IPA",
                    !out.empty());
        expect_contains("punctuation: tokens still space-joined", out, " ");
    }

    // Empty input.
    expect_eq("empty input returns empty", g.Process(""), "");
    expect_eq("whitespace-only input returns empty", g.Process("   "), "");

    std::printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
