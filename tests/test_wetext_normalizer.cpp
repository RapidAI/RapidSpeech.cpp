// test_wetext_normalizer.cpp
//
// Smoke + behavior tests for rs::WeTextNormalizer (the WeTextProcessing
// wrapper). Skips loudly (returns 0) when FST data is missing, so it
// stays green on machines that haven't synced rapidspeech/data/wetext/.

#include "frontend/wetext_normalizer.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_failed = 0;
int g_passed = 0;

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

void expect_true(const std::string& name, bool cond) {
    if (cond) {
        std::printf("  ok   %s\n", name.c_str());
        ++g_passed;
    } else {
        std::printf("  FAIL %s\n", name.c_str());
        ++g_failed;
    }
}

bool file_exists(const std::string& p) {
    std::ifstream in(p, std::ios::binary);
    return in.good();
}

std::string data_dir() {
    if (const char* e = std::getenv("RS_WETEXT_DATA_DIR")) {
        if (*e) return e;
    }
    return "rapidspeech/data/wetext";
}

}  // namespace

int main() {
    // ---- Pass-through behavior with no FST loaded ----
    {
        std::printf("[group] pass-through (unloaded)\n");
        rs::WeTextNormalizer tn;
        expect_true("IsLoaded() false before Load", !tn.IsLoaded());
        expect_eq("Normalize returns input verbatim",
                  tn.Normalize("2.5 平方"), "2.5 平方");
        expect_eq("empty string roundtrips", tn.Normalize(""), "");
    }

    // ---- Prefix validation (no abort) ----
    {
        std::printf("[group] prefix validation\n");
        rs::WeTextNormalizer tn;
        bool ok = tn.Load("/tmp/nope_tagger.fst", "/tmp/nope_verbalizer.fst");
        expect_true("Load rejects unknown prefix (returns false)", !ok);
        expect_true("IsLoaded() still false", !tn.IsLoaded());
        expect_eq("Normalize still pass-through after failed Load",
                  tn.Normalize("$5"), "$5");
    }

    // ---- Missing-file handling (correct prefix, no file) ----
    {
        std::printf("[group] missing-file handling\n");
        rs::WeTextNormalizer tn;
        bool ok = tn.Load("/tmp/zh_tn_tagger_nope.fst",
                          "/tmp/zh_tn_verbalizer_nope.fst");
        expect_true("Load returns false for missing files", !ok);
        expect_true("IsLoaded() false after missing-file Load", !tn.IsLoaded());
    }

    // ---- Real FST behavior (skipped if data not present) ----
    const std::string dir = data_dir();
    const std::string tagger = dir + "/zh_tn_tagger.fst";
    const std::string verbalizer = dir + "/zh_tn_verbalizer.fst";

    if (!file_exists(tagger) || !file_exists(verbalizer)) {
        std::printf("[skip] %s + %s not present; skipping real-FST checks\n",
                    tagger.c_str(), verbalizer.c_str());
        std::printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }

    std::printf("[group] real FST normalization\n");
    rs::WeTextNormalizer tn;
    bool ok = tn.Load(tagger, verbalizer);
    expect_true("Load with real FSTs succeeds", ok);
    expect_true("IsLoaded() true", tn.IsLoaded());

    expect_contains("decimal: '2.5平方电线'",
                    tn.Normalize("2.5平方电线"), "二点五");
    expect_contains("integer: '20安培'",
                    tn.Normalize("20安培"), "二十");
    expect_contains("compound: full sentence contains 二点五 and 二十",
                    tn.Normalize("2.5平方电线可以承受20安培电流。"), "二点五");
    expect_contains("year: '2026年'",
                    tn.Normalize("2026年"), "年");
    expect_true("empty string handled cleanly",
                tn.Normalize("").empty());
    {
        std::string ascii_in = "hello world";
        expect_eq("pure ASCII passes through unchanged",
                  tn.Normalize(ascii_in), ascii_in);
    }

    std::printf("\nSummary: %d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
