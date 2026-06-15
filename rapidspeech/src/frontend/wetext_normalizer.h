#pragma once

#include <memory>
#include <string>

namespace rs {

// Thin C++ wrapper around wetext::Processor (WeTextProcessing runtime).
//
// PIMPL'd so the header carries no OpenFST / wetext dependencies — only
// <memory> + <string>. Construct, optionally Load(...), then call Normalize.
//
// File-naming convention is significant: wetext::Processor decides the
// parse type from the tagger path's prefix (zh_tn_ / zh_itn_ / en_tn_ /
// ja_tn_); rename at your peril.
class WeTextNormalizer {
public:
    WeTextNormalizer();
    ~WeTextNormalizer();

    WeTextNormalizer(const WeTextNormalizer&) = delete;
    WeTextNormalizer& operator=(const WeTextNormalizer&) = delete;

    // Loads tagger + verbalizer FSTs. Returns true on success.
    //
    // On failure (file missing, FST format error, prefix mismatch) returns
    // false and IsLoaded() stays false — Normalize() then becomes a
    // pass-through. Failures are logged but never abort.
    bool Load(const std::string& tagger_path,
              const std::string& verbalizer_path);

    bool IsLoaded() const { return loaded_; }

    // Returns input unchanged if not loaded; otherwise the FST-normalized
    // form (e.g. "2.5平方" → "二点五平方").
    std::string Normalize(const std::string& text);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_ = false;
};

}  // namespace rs
