// wetext_normalizer.cpp — see header for overview.
//
// PIMPL boundary: this is the only TU that pulls in wetext / OpenFST
// headers, so consumers of WeTextNormalizer pay no cost in include time
// or header pollution.

#include "frontend/wetext_normalizer.h"
#include "utils/rs_log.h"

#include "processor/wetext_processor.h"

#include <fstream>
#include <utility>

namespace rs {

struct WeTextNormalizer::Impl {
    std::unique_ptr<wetext::Processor> proc;
};

WeTextNormalizer::WeTextNormalizer() : impl_(std::make_unique<Impl>()) {}
WeTextNormalizer::~WeTextNormalizer() = default;

static bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

// wetext::Processor decides parse type by substring match in the tagger
// path; bail early with a clear message rather than letting LOG(FATAL)
// abort the host process.
static bool has_known_prefix(const std::string& tagger_path) {
    return tagger_path.find("zh_tn_")  != std::string::npos
        || tagger_path.find("zh_itn_") != std::string::npos
        || tagger_path.find("en_tn_")  != std::string::npos
        || tagger_path.find("ja_tn_")  != std::string::npos;
}

bool WeTextNormalizer::Load(const std::string& tagger_path,
                            const std::string& verbalizer_path) {
    loaded_ = false;
    impl_->proc.reset();

    if (!has_known_prefix(tagger_path)) {
        RS_LOG_WARN(
            "WeTextNormalizer: tagger path '%s' has no recognized prefix "
            "(expected zh_tn_/zh_itn_/en_tn_/ja_tn_); skipping",
            tagger_path.c_str());
        return false;
    }
    if (!file_exists(tagger_path)) {
        RS_LOG_WARN("WeTextNormalizer: tagger FST not found: %s",
                    tagger_path.c_str());
        return false;
    }
    if (!file_exists(verbalizer_path)) {
        RS_LOG_WARN("WeTextNormalizer: verbalizer FST not found: %s",
                    verbalizer_path.c_str());
        return false;
    }

    try {
        impl_->proc = std::make_unique<wetext::Processor>(tagger_path,
                                                          verbalizer_path);
    } catch (const std::exception& e) {
        RS_LOG_ERR("WeTextNormalizer: Processor init failed: %s", e.what());
        impl_->proc.reset();
        return false;
    }

    loaded_ = true;
    RS_LOG_INFO("WeTextNormalizer: loaded tagger=%s verbalizer=%s",
                tagger_path.c_str(), verbalizer_path.c_str());
    return true;
}

std::string WeTextNormalizer::Normalize(const std::string& text) {
    if (!loaded_ || !impl_->proc) return text;
    try {
        return impl_->proc->Normalize(text);
    } catch (const std::exception& e) {
        RS_LOG_WARN("WeTextNormalizer: Normalize threw: %s — passing through",
                    e.what());
        return text;
    }
}

}  // namespace rs
