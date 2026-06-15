// kokoro_g2p_en.cpp — see header for overview.
//
// Binary format mirrors PypinyinLite's PKYS/PKYP scheme:
//   header  : <I magic=0x4E45474D> <I version=1> <I count>
//   record  : <B key_len> <key utf-8> <B ipa_len> <ipa utf-8>
//
// No string interning, no compression — the gold dict is ~2.2 MB on disk
// and lives in std::unordered_map after Load.

#include "frontend/kokoro_g2p_en.h"
#include "utils/rs_log.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace rs::kokoro_en {

namespace {

constexpr uint32_t kMagic = 0x4E45474D;  // 'MGEN'

bool read_u32(std::ifstream& in, uint32_t& v) {
    return (bool)in.read(reinterpret_cast<char*>(&v), 4);
}
bool read_u8(std::ifstream& in, uint8_t& v) {
    return (bool)in.read(reinterpret_cast<char*>(&v), 1);
}
bool read_bytes(std::ifstream& in, std::string& s, size_t n) {
    s.resize(n);
    return (bool)in.read(s.data(), (std::streamsize)n);
}

std::string to_lower_ascii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return out;
}

std::string to_upper_ascii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    }
    return out;
}

bool is_token_inner(char c) {
    // tokens hold ASCII letters, digits, and intra-word ' / -
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9') || c == '\'' || c == '-';
}

bool is_token_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

}  // namespace

EnG2P::EnG2P() = default;

bool EnG2P::Load(const std::string& data_dir) {
    loaded_ = false;
    gold_.clear();

    const std::string path = data_dir + "/us_gold.bin";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        RS_LOG_WARN("EnG2P: cannot open '%s'", path.c_str());
        return false;
    }

    uint32_t magic = 0, version = 0, count = 0;
    if (!read_u32(in, magic) || !read_u32(in, version) || !read_u32(in, count)) {
        RS_LOG_ERR("EnG2P: short header in '%s'", path.c_str());
        return false;
    }
    if (magic != kMagic) {
        RS_LOG_ERR("EnG2P: bad magic 0x%08x in '%s' (expected MGEN)",
                   magic, path.c_str());
        return false;
    }
    if (version != 1) {
        RS_LOG_ERR("EnG2P: unsupported version %u in '%s'",
                   version, path.c_str());
        return false;
    }

    gold_.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t kl = 0, vl = 0;
        std::string key, ipa;
        if (!read_u8(in, kl) || !read_bytes(in, key, kl)
            || !read_u8(in, vl) || !read_bytes(in, ipa, vl)) {
            RS_LOG_ERR("EnG2P: truncated record #%u in '%s'", i, path.c_str());
            return false;
        }
        gold_.emplace(std::move(key), std::move(ipa));
    }

    loaded_ = true;
    RS_LOG_INFO("EnG2P: loaded %u entries from %s",
                (unsigned)gold_.size(), path.c_str());
    return true;
}

std::vector<std::string> EnG2P::Tokenize(const std::string& en_run) {
    std::vector<std::string> out;
    const size_t n = en_run.size();
    size_t i = 0;
    while (i < n) {
        while (i < n && !is_token_start(en_run[i])) ++i;
        size_t j = i;
        while (j < n && is_token_inner(en_run[j])) ++j;
        if (j > i) out.emplace_back(en_run.substr(i, j - i));
        i = j;
    }
    return out;
}

std::string EnG2P::LookupOrSpell(const std::string& word) const {
    if (word.empty()) return "";

    // 1) Verbatim
    if (auto it = gold_.find(word); it != gold_.end()) return it->second;

    // 2) Lowercase
    const std::string lower = to_lower_ascii(word);
    if (lower != word) {
        if (auto it = gold_.find(lower); it != gold_.end()) return it->second;
    }

    // 3) Letter-by-letter (acronym path). Uppercase single-letter entries
    //    exist in misaki gold (A, B, ..., Z, plus digits 0-9 in their
    //    spoken forms).
    std::string spelled;
    for (char c : word) {
        char up = c;
        if (up >= 'a' && up <= 'z') up = (char)(up - 'a' + 'A');
        if (!((up >= 'A' && up <= 'Z') || (up >= '0' && up <= '9'))) continue;
        std::string key(1, up);
        auto it = gold_.find(key);
        if (it == gold_.end()) continue;
        if (!spelled.empty()) spelled += ' ';
        spelled += it->second;
    }
    return spelled;  // may be empty if nothing landed
}

std::string EnG2P::Process(const std::string& en_run) const {
    if (!loaded_ || en_run.empty()) return "";

    auto toks = Tokenize(en_run);
    if (toks.empty()) return "";

    std::string out;
    for (auto& t : toks) {
        std::string ipa = LookupOrSpell(t);
        if (ipa.empty()) continue;
        if (!out.empty()) out += ' ';
        out += ipa;
    }
    return out;
}

}  // namespace rs::kokoro_en
