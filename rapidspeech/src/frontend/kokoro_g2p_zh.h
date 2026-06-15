// kokoro_g2p_zh.h — Chinese G2P (ZHG2P + ZHFrontend) for Kokoro v1.1.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cppjieba { class Jieba; }

namespace rs::kokoro_zh {

class PypinyinLite;
class ToneSandhi;

class ZHG2P {
public:
    ZHG2P();
    ~ZHG2P();

    // Lazy-load: jieba dict dir + pypinyin .bin dir.
    bool Load(const std::string& jieba_dict_dir,
              const std::string& pypinyin_data_dir);

    bool IsLoaded() const { return loaded_; }

    // misaki/zh.py:66 __call__ + misaki/zh_frontend.py:156 __call__.
    std::string Process(const std::string& text) const;

    // misaki/zh.py:40 map_punctuation.
    static std::string MapPunctuation(const std::string& text);

    // Simplified an2cn: char-by-char '0'..'9' → '零'..'九'. NOT the full
    // cn2an conversion (no "一百二十三" handling).
    static std::string An2cn(const std::string& text);

    void SetUnk(const std::string& s) { unk_ = s; }
    const std::string& Unk() const { return unk_; }

    // Optional English G2P callback: when ZHG2P encounters a Latin-script
    // segment in mixed text, this gets called with the raw English run and
    // is expected to return space-joined IPA. Empty return → fall back to
    // unk_. If unset, Latin segments always fall back to unk_.
    using EnCallable = std::function<std::string(const std::string&)>;
    void SetEnCallable(EnCallable cb) { en_callable_ = std::move(cb); }

private:
    std::string ProcessZHSegment(const std::string& seg) const;

    // _merge_erhua (zh_frontend.py:118-154).
    void merge_erhua(std::vector<std::string>& initials,
                     std::vector<std::string>& finals,
                     const std::string& word,
                     const std::string& pos) const;

    // Apply the z/c/s -> ii, zh/ch/sh/r -> iii rewrite to a final.
    static std::string discriminate_i(const std::string& initial,
                                      const std::string& final_in);

    bool loaded_ = false;
    std::string unk_ = "❓";

    std::unique_ptr<cppjieba::Jieba> jieba_;
    std::unique_ptr<PypinyinLite> py_;
    std::unique_ptr<ToneSandhi> sandhi_;

    // ZH_MAP from zh_frontend.py:39 + the identity-map characters.
    std::unordered_map<std::string, std::string> zh_map_;
    // Frontend punc (zh_frontend.py:47).
    std::unordered_set<std::string> frontend_punc_;

    std::unordered_set<std::string> must_erhua_;
    std::unordered_set<std::string> not_erhua_;

    EnCallable en_callable_;
};

} // namespace rs::kokoro_zh
