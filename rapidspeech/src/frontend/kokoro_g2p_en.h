#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace rs::kokoro_en {

// Minimal English G2P used as the en_callable for ZHG2P (Latin-script
// segments inside otherwise-Chinese text). Pure lookup against a vendored
// misaki[en] gold dictionary; OOD words fall back to letter-by-letter
// spelling using the same dict's single-letter entries.
//
// Output uses misaki's IPA conventions (including the uppercase
// pseudo-vowels A/I/O and stress markers ˈ ˌ ː). These are all present
// in Kokoro v1.1-zh's 178-symbol vocab.
class EnG2P {
public:
    EnG2P();

    // Loads us_gold.bin (magic 0x4E45474D) from `data_dir`. Returns false
    // on missing file or bad header; IsLoaded() then stays false.
    bool Load(const std::string& data_dir);
    bool IsLoaded() const { return loaded_; }

    // Tokenises the English run, looks each token up in gold, joins the
    // IPA results with spaces. Returns an empty string when the wrapper
    // is not loaded (the caller is then free to emit an unk marker).
    std::string Process(const std::string& en_run) const;

private:
    // Token-level lookup: try as-is, then lowercase, then letter-spell.
    // Returns empty string if even the spell fallback can't produce
    // anything.
    std::string LookupOrSpell(const std::string& word) const;

    // Splits an English run on non-token characters. Keeps ' and - inside
    // tokens ("'em", "co-op"). Drops digits and other punctuation.
    static std::vector<std::string> Tokenize(const std::string& en_run);

    std::unordered_map<std::string, std::string> gold_;
    bool loaded_ = false;
};

}  // namespace rs::kokoro_en
