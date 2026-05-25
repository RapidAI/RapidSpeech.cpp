#include "frontend/text_frontend.h"
#include "frontend/cjk_pinyin_table.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

// =====================================================================
// UTF-8 helpers
// =====================================================================

static uint32_t utf8_decode_char(const char*& p, const char* end) {
  if (p >= end) return 0;
  uint8_t c = static_cast<uint8_t>(*p);
  uint32_t cp = 0;
  int extra = 0;

  if (c < 0x80) { cp = c; extra = 0; }
  else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
  else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
  else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
  else { ++p; return 0xFFFD; }  // invalid

  ++p;
  for (int i = 0; i < extra && p < end; i++) {
    cp = (cp << 6) | (static_cast<uint8_t>(*p) & 0x3F);
    ++p;
  }
  return cp;
}

static bool is_chinese_char(uint32_t cp) {
  // CJK Unified Ideographs: U+4E00 - U+9FFF
  // CJK Extension A: U+3400 - U+4DBF
  return (cp >= 0x4E00 && cp <= 0x9FFF) ||
         (cp >= 0x3400 && cp <= 0x4DBF);
}

static std::string to_lower(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(), ::tolower);
  return r;
}

// =====================================================================
// TextFrontend implementation
// =====================================================================

TextFrontend::TextFrontend() {}

bool TextFrontend::Init(const char* vocab_path) {
  if (vocab_path) {
    // Load vocab from file: one line per phoneme, ID = line number
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
      RS_LOG_ERR("Failed to open vocab file: %s", vocab_path);
      return false;
    }
    std::string line;
    int32_t id = 0;
    while (std::getline(f, line)) {
      if (!line.empty()) {
        phoneme_to_id_[line] = id++;
      }
    }
    RS_LOG_INFO("Loaded %d phonemes from %s", id, vocab_path);
  } else {
    InitBuiltinVocab();
  }

  InitChinesePinyin();
  InitEnglishLexicon();
  return true;
}

bool TextFrontend::InitFromSymbols(const std::vector<std::string>& symbols) {
  phoneme_to_id_.clear();
  for (size_t i = 0; i < symbols.size(); i++) {
    phoneme_to_id_[symbols[i]] = static_cast<int32_t>(i);
  }
  RS_LOG_INFO("Loaded %d phonemes from symbol list", (int)symbols.size());

  InitChinesePinyin();
  InitEnglishLexicon();
  return true;
}

void TextFrontend::InitBuiltinVocab() {
  // MeloTTS/OpenVoice2 phoneme set (simplified)
  // Special tokens
  const char* phonemes[] = {
    "<pad>", "<sos>", "<eos>", "<unk>", " ", ".",  ",",  "!",  "?",
    // Vowels (English ARPAbet + Chinese finals)
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
    // Consonants
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",
    // Chinese initials (pinyin)
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
    // Chinese finals (pinyin)
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
    // Tones (1-5, 5=neutral)
    "1", "2", "3", "4", "5",
    // Silence/breath
    "sil", "sp",
  };

  int32_t id = 0;
  for (const char* ph : phonemes) {
    phoneme_to_id_[ph] = id++;
  }
  RS_LOG_INFO("Built-in vocab: %d phonemes", id);
}

void TextFrontend::InitChinesePinyin() {
  // Simplified pinyin table for common Chinese characters
  // In production, this would be a much larger table loaded from file
  // Format: Unicode codepoint -> pinyin with tone number
  static const struct { uint32_t cp; const char* py; } table[] = {
    {0x4F60, "ni3"}, {0x597D, "hao3"}, {0x4E16, "shi4"}, {0x754C, "jie4"},
    {0x6211, "wo3"}, {0x4EEC, "men5"}, {0x662F, "shi4"}, {0x4E2D, "zhong1"},
    {0x56FD, "guo2"}, {0x4EBA, "ren2"}, {0x5927, "da4"},  {0x5B66, "xue2"},
    {0x751F, "sheng1"}, {0x65F6, "shi2"}, {0x5019, "hou4"}, {0x5E74, "nian2"},
    {0x6765, "lai2"}, {0x5C31, "jiu4"}, {0x8FD9, "zhe4"}, {0x90A3, "na4"},
    {0x4E0D, "bu4"}, {0x4E86, "le5"},  {0x7684, "de5"},  {0x5730, "di4"},
    {0x5F97, "de2"}, {0x548C, "he2"},  {0x5728, "zai4"}, {0x6709, "you3"},
    {0x4E3A, "wei4"}, {0x4E0A, "shang4"}, {0x4E0B, "xia4"}, {0x4E00, "yi1"},
    {0x4E8C, "er4"}, {0x4E09, "san1"}, {0x56DB, "si4"},  {0x4E94, "wu3"},
    {0x516D, "liu4"}, {0x4E03, "qi1"},  {0x516B, "ba1"},  {0x4E5D, "jiu3"},
    {0x5341, "shi2"}, {0x767E, "bai3"}, {0x5343, "qian1"}, {0x4E07, "wan4"},
    {0x8BF4, "shuo1"}, {0x8981, "yao4"}, {0x80FD, "neng2"}, {0x4F1A, "hui4"},
    {0x53EF, "ke3"}, {0x4EE5, "yi3"},  {0x6240, "suo3"}, {0x5C31, "jiu4"},
    {0x8FD8, "hai2"}, {0x5F88, "hen3"}, {0x591A, "duo1"}, {0x5C11, "shao3"},
    {0x8BA9, "rang4"}, {0x7ED9, "gei3"}, {0x53BB, "qu4"},  {0x8FC7, "guo4"},
    {0x505A, "zuo4"}, {0x770B, "kan4"}, {0x60F3, "xiang3"}, {0x77E5, "zhi1"},
    {0x9053, "dao4"}, {0x5929, "tian1"}, {0x6C34, "shui3"}, {0x706B, "huo3"},
    {0x5C71, "shan1"}, {0x6728, "mu4"}, {0x91D1, "jin1"}, {0x571F, "tu3"},
    {0x6708, "yue4"}, {0x65E5, "ri4"}, {0x661F, "xing1"}, {0x98CE, "feng1"},
    {0x96E8, "yu3"},  {0x96EA, "xue3"}, {0x82B1, "hua1"}, {0x8349, "cao3"},
    {0x9E1F, "niao3"}, {0x9C7C, "yu2"}, {0x9A6C, "ma3"}, {0x725B, "niu2"},
    {0x7F8A, "yang2"}, {0x732B, "mao1"}, {0x72D7, "gou3"}, {0x7236, "fu4"},
    {0x6BCD, "mu3"}, {0x5144, "xiong1"}, {0x59D0, "jie3"}, {0x5F1F, "di4"},
    {0x59B9, "mei4"}, {0x513F, "er2"}, {0x5973, "nv3"}, {0x7537, "nan2"},
    {0x8001, "lao3"}, {0x5C0F, "xiao3"}, {0x65B0, "xin1"}, {0x957F, "chang2"},
    {0x77ED, "duan3"}, {0x9AD8, "gao1"}, {0x4F4E, "di1"}, {0x5FEB, "kuai4"},
    {0x6162, "man4"}, {0x5DE6, "zuo3"}, {0x53F3, "you4"}, {0x524D, "qian2"},
    {0x540E, "hou4"}, {0x91CC, "li3"}, {0x5916, "wai4"}, {0x5F00, "kai1"},
    {0x5173, "guan1"}, {0x95E8, "men2"}, {0x7A97, "chuang1"},
    // Added: test text + common chars
    {0x6B22, "huan1"}, {0x8FCE, "ying2"}, {0x4F7F, "shi3"}, {0x7528, "yong4"},
    {0x8BED, "yu3"}, {0x97F3, "yin1"}, {0x5408, "he2"}, {0x6210, "cheng2"},
    {0x4E66, "shu1"}, {0x5199, "xie3"}, {0x8BFB, "du2"}, {0x542C, "ting1"},
    {0x8D70, "zou3"}, {0x8DD1, "pao3"}, {0x7761, "shui4"}, {0x5403, "chi1"},
    {0x559D, "he1"}, {0x7231, "ai4"}, {0x6068, "hen4"}, {0x559C, "xi3"},
    {0x6012, "nu4"}, {0x54ED, "ku1"}, {0x7B11, "xiao4"}, {0x6B4C, "ge1"},
    {0x5531, "chang4"}, {0x8DF3, "tiao4"}, {0x821E, "wu3"}, {0x753B, "hua4"},
    {0x5BB6, "jia1"}, {0x56DE, "hui2"}, {0x8FDB, "jin4"}, {0x51FA, "chu1"},
    {0x6253, "da3"}, {0x5EA6, "du4"}, {0x592A, "tai4"}, {0x771F, "zhen1"},
    {0x5047, "jia3"}, {0x7F8E, "mei3"}, {0x4E11, "chou3"},
    {0x70ED, "re4"}, {0x51B7, "leng3"}, {0x6696, "nuan3"}, {0x51C9, "liang2"},
    {0x6625, "chun1"}, {0x590F, "xia4"}, {0x79CB, "qiu1"}, {0x51AC, "dong1"},
    {0x65E9, "zao3"}, {0x665A, "wan3"}, {0x663C, "zhou4"}, {0x591C, "ye4"},
    {0x660E, "ming2"}, {0x6697, "an4"}, {0x8F7B, "qing1"}, {0x91CD, "zhong4"},
    {0x8F6F, "ruan3"}, {0x786C, "ying4"}, {0x6DF1, "shen1"}, {0x6D45, "qian3"},
    {0x7C97, "cu1"}, {0x7EC6, "xi4"}, {0x5BBD, "kuan1"}, {0x7A84, "zhai3"},
    {0x539A, "hou4"}, {0x8584, "bao2"}, {0x65E7, "jiu4"}, {0x5E78, "xing4"},
    {0x798F, "fu2"}, {0x4E50, "le4"}, {0x60B2, "bei1"}, {0x4F24, "shang1"},
    {0x5B89, "an1"}, {0x5168, "quan2"}, {0x5371, "wei1"}, {0x9669, "xian3"},
    {0x529F, "gong1"}, {0x5931, "shi1"}, {0x8D25, "bai4"}, {0x5E0C, "xi1"},
    {0x671B, "wang4"}, {0x613F, "yuan4"}, {0x610F, "yi4"}, {0x601D, "si1"},
    {0x8DEF, "lu4"}, {0x65B9, "fang1"}, {0x5411, "xiang4"}, {0x8FB9, "bian1"},
    {0x95F4, "jian1"}, {0x4E1C, "dong1"}, {0x897F, "xi1"}, {0x5357, "nan2"},
    {0x5317, "bei3"}, {0x8EAB, "shen1"}, {0x4F53, "ti3"}, {0x5065, "jian4"},
    {0x5EB7, "kang1"}, {0x75BE, "ji2"}, {0x75C5, "bing4"}, {0x533B, "yi1"},
    {0x9662, "yuan4"}, {0x6821, "xiao4"}, {0x6559, "jiao4"}, {0x5BA4, "shi4"},
    {0x5E08, "shi1"}, {0x540C, "tong2"}, {0x670B, "peng2"}, {0x53CB, "you3"},
    {0x4EB2, "qin1"}, {0x5988, "ma1"}, {0x7238, "ba4"}, {0x54E5, "ge1"},
    {0x4ED6, "ta1"}, {0x5B83, "ta1"}, {0x81EA, "zi4"}, {0x5DF1, "ji3"},
    {0x522B, "bie2"}, {0x54EA, "na3"}, {0x8C01, "shui2"}, {0x4EC0, "shen2"},
    {0x4E48, "me5"}, {0x600E, "zen3"}, {0x6837, "yang4"}, {0x56E0, "yin1"},
  };

  for (const auto& entry : table) {
    char_to_pinyin_[entry.cp] = entry.py;
  }

  // Comprehensive coverage from pypinyin (CJK Unified + Ext A, ~26K chars).
  // Hand-coded `table` entries above already loaded — keep them as overrides
  // for any common polyphone we know the right reading for. Only insert if
  // not already present.
  for (int i = 0; i < rs::CJK_PINYIN_COUNT; i++) {
    const auto& e = rs::CJK_PINYIN_TABLE[i];
    if (char_to_pinyin_.find(e.cp) == char_to_pinyin_.end()) {
      char_to_pinyin_[e.cp] = e.py;
    }
  }
}

void TextFrontend::InitEnglishLexicon() {
  // Minimal CMU dictionary subset for common words
  static const struct { const char* word; const char* phonemes; } dict[] = {
    {"the",    "DH AH"},
    {"a",      "AH"},
    {"is",     "IH Z"},
    {"are",    "AA R"},
    {"was",    "W AA Z"},
    {"were",   "W ER"},
    {"be",     "B IY"},
    {"have",   "HH AE V"},
    {"has",    "HH AE Z"},
    {"had",    "HH AE D"},
    {"do",     "D UW"},
    {"does",   "D AH Z"},
    {"did",    "D IH D"},
    {"will",   "W IH L"},
    {"would",  "W UH D"},
    {"can",    "K AE N"},
    {"could",  "K UH D"},
    {"should", "SH UH D"},
    {"may",    "M EY"},
    {"might",  "M AY T"},
    {"must",   "M AH S T"},
    {"shall",  "SH AE L"},
    {"i",      "AY"},
    {"you",    "Y UW"},
    {"he",     "HH IY"},
    {"she",    "SH IY"},
    {"it",     "IH T"},
    {"we",     "W IY"},
    {"they",   "DH EY"},
    {"this",   "DH IH S"},
    {"that",   "DH AE T"},
    {"hello",  "HH AH L OW"},
    {"world",  "W ER L D"},
    {"good",   "G UH D"},
    {"morning","M AO R N IH NG"},
    {"thank",  "TH AE NG K"},
    {"please", "P L IY Z"},
    {"yes",    "Y EH S"},
    {"no",     "N OW"},
    {"not",    "N AA T"},
    {"and",    "AE N D"},
    {"or",     "AO R"},
    {"but",    "B AH T"},
    {"in",     "IH N"},
    {"on",     "AA N"},
    {"at",     "AE T"},
    {"to",     "T UW"},
    {"for",    "F AO R"},
    {"with",   "W IH DH"},
    {"from",   "F R AH M"},
    {"of",     "AH V"},
  };

  for (const auto& entry : dict) {
    std::vector<std::string> phs;
    std::istringstream iss(entry.phonemes);
    std::string ph;
    while (iss >> ph) phs.push_back(ph);
    en_lexicon_[entry.word] = phs;
  }
}

// =====================================================================
// G2P methods
// =====================================================================

std::vector<std::string> TextFrontend::ChineseG2P(const std::string& text,
    std::vector<int32_t>* out_tones,
    std::vector<int32_t>* out_word2ph) const {
  std::vector<std::string> phonemes;
  // Track [start, end) of each Chinese character's phonemes so we can apply
  // tone sandhi after collecting all chars. MeloTTS uses pypinyin which
  // applies third-tone sandhi (consecutive 3s → 2,3) word-by-word; we use
  // a simpler pairwise rule which covers the most common case.
  struct ChunkSpan { size_t begin; size_t end; int tone; };
  std::vector<ChunkSpan> hanzi_spans;
  const char* p = text.c_str();
  const char* end = p + text.size();

  while (p < end) {
    uint32_t cp = utf8_decode_char(p, end);
    if (cp == 0) break;

    if (is_chinese_char(cp)) {
      auto it = char_to_pinyin_.find(cp);
      if (it != char_to_pinyin_.end()) {
        size_t begin_idx = phonemes.size();
        // Split pinyin into initial + tone-stripped final + tone digit
        // e.g. "ni3" -> "n" + "i" + tone=3,  "zhuang1" -> "zh" + "uang" + tone=1
        const std::string& py = it->second;
        std::string initial, final;
        int tone = 0;
        const char* initials[] = {"zh", "ch", "sh", "b", "p", "m", "f",
          "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
          "r", "z", "c", "s", "y", "w"};
        for (const char* init : initials) {
          size_t len = strlen(init);
          if (py.compare(0, len, init) == 0) {
            initial = init;
            final = py.substr(len);
            break;
          }
        }
        if (initial.empty()) final = py;
        // Extract tone digit from final
        std::string final_clean;
        for (char c : final) {
          if (c >= '0' && c <= '9') {
            tone = c - '0';
          } else {
            final_clean += c;
          }
        }
        if (final_clean.empty()) final_clean = final;

        // Map apical vowel for zh/ch/sh/r -> ir,  z/c/s -> ih
        if (final_clean == "i") {
          if (initial == "zh" || initial == "ch" || initial == "sh" || initial == "r")
            final_clean = "ir";
          else if (initial == "z" || initial == "c" || initial == "s")
            final_clean = "ih";
        }

        // MeloTTS Chinese G2P: zero-initial syllables (e.g. 爱/安/我) get an
        // explicit `AA` token prepended. Without it, the phoneme count
        // mismatches the model's training-time layout.
        if (initial.empty() && !final_clean.empty()) {
          phonemes.push_back("AA");
          if (out_tones) out_tones->push_back(tone);
        }

        if (!initial.empty()) { phonemes.push_back(initial); if (out_tones) out_tones->push_back(tone); }
        if (!final_clean.empty()) { phonemes.push_back(final_clean); if (out_tones) out_tones->push_back(tone); }

        hanzi_spans.push_back({begin_idx, phonemes.size(), tone});
        if (out_word2ph) out_word2ph->push_back((int32_t)(phonemes.size() - begin_idx));
      } else {
        phonemes.push_back("UNK");
        if (out_tones) out_tones->push_back(0);
        hanzi_spans.push_back({phonemes.size() - 1, phonemes.size(), 0});
        if (out_word2ph) out_word2ph->push_back(1);
      }
    } else if (cp == ' ' || cp == '\t' || cp == '\n') {
      // spaces ignored in Chinese
    } else if (cp == '.' || cp == 0x3002) {  // period or Chinese period
      phonemes.push_back(".");
      if (out_tones) out_tones->push_back(0);
      if (out_word2ph) out_word2ph->push_back(1);
    } else if (cp == ',' || cp == 0xFF0C) {
      phonemes.push_back(",");
      if (out_tones) out_tones->push_back(0);
      if (out_word2ph) out_word2ph->push_back(1);
    } else if (cp == '!' || cp == 0xFF01) {
      phonemes.push_back("!");
      if (out_tones) out_tones->push_back(0);
      if (out_word2ph) out_word2ph->push_back(1);
    } else if (cp == '?' || cp == 0xFF1F) {
      phonemes.push_back("?");
      if (out_tones) out_tones->push_back(0);
      if (out_word2ph) out_word2ph->push_back(1);
    } else if (cp < 0x80 && std::isalpha(static_cast<char>(cp))) {
      // English word embedded in Chinese text — collect full word
      std::string word;
      word += static_cast<char>(cp);
      while (p < end) {
        const char* save = p;
        uint32_t next = utf8_decode_char(p, end);
        if (next < 0x80 && std::isalpha(static_cast<char>(next))) {
          word += static_cast<char>(next);
        } else {
          p = save;  // put back
          break;
        }
      }
      size_t before = phonemes.size();
      auto en_phs = EnglishG2P(word);
      phonemes.insert(phonemes.end(), en_phs.begin(), en_phs.end());
      // Treat the entire English word as a single logical unit. NOTE: BERT
      // WordPiece may split it into multiple subwords; alignment with BERT
      // for mixed Chinese-English text is approximate.
      if (out_word2ph) out_word2ph->push_back((int32_t)(phonemes.size() - before));
    }
  }

  // --- Third-tone sandhi ---
  // Mandarin rule: when two third-tone syllables are adjacent, the first
  // becomes second tone. Without word boundaries we apply pairwise across
  // the sequence (greedy left-to-right). This matches pypinyin's default
  // tone3 sandhi for most short utterances.
  if (out_tones && hanzi_spans.size() >= 2) {
    std::vector<bool> changed(hanzi_spans.size(), false);
    for (size_t i = 0; i + 1 < hanzi_spans.size(); i++) {
      if (changed[i]) continue;
      if (hanzi_spans[i].tone == 3 && hanzi_spans[i + 1].tone == 3) {
        // Change tones of hanzi_spans[i] phonemes from 3 to 2
        for (size_t k = hanzi_spans[i].begin; k < hanzi_spans[i].end; k++) {
          if (k < out_tones->size() && (*out_tones)[k] == 3) {
            (*out_tones)[k] = 2;
          }
        }
        hanzi_spans[i].tone = 2;
        changed[i] = true;
      }
    }
  }


  // Pad out_tones to match phonemes length
  if (out_tones && out_tones->size() < phonemes.size()) {
    out_tones->resize(phonemes.size(), 0);
  }
  return phonemes;
}

std::vector<std::string> TextFrontend::EnglishG2P(const std::string& text) const {
  std::vector<std::string> phonemes;
  std::istringstream iss(text);
  std::string token;

  while (iss >> token) {
    // Strip punctuation from edges
    std::string punct_after;
    while (!token.empty() && std::ispunct(token.back())) {
      punct_after = std::string(1, token.back()) + punct_after;
      token.pop_back();
    }
    while (!token.empty() && std::ispunct(token.front())) {
      token.erase(token.begin());
    }

    if (token.empty()) {
      if (!punct_after.empty()) {
        for (char c : punct_after) phonemes.push_back(std::string(1, c));
      }
      continue;
    }

    std::string lower = to_lower(token);
    std::vector<std::string> word_phs;
    auto it = en_lexicon_.find(lower);
    if (it != en_lexicon_.end()) {
      word_phs = it->second;
    } else {
      word_phs = EnglishRuleFallback(lower);
    }
    // OpenVoice2/MeloTTS English symbols are lowercase ARPABET (hh, ah, l, ow, ...).
    // CMU dict + rule fallback return uppercase, so lowercase before lookup.
    for (auto& ph : word_phs) {
      ph = to_lower(ph);
      phonemes.push_back(ph);
    }

    // Add punctuation
    for (char c : punct_after) {
      phonemes.push_back(std::string(1, c));
    }

    phonemes.push_back("_");  // word boundary (blank token, matches symbol table)
  }


  return phonemes;
}

std::vector<std::string> TextFrontend::EnglishRuleFallback(const std::string& word) const {
  // Very simplified letter-to-phoneme rules
  std::vector<std::string> phonemes;
  for (size_t i = 0; i < word.size(); i++) {
    char c = word[i];
    switch (c) {
      case 'a': phonemes.push_back("AE"); break;
      case 'e': phonemes.push_back("EH"); break;
      case 'i': phonemes.push_back("IH"); break;
      case 'o': phonemes.push_back("AA"); break;
      case 'u': phonemes.push_back("AH"); break;
      case 'b': phonemes.push_back("B"); break;
      case 'c': phonemes.push_back("K"); break;
      case 'd': phonemes.push_back("D"); break;
      case 'f': phonemes.push_back("F"); break;
      case 'g': phonemes.push_back("G"); break;
      case 'h': phonemes.push_back("HH"); break;
      case 'j': phonemes.push_back("JH"); break;
      case 'k': phonemes.push_back("K"); break;
      case 'l': phonemes.push_back("L"); break;
      case 'm': phonemes.push_back("M"); break;
      case 'n': phonemes.push_back("N"); break;
      case 'p': phonemes.push_back("P"); break;
      case 'q': phonemes.push_back("K"); break;
      case 'r': phonemes.push_back("R"); break;
      case 's': phonemes.push_back("S"); break;
      case 't': phonemes.push_back("T"); break;
      case 'v': phonemes.push_back("V"); break;
      case 'w': phonemes.push_back("W"); break;
      case 'x': phonemes.push_back("K"); phonemes.push_back("S"); break;
      case 'y': phonemes.push_back("Y"); break;
      case 'z': phonemes.push_back("Z"); break;
      default: break;
    }
  }

  return phonemes;
}

std::vector<int32_t> TextFrontend::TextToPhonemeIds(const std::string& text,
                                                     const std::string& language,
                                                     std::vector<int32_t>* out_tones,
                                                     std::vector<int32_t>* out_lang_ids,
                                                     int language_id,
                                                     bool add_blank,
                                                     std::vector<int32_t>* out_word2ph) const {
  // Auto-detect Chinese characters: if text contains CJK codepoints, use zh G2P
  // regardless of language parameter (handles the case where language defaults to "English")
  bool has_chinese = false;
  {
    const char* p = text.c_str();
    const char* end = p + text.size();
    while (p < end) {
      uint32_t cp = utf8_decode_char(p, end);
      if (is_chinese_char(cp)) { has_chinese = true; break; }
    }
  }
  bool use_zh = has_chinese || language == "zh" || language == "ZH";

  std::vector<std::string> phonemes;
  std::vector<int32_t> tones_local;
  std::vector<int32_t>* tones_ptr = out_tones ? out_tones : &tones_local;
  tones_ptr->clear();

  std::vector<int32_t> word2ph_local;
  std::vector<int32_t>* w2p_ptr = out_word2ph ? out_word2ph : &word2ph_local;
  w2p_ptr->clear();

  if (use_zh) {
    phonemes = ChineseG2P(text, tones_ptr, w2p_ptr);
  } else {
    phonemes = EnglishG2P(text);
    tones_ptr->assign(phonemes.size(), 0);
    // Fallback: lump all English phonemes under one logical unit. This is
    // only used when BERT alignment is not needed (en-only path).
    if (!phonemes.empty()) w2p_ptr->push_back((int32_t)phonemes.size());
  }

  // Get special IDs
  int32_t blank_id = 0;  // "_"
  auto blank_it = phoneme_to_id_.find("_");
  if (blank_it != phoneme_to_id_.end()) blank_id = blank_it->second;
  int32_t unk_id = 0;
  auto unk_it = phoneme_to_id_.find("UNK");
  if (unk_it != phoneme_to_id_.end()) unk_id = unk_it->second;

  // Step 1: Add leading/trailing `_` to match MeloTTS g2p output convention.
  std::vector<int32_t> raw_ids;
  std::vector<int32_t> raw_tones;
  std::vector<int32_t> raw_langs;
  raw_ids.reserve(phonemes.size() + 2);
  raw_tones.reserve(phonemes.size() + 2);
  raw_langs.reserve(phonemes.size() + 2);

  raw_ids.push_back(blank_id);
  raw_tones.push_back(0);
  raw_langs.push_back(language_id);
  for (size_t i = 0; i < phonemes.size(); i++) {
    auto it = phoneme_to_id_.find(phonemes[i]);
    raw_ids.push_back(it != phoneme_to_id_.end() ? it->second : unk_id);
    raw_tones.push_back(i < tones_ptr->size() ? (*tones_ptr)[i] : 0);
    raw_langs.push_back(language_id);
  }
  raw_ids.push_back(blank_id);
  raw_tones.push_back(0);
  raw_langs.push_back(language_id);

  // Step 2: If add_blank, intersperse `_` (id=0) between every pair AND at boundaries.
  // MeloTTS commons.intersperse: result has length 2N+1, with blanks at even indices.
  // Tones and lang_ids for the interspersed blanks are 0.
  std::vector<int32_t> ids;
  if (add_blank) {
    size_t n = raw_ids.size();
    ids.reserve(2 * n + 1);
    if (out_tones) { out_tones->clear(); out_tones->reserve(2 * n + 1); }
    if (out_lang_ids) { out_lang_ids->clear(); out_lang_ids->reserve(2 * n + 1); }
    ids.push_back(blank_id);
    if (out_tones) out_tones->push_back(0);
    if (out_lang_ids) out_lang_ids->push_back(0);
    for (size_t i = 0; i < n; i++) {
      ids.push_back(raw_ids[i]);
      if (out_tones) out_tones->push_back(raw_tones[i]);
      if (out_lang_ids) out_lang_ids->push_back(raw_langs[i]);
      ids.push_back(blank_id);
      if (out_tones) out_tones->push_back(0);
      if (out_lang_ids) out_lang_ids->push_back(0);
    }
  } else {
    ids = std::move(raw_ids);
    if (out_tones) *out_tones = std::move(raw_tones);
    if (out_lang_ids) *out_lang_ids = std::move(raw_langs);
  }

  // Post-process word2ph: prepend 1 for [CLS] and append 1 for [SEP] (MeloTTS
  // convention so word2ph aligns with BERT subwords including specials).
  // If add_blank: word2ph *= 2 (each entry now spans both the phoneme and
  // the interspersed blank that follows), then word2ph[0] += 1 to account
  // for the extra leading blank inserted at index 0.
  if (out_word2ph) {
    std::vector<int32_t> w2p;
    w2p.reserve(w2p_ptr->size() + 2);
    w2p.push_back(1);  // [CLS]
    for (int32_t c : *w2p_ptr) w2p.push_back(c);
    w2p.push_back(1);  // [SEP]
    if (add_blank) {
      for (auto& c : w2p) c *= 2;
      w2p[0] += 1;
    }
    *out_word2ph = std::move(w2p);
  }

  return ids;
}

