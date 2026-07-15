#include "hotword_retriever.h"

#include "frontend/kokoro_pinyin.h"
#include "utils/rs_data_paths.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace rs {

HotwordRetriever::HotwordRetriever() = default;
HotwordRetriever::~HotwordRetriever() = default;

bool HotwordRetriever::Load() {
  if (load_attempted_)
    return loaded_;
  load_attempted_ = true;

  std::vector<std::string> tried;
  std::string zh_dir = find_data_dir("kokoro_zh", "RS_KOKORO_ZH_DATA_DIR", &tried);
  if (zh_dir.empty()) {
    std::string joined;
    for (auto &p : tried) {
      joined += "\n    ";
      joined += p;
    }
    RS_LOG_WARN("HotwordRetriever: kokoro_zh pinyin data not found; falling back "
                "to character-level matching (Chinese homophones won't be "
                "corrected). Searched:%s\n  Set RS_KOKORO_ZH_DATA_DIR to enable "
                "pinyin retrieval.",
                joined.c_str());
    loaded_ = false;
    return false;
  }

  py_ = std::make_unique<kokoro_zh::PypinyinLite>();
  if (!py_->Load(zh_dir)) {
    RS_LOG_WARN("HotwordRetriever: failed to load pinyin tables from %s; "
                "falling back to character-level matching.",
                zh_dir.c_str());
    py_.reset();
    loaded_ = false;
    return false;
  }

  RS_LOG_INFO("HotwordRetriever: loaded pinyin tables from %s", zh_dir.c_str());
  loaded_ = true;
  return true;
}

// Normalize one raw token from the pinyin scan:
//   - lowercase ASCII letters
//   - strip the trailing tone digit(s) so cross-tone homophones match
//     ("shi4" -> "shi", "an1" -> "an")
//   - drop pure punctuation / whitespace tokens
// Returns "" for tokens that should be skipped.
static std::string normalize_token(const std::string &tok) {
  std::string out;
  out.reserve(tok.size());
  bool has_letter_or_han = false;
  for (unsigned char c : tok) {
    if (c < 0x80) {
      if (std::isalpha(c)) {
        out.push_back((char)std::tolower(c));
        has_letter_or_han = true;
      } else if (std::isdigit(c)) {
        // tone digit or numeral — keep for now, trimmed below
        out.push_back((char)c);
      }
      // ASCII punctuation / whitespace: dropped
    } else {
      // Non-ASCII byte (Han codepoint returned verbatim when pinyin missing).
      out.push_back((char)c);
      has_letter_or_han = true;
    }
  }
  // Strip trailing tone digits (only when there is an alphabetic/Han body).
  if (has_letter_or_han) {
    while (!out.empty() && (unsigned char)out.back() < 0x80 &&
           std::isdigit((unsigned char)out.back())) {
      out.pop_back();
    }
  }
  if (out.empty() || !has_letter_or_han)
    return "";
  return out;
}

std::vector<std::string>
HotwordRetriever::ToTokens(const std::string &text) const {
  std::vector<std::string> raw;
  if (loaded_ && py_) {
    raw = py_->LazyPinyinTone3(text);
  } else {
    // Character-level fallback: one UTF-8 codepoint per token.
    for (uint32_t cp : kokoro_zh::utf8_codepoints(text)) {
      std::string s;
      kokoro_zh::utf8_encode(cp, s);
      raw.push_back(std::move(s));
    }
  }

  std::vector<std::string> out;
  out.reserve(raw.size());
  for (const auto &t : raw) {
    std::string n = normalize_token(t);
    if (!n.empty())
      out.push_back(std::move(n));
  }
  return out;
}

void HotwordRetriever::SetHotwords(const std::vector<std::string> &hotwords) {
  entries_.clear();
  entries_.reserve(hotwords.size());
  for (const auto &hw : hotwords) {
    if (hw.empty())
      continue;
    Entry e;
    e.text = hw;
    e.tokens = ToTokens(hw);
    if (e.tokens.empty()) {
      RS_LOG_WARN("HotwordRetriever: hotword '%s' produced no match tokens; "
                  "skipping.",
                  hw.c_str());
      continue;
    }
    entries_.push_back(std::move(e));
  }
  RS_LOG_INFO("HotwordRetriever: indexed %zu/%zu hotwords (pinyin=%s)",
              entries_.size(), hotwords.size(), loaded_ ? "yes" : "no");
}

// Minimum edit distance of pattern `p` against ANY contiguous substring of
// `t` (approximate substring match). First DP row is all zeros so the match may
// start anywhere in `t`; the answer takes the min over all end positions.
static int approx_substring_distance(const std::vector<std::string> &p,
                                     const std::vector<std::string> &t) {
  const int m = (int)p.size();
  const int n = (int)t.size();
  if (m == 0)
    return 0;
  if (n == 0)
    return m;

  std::vector<int> prev(n + 1, 0), curr(n + 1, 0);
  for (int i = 1; i <= m; ++i) {
    curr[0] = i; // deleting all of pattern's prefix costs i
    for (int j = 1; j <= n; ++j) {
      int cost = (p[i - 1] == t[j - 1]) ? 0 : 1;
      curr[j] = std::min({prev[j] + 1,        // deletion in pattern
                          curr[j - 1] + 1,     // insertion (skip text token)
                          prev[j - 1] + cost}); // match / substitution
    }
    std::swap(prev, curr);
  }
  int best = prev[0];
  for (int j = 1; j <= n; ++j)
    best = std::min(best, prev[j]);
  return best;
}

std::vector<std::string> HotwordRetriever::Retrieve(const std::string &ctc_text,
                                                    float tau, int topk) const {
  std::vector<std::string> result;
  if (entries_.empty())
    return result;

  std::vector<std::string> ctc_tokens = ToTokens(ctc_text);
  if (ctc_tokens.empty())
    return result;

  const bool dbg = std::getenv("RS_HOTWORD_DEBUG") != nullptr;
  if (dbg) {
    std::string joined;
    for (const auto &t : ctc_tokens) {
      joined += t;
      joined += ' ';
    }
    RS_LOG_INFO("[hotword] CTC pinyin: %s", joined.c_str());
  }

  struct Hit {
    float dist;
    const std::string *text;
  };
  std::vector<Hit> hits;
  for (const auto &e : entries_) {
    int edits = approx_substring_distance(e.tokens, ctc_tokens);
    float d = (float)edits / (float)e.tokens.size();
    if (dbg) {
      std::string joined;
      for (const auto &t : e.tokens) {
        joined += t;
        joined += ' ';
      }
      RS_LOG_INFO("[hotword] '%s' pinyin=[%s] edits=%d d=%.3f tau=%.3f -> %s",
                  e.text.c_str(), joined.c_str(), edits, d, tau,
                  (d <= tau ? "HIT" : "miss"));
    }
    if (d <= tau)
      hits.push_back({d, &e.text});
  }

  std::stable_sort(hits.begin(), hits.end(),
                   [](const Hit &a, const Hit &b) { return a.dist < b.dist; });

  std::vector<std::string> seen;
  for (const auto &h : hits) {
    if (std::find(seen.begin(), seen.end(), *h.text) != seen.end())
      continue;
    seen.push_back(*h.text);
    result.push_back(*h.text);
    if (topk > 0 && (int)result.size() >= topk)
      break;
  }
  return result;
}

std::string
HotwordRetriever::ApplySubstitutions(const std::string &text) const {
  if (entries_.empty() || text.empty())
    return text;

  // Byte offset + length of each UTF-8 codepoint.
  std::vector<size_t> cp_start, cp_len;
  for (size_t i = 0; i < text.size();) {
    size_t s = i;
    (void)kokoro_zh::utf8_decode(text, i); // advances i past the codepoint
    cp_start.push_back(s);
    cp_len.push_back(i - s);
  }
  const size_t N = cp_start.size();

  // Per-codepoint normalized (toneless) pinyin. LazyPinyinTone3 yields exactly
  // one syllable per codepoint; punctuation/non-Han come back verbatim and
  // normalize to their own string (never equal to a hotword syllable, so they
  // act as barriers). Fall back to per-char normalization if unavailable.
  std::vector<std::string> py(N);
  bool aligned = false;
  if (loaded_ && py_) {
    std::vector<std::string> raw = py_->LazyPinyinTone3(text);
    if (raw.size() == N) {
      for (size_t k = 0; k < N; ++k)
        py[k] = normalize_token(raw[k]);
      aligned = true;
    }
  }
  if (!aligned) {
    for (size_t k = 0; k < N; ++k)
      py[k] = normalize_token(text.substr(cp_start[k], cp_len[k]));
  }

  // Longest hotword first so a longer name wins over a contained shorter one.
  std::vector<const Entry *> order;
  order.reserve(entries_.size());
  for (const auto &e : entries_)
    order.push_back(&e);
  std::stable_sort(order.begin(), order.end(),
                   [](const Entry *a, const Entry *b) {
                     return a->tokens.size() > b->tokens.size();
                   });

  // Collect non-overlapping matches (exact pinyin equality only).
  struct Span {
    size_t b, e;
    const std::string *repl;
  };
  std::vector<Span> spans;
  std::vector<char> used(N, 0);
  for (const Entry *e : order) {
    const size_t m = e->tokens.size();
    if (m == 0 || m > N)
      continue;
    for (size_t i = 0; i + m <= N; ++i) {
      bool ok = true;
      for (size_t k = 0; k < m; ++k) {
        if (used[i + k] || py[i + k].empty() || py[i + k] != e->tokens[k]) {
          ok = false;
          break;
        }
      }
      if (ok) {
        for (size_t k = 0; k < m; ++k)
          used[i + k] = 1;
        spans.push_back({i, i + m, &e->text});
        i += m - 1;
      }
    }
  }
  if (spans.empty())
    return text;

  std::sort(spans.begin(), spans.end(),
            [](const Span &a, const Span &b) { return a.b < b.b; });

  // Rebuild the string, splicing hotword text over matched codepoint runs.
  std::string out;
  out.reserve(text.size());
  size_t k = 0, si = 0;
  while (k < N) {
    if (si < spans.size() && k == spans[si].b) {
      out += *spans[si].repl;
      k = spans[si].e;
      ++si;
    } else {
      out.append(text, cp_start[k], cp_len[k]);
      ++k;
    }
  }
  return out;
}

} // namespace rs