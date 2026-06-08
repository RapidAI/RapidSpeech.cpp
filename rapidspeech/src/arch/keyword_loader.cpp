#include "keyword_loader.h"

#include "utils/rs_log.h"

#include <fstream>
#include <sstream>

namespace rs {

namespace {

// Build the token→id reverse map once, since id_to_token is the canonical
// direction loaded from the gguf and we need to look up by string.
std::unordered_map<std::string, int> BuildTokenToId(
    const std::unordered_map<int, std::string> &id_to_token) {
  std::unordered_map<std::string, int> m;
  m.reserve(id_to_token.size());
  for (const auto &kv : id_to_token) {
    m.emplace(kv.second, kv.first);
  }
  return m;
}

} // namespace

ContextGraphPtr LoadKeywordsFromString(
    const std::string &content,
    const std::unordered_map<int, std::string> &id_to_token,
    const KWSLoaderConfig &cfg) {
  const auto tok2id = BuildTokenToId(id_to_token);

  std::vector<std::vector<int32_t>> ids_list;
  std::vector<std::string> phrases;
  std::vector<float> scores;
  std::vector<float> thresholds;

  std::istringstream is(content);
  std::string line;
  int line_no = 0;
  while (std::getline(is, line)) {
    ++line_no;
    if (line.empty()) continue;

    std::vector<int32_t> ids;
    std::string phrase;
    float score = 0.0f;
    float threshold = 0.0f;
    bool oov = false;

    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) {
      if (tok.empty()) continue;
      switch (tok[0]) {
        case ':':
          try {
            score = std::stof(tok.substr(1));
          } catch (...) {
            RS_LOG_ERR("keywords:%d: bad boost score '%s'", line_no, tok.c_str());
            oov = true;
          }
          break;
        case '#':
          try {
            threshold = std::stof(tok.substr(1));
          } catch (...) {
            RS_LOG_ERR("keywords:%d: bad threshold '%s'", line_no, tok.c_str());
            oov = true;
          }
          break;
        case '@':
          phrase = tok.substr(1);
          break;
        default: {
          auto it = tok2id.find(tok);
          if (it == tok2id.end()) {
            RS_LOG_ERR("keywords:%d: unknown token '%s' (skipping line)",
                       line_no, tok.c_str());
            oov = true;
          } else {
            ids.push_back(it->second);
          }
          break;
        }
      }
    }

    if (oov || ids.empty()) continue;

    ids_list.push_back(std::move(ids));
    phrases.push_back(phrase.empty() ? std::string() : phrase);
    scores.push_back(score);
    thresholds.push_back(threshold);
  }

  if (ids_list.empty()) {
    RS_LOG_ERR("LoadKeywords: no valid keyword lines parsed");
    return nullptr;
  }

  return std::make_shared<ContextGraph>(ids_list, cfg.default_score,
                                        cfg.default_threshold, scores, phrases,
                                        thresholds);
}

ContextGraphPtr LoadKeywordsFromFile(
    const std::string &path,
    const std::unordered_map<int, std::string> &id_to_token,
    const KWSLoaderConfig &cfg) {
  std::ifstream f(path);
  if (!f) {
    RS_LOG_ERR("LoadKeywords: cannot open %s", path.c_str());
    return nullptr;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return LoadKeywordsFromString(ss.str(), id_to_token, cfg);
}

} // namespace rs
