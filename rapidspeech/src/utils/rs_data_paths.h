#pragma once

#include <string>
#include <vector>

namespace rs {

// Locate a runtime data subdirectory shipped alongside the binary.
//
// Resolution order (first existing wins):
//   1. $env_var (if set and non-empty)
//   2. <exe_dir>/../share/rapidspeech/<rel>      (FHS / release tarball)
//   3. <exe_dir>/share/rapidspeech/<rel>
//   4. <exe_dir>/data/<rel>                      (portable layout)
//   5. <CWD>/rapidspeech/data/<rel>              (dev tree)
//      OR, when rel starts with "cppjieba/",     <CWD>/third_party/<rel>
//   6. $HOME/.cache/rapidspeech/<rel>            (user override)
//
// Returns "" if no candidate exists. On success, returns the absolute path.
//
// `rel` examples: "kokoro_zh", "kokoro_en", "wetext", "cppjieba/dict".
// `env_var` examples: "RS_KOKORO_ZH_DATA_DIR" (nullptr ⇒ no env override).
//
// If `tried_out` is non-null, every candidate path is appended to it so the
// caller can include them in an error message.
std::string find_data_dir(const std::string &rel, const char *env_var,
                          std::vector<std::string> *tried_out = nullptr);

// Return the directory containing the current executable (no trailing slash),
// or "" if it cannot be determined.
std::string get_executable_dir();

} // namespace rs
