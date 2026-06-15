// SPDX-License-Identifier: Apache-2.0
//
// Vendored shim for wenet-e2e/WeTextProcessing.
// Upstream `utils/wetext_log.h` re-exports glog. RapidSpeech does not depend
// on glog, so this header reimplements the subset of glog macros used by the
// vendored sources on top of plain stderr + abort. Kept self-contained so
// `third_party/wetext` has no coupling back to RapidSpeech internals.
//
// Macros provided:
//   LOG(level) << ...   (level ∈ INFO, WARNING, ERROR, FATAL)
//   CHECK(cond)
//   CHECK_EQ / NE / LT / LE / GT / GE
//
// FATAL streams and failed CHECKs print to stderr and then std::abort() —
// matching glog's process-terminating behavior.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

namespace wetext_log_shim {

enum class Severity { kInfo, kWarning, kError, kFatal };

class LogStream {
 public:
  explicit LogStream(Severity sev) : sev_(sev) {}

  LogStream(const LogStream&) = delete;
  LogStream& operator=(const LogStream&) = delete;

  ~LogStream() {
    const char* tag = "INFO";
    switch (sev_) {
      case Severity::kInfo:    tag = "INFO";    break;
      case Severity::kWarning: tag = "WARNING"; break;
      case Severity::kError:   tag = "ERROR";   break;
      case Severity::kFatal:   tag = "FATAL";   break;
    }
    const std::string msg = oss_.str();
    std::fprintf(stderr, "[wetext %s] %s\n", tag, msg.c_str());
    if (sev_ == Severity::kFatal) {
      std::abort();
    }
  }

  template <typename T>
  LogStream& operator<<(const T& v) {
    oss_ << v;
    return *this;
  }

 private:
  Severity sev_;
  std::ostringstream oss_;
};

}  // namespace wetext_log_shim

#define WETEXT_LOG_SHIM_SEVERITY_INFO    ::wetext_log_shim::Severity::kInfo
#define WETEXT_LOG_SHIM_SEVERITY_WARNING ::wetext_log_shim::Severity::kWarning
#define WETEXT_LOG_SHIM_SEVERITY_ERROR   ::wetext_log_shim::Severity::kError
#define WETEXT_LOG_SHIM_SEVERITY_FATAL   ::wetext_log_shim::Severity::kFatal

// OpenFST's fst/log.h defines LOG / CHECK / CHECK_* with different
// semantics. Undef before redefining so the wetext sources see the shim.
#ifdef LOG
#  undef LOG
#endif
#ifdef CHECK
#  undef CHECK
#endif
#ifdef CHECK_EQ
#  undef CHECK_EQ
#endif
#ifdef CHECK_NE
#  undef CHECK_NE
#endif
#ifdef CHECK_LT
#  undef CHECK_LT
#endif
#ifdef CHECK_LE
#  undef CHECK_LE
#endif
#ifdef CHECK_GT
#  undef CHECK_GT
#endif
#ifdef CHECK_GE
#  undef CHECK_GE
#endif

#define LOG(severity) \
  ::wetext_log_shim::LogStream(WETEXT_LOG_SHIM_SEVERITY_##severity)

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      LOG(FATAL) << "CHECK failed: " #cond " at " __FILE__ ":" << __LINE__; \
    }                                                                    \
  } while (0)

#define WETEXT_CHECK_OP(name, op, a, b)                                   \
  do {                                                                    \
    const auto _a = (a);                                                  \
    const auto _b = (b);                                                  \
    if (!(_a op _b)) {                                                    \
      LOG(FATAL) << "CHECK_" #name " failed: " #a " " #op " " #b          \
                 << " (" << _a << " vs " << _b << ") at "                 \
                 << __FILE__ << ":" << __LINE__;                          \
    }                                                                     \
  } while (0)

#define CHECK_EQ(a, b) WETEXT_CHECK_OP(EQ, ==, a, b)
#define CHECK_NE(a, b) WETEXT_CHECK_OP(NE, !=, a, b)
#define CHECK_LT(a, b) WETEXT_CHECK_OP(LT, <,  a, b)
#define CHECK_LE(a, b) WETEXT_CHECK_OP(LE, <=, a, b)
#define CHECK_GT(a, b) WETEXT_CHECK_OP(GT, >,  a, b)
#define CHECK_GE(a, b) WETEXT_CHECK_OP(GE, >=, a, b)
