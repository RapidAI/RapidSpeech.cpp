// rs_data_paths.cpp — runtime data directory discovery.
//
// Implements find_data_dir() which lets the binary locate the G2P, WeText,
// and cppjieba dictionary data shipped alongside it. Resolution order is
// documented in the header.

#include "utils/rs_data_paths.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <climits>
#else
#  include <climits>
#  include <unistd.h>
#endif

namespace rs {

namespace {

bool is_dir(const std::string &p) {
    if (p.empty()) return false;
    struct stat st {};
    if (::stat(p.c_str(), &st) != 0) return false;
    return (st.st_mode & S_IFDIR) != 0;
}

std::string dirname_of(const std::string &p) {
    if (p.empty()) return "";
    const auto pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return "";
    return p.substr(0, pos);
}

} // anon

std::string get_executable_dir() {
#if defined(_WIN32)
    char buf[MAX_PATH];
    DWORD n = ::GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return "";
    return dirname_of(std::string(buf, n));
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buf(size + 1, 0);
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return "";
    // _NSGetExecutablePath may return a path with embedded "..". realpath() it.
    char resolved[PATH_MAX];
    if (::realpath(buf.data(), resolved) == nullptr) {
        return dirname_of(std::string(buf.data()));
    }
    return dirname_of(std::string(resolved));
#else
    char resolved[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", resolved, sizeof(resolved) - 1);
    if (n <= 0) return "";
    resolved[n] = '\0';
    return dirname_of(std::string(resolved));
#endif
}

std::string find_data_dir(const std::string &rel, const char *env_var,
                          std::vector<std::string> *tried_out) {
    auto try_path = [&](const std::string &p) -> std::string {
        if (p.empty()) return "";
        if (tried_out) tried_out->push_back(p);
        return is_dir(p) ? p : std::string();
    };

    // 1. env var override
    if (env_var) {
        if (const char *v = std::getenv(env_var)) {
            if (v[0]) {
                if (auto hit = try_path(v); !hit.empty()) return hit;
            }
        }
    }

    const std::string exe = get_executable_dir();

    // 2. <exe>/../share/rapidspeech/<rel>   (FHS / tarball)
    if (!exe.empty()) {
        if (auto hit = try_path(exe + "/../share/rapidspeech/" + rel); !hit.empty()) return hit;
        // 3. <exe>/share/rapidspeech/<rel>
        if (auto hit = try_path(exe + "/share/rapidspeech/" + rel); !hit.empty()) return hit;
        // 4. <exe>/data/<rel>   (portable)
        if (auto hit = try_path(exe + "/data/" + rel); !hit.empty()) return hit;
    }

    // 5. dev tree relative to CWD
    if (rel.rfind("cppjieba/", 0) == 0) {
        if (auto hit = try_path("third_party/" + rel); !hit.empty()) return hit;
    } else {
        if (auto hit = try_path("rapidspeech/data/" + rel); !hit.empty()) return hit;
    }

    // 6. $HOME/.cache/rapidspeech/<rel>
    if (const char *home = std::getenv("HOME")) {
        if (home[0]) {
            if (auto hit = try_path(std::string(home) + "/.cache/rapidspeech/" + rel); !hit.empty())
                return hit;
        }
    }
#if defined(_WIN32)
    if (const char *app = std::getenv("LOCALAPPDATA")) {
        if (app[0]) {
            if (auto hit = try_path(std::string(app) + "\\rapidspeech\\" + rel); !hit.empty())
                return hit;
        }
    }
#endif

    return "";
}

} // namespace rs
