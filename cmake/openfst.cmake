# OpenFST as a FetchContent dependency for wetext_processor.
#
# Mirrors `runtime/cmake/openfst.cmake` from wenet-e2e/WeTextProcessing,
# but stripped of the gflags/glog dependency chain — the vendored wetext
# sources route through `third_party/wetext/utils/wetext_log.h`, our shim
# that needs neither glog nor gflags.
#
# Exposes the `fst` target (static library) for consumers.

include(FetchContent)

set(HAVE_BIN       OFF CACHE BOOL "Build the fst binaries" FORCE)
set(HAVE_SCRIPT    OFF CACHE BOOL "Build the fstscript"    FORCE)
set(HAVE_COMPACT   OFF CACHE BOOL "Build compact"          FORCE)
set(HAVE_CONST     OFF CACHE BOOL "Build const"            FORCE)
set(HAVE_GRM       OFF CACHE BOOL "Build grm"              FORCE)
set(HAVE_FAR       OFF CACHE BOOL "Build far"              FORCE)
set(HAVE_PDT       OFF CACHE BOOL "Build pdt"              FORCE)
set(HAVE_MPDT      OFF CACHE BOOL "Build mpdt"             FORCE)
set(HAVE_LINEAR    OFF CACHE BOOL "Build linear"           FORCE)
set(HAVE_LOOKAHEAD OFF CACHE BOOL "Build lookahead"        FORCE)
set(HAVE_NGRAM     OFF CACHE BOOL "Build ngram"            FORCE)
set(HAVE_SPECIAL   OFF CACHE BOOL "Build special"          FORCE)

if(MSVC)
    add_compile_options(/W0 /wd4244 /wd4267 /utf-8)
    # Prevent <windows.h> max/min macros from clobbering
    # std::numeric_limits<...>::max() in openfst/randgen.h.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

FetchContent_Declare(openfst
    URL      https://github.com/csukuangfj/openfst/archive/refs/tags/v1.8.5-2026-06-15.tar.gz
    URL_HASH SHA256=5f9323ded5c9cf4d4e23325dd92652b18b553556ad92b59996e687ebd9688490
)

# Force fst to build as a static archive regardless of the parent project's
# BUILD_SHARED_LIBS. The Python wheel build flips BUILD_SHARED_LIBS=ON, which
# would otherwise produce a libfst.so that rapidspeech-core.so picks up as a
# DT_NEEDED — and downstream test executables fail to link because libfst.so
# isn't on their search path. fst is a purely internal dep of wetext_processor,
# so static is the right default everywhere.
set(_RS_PREV_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
# Position-independent code is still required so the static archive can be
# linked into the eventual rapidspeech-core shared lib / Python module.
set(_RS_PREV_CMAKE_PIC ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

FetchContent_MakeAvailable(openfst)

set(BUILD_SHARED_LIBS ${_RS_PREV_BUILD_SHARED_LIBS})
set(CMAKE_POSITION_INDEPENDENT_CODE ${_RS_PREV_CMAKE_PIC})

# Public include path so consumers can `#include "fst/fst.h"` etc.
if(TARGET fst)
    target_include_directories(fst PUBLIC
        $<BUILD_INTERFACE:${openfst_SOURCE_DIR}/src/include>)
endif()
