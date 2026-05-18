cmake_minimum_required(VERSION 3.14)

project(RapidSpeech.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# --- Compilation Options ---
option(RS_BUILD_TESTS "Build test executables" ON)
option(RS_CUDA "Enable CUDA acceleration" OFF)
option(RS_VULKAN "Enable Vulkan acceleration" OFF)
option(RS_CANN "Enable CANN acceleration (Huawei)" OFF)
option(RS_OPENCL "Enable OpenCL acceleration" OFF)
option(RS_ENABLE_PYTHON "Enable Python bindings via pybind11" OFF)
option(RS_BUILD_CLI "Build command-line example executables" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (RS_ENABLE_PYTHON)
    include(pybind11)
endif()

# Auto-detect Apple platform to set default value for Metal
if (APPLE)
    set(RS_METAL_DEFAULT ON)
else()
    set(RS_METAL_DEFAULT OFF)
endif()
option(RS_METAL "Enable Metal acceleration (Apple only)" ${RS_METAL_DEFAULT})

# --- Dependencies Configuration ---
set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/ggml")

if (NOT EXISTS "${GGML_PATH}/CMakeLists.txt")
    message(FATAL_ERROR "ggml not found at ${GGML_PATH}. Please init submodules.")
endif()

# Note: Pass options to ggml before adding the directory
if (RS_CUDA)
    set(GGML_CUDA ON CACHE BOOL "ggml: enable CUDA" FORCE)
endif()

if (RS_METAL AND APPLE)
    set(GGML_METAL ON CACHE BOOL "ggml: enable Metal" FORCE)
endif()

if (RS_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "ggml: enable Vulkan" FORCE)
endif()

if (RS_CANN)
    set(GGML_CANN ON CACHE BOOL "ggml: enable CANN" FORCE)
endif()

if (RS_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "ggml: enable OpenCL" FORCE)
endif()

# Override ggml's install directories so all shared libraries end up
# in the same directory as the pybind module (required for wheel packaging).
# Without this, GNUInstallDirs puts backends in lib/ and the pybind in .
if (RS_ENABLE_PYTHON)
    set(CMAKE_INSTALL_LIBDIR "." CACHE PATH "lib install dir" FORCE)
    set(CMAKE_INSTALL_BINDIR "." CACHE PATH "bin install dir" FORCE)
    set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE PATH "header install dir" FORCE)
    # Disable library SONAME versioning for Python wheel packaging.
    # Versioned sonames (e.g., libfoo.so.1) confuse auditwheel/delocate
    # because the symlink .so.1 -> .so.1.0.0 is not preserved in the wheel.
    set(GGML_VERSION_MAJOR 0 CACHE STRING "" FORCE)
    set(GGML_VERSION_MINOR 10 CACHE STRING "" FORCE)
    set(GGML_VERSION_PATCH 0 CACHE STRING "" FORCE)
    set(GGML_VERSION "0.10.0" CACHE STRING "" FORCE)
    set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

    # Platform-appropriate runtime library search path.
    # macOS uses @loader_path, Linux uses $ORIGIN.
    if (APPLE)
        set(RS_RPATH "@loader_path")
    else()
        set(RS_RPATH "\$ORIGIN")
    endif()
endif()

add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(${GGML_PATH} ggml)

# Suppress implicit integer-to-enum and pointer-to-id conversion warnings in
# third-party ggml Metal code (MTLGPUFamilyMetal*_GGML are defined as integers).
if (TARGET ggml-metal)
    target_compile_options(ggml-metal PRIVATE -Wno-int-conversion)
endif()

# Remove SONAME versioning from ggml targets for Python wheel packaging.
# ggml explicitly sets SOVERSION on its targets, which creates versioned symlinks
# (libggml.so.0 -> libggml.so.0.10.0) that aren't preserved in wheels, causing
# auditwheel/delocate to fail.
if (RS_ENABLE_PYTHON)
    set(_ggml_targets ggml ggml-base ggml-cpu ggml-metal ggml-blas ggml-vulkan)
    foreach(_t IN LISTS _ggml_targets)
        if (TARGET ${_t})
            if (APPLE)
                # On macOS, NO_SONAME ON causes install_name to be the build-tree
                # path, which delocate-wheel cannot resolve. Use INSTALL_NAME_DIR
                # instead so install_name becomes @rpath/libname.dylib.
                set_target_properties(${_t} PROPERTIES
                    SOVERSION ""
                    VERSION ""
                    INSTALL_NAME_DIR "@rpath"
                    INSTALL_RPATH "${RS_RPATH}"
                    BUILD_WITH_INSTALL_RPATH ON
                )
            else()
                set_target_properties(${_t} PROPERTIES
                    SOVERSION ""
                    VERSION ""
                    NO_SONAME ON
                    INSTALL_RPATH "${RS_RPATH}"
                    BUILD_WITH_INSTALL_RPATH ON
                )
            endif()
        endif()
    endforeach()
endif()

# --- Core Source Files ---
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/c_api
)

# Note: LLM-related files are now included for Qwen3 integration
set(RS_SOURCES
        rapidspeech/src/core/rs_context.cpp
        rapidspeech/src/core/rs_processor.cpp
        rapidspeech/src/frontend/audio_processor.cpp
        rapidspeech/src/frontend/fftsg.c
        rapidspeech/src/utils/rs_log.cpp
        rapidspeech/src/arch/sensevoice.cpp
        rapidspeech/src/arch/sensevoice_encoder.cpp
        rapidspeech/src/arch/funasr-nano.cpp
        rapidspeech/src/arch/silero_vad.cpp
        rapidspeech/src/arch/llm_model.h
        rapidspeech/src/arch/llm_model.cpp
        rapidspeech/src/arch/llm_graph.h
        rapidspeech/src/arch/llm_graph.cpp
        rapidspeech/src/arch/llm_kv_cache.h
        rapidspeech/src/arch/llm_kv_cache.cpp
        rapidspeech/src/arch/qwen3.h
        rapidspeech/src/arch/qwen3.cpp
	    rapidspeech/src/arch/openvoice2.cpp
        rapidspeech/src/arch/omnivoice.cpp
        rapidspeech/src/frontend/text_frontend.cpp
#        rapidspeech/src/arch/qwen3_asr.h
#        rapidspeech/src/arch/qwen3_asr.cpp
        rapidspeech/src/utils/rs_wav.cpp
#        rapidspeech/src/arch/mimo_tokenizer.cpp
#        rapidspeech/src/arch/mimo_tokenizer.h
#        rapidspeech/src/arch/mimo_asr.cpp
#        rapidspeech/src/arch/mimo_asr.h
)


# --- Library Compilation ---
add_library(rapidspeech-core SHARED
        ${RS_SOURCES}
        rapidspeech/src/c_api/rapidspeech_c.cpp
)

target_include_directories(rapidspeech-core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/c_api
        ${GGML_PATH}/include
        ${GGML_PATH}/src
)

# Define RAPIDSPEECH_BUILD for proper symbol export
target_compile_definitions(rapidspeech-core PRIVATE RAPIDSPEECH_BUILD)

# MSVC portability: _USE_MATH_DEFINES enables M_PI from <cmath>,
# _CRT_SECURE_NO_WARNINGS suppresses getenv deprecation,
# /wd4244 suppresses int64→int truncation warnings from ggml API.
if (MSVC)
    target_compile_definitions(rapidspeech-core PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    target_compile_options(rapidspeech-core PRIVATE /wd4244)
endif()

# MSVC portability: _USE_MATH_DEFINES enables M_PI from <cmath>,
# _CRT_SECURE_NO_WARNINGS suppresses getenv deprecation warning,
# /wd4244 suppresses int64→int truncation warnings from ggml API.
if (MSVC)
    target_compile_definitions(rapidspeech-core PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    target_compile_options(rapidspeech-core PRIVATE /wd4244)
endif()

# Set library version (skip SOVERSION for Python wheels to avoid auditwheel issues)
if (NOT RS_ENABLE_PYTHON)
    set_target_properties(rapidspeech-core PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
else()
    if (APPLE)
        set_target_properties(rapidspeech-core PROPERTIES
            INSTALL_NAME_DIR "@rpath"
            INSTALL_RPATH "${RS_RPATH}"
            BUILD_WITH_INSTALL_RPATH ON
        )
    else()
        set_target_properties(rapidspeech-core PROPERTIES
            NO_SONAME ON
            INSTALL_RPATH "${RS_RPATH}"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endif()
endif()
set_target_properties(rapidspeech-core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_link_libraries(rapidspeech-core PRIVATE ggml)

# --- Hardware Acceleration Logic ---

# 1. CUDA Configuration
if (RS_CUDA)
    message(STATUS "RapidSpeech: CUDA acceleration enabled.")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_CUDA)
    if (TARGET ggml-cuda)
        target_link_libraries(rapidspeech-core PRIVATE ggml-cuda)
    endif()
endif()

# 2. Metal Configuration (Apple Platforms)
if (APPLE)
    # Always link Metal + Foundation — used by our custom DAC vocoder shader.
    # ggml-metal is not required for our Metal path (we use MTL API directly).
    target_link_libraries(rapidspeech-core PRIVATE "-framework Foundation" "-framework Metal" "-framework Accelerate")

    # Custom Metal DAC vocoder (fused compute kernels, bypasses ggml per-op overhead)
    target_sources(rapidspeech-core PRIVATE rapidspeech/src/arch/dac_metal.mm)
    set_source_files_properties(rapidspeech/src/arch/dac_metal.mm PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_METAL_DAC)
endif()

if (RS_METAL AND APPLE)
    message(STATUS "RapidSpeech: Metal acceleration enabled.")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_METAL)
    if (TARGET ggml-metal)
        target_link_libraries(rapidspeech-core PRIVATE ggml-metal)
    endif()
    target_link_libraries(rapidspeech-core PRIVATE "-framework MetalKit")
endif()

# 3. Vulkan Configuration
if (RS_VULKAN)
    message(STATUS "RapidSpeech: Vulkan acceleration enabled.")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_VULKAN)
    if (TARGET ggml-vulkan)
        target_link_libraries(rapidspeech-core PRIVATE ggml-vulkan)
    endif()
endif()

# 4. CANN Configuration (Huawei Ascend)
if (RS_CANN)
    message(STATUS "RapidSpeech: CANN acceleration enabled.")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_CANN)
    if (TARGET ggml-cann)
        target_link_libraries(rapidspeech-core PRIVATE ggml-cann)
    endif()
endif()

# 5. OpenCL Configuration
if (RS_OPENCL)
    message(STATUS "RapidSpeech: OpenCL acceleration enabled.")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_OPENCL)
    if (TARGET ggml-opencl)
        target_link_libraries(rapidspeech-core PRIVATE ggml-opencl)
    endif()
endif()

if (RS_ENABLE_PYTHON)

    pybind11_add_module(rapidspeech
            rapidspeech/python/pybind_rapidspeech.cpp
    )

    # Install the pybind11 module directly into the package root
    # so that `import rapidspeech` finds it without path tricks.
    install(TARGETS rapidspeech
            LIBRARY DESTINATION .     # Linux/macOS: rapidspeech/*.so
            RUNTIME DESTINATION .     # Windows:     rapidspeech/*.pyd
    )

    # Install rapidspeech-core shared library next to the pybind module
    # so that the OS loader can find it at runtime.
    install(TARGETS rapidspeech-core
            LIBRARY DESTINATION .     # Linux/macOS
            RUNTIME DESTINATION .     # Windows (.dll)
    )

    # ggml shared libraries are installed by ggml's own install rules.
    # We overrode CMAKE_INSTALL_LIBDIR/BINDIR to "." above so they end up
    # in the same directory as the pybind module.

    # Set @rpath relative to install location so dylibs find each other
    # At runtime, @loader_path (macOS) / $ORIGIN (Linux) refers to the
    # directory containing the loading dylib
    set(CMAKE_INSTALL_RPATH "${RS_RPATH}")
    set(CMAKE_BUILD_RPATH "${RS_RPATH}")

    target_link_libraries(rapidspeech
            PRIVATE rapidspeech-core
    )

    target_include_directories(rapidspeech
            PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Linux / macOS：确保符号可见
    set_target_properties(rapidspeech PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
    )

endif()

# --- Example Executables ---
if (RS_BUILD_CLI)
    # Download miniaudio.h if not present (3rd-party single-header library)
    set(MINIAUDIO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/utils/miniaudio.h")
    if (NOT EXISTS "${MINIAUDIO_PATH}")
        message(STATUS "Downloading miniaudio.h ...")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/mackron/miniaudio/refs/tags/0.11.21/miniaudio.h"
            "${MINIAUDIO_PATH}"
            STATUS _ma_status
            TIMEOUT 60
        )
        list(GET _ma_status 0 _ma_code)
        if (NOT _ma_code EQUAL 0)
            message(FATAL_ERROR "Failed to download miniaudio.h. "
                                "Place it at third_party/miniaudio/miniaudio.h manually.")
        endif()
    endif()

    add_executable(rs-asr-offline
            examples/asr/rs-asr-offline.cpp
    )

    add_executable(rs-asr-online
            examples/asr/rs-asr-online.cpp
    )

    add_executable(rs-tts-offline
            examples/tts/rs-tts-offline.cpp
    )

    add_executable(rs-quantize
            examples/quantize/quantize.cpp
            examples/quantize/common_quantize.cpp
    )
    add_executable(rs-imatrix
            examples/imatrix/rs-imatrix.cpp
    )
    target_link_libraries(rs-asr-offline PRIVATE rapidspeech-core ggml)
    target_link_libraries(rs-asr-online  PRIVATE rapidspeech-core ggml)
    target_link_libraries(rs-tts-offline PRIVATE rapidspeech-core ggml)
    target_link_libraries(rs-quantize PRIVATE ggml)
    target_link_libraries(rs-imatrix PRIVATE rapidspeech-core ggml)

    if (NOT MSVC)
        target_link_libraries(rs-quantize PRIVATE pthread)
    endif()

    target_include_directories(rs-quantize PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/quantize
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/imatrix
        ${GGML_PATH}/include
        ${GGML_PATH}/src
    )
    target_include_directories(rs-imatrix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/imatrix
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src
        ${GGML_PATH}/include
        ${GGML_PATH}/src
    )
    target_include_directories(rs-asr-offline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src
        ${GGML_PATH}/include
        ${GGML_PATH}/src
    )

    target_include_directories(rs-asr-online PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src
        ${GGML_PATH}/include
        ${GGML_PATH}/src
    )
    target_include_directories(rs-tts-offline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/rapidspeech/src
        ${GGML_PATH}/include
        ${GGML_PATH}/src
    )
    # miniaudio uses dlopen on Linux for ALSA/PulseAudio/JACK backend discovery
    if(UNIX AND NOT APPLE)
        target_link_libraries(rs-asr-online PRIVATE dl)
    endif()

    install(TARGETS rs-asr-offline
            RUNTIME DESTINATION bin
    )
    install(TARGETS rs-asr-online
            RUNTIME DESTINATION bin
    )
    install(TARGETS rs-tts-offline
            RUNTIME DESTINATION bin
    )
endif()
# --- Tests ---
if (RS_BUILD_TESTS)
    message(STATUS "Tests are enabled.")
    enable_testing()

    set(TEST_MODULES
            audio_frontend
    )

    foreach(module ${TEST_MODULES})
        set(test_target "test_${module}")
        set(test_source "tests/${test_target}.cpp")

        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_source}")
            add_executable(${test_target} ${test_source})
            target_link_libraries(${test_target} PRIVATE rapidspeech-core ggml)
            add_test(NAME ${test_target} COMMAND ${test_target})
        endif()
    endforeach()
endif()