cmake_minimum_required(VERSION 3.14)
project(rapidspeech-wasm VERSION 0.1.0 LANGUAGES C CXX)

# ── Emscripten-specific settings ─────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable all GPU backends for WASM
set(RS_CUDA OFF CACHE BOOL "" FORCE)
set(RS_METAL OFF CACHE BOOL "" FORCE)
set(RS_VULKAN OFF CACHE BOOL "" FORCE)
set(RS_CANN OFF CACHE BOOL "" FORCE)
set(RS_OPENCL OFF CACHE BOOL "" FORCE)
set(RS_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(RS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)

set(GGML_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml")

if(NOT EXISTS "${GGML_PATH}/CMakeLists.txt")
    message(FATAL_ERROR "ggml not found at ${GGML_PATH}")
endif()

# Pass Emscripten flags to ggml
set(GGML_WASM_SINGLE_FILE ON CACHE BOOL "ggml: single-file WASM" FORCE)

add_subdirectory(${GGML_PATH} ggml)

# ── RapidSpeech core sources (subset needed for ASR) ─────────
set(RS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/rs_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/core/rs_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/frontend/audio_processor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/frontend/fftsg.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils/rs_log.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/sensevoice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/sensevoice_encoder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/funasr-nano.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/arch/silero_vad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/c_api/rapidspeech_c.cpp
)

# ── Static library ───────────────────────────────────────────
add_library(rapidspeech-core-wasm STATIC ${RS_SOURCES})

target_include_directories(rapidspeech-core-wasm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/c_api
    ${GGML_PATH}/include
    ${GGML_PATH}/src
)

target_compile_definitions(rapidspeech-core-wasm PRIVATE RAPIDSPEECH_BUILD)
target_link_libraries(rapidspeech-core-wasm PRIVATE ggml)

# ── WASM executable ──────────────────────────────────────────
add_executable(rapidspeech-wasm
    rapidspeech_wasm.cpp
)

target_include_directories(rapidspeech-wasm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${GGML_PATH}/include
)

target_link_libraries(rapidspeech-wasm PRIVATE rapidspeech-core-wasm ggml)

# ── Emscripten link flags ────────────────────────────────────
set_target_properties(rapidspeech-wasm PROPERTIES
    SUFFIX ".js"
)

target_link_options(rapidspeech-wasm PRIVATE
    # Memory
    -sINITIAL_MEMORY=256MB
    -sALLOW_MEMORY_GROWTH=1
    -sSTACK_SIZE=10MB

    # Enable WASM SIMD for better performance
    -msimd128

    # Filesystem support (for reading .gguf model)
    -sFORCE_FILESYSTEM=1

    # Export the WASM entry-point functions to JS
    -sEXPORTED_FUNCTIONS=[_rs_wasm_init,_rs_wasm_free,_rs_wasm_push_audio,_rs_wasm_process,_rs_wasm_get_text,_rs_wasm_reset,_rs_wasm_get_sample_rate,_rs_wasm_get_arch_name,_rs_wasm_is_ready,_rs_wasm_get_version,_malloc,_free]
    -sEXPORTED_RUNTIME_METHODS=[ccall,cwrap,UTF8ToString,stringToUTF8,lengthBytesUTF8,getValue,setValue,FS]

    # No closure compiler (easier debugging)
    --closure 0

    # Modularize for clean global scope
    -sMODULARIZE=1
    -sEXPORT_NAME=RapidSpeechModule

    # Async model loading
    -sASYNCIFY=1
)

# If building with pthreads support, add:
# target_link_options(rapidspeech-wasm PRIVATE -pthread -sUSE_PTHREADS=1)
