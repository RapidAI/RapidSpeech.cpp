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

add_subdirectory(${GGML_PATH} ggml)

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
        rapidspeech/src/arch/llm_model.h
        rapidspeech/src/arch/llm_model.cpp
        rapidspeech/src/arch/llm_graph.h
        rapidspeech/src/arch/llm_graph.cpp
        rapidspeech/src/arch/llm_kv_cache.h
        rapidspeech/src/arch/llm_kv_cache.cpp
        rapidspeech/src/arch/qwen3.h
        rapidspeech/src/arch/qwen3.cpp
        rapidspeech/src/utils/rs_wav.cpp
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

# Set library version
set_target_properties(rapidspeech-core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
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
if (RS_METAL AND APPLE)
    message(STATUS "RapidSpeech: Metal acceleration enabled.")
    target_compile_definitions(rapidspeech-core PRIVATE RS_USE_METAL)
    if (TARGET ggml-metal)
        target_link_libraries(rapidspeech-core PRIVATE ggml-metal)
    endif()
    target_link_libraries(rapidspeech-core PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalKit")
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

    # Install ggml shared libraries next to the pybind module
    # so that the OS loader finds all transitive dependencies.
    install(TARGETS ggml ggml-base
            LIBRARY DESTINATION .
            RUNTIME DESTINATION .
    )
    if (TARGET ggml-cpu)
        install(TARGETS ggml-cpu
                LIBRARY DESTINATION .
                RUNTIME DESTINATION .
        )
    endif()

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
add_executable(rs-asr-offline
        examples/asr/rs-asr-offline.cpp
)

add_executable(rs-quantize
        examples/quantize/quantize.cpp
        examples/quantize/common_quantize.cpp
)
target_link_libraries(rs-asr-offline PRIVATE rapidspeech-core ggml)
target_link_libraries(rs-quantize PRIVATE ggml)

if (NOT MSVC)
    target_link_libraries(rs-quantize PRIVATE pthread)
endif()

target_include_directories(rs-quantize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/quantize
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

install(TARGETS rs-asr-offline
        RUNTIME DESTINATION bin
)
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