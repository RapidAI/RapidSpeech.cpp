cmake_minimum_required(VERSION 3.14)

project(RapidSpeech.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Compilation Options ---
option(RS_BUILD_TESTS "Build test executables" ON)
option(RS_CUDA "Enable CUDA acceleration" OFF)
option(RS_VULKAN "Enable Vulkan acceleration" OFF)
option(RS_CANN "Enable CANN acceleration (Huawei)" OFF)
option(RS_OPENCL "Enable OpenCL acceleration" OFF)

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
include_directories(include src)

set(RS_SOURCES
        src/core/rs_context.cpp
        src/core/rs_processor.cpp
        src/frontend/audio_processor.cpp
        src/frontend/fftsg.c
        src/utils/rs_log.cpp
        src/arch/sensevoice.cpp
        src/utils/rs_wav.cpp
)

# --- Library Compilation ---
add_library(rapidspeech SHARED
        ${RS_SOURCES}
        src/c_api/rapidspeech_c.cpp
)

target_link_libraries(rapidspeech PRIVATE ggml)

# --- Hardware Acceleration Logic ---

# 1. CUDA Configuration
if (RS_CUDA)
    message(STATUS "RapidSpeech: CUDA acceleration enabled.")
    target_compile_definitions(rapidspeech PRIVATE RS_USE_CUDA)
    if (TARGET ggml-cuda)
        target_link_libraries(rapidspeech PRIVATE ggml-cuda)
    endif()
endif()

# 2. Metal Configuration (Apple Platforms)
if (RS_METAL AND APPLE)
    message(STATUS "RapidSpeech: Metal acceleration enabled.")
    target_compile_definitions(rapidspeech PRIVATE RS_USE_METAL)
    if (TARGET ggml-metal)
        target_link_libraries(rapidspeech PRIVATE ggml-metal)
    endif()
    target_link_libraries(rapidspeech PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalKit")
endif()

# 3. Vulkan Configuration
if (RS_VULKAN)
    message(STATUS "RapidSpeech: Vulkan acceleration enabled.")
    target_compile_definitions(rapidspeech PRIVATE RS_USE_VULKAN)
    if (TARGET ggml-vulkan)
        target_link_libraries(rapidspeech PRIVATE ggml-vulkan)
    endif()
endif()

# 4. CANN Configuration (Huawei Ascend)
if (RS_CANN)
    message(STATUS "RapidSpeech: CANN acceleration enabled.")
    target_compile_definitions(rapidspeech PRIVATE RS_USE_CANN)
    if (TARGET ggml-cann)
        target_link_libraries(rapidspeech PRIVATE ggml-cann)
    endif()
endif()

# 5. OpenCL Configuration
if (RS_OPENCL)
    message(STATUS "RapidSpeech: OpenCL acceleration enabled.")
    target_compile_definitions(rapidspeech PRIVATE RS_USE_OPENCL)
    if (TARGET ggml-opencl)
        target_link_libraries(rapidspeech PRIVATE ggml-opencl)
    endif()
endif()

# --- Example Executables ---
add_executable(rs-cli examples/main.cpp)
target_link_libraries(rs-cli PRIVATE rapidspeech)

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
            target_link_libraries(${test_target} PRIVATE rapidspeech ggml)
            add_test(NAME ${test_target} COMMAND ${test_target})
        endif()
    endforeach()
endif()