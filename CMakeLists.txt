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
include_directories(include rapidspeech/src)

set(RS_SOURCES
        rapidspeech/src/core/rs_context.cpp
        rapidspeech/src/core/rs_processor.cpp
        rapidspeech/src/frontend/audio_processor.cpp
        rapidspeech/src/frontend/fftsg.c
        rapidspeech/src/utils/rs_log.cpp
        rapidspeech/src/arch/sensevoice.cpp
        rapidspeech/src/arch/sensevoice_encoder.cpp
        rapidspeech/src/arch/funasr-nano.cpp
        rapidspeech/src/utils/rs_wav.cpp
)


# --- Library Compilation ---
add_library(rapidspeech-core SHARED
        ${RS_SOURCES}
        rapidspeech/src/c_api/rapidspeech_c.cpp
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


    install(TARGETS rapidspeech
            LIBRARY DESTINATION lib
    )

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
add_executable(rs-asr-offline examples/rs-asr-offline.cpp)
target_link_libraries(rs-asr-offline PRIVATE rapidspeech-core)
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