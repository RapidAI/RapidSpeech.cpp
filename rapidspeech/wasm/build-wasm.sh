#!/bin/bash
# Build RapidSpeech as WebAssembly using Emscripten.
#
# Prerequisites:
#   Install emsdk: https://emscripten.org/docs/getting_started/downloads.html
#   cd /path/to/emsdk && ./emsdk install latest && ./emsdk activate latest
#   source /path/to/emsdk/emsdk_env.sh
#
# Usage:
#   ./rapidspeech/wasm/build-wasm.sh
#   # Output: rapidspeech/wasm/build/rapidspeech-wasm.js + .wasm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== RapidSpeech WASM Build ==="
echo "Project:  $PROJECT_DIR"
echo "Build:    $BUILD_DIR"

# Verify Emscripten is available
if ! command -v emcc &>/dev/null; then
    echo "ERROR: emcc not found. Source emsdk_env.sh first."
    echo "  source /path/to/emsdk/emsdk_env.sh"
    exit 1
fi
echo "emcc:     $(which emcc)"
echo "emsdk:    $(emcc --version | head -1)"

# Copy output to wasm-examples directory
WASM_EXAMPLES_DIR="$PROJECT_DIR/wasm-examples"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRS_WASM_WEBGPU=ON \
    -DRS_WASM_PTHREADS=ON \
    -G "Unix Makefiles"

make -j$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

# Copy output files to wasm-examples
echo ""
echo "=== Copying output to $WASM_EXAMPLES_DIR ==="
cp -v rapidspeech-wasm.js "$WASM_EXAMPLES_DIR/"
cp -v rapidspeech-wasm.wasm "$WASM_EXAMPLES_DIR/"
[ -f rapidspeech-wasm.worker.js ] && cp -v rapidspeech-wasm.worker.js "$WASM_EXAMPLES_DIR/"

echo ""
echo "=== Done ==="
echo "Web demo:  $WASM_EXAMPLES_DIR/index.html"
echo "Node demo: $PROJECT_DIR/node-api-example/index.js"
