# RapidSpeech WebAssembly Demo

Browser-based speech recognition powered by WebAssembly. Runs entirely locally — no server needed after initial page load.

## Quick Start

1. Build the WASM module:

```bash
cd rapidspeech/wasm
./build-wasm.sh
```

This produces `rapidspeech-wasm.js` and `rapidspeech-wasm.wasm` in `wasm-examples/`.

2. Serve the demo locally:

```bash
cd wasm-examples
python3 -m http.server 8080
```

3. Open http://localhost:8080, enter a GGUF model URL, and click **Load Model**.

## Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/new-space
   - Choose **Static** as the Space type

2. Clone the Space repository:

```bash
git clone https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
cd YOUR_SPACE
```

3. Copy the demo files:

```bash
cp /path/to/RapidSpeech.cpp/wasm-examples/* .
```

4. Upload your model to the Space (Git LFS):

```bash
git lfs install
git lfs track "*.gguf"
git add .gitattributes
cp /path/to/model.gguf .
git add model.gguf index.html app.js rapidspeech-bridge.js rapidspeech-wasm.js rapidspeech-wasm.wasm
git commit -m "Add RapidSpeech WASM demo"
git push
```

5. Set the default model URL in `index.html`:

```html
<input id="model-url" type="text" value="model.gguf">
```

## Deploy to ModelScope

Same process as HuggingFace Spaces — create a Space at https://modelscope.cn/studios, clone, copy files, and push.

## File Overview

| File | Description |
|------|-------------|
| `index.html` | Main page with UI |
| `app.js` | Application logic (mic, VAD, ASR pipeline) |
| `rapidspeech-bridge.js` | JavaScript bridge for the C API |
| `rapidspeech-wasm.js` | Emscripten-generated JS glue (build output) |
| `rapidspeech-wasm.wasm` | Compiled WebAssembly binary (build output) |

## Browser Requirements

- WebAssembly SIMD support (all modern browsers)
- Microphone access (HTTPS or localhost)
- Recommended: Chrome 91+, Firefox 89+, Safari 16.4+

## Model Format

RapidSpeech uses GGUF format models. You can convert your own models or use pre-built ones from HuggingFace.
