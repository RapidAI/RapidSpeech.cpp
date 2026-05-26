# RapidSpeech WebAssembly Demo

Three speech demos in your browser, no server side: **offline ASR**, **online
ASR (mic + VAD)**, and **offline TTS** — all running locally via
WebAssembly + WebGPU. The page loads a single shared WASM module and uses
three independent contexts (one per tab) so you can hot-switch between ASR
and TTS without rebuilding.

```
wasm-examples/
├── index.html                 # Tabbed shell (ASR Offline / ASR Online / TTS)
├── styles.css                 # Shared styles
├── app.js                     # Tab router
├── rapidspeech-bridge.js      # JS wrapper around the WASM exports
├── rapidspeech-wasm.{js,wasm} # Build output (Emscripten)
├── utils.js                   # WAV decode / encode / format helpers
├── pages/
│   ├── asr-offline.js         # Upload WAV or record → optional VAD → ASR
│   ├── asr-online.js          # Mic capture → neural VAD (or energy gate) → ASR
│   └── tts-offline.js         # text → synth → playback / download
└── serve.py                   # Tiny dev server with COOP/COEP headers
```

## Quick start

```bash
# 1. Build the WASM module (once; ~minutes the first time)
cd rapidspeech/wasm
./build-wasm.sh
# Produces rapidspeech-wasm.{js,wasm} → copied into wasm-examples/

# 2. Serve the demo (COOP/COEP required for pthreads + SharedArrayBuffer)
cd wasm-examples
python3 serve.py 8000

# 3. Open http://localhost:8000 — defaults to the ASR Offline tab.
#    Direct-link a tab via #asr-online or #tts-offline in the URL.
```

> ⚠️ **Use `serve.py`, not `python -m http.server`.** The plain HTTP server
> does not emit the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`
> headers that pthreads + SharedArrayBuffer require — without them the WASM
> module aborts at startup.

## Tabs

### ASR Offline
- Paste a model URL (e.g. an HF GGUF) → **Load Model**.
- Optionally load a VAD GGUF (silero-vad / firered-vad) — when present, long
  clips are pre-segmented by speech activity and each region is transcribed
  separately with `[mm:ss → mm:ss]` timestamps.
- Source: upload a WAV or record a clip with the mic. Re-decode with LLM
  on/off using the **Re-decode (LLM)** button.

### ASR Online
- Same model loader; add a neural VAD GGUF for best segmentation, or leave it
  blank to fall back to an RMS-energy gate.
- **Two-pass** checkbox: emits a CTC-greedy partial (grey, italic) the moment
  the segment closes, then updates the same line in place with the LLM-rescored
  result.
- A warning banner appears when the rolling-average RTF crosses 1.0 — your
  device cannot keep up with audio at the current settings. Hints suggest
  turning off LLM or two-pass.

### TTS Offline
- Load an OmniVoice or OpenVoice2 GGUF.
- Adjust `instruct`, language, seed, and diffusion steps. Generated audio
  appears as an in-page `<audio>` element with a one-click WAV download.
- **Voice cloning**: upload a reference WAV + its transcript (OmniVoice).

## Build prerequisites

- Emscripten SDK (emsdk activated in the shell):
  ```bash
  source /path/to/emsdk/emsdk_env.sh
  ```
- The build script enables both **WebGPU** and **pthreads**:
  ```bash
  cmake -DRS_WASM_WEBGPU=ON -DRS_WASM_PTHREADS=ON ...
  ```
  WebGPU is only effective on Chromium-based browsers as of 2026. On Firefox /
  Safari the runtime silently falls back to CPU.

## Browser requirements

- Chromium 113+ for WebGPU (Chrome / Edge / Opera). WebGPU on Firefox is still
  behind `dom.webgpu.enabled`. Safari has it shipping but with quirks.
- Cross-origin isolation (COOP+COEP) — see the `serve.py` note above.
- Mic input requires HTTPS or `localhost`.

## Deploying

The whole `wasm-examples/` directory is a static site. Drop it on any host
that lets you set the COOP/COEP headers (HuggingFace Spaces "Static",
Cloudflare Pages with `_headers`, ModelScope Studios, S3+CloudFront with a
Lambda@Edge, etc.). Without COOP/COEP the module loads but pthreads aborts.

### HuggingFace Spaces (Static)

```bash
git clone https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE
cd YOUR_SPACE
cp /path/to/RapidSpeech.cpp/wasm-examples/* .
git lfs install && git lfs track "*.gguf" && git add .gitattributes
cp /path/to/model.gguf .
git add . && git commit -m "RapidSpeech WASM demo" && git push
```

HuggingFace Spaces emit COOP/COEP automatically for static spaces.

## API surface (JS bridge → C exports)

The `RapidSpeechWASM` and `RapidSpeechVAD` classes in
[`rapidspeech-bridge.js`](rapidspeech-bridge.js) are the only thing pages talk
to. They wrap the following C exports (defined in
[`rapidspeech/wasm/rapidspeech_wasm.cpp`](../rapidspeech/wasm/rapidspeech_wasm.cpp)):

| Export | Description |
|--------|-------------|
| `rs_wasm_init_ex(path, task_type, threads)` | Load model with explicit task |
| `rs_wasm_push_audio(ptr, n)` | Push float32 PCM (ASR / TTS reference) |
| `rs_wasm_push_text(text)` | Push UTF-8 text for TTS |
| `rs_wasm_push_reference_audio / text(...)` | Voice cloning |
| `rs_wasm_process()` *async* | Run one inference step (await it!) |
| `rs_wasm_redecode()` *async* | Re-run decoder only (2-pass ASR) |
| `rs_wasm_get_text / get_audio_ptr / get_audio_len` | Read outputs |
| `rs_wasm_set_use_llm / set_ctc_precheck / set_tts_params / set_tts_diffusion_steps` | Knobs |
| `rs_wasm_vad_init / push_audio / drain_segments / drain_frames` | VAD streaming API |

`process` / `redecode` are bound with `cwrap({async: true})` because they
suspend through WebGPU's queue ops under ASYNCIFY — always `await` them or
you'll get a `Promise` instead of a result.

## Reusing the bridge outside the browser

The same `rapidspeech-bridge.js` runs under Node.js. See
[`../node-api-example/`](../node-api-example/) for a CLI that loads GGUF
files from disk and runs ASR / TTS without a browser.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `GET /rapidspeech-wasm.wasm net::ERR_CONNECTION_REFUSED` | `serve.py` not running. |
| `SharedArrayBuffer is not defined` / pthreads abort | Missing COOP/COEP headers — don't use `python -m http.server`. |
| `Queue work failed with status 3` | (Old bug) VAD on WebGPU. Already fixed — VAD is pinned to CPU in `rs_wasm_vad_init`. |
| `gguf_init_from_file_ptr: tensor name … is too long: 68 >= 64` | WASM build was configured without `GGML_MAX_NAME=128`. Rebuild with `rapidspeech/wasm` — its CMakeLists adds this define. |
| RTF warning banner | Your device can't keep up at the current settings. Turn off LLM, turn off two-pass, or shorten the VAD silence window. |

## See also

- [`../node-api-example/README.md`](../node-api-example/README.md) — same WASM module, Node CLI
- [`../python-api-examples/README.md`](../python-api-examples/README.md) — native Python bindings (faster than WASM)
- [`../README.md`](../README.md) — project overview and C++ CLI usage
