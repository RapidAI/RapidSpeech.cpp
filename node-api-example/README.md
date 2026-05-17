# RapidSpeech Node.js API Example

Transcribe WAV files locally using RapidSpeech compiled to WebAssembly.
No network calls — the model and audio never leave your machine.

## Prerequisites

- Node.js >= 18
- A GGUF ASR model (SenseVoice Small, FunASRNano, etc.)
- A WAV audio file

## Quick Start

1. Build the WASM module:

```bash
cd rapidspeech/wasm
./build-wasm.sh
```

2. Run the example:

```bash
cd node-api-example
node index.js -m /path/to/model.gguf -w /path/to/audio.wav
```

## Usage

```
node index.js -m <model.gguf> -w <audio.wav> [options]

Required:
  -m, --model <path>    Path to GGUF ASR model
  -w, --wav <path>      Path to WAV file (8/16/24/32-bit PCM, mono or stereo)

Options:
  -t, --threads <n>     Number of CPU threads (default: 2)
  -r, --runs <n>        Benchmark runs (default: 1)
  -h, --help            Show this help

Examples:
  node index.js -m sense-voice-small.gguf -w test.wav
  node index.js -m funasr-nano.gguf -w audio.wav --threads 4 --runs 10
```

## Example Output

```
=== RapidSpeech Node.js API Example ===

WAV:    ./test.wav
        16000 Hz, 1 ch, 16-bit
        28160 samples, 1.76 s

Loading model into WASM...
        Architecture: SenseVoiceSmall, Sample rate: 16000 Hz

Running ASR (1 run(s))...

──────────────────────────────────────────────────────────
 你好讲古佬。
──────────────────────────────────────────────────────────

Elapsed: 0.45s | Audio: 1.76s | RTF: 0.256
Done.
```

Benchmark mode (`--runs 10`):

```
Running ASR (10 run(s))...
  Run 1: 0.452s  RTF=0.257  你好讲古佬。
  Run 2: 0.438s  RTF=0.249  你好讲古佬。
  ...
  Avg: 0.443s  Min: 0.428s  Max: 0.461s  RTF=0.252
```

## How It Works

1. **WASM module loading** — The Emscripten-compiled `rapidspeech-wasm.js` is required
   as a Node.js module. With `MODULARIZE=1`, it exports a factory function; calling it
   returns a Promise that resolves to the WASM instance.

2. **Model loading** — The GGUF file is read from disk and written to the Emscripten
   virtual filesystem via `FS.writeFile()`. `rs_wasm_init()` loads and initializes it.

3. **WAV reading** — Built-in WAV parser supporting 8/16/24/32-bit PCM. Multi-channel
   audio is downmixed to mono by reading only the first channel.

4. **Inference** — Audio samples are written to the WASM heap, `rs_wasm_process()` runs
   the full encoder→decoder pipeline, and the transcription is retrieved via
   `rs_wasm_get_text()`.

5. **All local** — No network requests after initial module load. Model and audio never
   leave your machine.

## API Functions (WASM)

| Function | Signature | Description |
|----------|-----------|-------------|
| `rs_wasm_init` | `(path: string, threads: number) -> number` | Load model, returns 0 on success |
| `rs_wasm_free` | `() -> void` | Release all resources |
| `rs_wasm_push_audio` | `(ptr: number, samples: number) -> number` | Push float32 PCM to pipeline |
| `rs_wasm_process` | `() -> number` | Run inference, returns >0 if output available |
| `rs_wasm_get_text` | `() -> string` | Get transcription result |
| `rs_wasm_reset` | `() -> number` | Reset internal state for a new segment |
| `rs_wasm_get_version` | `() -> string` | Get library version |
| `rs_wasm_get_arch_name` | `() -> string` | Get model architecture name |
| `rs_wasm_get_sample_rate` | `() -> number` | Get expected input sample rate |
| `rs_wasm_is_ready` | `() -> number` | Check if model is loaded and ready |

## Notes

- WASM performance is slower than native builds due to the lack of SIMD/Metal
  acceleration in the browser/Node.js WASM runtime.
- The WASM module currently supports ASR only. TTS support via WASM is planned.
- Memory usage scales with model size — larger models may hit Node.js memory limits.
- For production use, prefer the native Python bindings (`pip install rapidspeech`)
  or the C API for maximum performance.
