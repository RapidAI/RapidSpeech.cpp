# rs-server — OpenAI-compatible HTTP API + MCP

A single native C++ server that exposes RapidSpeech.cpp's ASR and TTS pipelines
over three protocols:

- **OpenAI-compatible HTTP API** — drop-in for OpenAI audio clients:
  - `POST /v1/audio/transcriptions` — audio (WAV) → text (ASR)
  - `POST /v1/audio/speech` — text → audio (WAV) (TTS)
  - `GET /v1/models`, `GET /health`
- **Model Context Protocol (MCP)**, JSON-RPC 2.0 — tools for LLM agents:
  - `transcribe` (audio file path → text), `synthesize` (text → WAV file)
  - transports: **stdio** (Claude Desktop et al.) and **HTTP** (`POST /mcp`)
- **WebSocket streaming** — four low-latency endpoints (see below):
  - `/asr/stream`, `/asr/vad`, `/tts/segmented`, `/tts/stream`

One process serves ASR, TTS, or both. Each model context is mutex-guarded, so
concurrent HTTP requests are serialized per model.

Dependencies are vendored single headers: [`httplib.h`](httplib.h)
(cpp-httplib, MIT) for HTTP, [`rs_ws.h`](rs_ws.h) (self-contained RFC 6455
WebSocket) for streaming, and [`rs_json.h`](rs_json.h) (self-contained) for
JSON. No Python, no external services.

---

## Build

```bash
cmake -B build && cmake --build build --target rs-server -j
# → build/rs-server
```

## Run

```bash
# Both ASR and TTS
./build/rs-server \
    --asr-model xasr-q4_k_m.gguf \
    --tts-model omnivoice-f16.gguf \
    --vad-model silero_vad_v6.gguf \
    --host 127.0.0.1 --port 8080          # WebSocket on 8081

# ASR only / TTS only — just omit the other flag
./build/rs-server --asr-model xasr-q4_k_m.gguf --port 8080
```

Options: `--asr-model`, `--tts-model`, `--vad-model` (for `/asr/vad`),
`--vad-threshold` (default 0.5), `--vad-partial-ms` (default 600, `0`=off),
`--vad-end-silence-ms` (default 600; larger values merge short pauses),
`--vad-min-segment-ms` (default 500), `--vad-max-segment-ms` (default `0`=off),
`--two-pass` (FunASR-Nano CTC first pass + LLM final re-decode),
`--ctc-precheck` (skip LLM on CTC-empty silence/noise),
`--mem-log` (log ASR scheduler compute-buffer high-water marks),
`--asr-reload-every N` (reload ASR after N final segments to release backend
buffers; default `0`=off),
`--host` (default 127.0.0.1), `--port` (8080), `--ws-port` (default port+1),
`--web-dir` (serve a static UI dir at `/`), `--mcp-stdio`, `-t/--threads` (4),
`--cpu`, `-h`.

## Web test console

[`webui.html`](webui.html) is a self-contained browser UI (no build step) that
exercises every endpoint: file/mic transcription, streaming ASR with live
partials, VAD segments, and streaming TTS playback. Serve it from the server so
it's same-origin (needed for microphone access):

```bash
./build/rs-server --asr-model xasr.gguf --tts-model tts.gguf \
    --vad-model silero.gguf --web-dir examples/server --port 8080
# open http://127.0.0.1:8080/webui.html
```

(CORS is enabled, so you can also just open the file directly — HTTP calls work,
but microphone capture requires the same-origin `--web-dir` route.)

---

## OpenAI HTTP API

### Transcription (ASR)

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
    -F file=@audio.wav \
    -F model=rapidspeech-asr
# → {"text":"..."}
# add  -F response_format=text  for plain text
```

For long WAV files, load a VAD model and ask the HTTP endpoint to split speech
before ASR:

```bash
curl "http://127.0.0.1:8080/v1/audio/transcriptions" \
    -F file=@long.wav \
    -F model=rapidspeech-asr \
    -F vad=true \
    -F vad_end_silence_ms=900 \
    -F vad_min_segment_ms=300 \
    -F vad_max_segment_ms=15000
# → {"text":"...","vad":true,"segments":[{"id":0,"start":...,"end":...,"text":"..."}]}
```

HTTP VAD mode uses the same server-wide VAD defaults as `/asr/vad`, and accepts
the same override fields: `vad_threshold`, `vad_end_silence_ms`,
`vad_min_segment_ms`, `vad_max_segment_ms`, `two_pass`, and `ctc_precheck`.
The fields can be sent either as multipart form fields or URL query parameters.

With the official OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="rapidspeech-asr", file=f).text)
```

### Speech (TTS)

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
    -H "content-type: application/json" \
    -d '{"model":"rapidspeech-tts","input":"你好，欢迎使用 RapidSpeech","voice":"female","language":"Chinese"}' \
    --output out.wav
```

Recognized body fields: `input` (required), `voice` (→ instruct/speaker),
`language`, `seed`, `n_steps` (TTS diffusion steps), plus `instruct` as an
explicit alias for `voice`. Output is a 16-bit mono WAV at the model's rate.

---

## MCP

### stdio (Claude Desktop / local MCP clients)

```bash
./build/rs-server --asr-model xasr.gguf --tts-model tts.gguf --mcp-stdio
```

In stdio mode the JSON-RPC stream owns stdout; all framework/ggml logs are
diverted to stderr so they never corrupt it.

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rapidspeech": {
      "command": "/abs/path/to/build/rs-server",
      "args": ["--asr-model", "/abs/xasr.gguf",
               "--tts-model", "/abs/tts.gguf",
               "--mcp-stdio"]
    }
  }
}
```

Then ask the agent to *"transcribe /path/to/audio.wav"* or *"synthesize 'hello'
to /tmp/hi.wav"*.

### HTTP (Streamable-HTTP style)

The same JSON-RPC is available at `POST /mcp` (one message per request):

```bash
curl -X POST http://127.0.0.1:8080/mcp \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
curl -X POST http://127.0.0.1:8080/mcp \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call",
       "params":{"name":"transcribe","arguments":{"path":"/abs/audio.wav"}}}'
```

Tools:

| Tool | Arguments | Result |
|------|-----------|--------|
| `transcribe` | `path` (WAV file) | transcribed text |
| `synthesize` | `text`, optional `output_path` / `voice` / `language` | writes a WAV, returns its path + duration |

---

## WebSocket streaming

WebSocket endpoints run on a **separate port** (`--ws-port`, default HTTP
port + 1). Wire format: **binary frames = raw 16-bit signed little-endian mono
PCM**; **text frames = JSON** (control input, and result/marker output). The
server serializes one session per model (ggml contexts are single-threaded); a
second concurrent session on the same model is rejected with
`{"type":"error","error":"...busy"}`.

| Endpoint | Direction | Purpose |
|----------|-----------|---------|
| `/asr/stream` | PCM in → text out | Pure streaming ASR (X-ASR). `partial` per token update, `final` on `{"type":"eos"}`/close. Optional `?chunk_len=16\|32\|48\|96`. |
| `/asr/vad` | PCM in → text out | VAD-segmented offline ASR. `partial` while a segment grows, `final` per segment. Needs `--vad-model`. |
| `/tts/segmented` | text in → audio out | Splits input into sentences, synthesizes each fully, streams `segment` marker + audio per sentence. Works for any TTS model. |
| `/tts/stream` | text in → audio out | Pure streaming TTS — emits each audio chunk as generated (low first-chunk latency on CosyVoice3). |

`/asr/vad` uses the server-wide VAD settings from the command line. A client can
override them for one WebSocket connection with query parameters:

```text
ws://127.0.0.1:8081/asr/vad?vad_threshold=0.45&vad_end_silence_ms=900&vad_partial_ms=1000&vad_max_segment_ms=10000
```

Supported query keys: `vad_threshold`, `vad_end_silence_ms`, `vad_partial_ms`,
`vad_min_segment_ms`, `vad_max_segment_ms`, `two_pass`, and `ctc_precheck`.
Short aliases without the `vad_` prefix are also accepted for VAD timings.
`two_pass` makes partials use the fast CTC pass and finals use LLM re-decode.

Message shapes:

```jsonc
// ASR (/asr/stream, /asr/vad) — server → client
{"type":"ready"}
{"type":"partial","text":"…","segment":0}   // segment only on /asr/vad
{"type":"final","text":"…","segment":0}
// client → server: binary PCM frames, then
{"type":"eos"}

// TTS (/tts/segmented, /tts/stream) — client → server
{"type":"text","text":"你好，世界。","voice":"female","language":"Chinese"}
// server → client: JSON marker, then a binary PCM frame, …
{"type":"audio_info","sample_rate":24000,"format":"pcm_s16le"}
{"type":"segment","index":0,"text":"你好，世界。"}   // /tts/segmented only
{"type":"final"}
```

Minimal Python client (streaming ASR):

```python
import asyncio, json, wave, websockets

async def main():
    wf = wave.open("audio_16k.wav", "rb")           # 16 kHz mono s16le
    async with websockets.connect("ws://127.0.0.1:8081/asr/stream") as ws:
        assert json.loads(await ws.recv())["type"] == "ready"
        async def reader():
            async for m in ws:
                print(json.loads(m))                # partial / final
        task = asyncio.create_task(reader())
        while (frames := wf.readframes(3200)):       # 200 ms chunks
            await ws.send(frames)                    # binary PCM
            await asyncio.sleep(0.2)                 # pace ~ real time
        await ws.send(json.dumps({"type": "eos"}))
        await asyncio.sleep(1); task.cancel()

asyncio.run(main())
```

Streaming TTS: send one text frame, then read a binary audio frame per chunk
(concatenate + wrap in a WAV header at the `sample_rate` from `audio_info`).

---

- **Audio input is WAV** (any rate/bit-depth; auto-resampled to the model's
  rate). mp3/other formats aren't decoded — transcode to WAV first, or extend
  `decode_audio()` in `rs-server.cpp` to use miniaudio.
- **Concurrency**: HTTP is multi-threaded, but each model runs one request at a
  time (ggml contexts are single-threaded); requests queue on a per-model mutex.
  WebSocket sessions hold the model for the whole session — a second concurrent
  session on the same model is rejected as "busy" (one live stream per model).
- `/asr/stream` needs an X-ASR model (true chunked streaming); other ASR models
  reply with an error on that endpoint (use `/asr/vad` or the HTTP API instead).
- `/tts/stream` streams per-chunk only on CosyVoice3; other TTS models emit a
  single chunk at the end (use `/tts/segmented` for sentence-level streaming).
- **GPU memory**: ggml/CUDA keeps model weights and scheduler compute buffers
  resident for reuse, so `nvidia-smi` will not drop after each segment. For
  long VAD runs, set `vad_max_segment_ms` to bound the largest offline ASR graph.
  Use `--mem-log` to check whether internal compute buffers plateau. If you need
  the process to return CUDA buffers during a soak test, use
  `--asr-reload-every N`; this trades a reload pause for lower high-water memory.
- This is an **example** server (plain HTTP, no auth/TLS/rate-limiting). Put it
  behind a reverse proxy for anything exposed.
- HTTP ASR uses the offline whole-clip path by default. For long audio, pass
  `vad=true` with `--vad-model` to segment before ASR, or use `/asr/stream` with
  an X-ASR model for true streaming.
