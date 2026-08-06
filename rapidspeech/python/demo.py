"""Legacy local speech demo server.

The public browser-console entry point is `rapidspeech serve`; this module is
kept only for source compatibility with callers that import `run_demo`.

A pure-stdlib HTTP server (no Flask, no websockets) serving a single-page
UI: microphone → ASR, text → TTS, with backend/model/latency/offline status
visible. The browser records raw PCM via AudioContext and uploads 16-bit
WAV, so the server never needs an audio-codec dependency.

Endpoints:
    GET  /            the demo page (rapidspeech.python._demo.html)
    GET  /api/status  models, backends, offline flag
    POST /api/transcribe   body: WAV bytes            → JSON {text, ms, rtf}
    POST /api/speak        body: JSON {text, voice}   → WAV bytes (+timing headers)

One native context per task, serialized with a lock — the demo favours
simplicity and predictability over concurrency.
"""

import json
import os
import socket
import struct
import sys
import threading
import time
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path


class DemoState:
    def __init__(self):
        self.asr = None
        self.tts = None
        self.asr_name = None
        self.tts_name = None
        self.asr_sr = 16000
        self.lock = threading.Lock()
        self.last = {"asr_ms": None, "asr_rtf": None,
                     "tts_ms": None, "tts_rtf": None}


STATE = DemoState()


def _read_wav(data):
    import numpy as np
    with wave.open(BytesIO(data), "rb") as wf:
        nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError("expected 16-bit PCM WAV from the demo page")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return np.ascontiguousarray(pcm), sr


def _wav_bytes(pcm, sr):
    import numpy as np
    pcm = np.asarray(pcm, dtype=np.float32)
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if peak > 1.0:
        pcm = pcm * (0.95 / peak)
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(pcm, -1, 1) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


def _resample(pcm, src, dst):
    import math
    import numpy as np
    if src == dst:
        return pcm
    n = int(math.ceil(len(pcm) * dst / src))
    x = np.arange(n, dtype=np.float64) * (src / dst)
    return np.interp(x, np.arange(len(pcm), dtype=np.float64), pcm).astype(np.float32)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep the terminal clean; errors go through _json/_err

    # -- helpers ------------------------------------------------------
    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _err(self, what, next_step, code=500):
        self._json({"error": what, "next_step": next_step}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n)

    # -- routes -------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = (Path(__file__).parent / "_demo.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._json({
                "asr_model": STATE.asr_name,
                "tts_model": STATE.tts_name,
                "asr_backend": STATE.asr.get_backend_name() if STATE.asr else None,
                "tts_backend": STATE.tts.get_backend_name() if STATE.tts else None,
                "offline_enforced": os.environ.get("RAPIDSPEECH_OFFLINE", "") != "",
                "last": STATE.last,
            })
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/transcribe":
            self._transcribe()
        elif self.path == "/api/speak":
            self._speak()
        else:
            self._send(404, b"not found", "text/plain")

    def _transcribe(self):
        if STATE.asr is None:
            return self._err("ASR model not loaded",
                             "restart with `rapidspeech serve`")
        try:
            pcm, sr = _read_wav(self._body())
        except Exception as e:
            return self._err("could not parse audio: %s" % e,
                             "reload the page and record again", 400)
        pcm = _resample(pcm, sr, STATE.asr_sr)
        dur = len(pcm) / STATE.asr_sr
        if dur < 0.2:
            return self._err("recording too short (%.2fs)" % dur,
                             "hold the button and speak", 400)
        with STATE.lock:
            t0 = time.perf_counter()
            try:
                STATE.asr.reset()
                STATE.asr.push_audio(pcm)
                STATE.asr.process()
                text = STATE.asr.get_text()
            except Exception as e:
                return self._err("ASR failed: %s" % e,
                                 "run `rapidspeech doctor`")
            dt = time.perf_counter() - t0
        STATE.last["asr_ms"] = round(dt * 1000)
        STATE.last["asr_rtf"] = round(dt / dur, 3)
        self._json({"text": text, "ms": round(dt * 1000),
                    "rtf": round(dt / dur, 3), "audio_s": round(dur, 2)})

    def _speak(self):
        if STATE.tts is None:
            return self._err("TTS model not loaded",
                             "restart with `rapidspeech serve`")
        try:
            req = json.loads(self._body().decode("utf-8"))
            text = (req.get("text") or "").strip()
        except Exception:
            return self._err("bad request body", "reload the page", 400)
        if not text:
            return self._err("empty text", "type something to speak", 400)
        with STATE.lock:
            t0 = time.perf_counter()
            try:
                pcm = STATE.tts.synthesize(text)
            except Exception as e:
                return self._err("TTS failed: %s" % e,
                                 "run `rapidspeech doctor`")
            dt = time.perf_counter() - t0
        if pcm is None or len(pcm) == 0:
            return self._err(
                "synthesis produced no audio",
                "the default model speaks Chinese — try Chinese text")
        sr = STATE.tts.get_sample_rate()
        dur = len(pcm) / sr
        STATE.last["tts_ms"] = round(dt * 1000)
        STATE.last["tts_rtf"] = round(dt / dur, 3)
        self._send(200, _wav_bytes(pcm, sr), "audio/wav", extra={
            "X-Time-Ms": str(round(dt * 1000)),
            "X-RTF": "%.3f" % (dt / dur),
            "X-Audio-S": "%.2f" % dur,
        })


def _pick_port(host, want):
    s = socket.socket()
    try:
        s.bind((host, want))
        return want
    except OSError:
        s.close()
        s = socket.socket()
        s.bind((host, 0))
        return s.getsockname()[1]
    finally:
        s.close()


def run_demo(args, model_path_fn, registry, export_g2p, import_native, fail):
    """Wired up from cli.py (shares its model resolution and error style)."""
    print("Preparing models (first run downloads ~800 MB)...", file=sys.stderr)
    asr_path, _ = model_path_fn(registry, args.asr_model, "asr")
    tts_path, tts_name = model_path_fn(registry, args.tts_model, "tts")

    if tts_name and registry.models[tts_name].get("voices"):
        from rapidspeech import download as dl
        voice = registry.default_voice(tts_name)
        vfile = dl.ensure_file(registry.resolve_voice(tts_name, voice))
        os.environ.setdefault("RS_KOKORO_VOICE_PATH", str(vfile))
    missing = export_g2p()
    if missing:
        fail("TTS runtime data missing: %s" % ", ".join(missing),
             next_step="pip install --upgrade rapidspeech")

    native = import_native()
    print("Loading ASR: %s" % asr_path.name, file=sys.stderr)
    STATE.asr = native.asr_offline(model_path=str(asr_path),
                                   n_threads=args.threads,
                                   use_gpu=not args.cpu)
    STATE.asr_name = asr_path.name
    STATE.asr_sr = STATE.asr.get_model_meta()["audio_sample_rate"]
    print("Loading TTS: %s" % tts_path.name, file=sys.stderr)
    STATE.tts = native.tts_synthesizer(model_path=str(tts_path),
                                       n_threads=args.threads,
                                       use_gpu=not args.cpu)
    STATE.tts.set_params(language="zh", seed=42)
    STATE.tts_name = tts_path.name

    port = _pick_port(args.host, args.port)
    url = "http://%s:%d" % (args.host, port)
    srv = ThreadingHTTPServer((args.host, port), Handler)
    print("\n  RapidSpeech legacy demo ready:  %s" % url, file=sys.stderr)
    print("  ASR %s (%s) | TTS %s (%s)\n  Ctrl-C to stop.\n" % (
        STATE.asr_name, STATE.asr.get_backend_name(),
        STATE.tts_name, STATE.tts.get_backend_name()), file=sys.stderr)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    return 0
