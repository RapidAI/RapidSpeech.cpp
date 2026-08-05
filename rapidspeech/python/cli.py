"""RapidSpeech command-line interface.

Zero-config first run::

    pip install rapidspeech
    rapidspeech transcribe audio.wav          # ASR (auto-downloads the model)
    rapidspeech speak "你好，世界" -o out.wav   # TTS (auto-downloads the model)

Subcommands:

- ``doctor``      environment / install diagnostics
- ``models``      list the model registry and what is cached locally
- ``pull``        download a model by ID (resumable, sha256-verified)
- ``transcribe``  WAV -> text with the default (or named) ASR model
- ``speak``       text -> WAV with the default (or named) TTS model
- ``serve``        launch the local API and browser speech console

Errors follow the project remediation rule: say what happened, the most
likely cause, and the exact next command to run.
"""

import argparse
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
import webbrowser
from pathlib import Path

from rapidspeech.python import download as dl
from rapidspeech.python import registry as reg


# ──────────────────────────────────────────────────────────────────────
# Error helpers (what happened / likely cause / next step)
# ──────────────────────────────────────────────────────────────────────
def fail(what, cause=None, next_step=None, code=1):
    print("error: %s" % what, file=sys.stderr)
    if cause:
        print("  likely cause: %s" % cause, file=sys.stderr)
    if next_step:
        print("  next step:    %s" % next_step, file=sys.stderr)
    raise SystemExit(code)


def _fail_from(exc):
    hint = getattr(exc, "hint", None)
    fail(str(exc), next_step=hint)


def _import_native():
    """Import the compiled extension lazily with an actionable error."""
    try:
        import rapidspeech.rapidspeech as native
        return native
    except ImportError as e:
        fail(
            "the rapidspeech native module failed to import (%s)" % e,
            cause="the wheel does not match this Python/OS, or a shared "
                  "library is missing",
            next_step="run `rapidspeech doctor` for details, or reinstall: "
                      "pip install --force-reinstall rapidspeech",
        )


# ──────────────────────────────────────────────────────────────────────
# WAV I/O (kept dependency-free: wave + numpy)
# ──────────────────────────────────────────────────────────────────────
def read_wav_mono_f32(path):
    """Read WAV -> (float32 mono PCM in [-1,1], sample_rate)."""
    import numpy as np
    try:
        wf = wave.open(path, "rb")
    except FileNotFoundError:
        fail("audio file not found: %s" % path,
             next_step="check the path, or record one: `say -o t.aiff ...` / arecord")
    except wave.Error as e:
        fail("could not parse %s as WAV (%s)" % (path, e),
             cause="the file is not a PCM WAV (mp3/m4a/opus are not supported yet)",
             next_step="convert it first: ffmpeg -i input -ar 16000 -ac 1 out.wav")
    with wf:
        nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sw == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
        i32 = np.where(i32 & 0x800000, i32 | ~0xFFFFFF, i32)
        pcm = i32.astype(np.float32) / (1 << 23)
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    else:
        fail("unsupported WAV sample width: %d bytes" % sw,
             next_step="convert it: ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav")

    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), sr


def resample_linear(pcm, src_sr, dst_sr):
    import numpy as np
    if src_sr == dst_sr:
        return pcm
    n_out = int(math.ceil(len(pcm) * dst_sr / src_sr))
    x_src = np.arange(len(pcm), dtype=np.float64)
    x_dst = np.arange(n_out, dtype=np.float64) * (src_sr / dst_sr)
    return np.interp(x_dst, x_src, pcm).astype(np.float32)


def write_wav_pcm16(path, pcm, sample_rate):
    import numpy as np
    pcm = np.asarray(pcm, dtype=np.float32)
    # Some vocoders (kokoro DAC head in particular) emit PCM way outside
    # [-1,1]; clipping would square-wave the audio. Match the C++ CLI:
    # peak-normalize to 0.95 whenever the signal would clip.
    peak = float(np.abs(pcm).max()) if len(pcm) else 0.0
    if peak > 1.0:
        pcm = pcm * (0.95 / peak)
    i16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(i16.tobytes())


# ──────────────────────────────────────────────────────────────────────
# Model resolution helpers
# ──────────────────────────────────────────────────────────────────────
def _load_registry():
    try:
        return reg.load_registry()
    except reg.RegistryError as e:
        _fail_from(e)


def _model_path(registry, spec, task):
    """Turn --model into a local file path.

    Accepts a registry spec (``funasr-nano``, ``kokoro-zh:q8_0``), a
    direct .gguf path, or None (task default). Downloads on demand.
    """
    if spec and (os.path.sep in spec or spec.endswith(".gguf")):
        p = Path(spec).expanduser()
        if not p.is_file():
            fail("model file not found: %s" % p,
                 next_step="use a registry model instead: rapidspeech models")
        return p, None
    try:
        name = spec if spec else registry.default_model(task)
        resolved = registry.resolve(name)
        if resolved.known_issues:
            print("warning: %s: %s" % (resolved.name, resolved.known_issues),
                  file=sys.stderr)
        return dl.ensure_file(resolved), registry.split_spec(name)[0]
    except (reg.RegistryError, dl.DownloadError) as e:
        _fail_from(e)


# ──────────────────────────────────────────────────────────────────────
# Kokoro runtime data (G2P dictionaries shipped inside the wheel)
# ──────────────────────────────────────────────────────────────────────
def _pkg_data_dir():
    # This file lives in rapidspeech/python/; the data ships one level up
    # at rapidspeech/data/.
    return Path(__file__).resolve().parents[1] / "data"


def _data_dir_candidates(rel):
    """Mirror of rs::find_data_dir search order, for the pip-install case."""
    cands = [_pkg_data_dir() / rel]
    # Dev tree: repo_root/rapidspeech/data/<rel> and repo_root/third_party/<rel>
    repo_root = Path(__file__).resolve().parents[2]
    cands.append(repo_root / "rapidspeech" / "data" / rel)
    if rel.startswith("cppjieba/"):
        cands.append(repo_root / "third_party" / rel)
    cands.append(Path.home() / ".cache" / "rapidspeech" / rel)
    return cands


def _find_data_dir(rel):
    for c in _data_dir_candidates(rel):
        if c.is_dir():
            return c
    return None


_KOKORO_DATA_ENV = {
    "RS_KOKORO_ZH_DATA_DIR": "kokoro_zh",
    "RS_KOKORO_EN_DATA_DIR": "kokoro_en",
    "RS_WETEXT_DATA_DIR": "wetext",
    "RS_KOKORO_JIEBA_DICT_DIR": "cppjieba/dict",
}


def _export_kokoro_data_env():
    """Point the C++ runtime at the G2P data bundled with the wheel.
    Returns the list of rel-dirs that could not be located."""
    missing = []
    for env, rel in _KOKORO_DATA_ENV.items():
        if os.environ.get(env):
            continue  # user override wins
        d = _find_data_dir(rel)
        if d is not None:
            os.environ[env] = str(d)
        else:
            missing.append(rel)
    return missing


# ──────────────────────────────────────────────────────────────────────
# doctor
# ──────────────────────────────────────────────────────────────────────
def _pkg_version():
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # py3.7
        try:
            from importlib_metadata import version, PackageNotFoundError
        except ImportError:
            return None, None
    for name in ("rapidspeech", "rapidspeech-metal", "rapidspeech-cuda"):
        try:
            return name, version(name)
        except PackageNotFoundError:
            continue
    return None, None


def cmd_doctor(args):
    ok = True

    print("== system ==")
    print("  os:        %s %s (%s)" % (platform.system(), platform.release(),
                                       platform.machine()))
    print("  python:    %s (%s)" % (platform.python_version(), sys.executable))

    print("== package ==")
    dist, ver = _pkg_version()
    if dist:
        print("  dist:      %s %s" % (dist, ver))
    else:
        print("  dist:      (not installed via pip — running from source tree?)")
    try:
        import rapidspeech.rapidspeech as native
        print("  native:    OK (version %s)" % native.version())
        api = [n for n in ("asr_offline", "tts_synthesizer", "vad")
               if hasattr(native, n)]
        print("  api:       %s" % ", ".join(api))
    except ImportError as e:
        ok = False
        print("  native:    FAILED — %s" % e)
        print("             fix: pip install --force-reinstall rapidspeech")
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            print("             on Apple Silicon prefer: pip install rapidspeech-metal")

    print("== cache ==")
    print("  dir:       %s" % dl.models_dir())
    files = dl.cached_files()
    if files:
        for p, size in files:
            print("    %-42s %10.1f MB" % (p.name, size / 1e6))
    else:
        print("    (empty — `rapidspeech pull funasr-nano` to prefetch)")
    try:
        free = shutil.disk_usage(str(dl.cache_home().anchor or "/")).free
        print("  disk free: %.1f GB" % (free / 1e9))
        if free < 2e9:
            ok = False
            print("             warning: <2 GB free; the default ASR model needs ~0.7 GB")
    except OSError:
        pass

    print("== tts data (kokoro G2P) ==")
    for env, rel in _KOKORO_DATA_ENV.items():
        d = os.environ.get(env) or _find_data_dir(rel)
        status = str(d) if d else "MISSING"
        if not d:
            ok = False
        print("  %-12s %s" % (rel + ":", status))
    if any(_find_data_dir(rel) is None and not os.environ.get(env)
           for env, rel in _KOKORO_DATA_ENV.items()):
        print("  fix: pip install --upgrade rapidspeech  (data ships inside the wheel)")

    print("== registry ==")
    try:
        registry = reg.load_registry()
        print("  file:      %s" % reg.registry_path())
        print("  defaults:  asr=%s  tts=%s  vad=%s" % (
            registry.defaults.get("asr"), registry.defaults.get("tts"),
            registry.defaults.get("vad")))
    except reg.RegistryError as e:
        ok = False
        print("  FAILED: %s" % e)

    print("== network ==")
    if dl.is_offline():
        print("  offline mode (RAPIDSPEECH_OFFLINE set) — downloads disabled")
    else:
        import urllib.request
        for name, url in (("modelscope", "https://www.modelscope.cn"),
                          ("huggingface", "https://huggingface.co")):
            try:
                urllib.request.urlopen(
                    urllib.request.Request(url, method="HEAD"), timeout=5)
                print("  %-12s reachable" % name)
            except Exception:
                print("  %-12s UNREACHABLE (downloads will fall back to the other source)"
                      % name)

    print()
    print("doctor: %s" % ("all checks passed" if ok else
                          "problems found — see lines marked FAILED/MISSING above"))
    return 0 if ok else 1


# ──────────────────────────────────────────────────────────────────────
# models / pull
# ──────────────────────────────────────────────────────────────────────
def cmd_models(args):
    registry = _load_registry()
    cached = {p.name for p, _ in dl.cached_files()}
    print("%-24s %-6s %-10s %-9s %s" % ("ID", "TASK", "SIZE", "CACHED", "DESCRIPTION"))
    for name in sorted(registry.models):
        entry = registry.models[name]
        for variant in sorted(entry["variants"]):
            resolved = registry.resolve("%s:%s" % (name, variant))
            is_default = variant == entry["default_variant"]
            mid = "%s:%s" % (name, variant)
            if is_default:
                mid += " *"
            print("%-24s %-6s %-10s %-9s %s" % (
                mid, entry["task"],
                "%.0f MB" % (resolved.size_bytes / 1e6),
                "yes" if resolved.filename in cached else "-",
                entry["description"] if is_default
                else ("(!) " + resolved.known_issues if resolved.known_issues else "")))
        for voice in sorted(entry.get("voices", {})):
            resolved = registry.resolve_voice(name, voice)
            tag = " *" if voice == entry.get("default_voice") else ""
            print("%-24s %-6s %-10s %-9s %s" % (
                "  voice:%s%s" % (voice, tag), "", "0.5 MB",
                "yes" if resolved.filename in cached else "-", ""))
    print("\n(* = default; `rapidspeech pull <id>` to prefetch)")
    return 0


def cmd_pull(args):
    registry = _load_registry()
    for spec in args.model:
        name, _ = registry.split_spec(spec)
        try:
            resolved = registry.resolve(spec)
            path = dl.ensure_file(resolved)
            print(path)
            # TTS models also need their default voice pack.
            voice = registry.default_voice(name)
            if voice:
                vpath = dl.ensure_file(registry.resolve_voice(name, voice))
                print(vpath)
        except (reg.RegistryError, dl.DownloadError) as e:
            _fail_from(e)
    return 0


# ──────────────────────────────────────────────────────────────────────
# transcribe
# ──────────────────────────────────────────────────────────────────────
def cmd_transcribe(args):
    registry = _load_registry()
    model_path, _ = _model_path(registry, args.model, "asr")
    native = _import_native()

    print("loading %s ..." % model_path.name, file=sys.stderr)
    try:
        ctx = native.asr_offline(model_path=str(model_path),
                                 n_threads=args.threads,
                                 use_gpu=not args.cpu)
    except Exception as e:
        fail("failed to load ASR model %s (%s)" % (model_path, e),
             cause="corrupt download or out of memory",
             next_step="rm '%s' && rapidspeech pull %s" % (
                 model_path, args.model or registry.default_model("asr")))

    meta = ctx.get_model_meta()
    target_sr = meta["audio_sample_rate"]
    print("model: %s  backend: %s" % (meta["arch_name"], ctx.get_backend_name()),
          file=sys.stderr)

    pcm, sr = read_wav_mono_f32(args.audio)
    if sr != target_sr:
        pcm = resample_linear(pcm, sr, target_sr)
    duration = len(pcm) / target_sr

    ctx.reset()
    ctx.push_audio(pcm)
    if args.two_pass:
        ctx.set_use_llm(False)
    t0 = time.perf_counter()
    ctx.process()
    elapsed = time.perf_counter() - t0
    text = ctx.get_text()

    if args.two_pass:
        ctx.set_use_llm(True)
        t0 = time.perf_counter()
        ctx.redecode()
        elapsed += time.perf_counter() - t0
        text = ctx.get_text()

    print("%.2fs audio in %.2fs (RTF %.3f)" % (duration, elapsed,
                                               elapsed / max(duration, 1e-9)),
          file=sys.stderr)
    print(text if text else "(no speech detected)")
    return 0


# ──────────────────────────────────────────────────────────────────────
# speak
# ──────────────────────────────────────────────────────────────────────
def cmd_speak(args):
    out_parent = Path(args.output).expanduser().resolve().parent
    if not out_parent.is_dir():
        fail("output directory does not exist: %s" % out_parent,
             next_step="mkdir -p '%s'  (or pass a different -o path)" % out_parent)

    registry = _load_registry()
    model_path, model_name = _model_path(registry, args.model, "tts")

    # Voice pack (kokoro): resolve + download + export env BEFORE model load.
    if model_name and registry.models[model_name].get("voices"):
        voice = args.voice or registry.default_voice(model_name)
        try:
            vfile = dl.ensure_file(registry.resolve_voice(model_name, voice))
        except (reg.RegistryError, dl.DownloadError) as e:
            _fail_from(e)
        os.environ.setdefault("RS_KOKORO_VOICE_PATH", str(vfile))
    elif args.voice and not model_name:
        fail("--voice is only supported with registry TTS models",
             next_step="drop --voice, or use --model kokoro-zh")

    missing = _export_kokoro_data_env()
    if missing:
        fail("TTS runtime data missing: %s" % ", ".join(missing),
             cause="the wheel was built without bundled G2P data, or you are "
                   "running from a stripped source tree",
             next_step="pip install --upgrade rapidspeech  (or set "
                       "RS_KOKORO_ZH_DATA_DIR / RS_KOKORO_JIEBA_DICT_DIR manually)")

    native = _import_native()
    print("loading %s ..." % model_path.name, file=sys.stderr)
    try:
        tts = native.tts_synthesizer(model_path=str(model_path),
                                     n_threads=args.threads,
                                     use_gpu=not args.cpu)
    except Exception as e:
        fail("failed to load TTS model %s (%s)" % (model_path, e),
             cause="corrupt download or out of memory",
             next_step="rm '%s' && rapidspeech pull %s" % (
                 model_path, args.model or registry.default_model("tts")))

    meta = tts.get_model_meta()
    print("model: %s  backend: %s" % (meta["arch_name"], tts.get_backend_name()),
          file=sys.stderr)

    tts.set_params(language=args.lang, seed=args.seed)

    t0 = time.perf_counter()
    pcm = tts.synthesize(args.text)
    elapsed = time.perf_counter() - t0
    if pcm is None or len(pcm) == 0:
        fail("synthesis produced no audio",
             cause="text could not be phonemized (only Chinese G2P is "
                   "implemented for kokoro-zh; English words inside Chinese "
                   "text are fine)",
             next_step="try Chinese text, e.g. rapidspeech speak \"你好\"")

    sr = tts.get_sample_rate()
    write_wav_pcm16(args.output, pcm, sr)
    duration = len(pcm) / sr
    print("%.2fs audio in %.2fs (RTF %.3f)" % (duration, elapsed,
                                               elapsed / max(duration, 1e-9)),
          file=sys.stderr)
    print(args.output)
    return 0


# ──────────────────────────────────────────────────────────────────────
# demo
# ──────────────────────────────────────────────────────────────────────
def cmd_demo(args):
    from rapidspeech.python.demo import run_demo
    registry = _load_registry()
    args.asr_model = getattr(args, "asr_model", None)
    args.tts_model = getattr(args, "tts_model", None)
    return run_demo(args, _model_path, registry, _export_kokoro_data_env,
                    _import_native, fail)


# ──────────────────────────────────────────────────────────────────────
# serve
# ──────────────────────────────────────────────────────────────────────
def _server_candidates():
    """Return likely locations for the packaged or source-tree server."""
    candidates = []
    configured = os.environ.get("RAPIDSPEECH_SERVER_BIN")
    if configured:
        candidates.append(Path(configured).expanduser())

    package_dir = Path(__file__).resolve().parents[1]
    candidates.extend([
        package_dir / "bin" / "rs-server",
        package_dir / "rs-server",
    ])

    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        repo_root / "build" / "rs-server",
        repo_root / "build" / "bin" / "rs-server",
    ])

    path_server = shutil.which("rs-server")
    if path_server:
        candidates.append(Path(path_server))
    return candidates


def _find_server_binary():
    for candidate in _server_candidates():
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    return None


def _server_web_dir():
    package_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        package_dir / "server",
        repo_root / "examples" / "server",
    ]
    for candidate in candidates:
        if (candidate / "webui.html").is_file():
            return candidate
    return None


def _wait_for_server(url, process, timeout):
    deadline = time.monotonic() + timeout
    health_url = url.rstrip("/") + "/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.15)
    return False


def _terminate_server(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def cmd_serve(args):
    registry = _load_registry()
    server = _find_server_binary()
    if server is None:
        fail(
            "the native `rs-server` executable is not installed",
            cause="this package was installed without the server component, "
                  "or the source tree has not been built",
            next_step="reinstall a wheel that includes the server, or build it: "
                      "cmake -B build && cmake --build build --target rs-server",
        )

    model_note = "ASR + TTS" + ("" if args.no_vad else " + VAD")
    print("Preparing models (first run downloads %s)..." % model_note,
          file=sys.stderr)
    asr_path, _ = _model_path(registry, args.asr_model, "asr")
    tts_path, tts_name = _model_path(registry, args.tts_model, "tts")
    asr_path = asr_path.resolve()
    tts_path = tts_path.resolve()
    vad_path = None
    if not args.no_vad:
        vad_path, _ = _model_path(registry, args.vad_model, "vad")
        vad_path = vad_path.resolve()

    if tts_name and registry.models[tts_name].get("voices"):
        voice = args.voice or registry.default_voice(tts_name)
        try:
            vfile = dl.ensure_file(registry.resolve_voice(tts_name, voice))
        except (reg.RegistryError, dl.DownloadError) as e:
            _fail_from(e)
        os.environ.setdefault("RS_KOKORO_VOICE_PATH", str(vfile))

    missing = _export_kokoro_data_env()
    if missing:
        fail(
            "TTS runtime data missing: %s" % ", ".join(missing),
            cause="the wheel was built without bundled G2P data, or you are "
                  "running from a stripped source tree",
            next_step="pip install --upgrade rapidspeech",
        )

    web_dir = _server_web_dir()
    if web_dir is None:
        fail(
            "the browser console assets are not installed",
            cause="the package is incomplete or the source tree is missing "
                  "examples/server/webui.html",
            next_step="pip install --force-reinstall rapidspeech",
        )

    ws_port = args.ws_port if args.ws_port is not None else args.port + 1
    cmd = [
        str(server),
        "--asr-model", str(asr_path),
        "--tts-model", str(tts_path),
        "--web-dir", str(web_dir),
        "--host", args.host,
        "--port", str(args.port),
        "--ws-port", str(ws_port),
        "--threads", str(args.threads),
    ]
    if vad_path is not None:
        cmd.extend(["--vad-model", str(vad_path)])
    if args.two_pass:
        cmd.append("--two-pass")
    if args.ctc_precheck:
        cmd.append("--ctc-precheck")
    if args.cpu:
        cmd.append("--cpu")
    if args.ws_log:
        cmd.append("--ws-log")

    http_url = "http://%s:%d" % (args.host, args.port)
    console_url = http_url + "/webui.html"
    print("  command: %s" % " ".join(cmd), file=sys.stderr)
    process = None
    try:
        print("Starting RapidSpeech server...", file=sys.stderr)
        process = subprocess.Popen(cmd)
        if not _wait_for_server(http_url, process, args.startup_timeout):
            code = process.poll()
            fail(
                "RapidSpeech server did not become ready",
                cause="native model loading failed or the selected port is busy"
                      + ((" (exit code %d)" % code) if code is not None else ""),
                next_step="run the printed server command directly for logs",
            )

        print("\nRapidSpeech is ready", file=sys.stderr)
        print("  browser console: %s" % console_url, file=sys.stderr)
        print("  OpenAI API:      %s/v1" % http_url, file=sys.stderr)
        print("  WebSocket API:   ws://%s:%d" % (args.host, ws_port),
              file=sys.stderr)
        print("  Ctrl-C to stop.\n", file=sys.stderr)
        if not args.no_browser:
            webbrowser.open(console_url)
        return process.wait()
    except KeyboardInterrupt:
        print("\nstopping RapidSpeech server...", file=sys.stderr)
        return 130
    finally:
        if process is not None:
            _terminate_server(process)


# ──────────────────────────────────────────────────────────────────────
# entry point
# ──────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        prog="rapidspeech",
        description="On-device speech AI: ASR, TTS and VAD from one CLI.")
    _, ver = _pkg_version()
    p.add_argument("--version", action="version",
                   version="rapidspeech %s" % (ver or "(source tree)"))
    sub = p.add_subparsers(dest="cmd", metavar="command")

    sp = sub.add_parser("doctor", help="diagnose install / backend / cache problems")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("models", help="list downloadable models and cache status")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("pull", help="download a model (resumable, sha256-verified)")
    sp.add_argument("model", nargs="+",
                    help="model ID, e.g. funasr-nano or kokoro-zh:q8_0")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("transcribe", help="transcribe a WAV file to text")
    sp.add_argument("audio", help="input WAV file")
    sp.add_argument("--model", default=None,
                    help="registry ID (default: funasr-nano) or a .gguf path")
    sp.add_argument("--two-pass", action="store_true",
                    help="CTC first, then LLM rescoring (slower, more accurate)")
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--cpu", action="store_true", help="disable GPU")
    sp.set_defaults(func=cmd_transcribe)

    sp = sub.add_parser("speak", help="synthesize text to a WAV file")
    sp.add_argument("text", help="text to speak")
    sp.add_argument("-o", "--output", default="output.wav",
                    help="output WAV path (default: output.wav)")
    sp.add_argument("--model", default=None,
                    help="registry ID (default: kokoro-zh) or a .gguf path")
    sp.add_argument("--voice", default=None,
                    help="voice pack name (kokoro), e.g. zf_001 / zm_009")
    sp.add_argument("--lang", default="zh", help="text language (default: zh)")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--cpu", action="store_true", help="disable GPU")
    sp.set_defaults(func=cmd_speak)

    sp = sub.add_parser("demo", help="launch the local speech demo in your browser")
    sp.add_argument("--asr-model", default=None,
                    help="registry ID or .gguf path (default: funasr-nano)")
    sp.add_argument("--tts-model", default=None,
                    help="registry ID or .gguf path (default: kokoro-zh)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8090)
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--cpu", action="store_true", help="disable GPU")
    sp.add_argument("--no-browser", action="store_true",
                    help="don't open the browser automatically")
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser(
        "serve",
        help="start the OpenAI-compatible API and browser speech console",
    )
    sp.add_argument("--asr-model", default=None,
                    help="registry ID or .gguf path (default: funasr-nano)")
    sp.add_argument("--tts-model", default=None,
                    help="registry ID or .gguf path (default: kokoro-zh)")
    sp.add_argument("--vad-model", default=None,
                    help="registry ID or .gguf path (default: silero-vad)")
    sp.add_argument("--voice", default=None,
                    help="voice pack name for registry TTS models")
    sp.add_argument("--no-vad", action="store_true",
                    help="don't download/load a VAD model")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--ws-port", type=int, default=None)
    sp.add_argument("--threads", type=int, default=4)
    sp.add_argument("--startup-timeout", type=float, default=120.0,
                    help="seconds to wait for native models to load")
    sp.add_argument("--two-pass", action="store_true",
                    help="use CTC first pass and LLM final re-decode")
    sp.add_argument("--ctc-precheck", action="store_true",
                    help="skip LLM decoding for CTC-empty audio")
    sp.add_argument("--ws-log", action="store_true",
                    help="log WebSocket connection lifecycle events")
    sp.add_argument("--cpu", action="store_true", help="disable GPU")
    sp.add_argument("--no-browser", action="store_true",
                    help="don't open the browser automatically")
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted (partial downloads resume automatically)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
