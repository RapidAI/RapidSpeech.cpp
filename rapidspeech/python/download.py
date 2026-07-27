"""Model downloader for RapidSpeech.

Pure-stdlib (urllib) downloader with:

- stable cache directory (``~/.cache/rapidspeech/models``, override with
  ``RAPIDSPEECH_HOME``)
- resumable downloads (HTTP Range + ``.part`` file)
- sha256 verification after every download
- ordered mirror fallback (ModelScope -> Hugging Face by default)
- offline mode (``RAPIDSPEECH_OFFLINE=1`` forbids any network access)

Every user-facing error message answers: what happened, the most likely
cause, and the exact next command to run.
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


class DownloadError(Exception):
    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint


def cache_home():
    """Base cache directory. RAPIDSPEECH_HOME overrides ~/.cache/rapidspeech."""
    env = os.environ.get("RAPIDSPEECH_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "rapidspeech"


def models_dir():
    return cache_home() / "models"


def is_offline():
    return os.environ.get("RAPIDSPEECH_OFFLINE", "").strip() in ("1", "true", "yes")


def _sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%d B" % n


def _progress(done, total, t0, final=False):
    if not sys.stderr.isatty():
        return
    pct = 100.0 * done / total if total else 0.0
    speed = done / max(time.monotonic() - t0, 1e-6)
    msg = "\r  %6.1f%%  %s / %s  (%s/s)" % (
        pct, _fmt_size(done), _fmt_size(total), _fmt_size(speed))
    sys.stderr.write(msg + ("\n" if final else ""))
    sys.stderr.flush()


def _open_url(url, start_byte=0, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "rapidspeech-cli"})
    if start_byte > 0:
        req.add_header("Range", "bytes=%d-" % start_byte)
    return urllib.request.urlopen(req, timeout=timeout)


def _download_from(url, part_path, expected_size, t0):
    """Download url into part_path, resuming if part_path exists.
    Returns when the file is complete. Raises on any network error."""
    start = part_path.stat().st_size if part_path.exists() else 0
    if start > expected_size:
        # Corrupt partial (e.g. from a different source) — start over.
        part_path.unlink()
        start = 0

    if start == expected_size:
        return

    resp = _open_url(url, start_byte=start)
    status = resp.getcode()
    mode = "ab"
    if start > 0 and status != 206:
        # Server ignored the Range header — restart from scratch.
        mode = "wb"
        start = 0

    done = start
    with open(part_path, mode) as out:
        while True:
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            _progress(done, expected_size, t0)
    _progress(done, expected_size, t0, final=True)

    if done != expected_size:
        raise DownloadError(
            "download of %s stopped at %s of %s" %
            (url, _fmt_size(done), _fmt_size(expected_size)),
            hint="re-run the same command to resume from where it stopped",
        )


def ensure_file(resolved, quiet=False):
    """Make sure `resolved` (a registry.ResolvedFile) exists in the cache.

    Returns the local Path. Downloads (with resume + hash check + mirror
    fallback) when missing. Never touches the network if the file is
    already complete.
    """
    dest = models_dir() / resolved.filename
    if dest.is_file():
        if dest.stat().st_size == resolved.size_bytes:
            return dest
        # Size mismatch: stale or truncated file from a manual copy.
        raise DownloadError(
            "cached file %s has size %s but the registry expects %s" % (
                dest, _fmt_size(dest.stat().st_size), _fmt_size(resolved.size_bytes)),
            hint="delete it and re-download: rm '%s' && rapidspeech pull %s" % (
                dest, resolved.name),
        )

    if is_offline():
        raise DownloadError(
            "model file %s is not in the cache and RAPIDSPEECH_OFFLINE is set" %
            resolved.filename,
            hint="unset RAPIDSPEECH_OFFLINE, or copy the file to %s" % dest,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if not quiet:
        print("Downloading %s (%s) -> %s" % (
            resolved.name, _fmt_size(resolved.size_bytes), dest), file=sys.stderr)

    errors = []
    for source, url in resolved.urls:
        t0 = time.monotonic()
        if not quiet:
            print("  source: %s" % source, file=sys.stderr)
        try:
            _download_from(url, part, resolved.size_bytes, t0)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                DownloadError) as e:
            errors.append("%s: %s" % (source, e))
            continue

        digest = _sha256_of(part)
        if digest != resolved.sha256:
            errors.append("%s: sha256 mismatch (got %s, want %s)" % (
                source, digest[:16], resolved.sha256[:16]))
            if part.exists():
                part.unlink()
            continue

        os.replace(part, dest)
        if not quiet:
            print("  verified sha256, saved to %s" % dest, file=sys.stderr)
        return dest

    raise DownloadError(
        "could not download %s from any source:\n    %s" % (
            resolved.name, "\n    ".join(errors)),
        hint=("check your network; behind the GFW try "
              "RAPIDSPEECH_SOURCE=modelscope, or download manually from "
              "https://www.modelscope.cn/models/RapidAI/RapidSpeech and place "
              "the file at %s" % dest),
    )


def cached_files():
    """List (Path, size) of files currently in the model cache."""
    d = models_dir()
    if not d.is_dir():
        return []
    return sorted((p, p.stat().st_size) for p in d.iterdir() if p.is_file())
