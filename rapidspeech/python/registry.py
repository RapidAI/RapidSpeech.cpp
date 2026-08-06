"""Model registry for RapidSpeech.

Loads the machine-readable model list shipped with the package
(``rapidspeech.python._registry.json``) and resolves user-facing model specs
such as ``funasr-nano`` or ``kokoro-zh:q8_0`` into concrete download
entries (size, sha256, mirror paths).

This module is pure Python and must stay importable even when the
native ``rapidspeech.rapidspeech`` extension is missing, so that
``rapidspeech doctor`` can diagnose broken installs.
"""

import json
import os
from pathlib import Path


class RegistryError(Exception):
    """Raised for unknown model IDs / variants. Carries an actionable hint."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint


class ResolvedFile:
    """A single downloadable file resolved from the registry."""

    def __init__(self, name, filename, size_bytes, sha256, urls,
                 known_issues=None, env=None):
        self.name = name              # e.g. "funasr-nano:q4_k_m" or "kokoro-zh/voice:zf_001"
        self.filename = filename      # cache filename, e.g. "funasr-nano-q4_k_m.gguf"
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.urls = urls              # ordered list of (source_name, url)
        self.known_issues = known_issues
        self.env = env                # optional environment variable for companion files


class Registry:
    def __init__(self, data):
        self.data = data

    @property
    def models(self):
        return self.data["models"]

    @property
    def defaults(self):
        return self.data.get("defaults", {})

    def default_model(self, task):
        """Return the default model name for a task ("asr" / "tts" / "vad")."""
        name = self.defaults.get(task)
        if not name or name not in self.models:
            raise RegistryError(
                "registry has no default %s model" % task,
                hint="run `rapidspeech models` to list available models",
            )
        return name

    # -- spec resolution ------------------------------------------------

    def split_spec(self, spec):
        """Split "name[:variant]" into (name, variant-or-None)."""
        if ":" in spec:
            name, variant = spec.split(":", 1)
            return name.strip(), variant.strip()
        return spec.strip(), None

    def resolve(self, spec):
        """Resolve a model spec ("name" or "name:variant") to a ResolvedFile."""
        name, variant = self.split_spec(spec)
        entry = self.models.get(name)
        if entry is None:
            known = ", ".join(sorted(self.models))
            raise RegistryError(
                "unknown model '%s'" % name,
                hint="known models: %s (see `rapidspeech models`)" % known,
            )
        if variant is None:
            variant = entry["default_variant"]
        var = entry["variants"].get(variant)
        if var is None:
            known = ", ".join(sorted(entry["variants"]))
            raise RegistryError(
                "model '%s' has no variant '%s'" % (name, variant),
                hint="available variants: %s" % known,
            )
        return self._resolved("%s:%s" % (name, variant), var)

    def resolve_voice(self, model_name, voice):
        """Resolve a named voice pack of a TTS model to a ResolvedFile."""
        entry = self.models.get(model_name)
        if entry is None or "voices" not in entry:
            raise RegistryError("model '%s' has no voice packs" % model_name)
        v = entry["voices"].get(voice)
        if v is None:
            known = ", ".join(sorted(entry["voices"]))
            raise RegistryError(
                "model '%s' has no voice '%s'" % (model_name, voice),
                hint="available voices: %s" % known,
            )
        return self._resolved("%s/voice:%s" % (model_name, voice), v)

    def resolve_dependencies(self, model_name):
        """Resolve companion files needed by a registry model.

        Dependencies are downloaded with the main model and, when they carry an
        env key, exported before native model load.
        """
        entry = self.models.get(model_name)
        if entry is None:
            known = ", ".join(sorted(self.models))
            raise RegistryError(
                "unknown model '%s'" % model_name,
                hint="known models: %s (see rapidspeech models)" % known,
            )
        deps = entry.get("dependencies", {})
        return [
            self._resolved("%s/dependency:%s" % (model_name, dep_name), dep)
            for dep_name, dep in sorted(deps.items())
        ]

    def default_voice(self, model_name):
        entry = self.models.get(model_name, {})
        return entry.get("default_voice")

    def _resolved(self, name, file_entry):
        urls = []
        for source in self._source_order():
            rel = file_entry["files"].get(source)
            if rel is None:
                continue
            base = self._source_base(source)
            urls.append((source, base + rel))
        if not urls:
            raise RegistryError("registry entry '%s' has no download sources" % name)
        filename = os.path.basename(next(iter(file_entry["files"].values())))
        return ResolvedFile(
            name=name,
            filename=filename,
            size_bytes=file_entry["size_bytes"],
            sha256=file_entry["sha256"],
            urls=urls,
            known_issues=file_entry.get("known_issues"),
            env=file_entry.get("env"),
        )

    # -- mirror configuration --------------------------------------------

    def _source_order(self):
        """Download source priority. Override with RAPIDSPEECH_SOURCE
        (comma-separated, e.g. "huggingface,modelscope")."""
        env = os.environ.get("RAPIDSPEECH_SOURCE", "")
        if env.strip():
            order = [s.strip() for s in env.split(",") if s.strip()]
        else:
            order = ["modelscope", "huggingface"]
        return [s for s in order if s in self.data["sources"]]

    def _source_base(self, source):
        base = self.data["sources"][source]
        # Allow rerouting huggingface through a mirror (e.g. hf-mirror.com).
        if source == "huggingface":
            ep = os.environ.get("RAPIDSPEECH_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT")
            if ep:
                base = ep.rstrip("/") + "/" + self.data["repo"] + "/resolve/main/"
        return base


def registry_path():
    env = os.environ.get("RAPIDSPEECH_REGISTRY_PATH")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent / "_registry.json"


def load_registry(path=None):
    p = Path(path) if path else registry_path()
    if not p.is_file():
        raise RegistryError(
            "model registry not found at %s" % p,
            hint="reinstall the package: pip install --force-reinstall rapidspeech",
        )
    with open(p, "r", encoding="utf-8") as f:
        return Registry(json.load(f))
