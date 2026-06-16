"""Shared fixtures + env-driven model paths.

Tests are split into three layers:

- Surface tests  — never load a model; check that classes / methods exist
                    with the expected signatures and that constructors raise
                    on bogus paths. Runs on every PR.
- Functional L2 — pushes audio through silero_vad + the smallest SenseVoice
                    quant. Requires RAPIDAI_MODELS_DIR pointing at a folder
                    that has both .gguf files. Skipped otherwise.
- Functional L3 — covers OmniVoice / Kokoro / CosyVoice3 / OpenVoice2 /
                    Speaker / KWS once their .gguf are published. Each test
                    looks for a specific env var and skips when missing.

The CI driver fills these envs after running download_models.sh.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _models_dir() -> Path | None:
    val = os.environ.get("RAPIDAI_MODELS_DIR")
    return Path(val) if val else None


def _has(name: str) -> bool:
    d = _models_dir()
    if d is None:
        return False
    return (d / name).is_file()


def _path(name: str) -> str:
    d = _models_dir()
    assert d is not None
    return str(d / name)


@pytest.fixture(scope="session")
def models_dir() -> Path:
    d = _models_dir()
    if d is None or not d.is_dir():
        pytest.skip("RAPIDAI_MODELS_DIR not set; functional tests skipped")
    return d


@pytest.fixture(scope="session")
def vad_model_path(models_dir: Path) -> str:
    name = "silero_vad_v6.gguf"
    if not (models_dir / name).is_file():
        pytest.skip(f"{name} not in $RAPIDAI_MODELS_DIR")
    return str(models_dir / name)


@pytest.fixture(scope="session")
def asr_model_path(models_dir: Path) -> str:
    name = "sense-voice-small-q3_k.gguf"
    if not (models_dir / name).is_file():
        pytest.skip(f"{name} not in $RAPIDAI_MODELS_DIR")
    return str(models_dir / name)


@pytest.fixture(scope="session")
def tts_model_path(models_dir: Path) -> str:
    name = "omnivoice-q2_k.gguf"
    if not (models_dir / name).is_file():
        pytest.skip(f"{name} not in $RAPIDAI_MODELS_DIR (set FULL=1 to fetch)")
    return str(models_dir / name)
