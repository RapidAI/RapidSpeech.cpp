"""Surface tests: no model files needed.

Every check here catches a class of regression the existing
`python-bindings-test` job (single `import rapidspeech`) cannot:

- `import rapidspeech` succeeds **and** carries a non-empty version string.
- The three documented classes are exported and callable.
- Constructors raise (don't crash) on a non-existent model path.
- The aliases used in older examples (AsrOffline / TtsSynthesizer / Vad)
  still resolve to the same classes.
"""

from __future__ import annotations

import os

import pytest


def test_import_and_version():
    import rapidspeech

    assert hasattr(rapidspeech, "version")
    v = rapidspeech.version()
    assert isinstance(v, str) and len(v) > 0


def test_classes_exported():
    import rapidspeech

    for name in ("asr_offline", "tts_synthesizer", "vad"):
        assert hasattr(rapidspeech, name), f"missing {name}"
        assert callable(getattr(rapidspeech, name))


def test_legacy_aliases():
    import rapidspeech

    assert rapidspeech.AsrOffline is rapidspeech.asr_offline
    assert rapidspeech.TtsSynthesizer is rapidspeech.tts_synthesizer
    assert rapidspeech.Vad is rapidspeech.vad


@pytest.mark.parametrize(
    "factory_name",
    ["asr_offline", "tts_synthesizer", "vad"],
)
def test_bogus_path_raises(factory_name: str, tmp_path):
    """Loading a path that doesn't exist must raise — not abort."""
    import rapidspeech

    factory = getattr(rapidspeech, factory_name)
    bogus = str(tmp_path / "does_not_exist.gguf")
    with pytest.raises(RuntimeError):
        factory(bogus)


def test_asr_methods_present():
    """Catch pybind signature drift before users do."""
    import rapidspeech

    expected = {
        "push_audio",
        "process",
        "redecode",
        "reset",
        "get_text",
        "set_user_input_prompt",
        "set_use_llm",
        "set_ctc_precheck",
        "get_model_meta",
        "get_backend_name",
    }
    missing = expected - set(dir(rapidspeech.asr_offline))
    assert not missing, f"asr_offline missing methods: {missing}"


def test_tts_methods_present():
    import rapidspeech

    expected = {
        "set_params",
        "set_diffusion_steps",
        "set_reference_audio",
        "set_reference_text",
        "synthesize",
        "synthesize_streaming",
        "get_sample_rate",
        "get_model_meta",
        "get_backend_name",
    }
    missing = expected - set(dir(rapidspeech.tts_synthesizer))
    assert not missing, f"tts_synthesizer missing methods: {missing}"


def test_vad_methods_present():
    import rapidspeech

    expected = {
        "reset",
        "set_threshold",
        "push_audio",
        "is_speech",
        "get_probability",
        "get_arch",
        "drain_segments",
        "drain_frames",
        "detect_full",
    }
    missing = expected - set(dir(rapidspeech.vad))
    assert not missing, f"vad missing methods: {missing}"
