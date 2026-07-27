"""RapidSpeech Python package.

The compiled pybind11 extension (``rapidspeech.rapidspeech``) provides
``asr_offline`` / ``tts_synthesizer`` / ``vad``. Its import is deferred
behind a guard so that the pure-Python CLI (``rapidspeech doctor`` in
particular) still works when the native module is broken or missing —
that is exactly the situation ``doctor`` exists to diagnose.
"""

try:
    from rapidspeech.rapidspeech import *   # noqa: F401,F403
    _NATIVE_IMPORT_ERROR = None
except ImportError as _e:
    _NATIVE_IMPORT_ERROR = _e


def __getattr__(name):
    # Pure-Python submodules live in rapidspeech.python.* — keep them
    # importable as rapidspeech.* for backwards compat and because the
    # CLI and tests use the flat namespace.
    import importlib
    for pkg in ("rapidspeech", "rapidspeech.python"):
        try:
            return importlib.import_module(pkg + "." + name)
        except ImportError:
            pass
    if _NATIVE_IMPORT_ERROR is not None:
        raise ImportError(
            "rapidspeech native module failed to import: %s\n"
            "Run `rapidspeech doctor` to diagnose, or reinstall: "
            "pip install --force-reinstall rapidspeech" % _NATIVE_IMPORT_ERROR
        )
    raise AttributeError("module 'rapidspeech' has no attribute %r" % name)
