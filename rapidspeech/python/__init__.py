"""Pure-Python layer of rapidspeech: cli, registry and download.

Kept separate from the package root so the compiled extension
(``rapidspeech.rapidspeech``) and its shared libraries stay the only
things there. ``rapidspeech.__getattr__`` re-exports these modules under
the flat namespace (``rapidspeech.cli`` etc.) for backwards compat.
"""
