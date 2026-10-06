"""
Figure output policy: PDF only by default.

Figures are kept as PDF (vector, paper-ready, ~5x smaller than the former 300-dpi PNGs). A request to save a
.png is redirected to the .pdf with the same stem, so every figure is still written exactly once, including
code paths that only ever saved a PNG.

Set SAVE_PNG=1 to also write the PNGs (e.g. for slides); they are then rendered at PNG_DPI (default 200).

Installed on import of the `core` or `plots` package, which covers every pipeline entry point.
"""
import os

import matplotlib.figure

_original_savefig = matplotlib.figure.Figure.savefig


def png_enabled() -> bool:
    return os.environ.get("SAVE_PNG", "0").lower() in ("1", "true", "yes")


def _policy_savefig(self, fname, *args, **kwargs):
    if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png"):
        if png_enabled():
            kwargs["dpi"] = int(os.environ.get("PNG_DPI", 200))
        else:
            fname = str(fname)[:-4] + ".pdf"
            kwargs.pop("format", None)
    return _original_savefig(self, fname, *args, **kwargs)


def install():
    if not getattr(matplotlib.figure.Figure.savefig, "_pdf_only_policy", False):
        _policy_savefig._pdf_only_policy = True
        matplotlib.figure.Figure.savefig = _policy_savefig


install()
