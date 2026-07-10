"""
Vector glyph strokes for ARBITRARY Unicode text (incl. Chinese, Japanese,
Korean) from a real TrueType/OpenType font, via matplotlib's TextPath.

Unlike the Hershey fonts (Latin single-stroke only), this works for any
character the chosen font contains.  The catch: TTF glyphs are OUTLINES —
the marker traces the contour of each stroke, so characters look outlined
rather than drawn with a single brush stroke.  For pen-plotting that is the
normal, expected result.

Output matches hershey.text_to_strokes: (strokes, width, height) with each
stroke an (N, 2) array in meters, u right / v up, text bottom-left at (0, 0).

Preview:
  python -m control.writing.glyphs "你好世界" preview.png
"""

import os

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

# CJK-capable fonts commonly present on macOS, tried in order.
_DEFAULT_CJK_FONTS = [
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]

# Reference glyph whose height defines char_height.  '永' fills the CJK em box;
# fall back to 'H' for fonts without it.
_REF_CJK = "永"  # 永
_REF_LATIN = "H"


def default_cjk_font():
    """Return the first available bundled CJK font path, or None."""
    for p in _DEFAULT_CJK_FONTS:
        if os.path.exists(p):
            return p
    return None


def _polylines(text, fp, size=100.0):
    """Flattened outline polylines for `text` at nominal `size` (font units)."""
    tp = TextPath((0, 0), text, size=size, prop=fp)
    return [np.asarray(p, dtype=np.float64) for p in tp.to_polygons() if len(p) >= 2]


def text_to_strokes_ttf(text, char_height=0.05, font_path=None,
                        line_spacing=1.6):
    """Convert text to outline strokes using a TTF/OTF font.

    Args:
        text: text to render; '\\n' starts a new line.
        char_height: target glyph height in meters.
        font_path: path to a .ttf/.otf/.ttc font.  Defaults to a bundled CJK
            font (STHeiti, ...).
        line_spacing: baseline-to-baseline distance as a multiple of char_height.

    Returns:
        (strokes, width, height): strokes in meters, u right / v up, with the
        text block's bottom-left at (0, 0).

    Raises:
        FileNotFoundError if no usable font is found.
    """
    if font_path is None:
        font_path = default_cjk_font()
    if not font_path or not os.path.exists(font_path):
        raise FileNotFoundError(
            "No usable font found for non-Latin text. Pass --font-file <path> "
            "to a .ttf/.otf/.ttc that contains the characters.")

    fp = FontProperties(fname=font_path)

    # Height scale from a reference glyph so char_height is respected.
    ref = _polylines(_REF_CJK, fp) or _polylines(_REF_LATIN, fp)
    if not ref:
        raise ValueError(f"Font {font_path} produced no glyph outlines.")
    ref_pts = np.vstack(ref)
    ref_h = ref_pts[:, 1].max() - ref_pts[:, 1].min()
    scale = char_height / ref_h

    strokes = []
    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            continue
        v_offset = -i * line_spacing * char_height
        for p in _polylines(line, fp):
            p = p * scale
            p[:, 1] += v_offset
            strokes.append(p)

    if not strokes:
        return [], 0.0, 0.0

    all_pts = np.vstack(strokes)
    u_min, v_min = all_pts.min(axis=0)
    u_max, v_max = all_pts.max(axis=0)
    for s in strokes:
        s -= [u_min, v_min]

    return strokes, float(u_max - u_min), float(v_max - v_min)


if __name__ == "__main__":
    import sys

    from .hershey import preview_strokes

    text = sys.argv[1] if len(sys.argv) > 1 else "你好世界"
    out = sys.argv[2] if len(sys.argv) > 2 else "glyphs_preview.png"
    strokes, w, h = text_to_strokes_ttf(text, char_height=0.05)
    n = sum(len(s) for s in strokes)
    print(f"'{text}': {len(strokes)} strokes, {n} points, {w*100:.1f} x {h*100:.1f} cm")
    preview_strokes(strokes, out)
    print(f"Preview saved to {out}")
