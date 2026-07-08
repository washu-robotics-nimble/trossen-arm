"""
Text -> pen strokes using single-stroke Hershey vector fonts.

Wraps the Hershey-Fonts package (pip install Hershey-Fonts).  The default
'futural' (Hershey simplex) font is single-stroke, which is what a marker
draws naturally — no outlines to fill in.

Coordinates are meters in 2D whiteboard coordinates: u to the right, v up,
origin at the text's bottom-left.  A stroke is an (N, 2) array drawn pen-down;
the pen lifts between strokes.

Run a self-test / preview (no robot needed):
  python -m control.writing.hershey "Hello!" preview.png
"""

import numpy as np
from HersheyFonts import HersheyFonts

_EPS = 1e-9


def _segments_to_strokes(segments):
    """Chain contiguous line segments into polyline strokes."""
    strokes = []
    current = []
    for (x1, y1), (x2, y2) in segments:
        if current and abs(current[-1][0] - x1) < _EPS and abs(current[-1][1] - y1) < _EPS:
            current.append((x2, y2))
        else:
            if len(current) >= 2:
                strokes.append(np.array(current))
            current = [(x1, y1), (x2, y2)]
    if len(current) >= 2:
        strokes.append(np.array(current))
    return strokes


def text_to_strokes(text: str, char_height: float = 0.05, line_spacing: float = 1.5,
                    font: str = "futural"):
    """Convert text to pen strokes.

    Args:
        text: the text to write; '\\n' starts a new line.
        char_height: capital letter height in meters.
        line_spacing: baseline-to-baseline distance as a multiple of char_height.
        font: Hershey font name ('futural' simplex, 'futuram' bold,
            'cursive', 'scripts', ...).

    Returns:
        (strokes, width, height): strokes are (N, 2) float arrays in meters,
        u right / v up, with the text block's bottom-left at (0, 0).
    """
    hf = HersheyFonts()
    hf.load_default_font(font)
    hf.normalize_rendering(1.0)  # nominal font size 1.0, scaled below

    lines = text.split("\n")

    # Measure the cap height in normalized units using a reference glyph so
    # char_height means "height of a capital letter".
    ref = _segments_to_strokes(list(hf.lines_for_text("H")))
    ref_pts = np.vstack(ref)
    cap = ref_pts[:, 1].max() - ref_pts[:, 1].min()
    scale = char_height / cap

    strokes = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        line_strokes = _segments_to_strokes(list(hf.lines_for_text(line)))
        v_offset = -i * line_spacing * char_height
        for s in line_strokes:
            s = s * scale
            s[:, 1] += v_offset
            strokes.append(s)

    if not strokes:
        return [], 0.0, 0.0

    all_pts = np.vstack(strokes)
    u_min, v_min = all_pts.min(axis=0)
    u_max, v_max = all_pts.max(axis=0)
    # shift so the text block's bottom-left is at (0, 0)
    for s in strokes:
        s -= [u_min, v_min]

    return strokes, float(u_max - u_min), float(v_max - v_min)


def preview_strokes(strokes, path: str, px_per_m: float = 4000):
    """Render strokes to a PNG for a quick visual check."""
    import cv2

    all_pts = np.vstack(strokes)
    w_m, h_m = all_pts[:, 0].max(), all_pts[:, 1].max()
    pad = 40
    w_px, h_px = int(w_m * px_per_m) + 2 * pad, int(h_m * px_per_m) + 2 * pad
    img = np.full((max(h_px, 1), max(w_px, 1), 3), 255, dtype=np.uint8)
    for s in strokes:
        pts = np.round(s * px_per_m).astype(int)
        pts[:, 0] += pad
        pts[:, 1] = h_px - pad - pts[:, 1]  # flip v (image y is down)
        cv2.polylines(img, [pts], False, (40, 40, 40), 3, cv2.LINE_AA)
    cv2.imwrite(path, img)


if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Hello\nWorld!"
    out = sys.argv[2] if len(sys.argv) > 2 else "hershey_preview.png"
    strokes, w, h = text_to_strokes(text, char_height=0.05)
    n_pts = sum(len(s) for s in strokes)
    print(f"'{text}': {len(strokes)} strokes, {n_pts} points, {w*100:.1f} x {h*100:.1f} cm")
    preview_strokes(strokes, out)
    print(f"Preview saved to {out}")
