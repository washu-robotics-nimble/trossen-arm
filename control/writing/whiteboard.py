"""
Whiteboard plane model built from a 3-corner touch calibration.

The operator guides the marker tip to three corners of the writing area
(bottom-left, bottom-right, top-left as seen facing the board); FK converts
the recorded joint angles into 3D points, from which we build an orthonormal
board frame:

  origin = bottom-left corner
  u_hat  = along the bottom edge (toward bottom-right)
  v_hat  = up the board (orthogonalized)
  normal = u_hat x v_hat, pointing off the board back toward the robot

point(u, v, offset) maps 2D board coordinates (meters) to 3D robot-base
coordinates, with positive offset hovering off the writing surface.
"""

import json
import os

import numpy as np

CALIBRATION_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../config/whiteboard_calibration.json")
)


class WhiteboardPlane:
    def __init__(self, origin, u_hat, v_hat, width, height, tip_length):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.u_hat = np.asarray(u_hat, dtype=np.float64)
        self.v_hat = np.asarray(v_hat, dtype=np.float64)
        self.normal = np.cross(self.u_hat, self.v_hat)
        self.width = float(width)
        self.height = float(height)
        self.tip_length = float(tip_length)

    @classmethod
    def from_corners(cls, p_bl, p_br, p_tl, tip_length):
        p_bl, p_br, p_tl = (np.asarray(p, dtype=np.float64) for p in (p_bl, p_br, p_tl))
        u_raw = p_br - p_bl
        width = np.linalg.norm(u_raw)
        if width < 0.05:
            raise ValueError("Bottom-left and bottom-right corners are less than 5 cm apart.")
        u_hat = u_raw / width

        v_raw = p_tl - p_bl
        v_orth = v_raw - (v_raw @ u_hat) * u_hat
        height = np.linalg.norm(v_orth)
        if height < 0.05:
            raise ValueError("Top-left corner is less than 5 cm off the bottom edge.")
        v_hat = v_orth / height

        return cls(p_bl, u_hat, v_hat, width, height, tip_length)

    def point(self, u, v, offset=0.0):
        """3D robot-base position for board coords (u, v), offset off the surface."""
        return self.origin + u * self.u_hat + v * self.v_hat + offset * self.normal

    def approach_dir(self):
        """Marker pointing direction: into the board (anti-normal)."""
        return -self.normal

    def contains(self, u, v, margin=0.0):
        return (margin <= u <= self.width - margin) and (margin <= v <= self.height - margin)

    def save(self, path=CALIBRATION_PATH):
        data = {
            "origin": self.origin.tolist(),
            "u_hat": self.u_hat.tolist(),
            "v_hat": self.v_hat.tolist(),
            "width": self.width,
            "height": self.height,
            "tip_length": self.tip_length,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path=CALIBRATION_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No whiteboard calibration at {path}. "
                "Run: python control/scripts/calibrate_whiteboard.py"
            )
        with open(path) as f:
            d = json.load(f)
        return cls(d["origin"], d["u_hat"], d["v_hat"], d["width"], d["height"], d["tip_length"])
