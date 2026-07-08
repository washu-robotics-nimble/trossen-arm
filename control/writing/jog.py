"""
Cartesian keyboard-jog helper for the WXAI arm.

The arm stays in POSITION mode (holding itself up — no backdriving), and the
operator nudges the marker tip along base-frame axes with single keypresses.
Each keypress solves IK for the new tip target and commands a short blocking
move.  Used by calibrate_whiteboard.py to capture corner points without
fighting joint friction.

Base frame: x forward (toward the board), y left, z up.
"""

import sys
import termios
import tty

import numpy as np

from .kinematics import forward_kinematics, inverse_kinematics, JOINT_LIMITS

# key -> unit displacement of the tip in base frame (x, y, z)
_MOVES = {
    "\x1b[A": (0.0, 0.0, 1.0),    # up arrow    -> +z (up)
    "\x1b[B": (0.0, 0.0, -1.0),   # down arrow  -> -z (down)
    "\x1b[D": (0.0, 1.0, 0.0),    # left arrow  -> +y (left)
    "\x1b[C": (0.0, -1.0, 0.0),   # right arrow -> -y (right)
    "w": (1.0, 0.0, 0.0),         # w -> +x (toward board / pen in)
    "s": (-1.0, 0.0, 0.0),        # s -> -x (away / pen out)
}
_STEP_SIZES = [0.001, 0.002, 0.005, 0.01, 0.02]  # meters


def _getch():
    """Read one keypress (handles 3-byte arrow escape sequences) in raw mode."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _status(msg):
    sys.stdout.write("\r\x1b[K" + msg)
    sys.stdout.flush()


def jog_to_point(driver, q_start, tip_length, target_dir, label,
                 move_time=0.25):
    """Interactively jog the marker tip; returns (q, tip_pos) when captured.

    Args:
        driver: configured TrossenArmDriver, already in position mode.
        q_start: current joint angles (IK seed / starting pose).
        tip_length: marker-tip offset (m).
        target_dir: tool x-axis to hold during jogging (unit vector).
        label: corner name shown in the prompt.
        move_time: seconds per commanded step (smaller = snappier).

    Returns:
        (q (6,), tip_pos (3,)) on capture, or (None, None) if aborted.
    """
    q = np.array(q_start, dtype=np.float64)
    pos, _ = forward_kinematics(q, tip_length)
    step_idx = 2  # default 5 mm

    print(f"\n── Jog to {label} ──")
    print("  arrows = move along board (up/down/left/right)")
    print("  w / s  = pen in / out (toward / away from board)")
    print("  [ / ]  = smaller / larger step    SPACE = capture    q = abort")

    while True:
        step = _STEP_SIZES[step_idx]
        _status(f"tip x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f} m | "
                f"step={step*1000:.0f} mm | [{label}]")

        ch = _getch()

        if ch in ("q", "\x03"):  # q or Ctrl-C
            print("\n  Aborted.")
            return None, None

        if ch == " ":
            # average a few samples of the true position for a steady capture
            # (get_positions returns 6 arm joints + gripper; keep the arm)
            samples = [np.array(driver.get_positions()[:6], dtype=np.float64)
                       for _ in range(10)]
            q_final = np.mean(samples, axis=0)
            pos_final, _ = forward_kinematics(q_final, tip_length)
            print(f"\n  Captured {label}: {np.round(pos_final, 4)} m")
            return q_final, pos_final

        if ch == "[":
            step_idx = max(0, step_idx - 1)
            continue
        if ch == "]":
            step_idx = min(len(_STEP_SIZES) - 1, step_idx + 1)
            continue

        if ch not in _MOVES:
            continue

        delta = np.array(_MOVES[ch]) * step
        target = pos + delta
        q_new, ok, err = inverse_kinematics(
            target, target_dir=target_dir, q_init=q, tip_length=tip_length
        )
        if not ok:
            _status(f"\x07unreachable that way (residual {err*1000:.0f} mm) — try another axis")
            continue
        if np.any(q_new <= JOINT_LIMITS[:, 0] + 1e-3) or np.any(q_new >= JOINT_LIMITS[:, 1] - 1e-3):
            _status("\x07joint limit reached — try another axis")
            continue

        driver.set_arm_positions(q_new, move_time, True)
        q = q_new
        pos, _ = forward_kinematics(q, tip_length)
