"""
Plan joint-space trajectories that draw 2D strokes on the calibrated whiteboard.

The full trajectory is planned (all IK solved) BEFORE any motion is sent to
the robot, so an unreachable stroke aborts the run instead of stopping the
marker mid-letter.

The planned trajectory is a list of Waypoint(q, duration, pen_down) executed
sequentially with blocking set_arm_positions calls, mirroring the playback
approach in control/scripts/teach_and_playback.py.

Run a self-test with a synthetic board (no robot needed):
  python -m control.writing.trajectory
"""

from dataclasses import dataclass

import numpy as np

from .kinematics import forward_kinematics, inverse_kinematics


@dataclass
class Waypoint:
    q: np.ndarray        # 6 joint angles
    duration: float      # seconds allotted to reach this waypoint
    pen_down: bool       # True while the marker is on the board


class PlanningError(RuntimeError):
    pass


def _resample(stroke: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at uniform arc-length spacing."""
    deltas = np.diff(stroke, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total = seg_lens.sum()
    if total < 1e-9:
        return stroke[:1]
    n = max(int(np.ceil(total / spacing)), 1)
    targets = np.linspace(0.0, total, n + 1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    out = np.empty((len(targets), 2))
    for i, t in enumerate(targets):
        k = min(np.searchsorted(cum, t, side="right") - 1, len(seg_lens) - 1)
        frac = (t - cum[k]) / seg_lens[k] if seg_lens[k] > 1e-12 else 0.0
        out[i] = stroke[k] + frac * deltas[k]
    return out


def plan_strokes(
    plane,
    strokes,
    q_start,
    write_speed: float = 0.04,
    travel_speed: float = 0.10,
    hover: float = 0.03,
    press: float = 0.0,
    dt: float = 0.1,
    margin: float = 0.01,
    surface_offset: float = 0.0,
):
    """Plan a joint trajectory drawing `strokes` (2D board coords, meters).

    Args:
        plane: WhiteboardPlane from calibration.
        strokes: list of (N, 2) arrays in board coordinates.
        q_start: current arm joint angles, used as the IK seed.
        write_speed: marker speed on the board (m/s).
        travel_speed: pen-up speed between strokes (m/s).
        hover: pen-up distance off the board (m).
        press: how far past the board plane to press while writing (m); a
            couple of millimeters compensates calibration error and marker
            tip compliance.
        dt: time per pen-down waypoint (s).
        margin: minimum distance from strokes to the board edge (m).
        surface_offset: measured offset (m) of the true writing surface along
            the board normal, relative to the calibrated plane (from a contact
            probe).  Positive = surface is nearer the robot than calibrated
            (marker protruding past the calibrated tip).  All hover/pen-down
            heights shift by this so writing tracks the real surface.

    Returns:
        list[Waypoint].  Raises PlanningError if any point is off the board
        or IK fails to converge.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if write_speed <= 0 or travel_speed <= 0:
        raise ValueError("write_speed and travel_speed must be > 0.")
    if hover < 0:
        raise ValueError("hover must be >= 0.")
    if margin < 0:
        raise ValueError("margin must be >= 0.")
    for i, s in enumerate(strokes):
        for u, v in s:
            if not plane.contains(u, v, margin):
                raise PlanningError(
                    f"Stroke {i} point (u={u:.3f}, v={v:.3f}) is outside the "
                    f"calibrated {plane.width:.2f} x {plane.height:.2f} m board "
                    f"(margin {margin} m). Reduce --height or move --start."
                )

    tip = plane.tip_length
    direction = plane.approach_dir()
    spacing = write_speed * dt

    def solve(pos, seed, what):
        q, ok, err = inverse_kinematics(
            pos, target_dir=direction, q_init=seed, tip_length=tip
        )
        if not ok:
            raise PlanningError(
                f"IK failed at {what} (target {np.round(pos, 3)}, "
                f"residual {err*1000:.1f} mm). The point may be out of reach — "
                "consider moving the board closer or writing lower/smaller."
            )
        return q

    waypoints = []
    seed = np.asarray(q_start, dtype=np.float64)
    prev_hover_3d = None

    for i, stroke in enumerate(strokes):
        pts = _resample(np.asarray(stroke, dtype=np.float64), spacing)

        hover_start = plane.point(*pts[0], offset=surface_offset + hover)
        hover_end = plane.point(*pts[-1], offset=surface_offset + hover)

        # pen-up travel to the start of this stroke
        travel_dist = (np.linalg.norm(hover_start - prev_hover_3d)
                       if prev_hover_3d is not None else 0.10)
        travel_time = max(travel_dist / travel_speed, 0.4)
        q = solve(hover_start, seed, f"stroke {i} hover-in")
        waypoints.append(Waypoint(q, travel_time, False))
        seed = q

        # pen down
        q = solve(plane.point(*pts[0], offset=surface_offset - press), seed, f"stroke {i} touch")
        waypoints.append(Waypoint(q, max(hover / write_speed, 0.3), True))
        seed = q

        # draw
        for j, (u, v) in enumerate(pts[1:], start=1):
            q = solve(plane.point(u, v, offset=surface_offset - press), seed, f"stroke {i} point {j}")
            waypoints.append(Waypoint(q, dt, True))
            seed = q

        # pen up
        q = solve(hover_end, seed, f"stroke {i} hover-out")
        waypoints.append(Waypoint(q, max(hover / write_speed, 0.3), False))
        seed = q
        prev_hover_3d = hover_end

    return waypoints


def execute(driver, waypoints, settle: float = 0.0):
    """Send a planned trajectory to the robot with blocking position moves."""
    import time

    for wp in waypoints:
        driver.set_arm_positions(np.asarray(wp.q, dtype=np.float64), wp.duration, True)
        if settle > 0:
            time.sleep(settle)


def resample_waypoints(waypoints, q_start, dt=0.1):
    """Resample a variable-duration trajectory into uniform dt joint targets.

    The planner emits waypoints with per-segment durations (slow travels, fast
    pen strokes).  For data recording we want evenly spaced samples so each
    observation/action pair covers the same wall-clock dt.  Linearly
    interpolates joint angles along each segment.

    Args:
        waypoints: list[Waypoint] from plan_strokes.
        q_start: joint angles the arm starts from (segment 0 origin).
        dt: sample period (s).

    Returns:
        list of (q (6,), pen_down: bool) at uniform dt spacing.
    """
    if not waypoints:
        return []

    # segment k: from prev_q -> waypoints[k].q over waypoints[k].duration
    prev_q = np.asarray(q_start, dtype=np.float64)
    segs = []
    t0 = 0.0
    for wp in waypoints:
        dur = max(wp.duration, 1e-3)
        segs.append((t0, t0 + dur, prev_q, np.asarray(wp.q, dtype=np.float64), wp.pen_down))
        t0 += dur
        prev_q = np.asarray(wp.q, dtype=np.float64)
    total = t0

    samples = []
    n = int(np.floor(total / dt)) + 1
    si = 0
    for i in range(n):
        t = i * dt
        while si < len(segs) - 1 and t > segs[si][1]:
            si += 1
        start_t, end_t, q0, q1, pen = segs[si]
        frac = 0.0 if end_t <= start_t else np.clip((t - start_t) / (end_t - start_t), 0.0, 1.0)
        samples.append((q0 + frac * (q1 - q0), pen))
    return samples


def _selftest():
    """Plan 'Hi' on a synthetic board 45 cm in front of the arm."""
    from .hershey import text_to_strokes
    from .whiteboard import WhiteboardPlane

    # vertical board facing the robot: bl/br/tl corners in base frame
    plane = WhiteboardPlane.from_corners(
        p_bl=[0.45, 0.15, 0.25],
        p_br=[0.45, -0.15, 0.25],
        p_tl=[0.45, 0.15, 0.55],
        tip_length=0.10,
    )
    print(f"Synthetic board: {plane.width:.2f} x {plane.height:.2f} m, "
          f"normal {np.round(plane.normal, 3)} (expect ~[-1, 0, 0])")

    strokes, w, h = text_to_strokes("Hi", char_height=0.06)
    # center the text on the board
    offset = np.array([(plane.width - w) / 2, (plane.height - h) / 2])
    strokes = [s + offset for s in strokes]

    q_home = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])
    wps = plan_strokes(plane, strokes, q_home, press=0.002)

    # verify every pen-down waypoint lands on the board plane
    max_err = 0.0
    for wp in wps:
        if wp.pen_down:
            pos, R = forward_kinematics(wp.q, tip_length=plane.tip_length)
            plane_dist = abs((pos - plane.origin) @ plane.normal + 0.002)
            max_err = max(max_err, plane_dist)

    n_down = sum(wp.pen_down for wp in wps)
    total_t = sum(wp.duration for wp in wps)
    print(f"Planned {len(wps)} waypoints ({n_down} pen-down), ~{total_t:.0f} s")
    print(f"Max pen-down distance from board plane: {max_err*1000:.2f} mm")
    ok = max_err < 0.002
    print("Self-test", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
