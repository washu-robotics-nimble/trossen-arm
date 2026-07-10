"""
Forward/inverse kinematics for the Trossen WXAI V0 arm.

The trossen_arm 1.7.8 driver exposes joint-space control only, so cartesian
writing motions require our own FK/IK.  Joint frames are taken verbatim from
the official URDF:
https://github.com/TrossenRobotics/trossen_arm_description
(urdf/generated/wxai/wxai_base.urdf)

Frames: robot base frame has x forward, y left, z up.  The tool point is the
marker tip, offset along the gripper x-axis from the ee_gripper_link.

Run a self-test (no robot needed):
  python -m control.writing.kinematics
"""

import numpy as np

# (translation from parent link, rotation axis) per revolute joint — from URDF
_JOINTS = [
    (np.array([0.0, 0.0, 0.05725]), np.array([0.0, 0.0, 1.0])),    # joint_0 base yaw
    (np.array([0.02, 0.0, 0.04625]), np.array([0.0, 1.0, 0.0])),   # joint_1 shoulder
    (np.array([-0.264, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])),    # joint_2 elbow
    (np.array([0.245, 0.0, 0.06]), np.array([0.0, -1.0, 0.0])),    # joint_3 wrist pitch
    (np.array([0.06775, 0.0, 0.0455]), np.array([0.0, 0.0, -1.0])),  # joint_4 wrist yaw
    (np.array([0.02895, 0.0, -0.0455]), np.array([1.0, 0.0, 0.0])),  # joint_5 wrist roll
]

# link_6 -> ee_gripper_link (center of gripper fingers), from URDF
_EE_OFFSET = np.array([0.156062, 0.0, 0.0])

JOINT_LIMITS = np.array([
    [-3.054, 3.054],
    [0.0, 3.1416],
    [0.0, 2.3562],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-3.1416, 3.1416],
])


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit axis."""
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def forward_kinematics(q, tip_length: float = 0.0):
    """Pose of the marker tip in the base frame.

    Args:
        q: 6 joint angles (rad), driver order.
        tip_length: distance (m) from the gripper-finger center to the marker
            tip, measured along the gripper x-axis.

    Returns:
        (position (3,), rotation (3,3)) — columns of the rotation are the tool
        frame axes; column 0 (x-axis) is the marker pointing direction.
    """
    q = np.asarray(q, dtype=np.float64)
    R = np.eye(3)
    p = np.zeros(3)
    for (offset, axis), angle in zip(_JOINTS, q):
        p = p + R @ offset
        R = R @ _rot(axis, angle)
    tool = _EE_OFFSET + np.array([tip_length, 0.0, 0.0])
    return p + R @ tool, R


def _pose_error(q, target_pos, target_dir, tip_length, dir_weight):
    pos, R = forward_kinematics(q, tip_length)
    e_pos = target_pos - pos
    if target_dir is None:
        return e_pos
    # cross product of current vs desired tool x-axis = axis-angle error (small angles)
    e_dir = np.cross(R[:, 0], target_dir)
    return np.concatenate([e_pos, dir_weight * e_dir])


def inverse_kinematics(
    target_pos,
    target_dir=None,
    q_init=None,
    tip_length: float = 0.0,
    dir_weight: float = 0.05,
    max_iters: int = 200,
    tol: float = 5e-4,
    damping: float = 1e-3,
):
    """Damped least-squares IK for the marker tip.

    Args:
        target_pos: desired tip position (3,) in base frame.
        target_dir: desired tool x-axis (marker pointing direction), unit (3,).
            None leaves orientation free.
        q_init: seed joint angles; pass the previous waypoint's solution for
            continuity along a stroke.
        tip_length: gripper-center-to-marker-tip distance (m).
        dir_weight: meters-per-radian weighting of the direction error.
        tol: position tolerance (m) for convergence.

    Returns:
        (q (6,), converged: bool, pos_error: float)
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    if target_dir is not None:
        target_dir = np.asarray(target_dir, dtype=np.float64)
        target_dir = target_dir / np.linalg.norm(target_dir)

    q = (np.array(q_init, dtype=np.float64) if q_init is not None
         else JOINT_LIMITS.mean(axis=1))

    eps = 1e-6
    for _ in range(max_iters):
        e = _pose_error(q, target_pos, target_dir, tip_length, dir_weight)
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e) < 2 * tol:
            return q, True, float(np.linalg.norm(e[:3]))

        # numerical Jacobian
        J = np.zeros((len(e), 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = eps
            J[:, j] = (_pose_error(q + dq, target_pos, target_dir, tip_length, dir_weight) - e) / eps

        # Newton/DLS step solving e + J*dq = 0  (J = de/dq, so step is -J^+ e)
        JJt = J @ J.T
        step = -J.T @ np.linalg.solve(JJt + damping * np.eye(len(e)), e)
        # limit step size for stability
        step_norm = np.linalg.norm(step)
        if step_norm > 0.5:
            step *= 0.5 / step_norm
        q = np.clip(q + step, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    e = _pose_error(q, target_pos, target_dir, tip_length, dir_weight)
    pos_err = float(np.linalg.norm(e[:3]))
    # Match the in-loop convergence test: BOTH position and orientation must
    # be satisfied. Reporting success on position alone would accept a
    # solution whose marker direction is far off (wrong pen tilt).
    converged = pos_err < tol and float(np.linalg.norm(e)) < 2 * tol
    return q, converged, pos_err


def _selftest():
    print("FK at zero pose:")
    pos, R = forward_kinematics(np.zeros(6))
    print(f"  tip = {np.round(pos, 4)}  (expect ~[0.254, 0, 0.164])")

    print("FK at draw_a.py touch pose [0, pi/2, pi/2, 0, 0, 0]:")
    pos, R = forward_kinematics([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])
    print(f"  tip = {np.round(pos, 4)}  (expect forward reach, ~[0.52, 0, 0.43])")
    print(f"  tool x-axis = {np.round(R[:, 0], 4)}  (expect ~[1, 0, 0])")

    print("IK round-trip tests:")
    rng_qs = [
        [0.3, 1.2, 1.0, 0.2, -0.3, 0.0],
        [-0.5, 1.5, 0.8, -0.4, 0.2, 0.1],
        [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0],
    ]
    ok = True
    for q_true in rng_qs:
        pos, R = forward_kinematics(q_true, tip_length=0.1)
        q_sol, converged, err = inverse_kinematics(
            pos, target_dir=R[:, 0], q_init=np.array(q_true) + 0.2, tip_length=0.1
        )
        pos2, R2 = forward_kinematics(q_sol, tip_length=0.1)
        pos_err = np.linalg.norm(pos2 - pos)
        dir_err = np.degrees(np.arccos(np.clip(R2[:, 0] @ R[:, 0], -1, 1)))
        status = "OK" if (converged and pos_err < 1e-3) else "FAIL"
        ok &= status == "OK"
        print(f"  {status}: pos_err={pos_err*1000:.2f} mm, dir_err={dir_err:.2f} deg")

    print("\nSelf-test", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
