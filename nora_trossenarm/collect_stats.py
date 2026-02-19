"""Collect action statistics by manually moving the arm in gravity-comp mode.

The arm enters external_effort mode (gravity compensation) so you can
freely move it by hand. It records Cartesian deltas between timesteps.
When done, it computes q01/q99 stats and saves them as a custom unnorm key.

Usage:
    python nora_trossenarm/collect_stats.py

    1. Arm enters gravity-comp mode — move it around by hand
    2. Perform the kinds of motions you want NORA to do (reaching, picking, etc.)
    3. Press Ctrl+C when done
    4. Stats are saved to nora_trossenarm/custom_norm_stats.json
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

import ikpy.chain
from scipy.spatial.transform import Rotation

import trossen_arm

sys.path.insert(0, str(Path(__file__).parent.parent))

from util.hamer import robot_ip

URDF_PATH = str(Path(__file__).parent / "src" / "wxai_base.urdf")
RECORD_HZ = 10  # recording frequency


def build_ik_chain():
    chain = ikpy.chain.Chain.from_urdf_file(
        URDF_PATH,
        base_elements=["base_link"],
        active_links_mask=[False, True, True, True, True, True, True, False, False],
        name="wxai",
    )
    return chain


def joint_angles_to_cartesian(chain, joint_angles_6):
    full = np.zeros(9)
    full[1:7] = joint_angles_6
    fk = chain.forward_kinematics(full)
    pos = fk[:3, 3]
    rot = Rotation.from_matrix(fk[:3, :3]).as_euler("xyz")
    return np.concatenate([pos, rot])


if __name__ == "__main__":
    print("Building IK chain...")
    chain = build_ik_chain()

    print(f"Connecting to arm at {robot_ip}...")
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        robot_ip,
        False,
    )

    # Enter gravity compensation mode — arm can be moved freely by hand
    print("Entering gravity-comp mode. Move the arm around by hand.")
    print("Perform reaching, picking, placing motions.")
    print("Press Ctrl+C when done.\n")
    driver.set_arm_modes(trossen_arm.Mode.external_effort)
    driver.set_arm_external_efforts(np.zeros(6).tolist(), 0.0, False)

    # Record Cartesian poses
    poses = []
    try:
        while True:
            joints = np.array(driver.get_positions()[:6])
            cart = joint_angles_to_cartesian(chain, joints)
            poses.append(cart)
            print(f"\r  Recorded {len(poses)} samples | pos={np.round(cart[:3], 4)}", end="")
            time.sleep(1.0 / RECORD_HZ)
    except KeyboardInterrupt:
        print(f"\n\nRecording stopped. {len(poses)} samples collected.")

    # Return arm to idle
    driver.set_arm_modes(trossen_arm.Mode.idle)

    if len(poses) < 20:
        print("Too few samples. Move the arm around more next time.")
        sys.exit(1)

    # Compute deltas between consecutive poses
    poses = np.array(poses)  # (N, 6)
    deltas = np.diff(poses, axis=0)  # (N-1, 6)

    # Add gripper column (0 to 1 range)
    gripper_col = np.zeros((deltas.shape[0], 1))
    deltas_7d = np.concatenate([deltas, gripper_col], axis=1)  # (N-1, 7)

    # Compute q01 and q99
    q01 = np.percentile(deltas_7d, 1, axis=0).tolist()
    q99 = np.percentile(deltas_7d, 99, axis=0).tolist()

    # Gripper stats: fixed 0 to 1
    q01[6] = 0.0
    q99[6] = 1.0

    stats = {
        "trossen_wxai_custom": {
            "action": {
                "q01": q01,
                "q99": q99,
                "mask": [True, True, True, True, True, True, True],
            }
        }
    }

    out_path = Path(__file__).parent / "custom_norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {out_path}")
    print(f"  q01: {[round(v, 6) for v in q01]}")
    print(f"  q99: {[round(v, 6) for v in q99]}")
    print(f"\nUse with: --unnorm-key trossen_wxai_custom")
