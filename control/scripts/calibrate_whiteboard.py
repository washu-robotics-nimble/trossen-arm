"""
Whiteboard plane calibration by jogging the marker tip to 3 corners.

The arm stays in POSITION mode and holds itself up — you do NOT backdrive it
(the WXAI's geared joints have too much friction for precise hand-guiding).
Instead you nudge the marker tip to each corner with the keyboard, then press
SPACE to capture:
    corner 1: BOTTOM-LEFT   (as you face the board)
    corner 2: BOTTOM-RIGHT
    corner 3: TOP-LEFT

FK converts the captured joint angles to 3D points and the plane is saved to
config/whiteboard_calibration.json.

The marker tip offset (--tip-length) is the distance from the center of the
gripper fingers to the marker tip, in meters.  Measure it with a ruler.

Usage:
  python control/scripts/calibrate_whiteboard.py --tip-length 0.10
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import trossen_arm

from control.writing.kinematics import forward_kinematics
from control.writing.jog import jog_to_point
from control.writing.safety import home_on_interrupt, glide_home
from control.writing.whiteboard import WhiteboardPlane, CALIBRATION_PATH

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml")

# forward-reaching starting pose (matches draw_a.py / write_text.py READY)
READY = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])

CORNERS = [
    ("BOTTOM-LEFT", "bottom-left corner of the writing area"),
    ("BOTTOM-RIGHT", "bottom-right corner of the writing area"),
    ("TOP-LEFT", "top-left corner of the writing area"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tip-length", type=float, required=True,
                        help="Distance (m) from gripper-finger center to marker tip, e.g. 0.10")
    parser.add_argument("--skip-gripper", action="store_true",
                        help="Skip the marker-gripping prompt (marker already held)")
    args = parser.parse_args()

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print("Initializing the driver...")
    driver = trossen_arm.TrossenArmDriver()

    # configure()'s 4th arg is clear_error — clear any stale fault on startup.
    def _configure(clear_error):
        driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            cfg["robot"]["ip"],
            clear_error,
        )

    _configure(True)

    # From here on, any Ctrl-C or crash glides the arm home instead of
    # leaving it jammed against the board.  clear_error lets a faulted arm
    # be recovered before homing (an errored arm ignores motion commands).
    with home_on_interrupt(driver, clear_error=lambda: _configure(True)):
        if not args.skip_gripper:
            print("\nOpening the gripper...")
            driver.set_gripper_mode(trossen_arm.Mode.external_effort)
            driver.set_gripper_external_effort(20.0, 3.0, True)
            input("\nPlace the marker in the gripper exactly as it will be held while "
                  "writing.\nPress Enter to close the gripper...")
            driver.set_gripper_external_effort(-20.0, 3.0, True)
            time.sleep(0.5)

        input("\nThe arm will now move to a forward-reaching ready pose. "
              "Clear the area and press Enter...")
        driver.set_arm_modes(trossen_arm.Mode.position)
        driver.set_arm_positions(READY, 3.0, True)

        # Hold the tool orientation fixed (marker pointing direction) during jogging.
        _, R_ready = forward_kinematics(READY, args.tip_length)
        target_dir = R_ready[:, 0]  # tool x-axis

        q_seed = READY.copy()
        points = []
        for name, desc in CORNERS:
            print(f"\nJog the MARKER TIP to the {name} ({desc}).")
            q, pos = jog_to_point(driver, q_seed, args.tip_length, target_dir, name)
            if q is None:
                print("\nCalibration aborted — gliding the arm home...")
                glide_home(driver, clear_error=lambda: _configure(True))
                return
            points.append(pos)
            q_seed = q  # start next corner from where this one ended

        plane = WhiteboardPlane.from_corners(points[0], points[1], points[2],
                                             tip_length=args.tip_length)
        plane.save()

        print(f"\nCalibrated writing area: {plane.width*100:.1f} x {plane.height*100:.1f} cm")
        print(f"Board normal (should point back toward the robot): {np.round(plane.normal, 3)}")
        print(f"Saved to {CALIBRATION_PATH}")

        input("\nPress Enter to return the arm to home position...")
        glide_home(driver, clear_error=lambda: _configure(True))
        print("\nDone. Next: python control/scripts/write_text.py \"Hello\" --dry-run")


if __name__ == "__main__":
    main()
