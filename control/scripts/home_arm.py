"""
Return the arm to its home position (all joints at zero) safely.

Unlike reset_arm.py, this reads the arm's current pose and scales the move
duration to the distance travelled, so it glides home instead of jerking —
important when the arm is far out at a writing pose or limp in idle mode.

Usage:
  python control/scripts/home_arm.py
  python control/scripts/home_arm.py --speed 0.5   # slower (rad/s)
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import trossen_arm

from control.writing.safety import glide_home, HOME

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.8,
                        help="Approximate joint speed (rad/s); lower is gentler")
    parser.add_argument("--min-time", type=float, default=2.0,
                        help="Minimum move duration (s)")
    args = parser.parse_args()

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print("Initializing the driver...")
    driver = trossen_arm.TrossenArmDriver()

    # configure()'s 4th arg is clear_error — this is the recovery tool, so
    # clear any fault so the arm will accept the home command.
    def _configure(clear_error):
        driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            cfg["robot"]["ip"],
            clear_error,
        )

    _configure(True)

    current = np.array(driver.get_positions()[:6], dtype=np.float64)
    max_delta = float(np.max(np.abs(current - HOME)))

    print(f"Current pose: {np.round(current, 3)}")
    print(f"Largest joint move: {np.degrees(max_delta):.0f} deg.")

    input("Clear the area and press Enter to move the arm home...")

    print("Switching to position control and homing...")
    duration = glide_home(driver, speed=args.speed, min_time=args.min_time,
                          clear_error=lambda: _configure(True))

    print(f"Done. Arm glided home over {duration:.1f} s.")


if __name__ == "__main__":
    main()
