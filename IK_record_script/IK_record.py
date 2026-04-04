"""
Kinesthetic Teaching — Record keyframes by physically moving the arm.

The arm is set to compliant mode (external_effort) so you can move it by hand.
Press keys to capture joint positions as keyframes, then save for replay.

Controls (type in terminal):
    r - Set task name
    k - Capture current joint positions as a keyframe
    p - Print all captured keyframes
    s - Save keyframes to file
    clear - Clear all keyframes
    g - Toggle gripper open/close (records gripper state with keyframe)
    e - EMERGENCY STOP (hold arm at current position)
    comp - Return to compliant mode after e-stop
    Ctrl+C - Emergency stop during replay or exit
    q - Quit

Usage:
    python teach_keyframes.py
    python teach_keyframes.py --ip 192.168.2.2
"""

import argparse
import os
import threading
import time
from datetime import datetime

import numpy as np
import trossen_arm
from zero import zero_position, reset_gripper


DEFAULT_IP = "192.168.2.2"
OUTPUT_FOLDER = "IK_RESULTS"
REPLAY_DURATION = 20.0  # seconds per move during replay
GRIPPER_OPEN_EFFORT = 30.0 # gripper effort in N 
GRIPPER_CLOSE_EFFORT = -30.0 # gripper effort in N
EFFORT_DURATION = 5.0 # gripper duration in s 
GRIPPER_OPEN_POS = 0.04 # gripper position in m 
GRIPPER_CLOSE_POS = 0.0 # gripper position in m 


class KeyframeTeacher:
    """Records joint-space keyframes from a compliant arm."""

    def __init__(self, ip=DEFAULT_IP):
        self.ip = ip
        self.driver = None
        self.keyframes = []  # list of (joint_angles, gripper_open)
        self.task_name = None 
        self.gripper_open = False # start with close gripper
        self.output_folder = OUTPUT_FOLDER
        os.makedirs(self.output_folder, exist_ok=True)

    def connect(self):
        """Initialize and configure the arm driver."""
        print("Initializing driver...")
        self.driver = trossen_arm.TrossenArmDriver()
        print(f"Connecting to arm at {self.ip}...")
        self.driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_base,
            self.ip,
            False,
        )
        print("Connected.")

    def set_compliant(self):
        """Set arm to compliant mode so it can be moved by hand."""
        self.driver.set_arm_modes(trossen_arm.Mode.external_effort)
        # Send zero effort to all arm joints so they don't drift
        self.driver.set_arm_external_efforts([0.0] * 6, 0.5, False)
        # Keep gripper in position mode so it holds in place
        self.driver.set_gripper_mode(trossen_arm.Mode.position)
        print("Arm is now compliant — move it by hand.")

    def capture(self):
        """Capture current joint positions as a keyframe."""
        joints = np.array(self.driver.get_positions()[:6])
        self.keyframes.append((joints.copy(), self.gripper_open))
        n = len(self.keyframes)
        gripper_str = "open" if self.gripper_open else "closed"
        print(f"[Keyframe {n}] joints={np.array2string(joints, precision=4)} gripper={gripper_str}")

    def toggle_gripper(self):
        """Toggle gripper state."""
        self.gripper_open = not self.gripper_open
        effort = GRIPPER_OPEN_EFFORT if self.gripper_open else GRIPPER_CLOSE_EFFORT
        self.driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        self.driver.set_gripper_external_effort(effort, EFFORT_DURATION, True)
        state = "OPEN" if self.gripper_open else "CLOSED"
        print(f"Gripper: {state}")

    def clear(self):
        """Clear all keyframes."""
        self.keyframes.clear()
        print("Keyframes cleared.")

    def set_task_name(self):
        """Prompt for task name."""
        name = input("Enter task name: ").strip()
        if name:
            self.task_name = name
            print(f"Task name: '{self.task_name}'")
        else:
            print("Empty name, not changed.")

    def print_keyframes(self):
        """Print all keyframes in copy-pasteable format."""
        if not self.keyframes:
            print("No keyframes captured.")
            return
        print(f"\n--- {len(self.keyframes)} keyframes ---")
        print("keyframes = [")
        for joints, grip in self.keyframes:
            grip_str = "True" if grip else "False"
            print(f"    (np.array([{', '.join(f'{j:.4f}' for j in joints)}]), {grip_str}),")
        print("]")
        print("---\n")

    def save(self):
        """Save keyframes to .npy file."""
        if not self.keyframes:
            print("No keyframes to save.")
            return
        if self.task_name is None:
            print("No task name set. Press 'r' to set one first.")
            return
        
        reset_gripper(self.driver)
        zero_position(self.driver)
        filepath = os.path.join(self.output_folder, f"{self.task_name}.npy")
        joints_list = [kf[0] for kf in self.keyframes]
        gripper_list = [kf[1] for kf in self.keyframes]
        data = {
            "task_name": self.task_name,
            "joint_pos": np.array(joints_list),
            "gripper_pos": np.array(gripper_list),
        }
        np.save(filepath, data)
        print(f"Saved {len(self.keyframes)} keyframes to {filepath}")
        self.print_keyframes()

    def emergency_stop(self):
        """Immediately hold the arm at its current position."""
        print("\n!!! EMERGENCY STOP !!!")
        self.driver.set_arm_modes(trossen_arm.Mode.position)
        current_joints = np.array(self.driver.get_positions()[:6])
        self.driver.set_arm_positions(current_joints, 0.0, False)
        print("Arm held at current position.")
        print("RETURNING TO ZERO POSITION NOW")

        zero_position(self.driver)

def main():
    parser = argparse.ArgumentParser(description="Kinesthetic teaching for Trossen arm")
    parser.add_argument("--ip", default=DEFAULT_IP, help="Arm IP address")
    args = parser.parse_args()

    teacher = KeyframeTeacher(ip=args.ip)
    teacher.connect()
    print("reseting gripper and zeroing position. ")
    reset_gripper(teacher.driver)
    zero_position(teacher.driver)
    teacher.set_compliant()

    print("\nControls:")
    print("  r     - Set task name")
    print("  k     - Capture keyframe")
    print("  g     - Toggle gripper open/close")
    print("  p     - Print keyframes")
    print("  s     - Save keyframes to file")
    print("  clear     - Clear keyframes")
    print("  e     - EMERGENCY STOP (hold position)")
    print("  comp  - Return to compliant mode")
    print("  q     - Quit")
    print("  Ctrl+C during replay = emergency stop\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "r":
            teacher.set_task_name()
        elif cmd == "k":
            teacher.capture()
        elif cmd == "g":
            teacher.toggle_gripper()
        elif cmd == "p":
            teacher.print_keyframes()
        elif cmd == "s":
            teacher.save()
        elif cmd == "clear":
            teacher.clear()
        elif cmd == "e":
            teacher.emergency_stop()
        elif cmd == "comp":
            teacher.set_compliant()
        elif cmd == "q":
            break
        elif cmd:
            print(f"Unknown command: '{cmd}'")

    # Zero all joints before exiting
    print("Returning arm to zero position... (Ctrl+C to emergency stop)")
    try:
        teacher.driver.set_arm_modes(trossen_arm.Mode.position)
        reset_gripper(teacher.driver)
        zero_position(teacher.driver)

    except KeyboardInterrupt:
        teacher.emergency_stop()
    print("Done.")


if __name__ == "__main__":
    main()
