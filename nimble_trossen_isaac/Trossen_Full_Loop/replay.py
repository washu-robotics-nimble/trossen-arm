"""
Replay saved keyframes on the arm.

Usage:
    python replay.py                     # lists available recordings
    python replay.py <task_name>         # replays the specified recording
"""

import sys
import os
import numpy as np
import trossen_arm

from config import *

from util import (
    zero_position, reset_gripper, arm_move_to_position, gripper_motion,
)

############### FUNCTIONS FOR FILE LOADING AND MOTION REPLAY ############
def list_recordings():
    """List all saved recordings."""
    if not os.path.exists(OUTPUT_FOLDER):
        print("No recordings folder found.")
        return []
    files = [f.replace('.npy', '') for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.npy')]
    if not files:
        print("No recordings found.")
    else:
        print(f"Available recordings ({len(files)}):")
        for f in files:
            print(f"  {f}")
    return files

def load_keyframes(task_name):
    """Load keyframes from .npy file."""
    filepath = os.path.join(OUTPUT_FOLDER, f"{task_name}.npy")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    data = np.load(filepath, allow_pickle=True).item()
    print(f"Loaded task: {data['task_name']}")
    print(f"  {len(data['joint_pos'])} keyframes")
    return data

def replay(driver, data):
    """Replay keyframes on the arm."""
    joint_pos = data['joint_pos']
    gripper_pos = data['gripper_pos']
    target_pos = data.get('target_pos', None)
    num_frames = len(joint_pos)

    print(f"\nReplaying {num_frames} keyframes... (Ctrl+C to stop)")
    prev_grip = None
    try:
        for i in range(num_frames):
            joints = joint_pos[i]
            grip = gripper_pos[i]
            target = target_pos[i] if target_pos is not None else "N/A"
            print(f"  Waypoint {i+1}/{num_frames}: target={target} gripper={'open' if grip else 'closed'}")
            arm_move_to_position(driver, joints, DT) 

            # adjust gripper status by comparing two adjacent frames status
            if grip != prev_grip:
                gripper_motion(driver, 1 if grip else 0)
                prev_grip = grip

    # Ctrl+C during replay — stop safely and return arm to zero    
    except KeyboardInterrupt:
        print("\n!!! Replay interrupted !!!")
        zero_position(driver)
        return

    print("Replay complete.")

###################### REPLAY MAIN #########################
def main():
    # List or load recording
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        files = list_recordings()
        if not files:
            return
        # if multiple files found, choose one from the list 
        task_name = input("\nEnter task name to replay: ").strip()
        if not task_name:
            print("No task name entered.")
            return
    # load npy file 
    data = load_keyframes(task_name)
    if data is None:
        return

    # Connect to arm
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        DEFAULT_IP,
        False,
    )

    # before replay, reset arm adn gripper for safety 
    print("Resetting arm...")
    zero_position(driver)
    reset_gripper(driver)

    # Replay loop based on the saved trajectory in npy 
    while True:
        replay(driver, data)

        # check if replay again is needed 
        again = input("\nReplay again? (y/n): ").strip().lower()
        if again != 'y':
            break

        # Reset arm before next replay iteration.
        print("Resetting arm for replay...")
        reset_gripper(driver)
        zero_position(driver)

    # Finalize
    # reset arm and return to zero position. 
    print("Resetting arm...")
    zero_position(driver)
    reset_gripper(driver)
    print("Done.")


if __name__ == "__main__":
    main()
