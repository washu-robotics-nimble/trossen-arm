"""
Full Loop: IK-guided pick & place with manual adjustment option.

Flow per waypoint:
    1. User enters target location [x, y, z] + gripper status
    2. Safety checks (table, reach, joint limits)
    3. IK computes joint angles
    4. Arm moves to IK result
    5. User confirms or switches to manual adjust
    6. Joint angles recorded as keyframe
    7. Repeat until task complete
    8. Save all keyframes for replay
"""

import sys
import os
import numpy as np
import trossen_arm
import ikpy.chain

# Import local config.
from config import *

# Import all needed functions from util script. 
from util import (
    zero_position, reset_gripper, motion_ik, arm_rom_check,
    joint_limits_check, arm_move_to_position, gripper_motion,
    plot_arm_motion_safety_check,
)


def build_chain():
    """Build ikpy chain from URDF."""
    return ikpy.chain.Chain.from_urdf_file(
        WXAI_URDF_PATH,
        base_elements=["base_link"],
        last_link_vector=EE_LAST_LINK_VECTOR,
        active_links_mask=[False, True, True, True, True, True, True,
                           False, False, False],
        name="wxai",
    )


def capture_current(driver, gripper_open, target_location):
    """Read current joint angles and return as a keyframe."""
    joints = np.array(driver.get_positions()[:6])
    print(f"[Captured] target={target_location} gripper={'open' if gripper_open else 'closed'}")
    return (joints.copy(), gripper_open, target_location)


def set_compliant(driver):
    """Set arm to compliant mode for manual adjustment."""
    driver.set_arm_modes(trossen_arm.Mode.external_effort)
    driver.set_arm_external_efforts([0.0] * 6, 0.5, False) # external_efforts, goal_time, blocking
    driver.set_gripper_mode(trossen_arm.Mode.position)
    print("Arm is now compliant — adjust by hand.")


def save_keyframes(keyframes, task_name):
    """Save recorded keyframes to .npy file."""
    if not keyframes:
        print("No keyframes to save.")
        return
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filepath = os.path.join(OUTPUT_FOLDER, f"{task_name}.npy")
    joints_list = [kf[0] for kf in keyframes]
    gripper_list = [kf[1] for kf in keyframes]
    target_list = [kf[2] for kf in keyframes]
    data = {
        "task_name": task_name,
        "joint_pos": np.array(joints_list),
        "gripper_pos": np.array(gripper_list),
        "target_pos": np.array(target_list),
    }
    np.save(filepath, data)
    print(f"Saved {len(keyframes)} keyframes to {filepath}")


def main():
    # Connect to arm
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        DEFAULT_IP,
        False,
    )

    chain = build_chain()

    print("Resetting arm...")
    zero_position(driver)
    reset_gripper(driver)

    # Task setup
    task_name = input("Enter task name: ").strip()
    if not task_name:
        task_name = "untitled"
    print(f"Task: '{task_name}'")

    keyframes = []
    gripper_open = False  # start closed
    waypoint_num = 0

    print("\n=== Full Loop: IK + Manual Adjust ===")
    print("Each waypoint: enter target [x,y,z,gripper] or command")
    print("  p        — print recorded keyframes")
    print("  e        — emergency stop")
    print("  Example: 0.18, 0.10, 0.06, 1")
    print("  (move EE to [0.18, 0.10, 0.06], open gripper)\n")
    print("  the location should be incremental, not final location!")

    while True:
        waypoint_num += 1
        print(f"\n--- Waypoint {waypoint_num} ---")

        # input target location 
        try:
            user_input = input("target [x,y,z,gripper]: ").strip()
        except (EOFError, KeyboardInterrupt): # for ctl+c, ctl+z, ctl+d
            print("\nInterrupted at input prompt.")
            # exit confirmation 
            confirm = input("Exit task? (y/n): ").strip().lower()
            if confirm == 'y':
                break
            waypoint_num -= 1
            continue

        # Commands
        # print out recorded keyframes 
        if user_input.lower() == 'p':
            for i, (j, g, t) in enumerate(keyframes):
                gs = "open" if g else "closed"
                print(f"  {i+1}: target={t} gripper={gs}")
            waypoint_num -= 1 # since nothing recorded, so -1 
            continue
        
        # anything emergent happened,
        # the arm stop at current position and then return to zero position
        if user_input.lower() == 'e':
            print("\n!!! EMERGENCY STOP !!!")
            driver.set_arm_modes(trossen_arm.Mode.position)
            current = np.array(driver.get_positions()[:6]) # curr position
            driver.set_arm_positions(current, 0.0, False) # stop at curr position
            zero_position(driver) # arm back to initial position
            resume = input(f"Continue task '{task_name}'? (y/n): ").strip().lower()
            if resume == 'y':
                # replay saved keyframes to restore arm state
                if len(keyframes)>0:
                    print(f"Replaying {len(keyframes)} saved keyframes...")
                    for i, (kf_joints, kf_grip, _) in enumerate(keyframes):
                        print(f"  Replaying waypoint {i+1}/{len(keyframes)}...")
                        arm_move_to_position(driver, kf_joints, DT)
                        gripper_motion(driver, 1 if kf_grip else 0)
                    print("Replay complete. Resuming task.")
                else: 
                    print(f"no previous keyframes recorded. ")

                waypoint_num -= 1
                continue
            # if not resume on the current task, save and exit. 
            else:
                # end task, go to finalize
                break
        
        # if input is empty 
        if not user_input: 
            waypoint_num -= 1
            continue

        # Parse input
        try:
            inputs = [float(x) for x in user_input.replace(",", " ").split()]
        except ValueError:
            print("Invalid input.")
            waypoint_num -= 1
            continue
        
        # check if input is valid. 
        if len(inputs) != 4: 
            print(f"Need 3 coords + 1 gripper status, got {len(inputs)}.")
            waypoint_num -= 1
            continue
        # separate input info into joints and gripper 
        target_location = inputs[0:3]
        gripper_status = int(inputs[-1])
        # set gripper open if gripper open == gripper status == 1, else close. 
        gripper_open = gripper_status == 1

        # --- Same target check ---
        if keyframes:
            last_target = keyframes[-1][2]
            if list(last_target) == list(target_location):
                print("Same target as previous — gripper only.")
                gripper_motion(driver, gripper_status)
                kf = capture_current(driver, gripper_open, target_location)
                keyframes.append(kf)
                print(f"Keyframe {len(keyframes)} recorded.")
                done = input("Add another waypoint? (y/n): ").strip().lower()
                if done == 'n':
                    break
                continue

        # --- Safety checks ---
        if target_location[2] <= 0: # target location in [x,y,z]
            print("REJECTED — target z is below table.")
            waypoint_num -= 1
            continue
        
        # check if target location is outof arm span. 
        if not arm_rom_check(target_location):
            print("REJECTED — target out of reach.")
            waypoint_num -= 1
            continue

        # --- IK ---
        print("Computing IK...")
        ik_joint_angles = motion_ik(chain, driver, target_location,
                                     target_orientation=[0, 0, -1], # gripper point down for pick and place
                                     orientation_mode="X")
        # Preserve wrist angle from previous manual adjustment, otherwise 0
        curr_wrist = driver.get_positions()[5]
        ik_joint_angles[-1] = curr_wrist if abs(curr_wrist) > 1e-3 else 0.0
        # check if ik infered joint angles exceed trossen arm's joint limits
        if not joint_limits_check(ik_joint_angles):
            print("REJECTED — IK result exceeds joint limits.")
            waypoint_num -= 1
            continue

        # --- Path safety with visualization ---
        # check if ee points below baseline through the ik infered path
        if not plot_arm_motion_safety_check(chain, driver, ik_joint_angles, baseline=0.0):
            print("REJECTED — path goes below table.")
            waypoint_num -= 1
            continue

        # --- Execute ---
        # once all safety check passed, execute real arm motion. 
        print("Moving arm...")
        try:
            arm_move_to_position(driver, ik_joint_angles, DT)
            gripper_motion(driver, gripper_status)
            print("Arm at target position.")
        except KeyboardInterrupt:
            print("\n!!! Interrupted !!!")
            print("\n return to zero position... ")
            zero_position(driver)
            # Replay all saved keyframes to restore arm state
            if keyframes:
                print(f"Replaying {len(keyframes)} saved keyframes...")
                for i, (kf_joints, kf_grip, _) in enumerate(keyframes):
                    print(f"  Replaying waypoint {i+1}/{len(keyframes)}...")
                    arm_move_to_position(driver, kf_joints, DT)
                    gripper_motion(driver, 1 if kf_grip else 0)
                print("Replay complete.")
            waypoint_num -= 1
            continue

        # --- Confirm or adjust ---
        # for each execute motion inferred from IK, 
        # user can choose to record, adjust, or redo with reset target position. 
        while True:
            response = input("Satisfied? (y=record / n=manual adjust / r=redo): ").strip().lower()

            # yes for record
            if response == 'y':
                kf = capture_current(driver, gripper_open, target_location)
                keyframes.append(kf)
                print(f"Keyframe {len(keyframes)} recorded.")
                break

            # redo: zero, replay saved keyframes, re-prompt for new target
            elif response == 'r':
                print("Redo — returning to zero...")
                zero_position(driver)
                reset_gripper(driver)
                if keyframes:
                    print(f"Replaying {len(keyframes)} saved keyframes...")
                    for i, (kf_joints, kf_grip, _) in enumerate(keyframes):
                        print(f"  Replaying waypoint {i+1}/{len(keyframes)}...")
                        arm_move_to_position(driver, kf_joints, DT)
                        gripper_motion(driver, 1 if kf_grip else 0)
                    print("Replay complete.")
                waypoint_num -= 1
                break

            # no for adjust by setting arm in compliant mode.
            elif response == 'n':
                set_compliant(driver)
                print("Adjust the arm by hand. Commands available are: ")
                print("  g — toggle gripper")
                print("  k — capture current position")

                while True:
                    adj_cmd = input("  adjust> ").strip().lower()
                    # after adjustment, press k for store current keyframe
                    if adj_cmd == 'k':
                        kf = capture_current(driver, gripper_open, target_location)
                        keyframes.append(kf)
                        print(f"Keyframe {len(keyframes)} recorded.")
                        break
                    # set gripper status open or close 
                    elif adj_cmd == 'g':
                        gripper_open = not gripper_open
                        effort = GRIPPER_OPEN_EFFORT if gripper_open else GRIPPER_CLOSE_EFFORT
                        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
                        driver.set_gripper_external_effort(effort, EFFORT_DURATION, True)
                        print(f"Gripper: {'OPEN' if gripper_open else 'CLOSED'}")
                break

        # --- Next or finish? ---
        done = input("Add another waypoint? (y/n): ").strip().lower()
        if done == 'n':
            break

    # --- Finish ---
    print("\nFinishing up...")
    # reset arm
    print("arm return to zero position")
    zero_position(driver) # arm return to zero position
    print("resetting gripper")
    reset_gripper(driver) # reset gripper

    # if there are recorded key frames, save it in npy file for replay 
    if len(keyframes)>0:
        save_keyframes(keyframes, task_name)
    else:
        print("No keyframes recorded.")

    print("Done.")


if __name__ == "__main__":
    main()
