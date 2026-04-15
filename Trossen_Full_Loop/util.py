from __future__ import annotations
import numpy as np
import trossen_arm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import (
    GRIPPER_OPEN_EFFORT, GRIPPER_CLOSE_EFFORT,
    EFFORT_DURATION, HOLD_EFFORT_DURATION,
    GRIPPER_OPEN_POS, GRIPPER_CLOSE_POS,
)

################## RESET FUNCTIONS FOR BOTH ARM AND GRIPPER #######################
# zero arm 
def zero_position(driver):
    """Return arm to zero position and close gripper."""
    arm_pos = driver.get_positions()  # rad for joint and m for gripper
    # print(f"cur arm positions are: {arm_pos}")
    # print(f"total dof {len(arm_pos)}")

    zero_pos = np.zeros(driver.get_num_joints() - 1)  # only for joints
    if np.any(arm_pos[0:6] - zero_pos >= np.ones_like(zero_pos) * 1e-3):
        driver.set_arm_modes(trossen_arm.Mode.position)
        driver.set_arm_positions(
            np.zeros(driver.get_num_joints() - 1),
            5.0,
            True,
        )
        print("arm returns to zero position")
    else:
        print("arm is at zero position")

    zero_grip = 0.0
    if np.any(arm_pos[-1] - zero_grip >= np.ones_like(zero_grip) * 3e-3):
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(GRIPPER_CLOSE_EFFORT, EFFORT_DURATION, True)
        print("gripper closing.")
    else:
        print("gripper is closed. ")

    print("The arm returns to zero position.")

# zero griper 
def reset_gripper(driver):
    "reset gripper by open first and then close. "
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(GRIPPER_OPEN_POS, EFFORT_DURATION, True)
    driver.set_gripper_position(GRIPPER_CLOSE_POS, EFFORT_DURATION, True)
    print("gripper resets succeed. ")

###################### SAFETY CHECKS TO PROTECT ARM ###########################
# Gripper should not move underneath baseline based on the current arm setup.  
def ee_safety_check(chain, driver, angles, baseline,
                     num_steps=30) -> bool:
    """
    check if with input angle, will ee pointing into table.

    forward_kinematic function transform joint angle w.r.t to zero position
    as set in urdf file and thus compute the ee position and ee rotation matrix.

    ee might point into table during the motion but not in the end.
    so the thru motion is interpolated and checked as well.

    Args:
    - chain: ikpy chain.
    - driver: trossen arm driver.
    - angles: target angles in rad.
    - baseline: tabletop height. Normally is 0 for table.
    - num_steps: number of intermediate steps to check along the path.

    Return:
    True if ee's z > baseline at ALL points along path, else False.
    """
    start = driver.get_positions()[:6]
    target = np.array(angles)

    # check all intermediate steps + final position
    for step in range(1, num_steps + 1):
        t = step / num_steps
        interp = start + t * (target - start)
        full = np.zeros(10)
        full[1:7] = interp
        fk = chain.forward_kinematics(full)
        ee_pos = fk[:3, 3]
        if ee_pos[2] < baseline:
            print(f"UNSAFE at step {step}/{num_steps}: EE position {ee_pos}")
            return False

    print(f"Path safe. Final EE position: {ee_pos}")
    return True

# Elevate ee_safety_check to include plot to visualize arm motion from IK inference.
# This visualization is particularly useful when setting trajectory without gui.
# NOTE: requires a display (matplotlib 3D window) — will not work headless / over SSH
# without X forwarding. For headless runs, use ee_safety_check instead.
def plot_arm_motion_safety_check(chain, driver, angles, baseline,
                     num_steps=30) -> bool:
    """
    Visualize the input angle motion with fk and run safety check.
    Returns True if path is safe, False otherwise.
    All angles must be in radians.

    alternate function is ee_safety_check which doesn't have
    visualization.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    start = driver.get_positions()[:6]
    target = np.array(angles)
    path_safe = [True]  # mutable so update() can modify it

    def update(frame):
        ax.clear()
        t = frame / num_steps
        interp = start + t * (target - start)
        full = np.zeros(10)
        full[1:7] = interp
        chain.plot(full, ax)
        fk = chain.forward_kinematics(full)
        ee_pos = fk[:3, 3]
        safe = ee_pos[2] > baseline
        if not safe:
            path_safe[0] = False
        color = "green" if safe else "red"
        status = "SAFE" if safe else "UNSAFE"
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(0, 0.8)
        ax.set_title(f"Step {frame}/{num_steps} | EE z={ee_pos[2]:.4f}m | {status}", color=color)

    ani = FuncAnimation(fig, update, frames=num_steps + 1, interval=100, repeat=False)
    plt.show()
    return path_safe[0]

# target location should not exceed arm span 
ARM_MAX_REACH = 0.264 + 0.245 + 0.06775 + 0.02895 + 0.156062  # ~0.762m from URDF link lengths

def arm_rom_check(target_location) -> bool:
    """
    this function checks if target_location is beyond
    arm's range of motion (total span of arm)

    args:
    - target_location: (x,y,z) in m

    return:
    - bool: False if target > rom, True if target < rom.
    """
    dist = np.linalg.norm(target_location)
    if dist > ARM_MAX_REACH:
        print(f"Target distance {dist:.4f}m exceeds arm reach {ARM_MAX_REACH:.4f}m")
        return False
    print(f"Target distance {dist:.4f}m within arm reach {ARM_MAX_REACH:.4f}m")
    return True

# IK inferred joint angles should not exceed what's specified in URDF. 
# WXAI joint limits from URDF/spec sheet (in radians)
JOINT_LIMITS = {
    "joint_0": (-3.054, 3.054),
    "joint_1": (0.0, 3.142),
    "joint_2": (0.0, 2.356),
    "joint_3": (-1.571, 1.571),
    "joint_4": (-1.571, 1.571),
    "joint_5": (-3.142, 3.142),
}

def joint_limits_check(joint_angles) -> bool:
    """
    Check if joint angles are within WXAI joint limits.

    args:
    - joint_angles: 6 joint angles in rad (from IK result)

    return:
    - bool: True if all within limits, False if any exceed.
    """
    for i, (name, (low, high)) in enumerate(JOINT_LIMITS.items()):
        angle = joint_angles[i]
        print(f"  {name}: {angle:.4f} rad  [{low:.3f}, {high:.3f}]")
        if angle < low:
            print(f"  !! {name} below lower limit")
            return False
        if angle > high:
            print(f"  !! {name} above upper limit")
            return False
    print("All joints within limits.")
    return True

########################### ARM MOTION BASIC FUNCTIONS #############################
# arm move controlled by joint angles 
def arm_move_to_position(driver, joint_angles, dt) -> None:
    """
    controlled with joint angles in rad. 
    This set arm to absolute position as specified in joint_angles. 
    Not relative position to the previous one. 

    args: 
    - joint_angles: angles of each joint in rad 
    - dt: time duration for this move
    """
    num_joints = driver.get_num_joints() - 1 
    if len(joint_angles) != num_joints:
        print(f"FALSE INPUTS")
        print(f"Expected {num_joints} joint angles, got {len(joint_angles)}.")
        return
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_arm_positions(
        np.array(joint_angles), 
        dt, 
        True
    )

# gripper motion controlled by user setup 
def gripper_motion(driver, status): 
    """
    open or close gripper 

    args: 
    - status: 0 for close and 1 for open 
    """
    # gripper motor activate
    driver.set_gripper_mode(trossen_arm.Mode.external_effort) 

    # if status set to open
    if status == 1:
        # gripper open
        driver.set_gripper_external_effort(GRIPPER_OPEN_EFFORT, EFFORT_DURATION, True)
        print("opening gripper")
        # After opening, hold position
        driver.set_gripper_mode(trossen_arm.Mode.position)
    else:
        # close gripper
        driver.set_gripper_external_effort(GRIPPER_CLOSE_EFFORT, EFFORT_DURATION, True)
        print("closing gripper")
        # If gripper not fully closed (holding object), keep applying effort
        grip_pos = driver.get_positions()[-1]
        if grip_pos < 3e-3: # if fully closed, stop external effort
            driver.set_gripper_mode(trossen_arm.Mode.position)
        else: # if not fully closed, continue external effort
            print(f"Gripper holding object (pos={grip_pos:.4f}m), maintaining effort.")
            driver.set_gripper_external_effort(GRIPPER_CLOSE_EFFORT, HOLD_EFFORT_DURATION, False)

########################### INVERSE KINEMATICS INFERENCE #############################
# Main Function for Inverse Kinematic Inference
def motion_ik(chain, driver, target_location,
              target_orientation=None, orientation_mode=None):
    """
    this function solves joint angles with ik w.r.t target location.

    args:
    - chain: ikpy chain
    - driver: trossen arm driver
    - target_location: [x, y, z] in meters
    - target_orientation: direction vector [x, y, z] for EE axis constraint
        e.g. [0, 0, -1] for gripper pointing down
    - orientation_mode: which EE axis to constrain ("X", "Y", "Z", or None)
        "X" for gripper pointing direction (since ee_gripper offset is along X)

    return: 
    - infered joint angles
    """
    seed = np.zeros(10)
    seed[1:7] = driver.get_positions()[:6] # cur joint angles
    # clamp seed to joint bounds so IK solver doesn't reject it
    for i, link in enumerate(chain.links):
        if hasattr(link, 'bounds') and link.bounds is not None:
            low, high = link.bounds
            seed[i] = np.clip(seed[i], low, high)
    ik = chain.inverse_kinematics(
        target_location,
        initial_position=seed,
        target_orientation=target_orientation,
        orientation_mode=orientation_mode,
    ) # len(ik) = 10
    ik_joint_angles = ik[1:7] # rad
    return ik_joint_angles
