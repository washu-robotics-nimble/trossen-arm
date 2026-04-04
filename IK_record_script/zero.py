import numpy as np
import trossen_arm
import sys, os


def zero_position(driver):
    """Return arm to zero position and close gripper."""
    arm_pos = driver.get_positions()  # rad for joint and m for gripper
    print(f"cur arm positions are: {arm_pos}")
    print(f"total dof {len(arm_pos)}")

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
        driver.set_gripper_external_effort(-20.0, 5.0, True)
        print("gripper closing.")
    else:
        print("gripper is closed. ")

    print("The arm returns to zero position.")

def reset_gripper(driver):
    "reset gripper by open first and then close. "
    open = 0.04 # in m 
    close = 0.0 # in m 
    driver.set_gripper_mode(trossen_arm.Mode.position)
    driver.set_gripper_position(open, 5, True)
    driver.set_gripper_position(close, 5, True)
    print("gripper resets succeed. ")

if __name__ == "__main__":
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        "192.168.2.2",
        False,
    )
    reset_gripper(driver)
    zero_position(driver)
