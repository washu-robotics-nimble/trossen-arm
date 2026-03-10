import numpy as np
import trossen_arm

robot_ip = "192.168.2.2"

def reset_arm(driver):
    """Reset the arm to rest position using an existing driver."""
    current_positions = driver.get_positions()
    print(f"Current joint positions: {np.round(current_positions[:6], 4)}")

    print("Opening gripper...")
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    driver.set_gripper_external_effort(20.0, 5.0, True)

    print("Moving arm to rest position...")
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_arm_positions(np.zeros(6).tolist(), 3.0, True)

    print("Closing gripper...")
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    driver.set_gripper_external_effort(-20.0, 5.0, True)

    print("Reset done.")
