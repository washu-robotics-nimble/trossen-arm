
import numpy as np
import trossen_arm

if __name__=='__main__':
    print("Initializing the drivers...")
    driver = trossen_arm.TrossenArmDriver()

    print("Configuring the drivers...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        "192.168.2.2",
        False
    ) 
    print("Opening the gripper...")
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    driver.set_gripper_external_effort(20.0, 5.0, True)

    print("Moving the arm...")
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_arm_positions(
        np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0]),
        5.0,
        True
    )

    input("Place the marker in the gripper and press Enter...")

    print("Closing the gripper...")
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    driver.set_gripper_external_effort(-20.0, 5.0, True)

    print("Moving the arm again...")
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_arm_positions(
        np.zeros(driver.get_num_joints() - 1),
        5.0,
        True
    )

