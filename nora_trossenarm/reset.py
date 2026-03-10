"""Reset the arm to its rest position (all joints to zero).

Usage:
    python nora_trossenarm/reset.py
"""

import trossen_arm
from util.hamer import reset_arm, robot_ip

if __name__ == "__main__":
    print("Connecting to arm...")
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        robot_ip,
        False,
    )
    reset_arm(driver)
