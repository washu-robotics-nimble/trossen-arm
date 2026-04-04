import numpy as np
import trossen_arm 
from zero import zero_position, reset_gripper  
import sys, os 

# read recorded file 
record_folder = f"IK_RESULTS"
task_name = "marker"
record_file = f"{record_folder}/{task_name}.npy"

data = np.load(record_file, allow_pickle=True).item()
task_name = data['task_name']
joint_pos = data['joint_pos']
gripper_pos = data['gripper_pos']
print(f"task name: {task_name}")
print(f"{len(joint_pos)} frames are recorded") 

# configure arm driver 
driver = trossen_arm.TrossenArmDriver()
driver.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_base,
    "192.168.2.2",
    False,
)

# zero position driver first just in case 
reset_gripper(driver)
zero_position(driver)

# rerun arm with same keyframes
REPLAY_DURATION = 5.0 # in second 
EFFORT_DURATION = 5.0 # in second 

driver.set_arm_modes(trossen_arm.Mode.position)
try:
    for i in range(len(joint_pos)):
        joints = joint_pos[i]
        grip = gripper_pos[i]
        print(f"="*30)
        print(f"  Moving to keyframe {i + 1}/{len(joint_pos)}...")
        print(f"joint pos: {joint_pos[i]}")
        print(f"gripper open status: {gripper_pos[i]}")
        driver.set_arm_positions(joints, REPLAY_DURATION, True)
        effort = 30.0 if grip else -20.0
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(effort, EFFORT_DURATION, True)
except KeyboardInterrupt:
    print("\n!!! EMERGENCY STOP !!!")
    driver.set_arm_modes(trossen_arm.Mode.position)
    current = np.array(driver.get_positions()[:6])
    driver.set_arm_positions(current, 0.0, False)

print("Replay complete.")
print("Returning to zero position. ")
zero_position(driver)


