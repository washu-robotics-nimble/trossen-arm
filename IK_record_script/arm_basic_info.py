import numpy as np

"""some basic info about wxai"""

# number of arm joints (exclude gripper)
NUM_ARM_JOINTS_WXAI = 6

# End effector offset from link_6 in meters [x, y, z]
EE_OFFSET = np.array([0.1055, 0.0, 0.0])

# Gripper position limits in meters
GRIPPER_OPEN_POSITION = 0.044 
GRIPPER_CLOSE_POSITION = 0.0 
