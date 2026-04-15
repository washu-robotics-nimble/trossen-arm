"""
Configuration constants for WXAI arm control — Full Loop.
"""

import os

# Arm connection
# Hardware-specific — change this to match your arm's IP on your network.
DEFAULT_IP = "192.168.2.2"

# Output
# Absolute path so recordings land in Trossen_Full_Loop/IK_RESULTS regardless of cwd.
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IK_RESULTS")

# Motion timing
REPLAY_DURATION = 5.0   # seconds per move
DT = 5.0                # seconds per motion command

# Gripper effort
GRIPPER_OPEN_EFFORT = 30.0   # N
GRIPPER_CLOSE_EFFORT = -30.0 # N
EFFORT_DURATION = 5.0        # s
HOLD_EFFORT_DURATION = 60.0  # s — sustained effort when gripper holds an object

# Gripper position
GRIPPER_OPEN_POS = 0.04  # m
GRIPPER_CLOSE_POS = 0.0  # m

# IK
# Absolute path so scripts can be launched from any working directory.
WXAI_URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "wxai_base.urdf")
EE_LAST_LINK_VECTOR = [0.08, 0, 0]  # ee offset from link_6

# Safety
ARM_MAX_REACH = 0.264 + 0.245 + 0.06775 + 0.02895 + 0.156062  # ~0.762m
