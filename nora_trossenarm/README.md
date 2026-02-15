# NORA + Trossen Arm Integration

## Overview

This folder contains scripts for integrating the NORA 1.5 vision-language-action model with the Trossen WidowX AI arm.

### Files

- **`camera_detection.py`** — Simple test to verify the USB camera is properly connected. A USB-A to USB-C adapter is required for connection.
- **`nora_camera_inference.py`** — Initial prototype for running NORA inference on live camera input. Uses YOLO marker detection for real-time visualization and NORA 1.5 for action prediction.

### Current Status

- NORA generates 7-DOF action predictions in Cartesian (end-effector) coordinates.
- **Next step:** Convert Cartesian coordinates to joint angles for Trossen arm execution via inverse kinematics.

## Setup

1. Create a conda environment with Python 3.10:
   ```bash
   conda create -n nora_robot_arm python=3.10 -y
   conda activate nora_robot_arm
   ```

2. Install dependencies:
   ```bash
   pip install -r nora_trossenarm/requirements.txt
   ```
   This includes all packages for YOLO, NORA, and Trossen arm control — no need to install their requirements separately.

3. Connect the Trossen arm (ethernet via USB adapter at `192.168.2.2`) and USB camera to your computer.

## Usage

```bash
python nora_trossenarm/nora_camera_inference.py --camera 0 --instruction "<your instruction>"
```

- On the first run, the NORA 1.5 model (~7-8 GB) will be downloaded from HuggingFace. This may take a while.
- Once loaded, a camera window will open with live YOLO marker detection.
- Click on the camera window, then press **s** to run NORA inference on the current frame.
- Press **q** to quit.

## Output

NORA outputs normalized 7-DOF actions in Cartesian space: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.
