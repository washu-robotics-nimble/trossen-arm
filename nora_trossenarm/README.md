# NORA + Trossen Arm Integration

## Overview

This folder contains scripts for integrating the NORA 1.5 vision-language-action model with the Trossen WXAI V0 arm. NORA predicts 7-DOF Cartesian actions from camera images and language instructions, which are sent directly to the arm via the Trossen SDK's Cartesian position API.

### Files

- **`nora_camera_inference.py`** — Main script. Runs NORA inference on live camera input, unnormalizes the predicted actions, and executes them on the Trossen arm using `set_cartesian_positions` + `set_gripper_position`. Supports a `--dry-run` mode for testing without the robot.
- **`camera_detection.py`** — Simple test to verify the USB camera is properly connected. A USB-A to USB-C adapter is required for connection.
- **`demo_template.py`** — Reference template showing the general Trossen arm driver API (modes, commands, feedback).
- **`src/cartesian_position.py`** — Trossen SDK example for moving the end effector in Cartesian space. Used as reference for the integration.
- **`src/gripper_torque.py`** — Trossen SDK example for opening/closing the gripper with effort (force) control.
- **`TECH_INFO.md`** — Additional technical notes.

## Important: SDK Version Requirement

The Cartesian position API (`set_cartesian_positions`, `get_cartesian_positions`) requires **trossen-arm SDK v1.9.0 or later**. Earlier versions (e.g., v1.7.8) only support joint-space control.

Check your version:
```bash
pip show trossen-arm
```

Upgrade if needed:
```bash
pip install trossen-arm --upgrade
```

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

3. Upgrade the Trossen arm SDK to v1.9.0+:
   ```bash
   pip install trossen-arm --upgrade
   ```

4. Connect the Trossen arm (ethernet via USB adapter at `192.168.1.2`) and USB camera to your computer.

## Usage

Dry run (no robot connected, just prints predicted actions):
```bash
python nora_trossenarm/nora_camera_inference.py --dry-run --instruction "pick up the marker"
```

With robot:
```bash
python nora_trossenarm/nora_camera_inference.py --robot-ip 192.168.1.2 --unnorm-key bridge_orig --instruction "pick up the marker"
```

- On the first run, the NORA 1.5 model (~7-8 GB) will be downloaded from HuggingFace. This may take a while.
- Once loaded, a camera window will open with live YOLO marker detection.
- Click on the camera window, then press **s** to run NORA inference and execute the action chunk on the robot.
- Press **q** to quit.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--robot-ip` | `192.168.1.2` | Trossen arm IP address |
| `--unnorm-key` | `bridge_orig` | Dataset key for action unnormalization (see below) |
| `--goal-time` | `2.0` | Seconds per action step |
| `--dry-run` | off | Run without connecting to the robot |
| `--instruction` | `pick up the object` | Language instruction for NORA |
| `--num-steps` | `10` | Flow matching denoising steps |

## Things to Be Aware Of

### Action Unnormalization

NORA outputs normalized actions in the range [-1, 1]. These must be mapped back to real-world Cartesian coordinates using dataset statistics. The `--unnorm-key` argument selects which dataset's statistics to use. Available keys include: `bridge_orig`, `fractal20220817_data`, `bc_z`, `kuka`, and others from the Open X-Embodiment collection.

**The choice of `--unnorm-key` directly affects the scale and range of the output coordinates.** If the arm moves too far/little or to unexpected positions, try a different key. `bridge_orig` (Bridge V2 dataset) is a reasonable starting point for tabletop manipulation.

### Action Format

NORA outputs `(1, chunk_length, 7)` where the 7 dimensions are:

| Index | Meaning | Unit |
|---|---|---|
| 0 | x position | meters |
| 1 | y position | meters |
| 2 | z position | meters |
| 3 | rx rotation | radians |
| 4 | ry rotation | radians |
| 5 | rz rotation | radians |
| 6 | gripper openness | continuous (position in meters) |

Indices 0-5 are sent to `set_cartesian_positions()` and index 6 is sent to `set_gripper_position()`.
