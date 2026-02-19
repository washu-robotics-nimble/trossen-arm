# NORA + Trossen Arm Integration

## Overview

This folder contains scripts for integrating the NORA 1.5 vision-language-action model with the Trossen WXAI V0 arm. NORA predicts 7-DOF Cartesian **delta** actions from camera images and language instructions. Since the Trossen SDK v1.7.x only supports joint-space control, we use **ikpy + URDF** for forward/inverse kinematics to convert between Cartesian deltas and joint angles.

### Hardware

- **Arm**: Trossen WXAI V0 (`trossen_arm` SDK v1.7.8, firmware v1.7.6)
- **Camera**: USB camera mounted on the gripper (eye-in-hand)
- **Connection**: Arm connected via Ethernet at `192.168.2.2`

### Files

| File | Description |
|---|---|
| `nora_camera_inference.py` | Main inference script. Uses NORA 1.5 base model (`declare-lab/nora-1.5`) with YOLO marker detection, FK/IK via ikpy, and gripper effort control. Default unnorm key: `bridge_orig`. |
| `nora_fractal_dpo.py` | Fractal DPO variant. Uses `declare-lab/nora-1.5-fractal-dpo` model optimized for real robot tasks. Default unnorm key: `fractal20220817_data`. Works better than base model for manipulation. |
| `collect_stats.py` | Collect custom normalization stats by moving the arm in gravity-comp mode. Saves q01/q99 percentiles to `custom_norm_stats.json`. |
| `pickup_marker.py` | Manual marker pickup script (bypasses NORA). Opens gripper, moves to position, waits for user, closes gripper. |
| `reset.py` | Reset arm to rest position (all joints to zero). |
| `util/hamer.py` | Shared utilities: `reset_arm()` function and `robot_ip` constant. |
| `src/wxai_base.urdf` | WXAI V0 URDF file for FK/IK (downloaded from TrossenRobotics/trossen_arm_description). |
| `camera_detection.py` | Simple camera test. USB-A to USB-C adapter required. |
| `config_test.py` | Prints controller firmware and SDK driver versions. |
| `src/cartesian_position.py` | Trossen SDK Cartesian control example (requires SDK v1.9.0+, not compatible with our firmware). |
| `src/gripper_torque.py` | Trossen SDK gripper effort control example. |
| `demo_template.py` | Reference template for Trossen arm driver API. |

## SDK / Firmware Compatibility

Our arm runs **firmware v1.7.6**, which is only compatible with **SDK v1.7.x**. The Cartesian position API (`set_cartesian_positions`) requires SDK v1.9.0+ and is **not available** on our setup. Upgrading firmware requires a Teensy Loader + USB cable.

**Workaround**: We use `ikpy` with the WXAI V0 URDF file to compute forward kinematics (joint angles -> Cartesian pose) and inverse kinematics (Cartesian pose -> joint angles) in Python.

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

3. Connect the Trossen arm (Ethernet at `192.168.2.2`) and USB camera.

4. Verify arm connection:
   ```bash
   python nora_trossenarm/config_test.py
   ```

## Usage

### NORA Base Model

```bash
# Dry run (no robot):
python nora_trossenarm/nora_camera_inference.py --dry-run

# With robot:
python nora_trossenarm/nora_camera_inference.py
```

### NORA Fractal DPO (recommended)

```bash
# Dry run:
python nora_trossenarm/nora_fractal_dpo.py --dry-run

# With robot:
python nora_trossenarm/nora_fractal_dpo.py
```

### Controls

| Key | Action |
|---|---|
| `i` | Set or change the language instruction |
| `s` | Sample NORA actions and execute on the robot |
| `r` | Reset arm to rest position (all joints to zero) |
| `q` | Reset arm and quit |

On the first run, the NORA model (~7-8 GB) will be downloaded from HuggingFace.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--robot-ip` | `192.168.2.2` | Trossen arm IP address |
| `--unnorm-key` | varies by script | Dataset key for action unnormalization |
| `--goal-time` | `2.0` | Seconds per action step |
| `--dry-run` | off | Run without connecting to the robot |
| `--instruction` | none | Language instruction (can also set at runtime with `i`) |
| `--num-steps` | `10` | Flow matching denoising steps |
| `--camera` | `0` | Camera device ID |
| `--conf` | `0.25` | YOLO confidence threshold |

### Other Scripts

```bash
# Reset arm to rest position:
python nora_trossenarm/reset.py

# Collect custom normalization stats (gravity-comp mode):
python nora_trossenarm/collect_stats.py

# Manual marker pickup (no NORA):
python nora_trossenarm/pickup_marker.py
```

## How It Works

### Action Pipeline

1. Camera captures a frame
2. YOLO detects marker position, draws a green circle on the image, adds spatial info to the instruction
3. NORA predicts normalized actions `(1, 5, 7)` — 5 timesteps, 7 dimensions
4. Actions are unnormalized using dataset-specific q01/q99 statistics
5. For each timestep:
   - Read current joint angles from the arm
   - FK: joint angles -> current Cartesian pose (via ikpy + URDF)
   - Add NORA's delta to get target Cartesian pose
   - IK: target Cartesian pose -> target joint angles (via ikpy + URDF)
   - Send joint angles to arm via `set_arm_positions()`
   - Control gripper via `set_gripper_external_effort()` (+20N open, -20N close)

### Action Format

NORA outputs `(1, chunk_length, 7)` where the 7 dimensions are **deltas**:

| Index | Meaning | Unit |
|---|---|---|
| 0-2 | dx, dy, dz position delta | meters |
| 3-5 | drx, dry, drz rotation delta | radians |
| 6 | gripper openness | 0.0 (closed) to 1.0 (open) |

### Unnormalization Keys

| Key | Dataset | Camera | Notes |
|---|---|---|---|
| `bridge_orig` | Bridge V2 (WidowX) | Fixed overhead | Default for base model |
| `fractal20220817_data` | Fractal / Google RT-1 | Head-mounted | Default for Fractal DPO |
| `bc_z` | BC-Z (Google) | Wrist camera | Eye-in-hand, closest to our setup |
| `stanford_hydra` | Stanford Hydra | Wrist camera | Eye-in-hand |

### YOLO + NORA Integration

When `s` is pressed, the script:
1. Runs YOLO to detect the marker
2. Draws a green circle on the detection to visually guide NORA
3. Appends spatial location (e.g., "The marker is at the top-left of the image") to the instruction
4. Passes the annotated image + enriched instruction to NORA

## Known Issues / Next Steps

- **Movements are "better but still off"**: Fractal DPO works better than the base model but predictions don't perfectly match the arm's workspace. Likely causes:
  - **Initial pose mismatch**: The arm's rest position (all zeros) puts the camera in a different orientation than the training data. Try starting from a "manipulation ready" pose where the camera looks down at the workspace.
  - **Coordinate frame mismatch**: NORA's training data may use a different axis convention than the Trossen arm's URDF.
  - **Scale mismatch**: The training robot's workspace may be a different size.
- **Custom stats not yet tested**: `collect_stats.py` can record your arm's motion range to create custom unnormalization stats (`--unnorm-key trossen_wxai_custom`).
- **Firmware upgrade blocked**: SDK v1.9.0+ (with native Cartesian API) requires firmware upgrade via Teensy Loader + USB cable.
