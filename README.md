# Nimble Trossen NORA Pipeline

> 📹 **Demo video** (trajectory recording): [Google Drive](https://drive.google.com/drive/folders/1Tns9Wm0CSOvfmWFekixoPJ7-kg43G-dX?usp=share_link)


Fine-tuning NORA 1.5 (Vision-Language-Action model) on Trossen WidowX AI robot data collected in NVIDIA Isaac Sim.

## Folders

- **`Trossen_Full_Loop/`** — Real-arm CLI pick-and-place loop (IK + safety checks + keyframe recording/replay)
- **`nimble_trossen_isaac/`** — Isaac Sim pick-and-place simulation + LeRobot dataset recording (uses `env_isaaclab` env)
- **`nora-1.5-main/`** — NORA 1.5 training, inference, and normalization stats (uses `trossen_nora` env, training requires 4090)
- **`nora_trossenarm/`** — Real robot scripts for Trossen WidowX AI (camera inference, marker pickup, YOLO detection)

## Trossen_Full_Loop (CLI)

Interactive command-line loop for real-arm pick-and-place. Each waypoint you type a line like
`0.18, 0.10, 0.06, 1` (x, y, z in meters + gripper 1=open/0=closed).

### Run

```bash
cd Trossen_Full_Loop
python main.py
```

### Commands (at the `target [x,y,z,gripper]:` prompt)

| Input           | Action                                                      |
|-----------------|-------------------------------------------------------------|
| `x, y, z, g`    | Run safety → IK → move to target                            |
| `p`             | Print recorded keyframes so far                             |
| `e`             | Emergency stop (freeze, zero, optionally resume via replay) |
| `y` / `n` / `r` | After motion: record / manual adjust / redo                 |
| `g` / `k`       | In manual adjust: toggle gripper / capture keyframe         |

### Replay a saved recording

```bash
python replay.py                 # lists available recordings, then prompts
python replay.py <task_name>     # replays that specific recording
```

### Outputs

- Saved trajectories: `Trossen_Full_Loop/IK_RESULTS/<task_name>.npy`
- Console output only (no text log files)
- Optional 3D path visualization via `plot_arm_motion_safety_check`
  (matplotlib window — requires a display; not for headless / SSH)

---

## Two Conda Environments

| Environment | Purpose |
|---|---|
| `env_isaaclab` | Simulation & data collection (Isaac Sim + Isaac Lab + LeRobot) |
| `trossen_nora` | Training & norm stats (transformers==4.54.0 + LeRobot) |

They are separate because Isaac Sim and NORA training have conflicting dependencies.

## Quick Start

```bash
# 1. Collect data in simulation
conda activate env_isaaclab
cd nimble_trossen_isaac
python nimble_trossen.py

# 2. Compute normalization stats
conda activate trossen_nora
cd nora-1.5-main/utils
python compute_norm_stats.py --dataset_path /path/to/dataset --delta_transform

# 3. Train NORA 1.5 (4090 only)
cd nora-1.5-main/training/lerobot
WANDB_MODE=disabled accelerate launch train_lerobot.py
```

## Reference

- [Trossen AI Isaac Docs](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html)
- [NORA 1.5 (declare-lab)](https://github.com/declare-lab/nora)
- [LeRobot](https://github.com/huggingface/lerobot)
