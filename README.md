# Nimble Trossen NORA Pipeline

Fine-tuning NORA 1.5 (Vision-Language-Action model) on Trossen WidowX AI robot data collected in NVIDIA Isaac Sim.

## Folders

- **`nimble_trossen_isaac/`** — Isaac Sim pick-and-place simulation + LeRobot dataset recording (uses `env_isaaclab` env)
- **`nora-1.5-main/`** — NORA 1.5 training, inference, and normalization stats (uses `trossen_nora` env, training requires 4090)
- **`nora_trossenarm/`** — Real robot scripts for Trossen WidowX AI (camera inference, marker pickup, YOLO detection)

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
