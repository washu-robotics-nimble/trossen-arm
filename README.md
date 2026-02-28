# Nimble Trossen NORA Pipeline

Simulation-to-training pipeline for fine-tuning NORA 1.5 (Vision-Language-Action model) on Trossen WidowX AI robot data collected in NVIDIA Isaac Sim.

## Pipeline Overview

1. **Data Collection** (`env_isaaclab` env) — Run pick-and-place in Isaac Sim, record EE state/action/camera as a LeRobot dataset
2. **Compute Norm Stats** (`trossen_nora` env) — Compute delta action normalization statistics
3. **Train NORA 1.5** (`trossen_nora` env, requires 4090) — Fine-tune NORA 1.5 via behavior cloning on the recorded dataset
4. **Deploy & Evaluate** — Run inference on real robot or sim

## Project Structure

```
nimble_trossen_anling/
├── nimble_trossen_isaac/    # Isaac Sim scripts & robot assets
│   ├── nimble_trossen.py    # Main script: pick-and-place + LeRobot recording
│   ├── controller.py        # WidowX AI differential IK controller
│   └── robots/              # Robot USD models
├── nora-1.5-main/           # NORA 1.5 model (declare-lab)
│   ├── training/lerobot/    # train_lerobot.py — fine-tuning script
│   ├── inference/           # Inference & model code
│   └── utils/               # compute_norm_stats.py, normalize.py
├── config/                  # Configuration files
├── control/                 # Control utilities
├── learning/                # Learning utilities
├── perception/              # Perception utilities
└── todo.md                  # Pipeline progress tracker
```

## Prerequisites

- Ubuntu 24.04
- NVIDIA GPU with latest drivers
  - **RTX 4060 (8GB)**: Simulation & data collection only
  - **RTX 4090 (24GB)**: Required for NORA 1.5 training (~7B params)
- Anaconda/Miniconda
- Git

## Environment Setup

This project requires **two separate conda environments** because Isaac Sim and NORA training have conflicting dependencies.

### Environment 1: `env_isaaclab` (Simulation & Data Collection)

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
```

#### Install Isaac Sim 5.1.0

```bash
pip install isaacsim==5.1.0.0 isaacsim-rl==5.1.0.0
```

Verify:

```bash
python -c "import isaacsim; print('Isaac Sim OK')"
```

#### Install Isaac Lab 2.3.0

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
pip install --no-build-isolation flatdict==4.0.1
./isaaclab.sh --install
```

If the core `isaaclab` package is missing after install:

```bash
pip install -e source/isaaclab
```

Verify:

```bash
python -c "from isaaclab.app import AppLauncher; print('IsaacLab OK')"
```

#### Install Trossen AI Isaac Extension

```bash
cd ~/Desktop/nimble
git clone https://github.com/TrossenRobotics/trossen_ai_isaac.git
pip install -e trossen_ai_isaac/source/trossen_ai_isaac
```

#### Install LeRobot (for data recording)

```bash
pip install lerobot==0.3.3
```

Isaac Sim pins specific versions — restore them after installing lerobot:

```bash
pip install gymnasium==1.2.0 packaging==23.0
```

### Environment 2: `trossen_nora` (Training & Norm Stats)

```bash
conda create -n trossen_nora python=3.11 -y
conda activate trossen_nora
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.54.0
pip install accelerate lerobot==0.3.3
pip install numpydantic pydantic scipy
```

Configure accelerate (single GPU, bf16):

```bash
accelerate config
```

> **Note**: `transformers==4.54.0` is required. v4.51 is missing `eager_attention_forward` for Qwen2.5-VL, and v5.2+ breaks the FAST tokenizer.

## Usage

### 1. Collect Data (4060 or 4090)

```bash
conda activate env_isaaclab
cd nimble_trossen_isaac
python nimble_trossen.py
```

This runs the pick-and-place demo and saves a LeRobot dataset to `nimble_trossen_isaac/generated_lerobot_datasets_<timestamp>/`.

### 2. Compute Normalization Stats (any GPU)

```bash
conda activate trossen_nora
cd nora-1.5-main/utils
python compute_norm_stats.py --dataset_path /path/to/generated_lerobot_datasets_<timestamp> --delta_transform
```

### 3. Train NORA 1.5 (requires 4090)

Update `data_root_dir` in `nora-1.5-main/training/lerobot/train_lerobot.py` to point to your dataset, then:

```bash
conda activate trossen_nora
cd nora-1.5-main/training/lerobot
WANDB_MODE=disabled accelerate launch train_lerobot.py
```

## Version Compatibility

| Component | Version | Environment |
|---|---|---|
| Isaac Sim | 5.1.0 | `env_isaaclab` |
| Isaac Lab | 2.3.0 | `env_isaaclab` |
| Trossen AI Isaac | 0.1.0 | `env_isaaclab` |
| LeRobot | 0.3.3 | both |
| transformers | 4.54.0 | `trossen_nora` |
| Python | 3.11 | both |

## Reference

- [Trossen AI Isaac Docs](https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html)
- [Isaac Lab Docs](https://isaac-sim.github.io/IsaacLab)
- [Isaac Sim Docs](https://docs.isaacsim.omniverse.nvidia.com)
- [NORA 1.5 (declare-lab)](https://github.com/declare-lab/nora)
- [LeRobot](https://github.com/huggingface/lerobot)
