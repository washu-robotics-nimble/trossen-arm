# Nimble Trossen NORA Pipeline — TODO

## Round 1 — Data Collection & Imitation Learning

### Completed
- [v] Modified Trossen WidowX AI pick-and-place demo into a LeRobot data recorder (`nimble_trossen.py`)
- [v] Records per-frame: EE state (7D), action (7D), wrist camera image (224x224x3)
- [v] Output saved as LeRobot dataset with timestamped folders
- [v] Computed delta action normalization stats (`compute_norm_stats.py` → `norm_stats.json`)
- [v] Fixed REMAP_KEY in `compute_norm_stats.py` and `train_lerobot.py` to match our dataset keys (`observation.state`, `observation.images.scene`)
- [v] Set up `trossen_nora` conda env with training dependencies
- [v] Configured accelerate (single GPU, bf16)
- [v] Set `data_root_dir` and lowered training params for test run (batch=4, steps=50)

### Marker & Environment Setup
- [ ] Create marker USD model (simple cylinder ~12mm x 140mm with grip point and tip contact)
- [ ] Add writing surface to the scene (flat plane at known height)
- [ ] Generate stroke trajectories from font data (single-stroke/plotter fonts for letters/words)
- [ ] Replace cube pick-and-place with marker pick-up + writing motion

### Edit `nimble_trossen.py` for Multi-Episode Recording
- [ ] Support auto-looping through many episodes without manual STOP/PLAY
- [ ] Randomize task each episode (e.g., pick random word from dictionary)
- [ ] Pass per-episode task description to `DataRecorder` (e.g., "Write the word hello")
- [ ] All episodes stored in ONE dataset folder (not separate timestamped folders per run)
- [ ] Compute norm stats once on the combined dataset

### Train NORA 1.5 (Behavior Cloning) — requires 4090
- [ ] Transfer repo + dataset to 4090 machine
- [ ] Set up `trossen_nora` env on 4090 (same deps: transformers==4.54.0, etc.)
- [ ] Run test training to verify pipeline end-to-end (50 steps)
- [ ] Scale up data collection (hundreds of episodes with varied tasks)
- [ ] Run full training with scaled dataset
- [ ] Save fine-tuned model checkpoint

## Round 2 — Deploy & Evaluate

### Inference on Real Robot / Sim
- [ ] Run fine-tuned NORA 1.5 with language instructions on real robot or Isaac Lab
- [ ] Record inference trajectories (what the model actually does)
- [ ] Convert inference data back to Isaac Lab format for visualization
- [ ] Evaluate success/failure rate per task

## Round 3 — Iterative Improvement (if needed)

### DPO / Further RL
- [ ] Compare successful vs failed trajectories from inference
- [ ] Use DPO (Direct Preference Optimization) to teach model preferences
- [ ] Or apply RL with reward function to push beyond demo quality
- [ ] Retrain → redeploy → evaluate → repeat

## Notes
- **GPU**: 4060 (8GB) for sim/data collection, 4090 (24GB) for training
  - NORA 1.5 (~7B params) takes ~6.4GB just to load — 4060 can't train, only collect data
- Image format used (not video) — may need to revisit for storage at scale
- `trossen_nora` conda env is for training; `env_isaaclab` is for simulation/data collection
- `transformers==4.54.0` required (4.51 missing eager_attention_forward, 5.2 breaks FAST tokenizer)
- Isaac Sim pins `gymnasium==1.2.0` and `packaging==23.0` — restore after any lerobot install
- Use `WANDB_MODE=disabled` to skip wandb login for test runs
- Fractal DPO not needed initially — fine-tuned model is the baseline; DPO is a refinement step
- Each round improves the model: scripted demos → behavior cloning → deploy → evaluate → DPO/RL → better model
