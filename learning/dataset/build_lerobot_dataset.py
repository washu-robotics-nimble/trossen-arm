"""
Convert raw collected episodes into a LeRobotDataset and optionally push to HuggingFace.

Raw episodes are produced by control/scripts/collect_dataset.py and live under
data/raw_episodes/.  Each episode_NNN/ directory contains:
  metadata.json      — task string, fps, num_frames, frame_paths list
  observations.npy   — float32 (N, 7): 6 arm joints + 1 gripper value
  frames/            — PNG images named frame_NNNN.png

The dataset uses keys expected by nora-1.5/training/lerobot/train_lerobot.py:
  observation.images.scene  — camera frames (video)
  observation_state         — 7-dim joint state
  action                    — 7-dim target state (next frame's state)

After building, run compute_norm_stats.py to generate norm_stats.json required
for training.

Usage:
  python learning/dataset/build_lerobot_dataset.py \\
      --repo-id yourname/trossen-whiteboard-writing

  # push to HuggingFace when ready:
  python learning/dataset/build_lerobot_dataset.py \\
      --repo-id yourname/trossen-whiteboard-writing --push
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]

_RAW_DEFAULT = os.path.join(os.path.dirname(__file__), "../../data/raw_episodes")
_NORM_STATS_SCRIPT = os.path.join(os.path.dirname(__file__), "../../nora-1.5/utils/compute_norm_stats.py")


def _load_episode(ep_dir: Path):
    with open(ep_dir / "metadata.json") as f:
        meta = json.load(f)
    observations = np.load(ep_dir / "observations.npy")  # (N, 7)
    frame_paths = [ep_dir / p for p in meta["frame_paths"]]
    return meta, observations, frame_paths


def _infer_image_shape(episode_dirs: list[Path]) -> tuple[int, int]:
    for ep_dir in episode_dirs:
        meta, _, frame_paths = _load_episode(ep_dir)
        if frame_paths:
            frame = cv2.imread(str(frame_paths[0]))
            if frame is not None:
                h, w = frame.shape[:2]
                return h, w
    raise RuntimeError("Could not read any frame to determine image shape.")


def _create_dataset(repo_id: str, fps: int, h: int, w: int) -> LeRobotDataset:
    features = {
        "observation.images.scene": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation_state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="trossen_wxai",
        features=features,
        image_writer_processes=0,
        image_writer_threads=4,
    )


def _add_episode(dataset: LeRobotDataset, meta: dict, observations: np.ndarray, frame_paths: list):
    n = len(frame_paths)
    # Action = next frame's joint state (predict where the arm moves to)
    for i in range(n - 1):
        bgr = cv2.imread(str(frame_paths[i]))
        if bgr is None:
            print(f"  WARNING: could not read {frame_paths[i]}, skipping frame.")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        dataset.add_frame({
            "observation.images.scene": rgb,
            "observation_state": observations[i].astype(np.float32),
            "action": observations[i + 1].astype(np.float32),
        })
    dataset.save_episode(task=meta["task"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True,
                        help="HuggingFace dataset repo, e.g. yourname/trossen-whiteboard-writing")
    parser.add_argument("--raw", default=_RAW_DEFAULT, help="Path to raw episodes directory")
    parser.add_argument("--fps", type=int, default=10, help="Recording FPS (must match collection)")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub after building")
    parser.add_argument("--skip-norm-stats", action="store_true",
                        help="Skip computing norm_stats.json (run compute_norm_stats.py manually)")
    args = parser.parse_args()

    raw_dir = Path(args.raw).resolve()
    if not raw_dir.exists():
        print(f"Error: raw episodes directory not found: {raw_dir}")
        print("Run control/scripts/collect_dataset.py first.")
        sys.exit(1)

    ep_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    if not ep_dirs:
        print(f"No episodes found in {raw_dir}.")
        sys.exit(1)

    print(f"Found {len(ep_dirs)} episodes in {raw_dir}.")

    h, w = _infer_image_shape(ep_dirs)
    print(f"Image size: {w}×{h}, FPS: {args.fps}")

    print(f"\nCreating LeRobotDataset '{args.repo_id}'…")
    dataset = _create_dataset(args.repo_id, args.fps, h, w)

    for i, ep_dir in enumerate(ep_dirs):
        print(f"  [{i+1}/{len(ep_dirs)}] {ep_dir.name}")
        meta, observations, frame_paths = _load_episode(ep_dir)
        _add_episode(dataset, meta, observations, frame_paths)

    total_frames = len(dataset)
    print(f"\nDataset built: {len(ep_dirs)} episodes, {total_frames} frames.")

    if not args.skip_norm_stats:
        print("\nComputing norm_stats.json…")
        result = subprocess.run(
            [sys.executable, _NORM_STATS_SCRIPT, "--dataset-path", args.repo_id],
            capture_output=False,
        )
        if result.returncode != 0:
            print("WARNING: norm_stats computation failed. Run compute_norm_stats.py manually before training.")

    if args.push:
        print(f"\nPushing to HuggingFace Hub: {args.repo_id}")
        dataset.push_to_hub(tags=["robotics", "manipulation", "whiteboard-writing", "trossen-wxai"])
        print("Pushed successfully.")
    else:
        print(f"\nDataset is saved locally. Add --push to upload to HuggingFace Hub.")


if __name__ == "__main__":
    main()
