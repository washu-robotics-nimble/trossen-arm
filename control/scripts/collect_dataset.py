"""
Teleoperation dataset collection for LeRobot fine-tuning.

The arm is placed in gravity-compensation (IDLE) mode so an operator can
physically guide it.  Camera frames and joint states are recorded at 10 Hz
and saved as raw episodes under data/raw_episodes/.

Controls (all require pressing Enter after the key):
  <Enter>        start a new episode / stop the current recording
  g <Enter>      toggle gripper open / closed between episodes
  q <Enter>      quit and exit

Usage:
  python control/scripts/collect_dataset.py
  python control/scripts/collect_dataset.py --task "write letter A on whiteboard"
  python control/scripts/collect_dataset.py --task "..." --output data/raw_episodes --target 50
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import trossen_arm

from perception.utils.camera import open_camera

RECORD_HZ = 10
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml")


def _load_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _init_robot(cfg):
    driver = trossen_arm.TrossenArmDriver()
    # 4th arg is clear_error — clear any stale fault so idle/gravity-comp
    # mode actually engages (an errored arm silently ignores mode commands).
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        cfg["robot"]["ip"],
        True,
    )
    return driver


def _init_camera(cfg):
    # Use the shared robust opener (tolerates macOS index reordering / a
    # camera that opens but delivers no frames) instead of a hand-rolled open.
    cap, _idx = open_camera(cfg["camera"])
    return cap


def _set_gripper(driver, open_gripper: bool):
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    effort = 20.0 if open_gripper else -20.0
    driver.set_gripper_external_effort(effort, 1.5, True)
    driver.set_gripper_mode(trossen_arm.Mode.idle)


def _record_episode(driver, cap, task: str, episode_dir: str, gripper_open: bool) -> int:
    frames_dir = os.path.join(episode_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    states: list[np.ndarray] = []
    frame_paths: list[str] = []
    timestamps: list[float] = []

    stop_event = threading.Event()

    def _wait():
        input()
        stop_event.set()

    listener = threading.Thread(target=_wait, daemon=True)
    listener.start()

    print("  Recording — press Enter to stop.")
    start = time.time()
    idx = 0

    while not stop_event.is_set():
        t0 = time.time()

        ret, frame = cap.read()
        # record the real 7-dim joint state (6 arm + gripper position), matching
        # record_writing.py — a fabricated 0/1 gripper flag would be a different
        # scale from record_writing's real gripper value and corrupt norm stats.
        state = np.array(driver.get_positions()[:7], dtype=np.float32)

        if ret:
            rel_path = f"frames/frame_{idx:04d}.png"
            cv2.imwrite(os.path.join(episode_dir, rel_path), frame)
            frame_paths.append(rel_path)
            states.append(state)
            timestamps.append(t0 - start)
            idx += 1
        else:
            print("  WARNING: camera read failed, skipping frame.")

        sleep = (1.0 / RECORD_HZ) - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    n = len(states)
    if n < 2:
        print("  Episode too short (< 2 frames) — discarding.")
        return 0

    np.save(os.path.join(episode_dir, "observations.npy"), np.array(states, dtype=np.float32))
    np.save(os.path.join(episode_dir, "timestamps.npy"), np.array(timestamps, dtype=np.float64))
    with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
        json.dump({"task": task, "fps": RECORD_HZ, "num_frames": n, "frame_paths": frame_paths}, f, indent=2)

    print(f"  Saved {n} frames → {episode_dir}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="write on whiteboard", help="Task description")
    parser.add_argument("--output", default="data/raw_episodes", help="Output directory")
    parser.add_argument("--target", type=int, default=50, help="Target episode count (informational)")
    args = parser.parse_args()

    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", args.output))
    os.makedirs(out_root, exist_ok=True)

    existing = sorted(d for d in os.listdir(out_root) if d.startswith("episode_"))
    ep_idx = len(existing)

    print("Connecting to robot…")
    cfg = _load_config()
    driver = _init_robot(cfg)

    print("Opening camera…")
    cap = _init_camera(cfg)

    print(f"\nTask : '{args.task}'")
    print(f"Output: {out_root}")
    print(f"Target: {args.target} episodes  (currently have {ep_idx})\n")

    print("Switching arm to IDLE (gravity compensation). HOLD THE ARM NOW.")
    driver.set_arm_modes(trossen_arm.Mode.idle)
    driver.set_gripper_mode(trossen_arm.Mode.idle)
    time.sleep(0.5)

    gripper_open = True  # track gripper state manually
    print("Gripper: OPEN\n")

    try:
        while True:
            print(f"── Episode {ep_idx} ──  [{ep_idx}/{args.target} collected]")
            print("  <Enter> = start recording  |  g<Enter> = toggle gripper  |  q<Enter> = quit")
            cmd = input("> ").strip().lower()

            if cmd == "q":
                break

            if cmd == "g":
                gripper_open = not gripper_open
                label = "OPEN" if gripper_open else "CLOSED"
                print(f"  Setting gripper → {label}…")
                _set_gripper(driver, gripper_open)
                # Return arm to idle after gripper move
                driver.set_arm_modes(trossen_arm.Mode.idle)
                print(f"  Gripper is now {label}.\n")
                continue

            ep_dir = os.path.join(out_root, f"episode_{ep_idx:03d}")
            n = _record_episode(driver, cap, args.task, ep_dir, gripper_open)
            if n > 0:
                ep_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        print(f"\nDone. {ep_idx} episodes in {out_root}")


if __name__ == "__main__":
    main()
