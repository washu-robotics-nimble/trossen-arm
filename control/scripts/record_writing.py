"""
Record scripted whiteboard writing as LeRobot-ready episodes (Phase 2 data engine).

Runs the same trajectory planner as write_text.py, but instead of just writing
it captures a camera frame + joint state at a fixed rate while the arm draws,
saving each run as a raw episode.  Text position (and optionally size) is
randomized per episode so the same instruction is demonstrated across the
board — the variation that lets a fine-tuned VLA compose letters into words.

Output is the SAME raw format collect_dataset.py produces, so
learning/dataset/build_lerobot_dataset.py converts it to a LeRobotDataset
with no changes:
    episode_NNN/ { frames/, observations.npy (N,7), timestamps.npy, metadata.json }

Because the board is fixed across a session, contact depth is measured ONCE at
the start (via --touch-off or --probe) and reused for every episode.

Usage:
  # 20 episodes of writing "A" at random board positions
  python control/scripts/record_writing.py "A" --episodes 20 --touch-off

  # custom instruction text and output dir
  python control/scripts/record_writing.py "A" --episodes 50 \\
      --task "write the letter A on the whiteboard" --output data/raw_episodes
"""

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import trossen_arm

from perception.utils.camera import open_camera
from control.writing.contact import probe_contact, ContactError
from control.writing.glyphs import text_to_strokes_ttf
from control.writing.hershey import text_to_strokes
from control.writing.jog import jog_to_point
from control.writing.kinematics import forward_kinematics, inverse_kinematics
from control.writing.safety import home_on_interrupt, glide_home
from control.writing.trajectory import plan_strokes, resample_waypoints, PlanningError
from control.writing.whiteboard import WhiteboardPlane

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml")
READY = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])
RECORD_HZ = 10


def _render(text, char_height, font, font_file, use_ttf):
    if use_ttf:
        return text_to_strokes_ttf(text, char_height=char_height, font_path=font_file)
    return text_to_strokes(text, char_height=char_height, font=font)


def _grab(cap_ref, reopen, retries=5):
    """Read a frame, retrying and re-opening the camera on failure.

    macOS/AVFoundation occasionally drops the capture stream; a few retries
    plus one re-open recovers it instead of losing the whole episode.
    cap_ref is a one-element list so a re-opened handle propagates to the
    caller.
    """
    for attempt in range(retries):
        ok, frame = cap_ref[0].read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
        if attempt == retries - 2:  # last-ditch: re-open the camera
            try:
                cap_ref[0].release()
            except Exception:
                pass
            cap_ref[0] = reopen()
    return None


def _record_episode(driver, cap_ref, reopen, samples, task, episode_dir):
    """Execute uniform-dt joint targets, capturing frame+state per step."""
    frames_dir = os.path.join(episode_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    states, frame_paths, timestamps = [], [], []
    dt = 1.0 / RECORD_HZ
    start = time.time()
    misses = 0

    for idx, (q, _pen) in enumerate(samples):
        t0 = time.time()
        # observation captured BEFORE the step is commanded (obs -> action)
        frame = _grab(cap_ref, reopen)
        state = np.array(driver.get_positions()[:7], dtype=np.float32)  # 6 arm + gripper
        if frame is None:
            misses += 1
            if misses <= 3:
                print("  WARNING: camera read failed (retried+reopened), skipping frame.")
            # still command the move so the arm doesn't stall mid-stroke
            driver.set_arm_positions(np.asarray(q, dtype=np.float64), dt, True)
            continue
        rel = f"frames/frame_{idx:04d}.png"
        cv2.imwrite(os.path.join(episode_dir, rel), frame)
        frame_paths.append(rel)
        states.append(state)
        timestamps.append(t0 - start)

        driver.set_arm_positions(np.asarray(q, dtype=np.float64), dt, True)

    n = len(states)
    if misses:
        print(f"  ({misses} frames dropped to camera hiccups)")
    if n < 2:
        print("  Episode too short — discarding.")
        return 0
    np.save(os.path.join(episode_dir, "observations.npy"), np.array(states, dtype=np.float32))
    np.save(os.path.join(episode_dir, "timestamps.npy"), np.array(timestamps, dtype=np.float64))
    with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
        json.dump({"task": task, "fps": RECORD_HZ, "num_frames": n,
                   "frame_paths": frame_paths}, f, indent=2)
    print(f"  Saved {n} frames -> {episode_dir}")
    return n


def _measure_surface(driver, plane, uv, args):
    """One-time contact measurement for the session (board is fixed)."""
    _, R = forward_kinematics(READY, plane.tip_length)
    target_dir = R[:, 0]
    if args.touch_off:
        print("\nOne-time TOUCH-OFF: jog the marker until it just touches, SPACE to set.")
        hover_pos = plane.point(*uv, offset=args.hover)
        q_hover, ok, _ = inverse_kinematics(hover_pos, target_dir=target_dir,
                                            q_init=READY, tip_length=plane.tip_length)
        driver.set_arm_positions(q_hover if ok else READY, 2.0, True)
        q_touch, pos_touch = jog_to_point(
            driver, q_hover if ok else READY, plane.tip_length, target_dir,
            "SURFACE ('w'=toward board, SPACE when it just touches)")
        if q_touch is None:
            raise ContactError("Touch-off aborted.")
        return float((pos_touch - plane.origin) @ plane.normal)
    if args.probe:
        print("\nOne-time PROBE for the board surface...")
        return probe_contact(driver, plane, uv, target_dir,
                             threshold=args.contact_threshold, hover=args.hover)
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", help="Text to write and record")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to record")
    p.add_argument("--task", default=None, help="Instruction string stored per episode "
                   "(default: \"write '<text>' on the whiteboard\")")
    p.add_argument("--output", default="data/raw_episodes", help="Raw episode output dir")
    p.add_argument("--height", type=float, default=0.05, help="Letter height (m)")
    p.add_argument("--randomize-height", action="store_true",
                   help="Also vary letter height ±20%% per episode")
    p.add_argument("--margin", type=float, default=0.015, help="Edge margin (m)")
    p.add_argument("--press", type=float, default=0.002, help="Press depth past surface (m)")
    p.add_argument("--speed", type=float, default=0.04, help="Writing speed (m/s)")
    p.add_argument("--hover", type=float, default=0.03, help="Pen-up hover (m)")
    p.add_argument("--font", default="futural", help="Hershey font (Latin)")
    p.add_argument("--font-file", default=None, help="TTF/OTF for non-Latin text")
    p.add_argument("--touch-off", action="store_true", help="Manual contact set (once)")
    p.add_argument("--probe", action="store_true", help="Force-probe contact (once)")
    p.add_argument("--contact-threshold", type=float, default=2.0, help="Probe threshold (Nm)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for position randomization")
    p.add_argument("--calibration", default=None, help="Whiteboard calibration JSON")
    args = p.parse_args()

    text = args.text.replace("\\n", "\n")
    task = args.task or f"write '{text}' on the whiteboard"
    rng = random.Random(args.seed)

    plane = WhiteboardPlane.load(args.calibration) if args.calibration else WhiteboardPlane.load()
    use_ttf = bool(args.font_file) or any(ord(c) > 0x7f for c in text)

    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", args.output))
    os.makedirs(out_root, exist_ok=True)
    ep_idx = len([d for d in os.listdir(out_root) if d.startswith("episode_")])

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    print(f"Board: {plane.width*100:.1f} x {plane.height*100:.1f} cm | task: '{task}'")
    print(f"Recording {args.episodes} episodes into {out_root} (starting at {ep_idx})")

    # NB: the camera is opened LATE (just before recording), not here — an
    # idle capture session dropped during the long touch-off jog is why reads
    # were failing.  reopen() re-acquires it robustly on demand.
    def reopen():
        return open_camera(cfg["camera"])[0]

    print("Connecting to robot...")
    driver = trossen_arm.TrossenArmDriver()

    def _configure(clear_error):
        driver.configure(trossen_arm.Model.wxai_v0,
                         trossen_arm.StandardEndEffector.wxai_v0_leader,
                         cfg["robot"]["ip"], clear_error)
    _configure(True)

    with home_on_interrupt(driver, clear_error=lambda: _configure(True)):
        print("Opening the gripper...")
        driver.set_gripper_mode(trossen_arm.Mode.external_effort)
        driver.set_gripper_external_effort(20.0, 3.0, True)
        input("\nPlace the marker in the gripper. Press Enter to close...")
        driver.set_gripper_external_effort(-20.0, 3.0, True)
        time.sleep(0.5)

        input("\nClear the area. Press Enter to move to ready pose...")
        driver.set_arm_modes(trossen_arm.Mode.position)
        driver.set_arm_positions(READY, 3.0, True)

        # one-time contact measurement at board center
        surface_offset = _measure_surface(driver, plane, (plane.width / 2, plane.height / 2), args)
        print(f"Surface offset: {surface_offset*1000:+.1f} mm (reused for all episodes)")

        # open the camera NOW (after the long interactive touch-off) so the
        # capture stream is fresh when recording starts.
        print("Opening camera...")
        cap, cam_idx = open_camera(cfg["camera"])
        cap_ref = [cap]
        print(f"Camera ready on index {cam_idx}.")

        recorded = 0
        target = args.episodes
        while recorded < target:
            # randomize letter height and position within the board
            ch = args.height * (rng.uniform(0.8, 1.2) if args.randomize_height else 1.0)
            strokes, w, h = _render(text, ch, args.font, args.font_file, use_ttf)
            if not strokes:
                print("Nothing to render — aborting.")
                break
            umax = plane.width - w - args.margin
            vmax = plane.height - h - args.margin
            if umax <= args.margin or vmax <= args.margin:
                print(f"Text {w*100:.1f}x{h*100:.1f}cm too big for board at this size — "
                      "reduce --height.")
                break
            u0 = rng.uniform(args.margin, umax)
            v0 = rng.uniform(args.margin, vmax)
            placed = [s + np.array([u0, v0]) for s in strokes]

            try:
                wps = plan_strokes(plane, placed, READY, write_speed=args.speed,
                                   hover=args.hover, press=args.press, margin=args.margin,
                                   surface_offset=surface_offset)
            except PlanningError as e:
                print(f"  [ep {ep_idx}] plan failed, retrying new position: {e}")
                continue

            samples = resample_waypoints(wps, READY, dt=1.0 / RECORD_HZ)
            print(f"\n[episode {ep_idx}]  pos=({u0*100:.1f},{v0*100:.1f})cm  "
                  f"height={ch*100:.1f}cm  {len(samples)} frames")

            driver.set_arm_positions(READY, 2.0, True)
            ep_dir = os.path.join(out_root, f"episode_{ep_idx:03d}")
            n = _record_episode(driver, cap_ref, reopen, samples, task, ep_dir)
            driver.set_arm_positions(READY, 2.0, True)
            if n > 0:
                ep_idx += 1
                recorded += 1
                print(f"  progress: {recorded}/{target}")

            # The board now has this episode's writing on it (even if the
            # recording was discarded, the arm still drew). Wipe it clean so
            # every episode starts from a consistent blank board.
            if recorded < target:
                input("\n  Wipe the whiteboard clean, then press Enter for the "
                      "next episode...")

        print(f"\nDone. Recorded {recorded} episodes.")
        cap_ref[0].release()
        print("Returning home...")
        glide_home(driver, clear_error=lambda: _configure(True))
        print("Next: python learning/dataset/build_lerobot_dataset.py "
              "--repo-id <you>/trossen-whiteboard --push")


if __name__ == "__main__":
    main()
