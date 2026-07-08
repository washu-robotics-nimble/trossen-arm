"""
Verify the camera is accessible before data collection / training.

Opens the camera from config/robot_config.yaml, confirms frames can be read,
reports the actual resolution/FPS, and scans a few device indices so you can
find the right one if the configured index is wrong.  Optionally saves a test
snapshot so you can confirm the whiteboard is in view.

Usage:
  python control/scripts/check_camera.py
  python control/scripts/check_camera.py --save cam_test.png
  python control/scripts/check_camera.py --scan          # try indices 0..4
"""

import argparse
import os
import sys

import cv2
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml")
_BACKENDS = {"AVFOUNDATION": cv2.CAP_AVFOUNDATION, "DSHOW": cv2.CAP_DSHOW,
             "V4L2": cv2.CAP_V4L2}


def _try_open(index, backend_name, width, height, warmup=5):
    backend = _BACKENDS.get(backend_name, cv2.CAP_ANY)
    cap = cv2.VideoCapture(index, backend)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        cap.release()
        return None
    # warm up — first reads often fail while the sensor initializes
    frame = None
    for _ in range(warmup):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame": frame,
    }
    cap.release()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default=None, help="Save a test snapshot to this path")
    p.add_argument("--scan", action="store_true", help="Scan device indices 0..4")
    args = p.parse_args()

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["camera"]
    idx, backend, w, h = cfg["device_id"], cfg.get("backend"), cfg["width"], cfg["height"]

    print(f"Config: device_id={idx}, backend={backend}, {w}x{h}\n")

    if args.scan:
        print("Scanning device indices 0..4...")
        found = []
        for i in range(5):
            info = _try_open(i, backend, w, h)
            if info and info["frame"] is not None:
                print(f"  index {i}: OK  {info['width']}x{info['height']} @ {info['fps']:.0f}fps")
                found.append(i)
            elif info:
                print(f"  index {i}: opened but no frame")
            else:
                print(f"  index {i}: not available")
        print(f"\nUsable camera indices: {found or 'NONE'}")
        if found and idx not in found:
            print(f"NOTE: configured device_id={idx} is not usable; "
                  f"set camera.device_id to {found[0]} in config/robot_config.yaml.")
        return 0 if found else 1

    # Use the SAME robust opener the data-collection scripts use, so this
    # check reflects reality even when macOS reorders the device index.
    print("Resolving camera (auto, tolerant of index reordering)...")
    try:
        from perception.utils.camera import open_camera
        cap, used_idx = open_camera(cfg)
    except RuntimeError as e:
        print(f"FAILED: {e}")
        return 1

    ok, frame = False, None
    for _ in range(5):
        r, f = cap.read()
        if r and f is not None:
            ok, frame = True, f
    aw, ah = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ok:
        print("FAILED: opened a camera but could not read frames.")
        return 1
    print(f"OK: camera works on index {used_idx} — {aw}x{ah} @ {afps:.0f} fps")
    if (aw, ah) != (w, h):
        print(f"NOTE: got {aw}x{ah}, config requested {w}x{h} (nearest supported mode).")

    if args.save:
        out = os.path.abspath(args.save)
        cv2.imwrite(out, frame)
        print(f"Saved a test snapshot to {out} — open it to confirm the whiteboard is in view.")
    else:
        print("Tip: rerun with --save cam_test.png to eyeball what the camera sees.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
