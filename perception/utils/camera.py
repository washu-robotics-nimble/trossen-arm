"""Shared camera utilities for perception tasks."""

import cv2
import time

_BACKENDS = {"AVFOUNDATION": cv2.CAP_AVFOUNDATION, "DSHOW": cv2.CAP_DSHOW,
             "V4L2": cv2.CAP_V4L2}


def open_camera(cfg, warmup=5, scan_max=5):
    """Open the camera robustly, tolerating macOS's unstable device indices.

    OpenCV's AVFoundation index for the same physical camera can change
    between runs, so a hardcoded device_id is unreliable.  This tries the
    configured index first, then scans other indices and picks the first one
    that actually DELIVERS a frame (the built-in often opens but returns no
    frame without Terminal camera permission, so "returns a frame" reliably
    finds the working external cam).

    Args:
        cfg: the "camera" section of robot_config.yaml (dict).
        warmup: reads to discard while the sensor initializes.
        scan_max: highest device index to scan.

    Returns:
        (cap, index): an opened cv2.VideoCapture and the index used.

    Raises:
        RuntimeError if no camera delivers a frame.
    """
    backend = _BACKENDS.get(cfg.get("backend"), cv2.CAP_ANY)
    w, h = cfg["width"], cfg["height"]
    preferred = cfg["device_id"]
    order = [preferred] + [i for i in range(scan_max) if i != preferred]

    for idx in order:
        cap = cv2.VideoCapture(idx, backend)
        if w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        good = 0
        if cap.isOpened():
            # require several CONSECUTIVE good reads — a single fluke frame
            # during warmup isn't enough to trust a flaky device.
            for _ in range(warmup + 5):
                ok, frame = cap.read()
                good = good + 1 if (ok and frame is not None) else 0
                if good >= 3:
                    break
        if good >= 3:
            if idx != preferred:
                print(f"[camera] configured index {preferred} unusable; "
                      f"using index {idx} instead (macOS indices are unstable).")
            return cap, idx
        cap.release()

    raise RuntimeError(
        f"No camera delivered a frame (scanned indices {order}). "
        "Check the Logitech is plugged in and Terminal has camera permission "
        "(System Settings > Privacy & Security > Camera).")


class Camera:
    """Simple camera wrapper with FPS tracking."""
    
    def __init__(self, camera_id=0, width=1280, height=720, backend=cv2.CAP_AVFOUNDATION):
        """
        Initialize camera.
        
        Args:
            camera_id: Camera device ID (default 0)
            width: Frame width
            height: Frame height
            backend: OpenCV backend (CAP_AVFOUNDATION for macOS, CAP_DSHOW for Windows)
        """
        self.cap = cv2.VideoCapture(camera_id, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.prev_time = time.time()
        
    def read(self):
        """Read frame from camera."""
        return self.cap.read()
    
    def get_fps(self):
        """Calculate and return current FPS."""
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time
        return fps
    
    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()
