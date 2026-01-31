"""Shared camera utilities for perception tasks."""

import cv2
import time


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
