"""
Live marker detection with YOLO inference.

Example usage:
    python live_detection.py
    python live_detection.py --conf 0.5 --imgsz 1280
"""

import sys
from pathlib import Path
import argparse
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from perception.utils.camera import Camera
from learning.inference.predictor import MarkerPredictor


def main(args):
    """Run live marker detection."""
    # Initialize camera and predictor
    camera = Camera(camera_id=args.camera, width=args.width, height=args.height)
    predictor = MarkerPredictor(device=args.device)
    
    print("Starting live detection... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Run inference and annotate
            annotated_frame = predictor.predict_and_plot(
                frame, 
                conf=args.conf, 
                imgsz=args.imgsz
            )
            
            # Add FPS counter
            fps = camera.get_fps()
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display
            cv2.imshow("Live Marker Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        print("Detection stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live marker detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    
    args = parser.parse_args()
    main(args)
