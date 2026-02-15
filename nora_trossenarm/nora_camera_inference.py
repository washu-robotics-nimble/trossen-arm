"""
Example usage:
    python nora_trossenarm/nora_camera_inference.py
    python nora_trossenarm/nora_camera_inference.py --camera 0 --instruction "pick up the marker"
"""

import sys
import argparse
import cv2
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "nora-1.5-main"))

from perception.utils.camera import Camera
from learning.inference.predictor import MarkerPredictor
from inference.modelling_expert import VLAWithExpert


def main(args):
    """Run NORA inference with camera and marker detection."""
    # Initialize camera
    camera = Camera(camera_id=args.camera, width=args.width, height=args.height)

    # Initialize YOLO marker detector
    print("Loading YOLO marker detector...")
    marker_predictor = MarkerPredictor(device=args.device)

    # Initialize NORA model
    print("Loading NORA 1.5 model (this may take a while)...")
    model = VLAWithExpert.from_pretrained("declare-lab/nora-1.5")
    model.eval()
    print("Model loaded.")

    print(f"Instruction: '{args.instruction}'")
    print("Press 's' to sample actions, 'q' to quit")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run YOLO marker detection on every frame
            annotated_frame = marker_predictor.predict_and_plot(
                frame, conf=args.conf, imgsz=args.imgsz
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
                2,
            )

            cv2.imshow("NORA + Marker Detection", annotated_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Convert BGR (OpenCV) to RGB (PIL) for NORA
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                print("Running NORA inference...")
                actions = model.sample_actions(
                    pil_image, args.instruction, num_steps=args.num_steps
                )
                print(f"Predicted actions (normalized): {actions}")
                print(f"Shape: {actions.shape}")  # (1, action_chunk_length, 7)

    finally:
        camera.release()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NORA + marker detection inference")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--instruction", type=str, default="pick up the object",
                        help="Task instruction for NORA")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow matching steps")

    args = parser.parse_args()
    main(args)
