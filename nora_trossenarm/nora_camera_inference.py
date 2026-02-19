"""
Example usage:
    python nora_trossenarm/nora_camera_inference.py
    python nora_trossenarm/nora_camera_inference.py --camera 0 --instruction "pick up the marker"

    # Dry run (no robot connected, just prints actions):
    python nora_trossenarm/nora_camera_inference.py --dry-run

    # With robot:
    python nora_trossenarm/nora_camera_inference.py --robot-ip 192.168.1.2 --unnorm-key bridge_orig
"""

import sys
import argparse
import numpy as np
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

import trossen_arm


def init_robot(robot_ip):
    """Initialize and configure the Trossen arm driver."""
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        robot_ip,
        False,
    )
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_gripper_mode(trossen_arm.Mode.position)
    return driver


def execute_actions(driver, actions, goal_time):
    """Send a chunk of unnormalized actions to the robot.

    Args:
        driver: TrossenArmDriver instance.
        actions: np.ndarray of shape (chunk_length, 7)
                 where [:6] = [x, y, z, rx, ry, rz] and [6] = gripper.
        goal_time: seconds for each step.
    """
    for step_idx, action in enumerate(actions):
        cartesian_pose = action[:6].tolist()
        gripper_pos = float(action[6])

        print(f"  Step {step_idx}: pose={np.round(cartesian_pose, 4)}, gripper={gripper_pos:.4f}")

        driver.set_cartesian_positions(
            cartesian_pose,
            trossen_arm.InterpolationSpace.cartesian,
            goal_time,
            True,  # blocking
        )
        driver.set_gripper_position(gripper_pos, goal_time, True)


def main(args):
    """Run NORA inference with camera, marker detection, and Trossen arm control."""
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

    # Initialize robot
    driver = None
    if not args.dry_run:
        print(f"Connecting to robot at {args.robot_ip}...")
        driver = init_robot(args.robot_ip)
        print("Robot connected.")
    else:
        print("Dry-run mode: no robot commands will be sent.")

    print(f"Instruction: '{args.instruction}'")
    print(f"Unnormalization key: '{args.unnorm_key}'")
    print("Press 's' to sample & execute actions, 'q' to quit")

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
                normalized_actions = model.sample_actions(
                    pil_image, args.instruction, num_steps=args.num_steps
                )
                print(f"Predicted actions (normalized), shape: {normalized_actions.shape}")

                # Unnormalize: (1, chunk_length, 7) -> (chunk_length, 7)
                actions = model.unnormalize_action(normalized_actions, args.unnorm_key)
                actions = actions[0]  # remove batch dim
                print(f"Unnormalized actions:\n{np.round(actions, 4)}")

                if driver is not None:
                    print("Executing on robot...")
                    execute_actions(driver, actions, args.goal_time)
                    print("Done.")

    finally:
        camera.release()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NORA + Trossen arm inference")
    # Camera
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    # NORA
    parser.add_argument("--instruction", type=str, default="pick up the object",
                        help="Task instruction for NORA")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow matching steps")
    parser.add_argument("--unnorm-key", type=str, default="bridge_orig",
                        help="Dataset key for action unnormalization")
    # Robot
    parser.add_argument("--robot-ip", type=str, default="192.168.1.2",
                        help="Trossen arm IP address")
    parser.add_argument("--goal-time", type=float, default=2.0,
                        help="Seconds per action step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without connecting to the robot")

    args = parser.parse_args()
    main(args)
