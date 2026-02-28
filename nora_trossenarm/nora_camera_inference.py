"""
Example usage:
    python nora_trossenarm/nora_camera_inference.py
    python nora_trossenarm/nora_camera_inference.py --camera 0 --instruction "pick up the marker"

    # Dry run (no robot connected, just prints actions):
    python nora_trossenarm/nora_camera_inference.py --dry-run

    # With robot:
    python nora_trossenarm/nora_camera_inference.py --instruction "pick up the marker"
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

import ikpy.chain
from scipy.spatial.transform import Rotation

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "nora-1.5-main"))

from perception.utils.camera import Camera
from learning.inference.predictor import MarkerPredictor
from inference.modelling_expert import VLAWithExpert

import trossen_arm
from util.hamer import reset_arm

robot_ip = "192.168.2.2" # specific to our arm
unnorm_key = "bridge_orig" # unnormalized stats for inference conversion. this one is trained on Bridge V2 dataset, similar to trossen arm.

# URDF path for FK/IK (SDK v1.7.x doesn't have set_cartesian_positions)
URDF_PATH = str(Path(__file__).parent / "src" / "wxai_base.urdf")


def build_ik_chain():
    """Build an ikpy chain from the WXAI V0 URDF for FK/IK."""
    chain = ikpy.chain.Chain.from_urdf_file(
        URDF_PATH,
        base_elements=["base_link"],
        active_links_mask=[False, True, True, True, True, True, True, False, False],
        name="wxai",
    )
    return chain


def joint_angles_to_cartesian(chain, joint_angles_6):
    """Forward kinematics: 6 joint angles -> [x, y, z, rx, ry, rz].

    Args:
        chain: ikpy Chain.
        joint_angles_6: array of 6 joint angles from the driver.
    Returns:
        np.ndarray of shape (6,): [x, y, z, rx, ry, rz]
    """
    # Pad to 9 elements: [base_fixed, j0..j5, carriage_fixed, gripper_fixed]
    full = np.zeros(9)
    full[1:7] = joint_angles_6
    fk = chain.forward_kinematics(full)
    pos = fk[:3, 3]
    rot = Rotation.from_matrix(fk[:3, :3]).as_euler("xyz")
    return np.concatenate([pos, rot])


def cartesian_to_joint_angles(chain, target_pose_6, current_joint_angles_6):
    """Inverse kinematics: [x, y, z, rx, ry, rz] -> 6 joint angles.

    Uses position-only IK (more robust for small deltas) and seeds with
    the current joint angles so the solution stays close to the current pose.

    Args:
        chain: ikpy Chain.
        target_pose_6: array [x, y, z, rx, ry, rz].
        current_joint_angles_6: current joint angles as IK seed.
    Returns:
        np.ndarray of shape (6,): joint angles.
    """
    target_pos = target_pose_6[:3]

    # Seed with current angles for continuity, clamped to URDF bounds
    seed = np.zeros(9)
    seed[1:7] = current_joint_angles_6

    for i, link in enumerate(chain.links):
        if hasattr(link, "bounds") and link.bounds is not None:
            lo, hi = link.bounds
            if lo is not None and hi is not None:
                seed[i] = np.clip(seed[i], lo, hi)

    ik = chain.inverse_kinematics(
        target_pos, initial_position=seed
    )
    return ik[1:7]


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
    return driver


def set_gripper(driver, openness):
    """Control gripper using effort mode.

    Args:
        driver: TrossenArmDriver instance.
        openness: 0.0 (closed) to 1.0 (open). Values near 0.5 are treated as hold.
    """
    if openness > 0.5:
        effort = 20.0   # open
    else:
        effort = -20.0  # close
    driver.set_gripper_mode(trossen_arm.Mode.external_effort)
    driver.set_gripper_external_effort(effort, 2.0, True)


def execute_actions(driver, chain, actions, goal_time):
    """Send a chunk of unnormalized delta actions to the robot.

    NORA outputs deltas (displacements), not absolute poses. For each step we:
    1. Read current joint angles -> FK to get current Cartesian pose
    2. Add NORA's delta to get target Cartesian pose
    3. IK to get target joint angles
    4. Send joint angles to the arm

    Args:
        driver: TrossenArmDriver instance.
        chain: ikpy Chain for FK/IK.
        actions: np.ndarray of shape (chunk_length, 7)
                 where [:6] = [dx, dy, dz, drx, dry, drz] deltas
                 and [6] = gripper position (absolute).
        goal_time: seconds for each step.
    """
    for step_idx, action in enumerate(actions):
        delta_pose = action[:6]
        # NORA gripper: ~0 = closed, ~1 = open
        gripper_openness = float(np.clip(action[6], 0.0, 1.0))

        # Read current joint angles (excludes gripper)
        all_positions = driver.get_positions()
        current_joints = np.array(all_positions[:6])

        # FK: current joints -> current Cartesian pose
        current_cart = joint_angles_to_cartesian(chain, current_joints)

        # Apply NORA delta
        target_cart = current_cart + delta_pose

        # IK: target Cartesian -> target joint angles
        target_joints = cartesian_to_joint_angles(chain, target_cart, current_joints)

        print(f"  Step {step_idx}: cart={np.round(current_cart[:3], 4)} + delta={np.round(delta_pose[:3], 4)} -> target={np.round(target_cart[:3], 4)}")
        print(f"           joints: {np.round(target_joints, 4)}, gripper={'open' if gripper_openness > 0.5 else 'close'} ({gripper_openness:.2f})")

        driver.set_arm_positions(
            target_joints.tolist(),
            goal_time,
            True,  # blocking
        )
        set_gripper(driver, gripper_openness)


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

    # Load custom norm stats if available
    custom_stats_path = Path(__file__).parent / "custom_norm_stats.json"
    if custom_stats_path.exists():
        import json
        with open(custom_stats_path) as f:
            custom_stats = json.load(f)
        model.norm_stats.update(custom_stats)
        print(f"Loaded custom norm stats: {list(custom_stats.keys())}")

    # Build IK chain from URDF
    print("Loading IK chain from URDF...")
    chain = build_ik_chain()
    print("IK chain loaded.")

    # Initialize robot
    driver = None
    if not args.dry_run:
        print(f"Connecting to robot at {args.robot_ip}...")
        driver = init_robot(args.robot_ip)
        print("Robot connected.")
    else:
        print("Dry-run mode: no robot commands will be sent.")

    instruction = args.instruction
    print(f"Unnormalization key: '{args.unnorm_key}'")
    if instruction:
        print(f"Instruction: '{instruction}'")
    else:
        print("No instruction set. Press 'i' to enter one before sampling.")
    print("Press 's' to sample & execute, 'i' to set instruction, 'r' to reset arm, 'q' to quit")

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
                if driver is not None:
                    reset_arm(driver)
                break
            elif key == ord('r'):
                if driver is not None:
                    reset_arm(driver)
                else:
                    print("Dry-run mode: skipping reset.")
            elif key == ord('i'):
                new_instruction = input("Enter new instruction: ")
                if new_instruction.strip():
                    instruction = new_instruction.strip()
                    print(f"Instruction updated: '{instruction}'")
            elif key == ord('s'):
                if not instruction:
                    print("No instruction set. Press 'i' first.")
                    continue
                # Run YOLO to find marker, annotate image for NORA
                yolo_result = marker_predictor.predict(frame, conf=args.conf, imgsz=args.imgsz)
                nora_frame = frame.copy()
                nora_instruction = instruction

                if len(yolo_result.boxes) > 0:
                    # Get the highest-confidence detection
                    box = yolo_result.boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    conf_val = float(box.conf[0])

                    # Draw a bright circle on the detection to guide NORA's attention
                    cv2.circle(nora_frame, (cx, cy), 30, (0, 255, 0), 3)
                    cv2.circle(nora_frame, (cx, cy), 5, (0, 255, 0), -1)

                    # Add spatial info to instruction
                    h, w = frame.shape[:2]
                    vert = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "center")
                    horiz = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
                    location = f"{vert}-{horiz}" if vert != "center" or horiz != "center" else "center"
                    nora_instruction = f"{instruction} The marker is at the {location} of the image."

                    print(f"YOLO: marker at ({cx}, {cy}), conf={conf_val:.2f}, location={location}")
                else:
                    print("YOLO: no marker detected, using raw image")

                # Convert BGR (OpenCV) to RGB (PIL) for NORA
                rgb_frame = cv2.cvtColor(nora_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                print(f"Running NORA inference with: '{nora_instruction}'")
                normalized_actions = model.sample_actions(
                    pil_image, nora_instruction, num_steps=args.num_steps
                )
                print(f"Predicted actions (normalized), shape: {normalized_actions.shape}")

                # Unnormalize: (1, chunk_length, 7) -> (chunk_length, 7)
                actions = model.unnormalize_action(normalized_actions, args.unnorm_key)
                actions = actions[0]  # remove batch dim
                print(f"Unnormalized actions:\n{np.round(actions, 4)}")

                if driver is not None:
                    print("Executing on robot...")
                    execute_actions(driver, chain, actions, args.goal_time)
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
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction for NORA")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow matching steps")
    parser.add_argument("--unnorm-key", type=str, default=unnorm_key,
                        help="Dataset key for action unnormalization")
    # Robot
    parser.add_argument("--robot-ip", type=str, default=robot_ip,
                        help="Trossen arm IP address")
    parser.add_argument("--goal-time", type=float, default=2.0,
                        help="Seconds per action step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without connecting to the robot")

    args = parser.parse_args()
    main(args)
