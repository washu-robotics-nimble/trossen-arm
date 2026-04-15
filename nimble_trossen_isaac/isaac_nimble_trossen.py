"""
WidowX AI Pick-and-Place with LeRobot Dataset Recording.

Modified from the Trossen Robotics WidowX AI pick-and-place demo. Runs a scripted
pick-and-place task in Isaac Sim and records each frame (EE state, action, wrist
camera image) into a LeRobot dataset for NORA 1.5 fine-tuning.

Currently picks up a blue cube and places it at a target position as the original demo. 
Future workincludes replacing the cube with a marker model and generating writing trajectories.

WARNING:
Currently, wrist camera capture images, which might cause storage issue. 
If so, video format should be considered. 

Pre-requisites:
    1. NVIDIA Isaac Sim 4.x  (provides the isaacsim Python package)
    2. Isaac Lab 0.47+        (pip install isaaclab)
    3. LeRobot 0.3.3          (pip install lerobot==0.3.3)
       Note: after installing lerobot, restore Isaac Sim deps:
           pip install gymnasium==1.2.0 packaging==23.0

Usage:
    ~/isaacsim/python.sh nimble_trossen.py

Output:
    nimble_trossen_isaac/generated_lerobot_datasets_<timestamp>/
        data/       - parquet files (state & action per episode)
        images/     - PNG frames from the wrist camera
        meta/       - info.json, episodes.jsonl, tasks.jsonl
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from isaacsim import SimulationApp

# Must initialize SimulationApp before importing other Isaac Sim modules
simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom, Gf, Sdf  # noqa: E402
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial  # noqa: E402
from isaacsim.core.experimental.objects import Cube  # noqa: E402
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController, GRIPPER_OPEN_POSITION, GRIPPER_CLOSED_POSITION  # noqa: E402

# Default configuration constants
DEFAULT_CUBE_SIZE = np.array([0.05, 0.05, 0.05])
DEFAULT_CUBE_POSITION = np.array([0.35, -0.15, 0.03])
DEFAULT_CUBE_ORIENTATION = np.array([1, 0, 0, 0])
DEFAULT_TARGET_POSITION = np.array([0.35, 0.15, 0.03])
DEFAULT_HOME_POSITION = np.array([0.2, 0.0, 0.3])
DEFAULT_EVENTS_DT = [80, 50, 10, 50, 80, 50, 10, 50, 80]

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15  # Height clearance above objects in meters
APPROACH_OFFSET = np.array([0.0, 0.0, 0.03])  # Vertical offset for approach in meters
DOWNWARD_ORIENTATION = np.array(
    [[0.7071068, 0.0, 0.7071068, 0.0]]
)  # Downward-facing quaternion [w, x, y, z]

# Scene configuration
ROBOT_USD_PATH = "./robots/wxai/wxai_base.usd"
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"
CUBE_SCENE_PATH = "/World/Cube"
WRIST_CAMERA_PATH = ROBOT_SCENE_PATH + "/link_6/wrist_camera"

# Robot controller configuration
WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044, 0.044]


def quat_to_euler(quat_wxyz):
    """Convert [w,x,y,z] quaternion to [rx,ry,rz] Euler XYZ angles."""
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return Rotation.from_quat(quat_xyzw).as_euler("xyz")


def gripper_to_scalar(joint_val):
    """Map gripper joint value (0.022=closed, 0.044=open) to 0.0-1.0 scalar."""
    return float(np.clip(
        (joint_val - GRIPPER_CLOSED_POSITION) / (GRIPPER_OPEN_POSITION - GRIPPER_CLOSED_POSITION),
        0.0, 1.0,
    ))


class DataRecorder:
    """Records simulation frames into a LeRobot dataset for policy training."""

    def __init__(
        self,
        repo_id="anling/wxai_pick_place",
        fps=30,
        task_description="Pick up the blue cube and place it at the target location.",
    ):
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
            "observation.images.scene": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
        }
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=f"generated_lerobot_datasets_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_videos=False,
            image_writer_processes=0,
            image_writer_threads=0,
        )
        self.task_description = task_description

    def record_frame(self, robot, camera, goal_position, goal_orientation, gripper_open):
        """Record a single frame of state, action, and camera image."""
        joint_positions, ee_pos, ee_orient = robot.get_current_state()

        # State: [x, y, z, rx, ry, rz, gripper]
        state_euler = quat_to_euler(ee_orient[0])
        gripper_val = gripper_to_scalar(joint_positions[0, WXAI_GRIPPER_DOF_INDEX])
        state = np.concatenate([ee_pos[0], state_euler, [gripper_val]]).astype(np.float32)

        # Action: [x, y, z, rx, ry, rz, gripper]
        action_euler = quat_to_euler(goal_orientation)
        action_gripper = 1.0 if gripper_open else 0.0
        action = np.concatenate([goal_position, action_euler, [action_gripper]]).astype(np.float32)

        # Image: (224, 224, 3) uint8
        rgb = camera.get_rgb()
        if rgb is None or rgb.ndim != 3:
            return
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        image = np.ascontiguousarray(rgb)

        frame = {
            "observation.state": state,
            "observation.images.scene": image,
            "action": action,
        }
        self.dataset.add_frame(frame, task=self.task_description)

    def save_episode(self):
        """Save the current episode to disk."""
        self.dataset.save_episode()
        print("Episode saved to dataset.")


class WXAIPickPlace:
    """Pick-and-place task with trajectory-based motion control."""

    def __init__(
        self,
        events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        cube_size: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
    ):
        """Initialize pick-and-place task.

        Args:
            events_dt: List of time deltas for events in the task sequence.
            cube_initial_position: Initial position [x, y, z] of the cube in meters.
            cube_initial_orientation: Initial orientation quaternion [w, x, y, z] of the cube.
            cube_size: Size of the cube [width, height, depth] in meters.
            target_position: Target position [x, y, z] for placing the cube in meters.
        """
        self.cube_size = cube_size if cube_size is not None else DEFAULT_CUBE_SIZE
        self.cube_initial_position = (
            cube_initial_position
            if cube_initial_position is not None
            else DEFAULT_CUBE_POSITION
        )
        self.cube_initial_orientation = (
            cube_initial_orientation
            if cube_initial_orientation is not None
            else DEFAULT_CUBE_ORIENTATION
        )
        self.target_position = (
            target_position if target_position is not None else DEFAULT_TARGET_POSITION
        )

        self.events_dt = events_dt if events_dt is not None else DEFAULT_EVENTS_DT

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET
        self.home_position = DEFAULT_HOME_POSITION

        self.cube = None
        self.robot = None
        self.camera = None
        self.trajectory = None
        self.trajectory_index = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        stage_utils.create_new_stage(template="sunlight")

        # Spawn robot in scene
        stage_utils.add_reference_to_stage(
            usd_path=ROBOT_USD_PATH,
            path=ROBOT_SCENE_PATH,
        )

        self.robot = TrossenAIController(
            robot_path=ROBOT_SCENE_PATH,
            robot_type=RobotType.WXAI,
            arm_dof_indices=WXAI_ARM_DOF_INDICES,
            gripper_dof_index=WXAI_GRIPPER_DOF_INDEX,
            default_dof_positions=WXAI_DEFAULT_DOF_POSITIONS,
        )

        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Grid/default_environment.usd",
            path=GROUND_SCENE_PATH,
        )

        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

        cube_shape = Cube(
            paths=CUBE_SCENE_PATH,
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=[1.0],
            scales=self.cube_size,
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

        # Add wrist camera on top of the gripper (eye-in-hand)
        self.camera = Camera(
            prim_path=WRIST_CAMERA_PATH,
            resolution=(224, 224),
            frequency=30,
        )
        # Set translation and rotation directly on the prim attributes
        camera_prim = self.camera.prim
        camera_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.01612, -0.01109, 0.20453))
        camera_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.5, 0.5, -0.5, -0.5))
        self.camera.set_focal_length(2.4)
        self.camera.set_clipping_range(0.001, 1000000.0)
        self.camera.initialize()
        print(f"Wrist camera created at {WRIST_CAMERA_PATH}")

    def forward(self, recorder=None) -> bool:
        """Execute one simulation step of the pick-and-place trajectory.

        Args:
            recorder: Optional DataRecorder to log each frame.

        Returns:
            bool: True if trajectory is in progress, False if complete.
        """
        if self.is_done():
            return False

        if self.trajectory is None:
            self.generate_pick_place_trajectory()

        if self.trajectory_index < len(self.trajectory):
            goal_position, goal_orientation, _ = self.trajectory[self.trajectory_index]

            self.robot.set_end_effector_pose(
                position=goal_position.reshape(1, -1),
                orientation=goal_orientation.reshape(1, -1),
            )

            self.trajectory_index += 1

            phase_boundaries = [0]
            cumulative = 0
            for duration in self.events_dt:
                cumulative += duration
                phase_boundaries.append(cumulative)

            if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                self.robot.close_gripper()
            elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                self.robot.open_gripper()

            # Determine gripper command for action recording
            if self.trajectory_index >= phase_boundaries[6]:
                gripper_open = True
            elif self.trajectory_index >= phase_boundaries[2]:
                gripper_open = False
            else:
                gripper_open = True

            if recorder is not None:
                recorder.record_frame(
                    self.robot, self.camera, goal_position, goal_orientation, gripper_open
                )

        return True

    def is_done(self) -> bool:
        """Check if pick-and-place task is complete.

        Returns:
            bool: True if all trajectory waypoints have been executed.
        """
        return self.trajectory is not None and self.trajectory_index >= len(
            self.trajectory
        )

    def reset(
        self,
        cube_position: np.ndarray | None = None,
        cube_orientation: np.ndarray | None = None,
    ) -> None:
        """Reset task to initial state."""
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)

    def reset_robot(self) -> None:
        """Reset robot to default pose and clear trajectory."""
        if self.robot is None:
            raise RuntimeError("Cannot reset robot: robot not initialized.")

        self.robot.reset_to_default_pose()
        self.trajectory = None
        self.trajectory_index = 0

    def reset_cube(
        self, position: np.ndarray | None = None, orientation: np.ndarray | None = None
    ) -> None:
        """Reset cube to specified or initial pose."""
        if self.cube is None:
            raise RuntimeError("Cannot reset cube: cube not initialized.")

        reset_position = (
            position if position is not None else self.cube_initial_position
        )
        reset_orientation = (
            orientation if orientation is not None else self.cube_initial_orientation
        )
        self.cube.set_world_poses(
            positions=reset_position.reshape(1, -1),
            orientations=reset_orientation.reshape(1, -1),
        )

    def make_trajectory(
        self,
        key_frames: list[np.ndarray],
        orientations: list[np.ndarray],
        dt: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Generate smooth trajectory via linear interpolation between keyframes.

        Args:
            key_frames: Position waypoints [x, y, z] in meters. Length must be len(dt) + 1.
            orientations: Orientation quaternions [w, x, y, z] for each keyframe.
            dt: Duration in steps for each trajectory segment.

        Returns:
            List of (position, orientation, cumulative_step) tuples.

        Raises:
            ValueError: If array lengths are incompatible.
        """
        if len(key_frames) != len(dt) + 1:
            raise ValueError(f"Expected {len(dt) + 1} keyframes for {len(dt)} segments")
        if len(orientations) != len(key_frames):
            raise ValueError("Orientations must match keyframe count")

        trajectory = []
        cumulative_step = 0

        for i in range(len(dt)):
            start_pos = np.array(key_frames[i], dtype=np.float64)
            end_pos = np.array(key_frames[i + 1], dtype=np.float64)
            start_ori = np.array(orientations[i], dtype=np.float64)
            n_steps = dt[i]

            # Linear interpolation for each step in this segment
            for step in range(n_steps):
                alpha = step / n_steps if n_steps > 0 else 0.0
                interpolated_pos = start_pos + alpha * (end_pos - start_pos)
                trajectory.append((interpolated_pos, start_ori, cumulative_step + step))

            cumulative_step += n_steps

        trajectory.append(
            (
                np.array(key_frames[-1], dtype=np.float64),
                np.array(orientations[-1], dtype=np.float64),
                cumulative_step,
            )
        )

        return trajectory

    def generate_pick_place_trajectory(self) -> None:
        """Generate complete pick-and-place trajectory from current state.

        Creates a 9-phase trajectory with smooth linear interpolation:
        1. Move to pre-pick position above cube
        2. Descend to pick approach height
        3. Close gripper
        4. Lift cube with clearance
        5. Move to pre-place position above target
        6. Descend to place approach height
        7. Open gripper
        8. Retreat with clearance
        9. Return to home position
        """
        cube_pos = self.cube.get_world_poses()[0].numpy().flatten()
        _, current_ee_pos, _ = self.robot.get_current_state()
        current_ee_pos = current_ee_pos[0]
        key_frames = [
            current_ee_pos,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            cube_pos + self.approach_offset,
            cube_pos + self.approach_offset,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.target_position + np.array([0.0, 0.0, self.clearance_height]),
            self.target_position + self.approach_offset,
            self.target_position + self.approach_offset,
            self.target_position + np.array([0.0, 0.0, self.clearance_height]),
            self.home_position,
        ]

        goal_orientation = DOWNWARD_ORIENTATION[0]
        orientations = [goal_orientation for _ in key_frames]

        self.trajectory = self.make_trajectory(key_frames, orientations, self.events_dt)
        self.trajectory_index = 0


def main():
    print("WidowX AI Pick-and-Place Demo")
    simulation_app.update()

    pick_place = WXAIPickPlace()
    pick_place.setup_scene()

    recorder = DataRecorder()

    # omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    task_completed = False
    needs_reset = True

    print("Press PLAY to start. After completion, press STOP then PLAY to replay.")

    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            if needs_reset:
                pick_place.reset()
                needs_reset = False

            if not task_completed:
                pick_place.forward(recorder=recorder)

            if pick_place.is_done() and not task_completed:
                print("Task complete. Press STOP then PLAY to replay.")
                recorder.save_episode()
                task_completed = True
        else:
            if task_completed:
                needs_reset = True
                task_completed = False

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping pick and place demo...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
