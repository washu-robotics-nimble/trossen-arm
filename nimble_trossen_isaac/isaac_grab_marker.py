"""
WidowX AI Marker Pickup with LeRobot Dataset Recording.

Scripted marker pickup task: the robot picks up a marker (cylinder) from a random
position on the ground and moves it to a ready position in front of a whiteboard.
Records each frame (EE state, action, wrist camera image) into a LeRobot dataset
for NORA 1.5 fine-tuning.

The whiteboard is a static surface; the marker is a rigid-body cylinder.
Writing trajectories will be added in a future step.

Pre-requisites:
    1. NVIDIA Isaac Sim 4.x  (provides the isaacsim Python package)
    2. Isaac Lab 0.47+        (pip install isaaclab)
    3. LeRobot 0.3.3          (pip install lerobot==0.3.3)
       Note: after installing lerobot, restore Isaac Sim deps:
           pip install gymnasium==1.2.0 packaging==23.0

Usage:
    ~/isaacsim/python.sh whiteboard_marker.py

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
from isaacsim.core.experimental.objects import Cube, Cylinder  # noqa: E402
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

sys.path.append(os.path.dirname(__file__))
from controller import RobotType, TrossenAIController, GRIPPER_OPEN_POSITION, GRIPPER_CLOSED_POSITION  # noqa: E402

# Marker configuration (cylinder: 12mm diameter, 140mm tall)
MARKER_RADIUS = 0.01
MARKER_HEIGHT = 0.127
GRIPPER_MARKER_POSITION = 0.4 * MARKER_RADIUS  # tighter grip for thin marker (vs 0.022 default)
GRIPPER_CLOSE_STEPS = 20  # number of steps to fully close gripper
MARKER_DEFAULT_POSITION = np.array([0.45, 0.0, MARKER_HEIGHT / 2])  # standing upright
MARKER_ORIENTATION = np.array([1, 0, 0, 0])

# Whiteboard configuration (thin slab: 1cm thick, 50cm wide, 40cm tall)
WHITEBOARD_SIZE = np.array([0.01, 0.5, 0.4])
WHITEBOARD_POSITION = np.array([0.8, 0.0, 0.20])
WHITEBOARD_ORIENTATION = np.array([1, 0, 0, 0])

# Task positions
READY_POSITION = np.array([0.59, 0.0, 0.25])  # in front of whiteboard
HOME_POSITION = np.array([0.2, 0.0, 0.3])

# Trajectory: 6 phases [above, lower behind, slide forward, grasp, lift, return]
DEFAULT_EVENTS_DT = [200, 200, 200, 200, 200]

# Trajectory parameters
CLEARANCE_HEIGHT = MARKER_HEIGHT * 1.5
APPROACH_OFFSET = np.array([-3*MARKER_RADIUS, 0.0, 0.25 * MARKER_HEIGHT])  # overshoot 5cm so marker sits between fingers, not at tip
SIDE_APPROACH_OFFSET = np.array([-0.10, 0.0, 0.0])  # 10cm behind marker (toward robot)
DOWNWARD_ORIENTATION = np.array(
    [[0.7071068, 0.0, 0.7071068, 0.0]]
)  # Downward-facing quaternion [w, x, y, z]
HORIZONTAL_ORIENTATION = np.array(
    [[1.0, 0.0, 0.0, 0.0]]
)  # Horizontal, gripper pointing +X (forward)

# Marker randomization workspace bounds
MARKER_X_RANGE = (0.35, 0.55)  # within arm reach (0.70m max)
MARKER_Y_RANGE = (-0.15, 0.15)

# Scene configuration
ROBOT_USD_PATH = os.path.join(os.path.dirname(__file__), "robots/wxai/wxai_base.usd")
ROBOT_SCENE_PATH = "/World/wxai_robot"
GROUND_SCENE_PATH = "/World/ground"
MARKER_SCENE_PATH = "/World/Marker"
WHITEBOARD_SCENE_PATH = "/World/Whiteboard"
WRIST_CAMERA_PATH = ROBOT_SCENE_PATH + "/link_6/wrist_camera"

# Robot controller configuration
WXAI_ARM_DOF_INDICES = [0, 1, 2, 3, 4, 5]
WXAI_GRIPPER_DOF_INDEX = 6
WXAI_DEFAULT_DOF_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_MARKER_POSITION, GRIPPER_MARKER_POSITION]  # gripper closed


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
        repo_id="anling/wxai_marker_pickup",
        fps=30,
        task_description="Pick up the marker and move to ready position in front of the whiteboard",
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


class WXAIMarkerPickup:
    """Marker pickup task: pick up marker from ground, move to ready position at whiteboard."""

    def __init__(self, events_dt: list[int] | None = None):
        self.events_dt = events_dt if events_dt is not None else DEFAULT_EVENTS_DT

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET
        self.home_position = HOME_POSITION
        self.ready_position = READY_POSITION

        self.marker = None
        self.robot = None
        self.camera = None
        self.trajectory = None
        self.trajectory_index = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, marker, whiteboard, and environment."""
        stage_utils.create_new_stage(template="sunlight")

        # Spawn robot
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

        # Ground plane
        stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path()
            + "/Isaac/Environments/Grid/default_environment.usd",
            path=GROUND_SCENE_PATH,
        )

        # Red marker (rigid body cylinder)
        marker_material = PreviewSurfaceMaterial("/Visual_materials/red")
        marker_material.set_input_values("diffuseColor", [1.0, 0.0, 0.0])

        marker_shape = Cylinder(
            paths=MARKER_SCENE_PATH,
            positions=MARKER_DEFAULT_POSITION,
            orientations=MARKER_ORIENTATION,
            radii=[MARKER_RADIUS],
            heights=[MARKER_HEIGHT],
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=marker_shape.paths, apply_collision_apis=True)
        self.marker = RigidPrim(paths=marker_shape.paths)
        marker_shape.apply_visual_materials(marker_material)

        # White whiteboard (static collider, no RigidPrim)
        whiteboard_material = PreviewSurfaceMaterial("/Visual_materials/white")
        whiteboard_material.set_input_values("diffuseColor", [1.0, 1.0, 1.0])

        whiteboard_shape = Cube(
            paths=WHITEBOARD_SCENE_PATH,
            positions=WHITEBOARD_POSITION,
            orientations=WHITEBOARD_ORIENTATION,
            sizes=[1.0],
            scales=WHITEBOARD_SIZE,
            reset_xform_op_properties=True,
        )

        GeomPrim(paths=whiteboard_shape.paths, apply_collision_apis=True)
        whiteboard_shape.apply_visual_materials(whiteboard_material)

        # Wrist camera (eye-in-hand)
        self.camera = Camera(
            prim_path=WRIST_CAMERA_PATH,
            resolution=(224, 224),
            frequency=30,
        )
        camera_prim = self.camera.prim
        camera_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.01612, -0.01109, 0.20453))
        camera_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.5, 0.5, -0.5, -0.5))
        self.camera.set_focal_length(2.4)
        self.camera.set_clipping_range(0.001, 1000000.0)
        self.camera.initialize()
        print(f"Wrist camera created at {WRIST_CAMERA_PATH}")

    def forward(self, recorder=None) -> bool:
        """Execute one simulation step of the marker pickup trajectory.

        Returns:
            bool: True if trajectory is in progress, False if complete.
        """
        if self.is_done():
            return False

        if self.trajectory is None:
            self.generate_marker_pickup_trajectory()

        if self.trajectory_index < len(self.trajectory):
            goal_position, goal_orientation, _ = self.trajectory[self.trajectory_index]

            self.robot.set_end_effector_pose(
                position=goal_position.reshape(1, -1),
                orientation=goal_orientation.reshape(1, -1),
            )

            self.trajectory_index += 1

            # Compute phase boundaries: [0, 80, 130, 140, 190, 270]
            phase_boundaries = [0]
            cumulative = 0
            for duration in self.events_dt:
                cumulative += duration
                phase_boundaries.append(cumulative)

            # 5 phases (6 keyframes):
            #   0: init → behind marker   (open)
            #   1: behind → at marker     (open)
            #   2: hold at marker         (close gripper here)
            #   3: lift                   (closed)
            #   4: return to init         (closed)
            if self.trajectory_index < phase_boundaries[2]:
                self.robot.open_gripper()
                gripper_open = True
            elif self.trajectory_index < phase_boundaries[3]:
                # Close gripper over GRIPPER_CLOSE_STEPS, then hold closed
                steps_into_grasp = self.trajectory_index - phase_boundaries[2]
                grasp_progress = min(steps_into_grasp / GRIPPER_CLOSE_STEPS, 1.0)
                grip_pos = GRIPPER_OPEN_POSITION + grasp_progress * (GRIPPER_MARKER_POSITION - GRIPPER_OPEN_POSITION)
                self.robot.set_gripper_position(grip_pos)
                gripper_open = grasp_progress < 0.5
            else:
                self.robot.set_gripper_position(GRIPPER_MARKER_POSITION)
                gripper_open = False

            if recorder is not None:
                recorder.record_frame(
                    self.robot, self.camera, goal_position, goal_orientation, gripper_open
                )

        return True

    def is_done(self) -> bool:
        return self.trajectory is not None and self.trajectory_index >= len(self.trajectory)

    def reset(self) -> None:
        """Reset task: robot to default pose, marker to random position."""
        self.reset_robot()
        self.reset_marker()

    def reset_robot(self) -> None:
        if self.robot is None:
            raise RuntimeError("Cannot reset robot: robot not initialized.")
        self.robot.reset_to_default_pose()
        self.trajectory = None
        self.trajectory_index = 0

    def reset_marker(self) -> None:
        """Randomize marker position within workspace bounds."""
        if self.marker is None:
            raise RuntimeError("Cannot reset marker: marker not initialized.")

        x = np.random.uniform(*MARKER_X_RANGE)
        y = np.random.uniform(*MARKER_Y_RANGE)
        z = MARKER_HEIGHT / 2  # standing upright, center above ground

        position = np.array([x, y, z])
        self.marker.set_world_poses(
            positions=position.reshape(1, -1),
            orientations=MARKER_ORIENTATION.reshape(1, -1),
        )
        print(f"Marker reset to position: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def make_trajectory(
        self,
        key_frames: list[np.ndarray],
        orientations: list[np.ndarray],
        dt: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Generate smooth trajectory via linear interpolation between keyframes."""
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

    def generate_marker_pickup_trajectory(self) -> None:
        """Generate 6-phase marker pickup trajectory with side approach.

        Phases:
        1. Move above marker (clearance height)
        2. Lower to marker height, behind marker (-X offset)
        3. Slide forward horizontally to marker center
        4. Hold & close gripper
        5. Lift marker with clearance
        6. Return to init pose
        """
        marker_pos = self.marker.get_world_poses()[0].numpy().flatten()
        print(f"current marker at {marker_pos}")
        _, current_ee_pos, _ = self.robot.get_current_state()
        current_ee_pos = current_ee_pos[0]

        # behind_pos = marker_pos + self.approach_offset + SIDE_APPROACH_OFFSET
        behind_pos = marker_pos + np.array([0.0, 0.0, self.clearance_height]) + SIDE_APPROACH_OFFSET
        key_frames = [
            current_ee_pos,                                          # start (init pose)
            # marker_pos + np.array([0.0, 0.0, self.clearance_height]),  # above and behind marker
            behind_pos,                                              # behind marker at grip height
            marker_pos + self.approach_offset,                       # at marker center (slide forward)
            marker_pos + self.approach_offset,                       # hold for grasp
            marker_pos + np.array([0.0, 0.0, self.clearance_height]),  # lift
            current_ee_pos,                                          # back to init pose
        ]

        down = DOWNWARD_ORIENTATION[0]
        horiz = HORIZONTAL_ORIENTATION[0]
        orientations = [
            horiz,   # start (init pose)
            horiz,   # above marker
            # horiz,  # behind marker — switch to horizontal
            horiz,  # at marker center (slide forward)
            horiz,  # hold for grasp
            horiz,   # lift
            horiz,   # back to init pose
        ]

        self.trajectory = self.make_trajectory(key_frames, orientations, self.events_dt)
        self.trajectory_index = 0


def main():
    print("WidowX AI Marker Pickup Demo")
    simulation_app.update()

    task = WXAIMarkerPickup()
    task.setup_scene()

    recorder = DataRecorder()

    simulation_app.update()

    task_completed = False
    needs_reset = True

    print("Press PLAY to start. After completion, press STOP then PLAY to replay.")

    while simulation_app.is_running():
        if SimulationManager.is_simulating():
            if needs_reset:
                task.reset()
                needs_reset = False

            if not task_completed:
                task.forward(recorder=recorder)

            if task.is_done() and not task_completed:
                print("Task complete. Resetting to initial pose.")
                task.robot.reset_to_default_pose()
                recorder.save_episode()
                task_completed = True
                print("Press STOP then PLAY to replay.")
        else:
            if task_completed:
                needs_reset = True
                task_completed = False

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping marker pickup demo...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
