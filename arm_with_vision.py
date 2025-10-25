# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Integrated Arm Control with Vision

This script demonstrates how to:
1. Detect ArUco markers and whiteboard using computer vision
2. Map marker positions to a coordinate grid
3. Control the Trossen arm to move to detected marker positions
4. Use the whiteboard as a workspace coordinate system
"""

import numpy as np
import cv2
import time

import trossen_arm
from object_detection import ObjectDetector


def grid_to_arm_position(grid_x: int, grid_y: int, 
                        grid_size: tuple = (10, 10),
                        workspace_bounds: tuple = ((0.2, 0.5), (-0.3, 0.3), 0.1)):
    """
    Convert grid coordinates to arm position in 3D space.
    
    Args:
        grid_x: X grid coordinate
        grid_y: Y grid coordinate  
        grid_size: (cols, rows) size of the grid
        workspace_bounds: ((x_min, x_max), (y_min, y_max), z_height)
            Defines the physical workspace in meters
    
    Returns:
        (x, y, z) position in meters for the arm
    """
    # Normalize grid coordinates to 0-1 range
    norm_x = grid_x / (grid_size[0] - 1)
    norm_y = grid_y / (grid_size[1] - 1)
    
    # Map to physical workspace
    x = workspace_bounds[0][0] + norm_x * (workspace_bounds[0][1] - workspace_bounds[0][0])
    y = workspace_bounds[1][0] + norm_y * (workspace_bounds[1][1] - workspace_bounds[1][0])
    z = workspace_bounds[2]
    
    return (x, y, z)


def position_to_joint_angles(position: tuple, driver: trossen_arm.TrossenArmDriver) -> np.ndarray:
    """
    Convert 3D position to joint angles using inverse kinematics.
    
    Note: This is a simplified placeholder. You should use proper IK or
    the arm's built-in IK if available.
    
    Args:
        position: (x, y, z) target position in meters
        driver: Trossen arm driver instance
        
    Returns:
        Array of joint angles
    """
    # This is a placeholder - implement proper inverse kinematics
    # or use the driver's IK function if available
    x, y, z = position
    
    # Simple example: adjust based on your arm's kinematics
    # You'll need to implement proper IK here
    joint_angles = np.array([
        np.arctan2(y, x),  # Base rotation
        np.pi / 2,          # Shoulder
        np.pi / 2,          # Elbow
        0.0,                # Wrist 1
        0.0,                # Wrist 2
        0.0                 # Wrist 3
    ])
    
    return joint_angles


def main():
    """Main program integrating vision and arm control."""
    print("=" * 60)
    print("Trossen Arm with Vision Integration")
    print("=" * 60)
    
    # Initialize object detector
    print("\n1. Initializing vision system...")
    detector = ObjectDetector(camera_index=0)
    detector.set_grid_size(10, 10)
    
    # Initialize arm driver
    print("2. Initializing arm driver...")
    driver = trossen_arm.TrossenArmDriver()
    
    print("3. Configuring arm driver...")
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_base,
        "192.168.1.2",
        False
    )
    
    # Calibrate whiteboard
    print("\n4. Calibrating whiteboard...")
    print("   Position camera to see the whiteboard clearly.")
    print("   Press SPACE when ready to detect whiteboard, 'q' to quit")
    
    whiteboard_detected = False
    while not whiteboard_detected:
        frame = detector.capture_frame()
        if frame is None:
            print("Failed to capture frame")
            break
        
        # Try to detect whiteboard
        corners = detector.detect_whiteboard(frame)
        
        # Visualize
        result = frame.copy()
        if corners is not None:
            result = detector.draw_whiteboard_grid(result)
            cv2.putText(result, "Whiteboard detected! Press SPACE to continue",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(result, "Move camera to see whiteboard. Press SPACE to try detection.",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Calibration', result)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            corners = detector.detect_whiteboard(frame)
            if corners is not None:
                whiteboard_detected = True
                print("   ✓ Whiteboard detected successfully!")
            else:
                print("   ✗ Whiteboard not detected. Adjust camera and try again.")
        elif key == ord('q'):
            print("   Calibration cancelled.")
            cv2.destroyAllWindows()
            return
    
    # Main control loop
    print("\n5. Starting main control loop...")
    print("   The arm will track detected markers.")
    print("   Press 'q' to quit, 's' to stop tracking")
    
    tracking = False
    target_marker_id = 0  # Track marker with ID 0 by default
    
    # Move arm to home position
    print("   Moving to home position...")
    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_arm_positions(
        np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0]),
        2.0,
        True
    )
    
    while True:
        frame = detector.capture_frame()
        if frame is None:
            print("Failed to capture frame")
            break
        
        # Detect markers
        markers = detector.detect_markers(frame)
        
        # Visualize
        result = detector.draw_whiteboard_grid(frame)
        result = detector.draw_markers(result, markers)
        
        # Track target marker
        if target_marker_id in markers:
            marker_pos = markers[target_marker_id]
            wb_coords = detector.pixel_to_whiteboard_coords(marker_pos)
            
            if wb_coords is not None:
                grid_coords = detector.whiteboard_to_grid_coords(wb_coords)
                
                # Display tracking info
                cv2.putText(result, f"Tracking marker {target_marker_id}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result, f"Grid: ({grid_coords[0]}, {grid_coords[1]})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Move arm if tracking is enabled
                if tracking:
                    # Convert grid coordinates to arm position
                    target_pos = grid_to_arm_position(grid_coords[0], grid_coords[1])
                    print(f"Moving to grid ({grid_coords[0]}, {grid_coords[1]}) -> "
                          f"position ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
                    
                    # Note: You'll need to implement proper IK here
                    # This is just a demonstration
                    # joint_angles = position_to_joint_angles(target_pos, driver)
                    # driver.set_arm_positions(joint_angles, 1.0, False)
        else:
            cv2.putText(result, f"Marker {target_marker_id} not detected",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display control instructions
        status = "TRACKING" if tracking else "PAUSED"
        cv2.putText(result, f"Status: {status} | 's': toggle tracking | 'q': quit",
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        cv2.imshow('Arm Vision Control', result)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tracking = not tracking
            print(f"Tracking {'enabled' if tracking else 'disabled'}")
    
    # Return to home position
    print("\n6. Returning to home position...")
    driver.set_arm_positions(
        np.zeros(driver.get_num_joints() - 1),
        2.0,
        True
    )
    
    cv2.destroyAllWindows()
    print("\n✓ Program completed successfully!")


if __name__ == '__main__':
    main()
