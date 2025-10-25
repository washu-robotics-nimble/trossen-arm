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
Object Detection Module for Trossen Arm

This module provides functionality to:
1. Detect ArUco markers in a camera feed
2. Detect a whiteboard in the scene
3. Create a coordinate grid system on the whiteboard
4. Transform detected marker positions to whiteboard coordinates
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, List

# Optional Roboflow import for ML-based whiteboard detection
try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Roboflow not installed. Using color-based whiteboard detection.")
    print("To enable ML detection: pip install roboflow")


class ObjectDetector:
    """Detects markers and whiteboard, and provides coordinate transformations."""
    
    def __init__(self, camera_index: int = 0, use_roboflow: bool = True, 
                 roboflow_api_key: str = "mSYCb2E062UlCPoLQOjO"):
        """
        Initialize the object detector.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
            use_roboflow: Use Roboflow ML model for whiteboard detection (default: True)
            roboflow_api_key: Roboflow API key for whiteboard detection model
        """
        self.camera = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        
        # ArUco marker detection setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Improve marker detection parameters
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Roboflow whiteboard detection
        self.use_roboflow = use_roboflow and ROBOFLOW_AVAILABLE
        self.roboflow_model = None
        
        if self.use_roboflow:
            try:
                print("Loading Roboflow whiteboard detection model...")
                rf = Roboflow(api_key=roboflow_api_key)
                project = rf.workspace("paul-artushkov").project("whiteboards-detection")
                version = project.version(15)
                self.roboflow_model = version.model
                print("✓ Roboflow model loaded successfully!")
            except Exception as e:
                print(f"⚠ Failed to load Roboflow model: {e}")
                print("  Falling back to color-based detection.")
                self.use_roboflow = False
                self.roboflow_model = None
        
        # Whiteboard properties
        self.whiteboard_corners = None
        self.grid_size = (10, 10)  # 10x10 grid by default
        self.whiteboard_width = 1.0  # 1 meter width by default
        self.whiteboard_height = 1.0  # 1 meter height by default
        self.auto_detect_whiteboard = True  # Continuously detect whiteboard
        self.whiteboard_detection_confidence = 0  # Track detection stability
        
    def __del__(self):
        """Release camera resources."""
        if self.camera.isOpened():
            self.camera.release()
    
    def _detect_whiteboard_roboflow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect whiteboard using Roboflow ML model.
        
        Args:
            frame: Input image frame
            
        Returns:
            Four corners of the whiteboard or None if not detected
        """
        if not self.use_roboflow or self.roboflow_model is None:
            return None
        
        try:
            # Save frame temporarily for Roboflow
            temp_path = "/tmp/temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Run prediction
            prediction = self.roboflow_model.predict(temp_path, confidence=40).json()
            
            # Check if whiteboard detected
            if 'predictions' in prediction and len(prediction['predictions']) > 0:
                # Get the highest confidence detection
                best_detection = max(prediction['predictions'], 
                                    key=lambda x: x.get('confidence', 0))
                
                # Extract bounding box
                x = best_detection['x']
                y = best_detection['y']
                width = best_detection['width']
                height = best_detection['height']
                
                # Convert center + size to corners
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)
                
                # Create corner points [TL, TR, BR, BL]
                corners = np.array([
                    [x1, y1],  # Top-left
                    [x2, y1],  # Top-right
                    [x2, y2],  # Bottom-right
                    [x1, y2]   # Bottom-left
                ], dtype=np.float32)
                
                return corners
        except Exception as e:
            print(f"Roboflow detection error: {e}")
            return None
        
        return None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.
        
        Returns:
            The captured frame or None if capture failed
        """
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def detect_whiteboard(self, frame: np.ndarray, 
                          min_area: float = 10000) -> Optional[np.ndarray]:
        """
        Detect a whiteboard in the frame using ML model or color-based detection.
        
        Args:
            frame: Input image frame
            min_area: Minimum contour area to be considered a whiteboard (for color-based method)
            
        Returns:
            Four corners of the whiteboard [top-left, top-right, bottom-right, bottom-left]
            or None if not detected
        """
        # Try Roboflow ML detection first if available
        if self.use_roboflow and self.roboflow_model is not None:
            corners = self._detect_whiteboard_roboflow(frame)
            if corners is not None:
                # Update with smoothing for stability
                if self.whiteboard_corners is not None and self.auto_detect_whiteboard:
                    alpha = 0.7  # Weight for new detection
                    self.whiteboard_corners = alpha * corners + (1 - alpha) * self.whiteboard_corners
                    self.whiteboard_detection_confidence = min(10, self.whiteboard_detection_confidence + 1)
                else:
                    self.whiteboard_corners = corners
                    self.whiteboard_detection_confidence = 1
                
                return self.whiteboard_corners
        
        # Fall back to color-based detection
        # Convert to multiple color spaces for robust detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: White color detection in HSV
        lower_white = np.array([0, 0, 180])  # More lenient threshold
        upper_white = np.array([180, 40, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Method 2: Bright region detection in grayscale
        _, mask_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_white, mask_bright)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Edge detection for better contour finding
        edges = cv2.Canny(mask, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour
        max_area = 0
        best_approx = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if it's roughly rectangular (4 corners)
                if len(approx) == 4:
                    # Additional check: make sure it's reasonably rectangular
                    # by checking aspect ratio and angles
                    if area > max_area:
                        max_area = area
                        best_approx = approx
        
        if best_approx is not None:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = best_approx.reshape(4, 2)
            corners = self._order_points(corners)
            
            # Update with smoothing for stability
            if self.whiteboard_corners is not None and self.auto_detect_whiteboard:
                # Smooth the transition
                alpha = 0.7  # Weight for new detection
                self.whiteboard_corners = alpha * corners + (1 - alpha) * self.whiteboard_corners
                self.whiteboard_detection_confidence = min(10, self.whiteboard_detection_confidence + 1)
            else:
                self.whiteboard_corners = corners
                self.whiteboard_detection_confidence = 1
            
            return self.whiteboard_corners
        else:
            # Decay confidence if not detected
            self.whiteboard_detection_confidence = max(0, self.whiteboard_detection_confidence - 1)
            if self.whiteboard_detection_confidence == 0:
                self.whiteboard_corners = None
        
        return self.whiteboard_corners
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in the order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and diff to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]      # Top-left has smallest sum
        rect[2] = pts[np.argmax(s)]      # Bottom-right has largest sum
        rect[1] = pts[np.argmin(diff)]   # Top-right has smallest difference
        rect[3] = pts[np.argmax(diff)]   # Bottom-left has largest difference
        
        return rect
    
    def detect_markers(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect ArUco markers in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary mapping marker IDs to their center positions [x, y]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection in varying lighting
        gray = cv2.equalizeHist(gray)
        
        # Detect markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        markers = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # Calculate center of marker
                corner = corners[i][0]
                center_x = int(corner[:, 0].mean())
                center_y = int(corner[:, 1].mean())
                markers[marker_id] = np.array([center_x, center_y])
        
        return markers, corners, ids
    
    def pixel_to_whiteboard_coords(self, pixel_pos: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Transform pixel coordinates to whiteboard coordinate system.
        
        Args:
            pixel_pos: [x, y] position in pixel coordinates
            
        Returns:
            (x, y) position in whiteboard coordinates (0-1 range) or None if whiteboard not detected
        """
        if self.whiteboard_corners is None:
            return None
        
        # Define destination points (normalized square)
        dst_points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform
        matrix = cv2.getPerspectiveTransform(
            self.whiteboard_corners.astype(np.float32), 
            dst_points
        )
        
        # Transform point
        point = np.array([[[pixel_pos[0], pixel_pos[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, matrix)
        
        x, y = transformed[0][0]
        return (x, y)
    
    def whiteboard_to_grid_coords(self, wb_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Convert whiteboard coordinates (0-1 range) to grid coordinates.
        
        Args:
            wb_pos: (x, y) position in whiteboard coordinates (0-1 range)
            
        Returns:
            (grid_x, grid_y) grid cell indices
        """
        grid_x = int(wb_pos[0] * self.grid_size[0])
        grid_y = int(wb_pos[1] * self.grid_size[1])
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_size[0] - 1))
        grid_y = max(0, min(grid_y, self.grid_size[1] - 1))
        
        return (grid_x, grid_y)
    
    def draw_whiteboard_grid(self, frame: np.ndarray, show_labels: bool = True) -> np.ndarray:
        """
        Draw a coordinate grid on the detected whiteboard with live projection effect.
        
        Args:
            frame: Input image frame
            show_labels: Whether to show grid coordinate labels
            
        Returns:
            Frame with grid overlay
        """
        if self.whiteboard_corners is None:
            return frame
        
        result = frame.copy()
        
        # Create a semi-transparent overlay for better visibility
        overlay = result.copy()
        
        # Draw whiteboard boundary
        pts = self.whiteboard_corners.astype(np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 4)
        
        # Calculate grid lines
        corners = self.whiteboard_corners
        
        # Draw vertical lines
        for i in range(self.grid_size[0] + 1):
            t = i / self.grid_size[0]
            # Interpolate along top and bottom edges
            top = corners[0] + t * (corners[1] - corners[0])
            bottom = corners[3] + t * (corners[2] - corners[3])
            
            # Use thicker lines for major grid lines (every 5th line)
            thickness = 3 if i % 5 == 0 else 1
            color = (0, 255, 255) if i % 5 == 0 else (0, 255, 0)
            
            cv2.line(overlay, tuple(top.astype(int)), tuple(bottom.astype(int)), 
                    color, thickness)
        
        # Draw horizontal lines
        for i in range(self.grid_size[1] + 1):
            t = i / self.grid_size[1]
            # Interpolate along left and right edges
            left = corners[0] + t * (corners[3] - corners[0])
            right = corners[1] + t * (corners[2] - corners[1])
            
            # Use thicker lines for major grid lines (every 5th line)
            thickness = 3 if i % 5 == 0 else 1
            color = (0, 255, 255) if i % 5 == 0 else (0, 255, 0)
            
            cv2.line(overlay, tuple(left.astype(int)), tuple(right.astype(int)), 
                    color, thickness)
        
        # Add grid labels if requested
        if show_labels:
            # X-axis labels (top)
            for i in range(0, self.grid_size[0] + 1, 2):  # Every other label to avoid clutter
                t = i / self.grid_size[0]
                pos = corners[0] + t * (corners[1] - corners[0])
                label_pos = tuple((pos - np.array([0, 20])).astype(int))
                cv2.putText(overlay, str(i), label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                # Add background for better readability
                (text_width, text_height), _ = cv2.getTextSize(str(i), 
                                                                cv2.FONT_HERSHEY_SIMPLEX, 
                                                                0.6, 2)
                cv2.rectangle(overlay, 
                            (label_pos[0] - 2, label_pos[1] - text_height - 2),
                            (label_pos[0] + text_width + 2, label_pos[1] + 2),
                            (0, 0, 0), -1)
                cv2.putText(overlay, str(i), label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Y-axis labels (left)
            for i in range(0, self.grid_size[1] + 1, 2):  # Every other label to avoid clutter
                t = i / self.grid_size[1]
                pos = corners[0] + t * (corners[3] - corners[0])
                label_pos = tuple((pos - np.array([40, -5])).astype(int))
                (text_width, text_height), _ = cv2.getTextSize(str(i), 
                                                                cv2.FONT_HERSHEY_SIMPLEX, 
                                                                0.6, 2)
                cv2.rectangle(overlay, 
                            (label_pos[0] - 2, label_pos[1] - text_height - 2),
                            (label_pos[0] + text_width + 2, label_pos[1] + 2),
                            (0, 0, 0), -1)
                cv2.putText(overlay, str(i), label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Blend overlay with original image for semi-transparent effect
        alpha = 0.6
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # Add corner markers for visual reference with coordinates
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        corner_colors = [(255, 0, 255), (255, 100, 0), (0, 255, 255), (100, 255, 0)]
        
        for i, corner in enumerate(corners):
            # Draw corner circle
            cv2.circle(result, tuple(corner.astype(int)), 10, corner_colors[i], -1)
            cv2.circle(result, tuple(corner.astype(int)), 12, (255, 255, 255), 2)
            
            # Draw corner label
            label = corner_labels[i]
            label_offset = np.array([15, -15]) if i < 2 else np.array([15, 25])
            label_pos = tuple((corner + label_offset).astype(int))
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                                            0.6, 2)
            cv2.rectangle(result, 
                        (label_pos[0] - 2, label_pos[1] - text_height - 2),
                        (label_pos[0] + text_width + 2, label_pos[1] + 2),
                        (0, 0, 0), -1)
            cv2.putText(result, label, label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, corner_colors[i], 2)
            
            # Draw corner coordinates
            coord_text = f"({int(corner[0])}, {int(corner[1])})"
            coord_offset = np.array([15, 0]) if i < 2 else np.array([15, 40])
            coord_pos = tuple((corner + coord_offset).astype(int))
            
            # Draw coordinates background
            (text_width, text_height), _ = cv2.getTextSize(coord_text, 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                                            0.45, 1)
            cv2.rectangle(result, 
                        (coord_pos[0] - 2, coord_pos[1] - text_height - 2),
                        (coord_pos[0] + text_width + 2, coord_pos[1] + 2),
                        (0, 0, 0), -1)
            cv2.putText(result, coord_text, coord_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return result
    
    def draw_markers(self, frame: np.ndarray, markers: Dict[int, np.ndarray], 
                    corners=None, ids=None) -> np.ndarray:
        """
        Draw detected markers on the frame with their grid coordinates.
        
        Args:
            frame: Input image frame
            markers: Dictionary of marker IDs to positions
            corners: ArUco corners (optional, for drawing full marker outline)
            ids: ArUco IDs (optional, for drawing full marker outline)
            
        Returns:
            Frame with markers drawn
        """
        result = frame.copy()
        
        # Draw ArUco marker outlines if available
        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(result, corners, ids, (0, 255, 255))
        
        for marker_id, position in markers.items():
            # Draw marker center position with larger circle
            cv2.circle(result, tuple(position.astype(int)), 12, (0, 0, 255), -1)
            cv2.circle(result, tuple(position.astype(int)), 15, (255, 255, 255), 2)
            
            # Create background for text
            text = f"ID: {marker_id}"
            (text_width, text_height), _ = cv2.getTextSize(text, 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                                            0.7, 2)
            text_pos = tuple(position.astype(int) + np.array([20, 5]))
            cv2.rectangle(result, 
                        (text_pos[0] - 2, text_pos[1] - text_height - 2),
                        (text_pos[0] + text_width + 2, text_pos[1] + 5),
                        (0, 0, 0), -1)
            cv2.putText(result, text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert to whiteboard coordinates and display
            wb_coords = self.pixel_to_whiteboard_coords(position)
            if wb_coords is not None:
                grid_coords = self.whiteboard_to_grid_coords(wb_coords)
                coord_text = f"Grid: ({grid_coords[0]}, {grid_coords[1]})"
                wb_text = f"WB: ({wb_coords[0]:.2f}, {wb_coords[1]:.2f})"
                
                # Draw grid coordinates
                (text_width, text_height), _ = cv2.getTextSize(coord_text, 
                                                                cv2.FONT_HERSHEY_SIMPLEX, 
                                                                0.6, 2)
                text_pos2 = tuple(position.astype(int) + np.array([20, 28]))
                cv2.rectangle(result, 
                            (text_pos2[0] - 2, text_pos2[1] - text_height - 2),
                            (text_pos2[0] + text_width + 2, text_pos2[1] + 5),
                            (0, 0, 0), -1)
                cv2.putText(result, coord_text, text_pos2,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw whiteboard coordinates
                (text_width, text_height), _ = cv2.getTextSize(wb_text, 
                                                                cv2.FONT_HERSHEY_SIMPLEX, 
                                                                0.5, 1)
                text_pos3 = tuple(position.astype(int) + np.array([20, 48]))
                cv2.rectangle(result, 
                            (text_pos3[0] - 2, text_pos3[1] - text_height - 2),
                            (text_pos3[0] + text_width + 2, text_pos3[1] + 5),
                            (0, 0, 0), -1)
                cv2.putText(result, wb_text, text_pos3,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        return result
    
    def set_grid_size(self, rows: int, cols: int):
        """Set the grid size for the whiteboard."""
        self.grid_size = (cols, rows)
    
    def set_whiteboard_dimensions(self, width: float, height: float):
        """
        Set physical dimensions of the whiteboard in meters.
        
        Args:
            width: Width in meters
            height: Height in meters
        """
        self.whiteboard_width = width
        self.whiteboard_height = height


def main():
    """Demo program showing marker and whiteboard detection with live grid projection."""
    print("=" * 60)
    print("Object Detection Demo - Markers & Whiteboard")
    print("=" * 60)
    print("\nControls:")
    print("  'q' - Quit the program")
    print("  'c' - Manual calibration/detect whiteboard")
    print("  'a' - Toggle auto-detect whiteboard (ON by default)")
    print("  'g' - Toggle grid labels")
    print("  's' - Save current frame")
    print("  '+' - Increase grid size")
    print("  '-' - Decrease grid size")
    print("\nTips:")
    print("  - Place ArUco markers (DICT_4X4_50) in view")
    print("  - Ensure whiteboard is well-lit and visible")
    print("  - Grid will auto-project once whiteboard is detected")
    print("=" * 60)
    
    detector = ObjectDetector(camera_index=0, use_roboflow=True)
    detector.set_grid_size(10, 10)
    
    # Show which detection method is active
    if detector.use_roboflow:
        print("\n✓ Using Roboflow ML model for whiteboard detection")
    else:
        print("\n⚠ Using color-based whiteboard detection")
    
    show_labels = True
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    # Check if camera opened successfully
    if not detector.camera.isOpened():
        print("\nERROR: Could not open camera!")
        print("Please check:")
        print("  1. Camera is connected")
        print("  2. Camera permissions are granted")
        print("  3. No other application is using the camera")
        return
    
    print("\nCamera initialized successfully!")
    print("Starting detection loop...\n")
    
    while True:
        frame = detector.capture_frame()
        if frame is None:
            print("Failed to capture frame")
            break
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Auto-detect whiteboard if enabled
        if detector.auto_detect_whiteboard:
            detector.detect_whiteboard(frame)
        
        # Detect markers
        markers, corners, ids = detector.detect_markers(frame)
        
        # Draw visualizations
        result = frame.copy()
        
        # Draw whiteboard grid if detected (with live projection effect)
        if detector.whiteboard_corners is not None:
            result = detector.draw_whiteboard_grid(result, show_labels=show_labels)
        
        # Draw markers
        result = detector.draw_markers(result, markers, corners, ids)
        
        # Create info panel
        info_y = 25
        line_height = 25
        
        # Status information with background
        detection_method = "ML (Roboflow)" if detector.use_roboflow else "Color-based"
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Markers detected: {len(markers)}",
            f"Whiteboard: {'DETECTED' if detector.whiteboard_corners is not None else 'NOT DETECTED'}",
            f"Detection: {detection_method}",
            f"Auto-detect: {'ON' if detector.auto_detect_whiteboard else 'OFF'}",
            f"Grid size: {detector.grid_size[0]}x{detector.grid_size[1]}",
            f"Labels: {'ON' if show_labels else 'OFF'}"
        ]
        
        # Draw semi-transparent background for info panel
        overlay = result.copy()
        cv2.rectangle(overlay, (5, 5), (400, 5 + len(status_lines) * line_height + 15), 
                     (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)
        
        # Draw status text
        for i, line in enumerate(status_lines):
            color = (0, 255, 0) if "DETECTED" in line else (255, 255, 255)
            if "NOT DETECTED" in line:
                color = (0, 165, 255)  # Orange
            cv2.putText(result, line, (10, info_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show detected marker IDs
        if markers:
            marker_ids_text = f"Marker IDs: {', '.join(map(str, sorted(markers.keys())))}"
            cv2.putText(result, marker_ids_text, (10, result.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display help text at bottom
        help_text = "Press 'c' to calibrate | 'a' auto-detect | 'g' labels | '+/-' grid size | 'q' quit"
        cv2.putText(result, help_text, (10, result.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show result
        cv2.imshow('Object Detection - Live Grid Projection', result)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('c'):
            print("Manual whiteboard calibration...")
            corners = detector.detect_whiteboard(frame)
            if corners is not None:
                print(f"✓ Whiteboard detected! Corners:\n{corners}")
            else:
                print("✗ Whiteboard not detected. Adjust lighting or camera position.")
        elif key == ord('a'):
            detector.auto_detect_whiteboard = not detector.auto_detect_whiteboard
            status = "ON" if detector.auto_detect_whiteboard else "OFF"
            print(f"Auto-detect whiteboard: {status}")
        elif key == ord('g'):
            show_labels = not show_labels
            print(f"Grid labels: {'ON' if show_labels else 'OFF'}")
        elif key == ord('s'):
            filename = f"detection_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, result)
            print(f"✓ Frame saved as: {filename}")
        elif key == ord('+') or key == ord('='):
            new_size = min(20, detector.grid_size[0] + 1)
            detector.set_grid_size(new_size, new_size)
            print(f"Grid size: {new_size}x{new_size}")
        elif key == ord('-') or key == ord('_'):
            new_size = max(5, detector.grid_size[0] - 1)
            detector.set_grid_size(new_size, new_size)
            print(f"Grid size: {new_size}x{new_size}")
    
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("Demo finished. Thank you!")
    print("=" * 60)


if __name__ == '__main__':
    main()
