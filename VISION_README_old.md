# Object Detection and Whiteboard Coordinate Grid

This module adds computer vision capabilities to the Trossen Arm project, including:
- ArUco marker detection
- Whiteboard detection and tracking
- Coordinate grid system on the whiteboard
- Integration with arm control for vision-guided manipulation

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-contrib-python` for computer vision and ArUco marker detection
- `numpy` for numerical operations
- `trossen_arm` for arm control

## Files

### `object_detection.py`
Core object detection module that provides:
- **`ObjectDetector`** class with the following capabilities:
  - Camera capture and frame processing
  - ArUco marker detection (DICT_4X4_50 dictionary)
  - Whiteboard detection using color-based segmentation
  - Coordinate transformation from pixels to whiteboard grid
  - Visualization with grid overlay

### `arm_with_vision.py`
Integrated example combining vision and arm control:
- Detects whiteboard and creates coordinate grid
- Tracks ArUco markers in real-time
- Maps marker positions to grid coordinates
- (Framework for) controlling arm to move to detected positions

## Usage

### 1. Standalone Object Detection Demo

Run the object detection demo to test marker and whiteboard detection:

```bash
python object_detection.py
```

**Controls:**
- Press `c` to detect/calibrate the whiteboard
- Press `q` to quit

**Tips:**
- Ensure good lighting conditions
- Use a white or light-colored board
- Print ArUco markers from the 4x4_50 dictionary (you can generate them online)
- Adjust the `lower_white` and `upper_white` HSV values in `detect_whiteboard()` if needed

### 2. Integrated Arm Control with Vision

Run the integrated example:

```bash
python arm_with_vision.py
```

**Workflow:**
1. Camera initializes and shows live feed
2. Position camera to see the whiteboard
3. Press `SPACE` to detect the whiteboard
4. Once detected, the system tracks ArUco markers
5. Press `s` to toggle tracking mode
6. Press `q` to quit

**Note:** The inverse kinematics implementation in `position_to_joint_angles()` is a placeholder. You need to implement proper IK based on your arm's kinematics or use the arm's built-in IK functionality if available.

## Coordinate Systems

### Pixel Coordinates
- Origin: Top-left corner of the camera image
- Units: Pixels
- Range: (0, 0) to (image_width, image_height)

### Whiteboard Coordinates
- Origin: Top-left corner of detected whiteboard
- Units: Normalized (0.0 to 1.0)
- Range: (0, 0) at top-left to (1, 1) at bottom-right

### Grid Coordinates
- Origin: Top-left cell
- Units: Grid cells
- Range: (0, 0) to (grid_cols-1, grid_rows-1)
- Default: 10x10 grid

### Arm Workspace Coordinates
- Origin: Arm base
- Units: Meters
- Range: Defined by `workspace_bounds` parameter

## Configuration

### Grid Size
Change the grid resolution:

```python
detector.set_grid_size(rows=10, cols=10)
```

### Whiteboard Detection Parameters

Adjust HSV color range for better whiteboard detection (in `object_detection.py`):

```python
# For whiteboards under different lighting:
lower_white = np.array([0, 0, 200])    # Adjust brightness threshold
upper_white = np.array([180, 30, 255])  # Adjust saturation threshold
```

### Workspace Bounds

Define the physical workspace in `arm_with_vision.py`:

```python
workspace_bounds = (
    (0.2, 0.5),   # X range (min, max) in meters
    (-0.3, 0.3),  # Y range (min, max) in meters  
    0.1           # Z height in meters
)
```

## ArUco Markers

### Generating Markers

You can generate ArUco markers online or using OpenCV:

```python
import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate marker with ID 0
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)
cv2.imwrite('marker_0.png', marker_image)
```

### Printing Markers

- Print markers at 3x3 inches or larger for best detection
- Use high-quality paper and printer
- Ensure markers are flat and well-lit
- Avoid glare and shadows

## Troubleshooting

### Whiteboard Not Detected
- Ensure adequate lighting
- Check that the whiteboard is white or very light colored
- Adjust HSV thresholds in `detect_whiteboard()`
- Increase or decrease `min_area` parameter
- Make sure the whiteboard takes up significant portion of frame

### Markers Not Detected
- Ensure markers are from the DICT_4X4_50 dictionary
- Check that markers are flat and clearly visible
- Improve lighting to reduce shadows
- Move camera closer or use higher resolution markers
- Ensure markers have sufficient white border

### Grid Misalignment
- Recalibrate the whiteboard by pressing `c`
- Ensure whiteboard corners are clearly visible
- Check that the whiteboard is rectangular and flat

## API Reference

### ObjectDetector Class

#### Methods

**`__init__(camera_index=0)`**
- Initialize the detector with specified camera

**`capture_frame()`**
- Capture a single frame from camera
- Returns: numpy array or None

**`detect_whiteboard(frame, min_area=10000)`**
- Detect whiteboard in frame
- Returns: 4 corner points or None

**`detect_markers(frame)`**
- Detect ArUco markers
- Returns: dict mapping marker IDs to center positions

**`pixel_to_whiteboard_coords(pixel_pos)`**
- Transform pixel coordinates to normalized whiteboard coordinates
- Returns: (x, y) tuple in 0-1 range or None

**`whiteboard_to_grid_coords(wb_pos)`**
- Convert whiteboard coordinates to grid indices
- Returns: (grid_x, grid_y) tuple

**`draw_whiteboard_grid(frame)`**
- Draw grid overlay on frame
- Returns: frame with grid visualization

**`draw_markers(frame, markers)`**
- Draw detected markers with labels
- Returns: frame with marker visualization

**`set_grid_size(rows, cols)`**
- Configure grid dimensions

**`set_whiteboard_dimensions(width, height)`**
- Set physical whiteboard size in meters

## Future Enhancements

- [ ] Implement proper inverse kinematics
- [ ] Add calibration for camera-to-arm transformation
- [ ] Support multiple marker tracking
- [ ] Add depth perception using stereo cameras
- [ ] Implement object grasping based on marker orientation
- [ ] Add gesture recognition for arm control
- [ ] Create GUI for easier calibration and configuration
