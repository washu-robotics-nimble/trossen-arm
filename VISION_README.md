# Vision System for Trossen Arm# Object Detection and Whiteboard Coordinate Grid



This vision system provides **real-time object detection** for ArUco markers and whiteboards, with **live coordinate grid projection** for precise spatial tracking and robot arm control.This module adds computer vision capabilities to the Trossen Arm project, including:

- ArUco marker detection

## 🎯 Features- Whiteboard detection and tracking

- Coordinate grid system on the whiteboard

- **ArUco Marker Detection**: Detects and tracks multiple ArUco markers (DICT_4X4_50)- Integration with arm control for vision-guided manipulation

- **Whiteboard Detection**: Automatically detects whiteboards in the camera view

- **Live Grid Projection**: Projects a coordinate grid onto the detected whiteboard in real-time## Installation

- **Coordinate Transformation**: Converts pixel coordinates to grid coordinates for robot control

- **Auto-Calibration**: Continuous whiteboard tracking with smooth transitionsInstall the required dependencies:

- **Visual Feedback**: Enhanced visualization with semi-transparent overlays and labels

```bash

## 📋 Requirementspip install -r requirements.txt

```

Make sure you have the required packages installed:

This will install:

```bash- `opencv-contrib-python` for computer vision and ArUco marker detection

pip install -r requirements.txt- `numpy` for numerical operations

```- `trossen_arm` for arm control



Required packages:## Files

- `opencv-python>=4.12.0` - Computer vision library

- `numpy>=2.2.0` - Numerical computing### `object_detection.py`

- `trossen_arm>=1.9.0` - Trossen arm control libraryCore object detection module that provides:

- **`ObjectDetector`** class with the following capabilities:

## 🚀 Quick Start  - Camera capture and frame processing

  - ArUco marker detection (DICT_4X4_50 dictionary)

### 1. Generate ArUco Markers  - Whiteboard detection using color-based segmentation

  - Coordinate transformation from pixels to whiteboard grid

First, generate printable ArUco markers:  - Visualization with grid overlay



```bash### `arm_with_vision.py`

python generate_markers.pyIntegrated example combining vision and arm control:

```- Detects whiteboard and creates coordinate grid

- Tracks ArUco markers in real-time

This creates:- Maps marker positions to grid coordinates

- Individual marker images in `aruco_markers/` folder- (Framework for) controlling arm to move to detected positions

- A printable sheet `aruco_markers_sheet.png` with multiple markers

## Usage

**Print the markers** and place them in your workspace.

### 1. Standalone Object Detection Demo

### 2. Run Object Detection

Run the object detection demo to test marker and whiteboard detection:

Start the object detection system:

```bash

```bashpython object_detection.py

python object_detection.py```

```

**Controls:**

The system will:- Press `c` to detect/calibrate the whiteboard

1. Open your camera- Press `q` to quit

2. Automatically detect the whiteboard

3. Project a coordinate grid onto the whiteboard**Tips:**

4. Detect and track ArUco markers- Ensure good lighting conditions

5. Display marker positions in both pixel and grid coordinates- Use a white or light-colored board

- Print ArUco markers from the 4x4_50 dictionary (you can generate them online)

### 3. Keyboard Controls- Adjust the `lower_white` and `upper_white` HSV values in `detect_whiteboard()` if needed



While the detection window is open:### 2. Integrated Arm Control with Vision



| Key | Action |Run the integrated example:

|-----|--------|

| `q` | Quit the program |```bash

| `c` | Manual whiteboard calibration |python arm_with_vision.py

| `a` | Toggle auto-detect whiteboard (ON by default) |```

| `g` | Toggle grid coordinate labels |

| `s` | Save current frame as image |**Workflow:**

| `+` | Increase grid size |1. Camera initializes and shows live feed

| `-` | Decrease grid size |2. Position camera to see the whiteboard

3. Press `SPACE` to detect the whiteboard

## 📐 Grid Projection System4. Once detected, the system tracks ArUco markers

5. Press `s` to toggle tracking mode

The live grid projection system provides:6. Press `q` to quit



- **10x10 grid** by default (adjustable with +/- keys)**Note:** The inverse kinematics implementation in `position_to_joint_angles()` is a placeholder. You need to implement proper IK based on your arm's kinematics or use the arm's built-in IK functionality if available.

- **Major gridlines** every 5 units (thicker, cyan colored)

- **Minor gridlines** for fine positioning (thin, green colored)## Coordinate Systems

- **Corner markers** (TL, TR, BR, BL) for reference

- **Coordinate labels** on X and Y axes### Pixel Coordinates

- **Semi-transparent overlay** for better visibility- Origin: Top-left corner of the camera image

- Units: Pixels

### Coordinate Systems- Range: (0, 0) to (image_width, image_height)



The system provides three coordinate representations:### Whiteboard Coordinates

- Origin: Top-left corner of detected whiteboard

1. **Pixel Coordinates**: Raw camera coordinates (x, y)- Units: Normalized (0.0 to 1.0)

2. **Whiteboard Coordinates**: Normalized coordinates (0-1 range) on the whiteboard- Range: (0, 0) at top-left to (1, 1) at bottom-right

3. **Grid Coordinates**: Discrete grid cell indices (0-9 for default 10x10 grid)

### Grid Coordinates

Example marker display:- Origin: Top-left cell

```- Units: Grid cells

ID: 3- Range: (0, 0) to (grid_cols-1, grid_rows-1)

Grid: (7, 4)- Default: 10x10 grid

WB: (0.72, 0.45)

```### Arm Workspace Coordinates

- Origin: Arm base

## 🎥 Camera Setup Tips- Units: Meters

- Range: Defined by `workspace_bounds` parameter

For best results:

## Configuration

1. **Lighting**: Ensure good, even lighting on the whiteboard

2. **Position**: Place camera to view entire whiteboard### Grid Size

3. **Focus**: Allow camera to auto-focus on the whiteboardChange the grid resolution:

4. **Markers**: Place ArUco markers with clear visibility

5. **Background**: Use a clean, uncluttered background```python

detector.set_grid_size(rows=10, cols=10)

### Troubleshooting Whiteboard Detection```



If whiteboard is not detected:### Whiteboard Detection Parameters



1. **Adjust lighting** - Avoid glare and shadowsAdjust HSV color range for better whiteboard detection (in `object_detection.py`):

2. **Check color** - Ensure whiteboard is sufficiently white/bright

3. **Clear view** - Remove obstructions in camera view```python

4. **Manual calibration** - Press `c` to force detection# For whiteboards under different lighting:

5. **HSV values** - Adjust `lower_white` and `upper_white` in code if neededlower_white = np.array([0, 0, 200])    # Adjust brightness threshold

upper_white = np.array([180, 30, 255])  # Adjust saturation threshold

## 🤖 Integration with Robot Arm```



Use the coordinate system with your Trossen arm:### Workspace Bounds



```pythonDefine the physical workspace in `arm_with_vision.py`:

from object_detection import ObjectDetector

```python

# Initialize detectorworkspace_bounds = (

detector = ObjectDetector(camera_index=0)    (0.2, 0.5),   # X range (min, max) in meters

    (-0.3, 0.3),  # Y range (min, max) in meters  

# Capture and detect    0.1           # Z height in meters

frame = detector.capture_frame())

markers, corners, ids = detector.detect_markers(frame)```



# Get marker coordinates## ArUco Markers

for marker_id, pixel_pos in markers.items():

    # Convert to whiteboard coordinates### Generating Markers

    wb_coords = detector.pixel_to_whiteboard_coords(pixel_pos)

    You can generate ArUco markers online or using OpenCV:

    # Convert to grid coordinates

    grid_coords = detector.whiteboard_to_grid_coords(wb_coords)```python

    import cv2

    # Use with arm controlimport numpy as np

    print(f"Marker {marker_id} at grid position: {grid_coords}")

    # Move arm to grid_coords...aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

```

# Generate marker with ID 0

See `arm_with_vision.py` for a complete integration example.marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)

cv2.imwrite('marker_0.png', marker_image)

## 📊 API Reference```



### ObjectDetector Class### Printing Markers



#### Methods- Print markers at 3x3 inches or larger for best detection

- Use high-quality paper and printer

**`__init__(camera_index=0)`**- Ensure markers are flat and well-lit

- Initialize the detector with specified camera- Avoid glare and shadows



**`capture_frame()`**## Troubleshooting

- Capture a single frame from camera

- Returns: numpy array or None### Whiteboard Not Detected

- Ensure adequate lighting

**`detect_whiteboard(frame, min_area=10000)`**- Check that the whiteboard is white or very light colored

- Detect whiteboard in frame- Adjust HSV thresholds in `detect_whiteboard()`

- Returns: 4 corner points or None- Increase or decrease `min_area` parameter

- Make sure the whiteboard takes up significant portion of frame

**`detect_markers(frame)`**

- Detect ArUco markers in frame### Markers Not Detected

- Returns: (markers dict, corners, ids)- Ensure markers are from the DICT_4X4_50 dictionary

- Check that markers are flat and clearly visible

**`pixel_to_whiteboard_coords(pixel_pos)`**- Improve lighting to reduce shadows

- Transform pixel to whiteboard coordinates- Move camera closer or use higher resolution markers

- Returns: (x, y) in 0-1 range or None- Ensure markers have sufficient white border



**`whiteboard_to_grid_coords(wb_pos)`**### Grid Misalignment

- Convert whiteboard to grid coordinates- Recalibrate the whiteboard by pressing `c`

- Returns: (grid_x, grid_y)- Ensure whiteboard corners are clearly visible

- Check that the whiteboard is rectangular and flat

**`draw_whiteboard_grid(frame, show_labels=True)`**

- Draw grid overlay on frame## API Reference

- Returns: frame with grid visualization

### ObjectDetector Class

**`draw_markers(frame, markers, corners=None, ids=None)`**

- Draw detected markers with labels#### Methods

- Returns: frame with marker visualization

**`__init__(camera_index=0)`**

**`set_grid_size(rows, cols)`**- Initialize the detector with specified camera

- Set the grid dimensions

**`capture_frame()`**

**`set_whiteboard_dimensions(width, height)`**- Capture a single frame from camera

- Set physical whiteboard size in meters- Returns: numpy array or None



## 🔧 Configuration**`detect_whiteboard(frame, min_area=10000)`**

- Detect whiteboard in frame

### Grid Size- Returns: 4 corner points or None



Adjust grid resolution for your needs:**`detect_markers(frame)`**

- Detect ArUco markers

```python- Returns: dict mapping marker IDs to center positions

detector.set_grid_size(rows=10, cols=10)  # 10x10 grid

```**`pixel_to_whiteboard_coords(pixel_pos)`**

- Transform pixel coordinates to normalized whiteboard coordinates

Or use keyboard shortcuts `+` and `-` during runtime.- Returns: (x, y) tuple in 0-1 range or None



### Camera Settings**`whiteboard_to_grid_coords(wb_pos)`**

- Convert whiteboard coordinates to grid indices

Camera properties are optimized automatically:- Returns: (grid_x, grid_y) tuple

- Resolution: 1280x720

- FPS: 30**`draw_whiteboard_grid(frame)`**

- Autofocus: Enabled- Draw grid overlay on frame

- Returns: frame with grid visualization

Modify in `__init__()` if needed.

**`draw_markers(frame, markers)`**

### Detection Parameters- Draw detected markers with labels

- Returns: frame with marker visualization

ArUco detection parameters are tuned for reliability. Adjust in `__init__()`:

**`set_grid_size(rows, cols)`**

```python- Configure grid dimensions

self.aruco_params.adaptiveThreshWinSizeMin = 3

self.aruco_params.minMarkerPerimeterRate = 0.03**`set_whiteboard_dimensions(width, height)`**

# ... etc- Set physical whiteboard size in meters

```

## Future Enhancements

## 📝 Output Information

- [ ] Implement proper inverse kinematics

The live display shows:- [ ] Add calibration for camera-to-arm transformation

- [ ] Support multiple marker tracking

- **FPS**: Current frame rate- [ ] Add depth perception using stereo cameras

- **Markers detected**: Number of markers found- [ ] Implement object grasping based on marker orientation

- **Whiteboard status**: Detection state- [ ] Add gesture recognition for arm control

- **Auto-detect status**: ON/OFF- [ ] Create GUI for easier calibration and configuration

- **Grid size**: Current grid dimensions
- **Labels status**: ON/OFF
- **Marker details**: ID, grid position, whiteboard position

## 🎨 Customization

### Change Grid Colors

Edit colors in `draw_whiteboard_grid()`:

```python
color = (0, 255, 255)  # Cyan (BGR format)
```

### Adjust Whiteboard Detection

Modify HSV thresholds in `detect_whiteboard()`:

```python
lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 40, 255])
```

### Marker Dictionary

To use different ArUco markers, change in `__init__()`:

```python
self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
```

## 🐛 Troubleshooting

### Camera Not Opening

```
ERROR: Could not open camera!
```

Solutions:
1. Check camera is connected
2. Grant camera permissions (macOS: System Preferences > Security & Privacy)
3. Close other apps using the camera
4. Try different camera_index (0, 1, 2, etc.)

### Poor Marker Detection

- Ensure good lighting
- Print markers at adequate size (at least 5cm x 5cm)
- Avoid glare on markers
- Keep markers flat and visible
- Use matte paper (not glossy)

### Grid Not Projecting

- Whiteboard must be detected first
- Press `c` for manual calibration
- Enable auto-detect with `a` key
- Adjust lighting and camera angle

## 📚 Examples

### Example 1: Basic Detection

```python
detector = ObjectDetector()

while True:
    frame = detector.capture_frame()
    markers, corners, ids = detector.detect_markers(frame)
    
    result = detector.draw_markers(frame, markers, corners, ids)
    cv2.imshow('Detection', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Example 2: Grid Coordinates

```python
detector = ObjectDetector()
detector.detect_whiteboard(frame)

markers, _, _ = detector.detect_markers(frame)

for marker_id, pos in markers.items():
    wb = detector.pixel_to_whiteboard_coords(pos)
    grid = detector.whiteboard_to_grid_coords(wb)
    print(f"Marker {marker_id}: Grid {grid}")
```

## 🔗 Related Files

- `object_detection.py` - Main detection module
- `arm_with_vision.py` - Robot arm integration
- `generate_markers.py` - ArUco marker generator
- `requirements.txt` - Python dependencies

## 📄 License

Copyright 2025 Trossen Robotics - See file headers for full license text.

---

**Need help?** Check the troubleshooting section or review the inline code documentation.
