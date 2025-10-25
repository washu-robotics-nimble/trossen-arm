# Quick Start Guide - Camera Object Detection

## 🎬 Getting Started in 3 Steps

### Step 1: Generate & Print Markers (1 minute)
```bash
python generate_markers.py
```
- Opens the `aruco_markers_sheet.png` file
- Print this sheet on regular paper
- Cut out individual markers or use the whole sheet

### Step 2: Setup Your Workspace (2 minutes)
1. Position your whiteboard where the camera can see it
2. Place printed markers on or near the whiteboard
3. Ensure good lighting (no harsh shadows or glare)

### Step 3: Run Detection (30 seconds)
```bash
python object_detection.py
```

That's it! You should see:
- ✅ Live camera feed
- ✅ Automatic whiteboard detection
- ✅ Coordinate grid projected on whiteboard
- ✅ Detected markers with their coordinates

## 🎨 What You'll See

### Visual Elements

**1. Whiteboard Detection**
- Green outline around detected whiteboard
- Corner markers labeled: TL (top-left), TR, BR, BL
- Automatically detected and tracked

**2. Coordinate Grid**
- 10x10 grid by default (adjustable)
- Major gridlines (cyan, thicker) every 5 units
- Minor gridlines (green, thinner) for precision
- X-axis labels along top (0, 2, 4, 6, 8, 10)
- Y-axis labels along left (0, 2, 4, 6, 8, 10)

**3. ArUco Markers**
- Yellow outline around detected markers
- Red center dot
- White circle around center
- Labels showing:
  - `ID: X` - Marker identification number
  - `Grid: (X, Y)` - Discrete grid cell position
  - `WB: (X.XX, Y.YY)` - Normalized whiteboard position

**4. Status Panel (Top Left)**
```
FPS: 30.0
Markers detected: 2
Whiteboard: DETECTED
Auto-detect: ON
Grid size: 10x10
Labels: ON
```

**5. Marker List (Bottom)**
```
Marker IDs: 0, 3, 7
```

**6. Help Text (Bottom)**
```
Press 'c' to calibrate | 'a' auto-detect | 'g' labels | '+/-' grid size | 'q' quit
```

## 🎮 Interactive Controls

### During Runtime

**Essential Controls:**
- `q` - **Quit** the program
- `c` - **Calibrate** whiteboard manually
- `a` - **Toggle auto-detect** (recommended: keep ON)

**Grid Controls:**
- `+` or `=` - **Increase** grid size (max 20x20)
- `-` or `_` - **Decrease** grid size (min 5x5)
- `g` - **Toggle** coordinate labels

**Utility:**
- `s` - **Save** current frame as JPG

### Tips for Best Results

**Whiteboard Setup:**
- ✅ Clean, white surface
- ✅ Good, even lighting
- ✅ Visible to camera
- ✅ Flat (not curved)
- ❌ Avoid glare from lights
- ❌ Avoid shadows

**Marker Placement:**
- ✅ Print at least 5cm x 5cm size
- ✅ Use matte paper (not glossy)
- ✅ Keep markers flat
- ✅ Place within whiteboard area
- ✅ Ensure all 4 corners visible
- ❌ Don't overlap markers
- ❌ Avoid shadows on markers

**Lighting:**
- ✅ Bright, diffuse lighting
- ✅ Multiple light sources
- ✅ Even illumination
- ❌ Avoid direct sunlight
- ❌ Avoid single harsh light source

## 📊 Understanding Coordinates

### Three Coordinate Systems

**1. Pixel Coordinates** (Internal)
- Raw camera coordinates
- Example: (845, 432)
- Not shown to user

**2. Whiteboard Coordinates** (Normalized)
- Range: 0.0 to 1.0 on each axis
- Example: `WB: (0.72, 0.45)`
- Useful for calculations
- (0, 0) = top-left, (1, 1) = bottom-right

**3. Grid Coordinates** (Discrete)
- Integer grid cell indices
- Example: `Grid: (7, 4)`
- For default 10x10: range 0-9 on each axis
- **Use these for robot arm control!**

### Example Marker Reading

When you see:
```
ID: 3
Grid: (7, 4)
WB: (0.72, 0.45)
```

This means:
- Marker with ID 3
- Located in grid cell column 7, row 4
- At 72% across, 45% down the whiteboard
- To move arm there, use grid coordinates (7, 4)

## 🤖 Using with Robot Arm

### Basic Example

```python
from object_detection import ObjectDetector

# Initialize
detector = ObjectDetector(camera_index=0)

# Detect whiteboard once
frame = detector.capture_frame()
detector.detect_whiteboard(frame)

# Main loop
while True:
    frame = detector.capture_frame()
    markers, corners, ids = detector.detect_markers(frame)
    
    for marker_id, pixel_pos in markers.items():
        # Get grid coordinates
        wb = detector.pixel_to_whiteboard_coords(pixel_pos)
        grid = detector.whiteboard_to_grid_coords(wb)
        
        print(f"Move arm to grid position: {grid}")
        # arm.move_to_grid(grid[0], grid[1])
```

### Integration Points

**Get marker positions:**
```python
markers, corners, ids = detector.detect_markers(frame)
```

**Convert to grid:**
```python
wb_coords = detector.pixel_to_whiteboard_coords(pixel_pos)
grid_coords = detector.whiteboard_to_grid_coords(wb_coords)
```

**Use in arm control:**
```python
arm.move_to_grid(grid_coords[0], grid_coords[1])
```

## 🔧 Troubleshooting

### "Camera 0 is not available"
**Solutions:**
1. Check camera connection
2. macOS: System Preferences → Security & Privacy → Camera → Allow Terminal/Python
3. Close other apps using camera (Zoom, Skype, etc.)
4. Try different camera index in code

### "Whiteboard: NOT DETECTED"
**Solutions:**
1. Press `c` to force calibration
2. Improve lighting
3. Move camera to see entire whiteboard
4. Ensure whiteboard is clean and white
5. Remove obstructions

### "Markers detected: 0"
**Solutions:**
1. Print markers larger (at least 5cm)
2. Improve lighting on markers
3. Use matte paper (not glossy)
4. Keep markers flat and unobstructed
5. Ensure markers are DICT_4X4_50 type

### Low FPS (< 15)
**Solutions:**
1. Close other applications
2. Reduce grid size with `-` key
3. Disable labels with `g` key
4. Use lower camera resolution (modify code)

## 📁 File Reference

**Run these:**
- `object_detection.py` - Main detection system
- `generate_markers.py` - Create markers
- `test_vision.py` - System test
- `arm_with_vision.py` - Arm integration example

**Read these:**
- `VISION_README.md` - Full documentation
- `SETUP_SUMMARY.md` - What was implemented
- `README.md` - Project overview

**Use these:**
- `aruco_markers/marker_X.png` - Individual markers
- `aruco_markers_sheet.png` - Printable sheet

## ⚡ Performance Tips

**For best performance:**
1. Close unnecessary applications
2. Use adequate lighting (reduces detection time)
3. Keep whiteboard steady (reduces false detections)
4. Use appropriate grid size (smaller = faster)
5. Disable labels if not needed (`g` key)

**Expected performance:**
- FPS: 20-30 on modern computers
- Detection latency: < 50ms
- Whiteboard detection: < 100ms
- Marker detection: < 30ms per marker

## ✅ Checklist

Before running:
- [ ] OpenCV installed (`pip install opencv-python`)
- [ ] Markers generated (`python generate_markers.py`)
- [ ] Markers printed
- [ ] Whiteboard positioned
- [ ] Camera connected
- [ ] Good lighting
- [ ] Camera permissions granted

Ready to run:
```bash
python object_detection.py
```

Press `q` to quit when done.

---

**Need more help?** See `VISION_README.md` for detailed documentation.
