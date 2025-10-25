#!/usr/bin/env python3
"""
Test script for object detection - verifies imports and basic functionality.
"""

import sys
import cv2
import numpy as np

print("=" * 60)
print("Object Detection System - Pre-flight Check")
print("=" * 60)

# Check OpenCV version
print(f"\n✓ OpenCV version: {cv2.__version__}")

# Check if ArUco is available
try:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    print(f"✓ ArUco module available")
except Exception as e:
    print(f"✗ ArUco module error: {e}")
    sys.exit(1)

# Check numpy
print(f"✓ NumPy version: {np.__version__}")

# Test ObjectDetector import
try:
    from object_detection import ObjectDetector
    print("✓ ObjectDetector class imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test camera availability
print("\n" + "-" * 60)
print("Camera Check:")
print("-" * 60)

camera = cv2.VideoCapture(0)
if camera.isOpened():
    print("✓ Camera 0 is available and opened successfully")
    ret, frame = camera.read()
    if ret:
        print(f"✓ Successfully captured frame: {frame.shape}")
        
        # Test whiteboard detection on a test frame
        detector = ObjectDetector(camera_index=0)
        print("✓ ObjectDetector initialized")
        
        # Test marker generation
        markers, corners, ids = detector.detect_markers(frame)
        print(f"✓ Marker detection works (found {len(markers)} markers)")
        
        # Test visualization methods
        result = detector.draw_markers(frame, markers, corners, ids)
        print(f"✓ Marker visualization works")
        
        print("\n" + "=" * 60)
        print("✓ All checks passed! System is ready.")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python object_detection.py")
        print("\nMake sure to:")
        print("  1. Print ArUco markers from aruco_markers/ folder")
        print("  2. Have a whiteboard visible to the camera")
        print("  3. Ensure good lighting conditions")
    else:
        print("✗ Failed to capture frame from camera")
        sys.exit(1)
    
    camera.release()
else:
    print("✗ Camera 0 is not available")
    print("\nPossible solutions:")
    print("  1. Check if camera is connected")
    print("  2. Grant camera permissions")
    print("  3. Close other apps using the camera")
    print("  4. Try running with a different camera index")
    sys.exit(1)
