#!/usr/bin/env python3
"""
Install Roboflow and set up ML-based whiteboard detection.
"""

import subprocess
import sys

print("=" * 60)
print("   Installing Roboflow ML Whiteboard Detection")
print("=" * 60)
print()

# Install roboflow
print("Installing roboflow package...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "supervision"])
    print("✓ Roboflow installed successfully!")
except Exception as e:
    print(f"✗ Failed to install roboflow: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("   Testing Roboflow Integration")
print("=" * 60)
print()

# Test the import
try:
    from roboflow import Roboflow
    print("✓ Roboflow import successful")
    
    # Test loading the model
    print("\nLoading whiteboard detection model...")
    print("(This may take a moment on first run)")
    
    rf = Roboflow(api_key="mSYCb2E062UlCPoLQOjO")
    project = rf.workspace("paul-artushkov").project("whiteboards-detection")
    version = project.version(15)
    model = version.model
    
    print("✓ Whiteboard detection model loaded successfully!")
    print()
    print("=" * 60)
    print("   Setup Complete!")
    print("=" * 60)
    print()
    print("The system will now use ML-based whiteboard detection.")
    print("This provides:")
    print("  ✓ More accurate whiteboard detection")
    print("  ✓ Better performance in various lighting")
    print("  ✓ Detection of whiteboards at angles")
    print("  ✓ Reduced false positives")
    print()
    print("To use the object detection system:")
    print("  python object_detection.py")
    print()
    print("The system will automatically use Roboflow detection.")
    print("If Roboflow fails, it will fall back to color-based detection.")
    
except Exception as e:
    print(f"✗ Error setting up Roboflow: {e}")
    print()
    print("The system will fall back to color-based detection.")
    print("You can still use: python object_detection.py")
    sys.exit(1)
