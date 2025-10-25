#!/bin/bash

# Vision System Setup and Test Script
# This script helps you get started with the camera object detection system

echo "=================================================="
echo "   Trossen Arm - Vision System Setup"
echo "=================================================="
echo ""

# Check Python environment
echo "1. Checking Python environment..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found! Please install Python 3."
    exit 1
fi
echo "✓ Python found: $PYTHON_CMD"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment found"
    PYTHON_CMD=".venv/bin/python"
else
    echo "⚠ No virtual environment found (will use system Python)"
fi

echo ""
echo "2. Checking dependencies..."

# Check if opencv is installed
$PYTHON_CMD -c "import cv2; print('✓ OpenCV version:', cv2.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ OpenCV not installed"
    echo ""
    echo "Installing dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✓ Dependencies installed"
else
    echo "✓ All dependencies installed"
fi

echo ""
echo "=================================================="
echo "   What would you like to do?"
echo "=================================================="
echo ""
echo "1) Generate ArUco markers (first time setup)"
echo "2) Test camera and detection system"
echo "3) Run object detection (main program)"
echo "4) View documentation"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Generating ArUco markers..."
        echo "=================================================="
        $PYTHON_CMD generate_markers.py
        echo ""
        echo "✓ Markers generated!"
        echo ""
        echo "Next steps:"
        echo "  1. Print the file: aruco_markers_sheet.png"
        echo "  2. Or print individual markers from aruco_markers/ folder"
        echo "  3. Place markers in camera view"
        echo "  4. Run this script again and choose option 3"
        ;;
    2)
        echo ""
        echo "Testing system..."
        echo "=================================================="
        $PYTHON_CMD test_vision.py
        ;;
    3)
        echo ""
        echo "Starting object detection..."
        echo "=================================================="
        echo ""
        echo "Controls:"
        echo "  - Press 'q' to quit"
        echo "  - Press 'c' to calibrate whiteboard"
        echo "  - Press 'a' to toggle auto-detect"
        echo "  - Press 'g' to toggle grid labels"
        echo "  - Press '+/-' to adjust grid size"
        echo ""
        echo "Starting in 3 seconds..."
        sleep 3
        $PYTHON_CMD object_detection.py
        ;;
    4)
        echo ""
        echo "Opening documentation..."
        echo "=================================================="
        echo ""
        echo "Available documentation files:"
        echo "  - QUICK_START.md     : Quick start guide"
        echo "  - VISION_README.md   : Complete documentation"
        echo "  - SETUP_SUMMARY.md   : Implementation summary"
        echo ""
        if command -v open &> /dev/null; then
            open QUICK_START.md
        elif command -v xdg-open &> /dev/null; then
            xdg-open QUICK_START.md
        else
            echo "Please open QUICK_START.md manually"
        fi
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "   Done!"
echo "=================================================="
