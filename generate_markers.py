#!/usr/bin/env python3
"""
Generate ArUco markers for object detection.

This script creates printable ArUco markers that can be used with the object detection system.
"""

import cv2
import numpy as np
import os


def generate_aruco_markers(marker_ids, marker_size=200, output_dir="aruco_markers"):
    """
    Generate ArUco markers and save them as images.
    
    Args:
        marker_ids: List of marker IDs to generate
        marker_size: Size of each marker in pixels
        output_dir: Directory to save markers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the same dictionary as in object detection
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    print(f"Generating {len(marker_ids)} ArUco markers...")
    print(f"Dictionary: DICT_4X4_50")
    print(f"Marker size: {marker_size}x{marker_size} pixels")
    print(f"Output directory: {output_dir}/")
    print("-" * 50)
    
    for marker_id in marker_ids:
        # Generate marker image
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Add border and ID label
        border = 50
        full_size = marker_size + 2 * border
        full_image = np.ones((full_size, full_size), dtype=np.uint8) * 255
        
        # Place marker in center
        full_image[border:border+marker_size, border:border+marker_size] = marker_image
        
        # Add ID label
        label = f"ID: {marker_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        text_x = (full_size - text_width) // 2
        text_y = full_size - 10
        
        cv2.putText(full_image, label, (text_x, text_y), 
                   font, font_scale, 0, thickness)
        
        # Save marker
        filename = f"{output_dir}/marker_{marker_id}.png"
        cv2.imwrite(filename, full_image)
        print(f"✓ Generated: {filename}")
    
    print("-" * 50)
    print(f"✓ All markers generated successfully!")
    print(f"\nTo use these markers:")
    print(f"  1. Print the images from '{output_dir}/' folder")
    print(f"  2. Place them in the camera view")
    print(f"  3. Run: python object_detection.py")


def generate_marker_sheet(marker_ids, sheet_size=(1000, 1400), marker_size=150):
    """
    Generate a single sheet with multiple markers for easy printing.
    
    Args:
        marker_ids: List of marker IDs to include
        sheet_size: Size of the sheet (width, height)
        marker_size: Size of each marker
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Create white sheet
    sheet = np.ones((sheet_size[1], sheet_size[0]), dtype=np.uint8) * 255
    
    # Calculate grid layout
    cols = 4
    spacing_x = sheet_size[0] // cols
    spacing_y = 220  # Vertical spacing
    
    print(f"\nGenerating marker sheet with {len(marker_ids)} markers...")
    
    for i, marker_id in enumerate(marker_ids):
        row = i // cols
        col = i % cols
        
        # Generate marker
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Calculate position (centered in grid cell)
        x = col * spacing_x + (spacing_x - marker_size) // 2
        y = row * spacing_y + 50
        
        if y + marker_size + 50 > sheet_size[1]:
            break  # Don't exceed sheet bounds
        
        # Place marker
        sheet[y:y+marker_size, x:x+marker_size] = marker_image
        
        # Add ID label
        label = f"ID: {marker_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        text_x = x + (marker_size - text_width) // 2
        text_y = y + marker_size + 30
        
        cv2.putText(sheet, label, (text_x, text_y), 
                   font, font_scale, 0, thickness)
    
    # Add title
    title = "ArUco Markers - DICT_4X4_50"
    cv2.putText(sheet, title, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    
    # Save sheet
    filename = "aruco_markers_sheet.png"
    cv2.imwrite(filename, sheet)
    print(f"✓ Generated: {filename}")
    print(f"  Ready to print!")


if __name__ == "__main__":
    # Generate individual markers (IDs 0-9)
    marker_ids = list(range(10))
    generate_aruco_markers(marker_ids)
    
    # Generate a printable sheet
    generate_marker_sheet(marker_ids)
    
    print("\n" + "=" * 50)
    print("Marker generation complete!")
    print("=" * 50)
