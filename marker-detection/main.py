from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

# Create predictions directory if it doesn't exist
if not os.path.exists('predictions/prediction-images'):
    os.makedirs('predictions/prediction-images')

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (frame_width, frame_height))

# Initialize both models
#yoloe_model = YOLO("yoloe-11s-seg.pt")  # Segmentation model
marker_model = YOLO("best.pt")  # Marker detection model

# Set text prompt for YOLOE model (if it supports text prompts)
#names = ["marker", "whiteboard"]
# if hasattr(yoloe_model, 'set_classes'):
#     yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))

# Frame counter and capture settings
frame_count = 0
last_yoloe_results = None
last_marker_results = None
capture_interval = 10  # Capture every 10 seconds
last_capture_time = time.time()
capture_count = 0

print("Starting detection... Press 'q' to quit")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Process detection only every 2 frames for performance
    if frame_count % 2 == 0:
        # Run both models on the same frame
        # last_yoloe_results = yoloe_model.predict(frame, verbose=False)
        last_marker_results = marker_model.predict(frame, verbose=False)
    
    # Start with original frame
    annotated_frame = frame.copy()
    
    # Apply YOLOE segmentation results
    # if last_yoloe_results is not None and len(last_yoloe_results) > 0:
    #     annotated_frame = last_yoloe_results[0].plot(img=annotated_frame)
    
    # Apply marker detection results on top
    if last_marker_results is not None and len(last_marker_results) > 0:
        annotated_frame = last_marker_results[0].plot(img=annotated_frame)
    
    # Capture frame every 10 seconds based on actual time
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions/prediction-images/capture_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Saved: {filename}")
        capture_count += 1
        last_capture_time = current_time
    
    frame_count += 1
    
    # Write the frame into the file 'output.mp4'
    out.write(annotated_frame)
    
    # Display the frame
    cv2.imshow('Camera', annotated_frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()

print(f"\nTotal captures saved: {capture_count}")
print("Detection stopped.")