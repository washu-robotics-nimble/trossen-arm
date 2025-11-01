import cv2

# --- 1. Load the Pre-trained Classifier ---
# We are using a pre-built model from OpenCV for detecting frontal faces.
# You will need to download this file (see instructions below).
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except cv2.error as e:
    print(f"Error: Failed to load cascade classifier.")
    print("Download the 'haarcascade_frontalface_default.xml' file and place it in the same directory as this script.")
    print(f"Full error: {e}")
    exit()

# --- 2. Initialize the Camera ---
# 0 refers to the default camera (usually your built-in webcam).
# If you have multiple cameras, you might try 1, 2, etc.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Camera feed opened. Press 'q' in the video window to quit.")

# --- 3. Start the Detection Loop ---
while True:
    # Read one frame from the camera
    ret, frame = cap.read()

    # If the frame was not captured correctly, stop.
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    # --- 4. Perform Detection ---
    # Haar cascades work best on grayscale images.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame.
    # This returns a list of rectangles (x, y, w, h) for each face found.
    # scaleFactor=1.1: How much the image size is reduced at each image scale.
    # minNeighbors=5: How many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # --- 5. Draw Rectangles on the Original Frame ---
    for (x, y, w, h) in faces:
        # Draw a blue rectangle (BGR color) with a thickness of 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # --- 6. Display the Result ---
    # Show the final frame (with rectangles) in a window
    cv2.imshow('Face Detector', frame)

    # --- 7. Check for Quit Key ---
    # Wait 1ms for a key press. If 'q' is pressed, break the loop.
    if cv2.waitKey(1) == ord('q'):
        break

# --- 8. Clean Up ---
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Script finished. Resources released.")
