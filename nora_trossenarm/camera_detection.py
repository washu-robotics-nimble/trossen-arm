import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No camera found on device 0")
    exit(1)

print("Camera opened. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    