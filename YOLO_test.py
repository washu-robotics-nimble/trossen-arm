from ultralytics import YOLO
import os 
import torch

project_folder = "/Users/zhuanling/Desktop/Nimble_Perception"
src_folder_pth = os.path.join(project_folder, "src")
model_path = os.path.join(project_folder, "runs/detect/train6/weights/best.pt")

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = YOLO(model_path)
model.to(device)

# check from camera 
import cv2, time
# On macOS, AVFoundation often works better:
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # or cv2.VideoCapture(0) / CAP_DSHOW on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Inference (stream=False returns a list; we take first result)
    results = model.predict(frame, imgsz=640, conf=0.25, device=device, verbose=False)
    vis = results[0].plot()  # Ultralytics renders boxes/labels for you

    # FPS
    now = time.time()
    fps = 1.0 / (now - prev)
    prev = now
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("YOLOv8 Live", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()