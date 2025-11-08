import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import time
# -------------------------------------------------------------------
# 1. Model Definition (Must be the same as in your training script)
# -------------------------------------------------------------------

def get_detection_model(num_classes):
    """
    Loads a pre-trained Faster R-CNN model and adapts it.
    
    Args:
        num_classes (int): The number of classes INCLUDING the background.
                           (e.g., 1 class + 1 background = 2)
    """
    # Load a model pre-trained on COCO
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# -------------------------------------------------------------------
# 2. Main Inference Function
# -------------------------------------------------------------------

def run_webcam_detector():
    # --- Configuration ---
    
    # !!! IMPORTANT: CHANGE THIS !!!
    # Path to your saved model weights
    MODEL_PATH = 'marker_detector.pth' 
    
    # We have 2 classes: 1 (background) + 1 (capped_marker)
    NUM_CLASSES = 2
    CLASS_NAMES = ['__background__', 'capped_marker']
    
    # Confidence threshold: only show detections with a score > 0.5
    CONFIDENCE_THRESHOLD = 0.4
    
    # Colors for drawing
    BOX_COLOR = (0, 255, 0) # Green
    TEXT_COLOR = (0, 255, 0) # Green
    
    # --- Setup ---
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- 1. Load the Model ---
    print("Loading model...")
    model = get_detection_model(NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please make sure you have trained the model and")
        print("saved it to 'marker_detector.pth', or update the MODEL_PATH variable.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()  # Set model to evaluation mode (VERY IMPORTANT)
    print("Model loaded successfully.")

    # --- 2. Initialize Webcam ---
    print("Starting webcam...")
    # '0' is usually the default built-in webcam
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read one frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # --- 3. Pre-process the Frame ---
        # 1. Convert frame from BGR (OpenCV default) to RGB (PIL/PyTorch default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to a PyTorch tensor
        # This also normalizes pixel values from [0, 255] to [0.0, 1.0]
        # and changes shape from (H, W, C) to (C, H, W)
        tensor_frame = F.to_tensor(rgb_frame)
        
        # 3. Add a batch dimension and send to device
        input_tensor = tensor_frame.unsqueeze(0).to(device)

        # --- 4. Run Inference ---
        with torch.no_grad(): # Disable gradient calculation
            predictions = model(input_tensor)
        
        # 'predictions' is a list with one element
        # This element is a dictionary containing 'boxes', 'labels', 'scores'
        pred = predictions[0]

        # --- 5. Draw Detections on the Frame ---
        for i in range(len(pred['scores'])):
            score = pred['scores'][i].cpu().item()
            
            # Filter out weak detections
            if score > CONFIDENCE_THRESHOLD:
                # Get box coordinates (and convert to integer)
                box = pred['boxes'][i].cpu().numpy().astype(int)
                xmin, ymin, xmax, ymax = box
                
                # Get label
                label_idx = pred['labels'][i].cpu().item()
                label_name = CLASS_NAMES[label_idx]
                
                # Format text
                text = f"{label_name}: {score:.2f}"
                
                # Draw the bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BOX_COLOR, 2)
                
                # Draw the label text
                cv2.putText(frame, text, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

        # --- 6. Display the Frame ---
        cv2.imshow('Marker Detection (Press "q" to quit)', frame)

        time.sleep(0.5)
        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 7. Cleanup ---
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detector()

