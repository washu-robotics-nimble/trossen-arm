import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import json
import os

# -------------------------------------------------------------------
# 1. Custom Dataset Class
# -------------------------------------------------------------------
# This class is built to read your specific JSON annotation file
# and load the corresponding images.
# -------------------------------------------------------------------

class MarkerDataset(Dataset):
    def __init__(self, json_file, image_dir, transforms=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            image_dir (string): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transforms = transforms
        
        # --- Define your classes ---
        # 0 is reserved for the background class
        # Your data only has one class: "capped_marker"
        self.class_map = {"capped_marker": 1}
        
        # Load the json file
        print(f"Loading annotations from: {json_file}")
        with open(json_file) as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} annotations.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the annotation data for one image
        item = self.data[idx]
        
        # Get the image file name from the json's 'data.image' field
        # and create the full image path
        try:
            image_name = os.path.basename(item['data']['image'])
        except KeyError:
            print(f"Error: 'data' or 'image' key not found in item {idx}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # FIX THE NAMES
        image_name = image_name[9:]


        image_path = os.path.join(self.image_dir, image_name)
        
        # Open the image
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Using next image.")
            # Recursively call __getitem__ with the next index
            return self.__getitem__((idx + 1) % len(self))
            
        width, height = image.size

        boxes = []
        labels = []

        # Check if there are annotations for this image
        if item['annotations'] and item['annotations'][0]['result']:
            annotations = item['annotations'][0]['result']
            
            for ann in annotations:
                # Get the label name
                try:
                    label_name = ann['value']['polygonlabels'][0]
                except (KeyError, IndexError):
                    continue # Skip if annotation is malformed
                
                if label_name not in self.class_map:
                    continue
                
                labels.append(self.class_map[label_name])

                # Your JSON has polygon points as percentages (0-100)
                points_pct = ann['value']['points']
                
                # --- Convert Polygon to Bounding Box [xmin, ymin, xmax, ymax] ---
                # We find the min/max x and y from the polygon points
                all_x = [p[0] * width / 100 for p in points_pct]
                all_y = [p[1] * height / 100 for p in points_pct]
                
                xmin = min(all_x)
                ymin = min(all_y)
                xmax = max(all_x)
                ymax = max(all_y)
                
                # Ensure box is valid
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])

        # Convert everything into torch tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            # Handle images with no objects
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)


        # Create the target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Apply transforms (e.g., ToTensor)
        # Note: 'target' is not transformed by standard torchvision transforms
        # The image is, however.
        if self.transforms:
            image = self.transforms(image)

        return image, target

# -------------------------------------------------------------------
# 2. Model Definition
# -------------------------------------------------------------------
# This function loads a pre-trained Faster R-CNN model and
# replaces the final classification layer to match our number of classes.
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
# 3. Utility Functions
# -------------------------------------------------------------------

# DataLoaders for object detection require a custom collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))

# Simple transform: just convert image to tensor
# You can add augmentations here later
class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)

# -------------------------------------------------------------------
# 4. Main Training Script
# -------------------------------------------------------------------

def main():
    # --- !!! YOU MUST CHANGE THESE PATHS !!! ---
    # Path to your JSON file
    JSON_PATH = './data/project-4-at-2025-11-01-18-19-9b493072.json' 
    # Path to the folder containing your images
    IMAGE_DIR = './data/marker_pics' 
    # -------------------------------------------
    
    # --- Hyperparameters ---
    # We have 2 classes: 1 (background) + 1 (capped_marker)
    NUM_CLASSES = 2
    NUM_EPOCHS = 2
    BATCH_SIZE = 2  # Start small, increase if your GPU memory allows
    LEARNING_RATE = 0.005

    # --- Setup ---
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # --- 1. Create Dataset and DataLoader ---
    print("Initializing Dataset and DataLoader...")
    dataset = MarkerDataset(json_file=JSON_PATH,
                            image_dir=IMAGE_DIR,
                            transforms=ToTensor()) # Apply the ToTensor transform
    
    # Optional: Split into train/validation
    # For simplicity, we'll use the whole dataset for training first
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn # Use the custom collate_fn
    )
    print("DataLoader created successfully.")

    # --- 2. Initialize Model ---
    print("Initializing model...")
    model = get_detection_model(NUM_CLASSES)
    model.to(device)
    print("Model moved to device.")

    # --- 3. Setup Optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=0.0005)
    
    # --- 4. Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(data_loader):
            # Move data to the device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            # This returns a loss dictionary when in train() mode
            try:
                loss_dict = model(images, targets)
            except Exception as e:
                print(f"Error during model forward pass: {e}")
                print("Skipping this batch.")
                continue # Skip batch if an error occurs

            # Sum all the losses
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            running_loss += loss_value

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (i + 1) % 10 == 0: # Print loss every 10 batches
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {loss_value:.4f}')

        print(f'--- Epoch {epoch+1} Average Loss: {running_loss / len(data_loader):.4f} ---')

    print("--- Finished Training ---")
    
    # (Optional) Save the trained model
    torch.save(model.state_dict(), 'marker_detector.pth')
    print("Model saved to marker_detector.pth")

if __name__ == "__main__":
    main()
