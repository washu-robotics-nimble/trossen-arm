from ultralytics import YOLO
import os 

# Train our model 
src_folder_pth = "/Users/zhuanling/Desktop/Nimble_Perception/src"
# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
data_yaml_pth = os.path.join(src_folder_pth, "capped_marker_dataset.yaml")
results = model.train(data=data_yaml_pth, epochs=100, imgsz=640)

