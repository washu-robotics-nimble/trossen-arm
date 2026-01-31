# Trossen Arm Robotics Project

A WashU Robotics project for Nimble.

## Project Structure

```
trossen-arm/
├── perception/              # Vision & sensing modules
│   ├── yolo/               # YOLO-based detection
│   ├── vla/               # VLA vision support
│   └── utils/              # Shared utilities
│
├── learning/               # Model training & inference
│   ├── models/            # Trained model weights
│   ├── training/          # Training notebooks & datasets
|       ├── marker_detection/          # Markers
|       ├── vla/                        # VLA
│   └── inference/         # Inference utilities
│
├── control/               # Robot control
│   ├── scripts/          # Scripts
│   └── vla/                # VLA
│
└── config/               # Configuration files
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Live Marker Detection
```bash
python perception/yolo/live_detection.py
python perception/yolo/live_detection.py --conf 0.5 --imgsz 1280
```

### Robot Control (simple)
```bash
python control/simple_move.py
```

### Model Training
Open and run the training notebook:
```bash
jupyter notebook learning/training/yolo_training.ipynb
```

## Configuration

Edit `config/robot_config.yaml` to customize:
- Robot settings (IP, model, end effector)
- Camera parameters
- Detection thresholds

## Marker Detection

Trained models are stored in `learning/models/`:
- `marker_detection/best.pt` - YOLO marker detection model

## Future Development

- **Perception**: Add new detection methods in `perception/`
- **Learning**: Train new models in `learning/training/`
- **Control**: Add robot behaviors in `control/`
- **VLA (NEXT STEP)**: Add model / preprocessing / inference / training info to 
`learning`, `perception`, and `control` folders. Update `config` folder as needed.

## Hardware

- Trossen WidowX AI arm
- Logitech Camera (?)
- IP: 192.168.2.2 (configurable)

## References

- [Trossen Robotics WidowX AI](https://www.trossenrobotics.com/widowx-ai)
- [NORA](?)