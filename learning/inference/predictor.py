"""YOLO model inference utilities."""

import os
from pathlib import Path
from ultralytics import YOLO
import torch


class MarkerPredictor:
    """Wrapper for YOLO marker detection inference."""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model weights (default: auto-detect)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto)
        """
        if model_path is None:
            # Auto-detect best model
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "learning/models/marker_detection/best.pt"
            
        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                     ("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.model = YOLO(str(model_path))
        self.model.to(device)
        self.device = device
        
    def predict(self, frame, conf=0.25, imgsz=640, verbose=False):
        """
        Run inference on a frame.
        
        Args:
            frame: Input image (numpy array)
            conf: Confidence threshold
            imgsz: Input image size
            verbose: Print inference details
            
        Returns:
            YOLO Results object
        """
        results = self.model.predict(
            frame,
            imgsz=imgsz,
            conf=conf,
            device=self.device,
            verbose=verbose
        )
        return results[0]  # Return first result
    
    def predict_and_plot(self, frame, **kwargs):
        """
        Run inference and return annotated frame.
        
        Args:
            frame: Input image
            **kwargs: Additional arguments for predict()
            
        Returns:
            Annotated frame
        """
        result = self.predict(frame, **kwargs)
        return result.plot()
