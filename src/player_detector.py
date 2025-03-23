import cv2
import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Any, Optional


class PlayerDetector:
    """
    Detects players and goalies in hockey broadcast footage.
    Creates bounding boxes and reference points for detected players.
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the player detector.
        
        Args:
            model_path: Path to the detection model
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()
        
        # Define class mapping (adjust based on actual model outputs)
        self.class_mapping = {
            0: "background",
            1: "player",
            2: "goalie",
            3: "referee"
            # Add more classes as needed
        }
        
        # Confidence threshold for detections
        self.conf_threshold = 0.5
        
        # Define input dimensions expected by the model
        self.input_width = 640
        self.input_height = 640
        
    def _load_model(self) -> Any:
        """
        Load the detection model.
        
        Returns:
            Loaded detection model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model using torch.load
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different model saving formats
        if isinstance(checkpoint, dict):
            # If it's a dictionary, it might contain the model under a key
            if "model" in checkpoint:
                model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                # We'd need a model architecture definition here
                # For now, just print the keys and return a mock model
                print(f"Detection model checkpoint keys: {checkpoint.keys()}")
                # Create a simple placeholder model for testing
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                model = fasterrcnn_resnet50_fpn(pretrained=False)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Try to find a key that might contain the model
                for key, value in checkpoint.items():
                    if isinstance(value, torch.nn.Module):
                        model = value
                        break
                else:
                    # If no model is found, print the keys for debugging
                    print(f"Detection model checkpoint keys: {checkpoint.keys()}")
                    raise ValueError("Could not extract model from checkpoint")
        else:
            # If it's directly a model
            model = checkpoint
        
        # Explicitly convert model to float32
        model = model.float()
        print("Converted player detection model to float32")
        
        model.eval()
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for model inference.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize the frame to the model's expected input dimensions
        resized_frame = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(resized_frame.transpose(2, 0, 1))
        # Explicitly convert to float32
        tensor = tensor.float()
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        # Normalize and send to device
        tensor = (tensor / 255.0).to(self.device)
        
        return tensor
    
    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a frame to detect players and goalies.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of dictionaries containing detection information
        """
        # Preprocess frame
        tensor = self.preprocess_frame(frame)
        
        # Ensure tensor is float32
        tensor = tensor.float()
        
        # Run inference
        with torch.no_grad():
            try:
                predictions = self.model(tensor)
                print(f"Player detector output type: {type(predictions)}")
            except Exception as e:
                print(f"Error during player detection inference: {str(e)}")
                # Return empty list for now to avoid crashing
                return []
        
        # Process detections
        detections = self._process_detections(predictions, frame.shape[:2])
        
        # Filter detections to only include players and goalies
        filtered_detections = [det for det in detections 
                             if det["class"] in ["player", "goalie"]]
        
        # Calculate reference point for each detection (1/3 height from bottom, center width)
        for det in filtered_detections:
            x1, y1, x2, y2 = det["bbox"]
            # Calculate reference point (1/3 height from bottom, center width)
            ref_x = (x1 + x2) / 2
            ref_y = y2 - (y2 - y1) / 3
            det["reference_point"] = (ref_x, ref_y)
        
        return filtered_detections
    
    def _process_detections(self, predictions: Any, frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Process model predictions to extract bounding boxes.
        
        Args:
            predictions: Model predictions
            frame_shape: Original frame shape (height, width)
            
        Returns:
            List of dictionaries containing detection information
        """
        # This processing depends on the specific model output format
        # Placeholder for now, to be implemented based on model specifics
        height, width = frame_shape
        detections = []
        
        # For example, if predictions contains boxes, scores, and classes:
        # boxes = predictions[0]['boxes'].cpu().numpy()
        # scores = predictions[0]['scores'].cpu().numpy()
        # labels = predictions[0]['labels'].cpu().numpy()
        #
        # for box, score, label in zip(boxes, scores, labels):
        #     if score < self.conf_threshold:
        #         continue
        #
        #     x1, y1, x2, y2 = box
        #     class_name = self.class_mapping.get(label, "unknown")
        #
        #     detections.append({
        #         "bbox": (x1, y1, x2, y2),
        #         "confidence": float(score),
        #         "class": class_name
        #     })
        
        return detections
    
    def get_player_crops(self, frame: np.ndarray, detections: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Extract crop images of detected players for orientation detection.
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Dictionary mapping detection indices to cropped images
        """
        crops = {}
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            # Convert to integers for cropping
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Only add if crop is valid
            if crop.size > 0:
                crops[i] = crop
        
        return crops
    
    def visualize_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detections on the frame.
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Convert to integers for drawing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            color = (0, 255, 0) if det["class"] == "player" else (0, 0, 255)  # Green for players, Red for goalies
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw reference point
            ref_x, ref_y = det["reference_point"]
            cv2.circle(vis_frame, (int(ref_x), int(ref_y)), 5, (255, 0, 0), -1)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
