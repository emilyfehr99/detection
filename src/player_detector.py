import cv2
import numpy as np
import torch
import os
from typing import Dict, List, Any
from ultralytics import YOLO


class PlayerDetector:
    """
    Detects players and goalies in hockey broadcast footage.
    Creates bounding boxes and reference points for detected players.
    """

    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = None
    ):
        """
        Initialize the player detector.
        
        Args:
            model_path: Path to the detection model
            device: Device to run inference on ("cuda" or "cpu")
            output_dir: Directory to save processed frames and data
        """
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()
        self.output_dir = output_dir
        
        if output_dir:
            # Create output directories
            self.frames_dir = os.path.join(output_dir, "processed_frames")
            os.makedirs(self.frames_dir, exist_ok=True)
        
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
            msg = f"Model not found at {self.model_path}"
            raise FileNotFoundError(msg)
        
        # Load YOLOv8 model
        model = YOLO(self.model_path)
        model.to(self.device)
        print("Loaded YOLOv8 model and converted to float32")
        
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
        
        # Resize frame to model's input dimensions
        resized = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        
        # Normalize and convert to tensor
        tensor = torch.from_numpy(resized.transpose(2, 0, 1))
        # Explicitly convert to float32
        tensor = tensor.float()
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        # Normalize and send to device
        tensor = (tensor / 255.0).to(self.device)
        
        return tensor
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = None) -> List[Dict]:
        """
        Process a frame to detect players and goalies.
        
        Args:
            frame: Input frame (BGR format)
            frame_idx: Index of the current frame
            
        Returns:
            List of dictionaries containing detection information
        """
        try:
            # Run inference with YOLOv8
            results = self.model(frame, verbose=False)
            
            # Process each detection
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf.item() < self.conf_threshold:
                        continue
                        
                    # Get box coordinates (already in x1,y1,x2,y2 format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get class name and confidence
                    class_id = int(box.cls.item())
                    class_name = self.class_mapping.get(class_id, "unknown")
                    confidence = float(box.conf.item())
                    
                    # Calculate reference point
                    ref_x = (x1 + x2) / 2
                    ref_y = y2 - (y2 - y1) / 3
                    
                    # Create detection dictionary
                    detection = {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence,
                        "class": class_name,
                        "reference_point": {"x": ref_x, "y": ref_y}
                    }
                    
                    detections.append(detection)
            
            # Visualize detections on frame
            vis_frame = self.visualize_detections(frame, detections)
            
            # Save processed frame if output directory is set
            if self.output_dir and frame_idx is not None:
                frame_path = os.path.join(
                    self.frames_dir, 
                    f"frame_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(frame_path, vis_frame)
            
            # Display frame
            cv2.imshow('Player Detection', vis_frame)
            cv2.waitKey(1)  # 1ms delay to allow window updates
            
            return detections
                
        except Exception as e:
            msg = f"Error during player detection inference: {str(e)}"
            print(msg)
            import traceback
            traceback.print_exc()
            cv2.destroyAllWindows()  # Clean up windows on error
            return []
    
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
            if det["class"] == "player":
                color = (0, 255, 0)  # Green for players
            elif det["class"] == "goalie":
                color = (0, 0, 255)  # Red for goalies
            else:
                color = (255, 255, 0)  # Cyan for referees
                
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw reference point
            ref_x, ref_y = det["reference_point"].values()
            cv2.circle(vis_frame, (int(ref_x), int(ref_y)), 5, (255, 0, 0), -1)
            
            # Draw label with confidence
            label = f"{det['class']} {det['confidence']:.2f}"
            # Position label above bounding box
            label_y = max(y1 - 10, 20)  # Ensure label is visible
            cv2.putText(vis_frame, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame

    def __del__(self):
        """Cleanup method to ensure windows are closed"""
        cv2.destroyAllWindows()
