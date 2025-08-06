import cv2
import numpy as np
import torch
import os
from typing import Dict, List, Any
from inference_sdk import InferenceHTTPClient
import tempfile


class PlayerDetector:
    """
    Detects players and goalies in hockey broadcast footage.
    Creates bounding boxes and reference points for detected players.
    """

    def __init__(
        self, 
        api_key: str = "YDZxw1AQEvclkzV0ZLOz",
        workspace_name: str = "hockey-fghn7", 
        workflow_id: str = "custom-workflow-3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = None
    ):
        """
        Initialize the player detector with Roboflow.
        
        Args:
            api_key: Roboflow API key
            workspace_name: Roboflow workspace name
            workflow_id: Roboflow workflow ID
            device: Device preference (kept for compatibility)
            output_dir: Directory to save processed frames and data
        """
        self.api_key = api_key
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.device = device
        self.client = self._initialize_client()
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
        
    def _initialize_client(self) -> InferenceHTTPClient:
        """
        Initialize the Roboflow inference client.
        
        Returns:
            Initialized Roboflow client
        """
        try:
            client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
            print("Initialized Roboflow inference client")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Roboflow client: {str(e)}")
    
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
            # Save frame temporarily for Roboflow API
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, frame)
                temp_path = tmp_file.name
            
            try:
                # Run inference with Roboflow workflow
                result = self.client.run_workflow(
                    workspace_name=self.workspace_name,
                    workflow_id=self.workflow_id,
                    images={"image": temp_path},
                    use_cache=True
                )
                

                # Process detections from Roboflow response
                detections = []
                if result and len(result) > 0:
                    if "label_visualization" in result[0]:
                        predictions = result[0]["label_visualization"].get("predictions", [])
                        
                        for pred in predictions:
                            confidence = pred.get("confidence", 0.0)
                            if confidence < self.conf_threshold:
                                continue
                            
                            # Convert Roboflow format (x, y, width, height) to (x1, y1, x2, y2)
                            x_center = pred.get("x", 0)
                            y_center = pred.get("y", 0)
                            width = pred.get("width", 0)
                            height = pred.get("height", 0)
                            
                            x1 = x_center - width / 2
                            y1 = y_center - height / 2
                            x2 = x_center + width / 2
                            y2 = y_center + height / 2
                            
                            # Get class information
                            class_name = pred.get("class", "unknown")
                            class_id = pred.get("class_id", 0)
                            
                            # Map Roboflow classes to expected format and filter for specific classes
                            target_classes = ["player", "puck", "stick_blade", "goalkeeper", "goalie", "goalzone"]
                            if class_name.lower() in target_classes:
                                # Normalize goalkeeper/goalie naming
                                if class_name.lower() in ["goalkeeper", "goalie"]:
                                    mapped_class = "goalkeeper"
                                else:
                                    mapped_class = class_name.lower()
                            else:
                                continue  # Skip all other classes (referee, etc.)
                            
                            # Calculate reference point at bottom center
                            ref_x = (x1 + x2) / 2  # x-coordinate at center of bbox
                            ref_y = y2  # y-coordinate at bottom of bbox
                            
                            # Create detection dictionary
                            detection = {
                                "bbox": (x1, y1, x2, y2),
                                "confidence": confidence,
                                "class": mapped_class,
                                "reference_point": {
                                    "x": float(ref_x),  # Ensure coordinates are float
                                    "y": float(ref_y),
                                    "pixel_x": int(ref_x),  # Add pixel-space coordinates
                                    "pixel_y": int(ref_y)
                                }
                            }
                            
                            detections.append(detection)
                    # If label_visualization not found, skip this frame
                    pass
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            # Visualize detections on frame
            vis_frame = self.visualize_detections(frame, detections)
            
            # Save processed frame if output directory is set
            if self.output_dir and frame_idx is not None:
                frame_path = os.path.join(
                    self.frames_dir, 
                    f"frame_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(frame_path, vis_frame)
            
            # Display frame (disabled for real-time processing)
            # cv2.imshow('Player Detection', vis_frame)
            # cv2.waitKey(1)  # 1ms delay to allow window updates
            
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
        Visualize detections on the frame using ellipses at bottom center.
        
        Args:
            frame: Input frame (BGR format)
            detections: List of detection dictionaries
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        for det in detections:
            # Get reference point (bottom center)
            ref_point = det["reference_point"]
            ref_x = int(ref_point["pixel_x"])
            ref_y = int(ref_point["pixel_y"])
            
            # Set color and ellipse size based on class
            if det["class"] == "player":
                color = (0, 255, 0)  # Green for players
                ellipse_size = (20, 12)  # Larger ellipse for better visibility
            elif det["class"] == "puck":
                color = (0, 255, 255)  # Yellow for puck
                ellipse_size = (12, 8)  # Larger ellipse for puck
            elif det["class"] == "stick_blade":
                color = (255, 0, 255)  # Magenta for stick blade
                ellipse_size = (16, 10)  # Larger ellipse for stick blade
            elif det["class"] == "goalkeeper":
                color = (0, 0, 255)  # Red for goalkeeper
                ellipse_size = (25, 15)  # Larger ellipse for goalkeeper
            elif det["class"] == "goalzone":
                color = (255, 165, 0)  # Orange for goal zone
                ellipse_size = (30, 20)  # Large ellipse for goal zone
            else:
                continue  # Skip other classes
            
            # Draw filled ellipse at bottom center
            cv2.ellipse(vis_frame, (ref_x, ref_y), ellipse_size, 0, 0, 360, color, -1)
            
            # Draw ellipse outline for better visibility
            cv2.ellipse(vis_frame, (ref_x, ref_y), ellipse_size, 0, 0, 360, (255, 255, 255), 2)
            
            # Draw class label above ellipse
            label = det['class'].upper()
            if label == 'STICK_BLADE':
                label = 'STICK'
            elif label == 'GOALZONE':
                label = 'GOAL'
            
            # Calculate label position (above the ellipse)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            label_x = ref_x - label_size[0] // 2
            label_y = ref_y - ellipse_size[1] - 5
            
            # Draw label background
            cv2.rectangle(vis_frame, 
                         (label_x - 2, label_y - label_size[1] - 2),
                         (label_x + label_size[0] + 2, label_y + 2),
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_frame

    def __del__(self):
        """Cleanup method to ensure windows are closed"""
        cv2.destroyAllWindows()