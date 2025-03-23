import cv2
import numpy as np
import torch
import os
from typing import Dict, List, Tuple, Any, Optional


class SegmentationProcessor:
    """
    Processes frames through the segmentation model to identify rink features.
    Converts segmentation masks to polygons and extracts key points.
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the segmentation processor.
        
        Args:
            model_path: Path to the segmentation model
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.model = self._load_model()
        
        # Define class mapping
        self.class_mapping = {
            0: "Background",
            1: "BlueLine",
            2: "RedCenterLine",
            3: "GoalLine",
            4: "RedCircle"
        }
        
        # Define input dimensions expected by the model
        # Common resolutions used for training models: 640x640, 1280x1280, etc.
        self.input_width = 640
        self.input_height = 640
        
    def _load_model(self) -> Any:
        """
        Load the segmentation model.
        
        Returns:
            Loaded segmentation model
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
                print(f"Model checkpoint keys: {checkpoint.keys()}")
                # Create a simple placeholder model for testing
                from torchvision.models.segmentation import fcn_resnet50
                model = fcn_resnet50(pretrained=False)
                model.load_state_dict(checkpoint["state_dict"])
            else:
                # Try to find a key that might contain the model
                for key, value in checkpoint.items():
                    if isinstance(value, torch.nn.Module):
                        model = value
                        break
                else:
                    # If no model is found, print the keys for debugging
                    print(f"Model checkpoint keys: {checkpoint.keys()}")
                    raise ValueError("Could not extract model from checkpoint")
        else:
            # If it's directly a model
            model = checkpoint
        
        # Explicitly convert model to float32
        model = model.float()
        print("Converted segmentation model to float32")
        
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
        
        # Print original frame shape
        print(f"Original frame shape: {rgb_frame.shape}")
        
        # Resize to model's expected input dimensions
        resized_frame = cv2.resize(rgb_frame, (self.input_width, self.input_height))
        print(f"Resized frame shape: {resized_frame.shape}")
        
        # Normalize and convert to tensor - explicitly use float32 throughout
        tensor = torch.from_numpy(resized_frame.transpose(2, 0, 1))
        # First convert to float32 before any other operations
        tensor = tensor.float()
        # Then add batch dimension 
        tensor = tensor.unsqueeze(0)
        # Finally normalize and send to device
        tensor = (tensor / 255.0).to(self.device)
        
        print(f"Final tensor shape: {tensor.shape}")
        return tensor
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Process a frame through the segmentation model.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary containing detected features with polygons and endpoints
        """
        # Preprocess frame
        tensor = self.preprocess_frame(frame)
        
        # Ensure the tensor is float32 before passing to model
        tensor = tensor.float()  # Explicitly convert again to ensure float32
        
        # Run inference
        with torch.no_grad():
            try:
                output = self.model(tensor)
                print(f"Model output type: {type(output)}")
                if isinstance(output, dict):
                    print(f"Output keys: {output.keys()}")
                elif isinstance(output, list):
                    print(f"Output is a list of length {len(output)}")
                elif isinstance(output, torch.Tensor):
                    print(f"Output tensor shape: {output.shape}")
            except Exception as e:
                print(f"Error during model inference: {str(e)}")
                # Return empty features for now to avoid crashing
                return {
                    "BlueLine": [],
                    "RedCenterLine": [],
                    "GoalLine": [],
                    "RedCircle": []
                }
        
        # Process output to extract features
        # This part needs to be adjusted based on model output format
        segmentation_results = self._process_model_output(output, frame.shape[:2])
        
        # Convert segmentation masks to polygons and extract endpoints
        features = self._extract_features_from_segmentation(segmentation_results)
        
        return features
    
    def _process_model_output(self, output: Any, frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Process model output to extract segmentation masks.
        
        Args:
            output: Model output
            frame_shape: Original frame shape (height, width)
            
        Returns:
            Dictionary mapping class names to segmentation masks
        """
        height, width = frame_shape
        results = {}
        
        # Add debug information
        print(f"Processing model output: {type(output)}")
        
        # Handle different output formats
        if isinstance(output, tuple) and len(output) > 0:
            # If output is a tuple, try to extract the first item which is likely the segmentation output
            output_tensor = output[0]
            print(f"Output[0] type: {type(output_tensor)}, shape: {output_tensor.shape if hasattr(output_tensor, 'shape') else 'no shape'}")
            
            if isinstance(output_tensor, torch.Tensor):
                # Convert to CPU and numpy
                mask = output_tensor.cpu().detach()
                
                # For the specific model shape [1, 44, 8400]
                if len(mask.shape) == 3 and mask.shape[0] == 1 and mask.shape[2] == 8400:
                    # This appears to be an unusual format - let's try to reshape it
                    num_classes = mask.shape[1]  # 44 classes
                    
                    # Assuming the 8400 is a flattened spatial dimension (e.g., 84×100 or similar)
                    # Let's try to determine what the actual width and height should be
                    # Common resolutions: 80×105=8400, 84×100=8400, 70×120=8400
                    reshape_height, reshape_width = 84, 100  # Most likely resolution
                    
                    try:
                        # Reshape to [1, num_classes, height, width]
                        reshaped = mask.reshape(1, num_classes, reshape_height, reshape_width)
                        print(f"Reshaped to: {reshaped.shape}")
                        
                        # Take argmax to get class index per pixel
                        class_indices = reshaped[0].argmax(dim=0).numpy()
                        print(f"Class indices shape: {class_indices.shape}, unique values: {np.unique(class_indices)}")
                        
                        # Map class indices to class names
                        # This mapping may need to be adjusted based on the model
                        class_mapping = {
                            1: "BlueLine",
                            2: "RedCenterLine", 
                            3: "GoalLine",
                            4: "RedCircle"
                        }
                        
                        # Create binary mask for each class
                        for class_id, class_name in class_mapping.items():
                            class_mask = (class_indices == class_id).astype(np.uint8) * 255
                            # Resize mask to original frame dimensions
                            if class_mask.shape[0] != height or class_mask.shape[1] != width:
                                class_mask = cv2.resize(class_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            
                            # Add to results if mask contains any non-zero pixels
                            if np.any(class_mask):
                                results[class_name] = class_mask
                                print(f"Found {class_name} with {np.count_nonzero(class_mask)} pixels")
                    except Exception as e:
                        print(f"Error reshaping mask: {str(e)}")
                        
                        # As a fallback, let's try another approach - permute/transpose the tensor
                        try:
                            # Let's try a different approach - assume it's some kind of attention map
                            # Try to extract specific classes directly
                            for class_id, class_name in {1: "BlueLine", 2: "RedCenterLine", 3: "GoalLine", 4: "RedCircle"}.items():
                                if class_id < num_classes:
                                    # Get the class channel and reshape it
                                    class_channel = mask[0, class_id].reshape(reshape_height, reshape_width).numpy()
                                    # Threshold to get binary mask (might need to adjust threshold)
                                    class_mask = (class_channel > 0.5).astype(np.uint8) * 255
                                    # Resize to original dimensions
                                    class_mask = cv2.resize(class_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                    
                                    if np.any(class_mask):
                                        results[class_name] = class_mask
                                        print(f"Found {class_name} with {np.count_nonzero(class_mask)} pixels (fallback method)")
                        except Exception as e2:
                            print(f"Error in fallback processing: {str(e2)}")
                else:
                    print(f"Unexpected tensor shape: {mask.shape}, don't know how to process")
        
        # If output processing fails or no masks are found, return empty results
        if not results:
            print("No segmentation masks were extracted from model output")
            
        return results
    
    def _extract_features_from_segmentation(self, segmentation_results: Dict[str, np.ndarray]) -> Dict[str, List]:
        """
        Extract features (polygons, endpoints) from segmentation masks.
        
        Args:
            segmentation_results: Dictionary mapping class names to segmentation masks
            
        Returns:
            Dictionary containing extracted features
        """
        features = {
            "BlueLine": [],
            "RedCenterLine": [],
            "GoalLine": [],
            "RedCircle": []
        }
        
        # Process each class mask
        for class_name, mask in segmentation_results.items():
            if class_name not in features:
                continue
                
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour based on class type
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < 100:
                    continue
                    
                # Process based on class type
                if class_name in ["BlueLine", "RedCenterLine", "GoalLine"]:
                    # For lines, extract endpoints
                    points = self._extract_line_endpoints(contour)
                    features[class_name].append({
                        "confidence": 0.9,  # Placeholder confidence
                        "points": points
                    })
                elif class_name == "RedCircle":
                    # For circles, extract center and radius
                    center, radius = self._extract_circle_parameters(contour)
                    features[class_name].append({
                        "confidence": 0.9,  # Placeholder confidence
                        "center": center,
                        "radius": radius
                    })
        
        return features
    
    def _extract_line_endpoints(self, contour: np.ndarray) -> List[Dict[str, float]]:
        """
        Extract endpoints from a line contour.
        
        Args:
            contour: Contour points
            
        Returns:
            List containing top and bottom points
        """
        # Get bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Sort points by y-coordinate
        sorted_points = sorted(box, key=lambda p: p[1])
        
        # Get the top and bottom points
        top_point = sorted_points[0]
        bottom_point = sorted_points[-1]
        
        # Convert to the expected format
        return [
            {"x": float(top_point[0]), "y": float(top_point[1])},
            {"x": float(bottom_point[0]), "y": float(bottom_point[1])}
        ]
    
    def _extract_circle_parameters(self, contour: np.ndarray) -> Tuple[Dict[str, float], float]:
        """
        Extract center and radius from a circle contour.
        
        Args:
            contour: Contour points
            
        Returns:
            Center point and radius
        """
        # Fit a circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Convert to the expected format
        center = {"x": float(x), "y": float(y)}
        
        return center, float(radius)
