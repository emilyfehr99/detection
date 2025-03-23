import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from typing import Dict, List, Tuple, Any
from torchvision.models import squeezenet1_1


class OrientationDetector:
    """
    Detects the orientation of hockey players (which direction they are facing).
    """

    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the orientation detector.
        
        Args:
            model_path: Path to the orientation model
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        
        # Define orientation classes - this needs to be defined before loading the model
        self.orientation_classes = ["left", "right", "neutral"]
        self.model_output_classes = 8  # From the error message
        
        # Now load the model after orientation_classes is defined
        self.model = self._load_model()
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Adjust size based on model requirements
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self) -> Any:
        """
        Load the orientation model.
        
        Returns:
            Loaded orientation model
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model using torch.load
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different model saving formats
        if isinstance(checkpoint, dict):
            # Based on the keys we saw in the output, this is likely a SqueezeNet model
            if 'features.0.weight' in checkpoint:
                # Create a SqueezeNet model
                model = squeezenet1_1(pretrained=False)
                
                # Replace the final classifier layer to match model's output classes
                model.classifier[1] = torch.nn.Conv2d(
                    in_channels=512,
                    out_channels=self.model_output_classes,
                    kernel_size=1
                )
                
                try:
                    # Try to load state dict directly
                    model.load_state_dict(checkpoint)
                    print("Successfully loaded orientation model as SqueezeNet")
                except Exception as e:
                    print(f"Error loading state dict directly: {e}")
                    print("Trying to match keys manually...")
                    
                    # Create a new state dict with matching keys
                    new_state_dict = {}
                    model_state_dict = model.state_dict()
                    
                    for key in model_state_dict.keys():
                        if key in checkpoint:
                            new_state_dict[key] = checkpoint[key]
                    
                    # Make sure the classifier layer is properly sized
                    if 'classifier.1.weight' in checkpoint:
                        new_state_dict['classifier.1.weight'] = checkpoint['classifier.1.weight']
                        new_state_dict['classifier.1.bias'] = checkpoint['classifier.1.bias']
                    
                    # Load the matched state dict
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"Loaded {len(new_state_dict)}/{len(model_state_dict)} parameters")
            elif "model" in checkpoint:
                model = checkpoint["model"]
            elif "state_dict" in checkpoint:
                # Create a SqueezeNet model
                model = squeezenet1_1(pretrained=False)
                
                # Replace the final classifier layer to match model's output classes
                model.classifier[1] = torch.nn.Conv2d(
                    in_channels=512,
                    out_channels=self.model_output_classes,
                    kernel_size=1
                )
                
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                # Try to find a key that might contain the model
                for key, value in checkpoint.items():
                    if isinstance(value, torch.nn.Module):
                        model = value
                        break
                else:
                    # If no model is found, create a SqueezeNet as a placeholder
                    print("Creating placeholder SqueezeNet model")
                    model = squeezenet1_1(pretrained=False)
                    
                    # Replace the final classifier layer to match model's output classes
                    model.classifier[1] = torch.nn.Conv2d(
                        in_channels=512,
                        out_channels=self.model_output_classes,
                        kernel_size=1
                    )
        else:
            # If it's directly a model
            model = checkpoint
        
        model.eval()
        return model
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        tensor = self.transform(rgb_image)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def predict_orientation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict orientation for a player image.
        
        Args:
            image: Input image of a player (BGR format)
            
        Returns:
            Dictionary containing orientation information
        """
        # Preprocess image
        tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(tensor)
        
        # Process output
        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        
        # The model has 8 classes but we only care about 3 orientations
        # Map the 8 model classes to our 3 orientation classes
        # This is a simplified mapping - you might need to adjust based on actual model behavior
        left_prob = probs[:3].sum()  # Assume first 3 classes are "left" variations
        right_prob = probs[3:6].sum()  # Assume next 3 are "right" variations
        neutral_prob = probs[6:].sum()  # Assume last 2 are "neutral" variations
        
        orientation_probs = [left_prob, right_prob, neutral_prob]
        predicted_class = np.argmax(orientation_probs)
        orientation = self.orientation_classes[predicted_class]
        confidence = float(orientation_probs[predicted_class])
        
        return {
            "orientation": orientation,
            "confidence": confidence,
            "probabilities": {
                self.orientation_classes[i]: float(orientation_probs[i])
                for i in range(len(self.orientation_classes))
            }
        }
    
    def process_player_crops(self, crops: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """
        Process multiple player crops to determine their orientations.
        
        Args:
            crops: Dictionary mapping detection indices to player crops
            
        Returns:
            Dictionary mapping detection indices to orientation information
        """
        orientations = {}
        
        for idx, crop in crops.items():
            # Skip invalid crops
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
                
            # Predict orientation
            result = self.predict_orientation(crop)
            orientations[idx] = result
        
        return orientations
    
    def visualize_orientation(self, image: np.ndarray, orientation: str, confidence: float) -> np.ndarray:
        """
        Visualize orientation on the player image.
        
        Args:
            image: Input image of a player (BGR format)
            orientation: Predicted orientation
            confidence: Confidence score
            
        Returns:
            Image with visualized orientation
        """
        vis_image = image.copy()
        height, width = image.shape[:2]
        
        # Draw orientation arrow
        arrow_length = int(width * 0.3)
        start_point = (width // 2, height // 2)
        
        if orientation == "left":
            end_point = (start_point[0] - arrow_length, start_point[1])
            color = (0, 0, 255)  # Red
        elif orientation == "right":
            end_point = (start_point[0] + arrow_length, start_point[1])
            color = (0, 255, 0)  # Green
        else:  # neutral
            end_point = (start_point[0], start_point[1] - arrow_length)
            color = (255, 0, 0)  # Blue
        
        # Draw arrow
        cv2.arrowedLine(vis_image, start_point, end_point, color, 2, tipLength=0.3)
        
        # Draw label
        label = f"{orientation} ({confidence:.2f})"
        cv2.putText(vis_image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
