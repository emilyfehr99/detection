import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RINK_CLASS_MAPPING = {
    0: "Rink",
    1: "BlueLine",
    2: "RedCenterLine",
    3: "GoalLine",
    4: "RedCircle",
    5: "FaceoffCircle"
}

class SegmentationProcessor:
    """Process frames through a segmentation model."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ):
        """
        Initialize the SegmentationProcessor.
        
        Args:
            model_path: Path to the segmentation model
            confidence_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Add state for tracking circles between frames
        self.prev_circles = {}  # Maps circle_id to (x, y, frame_last_seen)
        self.next_circle_id = 0  # Counter for assigning unique IDs
        self.max_frames_to_keep = 10  # How many frames to remember circles
        self.max_dist_for_match = 100  # Maximum pixel distance to consider same circle
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the segmentation model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_frame(
        self, frame: np.ndarray, frame_id: int = None, output_dir: str = None
    ) -> Dict[str, List[Dict]]:
        """
        Process a single frame through the segmentation model.
        
        Args:
            frame: The frame to process
            frame_id: Optional frame identifier for saving debug images
            output_dir: Optional directory to save debug outputs
            
        Returns:
            Dictionary containing segmentation results
        """
        # Run inference with the model
        results = self.model(frame)
        
        # No results
        if len(results) == 0:
            logger.warning("No segmentation results produced")
            return {"segmentation_mask": None, "features": {}}
        
        result = results[0]
        features = {}
        
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data
            if len(masks) > 0:
                # Extract classes from results
                classes = result.boxes.cls.cpu().numpy()
                
                # Create mask by class
                mask_by_class = {}
                
                for i, mask in enumerate(masks):
                    class_idx = int(classes[i])
                    
                    # Convert class index to name using mapping if possible
                    if hasattr(self.model, 'names') and self.model.names:
                        class_name = self.model.names.get(
                            class_idx, f"class_{class_idx}"
                        )
                    else:
                        # Use hardcoded mapping if model doesn't provide names
                        class_name = RINK_CLASS_MAPPING.get(
                            class_idx, f"class_{class_idx}"
                        )
                    
                    # Create or update mask for this class
                    if class_name not in mask_by_class:
                        mask_by_class[class_name] = np.zeros(
                            (frame.shape[0], frame.shape[1]), dtype=bool
                        )
                    
                    # Convert mask tensor to numpy and add to class mask
                    numpy_mask = mask.cpu().numpy()
                    resized_mask = cv2.resize(
                        numpy_mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0])
                    )
                    mask_by_class[class_name] = np.logical_or(
                        mask_by_class[class_name], 
                        resized_mask > 0
                    )
                
                # Extract features from segmentation masks
                features = self._extract_features_from_segmentation(mask_by_class)
                
                # Create colored segmentation mask for visualization
                colored_mask = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
                
                # Color mapping for different classes
                color_map = {
                    "Rink": (0, 200, 0),       # Green
                    "BlueLine": (255, 0, 0),   # Blue
                    "RedCenterLine": (0, 0, 255),  # Red
                    "GoalLine": (255, 0, 255),  # Magenta
                    "RedCircle": (0, 255, 255),  # Yellow
                    "FaceoffCircle": (255, 255, 0)  # Cyan
                }
                
                # Apply colors to the mask
                for class_name, mask in mask_by_class.items():
                    if class_name in color_map:
                        color = color_map[class_name]
                        colored_mask[mask] = color
                
                # Save debug visualizations if requested
                if output_dir and frame_id is not None:
                    vis_img = self._save_debug_visualizations(
                        frame, mask_by_class, features, output_dir, frame_id
                    )
                
                # Return colored mask with features and overlay visualization
                return {
                    "segmentation_mask": colored_mask,
                    "features": features,
                    "raw_masks": mask_by_class,
                    "overlay_visualization": (
                        vis_img if output_dir and frame_id is not None else None
                    )
                }
        
        # If we didn't get any masks, return empty results
        return {"segmentation_mask": None, "features": {}}

    def _extract_features_from_segmentation(
        self, mask_by_class: Dict[str, np.ndarray]
    ) -> Dict[str, List[Dict]]:
        """
        Extract features from segmentation masks.
        
        Args:
            mask_by_class: Dictionary mapping class names to mask arrays
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Process each class mask
        for class_name, mask in mask_by_class.items():
            # Convert mask to uint8 for contour detection (0-255)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Skip masks with no content
            mask_sum = np.sum(mask_uint8)
            if mask_sum == 0:
                continue
            
            # Add debug logging for BlueLine class
            if class_name == "BlueLine":
                logger.info(f"BlueLine mask sum: {mask_sum}")
                logger.info(f"BlueLine mask unique values: {np.unique(mask_uint8)}")
            
            # Log which classes are being processed
            logger.info(f"Processing mask for class: {class_name}")
            
            # Extract appropriate features based on class name
            if class_name in ["BlueLine", "RedCenterLine", "GoalLine"]:
                # For lines, extract line segments
                features[class_name] = self._extract_line_segments(mask_uint8, class_name)
            elif class_name in ["RedCircle", "FaceoffCircle"]:
                # For circles, extract ellipses
                if class_name == "RedCircle":
                    # Treat RedCircle as FaceoffCircle since that's what they actually are
                    circle_features = self._extract_circles(mask_uint8)
                    if circle_features:
                        if "FaceoffCircle" not in features:
                            features["FaceoffCircle"] = []
                        features["FaceoffCircle"].extend(circle_features)
                        logger.info(f"Converted {len(circle_features)} RedCircle features to FaceoffCircle")
                else:
                    features[class_name] = self._extract_circles(mask_uint8)
        
        # Remove the synthetic faceoff circle code since we're using the detected ones
        # If no faceoff circles were detected, we're fine with that
        
        return features

    def _extract_line_segments(self, binary_mask, class_name=None):
        """Extract line segments from binary mask."""
        # Special handling for blue lines to ensure proper separation
        if class_name == "BlueLine":
            # Get the center line position
            center_x = binary_mask.shape[1] // 2
            
            # Create separate masks for left and right blue lines
            left_mask = binary_mask.copy()
            right_mask = binary_mask.copy()
            
            # Zero out the right side of the left mask
            left_mask[:, center_x:] = 0
            # Zero out the left side of the right mask
            right_mask[:, :center_x] = 0
            
            # Process each side separately
            left_contours, _ = cv2.findContours(
                left_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            right_contours, _ = cv2.findContours(
                right_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter and process contours
            min_area = 200
            all_points = []  # Single list for all points
            
            # Process left contours
            for contour in left_contours:
                if cv2.contourArea(contour) > min_area:
                    contour_points = contour.reshape(-1, 2)
                    all_points.extend(contour_points)
            
            # Process right contours
            for contour in right_contours:
                if cv2.contourArea(contour) > min_area:
                    contour_points = contour.reshape(-1, 2)
                    all_points.extend(contour_points)
            
            # Return a single feature for the blue line
            features = []
            if all_points:
                # Convert to numpy array for easier manipulation
                all_points = np.array(all_points)
                # Sort by y-coordinate to find true endpoints
                sorted_by_y = all_points[np.argsort(all_points[:, 1])]
                top_point = sorted_by_y[0]
                bottom_point = sorted_by_y[-1]
                features.append({
                    'points': [
                        {"x": int(top_point[0]), "y": int(top_point[1])},
                        {"x": int(bottom_point[0]), "y": int(bottom_point[1])}
                    ]
                })
            return features
        
        # For other line types, use the original logic
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter out very small contours (noise)
        min_area = 200  # Increased from 100
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Special handling for goal lines to connect broken segments
        if class_name == "GoalLine":
            # Get all points from all contours
            all_points = []
            for contour in contours:
                contour_points = contour.reshape(-1, 2)
                all_points.extend(contour_points)
                
            if all_points:
                # Convert to numpy array for easier manipulation
                all_points = np.array(all_points)
                
                # Calculate average x position to determine if this is left or right goal line
                avg_x = np.mean(all_points[:, 0])
                frame_center = binary_mask.shape[1] / 2
                
                if avg_x < frame_center:
                    # Left goal line - find absolute bottom-left and top-right points
                    sorted_by_y = all_points[np.argsort(-all_points[:, 1])]
                    max_y = sorted_by_y[0][1]
                    lowest_points = sorted_by_y[sorted_by_y[:, 1] == max_y]
                    bottom_left = lowest_points[np.argmin(lowest_points[:, 0])]
                    
                    sorted_by_y = all_points[np.argsort(all_points[:, 1])]
                    min_y = sorted_by_y[0][1]
                    highest_points = sorted_by_y[sorted_by_y[:, 1] == min_y]
                    top_right = highest_points[np.argmax(highest_points[:, 0])]
                    
                    points = [
                        {"x": int(top_right[0]), "y": int(top_right[1])},
                        {"x": int(bottom_left[0]), "y": int(bottom_left[1])}
                    ]
                else:
                    # Right goal line - find absolute bottom-right and top-left points
                    sorted_by_y = all_points[np.argsort(-all_points[:, 1])]
                    max_y = sorted_by_y[0][1]
                    lowest_points = sorted_by_y[sorted_by_y[:, 1] == max_y]
                    bottom_right = lowest_points[np.argmax(lowest_points[:, 0])]
                    
                    sorted_by_y = all_points[np.argsort(all_points[:, 1])]
                    min_y = sorted_by_y[0][1]
                    highest_points = sorted_by_y[sorted_by_y[:, 1] == min_y]
                    top_left = highest_points[np.argmin(highest_points[:, 0])]
                    
                    points = [
                        {"x": int(top_left[0]), "y": int(top_left[1])},
                        {"x": int(bottom_right[0]), "y": int(bottom_right[1])}
                    ]
                
                return [{'points': points}]
        
        # For other line types or if goal line processing failed
        points = []
        for contour in contours:
            # Get all points from contour
            contour_points = contour.reshape(-1, 2)
            
            # Sort points by y-coordinate
            sorted_points = sorted(contour_points, key=lambda p: p[1])
            
            # Get the top and bottom points
            top_point = sorted_points[0]
            bottom_point = sorted_points[-1]
            
            # Add points in expected format
            points.extend([
                {"x": int(top_point[0]), "y": int(top_point[1])},
                {"x": int(bottom_point[0]), "y": int(bottom_point[1])}
            ])
        
        return [{'points': points}]

    def _extract_circles(self, mask: np.ndarray, min_radius: int = 10) -> List[Dict]:
        """
        Extract elliptical shapes from a binary mask and maintain consistent labeling.
        
        Args:
            mask: Binary mask containing circle/ellipse objects
            min_radius: Minimum equivalent radius for shapes to be considered
            
        Returns:
            List of dictionaries containing ellipse information with consistent IDs
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract circles without IDs first
        current_circles = []
        for contour in contours:
            # Get the area of the contour
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < np.pi * min_radius ** 2:
                continue
            
            # Need at least 5 points to fit an ellipse
            if len(contour) < 5:
                continue
                
            # Fit an ellipse to the contour
            try:
                ellipse = cv2.fitEllipse(contour)
                (x, y), (major_axis, minor_axis), angle = ellipse
                
                # Calculate equivalent radius (geometric mean of semi-axes)
                equivalent_radius = np.sqrt(major_axis * minor_axis) / 2
                
                if equivalent_radius < min_radius:
                    continue
                
                current_circles.append({
                    'x': float(x),
                    'y': float(y),
                    'major_axis': float(major_axis),
                    'minor_axis': float(minor_axis),
                    'angle': float(angle),
                    'equivalent_radius': float(equivalent_radius)
                })
            except cv2.error:
                continue
        
        # Remove old circles from tracking
        current_frame = getattr(self, 'current_frame', 0)
        self.prev_circles = {
            circle_id: info for circle_id, info in self.prev_circles.items()
            if current_frame - info[2] <= self.max_frames_to_keep
        }
        
        # Match current circles with previous circles
        matched_circles = []
        used_prev_circles = set()
        
        # Sort circles by x-coordinate for more stable matching
        current_circles.sort(key=lambda c: c['x'])
        
        for circle in current_circles:
            best_match_id = None
            best_match_dist = float('inf')
            
            # Find the closest previous circle
            for circle_id, (prev_x, prev_y, _) in self.prev_circles.items():
                if circle_id in used_prev_circles:
                    continue
                    
                dist = np.sqrt((circle['x'] - prev_x)**2 + (circle['y'] - prev_y)**2)
                if dist < self.max_dist_for_match and dist < best_match_dist:
                    best_match_id = circle_id
                    best_match_dist = dist
            
            if best_match_id is not None:
                # Update existing circle
                circle_id = best_match_id
                used_prev_circles.add(circle_id)
            else:
                # Create new circle ID
                circle_id = self.next_circle_id
                self.next_circle_id += 1
            
            # Update tracking state
            self.prev_circles[circle_id] = (circle['x'], circle['y'], current_frame)
            
            # Create final circle dict with consistent ID
            circle_dict = {
                "points": [{"x": int(circle['x']), "y": int(circle['y'])}],
                "major_axis": circle['major_axis'],
                "minor_axis": circle['minor_axis'],
                "angle": circle['angle'],
                "equivalent_radius": circle['equivalent_radius'],
                "circle_id": circle_id  # Add consistent ID to output
            }
            matched_circles.append(circle_dict)
        
        # Increment frame counter
        setattr(self, 'current_frame', current_frame + 1)
        
        return matched_circles

    def _save_debug_visualizations(self, frame: np.ndarray, mask_by_class: Dict[str, np.ndarray], 
                                  features: Dict[str, List[Dict]], output_dir: str, frame_id: int) -> np.ndarray:
        """
        Save debug visualizations for the segmentation process.
        
        Args:
            frame: Original frame
            mask_by_class: Dictionary mapping class names to mask arrays
            features: Extracted features
            output_dir: Directory to save visualizations
            frame_id: Frame identifier
        
        Returns:
            The overlay visualization image
        """
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(output_dir, "debug_segmentation")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save combined mask visualization
        vis_img = frame.copy()
        
        # Use different colors for each class
        colors = {
            "Rink": (0, 255, 0, 50),         # Green with low opacity
            "BlueLine": (255, 0, 0, 200),     # Blue with high opacity
            "RedCenterLine": (0, 0, 255, 200), # Red with high opacity
            "GoalLine": (255, 0, 255, 200),   # Magenta with high opacity
            "RedCircle": (0, 255, 255, 200),  # Yellow with high opacity
            "FaceoffCircle": (255, 255, 0, 200) # Cyan with high opacity
        }
        
        # Draw masks with different colors and opacities
        for class_name, mask in mask_by_class.items():
            if class_name in colors:
                color = colors[class_name]
                
                # Create a colored overlay for this mask
                overlay = np.zeros_like(frame, dtype=np.uint8)
                overlay[mask] = color[:3]
                
                # Apply the overlay with the specified opacity
                alpha = color[3] / 255.0
                cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
        
        # Save the visualization
        cv2.imwrite(os.path.join(debug_dir, f"segmentation_{frame_id:04d}.jpg"), vis_img)
        
        return vis_img
