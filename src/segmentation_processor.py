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
            if np.sum(mask_uint8) == 0:
                continue
            
            # Log which classes are being processed
            logger.info(f"Processing mask for class: {class_name}")
            
            # Extract appropriate features based on class name
            if class_name in ["BlueLine", "RedCenterLine", "GoalLine"]:
                # For lines, extract line segments
                features[class_name] = self._extract_line_segments(mask_uint8)
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

    def _extract_line_segments(self, binary_mask):
        """Extract line segments from binary mask."""
        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter out very small contours (noise)
        min_area = 50
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Extract line segments from contours
        segments = []
        for contour in contours:
            # Get min area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get endpoints (leftmost and rightmost points)
            leftmost = tuple(
                contour[contour[:, :, 0].argmin()][0]
            )
            rightmost = tuple(
                contour[contour[:, :, 0].argmax()][0]
            )
            topmost = tuple(
                contour[contour[:, :, 1].argmin()][0]
            )
            bottommost = tuple(
                contour[contour[:, :, 1].argmax()][0]
            )
            
            # Calculate segment length
            length = np.sqrt(
                (rightmost[0] - leftmost[0])**2 + 
                (rightmost[1] - leftmost[1])**2
            )
            
            # Use the longer dimension endpoints
            vert_length = np.sqrt(
                (bottommost[1] - topmost[1])**2
            )
            if length > vert_length:
                endpoints = [leftmost, rightmost]
            else:
                endpoints = [topmost, bottommost]
            
            # Calculate center
            center = (
                (endpoints[0][0] + endpoints[1][0])/2,
                (endpoints[0][1] + endpoints[1][1])/2
            )
            
            # Calculate angle
            dx = endpoints[1][0] - endpoints[0][0]
            dy = endpoints[1][1] - endpoints[0][1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            
            segments.append({
                'endpoints': endpoints,
                'center': center,
                'angle': angle,
                'length': length
            })

        # Sort segments by x-coordinate for left-to-right processing
        segments.sort(key=lambda s: s['center'][0])
        
        # Merge segments that are likely part of the same line
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(segments):
                j = i + 1
                while j < len(segments):
                    seg1 = segments[i]
                    seg2 = segments[j]
                    
                    # Calculate endpoints and distances
                    potential_endpoints = []
                    for ep1 in seg1['endpoints']:
                        for ep2 in seg2['endpoints']:
                            dist = np.sqrt(
                                (ep2[0] - ep1[0])**2 + 
                                (ep2[1] - ep1[1])**2
                            )
                            potential_endpoints.append(
                                (ep1, ep2, dist)
                            )
                    
                    # Sort by distance to find furthest endpoints
                    potential_endpoints.sort(
                        key=lambda x: x[2], 
                        reverse=True
                    )
                    ep1, ep2, dist = potential_endpoints[0]
                    
                    # Calculate angle of potential merged line
                    dx = ep2[0] - ep1[0]
                    dy = ep2[1] - ep1[1]
                    merged_angle = np.arctan2(dy, dx) * 180 / np.pi
                    
                    # Calculate x-distance between segments
                    x_dist = abs(
                        seg2['center'][0] - seg1['center'][0]
                    )
                    
                    # Allow more angle difference as x-distance increases
                    # Increase allowed angle difference with distance
                    max_angle_diff = 15 + (x_dist / 100) * 5
                    
                    # Allow more y-difference as x-distance increases
                    # Increase allowed y-difference with distance
                    max_y_diff = 50 + (x_dist / 100) * 20
                    
                    # Check if segments should be merged
                    angle_diff = abs(merged_angle - seg1['angle'])
                    y_diff = abs(
                        seg1['center'][1] - seg2['center'][1]
                    )
                    
                    # Maximum x-distance threshold
                    if (angle_diff < max_angle_diff and 
                            y_diff < max_y_diff and
                            x_dist < 500):
                        
                        # Create new merged segment
                        new_endpoints = [ep1, ep2]
                        new_center = (
                            (ep1[0] + ep2[0])/2,
                            (ep1[1] + ep2[1])/2
                        )
                        new_length = np.sqrt(
                            (ep2[0] - ep1[0])**2 + 
                            (ep2[1] - ep1[1])**2
                        )
                        
                        segments[i] = {
                            'endpoints': new_endpoints,
                            'center': new_center,
                            'angle': merged_angle,
                            'length': new_length
                        }
                        
                        # Remove the second segment
                        segments.pop(j)
                        merged = True
                        break
                    j += 1
                if merged:
                    break
                i += 1
        
        # Convert segments to list of points in expected format
        points = []
        for segment in segments:
            for endpoint in segment['endpoints']:
                points.append({
                    'x': int(endpoint[0]),
                    'y': int(endpoint[1])
                })
        
        # Return points in the expected format
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
