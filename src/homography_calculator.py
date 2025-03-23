import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import deque


class HomographyCalculator:
    """
    Calculates homography transformation between broadcast footage and the 2D rink model.
    Maps detected rink features in broadcast footage to their corresponding positions in the 2D rink.
    """

    def __init__(self, rink_coordinates_path: str, broadcast_width: int = 1280, broadcast_height: int = 720):
        """
        Initialize the HomographyCalculator with rink coordinates.
        
        Args:
            rink_coordinates_path: Path to the JSON file containing rink coordinates
            broadcast_width: Width of the broadcast footage (default: 1280)
            broadcast_height: Height of the broadcast footage (default: 720)
        """
        self.rink_coordinates_path = rink_coordinates_path
        self.broadcast_width = broadcast_width
        self.broadcast_height = broadcast_height
        
        # Default rink dimensions
        self.rink_width = 1400
        self.rink_height = 600
        
        # Load rink coordinates
        self.rink_coordinates = self._load_rink_coordinates()
        
        # Cache for homography matrices
        self.homography_cache = {}
        
        # For homography smoothing
        self.recent_matrices = deque(maxlen=10)  # Store recent valid matrices
        self.last_valid_matrix = None  # Store the last valid matrix
        self.matrix_age = 0  # Track how old the last_valid_matrix is
        
    def _load_rink_coordinates(self) -> Dict:
        """
        Load rink coordinates from JSON file.
        
        Returns:
            Dictionary containing rink coordinates
        """
        if not os.path.exists(self.rink_coordinates_path):
            raise FileNotFoundError(f"Rink coordinates file not found: {self.rink_coordinates_path}")
        
        with open(self.rink_coordinates_path, 'r') as f:
            rink_coordinates = json.load(f)
            
        # Update rink dimensions if available in coordinates
        if 'rink_dimensions' in rink_coordinates:
            self.rink_width = rink_coordinates['rink_dimensions'].get('width', self.rink_width)
            self.rink_height = rink_coordinates['rink_dimensions'].get('height', self.rink_height)
            
        return rink_coordinates
    
    def get_destination_points(self) -> Dict[str, Tuple[float, float]]:
        """
        Get destination points in the 2D rink model.
        
        Returns:
            Dictionary mapping point names to coordinates
        """
        dest_points = {}
        
        # Extract blue line points
        blue_lines = self.rink_coordinates.get("additional_points", {}).get("blue_lines", {})
        if blue_lines:
            dest_points["blue_left_top"] = (float(blue_lines["left_top"]["x"]), float(blue_lines["left_top"]["y"]))
            dest_points["blue_right_top"] = (float(blue_lines["right_top"]["x"]), float(blue_lines["right_top"]["y"]))
            dest_points["blue_left_bottom"] = (float(blue_lines["left_bottom"]["x"]), float(blue_lines["left_bottom"]["y"]))
            dest_points["blue_right_bottom"] = (float(blue_lines["right_bottom"]["x"]), float(blue_lines["right_bottom"]["y"]))
        
        # Extract center line points
        center_line = self.rink_coordinates.get("additional_points", {}).get("center_line", {})
        if center_line:
            dest_points["center_top"] = (float(center_line["top"]["x"]), float(center_line["top"]["y"]))
            dest_points["center_bottom"] = (float(center_line["bottom"]["x"]), float(center_line["bottom"]["y"]))
        
        # Extract goal line points
        goal_lines = self.rink_coordinates.get("additional_points", {}).get("goal_lines", {})
        if goal_lines:
            dest_points["goal_left_top"] = (float(goal_lines["left_top"]["x"]), float(goal_lines["left_top"]["y"]))
            dest_points["goal_left_bottom"] = (float(goal_lines["left_bottom"]["x"]), float(goal_lines["left_bottom"]["y"]))
            dest_points["goal_right_top"] = (float(goal_lines["right_top"]["x"]), float(goal_lines["right_top"]["y"]))
            dest_points["goal_right_bottom"] = (float(goal_lines["right_bottom"]["x"]), float(goal_lines["right_bottom"]["y"]))
        
        # Extract circle centers
        center_circle = self.rink_coordinates.get("additional_points", {}).get("center_circle", {})
        if center_circle and "center" in center_circle:
            dest_points["center_circle"] = (float(center_circle["center"]["x"]), float(center_circle["center"]["y"]))
        
        # Extract faceoff circle centers
        faceoff_circles = self.rink_coordinates.get("additional_points", {}).get("faceoff_circles", {})
        if faceoff_circles:
            for position, circle in faceoff_circles.items():
                if "center" in circle:
                    dest_points[f"faceoff_{position}"] = (float(circle["center"]["x"]), float(circle["center"]["y"]))
        
        return dest_points
    
    def extract_source_points(self, segmentation_features: Dict[str, List]) -> Dict[str, Tuple[float, float]]:
        """
        Extract source points from segmentation features.
        
        Args:
            segmentation_features: Dictionary containing detected features from segmentation
            
        Returns:
            Dictionary mapping point names to coordinates in broadcast footage
        """
        source_points = {}
        
        # Extract blue line points
        if "BlueLine" in segmentation_features:
            blue_lines = sorted(segmentation_features["BlueLine"], key=lambda x: x["points"][0]["x"])
            if len(blue_lines) >= 2:
                # Left blue line
                left_blue = blue_lines[0]
                left_points = sorted(left_blue["points"], key=lambda p: p["y"])
                if len(left_points) >= 2:
                    source_points["blue_left_top"] = (float(left_points[0]["x"]), float(left_points[0]["y"]))
                    source_points["blue_left_bottom"] = (float(left_points[-1]["x"]), float(left_points[-1]["y"]))
                
                # Right blue line
                right_blue = blue_lines[1]
                right_points = sorted(right_blue["points"], key=lambda p: p["y"])
                if len(right_points) >= 2:
                    source_points["blue_right_top"] = (float(right_points[0]["x"]), float(right_points[0]["y"]))
                    source_points["blue_right_bottom"] = (float(right_points[-1]["x"]), float(right_points[-1]["y"]))
        
        # Extract center line points
        if "RedCenterLine" in segmentation_features and segmentation_features["RedCenterLine"]:
            center_line = segmentation_features["RedCenterLine"][0]
            center_points = sorted(center_line["points"], key=lambda p: p["y"])
            if len(center_points) >= 2:
                source_points["center_top"] = (float(center_points[0]["x"]), float(center_points[0]["y"]))
                source_points["center_bottom"] = (float(center_points[-1]["x"]), float(center_points[-1]["y"]))
        
        # Extract goal line points
        if "GoalLine" in segmentation_features:
            goal_lines = sorted(segmentation_features["GoalLine"], key=lambda x: x["points"][0]["x"])
            if len(goal_lines) >= 2:
                # Left goal line
                left_goal = goal_lines[0]
                left_points = sorted(left_goal["points"], key=lambda p: p["y"])
                if len(left_points) >= 2:
                    source_points["goal_left_top"] = (float(left_points[0]["x"]), float(left_points[0]["y"]))
                    source_points["goal_left_bottom"] = (float(left_points[-1]["x"]), float(left_points[-1]["y"]))
                
                # Right goal line
                right_goal = goal_lines[1]
                right_points = sorted(right_goal["points"], key=lambda p: p["y"])
                if len(right_points) >= 2:
                    source_points["goal_right_top"] = (float(right_points[0]["x"]), float(right_points[0]["y"]))
                    source_points["goal_right_bottom"] = (float(right_points[-1]["x"]), float(right_points[-1]["y"]))
        
        # Extract circle centers
        if "RedCircle" in segmentation_features:
            for circle in segmentation_features["RedCircle"]:
                if "center" in circle:
                    # Need to classify circles based on position
                    # This is a simplified approach - in practice, you would need more sophisticated classification
                    center_x = float(circle["center"]["x"])
                    center_y = float(circle["center"]["y"])
                    
                    # For now, just add as an unclassified circle
                    source_points[f"circle_{len([k for k in source_points.keys() if k.startswith('circle')])}"] = (center_x, center_y)
        
        return source_points
    
    def get_average_matrix(self) -> Optional[np.ndarray]:
        """
        Calculate an average homography matrix from recent valid matrices.
        
        Returns:
            Average homography matrix or None if no recent matrices
        """
        if not self.recent_matrices:
            return None
        
        # Simple average of recent matrices
        # Note: In a more sophisticated implementation, you might want to use
        # a weighted average that gives more importance to more recent matrices
        avg_matrix = np.mean([m for m in self.recent_matrices], axis=0)
        return avg_matrix
    
    def calculate_homography(self, segmentation_features: Dict[str, List], frame_idx: int) -> Tuple[np.ndarray, bool]:
        """
        Calculate homography matrix from broadcast to rink coordinates.
        Uses a smoothing approach when direct calculation fails.
        
        Args:
            segmentation_features: Dictionary containing detected features from segmentation
            frame_idx: Current frame index for tracking
            
        Returns:
            Homography matrix and success flag
        """
        # Debug: Print the available features
        print(f"\nFeatures detected in frame {frame_idx}:")
        for feature_type, features in segmentation_features.items():
            print(f"  {feature_type}: {len(features)} instances")
            if features:
                for i, feature in enumerate(features):
                    if "points" in feature:
                        print(f"    Instance {i}: {len(feature['points'])} points")
                    elif "center" in feature:
                        print(f"    Instance {i}: center at ({feature['center']['x']:.1f}, {feature['center']['y']:.1f})")
        
        # Extract source and destination points
        source_points = self.extract_source_points(segmentation_features)
        dest_points = self.get_destination_points()
        
        # Debug: Print the extracted source points
        print(f"Extracted source points: {len(source_points)} points")
        for name, point in source_points.items():
            print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")
        
        # Find common point keys between source and destination
        common_keys = set(source_points.keys()) & set(dest_points.keys())
        print(f"Common points between source and destination: {len(common_keys)}")
        for key in common_keys:
            print(f"  {key}")
        
        # Check if we have enough points for direct homography calculation
        if len(common_keys) >= 4:
            # Create point arrays for homography calculation
            src_pts = np.array([source_points[k] for k in common_keys], dtype=np.float32)
            dst_pts = np.array([dest_points[k] for k in common_keys], dtype=np.float32)
            
            # Calculate homography matrix
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Cache the homography matrix
                self.homography_cache[frame_idx] = H
                # Update our recent matrices for smoothing
                self.recent_matrices.append(H)
                self.last_valid_matrix = H
                self.matrix_age = 0
                return H, True
        
        # If we reach here, either we didn't have enough points or the homography calculation failed
        self.matrix_age += 1
        
        # Try using average of recent matrices
        avg_matrix = self.get_average_matrix()
        if avg_matrix is not None:
            print(f"Using averaged homography matrix for frame {frame_idx}")
            self.homography_cache[frame_idx] = avg_matrix
            return avg_matrix, True
        
        # If no average available, use the last valid matrix if available
        if self.last_valid_matrix is not None:
            print(f"Using last valid homography matrix for frame {frame_idx} (age: {self.matrix_age})")
            self.homography_cache[frame_idx] = self.last_valid_matrix
            return self.last_valid_matrix, True
            
        # If we still don't have a matrix, return failure
        print(f"No valid homography matrix could be calculated or estimated for frame {frame_idx}")
        return None, False
    
    def apply_homography(self, points: List[Tuple[float, float]], homography_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """
        Apply homography transformation to points.
        
        Args:
            points: List of points to transform (x, y)
            homography_matrix: Homography matrix to apply
            
        Returns:
            List of transformed points
        """
        if homography_matrix is None or not points:
            return []
            
        # Convert points to homogeneous coordinates
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Apply transformation
        transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)
        
        # Convert back to list of tuples
        return [(pt[0][0], pt[0][1]) for pt in transformed_pts]
    
    def get_homography_matrix(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get cached homography matrix for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Homography matrix if available, None otherwise
        """
        return self.homography_cache.get(frame_idx)
        
    def warp_frame(self, frame: np.ndarray, homography_matrix: np.ndarray, rink_dims: Tuple[int, int]) -> np.ndarray:
        """
        Warp the broadcast frame to the rink perspective using the homography matrix.
        
        Args:
            frame: Original broadcast frame
            homography_matrix: Homography matrix to apply
            rink_dims: Dimensions of the rink image (width, height)
            
        Returns:
            Warped frame in rink perspective
        """
        if homography_matrix is None:
            return None
            
        # Apply perspective transformation to the entire frame
        warped_frame = cv2.warpPerspective(frame, homography_matrix, rink_dims)
        
        return warped_frame
        
    def draw_segmentation_lines(self, frame: np.ndarray, segmentation_features: Dict[str, List[Dict]]) -> np.ndarray:
        """
        Draw the detected rink features on the frame.
        
        Args:
            frame: Original frame
            segmentation_features: Dictionary of rink features
            
        Returns:
            Frame with drawn features
        """
        vis_frame = frame.copy()
        
        # Define colors for different features
        colors = {
            "BlueLine": (255, 0, 0),      # Blue
            "RedCenterLine": (0, 0, 255), # Red
            "GoalLine": (0, 255, 0),      # Green
            "RedCircle": (0, 0, 255)      # Red
        }
        
        # Draw line features
        for feature_type, features in segmentation_features.items():
            color = colors.get(feature_type, (255, 255, 0))  # Default to yellow
            
            for feature in features:
                if "points" in feature:
                    points = feature["points"]
                    for i in range(len(points) - 1):
                        pt1 = (int(points[i][0]), int(points[i][1]))
                        pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                        cv2.line(vis_frame, pt1, pt2, color, 2)
                elif "center" in feature:
                    center = (int(feature["center"]["x"]), int(feature["center"]["y"]))
                    radius = int(feature.get("radius", 10))
                    cv2.circle(vis_frame, center, radius, color, 2)
                    
        return vis_frame
