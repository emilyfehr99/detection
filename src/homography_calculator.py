import cv2
import numpy as np
import json
import os
import logging
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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
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
            dest_points["blue_line_left_top"] = (float(blue_lines["left_top"]["x"]), float(blue_lines["left_top"]["y"]))
            dest_points["blue_line_left_bottom"] = (float(blue_lines["left_bottom"]["x"]), float(blue_lines["left_bottom"]["y"]))
            dest_points["blue_line_right_top"] = (float(blue_lines["right_top"]["x"]), float(blue_lines["right_top"]["y"]))
            dest_points["blue_line_right_bottom"] = (float(blue_lines["right_bottom"]["x"]), float(blue_lines["right_bottom"]["y"]))
        
        # Extract center line points
        center_line = self.rink_coordinates.get("additional_points", {}).get("center_line", {})
        if center_line:
            dest_points["center_line_top"] = (float(center_line["top"]["x"]), float(center_line["top"]["y"]))
            dest_points["center_line_bottom"] = (float(center_line["bottom"]["x"]), float(center_line["bottom"]["y"]))
        
        # Extract goal line points
        goal_lines = self.rink_coordinates.get("additional_points", {}).get("goal_lines", {})
        if goal_lines:
            dest_points["goal_line_left_top"] = (float(goal_lines["left_top"]["x"]), float(goal_lines["left_top"]["y"]))
            dest_points["goal_line_left_bottom"] = (float(goal_lines["left_bottom"]["x"]), float(goal_lines["left_bottom"]["y"]))
            dest_points["goal_line_right_top"] = (float(goal_lines["right_top"]["x"]), float(goal_lines["right_top"]["y"]))
            dest_points["goal_line_right_bottom"] = (float(goal_lines["right_bottom"]["x"]), float(goal_lines["right_bottom"]["y"]))
        
        # Extract circle centers
        center_circle = self.rink_coordinates.get("additional_points", {}).get("center_circle", {})
        if center_circle and "center" in center_circle:
            dest_points["center_circle"] = (float(center_circle["center"]["x"]), float(center_circle["center"]["y"]))
            
            # Also add with numeric index to match source points format
            dest_points["center_circle_0"] = (float(center_circle["center"]["x"]), float(center_circle["center"]["y"]))
        
        # Extract faceoff circle centers
        faceoff_circles = self.rink_coordinates.get("additional_points", {}).get("faceoff_circles", {})
        if faceoff_circles:
            # Create a mapping for positions to indices to ensure consistent numbering
            position_to_index = {
                "top_left": 0,
                "top_right": 1, 
                "bottom_left": 2,
                "bottom_right": 3,
                "center_left": 4,
                "center_right": 5,
            }
            
            for position, circle in faceoff_circles.items():
                if "center" in circle:
                    # Original naming format
                    dest_points[f"faceoff_{position}"] = (float(circle["center"]["x"]), float(circle["center"]["y"]))
                    
                    # Add with numeric index to match source points format if position is in our map
                    if position in position_to_index:
                        idx = position_to_index[position]
                        dest_points[f"faceoff_circle_{idx}"] = (float(circle["center"]["x"]), float(circle["center"]["y"]))
        
        return dest_points
    
    def extract_source_points(self, segmentation_features: Dict[str, List[Dict]]) -> Dict[str, Tuple[float, float]]:
        """
        Extract source points from segmentation features.
        
        Args:
            segmentation_features: Dictionary of segmentation features
            
        Returns:
            Dictionary of source points for homography calculation
        """
        source_points = {}
        
        # Process blue lines (left to right)
        if "BlueLine" in segmentation_features:
            blue_lines = sorted(segmentation_features["BlueLine"], 
                               key=lambda x: x["points"][0]["x"] if isinstance(x["points"][0], dict) else x["points"][0][0])
            
            if len(blue_lines) >= 1:
                points = blue_lines[0]["points"]
                if len(points) >= 2:
                    # Get the top and bottom points of the blue line
                    if isinstance(points[0], dict):
                        top_point = (float(points[0]["x"]), float(points[0]["y"]))
                        bottom_point = (float(points[-1]["x"]), float(points[-1]["y"]))
                    else:
                        top_point = (float(points[0][0]), float(points[0][1]))
                        bottom_point = (float(points[-1][0]), float(points[-1][1]))
                    
                    source_points["blue_line_left_top"] = top_point
                    source_points["blue_line_left_bottom"] = bottom_point
            
            if len(blue_lines) >= 2:
                points = blue_lines[1]["points"]
                if len(points) >= 2:
                    # Get the top and bottom points of the blue line
                    if isinstance(points[0], dict):
                        top_point = (float(points[0]["x"]), float(points[0]["y"]))
                        bottom_point = (float(points[-1]["x"]), float(points[-1]["y"]))
                    else:
                        top_point = (float(points[0][0]), float(points[0][1]))
                        bottom_point = (float(points[-1][0]), float(points[-1][1]))
                    
                    source_points["blue_line_right_top"] = top_point
                    source_points["blue_line_right_bottom"] = bottom_point
        
        # Process center line
        if "RedCenterLine" in segmentation_features and len(segmentation_features["RedCenterLine"]) > 0:
            points = segmentation_features["RedCenterLine"][0]["points"]
            if len(points) >= 2:
                # Get the top and bottom points of the center line
                if isinstance(points[0], dict):
                    top_point = (float(points[0]["x"]), float(points[0]["y"]))
                    bottom_point = (float(points[-1]["x"]), float(points[-1]["y"]))
                else:
                    top_point = (float(points[0][0]), float(points[0][1]))
                    bottom_point = (float(points[-1][0]), float(points[-1][1]))
                
                source_points["center_line_top"] = top_point
                source_points["center_line_bottom"] = bottom_point
        
        # Process goal lines (left to right)
        if "GoalLine" in segmentation_features:
            goal_lines = sorted(segmentation_features["GoalLine"], 
                               key=lambda x: x["points"][0]["x"] if isinstance(x["points"][0], dict) else x["points"][0][0])
            
            if len(goal_lines) >= 1:
                points = goal_lines[0]["points"]
                if len(points) >= 2:
                    # Get the top and bottom points of the goal line
                    if isinstance(points[0], dict):
                        top_point = (float(points[0]["x"]), float(points[0]["y"]))
                        bottom_point = (float(points[-1]["x"]), float(points[-1]["y"]))
                    else:
                        top_point = (float(points[0][0]), float(points[0][1]))
                        bottom_point = (float(points[-1][0]), float(points[-1][1]))
                    
                    source_points["goal_line_left_top"] = top_point
                    source_points["goal_line_left_bottom"] = bottom_point
            
            if len(goal_lines) >= 2:
                points = goal_lines[1]["points"]
                if len(points) >= 2:
                    # Get the top and bottom points of the goal line
                    if isinstance(points[0], dict):
                        top_point = (float(points[0]["x"]), float(points[0]["y"]))
                        bottom_point = (float(points[-1]["x"]), float(points[-1]["y"]))
                    else:
                        top_point = (float(points[0][0]), float(points[0][1]))
                        bottom_point = (float(points[-1][0]), float(points[-1][1]))
                    
                    source_points["goal_line_right_top"] = top_point
                    source_points["goal_line_right_bottom"] = bottom_point
        
        # Process center red circle
        if "RedCircle" in segmentation_features and len(segmentation_features["RedCircle"]) > 0:
            for circle in segmentation_features["RedCircle"]:
                if "points" in circle and len(circle["points"]) > 0:
                    # Get the center point of the circle
                    if isinstance(circle["points"][0], dict):
                        center = (float(circle["points"][0]["x"]), float(circle["points"][0]["y"]))
                    else:
                        center = (float(circle["points"][0][0]), float(circle["points"][0][1]))
                    
                    source_points["center_circle"] = center
                    break
        
        # Process faceoff circles
        if "FaceoffCircle" in segmentation_features:
            # Try to identify faceoff circles by their position
            # This is a heuristic approach and might need refinement
            
            # Sort circles by x-coordinate (left to right)
            faceoff_circles = sorted(segmentation_features["FaceoffCircle"], 
                                   key=lambda x: x["points"][0]["x"] if isinstance(x["points"][0], dict) else x["points"][0][0])
            
            # Split circles into left and right sides
            left_circles = []
            right_circles = []
            
            if len(faceoff_circles) > 0:
                # Find the center x-coordinate (approximate)
                all_x = [circle["points"][0]["x"] if isinstance(circle["points"][0], dict) else circle["points"][0][0] 
                       for circle in faceoff_circles]
                center_x = sum(all_x) / len(all_x)
                
                # Split circles based on center x
                for circle in faceoff_circles:
                    circle_x = circle["points"][0]["x"] if isinstance(circle["points"][0], dict) else circle["points"][0][0]
                    if circle_x < center_x:
                        left_circles.append(circle)
                    else:
                        right_circles.append(circle)
                
                # Sort left and right circles by y-coordinate (top to bottom)
                left_circles = sorted(left_circles, 
                                    key=lambda x: x["points"][0]["y"] if isinstance(x["points"][0], dict) else x["points"][0][1])
                right_circles = sorted(right_circles, 
                                     key=lambda x: x["points"][0]["y"] if isinstance(x["points"][0], dict) else x["points"][0][1])
                
                # Extract points for left circles
                if len(left_circles) >= 1 and len(left_circles[0]["points"]) > 0:
                    if isinstance(left_circles[0]["points"][0], dict):
                        center = (float(left_circles[0]["points"][0]["x"]), float(left_circles[0]["points"][0]["y"]))
                    else:
                        center = (float(left_circles[0]["points"][0][0]), float(left_circles[0]["points"][0][1]))
                    source_points["faceoff_top_left"] = center
                
                if len(left_circles) >= 2 and len(left_circles[1]["points"]) > 0:
                    if isinstance(left_circles[1]["points"][0], dict):
                        center = (float(left_circles[1]["points"][0]["x"]), float(left_circles[1]["points"][0]["y"]))
                    else:
                        center = (float(left_circles[1]["points"][0][0]), float(left_circles[1]["points"][0][1]))
                    source_points["faceoff_bottom_left"] = center
                
                # Extract points for right circles
                if len(right_circles) >= 1 and len(right_circles[0]["points"]) > 0:
                    if isinstance(right_circles[0]["points"][0], dict):
                        center = (float(right_circles[0]["points"][0]["x"]), float(right_circles[0]["points"][0]["y"]))
                    else:
                        center = (float(right_circles[0]["points"][0][0]), float(right_circles[0]["points"][0][1]))
                    source_points["faceoff_top_right"] = center
                
                if len(right_circles) >= 2 and len(right_circles[1]["points"]) > 0:
                    if isinstance(right_circles[1]["points"][0], dict):
                        center = (float(right_circles[1]["points"][0]["x"]), float(right_circles[1]["points"][0]["y"]))
                    else:
                        center = (float(right_circles[1]["points"][0][0]), float(right_circles[1]["points"][0][1]))
                    source_points["faceoff_bottom_right"] = center
        
        # Log the number of source points found
        self.logger.info(f"Found {len(source_points)} source points for homography calculation")
        for point_name, point_coord in source_points.items():
            self.logger.debug(f"Source point {point_name}: {point_coord}")
        
        # Need at least 4 points to compute a valid homography matrix
        if len(source_points) < 4:
            self.logger.warning("Not enough source points for homography calculation")
        
        return source_points
    
    def calculate_homography(self, segmentation_features: Dict[str, List[Dict]]) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """
        Calculate the homography matrix between broadcast footage and the 2D rink.
        
        Args:
            segmentation_features: Dictionary containing segmentation features
            
        Returns:
            Tuple containing:
              - Success status (bool)
              - Homography matrix (numpy array) or None if failed
              - Debug info (dict) with extracted source points and more details
        """
        # Extract source points from segmentation
        source_points = self.extract_source_points(segmentation_features)
        
        debug_info = {
            "source_points": source_points,
            "extracted_points_count": len(source_points),
            "reason_for_failure": None
        }
        
        # Log the extracted source points
        self.logger.info(f"Found {len(source_points)} source points for homography calculation")
        
        if len(source_points) < 4:
            self.logger.warning("Not enough source points for homography calculation")
            debug_info["reason_for_failure"] = "insufficient_points"
            return False, None, debug_info
        
        # Get corresponding destination points
        dest_points = self.get_destination_points()
        
        # Find common points between source and destination
        common_points = sorted(set(source_points.keys()) & set(dest_points.keys()))
        
        # Verify we have enough common points
        if len(common_points) < 4:
            print(f"Common points between source and destination: {len(common_points)}")
            print(f"Common points: {common_points}")
            debug_info["common_points"] = common_points
            debug_info["reason_for_failure"] = "insufficient_common_points"
            return False, None, debug_info
        
        # Prepare point arrays for homography calculation
        src_pts = np.array([source_points[name] for name in common_points], dtype=np.float32)
        dst_pts = np.array([dest_points[name] for name in common_points], dtype=np.float32)
        
        try:
            # Calculate homography matrix
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                self.logger.warning("Homography calculation failed")
                debug_info["reason_for_failure"] = "findHomography_returned_none"
                return False, None, debug_info
            
            # Update smoothing cache
            self.recent_matrices.append(H)
            self.last_valid_matrix = H
            self.matrix_age = 0
            
            return True, H, debug_info
            
        except Exception as e:
            self.logger.error(f"Error calculating homography: {e}")
            debug_info["reason_for_failure"] = f"exception: {str(e)}"
            return False, None, debug_info
    
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
        
    def draw_segmentation_lines(self, frame: np.ndarray, segmentation_features: Dict) -> np.ndarray:
        """
        Draw all segmentation features on a frame.
        
        Args:
            frame: Input frame to draw on
            segmentation_features: Dictionary containing segmentation features
            
        Returns:
            Frame with drawn segmentation features
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Draw blue lines
        if "BlueLine" in segmentation_features:
            for blue_line in segmentation_features["BlueLine"]:
                if "points" in blue_line:
                    pts = [(int(p["x"]), int(p["y"])) for p in blue_line["points"]]
                    for i in range(len(pts) - 1):
                        cv2.line(vis_frame, pts[i], pts[i + 1], (255, 0, 0), 2)  # Blue color
                elif "center" in blue_line:
                    center = (int(blue_line["center"]["x"]), int(blue_line["center"]["y"]))
                    radius = int(blue_line.get("radius", 50))
                    cv2.circle(vis_frame, center, 10, (255, 0, 0), -1)  # Blue dot
                    cv2.putText(vis_frame, "Blue Line", (center[0] + 15, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw red center lines
        if "RedCenterLine" in segmentation_features:
            for red_line in segmentation_features["RedCenterLine"]:
                if "points" in red_line:
                    pts = [(int(p["x"]), int(p["y"])) for p in red_line["points"]]
                    for i in range(len(pts) - 1):
                        cv2.line(vis_frame, pts[i], pts[i + 1], (0, 0, 255), 2)  # Red color
                elif "center" in red_line:
                    center = (int(red_line["center"]["x"]), int(red_line["center"]["y"]))
                    radius = int(red_line.get("radius", 50))
                    cv2.circle(vis_frame, center, 10, (0, 0, 255), -1)  # Red dot
                    cv2.putText(vis_frame, "Red Center Line", (center[0] + 15, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw goal lines - changed to draw diagonal lines
        if "GoalLine" in segmentation_features:
            goal_lines = segmentation_features["GoalLine"]
            
            # Group goal lines by position (left and right)
            left_goal_points = []
            right_goal_points = []
            
            for goal_line in goal_lines:
                if "points" in goal_line:
                    pts = [(int(p["x"]), int(p["y"])) for p in goal_line["points"]]
                    
                    # Use the x-coordinate of the first point to determine if it's left or right
                    # This is a heuristic - if the points are on the left side of the frame, consider it left goal line
                    if pts[0][0] < frame.shape[1] // 2:
                        left_goal_points.extend(pts)
                    else:
                        right_goal_points.extend(pts)
                    
                    # Do not draw the individual segments anymore
                    # for i in range(len(pts) - 1):
                    #     cv2.line(vis_frame, pts[i], pts[i + 1], (255, 0, 255), 1)  # Magenta (thinner line)
                
                elif "center" in goal_line:
                    center = (int(goal_line["center"]["x"]), int(goal_line["center"]["y"]))
                    cv2.circle(vis_frame, center, 10, (255, 0, 255), -1)  # Magenta dot
                    cv2.putText(vis_frame, "Goal Line", (center[0] + 15, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Draw the diagonal goal lines if we have enough points
            if left_goal_points:
                # Find the furthest top and bottom points for true ends of line
                left_goal_points.sort(key=lambda p: p[1])  # Sort by y-coordinate
                top_left = left_goal_points[0]
                bottom_left = left_goal_points[-1]
                
                # Draw diagonal line connecting top and bottom points
                cv2.line(vis_frame, top_left, bottom_left, (255, 0, 255), 2)  # Magenta diagonal line
                
                # Add labels
                cv2.putText(vis_frame, "Left Goal Line", (top_left[0] - 40, top_left[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if right_goal_points:
                # Find the furthest top and bottom points for true ends of line
                right_goal_points.sort(key=lambda p: p[1])  # Sort by y-coordinate  
                top_right = right_goal_points[0]
                bottom_right = right_goal_points[-1]
                
                # Draw diagonal line connecting top and bottom points
                cv2.line(vis_frame, top_right, bottom_right, (255, 0, 255), 2)  # Magenta diagonal line
                
                # Add labels
                cv2.putText(vis_frame, "Right Goal Line", (top_right[0] - 40, top_right[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw red circles
        if "RedCircle" in segmentation_features:
            for circle in segmentation_features["RedCircle"]:
                if "center" in circle:
                    center = (int(circle["center"]["x"]), int(circle["center"]["y"]))
                    radius = int(circle.get("radius", 50))
                    cv2.circle(vis_frame, center, radius, (0, 255, 255), 2)  # Yellow
                    cv2.circle(vis_frame, center, 5, (0, 255, 255), -1)  # Yellow center dot
                    cv2.putText(vis_frame, "Red Circle", (center[0] + 15, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # If there are ellipse parameters
                if all(k in circle for k in ["center", "axes", "angle"]):
                    center = (int(circle["center"]["x"]), int(circle["center"]["y"]))
                    axes = (int(circle["axes"]["width"]), int(circle["axes"]["height"]))
                    angle = float(circle["angle"])
                    if all(a > 0 for a in axes):  # Ensure both axes are positive
                        cv2.ellipse(vis_frame, center, axes, angle, 0, 360, (255, 255, 0), 1)  # Yellow highlight
        
        # Draw faceoff circles
        if "FaceoffCircle" in segmentation_features:
            for circle in segmentation_features["FaceoffCircle"]:
                # Handle circle format with "center" key
                if "center" in circle:
                    center = (int(circle["center"]["x"]), int(circle["center"]["y"]))
                    radius = int(circle.get("radius", 50))
                    cv2.circle(vis_frame, center, radius, (0, 255, 0), 2)  # Green
                    cv2.circle(vis_frame, center, 5, (0, 255, 0), -1)  # Green center dot
                    
                    # Add coordinates as label
                    label = f"({center[0]}, {center[1]})"
                    cv2.putText(vis_frame, label, (center[0] + 5, center[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    cv2.putText(vis_frame, "Faceoff Circle", (center[0] + 15, center[1] + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Handle circle format with "points" key (our new format)
                elif "points" in circle and len(circle["points"]) > 0:
                    # Center point is the first point
                    center_point = circle["points"][0]
                    center = (int(center_point["x"]), int(center_point["y"]))
                    
                    # Get radius from major/minor axis
                    if "major_axis" in circle and "minor_axis" in circle:
                        major = float(circle["major_axis"])
                        minor = float(circle["minor_axis"])
                        axes = (int(major/2), int(minor/2))
                        angle = float(circle.get("angle", 0))
                        
                        # Draw the ellipse
                        if all(a > 0 for a in axes):  # Ensure both axes are positive
                            cv2.ellipse(vis_frame, center, axes, angle, 0, 360, (0, 255, 0), 2)  # Green ellipse
                            
                            # Draw a highlight outline
                            highlight_axes = (axes[0] + 2, axes[1] + 2)
                            cv2.ellipse(vis_frame, center, highlight_axes, angle, 0, 360, (255, 255, 0), 1)
                    elif "equivalent_radius" in circle:
                        # Use equivalent radius if available
                        radius = int(circle["equivalent_radius"])
                        cv2.circle(vis_frame, center, radius, (0, 255, 0), 2)
                    else:
                        # Default radius
                        radius = 50
                        cv2.circle(vis_frame, center, radius, (0, 255, 0), 2)
                    
                    # Draw center point
                    cv2.circle(vis_frame, center, 5, (0, 255, 0), -1)
                    
                    # Add coordinates label
                    label = f"({center[0]}, {center[1]})"
                    cv2.putText(vis_frame, label, (center[0] + 5, center[1] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Add faceoff circle label
                    cv2.putText(vis_frame, "Faceoff Circle", (center[0] + 15, center[1] + 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # If there are ellipse parameters in the old format
                if all(k in circle for k in ["center", "axes", "angle"]):
                    center = (int(circle["center"]["x"]), int(circle["center"]["y"]))
                    axes = (int(circle["axes"]["width"]), int(circle["axes"]["height"]))
                    angle = float(circle["angle"])
                    if all(a > 0 for a in axes):  # Ensure both axes are positive
                        cv2.ellipse(vis_frame, center, axes, angle, 0, 360, (0, 255, 0), 2)  # Green ellipse
                        # Draw a thin highlight outline
                        highlight_axes = (axes[0] + 2, axes[1] + 2)
                        cv2.ellipse(vis_frame, center, highlight_axes, angle, 0, 360, (255, 255, 0), 1)  # Yellow highlight
        
        return vis_frame

    def draw_visualization(self, frame: np.ndarray, segmentation_features: Dict[str, List[Dict]], homography: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw a visualization of the homography and segmentation on the frame.
        
        Args:
            frame: The broadcast frame
            segmentation_features: Dictionary containing segmentation features
            homography: Optional homography matrix
            
        Returns:
            Visualization image
        """
        # Create a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Always draw the segmentation features even if homography fails
        vis_frame = self.draw_segmentation_lines(vis_frame, segmentation_features)
        
        # Draw the homography points and lines only if we have a valid homography matrix
        if homography is not None:
            # Draw rink boundaries
            rink_corners = np.array([
                [0, 0],
                [self.rink_width, 0],
                [self.rink_width, self.rink_height],
                [0, self.rink_height]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Project rink corners to the broadcast frame
            try:
                projected_corners = cv2.perspectiveTransform(rink_corners, np.linalg.inv(homography))
                projected_corners = projected_corners.astype(int)
                
                # Comment out rink boundaries to remove the blue shape
                # cv2.polylines(vis_frame, [projected_corners], True, (255, 0, 0), 2)
                
            except Exception as e:
                self.logger.error(f"Error projecting rink corners: {e}")
        else:
            # If no homography, add a text message explaining
            cv2.putText(vis_frame, "Homography calculation failed", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return vis_frame
