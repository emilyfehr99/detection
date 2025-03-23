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
        
        # NEW: Maximum number of matrices to store for temporal smoothing
        self.max_matrices = 5
        
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
        
        # Process faceoff circles - IMPORTANT: Add these to source points
        if "FaceoffCircle" in segmentation_features and len(segmentation_features["FaceoffCircle"]) > 0:
            faceoff_circles = segmentation_features["FaceoffCircle"]
            
            # Sort faceoff circles by x-coordinate (left to right)
            sorted_circles = sorted(faceoff_circles, 
                                     key=lambda fc: fc["points"][0]["x"] if isinstance(fc["points"][0], dict) else fc["points"][0][0])
            
            # Get the faceoff circle names from the rink coordinates
            faceoff_names = ["left_top", "left_bottom", "right_top", "right_bottom"]
            
            # Process each faceoff circle - use the exact same naming convention as in get_destination_points
            if len(sorted_circles) >= 1:
                # If only one circle is detected, it's likely on the left or right side
                if isinstance(sorted_circles[0]["points"][0], dict):
                    center_x = float(sorted_circles[0]["points"][0]["x"])
                    center_y = float(sorted_circles[0]["points"][0]["y"])
                else:
                    center_x = float(sorted_circles[0]["points"][0][0])
                    center_y = float(sorted_circles[0]["points"][0][1])
                
                # Use the frame center to determine if it's on the left or right side
                frame_center = self.broadcast_width / 2
                if center_x < frame_center:
                    # Left side - could be top or bottom
                    if center_y < self.broadcast_height / 2:
                        source_points["faceoff_top_left"] = (center_x, center_y)
                        # Also add with numeric index to match destination points format
                        source_points["faceoff_circle_0"] = (center_x, center_y)
                    else:
                        source_points["faceoff_bottom_left"] = (center_x, center_y)
                        # Also add with numeric index to match destination points format
                        source_points["faceoff_circle_2"] = (center_x, center_y)
                else:
                    # Right side - could be top or bottom
                    if center_y < self.broadcast_height / 2:
                        source_points["faceoff_top_right"] = (center_x, center_y)
                        # Also add with numeric index to match destination points format
                        source_points["faceoff_circle_1"] = (center_x, center_y)
                    else:
                        source_points["faceoff_bottom_right"] = (center_x, center_y)
                        # Also add with numeric index to match destination points format
                        source_points["faceoff_circle_3"] = (center_x, center_y)
            
            # If we have two circles
            if len(sorted_circles) >= 2:
                for i, fc in enumerate(sorted_circles):
                    if "points" in fc and len(fc["points"]) > 0:
                        # Get center point
                        if isinstance(fc["points"][0], dict):
                            center_x = float(fc["points"][0]["x"])
                            center_y = float(fc["points"][0]["y"])
                        else:
                            center_x = float(fc["points"][0][0])
                            center_y = float(fc["points"][0][1])
                        
                        # Determine which faceoff circle this is based on position
                        frame_center_x = self.broadcast_width / 2
                        frame_center_y = self.broadcast_height / 2
                        
                        if center_x < frame_center_x:  # Left side
                            if center_y < frame_center_y:  # Top
                                source_points["faceoff_top_left"] = (center_x, center_y)
                                source_points["faceoff_circle_0"] = (center_x, center_y)
                            else:  # Bottom
                                source_points["faceoff_bottom_left"] = (center_x, center_y)
                                source_points["faceoff_circle_2"] = (center_x, center_y)
                        else:  # Right side
                            if center_y < frame_center_y:  # Top
                                source_points["faceoff_top_right"] = (center_x, center_y)
                                source_points["faceoff_circle_1"] = (center_x, center_y)
                            else:  # Bottom
                                source_points["faceoff_bottom_right"] = (center_x, center_y)
                                source_points["faceoff_circle_3"] = (center_x, center_y)
        
        # Process goal lines (left to right)
        if "GoalLine" in segmentation_features:
            # First, print some debug info about the goal lines
            print(f"Found {len(segmentation_features['GoalLine'])} goal lines")
            
            # Initialize variables to track positions of features
            left_faceoff_x = None
            right_faceoff_x = None
            
            # If we have faceoff circles, get their positions to use as references
            if "FaceoffCircle" in segmentation_features and len(segmentation_features["FaceoffCircle"]) > 0:
                faceoff_circles = segmentation_features["FaceoffCircle"]
                for fc in faceoff_circles:
                    if "points" in fc and len(fc["points"]) > 0:
                        # Get center point
                        if isinstance(fc["points"][0], dict):
                            center_x = fc["points"][0]["x"]
                        else:
                            center_x = fc["points"][0][0]
                            
                        # Track leftmost and rightmost faceoff circle x positions
                        if left_faceoff_x is None or center_x < left_faceoff_x:
                            left_faceoff_x = center_x
                        if right_faceoff_x is None or center_x > right_faceoff_x:
                            right_faceoff_x = center_x
            
            # If we have RedCircle (which can also be faceoff circles), check them too
            if "RedCircle" in segmentation_features and len(segmentation_features["RedCircle"]) > 0:
                red_circles = segmentation_features["RedCircle"]
                for rc in red_circles:
                    if "points" in rc and len(rc["points"]) > 0:
                        # Get center point
                        if isinstance(rc["points"][0], dict):
                            center_x = rc["points"][0]["x"]
                        else:
                            center_x = rc["points"][0][0]
                            
                        # Track leftmost and rightmost circle x positions
                        if left_faceoff_x is None or center_x < left_faceoff_x:
                            left_faceoff_x = center_x
                        if right_faceoff_x is None or center_x > right_faceoff_x:
                            right_faceoff_x = center_x
            
            # Print reference positions if available
            if left_faceoff_x is not None:
                print(f"Leftmost faceoff circle x position: {left_faceoff_x}")
            if right_faceoff_x is not None:
                print(f"Rightmost faceoff circle x position: {right_faceoff_x}")
            
            # Sort goal lines by their x-coordinate to get left to right order
            goal_lines = sorted(segmentation_features["GoalLine"], 
                               key=lambda x: x["points"][0]["x"] if isinstance(x["points"][0], dict) else x["points"][0][0])
            
            # Print positions of each goal line for debugging
            for i, gl in enumerate(goal_lines):
                first_point = gl["points"][0]
                x_pos = first_point["x"] if isinstance(first_point, dict) else first_point[0]
                print(f"Goal line {i} x-position: {x_pos}")
            
            # We need at least one goal line
            if len(goal_lines) >= 1:
                # IMPROVED APPROACH: Determine if we truly have left and right goal lines
                # by measuring the distance between furthest detected goal line segments
                
                # Get x positions for all goal line segments
                x_positions = []
                for gl in goal_lines:
                    points = gl["points"]
                    if isinstance(points[0], dict):
                        x_coords = [p["x"] for p in points]
                    else:
                        x_coords = [p[0] for p in points]
                    
                    avg_x = sum(x_coords) / len(x_coords)
                    x_positions.append(avg_x)
                
                # Calculate the span of goal line x positions and the frame width ratio
                x_min = min(x_positions)
                x_max = max(x_positions)
                x_span = x_max - x_min
                x_span_ratio = x_span / self.broadcast_width
                
                print(f"Goal line x-span: {x_span}, ratio to frame width: {x_span_ratio:.3f}")
                
                # If the x-span is large enough (typically > 65% of frame width), 
                # we likely have both left and right goal lines
                have_two_goal_lines = x_span_ratio > 0.65
                
                if have_two_goal_lines:
                    print("Detected both left and right goal lines")
                    # For two goal lines, use the leftmost 25% for left and rightmost 25% for right
                    left_threshold = x_min + (x_span * 0.25)
                    right_threshold = x_max - (x_span * 0.25)
                    
                    left_goal_segments = [gl for i, gl in enumerate(goal_lines) 
                                        if x_positions[i] < left_threshold]
                    right_goal_segments = [gl for i, gl in enumerate(goal_lines) 
                                         if x_positions[i] > right_threshold]
                else:
                    print("Detected only one goal line")
                    # If only one goal line is visible, determine if it's left or right
                    # based on its position relative to the center of the frame or relative to detected faceoff circles
                    frame_center = self.broadcast_width / 2
                    
                    # If we have faceoff circles, use them as reference
                    if left_faceoff_x is not None and right_faceoff_x is not None:
                        faceoff_center = (left_faceoff_x + right_faceoff_x) / 2
                        
                        # If goal lines are to the left of the faceoff circles center, they are left goal lines
                        if x_max < faceoff_center:
                            print("Goal line is on the left side of the rink (based on faceoff circles)")
                            left_goal_segments = goal_lines
                            right_goal_segments = []
                        # If goal lines are to the right of the faceoff circles center, they are right goal lines
                        else:
                            print("Goal line is on the right side of the rink (based on faceoff circles)")
                            right_goal_segments = goal_lines
                            left_goal_segments = []
                    else:
                        # Fall back to using frame center
                        if x_max < frame_center:
                            print("Goal line is on the left side of the frame")
                            left_goal_segments = goal_lines
                            right_goal_segments = []
                        else:
                            print("Goal line is on the right side of the frame")
                            right_goal_segments = goal_lines
                            left_goal_segments = []
                
                # NEW: Combine goal line segments into a single diagonal line
                # Process left goal line
                if left_goal_segments:
                    # Get all points from all segments
                    all_points = []
                    for segment in left_goal_segments:
                        for point in segment["points"]:
                            if isinstance(point, dict):
                                all_points.append((float(point["x"]), float(point["y"])))
                            else:
                                all_points.append((float(point[0]), float(point[1])))
                    
                    # Sort points by y-coordinate (top to bottom)
                    all_points.sort(key=lambda p: p[1])
                    
                    # Get the topmost and bottommost points
                    if all_points:
                        # Use the first (topmost) and last (bottommost) points
                        top_point = all_points[0]
                        bottom_point = all_points[-1]
                        
                        print(f"Selected left goal line at {top_point} (top) and {bottom_point} (bottom)")
                        
                        source_points["goal_line_left_top"] = top_point
                        source_points["goal_line_left_bottom"] = bottom_point
                
                # Process right goal line
                if right_goal_segments:
                    # Get all points from all segments
                    all_points = []
                    for segment in right_goal_segments:
                        for point in segment["points"]:
                            if isinstance(point, dict):
                                all_points.append((float(point["x"]), float(point["y"])))
                            else:
                                all_points.append((float(point[0]), float(point[1])))
                    
                    # Sort points by y-coordinate (top to bottom)
                    all_points.sort(key=lambda p: p[1])
                    
                    # Get the topmost and bottommost points
                    if all_points:
                        # Use the first (topmost) and last (bottommost) points
                        top_point = all_points[0]
                        bottom_point = all_points[-1]
                        
                        print(f"Selected right goal line at {top_point} (top) and {bottom_point} (bottom)")
                        
                        source_points["goal_line_right_top"] = top_point
                        source_points["goal_line_right_bottom"] = bottom_point
        
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
            "source_points": {name: point for name, point in source_points.items()},
            "destination_points": {name: point for name, point in self.get_destination_points().items()},
            "common_point_names": []
        }
        
        # Enhanced debugging output
        print("Source points available:")
        for name, point in source_points.items():
            print(f"  - {name}: {point}")
        
        # Get corresponding destination points
        dest_points = self.get_destination_points()
        
        print("Destination points available:")
        for name, point in dest_points.items():
            print(f"  - {name}: {point}")
        
        # Log the extracted source points
        self.logger.info(f"Found {len(source_points)} source points for homography calculation")
        
        if len(source_points) < 4:
            debug_info["reason_for_failure"] = f"Not enough source points found (need ≥4, found {len(source_points)})"
            return False, np.eye(3), debug_info
        
        # Find common points between source and destination 
        # AND create mappings for points that have different but related names
        common_point_names = list(set(source_points.keys()) & set(dest_points.keys()))
        print(f"Common point names before mapping: {common_point_names}")
        
        # Add mapped point pairs
        point_mapping = {}
        
        # Add specific mappings for faceoff circles - UPDATED to match actual point names
        faceoff_mapping = {
            "faceoff_left_top": ["faceoff_circle_0", "faceoff_top_left"],
            "faceoff_left_bottom": ["faceoff_circle_2", "faceoff_bottom_left"],
            "faceoff_right_top": ["faceoff_circle_1", "faceoff_top_right"],
            "faceoff_right_bottom": ["faceoff_circle_3", "faceoff_bottom_right"],
        }
        
        # Also check the reverse mapping - UPDATED to match actual point names
        reverse_faceoff_mapping = {
            "faceoff_circle_0": ["faceoff_left_top"],
            "faceoff_circle_1": ["faceoff_right_top"],
            "faceoff_circle_2": ["faceoff_left_bottom"],
            "faceoff_circle_3": ["faceoff_right_bottom"],
            "faceoff_top_left": ["faceoff_left_top"],
            "faceoff_bottom_left": ["faceoff_left_bottom"],
            "faceoff_top_right": ["faceoff_right_top"],
            "faceoff_bottom_right": ["faceoff_right_bottom"],
        }
        
        # IMPROVED MAPPINGS: Ensure correct mapping between detected lines and rink coordinates
        # First handle direct matches
        for name in list(source_points.keys()):
            if name in dest_points:
                common_point_names.append(name)
        
        # Handle goal line mappings more carefully
        # When we detect a goal line on the left side, it should map to the left goal line in the rink
        # Similarly for the right side
        if "goal_line_left_top" in source_points and "goal_line_left_bottom" in source_points:
            # Check if we have a proper match with destination points
            if "goal_line_left_top" not in dest_points or "goal_line_left_bottom" not in dest_points:
                # Map to any available goal line in destination
                if "goal_line_left_top" in dest_points:
                    point_mapping["goal_line_left_top"] = "goal_line_left_top"
                    point_mapping["goal_line_left_bottom"] = "goal_line_left_bottom"
                    common_point_names.extend(["goal_line_left_top", "goal_line_left_bottom"])
                elif "goal_line_right_top" in dest_points:
                    # If no left goal line in dest, try right side
                    point_mapping["goal_line_left_top"] = "goal_line_right_top"
                    point_mapping["goal_line_left_bottom"] = "goal_line_right_bottom"
                    common_point_names.extend(["goal_line_right_top", "goal_line_right_bottom"])
        
        if "goal_line_right_top" in source_points and "goal_line_right_bottom" in source_points:
            # Check if we have a proper match with destination points
            if "goal_line_right_top" not in dest_points or "goal_line_right_bottom" not in dest_points:
                # Map to any available goal line in destination
                if "goal_line_right_top" in dest_points:
                    point_mapping["goal_line_right_top"] = "goal_line_right_top"
                    point_mapping["goal_line_right_bottom"] = "goal_line_right_bottom"
                    common_point_names.extend(["goal_line_right_top", "goal_line_right_bottom"])
                elif "goal_line_left_top" in dest_points:
                    # If no right goal line in dest, try left side
                    point_mapping["goal_line_right_top"] = "goal_line_left_top"
                    point_mapping["goal_line_right_bottom"] = "goal_line_left_bottom"
                    common_point_names.extend(["goal_line_left_top", "goal_line_left_bottom"])
        
        # Add mappings from faceoff names to numeric indices
        for dest_name, src_names in faceoff_mapping.items():
            if dest_name in dest_points:
                for src_name in src_names:
                    if src_name in source_points and dest_name not in common_point_names:
                        point_mapping[src_name] = dest_name
                        common_point_names.append(dest_name)
                        print(f"Adding mapping: {src_name} -> {dest_name}")
        
        # Add mappings from numeric indices to faceoff names
        for src_name, dest_names in reverse_faceoff_mapping.items():
            if src_name in source_points:
                for dest_name in dest_names:
                    if dest_name in dest_points and dest_name not in common_point_names:
                        point_mapping[src_name] = dest_name
                        common_point_names.append(dest_name)
                        print(f"Adding reverse mapping: {src_name} -> {dest_name}")
        
        # Ensure no duplicate entries in common_point_names
        common_point_names = list(dict.fromkeys(common_point_names))
        
        debug_info["common_point_names"] = common_point_names
        debug_info["point_mapping"] = point_mapping
        
        print(f"Common point names after mapping: {common_point_names}")
        print(f"Point mappings: {point_mapping}")
        
        if len(common_point_names) < 4:
            debug_info["reason_for_failure"] = f"Not enough common points found (need ≥4, found {len(common_point_names)})"
            return False, np.eye(3), debug_info
        
        # Build the source and destination point arrays
        src_points = []
        dst_points = []
        
        used_common_points = []
        
        for point_name in common_point_names:
            # Check if this is a direct match or if we need to use a mapping
            if point_name in source_points:
                # Direct match - point name exists in both source and destination points
                src_points.append(source_points[point_name])
                dst_points.append(dest_points[point_name])
                used_common_points.append(point_name)
            elif point_name in point_mapping.values():
                # This is a destination name that we mapped to - need to find the source name
                src_name = None
                for s_name, d_name in point_mapping.items():
                    if d_name == point_name and s_name in source_points:
                        src_name = s_name
                        break
                
                if src_name:
                    src_points.append(source_points[src_name])
                    dst_points.append(dest_points[point_name])
                    used_common_points.append(f"{src_name} -> {point_name}")
            
        debug_info["used_common_points"] = used_common_points
        
        print(f"Used {len(src_points)} common points for homography calculation:")
        for i, name in enumerate(used_common_points):
            src_idx = i if i < len(src_points) else -1
            dst_idx = i if i < len(dst_points) else -1
            src_point = src_points[src_idx] if src_idx >= 0 else "N/A"
            dst_point = dst_points[dst_idx] if dst_idx >= 0 else "N/A"
            print(f"  - {name}: {src_point} -> {dst_point}")
        
        if len(src_points) < 4:
            debug_info["reason_for_failure"] = f"Not enough common points found for computation (need ≥4, found {len(src_points)})"
            return False, np.eye(3), debug_info
        
        # Convert to numpy arrays
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        
        # Use RANSAC with multiple thresholds to find the best homography
        best_homography = None
        best_inliers = 0
        best_error = float('inf')
        
        # Try multiple RANSAC thresholds to find the best homography
        # (higher threshold = more lenient, lower threshold = more strict)
        ransac_thresholds = [5.0, 10.0, 15.0, 20.0, 25.0]
        for threshold in ransac_thresholds:
            # Calculate homography using RANSAC
            homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
            
            if homography is None:
                continue
                
            # Count inliers
            inliers = np.sum(mask)
            
            # Calculate reprojection error for this homography
            error = self.compute_reprojection_error(homography, src_points, dst_points)
            
            # Choose homography with lowest error if it has enough inliers
            # or the one with the most inliers if the error is comparable
            if (inliers > best_inliers and error < best_error * 1.5) or \
               (inliers >= best_inliers * 0.8 and error < best_error):
                best_homography = homography
                best_inliers = inliers
                best_error = error
                debug_info["best_ransac_threshold"] = threshold
                debug_info["inlier_count"] = int(inliers)
                debug_info["reprojection_error"] = float(error)
        
        if best_homography is None:
            debug_info["reason_for_failure"] = "Failed to calculate homography with any RANSAC threshold"
            return False, np.eye(3), debug_info
        
        # Validate the homography to ensure it's reasonable
        if not self.validate_homography(best_homography):
            debug_info["reason_for_failure"] = "Homography validation failed"
            return False, best_homography, debug_info
        
        return True, best_homography, debug_info
    
    def compute_reprojection_error(self, homography: np.ndarray, 
                                   src_points: np.ndarray, dst_points: np.ndarray) -> float:
        """
        Compute the reprojection error for a homography matrix.
        
        Args:
            homography: The homography matrix
            src_points: Source points (from broadcast frame)
            dst_points: Destination points (from 2D rink)
            
        Returns:
            Average reprojection error
        """
        # Transform source points using the homography
        src_points_homogeneous = np.hstack([src_points, np.ones((src_points.shape[0], 1))])
        transformed_points_homogeneous = np.dot(homography, src_points_homogeneous.T).T
        
        # Convert back from homogeneous coordinates
        transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
        
        # Calculate euclidean distance between transformed points and destination points
        errors = np.sqrt(np.sum((transformed_points - dst_points) ** 2, axis=1))
        
        # Return average error
        return np.mean(errors)
    
    def validate_homography(self, homography_matrix: np.ndarray) -> bool:
        """
        Validate the homography matrix by checking if transformed corners stay within reasonable bounds
        and if the transformation preserves the expected orientation.
        
        Args:
            homography_matrix: Homography matrix to validate
            
        Returns:
            True if the homography is valid, False otherwise
        """
        # Transform the four corners of the broadcast frame
        h, w = self.broadcast_height, self.broadcast_width
        corners = np.array([
            [0, 0],        # top-left
            [0, h-1],      # bottom-left
            [w-1, h-1],    # bottom-right
            [w-1, 0]       # top-right
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform corners
        transformed_corners = cv2.perspectiveTransform(corners, homography_matrix).reshape(-1, 2)
        
        # Check if transformed corners are within a reasonable range of the rink
        # Allow some margin outside the rink (2x the rink dimensions)
        margin = 2.0
        rink_bounds_x = [-self.rink_width * (margin - 1), self.rink_width * margin]
        rink_bounds_y = [-self.rink_height * (margin - 1), self.rink_height * margin]
        
        within_bounds = all(
            rink_bounds_x[0] <= x <= rink_bounds_x[1] and
            rink_bounds_y[0] <= y <= rink_bounds_y[1]
            for x, y in transformed_corners
        )
        
        # NEW: Check if the transformation preserves orientation
        # In a hockey rink, top-left and top-right should have smaller y values than bottom-left and bottom-right
        def is_top(idx):
            return idx == 0 or idx == 3  # top-left or top-right
        
        def is_bottom(idx):
            return idx == 1 or idx == 2  # bottom-left or bottom-right
        
        orientation_preserved = True
        for i in range(4):
            for j in range(4):
                if is_top(i) and is_bottom(j):
                    # A top corner should have a smaller y value than a bottom corner
                    if transformed_corners[i][1] > transformed_corners[j][1]:
                        orientation_preserved = False
                        break
        
        if not orientation_preserved:
            print("Homography validation failed - orientation not preserved")
            
        return within_bounds and orientation_preserved
    
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
                    cv2.circle(vis_frame, center, radius, (255, 0, 0), 2)  # Blue circle
                    cv2.circle(vis_frame, center, 5, (255, 0, 0), -1)  # Blue center dot
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
                    cv2.circle(vis_frame, center, radius, (0, 0, 255), 2)  # Red circle
                    cv2.circle(vis_frame, center, 5, (0, 0, 255), -1)  # Red center dot
                    cv2.putText(vis_frame, "Red Center Line", (center[0] + 15, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw goal lines - changed to draw diagonal lines
        if "GoalLine" in segmentation_features:
            goal_lines = segmentation_features["GoalLine"]
            
            # Group goal lines by position (left and right)
            left_goal_points = []
            right_goal_points = []
            
            # First pass - classify lines using all points to get a more robust classification
            for goal_line in goal_lines:
                if "points" in goal_line:
                    pts = [(int(p["x"]), int(p["y"])) for p in goal_line["points"]]
                    
                    # Use the average x-coordinate of all points to determine if it's left or right
                    # This is more robust than just using the first point
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    
                    if avg_x < frame.shape[1] // 2:
                        left_goal_points.extend(pts)
                    else:
                        right_goal_points.extend(pts)
                    
                    # Do not draw the individual segments anymore
                    # for i in range(len(pts) - 1):
                    #     cv2.line(vis_frame, pts[i], pts[i + 1], (255, 0, 255), 1)  # Magenta (thinner line)
                
                elif "center" in goal_line:
                    center = (int(goal_line["center"]["x"]), int(goal_line["center"]["y"]))
                    cv2.circle(vis_frame, center, 10, (255, 0, 255), -1)  # Magenta dot
                    
                    # Use the x-coordinate to determine left/right
                    if center[0] < frame.shape[1] // 2:
                        label = "Left Goal Line"
                    else:
                        label = "Right Goal Line"
                    
                    cv2.putText(vis_frame, label, (center[0] + 15, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Draw the goal lines if we have enough points
            if left_goal_points:
                # Find the furthest top and bottom points for true ends of line
                left_goal_points.sort(key=lambda p: p[1])  # Sort by y-coordinate
                top_left = left_goal_points[0]
                bottom_left = left_goal_points[-1]
                
                # Draw vertical line
                cv2.line(vis_frame, top_left, bottom_left, (255, 0, 255), 2)  # Magenta vertical line
                
                # Add label only at the top
                cv2.putText(vis_frame, "Left Goal Line", (top_left[0] - 40, top_left[1] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            if right_goal_points:
                # Find the furthest top and bottom points for true ends of line
                right_goal_points.sort(key=lambda p: p[1])  # Sort by y-coordinate  
                top_right = right_goal_points[0]
                bottom_right = right_goal_points[-1]
                
                # Draw vertical line
                cv2.line(vis_frame, top_right, bottom_right, (255, 0, 255), 2)  # Magenta vertical line
                
                # Add label only at the top
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
