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
        
        # Process center line
        if "RedCenterLine" in segmentation_features and len(segmentation_features["RedCenterLine"]) > 0:
            center_line = segmentation_features["RedCenterLine"][0]
            if "points" in center_line and len(center_line["points"]) >= 2:
                # Get the top and bottom points of the center line
                if isinstance(center_line["points"][0], dict):
                    top_point = (float(center_line["points"][0]["x"]), float(center_line["points"][0]["y"]))
                    bottom_point = (float(center_line["points"][-1]["x"]), float(center_line["points"][-1]["y"]))
                else:
                    top_point = (float(center_line["points"][0][0]), float(center_line["points"][0][1]))
                    bottom_point = (float(center_line["points"][-1][0]), float(center_line["points"][-1][1]))
                
                source_points["center_line_top"] = top_point
                source_points["center_line_bottom"] = bottom_point
        
        # Process blue lines
        if "BlueLine" in segmentation_features and len(segmentation_features["BlueLine"]) > 0:
            blue_lines = sorted(segmentation_features["BlueLine"], 
                               key=lambda x: x["points"][0]["x"] if isinstance(x["points"][0], dict) else x["points"][0][0])
            
            if len(blue_lines) >= 1:
                # If only one blue line is detected, it's likely on the left or right side
                if isinstance(blue_lines[0]["points"][0], dict):
                    center_x = float(blue_lines[0]["points"][0]["x"])
                    center_y = float(blue_lines[0]["points"][0]["y"])
                else:
                    center_x = float(blue_lines[0]["points"][0][0])
                    center_y = float(blue_lines[0]["points"][0][1])
                
                # Use the frame center to determine if it's on the left or right side
                frame_center = self.broadcast_width / 2
                if center_x < frame_center:
                    # Left side - could be top or bottom
                    if center_y < self.broadcast_height / 2:
                        source_points["blue_line_left_top"] = (center_x, center_y)
                    else:
                        source_points["blue_line_left_bottom"] = (center_x, center_y)
                else:
                    # Right side - could be top or bottom
                    if center_y < self.broadcast_height / 2:
                        source_points["blue_line_right_top"] = (center_x, center_y)
                    else:
                        source_points["blue_line_right_bottom"] = (center_x, center_y)
            
            # If we have two or more blue lines, use them as left and right blue lines
            if len(blue_lines) >= 2:
                for i, bl in enumerate(blue_lines):
                    if "points" in bl and len(bl["points"]) > 0:
                        # Get center point
                        if isinstance(bl["points"][0], dict):
                            center_x = float(bl["points"][0]["x"])
                            center_y = float(bl["points"][0]["y"])
                        else:
                            center_x = float(bl["points"][0][0])
                            center_y = float(bl["points"][0][1])
                        
                        # Determine which blue line this is based on position
                        frame_center_x = self.broadcast_width / 2
                        frame_center_y = self.broadcast_height / 2
                        
                        if center_x < frame_center_x:  # Left side
                            if center_y < frame_center_y:  # Top
                                source_points["blue_line_left_top"] = (center_x, center_y)
                            else:  # Bottom
                                source_points["blue_line_left_bottom"] = (center_x, center_y)
                        else:  # Right side
                            if center_y < frame_center_y:  # Top
                                source_points["blue_line_right_top"] = (center_x, center_y)
                            else:  # Bottom
                                source_points["blue_line_right_bottom"] = (center_x, center_y)
        
        # Process faceoff circles with hockey domain knowledge
        faceoff_circles = []
        blue_line_points = []
        
        # First identify blue line points and calculate average x position
        blue_line_x = None
        left_blue_line_x = None
        if "BlueLine" in segmentation_features and segmentation_features["BlueLine"]:
            for bl in segmentation_features["BlueLine"]:
                if "points" in bl and bl["points"]:
                    for point in bl["points"]:
                        if isinstance(point, dict):
                            x = float(point["x"])
                            blue_line_points.append((x, float(point["y"])))
                        else:
                            x = float(point[0])
                            blue_line_points.append((x, float(point[1])))
            
            if blue_line_points:
                # Sort x positions to identify left and right blue lines
                x_positions = sorted(p[0] for p in blue_line_points)
                if len(x_positions) >= 2:
                    left_blue_line_x = x_positions[0]  # Leftmost blue line x position
                blue_line_x = sum(p[0] for p in blue_line_points) / len(blue_line_points)
                print(f"Average blue line x position: {blue_line_x:.1f}")
                if left_blue_line_x:
                    print(f"Left blue line x position: {left_blue_line_x:.1f}")

        # Check for left goal line presence
        has_left_goal_line = False
        if "GoalLine" in segmentation_features and segmentation_features["GoalLine"]:
            goal_lines = segmentation_features["GoalLine"]
            for gl in goal_lines:
                if "points" in gl and gl["points"]:
                    points = gl["points"]
                    if isinstance(points[0], dict):
                        x_positions = [float(p["x"]) for p in points]
                    else:
                        x_positions = [float(p[0]) for p in points]
                    avg_x = sum(x_positions) / len(x_positions)
                    
                    # If we have a blue line reference, use it to determine if this is a left goal line
                    if blue_line_x and avg_x < blue_line_x:
                        has_left_goal_line = True
                        print("Detected left goal line")
                        break
                    # If no blue line reference, use frame center
                    elif avg_x < self.broadcast_width / 2:
                        has_left_goal_line = True
                        print("Detected left goal line (using frame center)")
                        break

        # Process faceoff circles using consistent IDs
        if "FaceoffCircle" in segmentation_features and len(segmentation_features["FaceoffCircle"]) > 0:
            # First collect all faceoff circle points with their IDs
            for fc in segmentation_features["FaceoffCircle"]:
                if "points" in fc and len(fc["points"]) > 0:
                    if isinstance(fc["points"][0], dict):
                        center_x = float(fc["points"][0]["x"])
                        center_y = float(fc["points"][0]["y"])
                    else:
                        center_x = float(fc["points"][0][0])
                        center_y = float(fc["points"][0][1])
                    
                    # Use the consistent circle_id if available
                    circle_id = fc.get("circle_id", None)
                    if circle_id is not None:
                        faceoff_circles.append({
                            "id": circle_id,
                            "x": center_x,
                            "y": center_y
                        })
            
            # Log detected faceoff circles
            if faceoff_circles:
                x_positions = [(c["id"], c["x"]) for c in faceoff_circles]
                print(f"Found {len(faceoff_circles)} faceoff circles: {x_positions}")
                
                # Sort circles by their consistent IDs to maintain ordering
                faceoff_circles.sort(key=lambda c: c["id"])
                
                # HOCKEY RULE 1: If circles are left of ANY blue line, they are LEFT zone circles
                # HOCKEY RULE 2: If we see the left goal line, all circles are LEFT zone circles
                circle_side = None
                
                if has_left_goal_line:
                    # If we see the left goal line, all circles are LEFT zone circles
                    circle_side = "left"
                    print("All faceoff circles are LEFT circles (left goal line visible)")
                elif left_blue_line_x:
                    # If we see a blue line, anything to the left of it MUST be in the left zone
                    if any(circle["x"] < left_blue_line_x for circle in faceoff_circles):
                        circle_side = "left"
                        print("All faceoff circles are LEFT circles (at least one circle left of blue line)")
                    else:
                        circle_side = "right"
                        print("All faceoff circles are RIGHT circles (all circles right of blue line)")
                else:
                    # No blue line or goal line visible, fall back to frame position
                    frame_center_x = self.broadcast_width / 2
                    all_left = all(circle["x"] < frame_center_x for circle in faceoff_circles)
                    all_right = all(circle["x"] > frame_center_x for circle in faceoff_circles)
                    
                    if all_left:
                        print("All faceoff circles appear on left side of frame")
                        circle_side = "left"
                    elif all_right:
                        print("All faceoff circles appear on right side of frame")
                        circle_side = "right"
                    else:
                        print("WARNING: Faceoff circles appear on both sides of frame!")
                        # Use average position as fallback
                        avg_circle_x = sum(c["x"] for c in faceoff_circles) / len(faceoff_circles)
                        circle_side = "left" if avg_circle_x < frame_center_x else "right"
                
                # Label circles based on their consistent IDs
                if len(faceoff_circles) == 1:
                    # Just one circle, use y position to determine top/bottom
                    circle = faceoff_circles[0]
                    frame_center_y = self.broadcast_height / 2
                    position = "top" if circle["y"] < frame_center_y else "bottom"
                    
                    # Add to source points with consistent naming
                    source_points[f"faceoff_{circle_side}_{position}"] = (circle["x"], circle["y"])
                    source_points[f"faceoff_circle_{circle['id']}"] = (circle["x"], circle["y"])
                    
                    # Add redundant naming for backward compatibility
                    if circle_side == "left":
                        source_points[f"faceoff_{position}_left"] = (circle["x"], circle["y"])
                    else:
                        source_points[f"faceoff_{position}_right"] = (circle["x"], circle["y"])
                        
                elif len(faceoff_circles) >= 2:
                    # Multiple circles, use consistent IDs for top/bottom assignment
                    # The circle with lower ID is considered top
                    circles_by_id = sorted(faceoff_circles, key=lambda c: c["id"])
                    
                    # Top circle (lowest ID)
                    top_circle = circles_by_id[0]
                    source_points[f"faceoff_{circle_side}_top"] = (top_circle["x"], top_circle["y"])
                    source_points[f"faceoff_circle_{top_circle['id']}"] = (top_circle["x"], top_circle["y"])
                    
                    # Bottom circle (highest ID)
                    bottom_circle = circles_by_id[-1]
                    source_points[f"faceoff_{circle_side}_bottom"] = (bottom_circle["x"], bottom_circle["y"])
                    source_points[f"faceoff_circle_{bottom_circle['id']}"] = (bottom_circle["x"], bottom_circle["y"])
                    
                    # Add redundant naming for backward compatibility
                    if circle_side == "left":
                        source_points[f"faceoff_top_left"] = (top_circle["x"], top_circle["y"])
                        source_points[f"faceoff_bottom_left"] = (bottom_circle["x"], bottom_circle["y"])
                    else:
                        source_points[f"faceoff_top_right"] = (top_circle["x"], top_circle["y"])
                        source_points[f"faceoff_bottom_right"] = (bottom_circle["x"], bottom_circle["y"])
                    
                    print(f"Classified faceoff circles as {circle_side}_top (ID: {top_circle['id']}) and {circle_side}_bottom (ID: {bottom_circle['id']})")
        
        # Process goal lines as a SINGLE CONTINUOUS LINE regardless of segmentation
        if "GoalLine" in segmentation_features and len(segmentation_features["GoalLine"]) > 0:
            goal_lines = segmentation_features["GoalLine"]
            
            # Collect all goal line points
            all_goal_points = []
            for gl in goal_lines:
                if "points" in gl and len(gl["points"]) > 0:
                    for point in gl["points"]:
                        if isinstance(point, dict):
                            all_goal_points.append((float(point["x"]), float(point["y"])))
                        else:
                            all_goal_points.append((float(point[0]), float(point[1])))
            
            print(f"Found {len(goal_lines)} goal line segments with {len(all_goal_points)} total points")
            
            if all_goal_points:
                # Calculate goal line span
                min_x = min(p[0] for p in all_goal_points)
                max_x = max(p[0] for p in all_goal_points)
                goal_x_span = max_x - min_x
                goal_x_ratio = goal_x_span / self.broadcast_width
                print(f"Goal line x-span: {goal_x_span:.1f}, "
                      f"ratio to frame width: {goal_x_ratio:.3f}")
                
                # If any faceoff circle exists, classify goal line based on leftmost point
                if faceoff_circles:
                    # If leftmost point of goal line is left of any faceoff circle,
                    # it's a LEFT goal line
                    if any(min_x < circle["x"] for circle in faceoff_circles):
                        print(f"Goal line is LEFT (leftmost point {min_x:.1f} "
                              f"is left of faceoff circle)")
                        goal_side = "left"
                    else:
                        print(f"Goal line is RIGHT (leftmost point {min_x:.1f} "
                              f"is not left of any faceoff circle)")
                        goal_side = "right"
                else:
                    # No faceoff circles visible, use frame center
                    frame_center_x = self.broadcast_width / 2
                    goal_side = "left" if min_x < frame_center_x else "right"
                    print(f"Goal line is {goal_side.upper()} (using frame center)")
                
                # Sort goal points by y-coordinate to find top and bottom
                goal_points_sorted_by_y = sorted(all_goal_points, key=lambda p: p[1])
                top_point = goal_points_sorted_by_y[0]
                bottom_point = goal_points_sorted_by_y[-1]
                
                # Set goal line points with consistent naming
                source_points[f"goal_line_{goal_side}_top"] = top_point
                source_points[f"goal_line_{goal_side}_bottom"] = bottom_point
                
                # Remove any incorrect goal line points to avoid confusion
                opposite_side = "right" if goal_side == "left" else "left"
                if f"goal_line_{opposite_side}_top" in source_points:
                    del source_points[f"goal_line_{opposite_side}_top"]
                if f"goal_line_{opposite_side}_bottom" in source_points:
                    del source_points[f"goal_line_{opposite_side}_bottom"]
                
                print(f"Selected {goal_side} goal line endpoints at {top_point} (top) and {bottom_point} (bottom)")
        
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
        
        # ENHANCED: Add blue line mappings - even if we have only one blue line, try to map it
        # to either left or right blue line in the destination points
        blue_line_mapping = {
            "blue_line_left_top": ["blue_line_top", "blue_line_upper"],
            "blue_line_left_bottom": ["blue_line_bottom", "blue_line_lower"],
            "blue_line_right_top": ["blue_line_top", "blue_line_upper"],
            "blue_line_right_bottom": ["blue_line_bottom", "blue_line_lower"]
        }
        
        # Determine which side of the rink the faceoff circles are on
        circle_side = None
        for key in source_points:
            if key == "faceoff_left_top" or key == "faceoff_left_bottom":
                circle_side = "left"
                break
            elif key == "faceoff_right_top" or key == "faceoff_right_bottom":
                circle_side = "right"
                break
        
        print(f"Detected faceoff circles on the {circle_side if circle_side else 'UNKNOWN'} side of the rink")
        
        # Create dynamic faceoff mapping dictionary based on detected side
        faceoff_mapping = {}
        reverse_faceoff_mapping = {}
        
        if circle_side == "left":
            # All circles are on the LEFT side
            print("Creating faceoff mappings for LEFT side of rink")
            faceoff_mapping = {
                "faceoff_left_top": ["faceoff_top_left", "faceoff_circle_0", "faceoff_circle_1"],
                "faceoff_left_bottom": ["faceoff_bottom_left", "faceoff_circle_2", "faceoff_circle_3"],
            }
            reverse_faceoff_mapping = {
                "faceoff_top_left": ["faceoff_left_top"],
                "faceoff_bottom_left": ["faceoff_left_bottom"],
                "faceoff_circle_0": ["faceoff_left_top"],
                "faceoff_circle_1": ["faceoff_left_top"],
                "faceoff_circle_2": ["faceoff_left_bottom"],
                "faceoff_circle_3": ["faceoff_left_bottom"],
            }
        elif circle_side == "right":
            # All circles are on the RIGHT side
            print("Creating faceoff mappings for RIGHT side of rink")
            faceoff_mapping = {
                "faceoff_right_top": ["faceoff_top_right", "faceoff_circle_0", "faceoff_circle_1"],
                "faceoff_right_bottom": ["faceoff_bottom_right", "faceoff_circle_2", "faceoff_circle_3"],
            }
            reverse_faceoff_mapping = {
                "faceoff_top_right": ["faceoff_right_top"],
                "faceoff_bottom_right": ["faceoff_right_bottom"],
                "faceoff_circle_0": ["faceoff_right_top"],
                "faceoff_circle_1": ["faceoff_right_top"],
                "faceoff_circle_2": ["faceoff_right_bottom"],
                "faceoff_circle_3": ["faceoff_right_bottom"],
            }
        else:
            # Fallback: use empty mappings if side couldn't be determined
            print("WARNING: Could not determine faceoff circle side, using empty mappings")
        
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
                        print(f"Adding reverse mapping: {src_name} -> {dest_name} (side: {circle_side})")
        
        # Add blue line mappings
        for dest_name, src_names in blue_line_mapping.items():
            if dest_name in dest_points:
                for src_name in src_names:
                    if src_name in source_points and dest_name not in common_point_names:
                        point_mapping[src_name] = dest_name
                        common_point_names.append(dest_name)
                        print(f"Adding blue line mapping: {src_name} -> {dest_name}")
        
        # IMPROVED MAPPINGS: Ensure correct mapping between detected lines and rink coordinates
        # First handle direct matches for any remaining points
        for name in list(source_points.keys()):
            if name in dest_points and name not in common_point_names:
                common_point_names.append(name)
                print(f"Adding direct mapping: {name}")
        
        # Handle goal line mappings based on the determined side of the rink
        if circle_side == "left":
            # If circles are on the left, goal lines should also be mapped to left
            if "goal_line_left_top" in source_points and "goal_line_left_bottom" in source_points:
                if "goal_line_left_top" in dest_points:
                    if "goal_line_left_top" not in common_point_names:
                        point_mapping["goal_line_left_top"] = "goal_line_left_top"
                        common_point_names.append("goal_line_left_top")
                    if "goal_line_left_bottom" not in common_point_names:
                        point_mapping["goal_line_left_bottom"] = "goal_line_left_bottom"
                        common_point_names.append("goal_line_left_bottom")
                    print("Mapped LEFT goal lines to LEFT goal lines in rink model")
            
            # If we have right goal lines in source, try to map them properly
            if "goal_line_right_top" in source_points and "goal_line_right_bottom" in source_points:
                if "goal_line_right_top" in dest_points:
                    if "goal_line_right_top" not in common_point_names:
                        point_mapping["goal_line_right_top"] = "goal_line_right_top"
                        common_point_names.append("goal_line_right_top")
                    if "goal_line_right_bottom" not in common_point_names:
                        point_mapping["goal_line_right_bottom"] = "goal_line_right_bottom"
                        common_point_names.append("goal_line_right_bottom")
                    print("Mapped RIGHT goal lines to RIGHT goal lines in rink model")
        elif circle_side == "right":
            # If circles are on the right, goal lines should also be mapped to right
            if "goal_line_right_top" in source_points and "goal_line_right_bottom" in source_points:
                if "goal_line_right_top" in dest_points:
                    if "goal_line_right_top" not in common_point_names:
                        point_mapping["goal_line_right_top"] = "goal_line_right_top"
                        common_point_names.append("goal_line_right_top")
                    if "goal_line_right_bottom" not in common_point_names:
                        point_mapping["goal_line_right_bottom"] = "goal_line_right_bottom"
                        common_point_names.append("goal_line_right_bottom")
                    print("Mapped RIGHT goal lines to RIGHT goal lines in rink model")
            
            # If we have left goal lines in source, try to map them properly
            if "goal_line_left_top" in source_points and "goal_line_left_bottom" in source_points:
                if "goal_line_left_top" in dest_points:
                    if "goal_line_left_top" not in common_point_names:
                        point_mapping["goal_line_left_top"] = "goal_line_left_top"
                        common_point_names.append("goal_line_left_top")
                    if "goal_line_left_bottom" not in common_point_names:
                        point_mapping["goal_line_left_bottom"] = "goal_line_left_bottom"
                        common_point_names.append("goal_line_left_bottom")
                    print("Mapped LEFT goal lines to LEFT goal lines in rink model")
        
        # Ensure no duplicate entries in common_point_names
        common_point_names = list(dict.fromkeys(common_point_names))
        
        # IMPROVED: Reduce minimum required points if we have at least some key points
        min_required_points = 4  # Default minimum
        
        # If we have at least one goal line and one blue line, we can try with just 3 points
        has_goal_line = any("goal_line" in name for name in common_point_names)
        has_blue_line = any("blue_line" in name for name in common_point_names)
        
        if has_goal_line and has_blue_line:
            min_required_points = 3
            print("Relaxing homography requirements due to presence of goal line and blue line")
        
        if len(common_point_names) < min_required_points:
            debug_info["reason_for_failure"] = f"Not enough common points found (need ≥{min_required_points}, found {len(common_point_names)})"
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
        
        if len(src_points) < min_required_points:
            debug_info["reason_for_failure"] = f"Not enough common points found for computation (need ≥{min_required_points}, found {len(src_points)})"
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
        margin = 3.0  # Increase margin to be more permissive
        rink_bounds_x = [-self.rink_width * (margin - 1), self.rink_width * margin]
        rink_bounds_y = [-self.rink_height * (margin - 1), self.rink_height * margin]
        
        within_bounds = all(
            rink_bounds_x[0] <= x <= rink_bounds_x[1] and
            rink_bounds_y[0] <= y <= rink_bounds_y[1]
            for x, y in transformed_corners
        )
        
        if not within_bounds:
            print("Homography validation failed - corners outside reasonable bounds")
            return False
        
        # IMPROVED: Check for area distortion instead of strict orientation preservation
        # Calculate area of the quadrilateral in both source and transformed space
        def calculate_quad_area(points):
            # Use the shoelace formula to calculate area
            n = len(points)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            area = abs(area) / 2.0
            return area
            
        source_area = calculate_quad_area(corners.reshape(-1, 2))
        transformed_area = calculate_quad_area(transformed_corners)
        
        # Check if the area ratio is reasonable (not extremely distorted)
        # Get areas scaled to a common reference to compare them
        rink_area = self.rink_width * self.rink_height
        source_ratio = source_area / (w * h)
        transformed_ratio = transformed_area / rink_area
        
        area_ratio = transformed_ratio / source_ratio
        if area_ratio < 0.01 or area_ratio > 100:
            print(f"Homography validation failed - extreme area distortion: {area_ratio:.2f}")
            return False
            
        # IMPROVED: Instead of checking exact orientation preservation,
        # check if the transformation doesn't completely invert the frame
        # Extract the transformed directions
        width_vector = transformed_corners[3] - transformed_corners[0]  # top-right - top-left
        height_vector = transformed_corners[1] - transformed_corners[0]  # bottom-left - top-left
        
        # A reasonable homography should not completely invert directions
        # For hockey rinks, the width_vector should point generally rightward (x positive)
        # and the height_vector should point generally downward (y positive)
        # Allow flexibility by checking only the dominant direction
        
        # If the camera is viewing from above the rink, y coordinates are inverted
        # compared to screen coordinates, so allow for that case
        width_reasonable = width_vector[0] > -w/2  # Width vector should not point strongly left
        height_reasonable = True  # Don't strictly check height direction due to varying camera angles
        
        if not (width_reasonable and height_reasonable):
            print("Homography validation failed - orientation not preserved")
            return False
            
        # If all checks pass, the homography is valid
        return True
    
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
                    
                    # Calculate average position of this goal line
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    
                    # IMPROVED: Check if ANY part of this goal line is to the left of ANY faceoff circle
                    is_left_of_faceoff = False
                    min_goal_x = min(p[0] for p in pts)  # Leftmost point of goal line
                    
                    if "FaceoffCircle" in segmentation_features:
                        for circle in segmentation_features["FaceoffCircle"]:
                            if "center" in circle:
                                circle_x = float(circle["center"]["x"])
                                if min_goal_x < circle_x:  # If ANY part of goal line is left of circle
                                    is_left_of_faceoff = True
                                    break
                            elif "points" in circle and circle["points"]:
                                # For point-based circles, use the average x position
                                circle_points = circle["points"]
                                if isinstance(circle_points[0], dict):
                                    circle_x = sum(float(p["x"]) for p in circle_points) / len(circle_points)
                                else:
                                    circle_x = sum(float(p[0]) for p in circle_points) / len(circle_points)
                                if min_goal_x < circle_x:  # If ANY part of goal line is left of circle
                                    is_left_of_faceoff = True
                                    break
                    
                    # If ANY part of the goal line is to the left of any faceoff circle, it MUST be the left goal line
                    if is_left_of_faceoff:
                        left_goal_points.extend(pts)
                        print(f"Classified goal line at x={avg_x:.1f} as LEFT (leftmost point {min_goal_x:.1f} is left of faceoff circle)")
                    else:
                        # Only classify as right if we're sure no part is left
                        right_goal_points.extend(pts)
                        print(f"Classified goal line at x={avg_x:.1f} as RIGHT (no part left of any faceoff circle)")
                
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
