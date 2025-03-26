import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
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
            # Filter out invalid blue lines
            valid_blue_lines = []
            for bl in segmentation_features["BlueLine"]:
                if "points" not in bl or len(bl["points"]) < 2:
                    self.logger.info("Skipping blue line with insufficient points")
                    continue
                
                # Get both top and bottom points
                points = []
                try:
                    if isinstance(bl["points"][0], dict):
                        points = [(float(p["x"]), float(p["y"])) for p in bl["points"]]
                    else:
                        points = [(float(p[0]), float(p[1])) for p in bl["points"]]
                except (IndexError, KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing blue line points: {e}")
                    continue
                
                if not points:
                    self.logger.info("No valid points extracted from blue line")
                    continue
                
                # Calculate line properties
                dx = points[-1][0] - points[0][0]
                dy = points[-1][1] - points[0][1]
                line_length = np.sqrt(dx*dx + dy*dy)
                angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
                
                # Only validate y-range to avoid top/bottom of frame
                y_min = 0.2 * self.broadcast_height
                y_max = 0.8 * self.broadcast_height
                if y_min < points[0][1] < y_max:
                    valid_blue_lines.append({"points": points})
                    self.logger.info(
                        "Valid blue line: "
                        f"length={line_length:.1f}, angle={angle:.1f}"
                    )
                else:
                    self.logger.info(
                        "Rejected blue line: "
                        f"length={line_length:.1f}, angle={angle:.1f}, "
                        f"y={points[0][1]}"
                    )

            # Sort valid blue lines by x-coordinate
            if valid_blue_lines:
                valid_blue_lines.sort(key=lambda x: x["points"][0][0])
                
                # Process each valid blue line
                for bl in valid_blue_lines:
                    points = bl["points"]
                    if not points:
                        continue
                        
                    # Sort points by y-coordinate to get top and bottom
                    points.sort(key=lambda p: p[1])
                    top_point = points[0]
                    bottom_point = points[-1]
                    
                    # Use x-position to determine if it's left or right blue line
                    frame_center = self.broadcast_width / 2
                    if top_point[0] < frame_center:
                        # Left blue line
                        source_points["blue_line_left_top"] = top_point
                        source_points["blue_line_left_bottom"] = bottom_point
                        self.logger.info(
                            f"Added left blue line points: top={top_point}, "
                            f"bottom={bottom_point}"
                        )
                    else:
                        # Right blue line
                        source_points["blue_line_right_top"] = top_point
                        source_points["blue_line_right_bottom"] = bottom_point
                        self.logger.info(
                            f"Added right blue line points: top={top_point}, "
                            f"bottom={bottom_point}"
                        )
        
        # Process faceoff circles with hockey domain knowledge
        faceoff_circles = []
        blue_line_points = []
        
        # First identify blue line points and calculate average x position
        blue_line_x = None
        left_blue_line_x = None
        if "BlueLine" in segmentation_features and segmentation_features["BlueLine"]:
            for bl in segmentation_features["BlueLine"]:
                if "points" not in bl or not bl["points"]:
                    continue
                points = bl["points"]
                try:
                    if isinstance(points[0], dict):
                        x_positions = [float(p["x"]) for p in points]
                    else:
                        x_positions = [float(p[0]) for p in points]
                    avg_x = sum(x_positions) / len(x_positions)
                    blue_line_points.append(avg_x)
                except (IndexError, KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing blue line points: {e}")
                    continue
            
            if blue_line_points:
                # Sort blue line x positions
                blue_line_points.sort()
                if len(blue_line_points) >= 2:
                    # If we have two blue lines, use the leftmost one
                    left_blue_line_x = blue_line_points[0]
                    blue_line_x = blue_line_points[-1]
                else:
                    # If we only have one blue line, use its position
                    blue_line_x = blue_line_points[0]
                    left_blue_line_x = blue_line_x

        # Check for left goal line presence
        has_left_goal_line = False
        if "GoalLine" in segmentation_features and segmentation_features["GoalLine"]:
            goal_lines = segmentation_features["GoalLine"]
            for gl in goal_lines:
                if "points" not in gl or not gl["points"]:
                    continue
                points = gl["points"]
                try:
                    if isinstance(points[0], dict):
                        x_positions = [float(p["x"]) for p in points]
                    else:
                        x_positions = [float(p[0]) for p in points]
                    avg_x = sum(x_positions) / len(x_positions)
                    
                    # If we have a blue line reference, use it to determine if this is a left goal line
                    if blue_line_x and avg_x < blue_line_x:
                        has_left_goal_line = True
                        self.logger.info("Detected left goal line")
                        break
                    # If no blue line reference, use frame center
                    elif avg_x < self.broadcast_width / 2:
                        has_left_goal_line = True
                        self.logger.info("Detected left goal line (using frame center)")
                        break
                except (IndexError, KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing goal line points: {e}")
                    continue

        # Process faceoff circles
        if "FaceoffCircle" in segmentation_features:
            for fc in segmentation_features["FaceoffCircle"]:
                if "points" not in fc or not fc["points"]:
                    continue
                
                # Extract center point
                try:
                    points = fc["points"]
                    if isinstance(points[0], dict):
                        center_x = float(points[0]["x"])
                        center_y = float(points[0]["y"])
                    else:
                        center_x = float(points[0][0])
                        center_y = float(points[0][1])
                except (IndexError, KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing faceoff circle points: {e}")
                    continue
                
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
                # HOCKEY RULE 3: If we see both blue lines, circles between them are in the neutral zone
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
                    elif blue_line_x and all(left_blue_line_x < circle["x"] < blue_line_x for circle in faceoff_circles):
                        # Circles between blue lines are in neutral zone
                        circle_side = "neutral"
                        print("Faceoff circles are in NEUTRAL zone (between blue lines)")
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
                
                # Label circles based on their consistent IDs and y-positions
                if len(faceoff_circles) == 1:
                    # Just one circle, use y position to determine top/bottom
                    circle = faceoff_circles[0]
                    frame_center_y = self.broadcast_height / 2
                    position = "top" if circle["y"] < frame_center_y else "bottom"
                    
                    # Add to source points with consistent naming
                    if circle_side == "left":
                        source_points[f"faceoff_left_{position}"] = (circle["x"], circle["y"])
                        source_points[f"faceoff_circle_{circle['id']}"] = (circle["x"], circle["y"])
                        source_points[f"faceoff_{position}_left"] = (circle["x"], circle["y"])
                    elif circle_side == "right":
                        source_points[f"faceoff_right_{position}"] = (circle["x"], circle["y"])
                        source_points[f"faceoff_circle_{circle['id']}"] = (circle["x"], circle["y"])
                        source_points[f"faceoff_{position}_right"] = (circle["x"], circle["y"])
                        
                elif len(faceoff_circles) >= 2:
                    # Multiple circles, use y-positions for top/bottom assignment
                    circles_by_y = sorted(faceoff_circles, key=lambda c: c["y"])
                    
                    # Top circle (lowest y)
                    top_circle = circles_by_y[0]
                    if circle_side == "left":
                        source_points[f"faceoff_left_top"] = (top_circle["x"], top_circle["y"])
                        source_points[f"faceoff_circle_{top_circle['id']}"] = (top_circle["x"], top_circle["y"])
                        source_points[f"faceoff_top_left"] = (top_circle["x"], top_circle["y"])
                    elif circle_side == "right":
                        source_points[f"faceoff_right_top"] = (top_circle["x"], top_circle["y"])
                        source_points[f"faceoff_circle_{top_circle['id']}"] = (top_circle["x"], top_circle["y"])
                        source_points[f"faceoff_top_right"] = (top_circle["x"], top_circle["y"])
                    
                    # Bottom circle (highest y)
                    bottom_circle = circles_by_y[-1]
                    if circle_side == "left":
                        source_points[f"faceoff_left_bottom"] = (bottom_circle["x"], bottom_circle["y"])
                        source_points[f"faceoff_circle_{bottom_circle['id']}"] = (bottom_circle["x"], bottom_circle["y"])
                        source_points[f"faceoff_bottom_left"] = (bottom_circle["x"], bottom_circle["y"])
                    elif circle_side == "right":
                        source_points[f"faceoff_right_bottom"] = (bottom_circle["x"], bottom_circle["y"])
                        source_points[f"faceoff_circle_{bottom_circle['id']}"] = (bottom_circle["x"], bottom_circle["y"])
                        source_points[f"faceoff_bottom_right"] = (bottom_circle["x"], bottom_circle["y"])
                    
                    print(f"Classified faceoff circles as {circle_side}_top (ID: {top_circle['id']}) and {circle_side}_bottom (ID: {bottom_circle['id']})")
        
        # Process goal lines as a SINGLE CONTINUOUS LINE regardless of segmentation
        if "GoalLine" in segmentation_features and len(segmentation_features["GoalLine"]) > 0:
            goal_lines = segmentation_features["GoalLine"]
            all_goal_points = []
            
            # Collect all goal line points
            for gl in goal_lines:
                if "points" in gl:
                    for point in gl["points"]:
                        if isinstance(point, dict):
                            all_goal_points.append((float(point["x"]), float(point["y"])))
                        else:
                            all_goal_points.append((float(point[0]), float(point[1])))
            
            if all_goal_points:
                # Calculate goal line span
                min_x = min(x for x, _ in all_goal_points)
                max_x = max(x for x, _ in all_goal_points)
                goal_x_span = max_x - min_x
                goal_x_ratio = goal_x_span / self.broadcast_width
                print(f"Goal line x-span: {goal_x_span:.1f}, "
                      f"ratio to frame width: {goal_x_ratio:.3f}")
                
                # HOCKEY RULE 1: If ANY part of goal line is left of ANY faceoff circle, it MUST be left goal line
                # HOCKEY RULE 2: If we see the left goal line, all circles are LEFT zone circles
                is_left_goal = False
                if faceoff_circles:
                    if any(min_x < circle["x"] for circle in faceoff_circles):
                        is_left_goal = True
                        print(f"Goal line is LEFT (leftmost point {min_x:.1f} is left of faceoff circle)")
                    else:
                        print(f"Goal line is RIGHT (leftmost point {min_x:.1f} is not left of any faceoff circle)")
                else:
                    # Fallback to frame center if no faceoff circles
                    frame_center_x = self.broadcast_width / 2
                    is_left_goal = min_x < frame_center_x
                    print(f"Goal line is {'LEFT' if is_left_goal else 'RIGHT'} (using frame center)")
                
                # Sort goal points by y-coordinate to find top and bottom
                goal_points_sorted_by_y = sorted(all_goal_points, key=lambda p: p[1])
                top_point = goal_points_sorted_by_y[0]
                bottom_point = goal_points_sorted_by_y[-1]
                
                # Add points with consistent naming
                goal_side = "left" if is_left_goal else "right"
                source_points[f"goal_line_{goal_side}_top"] = top_point
                source_points[f"goal_line_{goal_side}_bottom"] = bottom_point
                
                # Remove any incorrect goal line points to avoid confusion
                opposite_side = "right" if is_left_goal else "left"
                if f"goal_line_{opposite_side}_top" in source_points:
                    del source_points[f"goal_line_{opposite_side}_top"]
                if f"goal_line_{opposite_side}_bottom" in source_points:
                    del source_points[f"goal_line_{opposite_side}_bottom"]
                
                print(f"Selected {goal_side} goal line endpoints at {top_point} (top) and {bottom_point} (bottom)")
        
        return source_points
    
    def calculate_center_weight(self, point, frame_width, frame_height):
        """
        Calculate weight based on point's distance from frame center.
        Uses a lenient Gaussian falloff with sigma=2.0.
        """
        # Normalize to [-1, 1]
        x = (point[0] - frame_width/2) / (frame_width/2)
        y = (point[1] - frame_height/2) / (frame_height/2)
        # More lenient Gaussian falloff (sigma = 2.0)
        return np.exp(-0.5 * (x*x + y*y) / 4.0)

    def calculate_line_weight(self, line_length, expected_length):
        """
        Calculate weight based on line length ratio.
        Uses lenient bounds (0.2 to 4.0) for ratio validation.
        """
        ratio = line_length / expected_length
        if ratio < 0.2 or ratio > 4.0:
            return 0.0
        # Smooth falloff near bounds
        weight = min(ratio, 1/ratio)
        return weight

    def calculate_homography(self, segmentation_features: Dict[str, List[Dict]]) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """
        Calculate the homography matrix between broadcast footage and the 2D rink.
        Uses weighted points based on line visibility and distance from frame center.
        """
        source_points = self.extract_source_points(segmentation_features)
        dest_points = self.get_destination_points()
        
        debug_info = {
            "source_points": {name: point for name, point in source_points.items()},
            "destination_points": {name: point for name, point in dest_points.items()},
            "common_point_names": [],
            "weights": {}  # Store weights for debugging
        }
        
        try:
            H = self._calculate_homography_internal(
                source_points, dest_points, 
                self.broadcast_width, self.broadcast_height
            )
            
            if self.validate_homography(
                H, self.broadcast_width, self.broadcast_height,
                self.rink_width, self.rink_height
            ):
                return True, H, debug_info
            else:
                debug_info["reason_for_failure"] = "Homography validation failed"
                return False, H, debug_info
            
        except Exception as e:
            debug_info["reason_for_failure"] = str(e)
            return False, np.eye(3), debug_info

    def _calculate_homography_internal(
        self, source_points, dest_points, frame_width, frame_height
    ):
        """Calculate homography with weighted point selection."""
        # Initialize weights
        point_weights = {}
        
        # Process each point type
        for point_type in source_points:
            point = source_points[point_type]
            if not point:
                continue
            
            if (point_type.startswith('blue_line') or 
                point_type.startswith('goal_line')):
                # For line points, calculate line length weight
                if point_type.endswith('_top'):
                    # Find corresponding bottom point
                    bottom_type = point_type.replace('_top', '_bottom')
                    if bottom_type in source_points:
                        top_point = point
                        bottom_point = source_points[bottom_type]
                        # Calculate line length
                        dx = top_point[0] - bottom_point[0]
                        dy = top_point[1] - bottom_point[1]
                        line_length = np.sqrt(dx*dx + dy*dy)
                        expected_length = frame_height * 0.8
                        line_weight = self.calculate_line_weight(
                            line_length, expected_length
                        )
                        
                        # Calculate weights for both points
                        center_weight_top = self.calculate_center_weight(
                            top_point, frame_width, frame_height
                        )
                        center_weight_bottom = self.calculate_center_weight(
                            bottom_point, frame_width, frame_height
                        )
                        
                        # Combine weights with more emphasis on center weight
                        point_weights[point] = (
                            0.8 * center_weight_top + 0.2 * line_weight
                        )
                        point_weights[bottom_point] = (
                            0.8 * center_weight_bottom + 0.2 * line_weight
                        )
            else:
                # For non-line points (e.g. faceoff circles), only use center weight
                center_weight = self.calculate_center_weight(
                    point, frame_width, frame_height
                )
                point_weights[point] = center_weight
        
        # Prepare point arrays
        src_pts = []
        dst_pts = []
        weights = []
        min_weight = 0.01  # Reduced from 0.02
        
        # Filter points by weight
        for point_type in source_points:
            if point_type not in dest_points:
                continue
            
            src = source_points[point_type]
            dst = dest_points[point_type]
            
            weight = point_weights.get(src, 0.0)
            if weight > min_weight:
                src_pts.append(src)
                dst_pts.append(dst)
                weights.append(weight)
        
        # Check if we have high-confidence points (weight > 0.5)
        high_confidence_points = sum(1 for w in weights if w > 0.5)
        min_points = 2 if high_confidence_points >= 2 else 3  # Allow 2 points if they're high confidence
        
        if len(src_pts) < min_points:
            msg = (
                f"Not enough weighted points "
                f"(need ≥{min_points}, found {len(src_pts)})"
            )
            raise ValueError(msg)
        
        # Convert to numpy arrays
        src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
        weights = np.array(weights)
        
        # Try multiple RANSAC thresholds with more lenient values
        thresholds = [
            5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0
        ]  # Added even higher thresholds
        best_homography = None
        best_inliers = 0
        
        for threshold in thresholds:
            try:
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, threshold
                )
                if H is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_homography = H
                        best_inliers = inliers
            except cv2.error:
                continue
        
        if best_homography is None:
            raise ValueError("Failed to find valid homography")
        
        return best_homography

    def validate_homography(
        self, H, frame_width, frame_height, rink_width, rink_height
    ):
        """
        Validate homography matrix using corner bounds, area ratio, and diagonal ratio.
        Uses extremely lenient bounds to allow for more perspective variations.
        """
        try:
            # Test corners of the frame
            corners = np.array([
                [0, 0],
                [frame_width, 0],
                [frame_width, frame_height],
                [0, frame_height]
            ], dtype=np.float32)
            
            # Transform corners
            transformed = cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), H
            ).reshape(-1, 2)
            
            # Check corner bounds - extremely lenient bounds
            max_width = rink_width * 10  # Was 6, now 10
            max_height = rink_height * 10  # Was 6, now 10
            
            # Check if any transformed point is outside reasonable bounds
            for point in transformed:
                x_out = point[0] < -max_width or point[0] > max_width * 5
                y_out = point[1] < -max_height or point[1] > max_height * 5
                if x_out or y_out:
                    raise ValueError("Corners outside reasonable bounds")
            
            # Check area ratio - extremely lenient bounds
            src_area = cv2.contourArea(corners)
            dst_area = cv2.contourArea(transformed)
            area_ratio = dst_area / src_area
            
            # Allow extremely extreme area ratios
            if area_ratio < 0.01 or area_ratio > 100:  # Was 0.02-50
                raise ValueError(f"Area ratio {area_ratio:.2f} invalid")
            
            # Check diagonal ratios - extremely lenient bounds
            src_diag1 = np.linalg.norm(corners[2] - corners[0])
            src_diag2 = np.linalg.norm(corners[3] - corners[1])
            src_ratio = src_diag1 / src_diag2
            
            dst_diag1 = np.linalg.norm(transformed[2] - transformed[0])
            dst_diag2 = np.linalg.norm(transformed[3] - transformed[1])
            dst_ratio = dst_diag1 / dst_diag2
            
            # Allow extremely extreme diagonal ratios
            ratio = dst_ratio / src_ratio
            if ratio < 0.05 or ratio > 20.0:  # Was 0.1-10
                raise ValueError(f"Diagonal ratio {ratio:.2f} invalid")
            
            return True
            
        except Exception as e:
            print(f"Homography validation failed - {str(e)}")
            return False
    
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
            self.logger.info(f"BlueLine features: {segmentation_features['BlueLine']}")
            for blue_line in segmentation_features["BlueLine"]:
                if "points" in blue_line and len(blue_line["points"]) >= 2:  # Ensure we have at least 2 points
                    pts = [(int(p["x"]), int(p["y"])) for p in blue_line["points"]]
                    # Calculate line length to validate
                    line_length = np.sqrt((pts[-1][0] - pts[0][0])**2 + (pts[-1][1] - pts[0][1])**2)
                    if line_length > 50:  # Only draw if line is long enough
                        for i in range(len(pts) - 1):
                            cv2.line(vis_frame, pts[i], pts[i + 1], (255, 0, 0), 2)  # Blue color
                elif "center" in blue_line and "radius" in blue_line:
                    center = (int(blue_line["center"]["x"]), int(blue_line["center"]["y"]))
                    radius = int(blue_line["radius"])
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
                
                # Draw rink boundaries in blue
                cv2.polylines(vis_frame, [projected_corners], True, (255, 0, 0), 2)
                
            except Exception as e:
                self.logger.error(f"Error projecting rink corners: {e}")
        else:
            # If no homography, add a text message explaining
            cv2.putText(vis_frame, "Homography calculation failed", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return vis_frame
