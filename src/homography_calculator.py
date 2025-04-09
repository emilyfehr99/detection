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

    def __init__(self, rink_coordinates_path: str, broadcast_width: int = 1948, broadcast_height: int = 1042):
        """
        Initialize the HomographyCalculator with rink coordinates.
        
        Args:
            rink_coordinates_path: Path to the JSON file containing rink coordinates
            broadcast_width: Width of the broadcast footage (default: 1948)
            broadcast_height: Height of the broadcast footage (default: 1042)
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
        Extract source points from segmentation features for homography calculation.
        
        Args:
            segmentation_features: Dictionary of segmentation features
            
        Returns:
            Dictionary of source points
        """
        source_points = {}
        
        # Debug: Print raw segmentation features
        self.logger.info(
            f"Raw segmentation features: {json.dumps(segmentation_features, indent=2)}"
        )
        
        # Track feature positions for relative positioning
        goal_line_positions = []  # List of (x, points) tuples
        faceoff_circle_positions = []  # List of (x, points) tuples
        blue_line_positions = []  # List of (x, points) tuples
        center_line_x = None
        
        # First collect all feature positions
        if "GoalLine" in segmentation_features:
            for gl in segmentation_features["GoalLine"]:
                if "points" in gl and gl["points"]:
                    points = gl["points"]
                    if isinstance(points[0], dict):
                        pts = [(float(p["x"]), float(p["y"])) for p in points]
                    else:
                        pts = [(float(p[0]), float(p[1])) for p in points]
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    goal_line_positions.append((avg_x, pts))
        
        if "FaceoffCircle" in segmentation_features:
            for fc in segmentation_features["FaceoffCircle"]:
                if "points" in fc and fc["points"]:
                    points = fc["points"]
                    if isinstance(points[0], dict):
                        x = float(points[0]["x"])
                        y = float(points[0]["y"])
                    else:
                        x = float(points[0][0])
                        y = float(points[0][1])
                    faceoff_circle_positions.append((x, [(x, y)]))
                elif "center" in fc:
                    x = float(fc["center"]["x"])
                    y = float(fc["center"]["y"])
                    faceoff_circle_positions.append((x, [(x, y)]))
        
        if "BlueLine" in segmentation_features:
            for bl in segmentation_features["BlueLine"]:
                if "points" in bl and bl["points"]:
                    points = bl["points"]
                    if isinstance(points[0], dict):
                        pts = [(float(p["x"]), float(p["y"])) for p in points]
                    else:
                        pts = [(float(p[0]), float(p[1])) for p in points]
                    avg_x = sum(p[0] for p in pts) / len(pts)
                    blue_line_positions.append((avg_x, pts))
        
        if "RedCenterLine" in segmentation_features and segmentation_features["RedCenterLine"]:
            for cl in segmentation_features["RedCenterLine"]:
                if "points" in cl and cl["points"]:
                    points = cl["points"]
                    if isinstance(points[0], dict):
                        x_positions = [float(p["x"]) for p in points]
                    else:
                        x_positions = [float(p[0]) for p in points]
                    center_line_x = sum(x_positions) / len(x_positions)
                    break
        
        # Now classify goal lines
        left_goal_line = None
        right_goal_line = None
        for goal_x, goal_points in goal_line_positions:
            # Check if this goal line is left of any faceoff circle
            is_left_of_faceoff = any(goal_x < fc_x for fc_x, _ in faceoff_circle_positions)
            # Check if this goal line is right of any faceoff circle
            is_right_of_faceoff = any(goal_x > fc_x for fc_x, _ in faceoff_circle_positions)
            
            if is_left_of_faceoff and not is_right_of_faceoff:
                left_goal_line = (goal_x, goal_points)
                # Add to source points
                source_points["goal_line_left_top"] = goal_points[0]
                source_points["goal_line_left_bottom"] = goal_points[-1]
                self.logger.info(f"Classified goal line at x={goal_x} as LEFT")
            elif is_right_of_faceoff and not is_left_of_faceoff:
                right_goal_line = (goal_x, goal_points)
                # Add to source points
                source_points["goal_line_right_top"] = goal_points[0]
                source_points["goal_line_right_bottom"] = goal_points[-1]
                self.logger.info(f"Classified goal line at x={goal_x} as RIGHT")
        
        # Classify faceoff circles
        for fc_x, fc_points in faceoff_circle_positions:
            is_left_circle = False
            is_right_circle = False
            
            # Check relative to goal lines
            if left_goal_line and fc_x > left_goal_line[0]:
                is_left_circle = True
            if right_goal_line and fc_x < right_goal_line[0]:
                is_right_circle = True
                
            # Check relative to blue lines
            for blue_x, _ in blue_line_positions:
                if fc_x < blue_x:
                    is_left_circle = True
                if fc_x > blue_x:
                    is_right_circle = True
            
            # Classify based on vertical position
            y = fc_points[0][1]
            if is_left_circle and not is_right_circle:
                if y < self.broadcast_height / 2:
                    source_points["faceoff_circle_0"] = fc_points[0]  # top left
                    self.logger.info(f"Added top left faceoff circle at ({fc_x}, {y})")
                else:
                    source_points["faceoff_circle_2"] = fc_points[0]  # bottom left
                    self.logger.info(f"Added bottom left faceoff circle at ({fc_x}, {y})")
            elif is_right_circle and not is_left_circle:
                if y < self.broadcast_height / 2:
                    source_points["faceoff_circle_1"] = fc_points[0]  # top right
                    self.logger.info(f"Added top right faceoff circle at ({fc_x}, {y})")
                else:
                    source_points["faceoff_circle_3"] = fc_points[0]  # bottom right
                    self.logger.info(f"Added bottom right faceoff circle at ({fc_x}, {y})")
        
        # Classify blue lines
        for blue_x, blue_points in blue_line_positions:
            is_left_blue = False
            is_right_blue = False
            
            # Check relative to center line if available
            if center_line_x:
                if blue_x < center_line_x:
                    is_left_blue = True
                else:
                    is_right_blue = True
            
            # Check relative to faceoff circles and goal lines
            if left_goal_line and blue_x > left_goal_line[0]:
                is_left_blue = True
            if right_goal_line and blue_x < right_goal_line[0]:
                is_right_blue = True
                
            for fc_x, _ in faceoff_circle_positions:
                if blue_x > fc_x:
                    is_left_blue = True
                if blue_x < fc_x:
                    is_right_blue = True
            
            # Classify based on strongest evidence
            if is_left_blue and not is_right_blue:
                source_points["blue_line_left_top"] = blue_points[0]
                source_points["blue_line_left_bottom"] = blue_points[-1]
                self.logger.info(f"Classified blue line at x={blue_x} as LEFT")
            elif is_right_blue and not is_left_blue:
                source_points["blue_line_right_top"] = blue_points[0]
                source_points["blue_line_right_bottom"] = blue_points[-1]
                self.logger.info(f"Classified blue line at x={blue_x} as RIGHT")
        
        # Add center line points if available
        if center_line_x is not None:
            for cl in segmentation_features["RedCenterLine"]:
                if "points" in cl and cl["points"]:
                    points = cl["points"]
                    if isinstance(points[0], dict):
                        pts = [(float(p["x"]), float(p["y"])) for p in points]
                    else:
                        pts = [(float(p[0]), float(p[1])) for p in points]
                    source_points["center_line_top"] = pts[0]
                    source_points["center_line_bottom"] = pts[-1]
                    self.logger.info(f"Added center line points at x={center_line_x}")
                    break
        
        # Log summary of found points
        self.logger.info(f"Found {len(source_points)} source points:")
        for name, point in source_points.items():
            self.logger.info(f"  {name}: {point}")
        
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

    def calculate_homography(
        self, 
        segmentation_features: Dict[str, List[Dict]]
    ) -> Optional[np.ndarray]:
        """
        Calculate homography matrix from segmentation features.
        
        Args:
            segmentation_features: Dictionary of segmentation features
            
        Returns:
            Homography matrix if successful, None otherwise
        """
        # Extract source points from segmentation features
        source_points = self.extract_source_points(segmentation_features)
        
        # Log available source points
        self.logger.info(
            f"Available source points: {list(source_points.keys())}"
        )
        
        # Get destination points
        dest_points = self.get_destination_points()
        
        # Log available destination points
        self.logger.info(
            f"Available destination points: {list(dest_points.keys())}"
        )
        
        # Log BlueLine features for debugging
        if "BlueLine" in segmentation_features:
            self.logger.info(
                f"BlueLine features: {json.dumps(segmentation_features['BlueLine'])}"
            )
        
        # Need at least 4 points for homography
        if len(source_points) < 4:
            self.logger.warning(
                f"Insufficient points for homography: "
                f"found {len(source_points)}, need 4"
            )
            return None
        
        # Get corresponding destination points
        source_pts = []
        dest_pts = []
        
        for name, point in source_points.items():
            if name in dest_points:
                source_pts.append(point)
                dest_pts.append(dest_points[name])
        
        # Need at least 4 matching points
        if len(source_pts) < 4:
            self.logger.warning(
                f"Insufficient matching points: found {len(source_pts)}, need 4"
            )
            return None
        
        # Convert to numpy arrays
        source_pts = np.array(source_pts, dtype=np.float32)
        dest_pts = np.array(dest_pts, dtype=np.float32)
        
        try:
            # Calculate homography matrix using RANSAC
            matrix, mask = cv2.findHomography(
                source_pts, dest_pts, cv2.RANSAC, 5.0
            )
            
            # Convert matrix to numpy array if it's not already
            if not isinstance(matrix, np.ndarray):
                matrix = np.array(matrix, dtype=np.float32)
            
            # Validate the homography matrix
            if not self.validate_homography(matrix):
                return None
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating homography: {str(e)}")
            return None

    def _calculate_feature_weights(self, common_points, segmentation_features):
        """
        Calculate weights for homography points based on feature reliability.
        
        Args:
            common_points: Dictionary of common points between source and destination
            segmentation_features: Dictionary of detected features
            
        Returns:
            Array of weights for each point
        """
        weights = np.ones(len(common_points))
        
        for i, (name, (src_pt, _)) in enumerate(common_points.items()):
            # Base weight on feature type
            if "goal_line" in name.lower():
                weights[i] *= 1.5  # Goal lines are usually very reliable
            elif "blue_line" in name.lower():
                weights[i] *= 1.2  # Blue lines are generally reliable
            elif "center_line" in name.lower():
                weights[i] *= 1.3  # Center line is important for orientation
            elif "faceoff" in name.lower():
                weights[i] *= 0.8  # Faceoff circles can be less reliable
            
            # Adjust weight based on visibility
            if name in segmentation_features:
                feature = segmentation_features[name]
                if "confidence" in feature:
                    weights[i] *= feature["confidence"]
                # Adjust weight based on feature quality
                if "quality_score" in feature:
                    weights[i] *= feature["quality_score"]
            
            # Adjust weight based on distance from frame center
            center_weight = self.calculate_center_weight(src_pt, self.broadcast_width, self.broadcast_height)
            weights[i] *= center_weight
        
        # Normalize weights
        weights = weights / np.sum(weights)
        return weights

    def validate_homography(self, h_matrix):
        """
        Validate homography matrix using multiple criteria.
        
        Args:
            h_matrix: Homography matrix to validate
            
        Returns:
            Boolean indicating if homography is valid
        """
        if h_matrix is None:
            self.logger.error("Homography matrix is None")
            return False
        
        # Check if matrix is well-formed
        if not isinstance(h_matrix, np.ndarray) or h_matrix.shape != (3, 3):
            shape_info = (h_matrix.shape if isinstance(h_matrix, np.ndarray) 
                         else type(h_matrix))
            self.logger.error(f"Invalid matrix shape: {shape_info}")
            return False
        
        # Check corner bounds
        corners = np.float32([
            [0, 0],
            [self.broadcast_width, 0],
            [self.broadcast_width, self.broadcast_height],
            [0, self.broadcast_height]
        ])
        
        transformed_corners = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), 
            h_matrix
        ).reshape(-1, 2)
        
        # Check if corners are within reasonable bounds
        rink_width = self.rink_width
        rink_height = self.rink_height
        
        # Increase margin to allow for more extreme perspectives
        margin = 1.0  # Allow 100% margin outside rink bounds
        min_x = -rink_width * margin
        max_x = rink_width * (1 + margin)
        min_y = -rink_height * margin
        max_y = rink_height * (1 + margin)
        
        # Log corner positions
        self.logger.info("Transformed corner positions:")
        for i, corner in enumerate(transformed_corners):
            self.logger.info(
                f"Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})"
            )
            self.logger.info(
                f"Bounds: x=[{min_x:.1f}, {max_x:.1f}], "
                f"y=[{min_y:.1f}, {max_y:.1f}]"
            )
            if not (min_x <= corner[0] <= max_x and min_y <= corner[1] <= max_y):
                self.logger.error(f"Corner {i} is out of bounds")
                return False
        
        # Check area ratio with more lenient bounds
        original_area = self.broadcast_width * self.broadcast_height
        transformed_area = cv2.contourArea(transformed_corners)
        area_ratio = transformed_area / original_area
        
        # Allow for more extreme area changes
        min_area_ratio = 0.01  # Allow much smaller projected area
        max_area_ratio = 50.0  # Allow much larger projected area
        
        self.logger.info(
            f"Area ratio: {area_ratio:.3f} "
            f"(should be between {min_area_ratio} and {max_area_ratio})"
        )
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            self.logger.error(f"Invalid area ratio: {area_ratio:.3f}")
            return False
        
        # Check diagonal ratio with more lenient bounds
        original_diag = np.sqrt(self.broadcast_width**2 + self.broadcast_height**2)
        transformed_diag1 = np.linalg.norm(transformed_corners[1] - transformed_corners[3])
        transformed_diag2 = np.linalg.norm(transformed_corners[0] - transformed_corners[2])
        diag_ratio = max(transformed_diag1, transformed_diag2) / original_diag
        
        # Allow for more extreme diagonal changes
        min_diag_ratio = 0.05  # Allow much smaller diagonals
        max_diag_ratio = 20.0  # Allow much larger diagonals
        
        self.logger.info(
            f"Diagonal ratio: {diag_ratio:.3f} "
            f"(should be between {min_diag_ratio} and {max_diag_ratio})"
        )
        if not (min_diag_ratio <= diag_ratio <= max_diag_ratio):
            self.logger.error(f"Invalid diagonal ratio: {diag_ratio:.3f}")
            return False
        
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
    
    def interpolate_homography(self, matrix1: np.ndarray, matrix2: np.ndarray, t: float) -> np.ndarray:
        """
        Linearly interpolate between two homography matrices.
        
        Args:
            matrix1: First homography matrix
            matrix2: Second homography matrix
            t: Interpolation factor (0.0 to 1.0)
            
        Returns:
            Interpolated homography matrix
        """
        # Ensure t is between 0 and 1
        t = max(0.0, min(1.0, t))
        
        # Linear interpolation of matrix elements
        interpolated = (1 - t) * matrix1 + t * matrix2
        
        # Normalize the matrix to ensure it's a valid homography
        interpolated = interpolated / interpolated[2, 2]
        
        return interpolated

    def get_homography_matrix(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get homography matrix for a specific frame, using interpolation if necessary.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Homography matrix if available, None otherwise
        """
        # If we have the exact matrix, return it
        if frame_idx in self.homography_cache:
            return self.homography_cache[frame_idx]
        
        # Find the closest valid matrices before and after this frame
        valid_indices = sorted(self.homography_cache.keys())
        if not valid_indices:
            return None
            
        # Find the closest indices before and after
        before_idx = None
        after_idx = None
        
        for idx in valid_indices:
            if idx <= frame_idx:
                before_idx = idx
            if idx >= frame_idx:
                after_idx = idx
                break
        
        # If we have both before and after matrices, interpolate
        if before_idx is not None and after_idx is not None:
            matrix1 = self.homography_cache[before_idx]
            matrix2 = self.homography_cache[after_idx]
            
            # Calculate interpolation factor
            t = (frame_idx - before_idx) / (after_idx - before_idx)
            
            # Interpolate and cache the result
            interpolated = self.interpolate_homography(matrix1, matrix2, t)
            self.homography_cache[frame_idx] = interpolated
            return interpolated
        
        # If we only have a matrix before this frame, use it
        elif before_idx is not None:
            return self.homography_cache[before_idx]
        
        # If we only have a matrix after this frame, use it
        elif after_idx is not None:
            return self.homography_cache[after_idx]
        
        return None
        
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
                    
                    # Skip if no points
                    if not pts:
                        continue
                    
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

    def draw_visualization(self, frame: np.ndarray, segmentation_features: Dict, homography: Optional[np.ndarray] = None) -> np.ndarray:
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
        
        # If no homography, add a text message explaining
        if homography is None:
            cv2.putText(vis_frame, "Homography calculation failed", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return vis_frame

    def project_point_to_rink(
        self, 
        point: Tuple[float, float], 
        homography_matrix: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Project a point from broadcast coordinates to rink coordinates.
        
        Args:
            point: Tuple of (x, y) coordinates in broadcast frame
            homography_matrix: Homography matrix to use for projection
            
        Returns:
            Dictionary with x, y coordinates in rink space if successful, None otherwise
        """
        try:
            # Convert homography matrix to numpy array if it's a list
            if isinstance(homography_matrix, list):
                homography_matrix = np.array(homography_matrix, dtype=np.float32)
            
            # Convert point to homogeneous coordinates
            pt = np.array([[point[0], point[1], 1]], dtype=np.float32)
            
            # Project point using homography matrix
            projected = homography_matrix.dot(pt.T)
            
            # Convert back from homogeneous coordinates
            x = float(projected[0] / projected[2])
            y = float(projected[1] / projected[2])
            
            # Check if point is within rink bounds with margin
            margin = 0.2  # 20% margin
            min_x = -self.rink_width * margin
            max_x = self.rink_width * (1 + margin)
            min_y = -self.rink_height * margin
            max_y = self.rink_height * (1 + margin)
            
            if min_x <= x <= max_x and min_y <= y <= max_y:
                # Convert to pixel coordinates (0 to rink_width/height)
                pixel_x = int((x + self.rink_width * margin) * self.rink_width / (max_x - min_x))
                pixel_y = int((y + self.rink_height * margin) * self.rink_height / (max_y - min_y))
                
                return {
                    "x": x,
                    "y": y,
                    "pixel_x": pixel_x,
                    "pixel_y": pixel_y
                }
            else:
                self.logger.warning(f"Projected point ({x}, {y}) is outside rink bounds")
                return None
                
        except Exception as e:
            self.logger.error(f"Error projecting point: {str(e)}")
            return None
