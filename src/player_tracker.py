import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from segmentation_processor import SegmentationProcessor
from player_detector import PlayerDetector
from orientation_detector import OrientationDetector
from homography_calculator import HomographyCalculator


class PlayerTracker:
    """
    Main module that integrates all components to track players in hockey broadcast footage.
    """

    def __init__(
        self,
        segmentation_model_path: str,
        detection_model_path: str,
        orientation_model_path: str,
        rink_coordinates_path: str,
        output_dir: str = None,
        device: str = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    ):
        """
        Initialize the player tracker.
        
        Args:
            segmentation_model_path: Path to segmentation model
            detection_model_path: Path to detection model
            orientation_model_path: Path to orientation model
            rink_coordinates_path: Path to rink coordinates JSON
            output_dir: Directory to save outputs
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = device
        
        # Initialize output directory
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize components
        self.segmentation_processor = SegmentationProcessor(segmentation_model_path, device)
        self.player_detector = PlayerDetector(detection_model_path, device)
        self.orientation_detector = OrientationDetector(orientation_model_path, device)
        self.homography_calculator = HomographyCalculator(rink_coordinates_path)
        
        # Initialize tracking data
        self.tracking_data = {}
        
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict[str, Any]:
        """
        Process a single frame to track players.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            
        Returns:
            Dictionary containing processed data for the frame
        """
        frame_height, frame_width = frame.shape[:2]
        frame_data = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "players": [],
            "homography_success": False
        }
        
        # Step 1: Process through segmentation model to get rink features
        segmentation_features = self.segmentation_processor.process_frame(frame)
        
        # Save the segmentation features for visualization
        frame_data["segmentation_features"] = segmentation_features
        
        # Step 2: Calculate homography based on segmentation features
        # Pass the frame_id to the homography calculator for temporal smoothing
        homography_matrix, homography_success = self.homography_calculator.calculate_homography(
            segmentation_features, frame_id
        )
        frame_data["homography_success"] = homography_success
        frame_data["homography_matrix"] = homography_matrix  # Store the matrix for visualization
        
        # No need to handle homography failure here - our calculator does that internally
        # with the temporal smoothing approach
        
        # Step 3: Detect players
        player_detections = self.player_detector.process_frame(frame)
        
        # Step 4: Get player crops for orientation detection
        player_crops = self.player_detector.get_player_crops(frame, player_detections)
        
        # Step 5: Detect player orientations
        player_orientations = self.orientation_detector.process_player_crops(player_crops)
        
        # Step 6: Map player positions to the rink and compile data
        for i, detection in enumerate(player_detections):
            player_data = {
                "player_id": f"player_{frame_id}_{i}",  # Temporary ID
                "type": detection["class"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "broadcast_position": detection["reference_point"]
            }
            
            # Add orientation if available
            if i in player_orientations:
                player_data["orientation"] = player_orientations[i]["orientation"]
                player_data["orientation_confidence"] = player_orientations[i]["confidence"]
            
            # Map position to rink if homography is available
            if homography_matrix is not None:
                # Use apply_homography instead of map_point
                rink_positions = self.homography_calculator.apply_homography(
                    [detection["reference_point"]], homography_matrix
                )
                if rink_positions:  # Check if we got any positions back
                    player_data["rink_position"] = rink_positions[0]
            
            frame_data["players"].append(player_data)
        
        # Store frame data
        self.tracking_data[frame_id] = frame_data
        
        return frame_data
    
    def visualize_frame(self, frame: np.ndarray, frame_data: Dict[str, Any], rink_image: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Create visualizations for a processed frame.
        
        Args:
            frame: Original input frame
            frame_data: Processed data for the frame
            rink_image: Optional rink image for visualization
            
        Returns:
            Dictionary containing different visualizations
        """
        visualizations = {}
        
        # Create broadcast visualization with players
        broadcast_vis = frame.copy()
        
        # Draw player detections and orientations
        for player in frame_data["players"]:
            x1, y1, x2, y2 = player["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Determine color based on player type
            color = (0, 255, 0) if player["type"] == "player" else (0, 0, 255)  # Green for players, Red for goalies
            
            # Draw bounding box
            cv2.rectangle(broadcast_vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw reference point
            ref_x, ref_y = player["broadcast_position"]
            cv2.circle(broadcast_vis, (int(ref_x), int(ref_y)), 5, (255, 0, 0), -1)
            
            # Draw orientation arrow if available
            if "orientation" in player:
                orientation = player["orientation"]
                arrow_length = int(min(x2-x1, y2-y1) * 0.3)
                arrow_start = (int(ref_x), int(ref_y))
                
                if orientation == "left":
                    arrow_end = (arrow_start[0] - arrow_length, arrow_start[1])
                elif orientation == "right":
                    arrow_end = (arrow_start[0] + arrow_length, arrow_start[1])
                else:  # neutral
                    arrow_end = (arrow_start[0], arrow_start[1] - arrow_length)
                
                cv2.arrowedLine(broadcast_vis, arrow_start, arrow_end, (255, 255, 0), 2, tipLength=0.3)
        
        visualizations["broadcast"] = broadcast_vis
        
        # Create segmentation visualization
        if "segmentation_features" in frame_data:
            segmentation_vis = self.homography_calculator.draw_segmentation_lines(frame.copy(), frame_data["segmentation_features"])
            visualizations["segmentation"] = segmentation_vis
        
        # Create rink visualization if a rink image is provided
        if rink_image is not None and frame_data["homography_success"]:
            rink_vis = rink_image.copy()
            rink_dims = (rink_image.shape[1], rink_image.shape[0])
            
            # Draw players on rink
            for player in frame_data["players"]:
                if "rink_position" in player:
                    x, y = player["rink_position"]
                    
                    # Determine color based on player type
                    color = (0, 255, 0) if player["type"] == "player" else (0, 0, 255)  # Green for players, Red for goalies
                    
                    # Draw player position
                    cv2.circle(rink_vis, (int(x), int(y)), 10, color, -1)
                    
                    # Draw orientation arrow if available
                    if "orientation" in player:
                        orientation = player["orientation"]
                        arrow_length = 20
                        arrow_start = (int(x), int(y))
                        
                        if orientation == "left":
                            arrow_end = (arrow_start[0] - arrow_length, arrow_start[1])
                        elif orientation == "right":
                            arrow_end = (arrow_start[0] + arrow_length, arrow_start[1])
                        else:  # neutral
                            arrow_end = (arrow_start[0], arrow_start[1] - arrow_length)
                        
                        cv2.arrowedLine(rink_vis, arrow_start, arrow_end, (255, 255, 0), 2, tipLength=0.3)
            
            # Create warped broadcast frame if homography matrix is available
            if "homography_matrix" in frame_data and frame_data["homography_matrix"] is not None:
                warped_frame = self.homography_calculator.warp_frame(
                    frame, frame_data["homography_matrix"], rink_dims
                )
                
                if warped_frame is not None:
                    visualizations["warped_broadcast"] = warped_frame
                    
                    # Create overlay of warped frame on rink
                    # Use alpha blending to overlay the warped frame onto the rink
                    alpha = 0.6  # Transparency factor
                    overlay = rink_vis.copy()
                    
                    # Create a mask for non-zero (non-black) pixels in the warped frame
                    non_zero_mask = (warped_frame.sum(axis=2) > 0).astype(np.uint8) * 255
                    
                    # Apply the mask to blend only the non-black parts of the warped frame
                    for c in range(3):  # For each color channel
                        overlay[:, :, c] = np.where(
                            non_zero_mask > 0,
                            overlay[:, :, c] * (1 - alpha) + warped_frame[:, :, c] * alpha,
                            overlay[:, :, c]
                        )
                    
                    visualizations["overlay"] = overlay
            
            visualizations["rink"] = rink_vis
            
            # Add frame ID to visualization
            cv2.putText(rink_vis, f"Frame: {frame_data['frame_id']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return visualizations
    
    def save_tracking_data(self, output_path: str = None) -> str:
        """
        Save tracking data to JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Path to the saved JSON file
        """
        if output_path is None:
            if self.output_dir is None:
                raise ValueError("No output directory specified")
            output_path = os.path.join(self.output_dir, "tracking_data.json")
        
        # Convert tracking data to serializable format
        serializable_data = {}
        for frame_id, frame_data in self.tracking_data.items():
            serializable_frame = frame_data.copy()
            serializable_data[str(frame_id)] = serializable_frame
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        
        return output_path
