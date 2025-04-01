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


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that can handle numpy arrays and other non-serializable types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


class PlayerTracker:
    """
    Main module that integrates all components to track players in hockey broadcast footage.
    """

    def __init__(
        self,
        detection_model_path: str,
        orientation_model_path: str,
        output_dir: str = None,
        segmentation_model_path: Optional[str] = None,
        rink_coordinates_path: Optional[str] = None,
        device: str = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    ):
        """
        Initialize the player tracker.
        
        Args:
            detection_model_path: Path to detection model
            orientation_model_path: Path to orientation model
            output_dir: Directory to save outputs
            segmentation_model_path: Optional path to segmentation model
            rink_coordinates_path: Optional path to rink coordinates JSON
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = device
        
        # Initialize output directory
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize components
        self.player_detector = PlayerDetector(
            model_path=detection_model_path, 
            device=device,
            output_dir=output_dir
        )
        self.orientation_detector = OrientationDetector(orientation_model_path, device)
        
        # Initialize optional components
        self.segmentation_processor = None
        if segmentation_model_path:
            self.segmentation_processor = SegmentationProcessor(segmentation_model_path, device)
            
        self.homography_calculator = None
        if rink_coordinates_path:
            self.homography_calculator = HomographyCalculator(rink_coordinates_path)
        
        # Initialize tracking data
        self.tracking_data = {}
        
    def process_frame(self, frame: np.ndarray, frame_id: int, debug_mode: bool = False) -> Dict:
        """
        Process a single frame to track players.
        
        Args:
            frame: Input frame (BGR format)
            frame_id: Frame identifier
            debug_mode: Enable extra debugging output
            
        Returns:
            Dictionary containing processed data for the frame
        """
        frame_height, frame_width = frame.shape[:2]
        frame_data = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "players": []
        }
        
        # Step 1: Process through segmentation model if available
        if self.segmentation_processor:
            segmentation_features = self.segmentation_processor.process_frame(
                frame, frame_id, self.output_dir
            )
            frame_data["segmentation_features"] = segmentation_features
            
            # Step 2: Calculate homography if rink coordinates are available
            if self.homography_calculator:
                success, homography_matrix, debug_info = (
                    self.homography_calculator.calculate_homography(
                        segmentation_features["features"]
                    )
                )
                frame_data["homography_success"] = success
                frame_data["homography_debug_info"] = debug_info
                
                if success and homography_matrix is not None:
                    frame_data["homography_matrix"] = homography_matrix
                    print(f"Homography calculation successful for frame {frame_id}")
                else:
                    print(
                        f"Homography calculation failed for frame {frame_id}: "
                        f"{debug_info.get('reason_for_failure', 'unknown')}"
                    )
        
        # Step 3: Detect players
        player_detections = self.player_detector.process_frame(frame, frame_id)
        
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
                "reference_point": {
                    "x": detection["reference_point"]["x"],
                    "y": detection["reference_point"]["y"],
                    "pixel_x": detection["reference_point"]["pixel_x"],
                    "pixel_y": detection["reference_point"]["pixel_y"]
                }
            }
            
            # Add orientation if available
            if i in player_orientations:
                player_data["orientation"] = player_orientations[i]["orientation"]
                player_data["orientation_confidence"] = player_orientations[i]["confidence"]
            
            # Map position to rink if homography is available
            if self.homography_calculator and frame_data.get("homography_success") and "homography_matrix" in frame_data:
                try:
                    # Extract reference point coordinates
                    ref_point = [(float(detection["reference_point"]["x"]), float(detection["reference_point"]["y"]))]
                    
                    # Transform reference point to rink coordinates
                    rink_positions = self.homography_calculator.apply_homography(ref_point, frame_data["homography_matrix"])
                    
                    if rink_positions:  # Check if we got any positions back
                        player_data["rink_position"] = {
                            "x": float(rink_positions[0][0]),  # Ensure coordinates are float
                            "y": float(rink_positions[0][1]),
                            "pixel_x": int(rink_positions[0][0]),  # Add pixel-space coordinates
                            "pixel_y": int(rink_positions[0][1])
                        }
                except Exception as e:
                    print(f"Error applying homography to player: {e}")
            
            frame_data["players"].append(player_data)
        
        # Store frame data
        self.tracking_data[frame_id] = frame_data
        
        return frame_data
    
    def visualize_frame(self, frame: np.ndarray, frame_data: Dict, rink_image: np.ndarray = None, debug_mode: bool = False) -> Dict[str, np.ndarray]:
        """
        Create visualizations for the processed frame.
        
        Args:
            frame: The original video frame
            frame_data: The processed frame data
            rink_image: The rink template image (optional)
            debug_mode: Whether to generate additional debug visualizations
            
        Returns:
            Dictionary containing different visualizations
        """
        visualizations = {}
        
        # Create a copy of the frame for visualization
        broadcast_vis = frame.copy()
        
        # Extract data
        homography_success = frame_data.get("homography_success", False)
        segmentation_features = frame_data.get("segmentation_features", {}).get("features", {})
        players = frame_data.get("players", [])
        frame_idx = frame_data.get("frame_id", 0)
        
        # --- BROADCAST VIEW VISUALIZATION ---
        
        # Draw segmentation features if available
        if self.homography_calculator and segmentation_features:
            if hasattr(self.homography_calculator, 'draw_visualization'):
                homography_matrix = None
                if homography_success and "homography_matrix" in frame_data:
                    homography_matrix = frame_data["homography_matrix"]
                
                broadcast_vis = self.homography_calculator.draw_visualization(
                    broadcast_vis, 
                    segmentation_features,
                    homography_matrix
                )
            else:
                broadcast_vis = self.homography_calculator.draw_segmentation_lines(
                    broadcast_vis, 
                    segmentation_features
                )
                
                if homography_success is not None:
                    status_text = "Homography: Success" if homography_success else "Homography: Failed"
                    cv2.putText(broadcast_vis, status_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw player detections on broadcast view
        for player in players:
            if "reference_point" in player:
                pos = player["reference_point"]
                x, y = int(pos["pixel_x"]), int(pos["pixel_y"])
                
                # Draw player marker (blue dot)
                cv2.circle(broadcast_vis, (x, y), 4, (255, 0, 0), -1)  # Blue dot
                
                # Add player ID if available
                if "player_id" in player:
                    cv2.putText(broadcast_vis, str(player["player_id"]), 
                              (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add frame number
        cv2.putText(broadcast_vis, f"Frame: {frame_idx}", (10, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        visualizations["broadcast"] = broadcast_vis
        
        # --- RINK VIEW VISUALIZATION ---
        if rink_image is not None and self.homography_calculator and homography_success:
            rink_vis = rink_image.copy()
            
            # Draw players on rink view
            for player in players:
                if "rink_position" in player:
                    rink_pos = player["rink_position"]
                    rx, ry = int(rink_pos["pixel_x"]), int(rink_pos["pixel_y"])
                    
                    # Only draw players within rink boundaries
                    if 0 <= rx < rink_vis.shape[1] and 0 <= ry < rink_vis.shape[0]:
                        # Draw player marker (blue dot)
                        cv2.circle(rink_vis, (rx, ry), 4, (255, 0, 0), -1)  # Blue dot
                        
                        # Add player ID and orientation if available
                        label = player.get("player_id", "")
                        if "orientation" in player:
                            label += f" ({player['orientation']})"
                        if label:
                            cv2.putText(rink_vis, label, (rx + 10, ry - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            visualizations["rink"] = rink_vis
        
        return visualizations
    
    def save_tracking_data(self, output_path: str) -> str:
        """
        Save tracking data to a JSON file.
        
        Args:
            output_path: Path to save tracking data
            
        Returns:
            Path to the saved file
        """
        # Create a serializable copy of the tracking data
        serializable_data = {}
        
        print(f"Preparing tracking data for saving ({len(self.tracking_data)} frames)...")
        
        # Helper function to filter out non-serializable objects
        def filter_non_serializable(obj):
            if isinstance(obj, dict):
                return {k: filter_non_serializable(v) for k, v in obj.items() 
                        if k not in ['segmentation_mask', 'raw_masks', 'overlay_visualization']}
            elif isinstance(obj, list):
                return [filter_non_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, np.ndarray):
                # Only keep small arrays
                if obj.size <= 100:
                    return obj.tolist()
                return "large_array_removed"
            else:
                return str(obj)
        
        # Process each frame separately
        for frame_id, frame_data in self.tracking_data.items():
            try:
                serializable_frame = filter_non_serializable(frame_data)
                serializable_data[str(frame_id)] = serializable_frame
            except Exception as e:
                print(f"Error processing frame {frame_id}: {str(e)}")
                continue
        
        try:
            print(f"Saving {len(serializable_data)} frames to {output_path}...")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with error handling
            with open(output_path, 'w') as f:
                json.dump(serializable_data, f, indent=2, cls=NumpyEncoder)
            
            # Verify the file was created successfully
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"Successfully saved tracking data to {output_path} ({file_size/1024:.1f} KB)")
            else:
                print(f"Failed to save tracking data to {output_path} - file not created")
                
            return output_path
            
        except Exception as e:
            print(f"Error saving tracking data: {str(e)}")
            
            # Try saving in a simpler format as backup
            backup_path = output_path.replace('.json', '_backup.json')
            try:
                print(f"Attempting to save basic tracking data to {backup_path}...")
                with open(backup_path, 'w') as f:
                    json.dump({"frames": list(serializable_data.keys())}, f)
                return backup_path
            except Exception as e2:
                print(f"Failed to save backup tracking data: {str(e2)}")
                return None
