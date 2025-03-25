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
            "players": [],
            "homography_success": False
        }
        
        # Step 1: Process through segmentation model to get rink features
        segmentation_features = self.segmentation_processor.process_frame(frame, frame_id, self.output_dir)
        
        # Save the segmentation features for visualization
        frame_data["segmentation_features"] = segmentation_features
        
        # Step 2: Calculate homography based on segmentation features
        # Extract the 'features' key from the segmentation results
        success, homography_matrix, debug_info = self.homography_calculator.calculate_homography(
            segmentation_features["features"]
        )
        
        # Store homography results in frame data
        frame_data["homography_success"] = success
        frame_data["homography_debug_info"] = debug_info
        
        if success and homography_matrix is not None:
            frame_data["homography_matrix"] = homography_matrix
            
            # Store additional debug info about homography
            print(f"Homography calculation successful for frame {frame_id}")
        else:
            print(f"Homography calculation failed for frame {frame_id}: {debug_info.get('reason_for_failure', 'unknown')}")
            
            # We still continue processing even if homography fails
            # The visualization will show segmentation features but player tracking will be limited
        
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
            if success and homography_matrix is not None:
                try:
                    rink_positions = self.homography_calculator.apply_homography(
                        [detection["reference_point"]], homography_matrix
                    )
                    
                    if rink_positions:  # Check if we got any positions back
                        player_data["rink_position"] = rink_positions[0]
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
        
        # Draw segmentation features using the homography calculator's draw_visualization 
        # This will ensure features are drawn even if homography calculation failed
        if hasattr(self.homography_calculator, 'draw_visualization'):
            # Use the new method if available (backward compatibility)
            homography_matrix = None
            if homography_success and "homography_matrix" in frame_data:
                homography_matrix = frame_data["homography_matrix"]
            
            broadcast_vis = self.homography_calculator.draw_visualization(
                broadcast_vis, 
                segmentation_features,
                homography_matrix
            )
        else:
            # Fallback to the old method (draw segmentation lines)
            broadcast_vis = self.homography_calculator.draw_segmentation_lines(
                broadcast_vis, 
                segmentation_features
            )
            
            # Add text about homography status
            status_text = "Homography: Success" if homography_success else "Homography: Failed"
            cv2.putText(broadcast_vis, status_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw player detections on broadcast view
        for player in players:
            if "position" in player:
                pos = player["position"]
                x, y = int(pos["x"]), int(pos["y"])
                
                # Draw player marker (circle)
                color = (0, 255, 0) if homography_success else (0, 165, 255)  # Green if homography succeeded, orange otherwise
                cv2.circle(broadcast_vis, (x, y), 10, color, -1)
                
                # Add player ID if available
                if "id" in player:
                    cv2.putText(broadcast_vis, str(player["id"]), 
                              (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame number
        cv2.putText(broadcast_vis, f"Frame: {frame_idx}", (10, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        visualizations["broadcast"] = broadcast_vis
        
        # --- RINK VIEW VISUALIZATION ---
        if rink_image is not None:
            rink_vis = rink_image.copy()
            
            # Draw players on rink view (only if homography succeeded)
            if homography_success:
                for player in players:
                    if "rink_position" in player:
                        rink_pos = player["rink_position"]
                        rx, ry = int(rink_pos["x"]), int(rink_pos["y"])
                        
                        # Only draw players within rink boundaries
                        if 0 <= rx < rink_vis.shape[1] and 0 <= ry < rink_vis.shape[0]:
                            # Draw player marker
                            cv2.circle(rink_vis, (rx, ry), 15, (0, 0, 255), -1)
                            
                            # Add player ID if available
                            if "id" in player:
                                cv2.putText(rink_vis, str(player["id"]), 
                                          (rx + 15, ry - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Add text about homography failure
                cv2.putText(rink_vis, "Homography failed - No player tracking", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Add frame number
            cv2.putText(rink_vis, f"Frame: {frame_idx}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            visualizations["rink"] = rink_vis
        
        # --- DEBUG VISUALIZATIONS ---
        if debug_mode:
            # If there are raw segmentation masks, visualize them
            if "raw_masks" in frame_data.get("segmentation_features", {}):
                raw_masks = frame_data["segmentation_features"]["raw_masks"]
                
                # Create a visualization of raw masks with different colors
                mask_vis = frame.copy()
                alpha = 0.5  # Transparency for overlay
                
                # Color mapping for different classes (same as in segmentation_processor.py)
                color_map = {
                    "Rink": (0, 200, 0),       # Green
                    "BlueLine": (255, 0, 0),    # Blue
                    "RedCenterLine": (0, 0, 255),  # Red
                    "GoalLine": (255, 0, 255),  # Magenta
                    "RedCircle": (0, 255, 255),  # Yellow
                    "FaceoffCircle": (255, 255, 0)  # Cyan
                }
                
                # Create an overlay image with all masks
                overlay = np.zeros_like(mask_vis)
                
                # Draw each mask class with its color
                for class_name, mask in raw_masks.items():
                    if class_name in color_map:
                        color = color_map[class_name]
                        
                        # Use much lower opacity for the Rink class
                        class_alpha = 0.2 if class_name == "Rink" else alpha
                        
                        # Apply the mask with the appropriate color
                        mask_area = np.where(mask)
                        overlay[mask_area] = color
                        
                # Blend with original image
                cv2.addWeighted(overlay, alpha, mask_vis, 1 - alpha, 0, mask_vis)
                
                # Add a title to the visualization
                cv2.putText(mask_vis, "Segmentation Raw Masks", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                visualizations["raw_masks"] = mask_vis
        
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
