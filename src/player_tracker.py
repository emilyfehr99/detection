import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import time

from segmentation_processor import SegmentationProcessor
from player_detector import PlayerDetector
from orientation_detector import OrientationDetector
from homography_calculator import HomographyCalculator
from jersey_color_detector import JerseyColorDetector
from ultralytics import YOLO


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
        orientation_model_path: str,
        detection_model_path: str = None,
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
            api_key="YDZxw1AQEvclkzV0ZLOz",
            workspace_name="hockey-fghn7", 
            workflow_id="custom-workflow-3",
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
        
        # Initialize jersey color detector for team identification
        self.jersey_detector = JerseyColorDetector()
        
        # Initialize tracking data
        self.tracking_data = {}
        
        # Initialize player tracking across frames
        self.next_player_id = 0
        self.player_tracks = {}  # Maps player_id to track info
        self.frame_player_matches = {}  # Maps frame_id to player_id mappings
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
    def track_player_across_frames(self, detection: Dict, frame_id: int) -> str:
        """
        Track a player across frames and assign consistent ID.
        
        Args:
            detection: Player detection data
            frame_id: Current frame ID
            
        Returns:
            Consistent player ID across frames
        """
        if frame_id == 0:
            # First frame: assign new ID
            player_id = f"player_{self.next_player_id}"
            self.next_player_id += 1
            
            # Initialize track
            self.player_tracks[player_id] = {
                "first_frame": frame_id,
                "last_frame": frame_id,
                "bbox_history": [detection["bbox"]],
                "roboflow_class": detection.get("roboflow_class", "unknown")
            }
            
            return player_id
        
        # Find the best matching player from previous frame
        best_match_id = None
        best_match_score = 0.5  # Minimum threshold for matching
        
        if frame_id - 1 in self.frame_player_matches:
            prev_matches = self.frame_player_matches[frame_id - 1]
            
            for prev_player_id, prev_player_data in prev_matches.items():
                if prev_player_id not in self.player_tracks:
                    continue
                
                # Calculate similarity score based on position and class
                score = self._calculate_player_similarity(detection, prev_player_data)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_id = prev_player_id
        
        if best_match_id:
            # Update existing track
            self.player_tracks[best_match_id]["last_frame"] = frame_id
            self.player_tracks[best_match_id]["bbox_history"].append(detection["bbox"])
            return best_match_id
        else:
            # New player: assign new ID
            player_id = f"player_{self.next_player_id}"
            self.next_player_id += 1
            
            # Initialize track
            self.player_tracks[player_id] = {
                "first_frame": frame_id,
                "last_frame": frame_id,
                "bbox_history": [detection["bbox"]],
                "roboflow_class": detection.get("roboflow_class", "unknown")
            }
            
            return player_id
    
    def _calculate_player_similarity(self, detection1: Dict, detection2: Dict) -> float:
        """
        Calculate similarity score between two player detections.
        
        Args:
            detection1: First detection
            detection2: Second detection
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Position similarity (using bounding box center)
        bbox1 = detection1["bbox"]
        bbox2 = detection2["bbox"]
        
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Convert distance to similarity score (closer = higher score)
        # Assuming reasonable movement between frames (e.g., max 100 pixels)
        max_distance = 100.0
        position_score = max(0, 1.0 - (distance / max_distance))
        
        # Class similarity (same Roboflow class = higher score)
        class1 = detection1.get("roboflow_class", "unknown")
        class2 = detection2.get("roboflow_class", "unknown")
        class_score = 1.0 if class1 == class2 else 0.5
        
        # Combined score (weighted average)
        final_score = 0.7 * position_score + 0.3 * class_score
        
        return final_score
    
    def calculate_player_metrics(self, current_player: Dict, frame_id: int, prev_frame_data: Optional[Dict] = None) -> Dict:
        """
        Calculate metrics for a player based on current and previous positions.
        
        Args:
            current_player: Current player data dictionary
            frame_id: Current frame ID
            prev_frame_data: Previous frame data dictionary
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {
            "speed": 0.0,
            "acceleration": 0.0,
            "orientation": current_player.get("orientation", 0.0)
        }
        
        # If no rink position, return default metrics
        if not current_player.get("rink_position"):
            return metrics
            
        current_pos = current_player["rink_position"]
        
        # Find the same player in previous frame
        if prev_frame_data and "players" in prev_frame_data:
            # Match player by closest position
            prev_players = [p for p in prev_frame_data["players"] if p.get("rink_position")]
            if prev_players:
                # Find closest previous player of same type
                closest_prev = min(
                    [p for p in prev_players if p["type"] == current_player["type"]], 
                    key=lambda p: np.sqrt(
                        (p["rink_position"][0] - current_pos[0])**2 + 
                        (p["rink_position"][1] - current_pos[1])**2
                    ),
                    default=None
                )
                
                if closest_prev:
                    prev_pos = closest_prev["rink_position"]
                    # Calculate speed (pixels/frame)
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                    metrics["speed"] = np.sqrt(dx**2 + dy**2)
                    
                    # Calculate acceleration if previous speed available
                    prev_speed = closest_prev.get("speed", 0.0)
                    metrics["acceleration"] = metrics["speed"] - prev_speed
                    
                    # Calculate orientation (in degrees)
                    if dx != 0 or dy != 0:
                        metrics["orientation"] = np.degrees(np.arctan2(dy, dx))
        
        return metrics

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
        
        # Get previous frame data if available
        prev_frame_data = self.tracking_data.get(frame_id - 1)
        
        # Step 1: Process through segmentation model if available
        if self.segmentation_processor:
            segmentation_result = self.segmentation_processor.process_frame(
                frame, frame_id, self.output_dir
            )
            frame_data["segmentation_features"] = segmentation_result
            
            # Calculate homography if we have a homography calculator
            if self.homography_calculator:
                try:
                    # Pass the features to the homography calculator
                    homography_matrix = self.homography_calculator.calculate_homography(
                        segmentation_result["features"],
                        frame_id  # Pass frame_id for caching
                    )
                    if homography_matrix is not None:
                        frame_data["homography_matrix"] = homography_matrix.tolist()
                        frame_data["homography_success"] = True
                        frame_data["homography_source"] = "original"  # Mark as an original calculation
                    else:
                        # Try to get an interpolated matrix
                        homography_matrix = self.homography_calculator.get_homography_matrix(frame_id)
                        if homography_matrix is not None:
                            frame_data["homography_matrix"] = homography_matrix.tolist()
                            frame_data["homography_success"] = True
                            frame_data["homography_interpolated"] = True
                            frame_data["homography_source"] = "fallback"  # Mark as a fallback, to be interpolated later
                        else:
                            frame_data["homography_success"] = False
                except Exception as e:
                    self.logger.error(f"Error calculating homography: {e}")
                    frame_data["homography_success"] = False
        
        # Step 2: Detect players
        if self.player_detector:
            detections = self.player_detector.process_frame(frame, frame_id)
            
            # Step 3: Process each detection
            for i, detection in enumerate(detections):
                # Track player across frames and assign consistent ID
                player_id = self.track_player_across_frames(detection, frame_id)
                
                player_data = {
                    "player_id": player_id,  # Consistent ID across frames
                    "type": detection["class"],
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                    "reference_point": detection["reference_point"],
                    "roboflow_class": detection.get("roboflow_class", "unknown")  # Include Roboflow class
                }
                
                # Project player position to rink coordinates if homography available
                if frame_data.get("homography_success", False):
                    try:
                        rink_pos = self.homography_calculator.project_point_to_rink(
                            (detection["reference_point"]["x"], detection["reference_point"]["y"]),
                            frame_data["homography_matrix"]
                        )
                        if rink_pos:
                            player_data["rink_position"] = rink_pos
                            
                            # Calculate metrics using previous frame data
                            metrics = self.calculate_player_metrics(player_data, frame_id, prev_frame_data)
                            player_data.update(metrics)
                    except Exception as e:
                        self.logger.error(f"Error projecting point: {e}")
                
                # Add team detection using jersey color analysis
                try:
                    team_info = self.jersey_detector.detect_team(frame, detection["bbox"])
                    player_data["team"] = self.jersey_detector.get_team_display_name(team_info)
                    player_data["team_confidence"] = team_info.get("confidence", 0.0)
                    player_data["team_detection_method"] = team_info.get("method", "unknown")
                except Exception as e:
                    self.logger.warning(f"Error in team detection: {e}")
                    player_data["team"] = "Unknown"
                    player_data["team_confidence"] = 0.0
                    player_data["team_detection_method"] = "unknown"
                
                frame_data["players"].append(player_data)
        
        # Store frame player matches for tracking
        self.frame_player_matches[frame_id] = {
            player_data["player_id"]: player_data 
            for player_data in frame_data["players"]
        }
        
        # Store frame data for next frame's calculations
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
        
        # Process each frame separately
        for frame_id, frame_data in self.tracking_data.items():
            try:
                # Only keep essential data
                serializable_frame = {
                    "frame_id": frame_data["frame_id"],
                    "timestamp": frame_data["timestamp"],
                    "players": [
                        {
                            "player_id": p["player_id"],
                            "type": p["type"],
                            "bbox": p["bbox"],
                            "rink_position": p.get("rink_position", None)
                        } for p in frame_data["players"]
                    ],
                    "homography_success": frame_data.get("homography_success", False)
                }
                
                # Only include homography matrix if successful
                if frame_data.get("homography_success", False):
                    serializable_frame["homography_matrix"] = frame_data.get("homography_matrix", None)
                
                # Only include essential segmentation features
                if "segmentation_features" in frame_data:
                    serializable_frame["segmentation_features"] = {
                        "features": {
                            k: v for k, v in frame_data["segmentation_features"].get("features", {}).items()
                            if k in ["blue_lines", "center_line", "goal_lines"]
                        }
                    }
                
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
            return None

    def track_players(self, frame, frame_idx, timestamp):
        """Track players in a single frame."""
        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "players": [],
            "homography_success": False
        }
        
        # Run segmentation if we have a segmentation processor
        if self.segmentation_processor:
            segmentation_result = self.segmentation_processor.process_frame(frame)
            frame_data["segmentation_features"] = segmentation_result["features"]
            
            # Calculate homography if we have a homography calculator
            if self.homography_calculator:
                try:
                    # Pass the features to the homography calculator
                    homography_matrix = self.homography_calculator.calculate_homography(
                        segmentation_result["features"],
                        frame_idx  # Pass frame_idx for caching
                    )
                    if homography_matrix is not None:
                        frame_data["homography_matrix"] = homography_matrix.tolist()
                        frame_data["homography_success"] = True
                    else:
                        # Try to get an interpolated matrix
                        homography_matrix = self.homography_calculator.get_homography_matrix(frame_idx)
                        if homography_matrix is not None:
                            frame_data["homography_matrix"] = homography_matrix.tolist()
                            frame_data["homography_success"] = True
                            frame_data["homography_interpolated"] = True
                        else:
                            frame_data["homography_success"] = False
                except Exception as e:
                    self.logger.error(f"Error calculating homography: {e}")
                    frame_data["homography_success"] = False
        
        # Run player detection
        detection_result = self.player_detector.detect_players(frame)
        
        # Extract player detections
        players = []
        for i, bbox in enumerate(detection_result["boxes"]):
            x1, y1, x2, y2 = bbox
            # Get class and confidence
            class_id = detection_result["classes"][i]
            confidence = detection_result["scores"][i]
            
            # Extract player orientation if orientation model is available
            orientation = None
            orientation_confidence = None
            if self.player_orientation_estimator:
                try:
                    player_img = frame[int(y1):int(y2), int(x1):int(x2)]
                    if player_img.size > 0:  # Ensure valid crop
                        orient_result = self.player_orientation_estimator.estimate_orientation(player_img)
                        orientation = orient_result["orientation"]
                        orientation_confidence = orient_result["confidence"]
                except Exception as e:
                    self.logger.error(f"Error estimating player orientation: {e}")
            
            # Calculate player position on the rink if homography is available
            rink_position = None
            if frame_data.get("homography_success", False) and "homography_matrix" in frame_data:
                try:
                    # Use the center bottom point of the bounding box as the player's position
                    player_x = (x1 + x2) / 2
                    player_y = y2  # Bottom of the bounding box
                    
                    # Convert from list back to numpy array
                    homography_matrix = np.array(frame_data["homography_matrix"])
                    
                    # Project the point to the rink coordinates
                    rink_position = self.homography_calculator.project_point_to_rink(
                        (player_x, player_y), 
                        homography_matrix
                    )
                except Exception as e:
                    self.logger.error(f"Error projecting player position: {e}")
            
            player_data = {
                "bbox": bbox.tolist(),
                "class_id": int(class_id),
                "confidence": float(confidence),
                "orientation": orientation,
                "orientation_confidence": orientation_confidence,
                "rink_position": rink_position
            }
            
            players.append(player_data)
        
        frame_data["players"] = players
        return frame_data

    def process_video_clip(self, video_path, start_second=0, num_seconds=5, frame_step=1, max_frames=None):
        """Process a clip from a video file."""
        results = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return {"error": f"Could not open video: {video_path}"}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            start_frame = int(start_second * fps)
            if num_seconds > 0:
                end_frame = int((start_second + num_seconds) * fps)
            else:
                end_frame = total_frames
            
            # Apply max_frames limit if provided
            if max_frames is not None and (end_frame - start_frame) > max_frames:
                end_frame = start_frame + max_frames
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_idx = start_frame
            frames_dir = os.path.join(self.output_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            
            frame_results = []
            
            # Process frames
            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every n-th frame
                if (frame_idx - start_frame) % frame_step == 0:
                    timestamp = frame_idx / fps
                    
                    # Track players
                    frame_data = self.track_players(frame, frame_idx, timestamp)
                    frame_results.append(frame_data)
                    
                    # Save frame
                    output_idx = (frame_idx - start_frame) // frame_step
                    frame_path = os.path.join(frames_dir, f"frame_{output_idx}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    self.logger.info(f"Processed frame {frame_idx} (output idx: {output_idx})")
                
                frame_idx += 1
                
                # Check for early termination
                if max_frames is not None and (frame_idx - start_frame) >= max_frames * frame_step:
                    break
            
            # Now interpolate missing homography matrices
            self.interpolate_missing_homography(frame_results)
            
            # Prepare final results
            results = {
                "video_info": {
                    "path": video_path,
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": len(frame_results)
                },
                "frames": frame_results
            }
            
            # Save results to file
            output_file = os.path.join(self.output_dir, f"player_detection_data_{int(time.time())}.json")
            with open(output_file, "w") as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)
            
            self.logger.info(f"Results saved to {output_file}")
            
            # Create visualization HTML
            self.create_visualization(results, self.output_dir)
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            results = {"error": str(e)}
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
        
        return results
    
    def interpolate_missing_homography(self, frame_results):
        """Interpolate missing homography matrices for frames where calculation failed or used fallback."""
        # Collect frame indices with successfully calculated homography (not fallback)
        successful_original_frames = {}
        for i, frame_data in enumerate(frame_results):
            # Only use frames with original homography as interpolation sources
            if frame_data.get("homography_success", False) and frame_data.get("homography_source") == "original":
                successful_original_frames[frame_data["frame_idx"]] = i
        
        # Skip if we have less than 2 original frames
        if len(successful_original_frames) < 2:
            self.logger.warning("Insufficient original homography frames for interpolation")
            return
        
        # Report initial state
        self.logger.info(f"Starting true homography interpolation:")
        self.logger.info(f"Found {len(successful_original_frames)} frames with original homography matrices")
        
        # Count frames that need interpolation (fallback frames)
        frames_needing_interpolation = []
        for i, frame_data in enumerate(frame_results):
            if frame_data.get("homography_source") == "fallback":
                frames_needing_interpolation.append(frame_data["frame_idx"])
        
        self.logger.info(f"Need to interpolate {len(frames_needing_interpolation)} frames with fallback matrices")
        
        # Get sorted list of frame indices with original homography
        sorted_indices = sorted(successful_original_frames.keys())
        self.logger.info(f"Original homography frames: {sorted_indices}")
        
        # Get sorted list of frames needing interpolation
        sorted_frames_to_interpolate = sorted(frames_needing_interpolation)
        
        # Do multiple passes to ensure all frames have a chance to be interpolated
        for pass_num in range(3):  # Do three passes to ensure better results
            self.logger.info(f"\nInterpolation pass {pass_num+1}")
            frames_interpolated_this_pass = 0
            
            # Interpolate for each frame with fallback homography
            for frame_idx in sorted_frames_to_interpolate[:]:  # Use a copy to safely modify the list
                # Get the frame data
                frame_idx_in_results = None
                for i, frame_data in enumerate(frame_results):
                    if frame_data["frame_idx"] == frame_idx:
                        frame_idx_in_results = i
                        break
                
                if frame_idx_in_results is None:
                    continue
                
                frame_data = frame_results[frame_idx_in_results]
                
                # Skip if this frame is no longer a fallback (it was interpolated in a previous pass)
                if frame_data.get("homography_source") != "fallback":
                    sorted_frames_to_interpolate.remove(frame_idx)
                    continue
                
                # Find the closest original frames before and after
                before_idx = None
                after_idx = None
                
                # Find closest before and after indices from original frames
                before_indices = [idx for idx in sorted_indices if idx < frame_idx]
                after_indices = [idx for idx in sorted_indices if idx > frame_idx]
                
                if before_indices:
                    before_idx = max(before_indices)
                
                if after_indices:
                    after_idx = min(after_indices)
                
                # Interpolate only if we have both before and after frames
                if before_idx is not None and after_idx is not None:
                    before_matrix = np.array(frame_results[successful_original_frames[before_idx]]["homography_matrix"])
                    after_matrix = np.array(frame_results[successful_original_frames[after_idx]]["homography_matrix"])
                    
                    # Calculate interpolation factor
                    t = (frame_idx - before_idx) / (after_idx - before_idx)
                    
                    # Use the homography calculator's interpolation method
                    interpolated = self.homography_calculator.interpolate_homography(before_matrix, after_matrix, t)
                    
                    # Update the frame data
                    frame_data["homography_matrix"] = interpolated.tolist()
                    frame_data["homography_success"] = True
                    frame_data["homography_source"] = "interpolated"  # Mark as properly interpolated
                    frame_data["interpolation_details"] = {
                        "method": "true_interpolation",
                        "before_frame": before_idx,
                        "after_frame": after_idx,
                        "t_factor": t
                    }
                    
                    # Remove this frame from the list of frames to interpolate
                    sorted_frames_to_interpolate.remove(frame_idx)
                    frames_interpolated_this_pass += 1
                    
                    self.logger.info(f"  Frame {frame_idx}: TRUE INTERPOLATION (t={t:.2f}, between frames {before_idx} and {after_idx})")
                
                # If we only have a before frame but no after frame, keep using the before frame
                elif before_idx is not None:
                    # We already have the before matrix, no need to change it
                    # Just update the metadata to be clearer about what happened
                    frame_data["interpolation_details"] = {
                        "method": "before_fallback",
                        "before_frame": before_idx,
                        "note": "Kept existing fallback, no after frame available for interpolation"
                    }
                    
                    sorted_frames_to_interpolate.remove(frame_idx)
                    frames_interpolated_this_pass += 1
                    self.logger.info(f"  Frame {frame_idx}: Kept fallback from frame {before_idx} (no after frame)")
                
                # If we only have an after frame but no before frame, use the after frame
                elif after_idx is not None:
                    after_matrix = np.array(frame_results[successful_original_frames[after_idx]]["homography_matrix"])
                    frame_data["homography_matrix"] = after_matrix.tolist()
                    frame_data["homography_source"] = "from_after"
                    frame_data["interpolation_details"] = {
                        "method": "after_fallback",
                        "after_frame": after_idx,
                        "note": "Used after frame instead of fallback, no before frame available"
                    }
                    
                    sorted_frames_to_interpolate.remove(frame_idx)
                    frames_interpolated_this_pass += 1
                    self.logger.info(f"  Frame {frame_idx}: Using matrix from next frame {after_idx}")
            
            # Update the number of successful frames after each pass
            self.logger.info(f"Pass {pass_num+1} complete: Interpolated {frames_interpolated_this_pass} frames")
            self.logger.info(f"Remaining frames to interpolate: {len(sorted_frames_to_interpolate)}")
            
            # If we didn't interpolate any frames in this pass, or all frames have been interpolated, break
            if frames_interpolated_this_pass == 0 or len(sorted_frames_to_interpolate) == 0:
                self.logger.info(f"No more frames to interpolate, ending interpolation passes")
                break
    
    def create_visualization(self, results, output_dir):
        # Implementation of create_visualization method
        pass
