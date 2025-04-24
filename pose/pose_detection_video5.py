from ultralytics import YOLO
import cv2
from pathlib import Path
import torch
import sys
import os
import numpy as np
import math

class PoseDetector:
    def __init__(self):
        """Initialize pose detection system"""
        # Set up device - optimize for Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon MPS acceleration")
        else:
            self.device = torch.device('cpu')
            print("MPS not available, falling back to CPU")
            
        # Initialize YOLOv8 model for pose detection
        print("Loading YOLOv8 pose model...")
        self.model = YOLO('yolov8x-pose.pt')
        self.model.to(self.device)
        
        # Player tracking system
        self.tracked_players = {}  # Dictionary to track players across frames
        self.max_player_id = 12    # Maximum player number
        self.clip_frame = 0        # Counter for frames within current 4-second clip
        self.total_frames = 0      # Counter for total frames processed
        self.reset_interval = 120  # Reset tracking every 120 frames (4 seconds at 30fps)
        
        # Player position history for better tracking
        self.player_history = {}   # Store recent positions for each player
        self.history_length = 10   # Number of frames to keep in history
        
        # Shot detection variables
        self.prev_wrist_pos = {}   # Track wrist positions for each player
        self.frame_time = 1/30.0   # Default to 30fps, will be updated from video
        self.shot_states = {}      # Track shooting states for each player
        self.total_shots = 0       # Counter for total shots detected
        self.debug_mode = True     # Enable debug visualization
        
    def find_next_available_id(self):
        """Find the next available player ID"""
        # Start from 1 to reserve 0 for uncertain cases
        for i in range(1, self.max_player_id):
            if i not in self.tracked_players:
                return i
        return 0  # Return 0 if no IDs are available
        
    def get_player_id(self, center, bbox, prev_id=None):
        """
        Get stable player ID based on position and history
        
        Args:
            center: (x, y) tuple of player center position
            bbox: (x1, y1, x2, y2) tuple of bounding box
            prev_id: Previous ID if known
            
        Returns:
            int: Player ID
        """
        # Reset tracking every 4 seconds (120 frames)
        if self.clip_frame >= self.reset_interval:
            self.clip_frame = 0
            self.tracked_players.clear()
            self.player_history.clear()
            return self.find_next_available_id()
            
        # Calculate search radius based on player size
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        search_radius = max(50, (box_width + box_height) / 4)  # Adaptive radius
        
        best_match_id = None
        min_distance = float('inf')
        
        # First, try to match with currently tracked players
        for player_id, player_data in self.tracked_players.items():
            dist = np.sqrt((center[0] - player_data['center'][0])**2 + 
                         (center[1] - player_data['center'][1])**2)
            if dist < search_radius and dist < min_distance:
                min_distance = dist
                best_match_id = player_id
        
        # If no direct match, check player history
        if best_match_id is None and prev_id is not None:
            if prev_id in self.player_history:
                for hist_center in self.player_history[prev_id]:
                    dist = np.sqrt((center[0] - hist_center[0])**2 + 
                                 (center[1] - hist_center[1])**2)
                    if dist < search_radius * 1.5:  # Slightly larger radius for history
                        best_match_id = prev_id
                        break
        
        # If still no match, assign new ID
        if best_match_id is None:
            best_match_id = self.find_next_available_id()
        
        # Update tracking data
        self.tracked_players[best_match_id] = {
            'center': center,
            'bbox': bbox,
            'last_seen': 0  # Reset last seen counter
        }
        
        # Update position history
        if best_match_id not in self.player_history:
            self.player_history[best_match_id] = []
        self.player_history[best_match_id].append(center)
        if len(self.player_history[best_match_id]) > self.history_length:
            self.player_history[best_match_id].pop(0)
        
        return best_match_id

    def update_missing_players(self):
        """Update missing frame counter and remove old players"""
        # Increment last_seen counter for all tracked players
        for player_id in list(self.tracked_players.keys()):
            self.tracked_players[player_id]['last_seen'] = \
                self.tracked_players[player_id].get('last_seen', 0) + 1
            
            # Remove players that haven't been seen for too long
            if self.tracked_players[player_id]['last_seen'] > 30:  # 1 second at 30fps
                del self.tracked_players[player_id]
                if player_id in self.player_history:
                    del self.player_history[player_id]

    def detect_shooting_motion(self, keypoints):
        """
        Detect if the player is in a shooting motion and determine the phase
        
        Args:
            keypoints: Dictionary of pose keypoints from YOLO
            
        Returns:
            tuple: (is_shooting, shot_phase, confidence)
            - is_shooting: bool indicating if player is shooting
            - shot_phase: str indicating phase ('windup', 'release', 'follow_through', None)
            - confidence: float 0-1 indicating confidence in detection
        """
        # Extract key points
        right_shoulder = keypoints.get('right_shoulder', None)
        right_elbow = keypoints.get('right_elbow', None)
        right_wrist = keypoints.get('right_wrist', None)
        hip_center = keypoints.get('hip_center', None)
        
        if not all([right_shoulder, right_elbow, right_wrist, hip_center]):
            return False, None, 0.0
            
        # Calculate key angles and positions
        elbow_angle = self.calculate_joint_angle(
            right_shoulder, right_elbow, right_wrist
        )
        shoulder_height = right_shoulder[1] - hip_center[1]  # Y coordinates
        wrist_height = right_wrist[1] - hip_center[1]
        
        # Track wrist velocity for shot release detection
        wrist_pos = (right_wrist[0], right_wrist[1])
        wrist_velocity = 0
        
        if hasattr(self, 'prev_wrist_pos'):
            dx = wrist_pos[0] - self.prev_wrist_pos[0]
            dy = wrist_pos[1] - self.prev_wrist_pos[1]
            wrist_velocity = math.sqrt(dx**2 + dy**2) / self.frame_time
        
        self.prev_wrist_pos = wrist_pos
        
        # Define shooting phases based on biomechanical indicators
        is_shooting = False
        shot_phase = None
        confidence = 0.0
        
        # Windup phase
        if (70 < elbow_angle < 120 and 
            shoulder_height < wrist_height and 
            wrist_velocity < 2.0):
            is_shooting = True
            shot_phase = 'windup'
            confidence = 0.7
        
        # Release phase - characterized by rapid wrist movement
        elif (120 < elbow_angle < 160 and 
              wrist_velocity > 5.0 and 
              shoulder_height > wrist_height):
            is_shooting = True
            shot_phase = 'release'
            confidence = 0.9
        
        # Follow through phase
        elif (elbow_angle > 150 and 
              shoulder_height > wrist_height and 
              2.0 < wrist_velocity < 5.0):
            is_shooting = True
            shot_phase = 'follow_through'
            confidence = 0.8
            
        return is_shooting, shot_phase, confidence

    def identify_shooter(self, keypoints):
        """
        Identify if this player is likely the shooter based on pose
        
        Args:
            keypoints: Dictionary of pose keypoints from YOLO
            
        Returns:
            tuple: (is_shooter, confidence)
            - is_shooter: bool indicating if this player is likely shooting
            - confidence: float 0-1 indicating confidence in identification
        """
        # Extract key points
        right_shoulder = keypoints.get('right_shoulder', None)
        right_elbow = keypoints.get('right_elbow', None)
        right_wrist = keypoints.get('right_wrist', None)
        hip_center = keypoints.get('hip_center', None)
        
        if not all([right_shoulder, right_elbow, right_wrist, hip_center]):
            return False, 0.0
        
        # Calculate shooting indicators
        elbow_angle = self.calculate_joint_angle(
            right_shoulder, right_elbow, right_wrist
        )
        shoulder_rotation = self.calculate_shoulder_rotation(keypoints)
        hip_rotation = self.calculate_hip_rotation(keypoints)
        weight_transfer = self.calculate_weight_transfer(keypoints)
        
        # Score different aspects of shooting form
        form_scores = {
            'elbow_angle': 1.0 if 70 < elbow_angle < 160 else 0.0,
            'rotation_alignment': 1.0 if abs(shoulder_rotation - hip_rotation) > 20 else 0.0,
            'weight_transfer': 1.0 if weight_transfer > 60 else 0.0
        }
        
        # Calculate overall confidence
        confidence = sum(form_scores.values()) / len(form_scores)
        is_shooter = confidence > 0.6
        
        return is_shooter, confidence

    def calculate_joint_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)

    def calculate_shoulder_rotation(self, keypoints):
        """Calculate shoulder rotation angle"""
        left_shoulder = keypoints.get('left_shoulder', None)
        right_shoulder = keypoints.get('right_shoulder', None)
        
        if not (left_shoulder and right_shoulder):
            return 0.0
            
        return np.degrees(math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ))

    def calculate_hip_rotation(self, keypoints):
        """Calculate hip rotation angle"""
        left_hip = keypoints.get('left_hip', None)
        right_hip = keypoints.get('right_hip', None)
        
        if not (left_hip and right_hip):
            return 0.0
            
        return np.degrees(math.atan2(
            right_hip[1] - left_hip[1],
            right_hip[0] - left_hip[0]
        ))

    def calculate_weight_transfer(self, keypoints):
        """Calculate weight transfer percentage to right side"""
        left_ankle = keypoints.get('left_ankle', None)
        right_ankle = keypoints.get('right_ankle', None)
        hip_center = keypoints.get('hip_center', None)
        
        if not all([left_ankle, right_ankle, hip_center]):
            return 50.0
        
        # Calculate horizontal distance from hip center to each ankle
        left_dist = abs(hip_center[0] - left_ankle[0])
        right_dist = abs(hip_center[0] - right_ankle[0])
        total_dist = left_dist + right_dist
        
        # Return percentage of weight on right leg
        return (right_dist / total_dist * 100) if total_dist > 0 else 50.0

    def draw_debug_info(self, frame, keypoints, is_shooting, shot_phase, 
                       shot_conf, wrist_velocity, elbow_angle):
        """Draw debug information for shot detection"""
        if not self.debug_mode or not keypoints:
            return frame
            
        height, width = frame.shape[:2]
        debug_x = width - 300  # Position debug info 300px from right edge
        
        # Draw debug values
        debug_lines = [
            f"Shot Phase: {shot_phase}",
            f"Confidence: {shot_conf:.2f}",
            f"Wrist Vel: {wrist_velocity:.1f}",
            f"Elbow Ang: {elbow_angle:.1f}"
        ]
        
        y_pos = 120  # Start below frame counter
        for line in debug_lines:
            cv2.putText(
                frame,
                line,
                (debug_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),  # Yellow color
                2
            )
            y_pos += 30
            
        # Draw keypoints and connections if available
        if keypoints:
            # Draw key points
            for name, point in keypoints.items():
                if point:
                    cv2.circle(
                        frame,
                        (int(point[0]), int(point[1])),
                        4,
                        (0, 255, 255),  # Yellow
                        -1
                    )
            
            # Draw connections for shooting arm
            connections = [
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist')
            ]
            
            for start_point, end_point in connections:
                if (start_point in keypoints and end_point in keypoints):
                    p1 = keypoints[start_point]
                    p2 = keypoints[end_point]
                    if p1 and p2:
                        color = (0, 255, 0) if is_shooting else (0, 0, 255)
                        cv2.line(
                            frame,
                            (int(p1[0]), int(p1[1])),
                            (int(p2[0]), int(p2[1])),
                            color,
                            2
                        )
        
        return frame

    def process_video(self, video_path, output_path=None):
        """
        Process video with pose detection
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path for output video. If None, will use input path with '_pose' suffix
        """
        # Convert to absolute path and check if file exists
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")
        
        # Handle output path
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / 
                            f"{video_path_obj.stem}_pose{video_path_obj.suffix}")
        output_path = os.path.abspath(output_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the input video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Output will be saved to: {output_path}")
        print(f"Video properties: {width}x{height} @ {fps}fps")
        print(f"Total frames to process: {total_frames}")
        
        # Get video FPS for timing calculations
        self.frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.total_frames += 1
                self.clip_frame += 1
                
                # Add frame counter to top-left corner
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Add shot counter to top-left corner
                cv2.putText(
                    frame,
                    f"Shots: {self.total_shots}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Process frame with YOLO
                results = self.model(frame, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    annotated_frame = result.plot()
                    
                    # Copy frame counters to annotated frame
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_count}/{total_frames}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Shots: {self.total_shots}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    
                    if hasattr(result, 'boxes') and len(result.boxes) > 0:
                        # Process detections in order of confidence
                        boxes_data = []
                        for i, box in enumerate(result.boxes):
                            conf = float(box.conf[0].cpu().numpy())
                            boxes_data.append((i, box, conf))
                        
                        boxes_data.sort(key=lambda x: x[2], reverse=True)
                        
                        # Process each detection
                        for orig_idx, box, conf in boxes_data:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            
                            # Get stable player ID
                            player_id = self.get_player_id(
                                center, 
                                (x1, y1, x2, y2),
                                prev_id=orig_idx
                            )
                            
                            # Extract keypoints for this detection
                            if (hasattr(result, 'keypoints') and 
                                len(result.keypoints) > orig_idx):
                                keypoints = self.extract_keypoints(
                                    result.keypoints[orig_idx]
                                )
                                
                                # Calculate shot detection metrics
                                is_shooting, shot_phase, shot_conf = \
                                    self.detect_shooting_motion(keypoints)
                                is_shooter, shooter_conf = \
                                    self.identify_shooter(keypoints)
                                
                                # Get additional metrics for debug display
                                wrist_velocity = 0
                                elbow_angle = 0
                                if all(k in keypoints for k in ['right_shoulder', 
                                                              'right_elbow', 
                                                              'right_wrist']):
                                    rs = keypoints['right_shoulder']
                                    re = keypoints['right_elbow']
                                    rw = keypoints['right_wrist']
                                    elbow_angle = self.calculate_joint_angle(rs, re, rw)
                                    
                                    if hasattr(self, 'prev_wrist_pos'):
                                        dx = rw[0] - self.prev_wrist_pos[0]
                                        dy = rw[1] - self.prev_wrist_pos[1]
                                        wrist_velocity = math.sqrt(dx**2 + dy**2) / self.frame_time
                                
                                # Draw debug information
                                annotated_frame = self.draw_debug_info(
                                    annotated_frame,
                                    keypoints,
                                    is_shooting,
                                    shot_phase,
                                    shot_conf,
                                    wrist_velocity,
                                    elbow_angle
                                )
                                
                                # Draw labels
                                label_lines = [f"Player {player_id}"]
                                if is_shooter and shooter_conf > 0.6:
                                    label_lines.append("SHOOTER")
                                if is_shooting and shot_conf > 0.7:
                                    label_lines.append(f"Shot: {shot_phase}")
                                    
                                    # If this is a release, add it to shot history
                                    if shot_phase == 'release':
                                        if player_id not in self.shot_states or \
                                           self.total_frames - self.shot_states[player_id]['frame'] > 30:
                                            self.total_shots += 1
                                            self.shot_states[player_id] = {
                                                'frame': self.total_frames,
                                                'confidence': shot_conf,
                                                'position': center
                                            }
                                
                                # Draw multi-line label
                                y_offset = y1 - 10
                                for line in label_lines:
                                    cv2.putText(
                                        annotated_frame,
                                        line,
                                        (x1, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,
                                        (0, 255, 0),
                                        2
                                    )
                                    y_offset -= 25
                    
                    writer.write(annotated_frame)
                else:
                    writer.write(frame)
                
                # Update progress
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            print("\nProcessing Statistics:")
            print(f"Total Frames Processed: {frame_count}")
            print(f"Total Shots Detected: {self.total_shots}")
            print(f"Output saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            raise
        
        finally:
            # Release resources
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
        
        return output_path
    
    def sort_detections(self, results):
        """
        Sort detections to maintain consistency with the biomechanical script
        
        Args:
            results: YOLO detection results
            
        Returns:
            Sorted boxes based on position (left to right, top to bottom)
        """
        # This method is kept for future improvements if needed
        # Currently we're using detection indices directly
        pass

    def extract_keypoints(self, pose_keypoints):
        """
        Extract keypoints from YOLO pose detection result
        
        Args:
            pose_keypoints: Keypoints from YOLO detection
            
        Returns:
            dict: Dictionary of keypoint coordinates
        """
        keypoints = {}
        
        # Map YOLO keypoint indices to names
        keypoint_map = {
            0: 'nose',
            5: 'left_shoulder',
            6: 'right_shoulder',
            7: 'left_elbow',
            8: 'right_elbow',
            9: 'left_wrist',
            10: 'right_wrist',
            11: 'left_hip',
            12: 'right_hip',
            13: 'left_knee',
            14: 'right_knee',
            15: 'left_ankle',
            16: 'right_ankle'
        }
        
        # Extract keypoints
        for idx, name in keypoint_map.items():
            if idx < len(pose_keypoints):
                kp = pose_keypoints[idx]
                if kp is not None and len(kp) >= 2:
                    keypoints[name] = (float(kp[0]), float(kp[1]))
        
        # Calculate hip center if both hips are detected
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            left_hip = keypoints['left_hip']
            right_hip = keypoints['right_hip']
            hip_center = (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2
            )
            keypoints['hip_center'] = hip_center
        
        return keypoints

def main():
    # Get current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Use command line arguments if provided
    if len(sys.argv) > 1:
        # Get the video path from command line
        video_path = sys.argv[1]
        print(f"Command line argument for video: {video_path}")
        
        # Check if input is a relative path and resolve it
        if not os.path.isabs(video_path):
            video_path = os.path.join(cwd, video_path)
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"ERROR: Input file does not exist: {video_path}")
            sys.exit(1)
            
        # Set output path if provided, otherwise use default
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
            if not os.path.isabs(output_path):
                output_path = os.path.join(cwd, output_path)
        else:
            # Create default output path in same directory as input
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / 
                            f"{video_path_obj.stem}_pose{video_path_obj.suffix}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        print("No input video provided. Please specify a video file.")
        print("Usage: python pose_detection5.py input_video.mp4 [output_video.mp4]")
        sys.exit(1)
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize detector and process video
    detector = PoseDetector()
    try:
        output_file = detector.process_video(video_path, output_path)
        print(f"\nProcessing complete! Output saved to: {output_file}")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()