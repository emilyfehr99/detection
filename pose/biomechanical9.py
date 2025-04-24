# Import necessary libraries
import cv2
import numpy as np
import os
import sys
import traceback
import argparse
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import torch
from ultralytics import YOLO
import math
import warnings
warnings.filterwarnings('ignore')


class HockeyAnalysis:
    def __init__(self, video_path):
        """
        Initialize the hockey analysis system
        
        Args:
            video_path (str): Path to the hockey video file
        """
        self.video_path = video_path
        
        # Get the appropriate device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Initialize pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize YOLO models
        print("Loading YOLO models...")
        # Initialize models without device parameter
        self.player_detector = YOLO('yolov8x.pt')
        self.player_detector.to(self.device)
        self.player_detector.verbose = False
        print(f"Player detection model loaded and running on: {self.device}")
        
        # Store metrics
        self.metrics_data = []
        
        # Player tracking system
        self.tracked_players = {}  # Dictionary to store player tracking info
        self.max_player_id = 12    # Maximum number of players to track
        self.last_seen = {}        # Frame number when player was last seen
        self.player_positions = {} # Last known positions of players
        self.frame_count = 0      # Counter for frame resets
        self.tracking_threshold = 30  # Number of frames to keep tracking a player
        self.reset_interval = 120  # Reset tracking every 4 seconds (30fps * 4)
    

    def process_video(self):
        """Process the specified video file"""
        print(f"Processing video: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create progress bars with reduced update frequency
        main_pbar = tqdm(
            total=frame_count,
            desc="Overall Progress",
            position=0,
            mininterval=1.0,  # Update at most once per second
            maxinterval=5.0   # Update at least every 5 seconds
        )
        
        frame_number = 0
        update_interval = max(1, frame_count // 100)  # Update every 1% of frames
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                metrics = self.analyze_frame(frame, frame_number)
                if metrics:
                    self.metrics_data.append(metrics)
                
                # Update progress less frequently
                if frame_number % update_interval == 0:
                    main_pbar.update(update_interval)
                
                frame_number += 1
                
                # Clear GPU/MPS cache periodically
                if frame_number % 30 == 0:  # Every 30 frames (about 1 second)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif (hasattr(torch.backends, 'mps') and 
                          torch.backends.mps.is_available()):
                        torch.mps.empty_cache()
                
        except Exception as e:
            print(f"\nError during video processing: {str(e)}")
            raise
        finally:
            # Clean up resources
            cap.release()
            main_pbar.close()
            
            # Final cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif (hasattr(torch.backends, 'mps') and 
                  torch.backends.mps.is_available()):
                torch.mps.empty_cache()
        
        print("\nVideo processing complete!")


    def analyze_frame(self, frame, frame_number):
        """Analyze a single frame"""
        metrics = {'frame': frame_number}
        
        try:
            # Reset tracking every 120 frames (4 seconds at 30fps)
            self.frame_count += 1
            if self.frame_count >= self.reset_interval:
                self.tracked_players.clear()
                self.last_seen.clear()
                self.player_positions.clear()
                self.frame_count = 0
            
            # Detect players with verbose=False to silence per-frame outputs
            with torch.no_grad():  # Reduce memory usage during inference
                player_results = self.player_detector(
                    frame, 
                    classes=[0],  # Only detect people
                    verbose=False
                )
            
            # Process detections in order of confidence
            boxes_data = []
            for i, player in enumerate(player_results[0].boxes.data):
                if player[4] > 0.5:  # Confidence threshold
                    boxes_data.append((i, player, float(player[4])))
            
            # Sort by confidence
            boxes_data.sort(key=lambda x: x[2], reverse=True)
            
            # Update tracking information
            current_frame_players = set()
            
            # Process each detected player
            for orig_idx, player, conf in boxes_data:
                x1, y1, x2, y2 = map(int, player[:4])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                current_pos = (center_x, center_y)
                
                # Find closest tracked player
                min_dist = float('inf')
                best_match_id = None
                
                for player_id, last_pos in self.player_positions.items():
                    if player_id not in current_frame_players and \
                       frame_number - self.last_seen.get(player_id, 0) <= self.tracking_threshold:
                        dist = ((current_pos[0] - last_pos[0])**2 + 
                               (current_pos[1] - last_pos[1])**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_match_id = player_id
                
                # If no match found or too far, assign new ID
                if best_match_id is None or min_dist > 100:  # Distance threshold
                    for i in range(self.max_player_id):
                        if i not in self.tracked_players:
                            best_match_id = i
                            break
                    else:
                        continue  # Skip if no available IDs
                
                # Update tracking information
                self.tracked_players[best_match_id] = True
                self.last_seen[best_match_id] = frame_number
                self.player_positions[best_match_id] = current_pos
                current_frame_players.add(best_match_id)
                
                # Extract player metrics
                player_frame = frame[y1:y2, x1:x2].copy()
                if player_frame.size == 0:
                    continue
                
                player_metrics = self.analyze_player(
                    player_frame=player_frame,
                    full_frame=frame,
                    bbox=(x1, y1, x2, y2),
                    frame_number=frame_number,
                    player_id=best_match_id
                )
                
                # Add player-specific metrics with ID
                for key, value in player_metrics.items():
                    metrics[f'player_{best_match_id}_{key}'] = value
                
                # Clear individual player frame
                del player_frame
            
            # Remove players not seen recently
            current_frame = frame_number
            to_remove = []
            for player_id in self.tracked_players:
                if current_frame - self.last_seen.get(player_id, 0) > self.tracking_threshold:
                    to_remove.append(player_id)
            
            for player_id in to_remove:
                self.tracked_players.pop(player_id, None)
                self.last_seen.pop(player_id, None)
                self.player_positions.pop(player_id, None)
            
            return metrics
            
        except Exception as e:
            print(f"\nError processing frame {frame_number}: {str(e)}")
            return None
        finally:
            # Clear any temporary tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif (hasattr(torch.backends, 'mps') and 
                  torch.backends.mps.is_available()):
                torch.mps.empty_cache()
    

    def analyze_player(self, player_frame, full_frame, bbox, frame_number, player_id):
        """
        Analyze individual player metrics
        
        Args:
            player_frame: Cropped frame containing the player
            full_frame: Complete video frame
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            frame_number: Current frame number
            player_id: Stable player ID for tracking
            
        Returns:
            dict: Dictionary of player metrics
        """
        metrics = {}
        
        # Pose detection
        pose_results = self.analyze_pose(player_frame)
        if not pose_results.pose_landmarks:
            return metrics
            
        landmarks = pose_results.pose_landmarks.landmark
        
        # Calculate hip center
        hip_center = self.get_hip_center(landmarks)
        
        # Save keypoint coordinates (excluding eyes and ears)
        keypoints_to_save = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_foot_index': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            'right_foot_index': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        }
        
        # Add keypoint coordinates to metrics
        for name, landmark_type in keypoints_to_save.items():
            landmark = landmarks[landmark_type.value]
            metrics[f'{name}_x'] = landmark.x
            metrics[f'{name}_y'] = landmark.y
            metrics[f'{name}_z'] = landmark.z
            metrics[f'{name}_visibility'] = landmark.visibility
        
        # Analyze hand differences
        hand_metrics = self.analyze_hand_differences(landmarks)
        metrics.update(hand_metrics)
        
        # Calculate stance width (distance between ankles)
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        metrics['stance_width'] = self.calculate_distance(left_ankle, right_ankle)
        
        # 1. Lower Body Metrics
        # Knee angles
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        metrics['left_knee_angle'] = self.calculate_joint_angle(
            left_hip, left_knee, left_ankle
        )
        
        metrics['right_knee_angle'] = self.calculate_joint_angle(
            right_hip, right_knee, right_ankle
        )
        
        # Hip angles
        metrics['left_hip_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        
        metrics['right_hip_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        )
        
        # Ankle angles
        metrics['left_ankle_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        )
        
        metrics['right_ankle_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        )
        
        # 2. Upper Body Metrics
        # Shoulder angles
        metrics['left_shoulder_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        
        metrics['right_shoulder_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        # Elbow angles
        metrics['left_elbow_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        metrics['right_elbow_angle'] = self.calculate_joint_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        
        # 3. Posture Metrics
        # Upper body lean (angle from vertical)
        metrics['spine_angle'] = self.calculate_vertical_angle(
            landmarks[self.mp_pose.PoseLandmark.NOSE.value],
            hip_center
        )
        
        return metrics
    
    def analyze_hand_differences(self, landmarks):
        """
        Analyze differences between left and right hands
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            dict: Dictionary containing hand difference metrics
        """
        metrics = {}
        
        # Get hand landmarks
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calculate distance between hands
        metrics['hand_distance'] = self.calculate_distance(left_wrist, right_wrist)
        
        # Calculate height difference (positive means right hand is higher)
        metrics['hand_height_diff'] = right_wrist.y - left_wrist.y
        
        # Calculate horizontal separation (positive means right hand is to the right)
        metrics['hand_horizontal_sep'] = right_wrist.x - left_wrist.x
        
        return metrics
    
    def get_hip_center(self, landmarks):
        """Calculate the center point between left and right hip"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        return type('Point', (), {
            'x': (left_hip.x + right_hip.x) / 2,
            'y': (left_hip.y + right_hip.y) / 2,
            'z': (left_hip.z + right_hip.z) / 2,
            'visibility': min(left_hip.visibility, right_hip.visibility)
        })
    
    def calculate_joint_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def calculate_distance(self, p1, p2):
        """Calculate distance between two points"""
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    
    def calculate_vertical_angle(self, top_point, bottom_point):
        """Calculate angle from vertical"""
        dx = top_point.x - bottom_point.x
        dy = top_point.y - bottom_point.y
        angle = np.arctan2(dx, dy)
        return np.degrees(angle)
    
    def analyze_pose(self, frame):
        """Detect and analyze pose in a frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(frame_rgb)
    
    def save_metrics(self, output_dir):
        """
        Save the collected metrics to CSV files
        
        Args:
            output_dir (str): Directory to save the metrics files
        """
        if not self.metrics_data:
            print("No metrics to save!")
            return
        
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics_data)
        
        # Save detailed metrics (including keypoints)
        detailed_output = os.path.join(output_dir, 'detailed_metrics.csv')
        df.to_csv(detailed_output, index=False)
        print(f"\nDetailed metrics saved to: {detailed_output}")
        
        # Create simplified metrics DataFrame with only specific measurements
        simplified_metrics = []
        for frame_data in self.metrics_data:
            frame_metrics = {'frame': frame_data['frame']}
            
            # For each player in the frame
            for player_id in range(self.max_player_id):
                prefix = f'player_{player_id}_'
                
                # Check if this player exists in the frame
                if any(key.startswith(prefix) for key in frame_data.keys()):
                    # Extract only the specific metrics we want
                    metrics_to_keep = [
                        'hand_distance',
                        'left_knee_angle',
                        'right_knee_angle',
                        'left_ankle_angle',
                        'right_ankle_angle',
                        'left_elbow_angle',
                        'right_elbow_angle',
                        'left_hip_angle',
                        'right_hip_angle',
                        'left_shoulder_angle',
                        'right_shoulder_angle',
                        'spine_angle',
                        'stance_width'
                    ]
                    
                    for metric in metrics_to_keep:
                        key = f'{prefix}{metric}'
                        if key in frame_data:
                            frame_metrics[key] = frame_data[key]
            
            simplified_metrics.append(frame_metrics)
        
        # Save simplified metrics
        simplified_df = pd.DataFrame(simplified_metrics)
        simplified_output = os.path.join(output_dir, 'simplified_metrics.csv')
        simplified_df.to_csv(simplified_output, index=False)
        print(f"Simplified metrics saved to: {simplified_output}")

def validate_video_path(video_path):
    """Validate that the video file exists and is a supported format"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    extension = os.path.splitext(video_path)[1].lower()
    supported_formats = ['.mov', '.mp4', '.avi']
    if extension not in supported_formats:
        raise ValueError(
            f"Unsupported video format: {extension}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
    
    return True

def setup_output_directory(video_path):
    """Set up output directory for analysis results"""
    # Create an output directory based on the video file name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(video_path)),
        f'{video_name}_analysis'
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Hockey Biomechanical Analysis'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the hockey video file'
    )
    return parser.parse_args()

def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        video_path = args.video_path
        
        # Validate video path
        validate_video_path(video_path)
        
        # Set up output directory
        output_dir = setup_output_directory(video_path)
        
        # Initialize and run analysis
        print("\nInitializing Hockey Analysis System...")
        analyzer = HockeyAnalysis(video_path)
        
        print("\nBeginning video processing...")
        analyzer.process_video()
        
        # Save results
        metrics_path, summary_path = (
            analyzer.save_metrics(output_dir)
        )
        
        print("\nAnalysis Complete!")
        print(f"Results saved to: {output_dir}")
        print("\nMetrics saved:")
        print(f"- Detailed metrics: {metrics_path}")
        print(f"- Summary statistics: {summary_path}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please check that the video file path is correct.")
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        print("\nProgram finished.")

if __name__ == "__main__":
    try:
        # Set up basic configurations
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Configure numpy settings
        np.set_printoptions(precision=3, suppress=True)
        
        # Configure pandas display settings
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        
        # Hardware acceleration setup
        import platform
        import multiprocessing
        
        # Get number of available CPU cores
        cpu_count = multiprocessing.cpu_count()
        recommended_threads = max(1, cpu_count - 1)  # Leave one core free for system
        
        # Check for hardware acceleration options
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU acceleration.")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            device = torch.device('cuda')
            cv2.setNumThreads(1)  # Minimal CPU threads when using GPU
        elif platform.processor() == 'arm' and platform.system() == 'Darwin':
            # Check for Apple Silicon (M-series)
            try:
                import torch.backends.mps as mps
                if mps.is_available():
                    print("Apple Metal acceleration is available. Using MPS device.")
                    device = torch.device('mps')
                    cv2.setNumThreads(recommended_threads)
                else:
                    print("Metal not available. Falling back to CPU optimization.")
                    device = torch.device('cpu')
                    cv2.setNumThreads(recommended_threads)
            except ImportError:
                print("Metal support not installed. Falling back to CPU optimization.")
                device = torch.device('cpu')
                cv2.setNumThreads(recommended_threads)
        else:
            print(f"Using CPU optimization with {recommended_threads} threads.")
            device = torch.device('cpu')
            cv2.setNumThreads(recommended_threads)
            
        # Additional CPU optimizations when not using CUDA
        if device.type == 'cpu' or device.type == 'mps':
            # Set OpenCV optimization flags
            cv2.ocl.setUseOpenCL(True)  # Enable OpenCL acceleration if available
            
            # Set numpy multithreading
            os.environ["OMP_NUM_THREADS"] = str(recommended_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(recommended_threads)
            os.environ["MKL_NUM_THREADS"] = str(recommended_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(recommended_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(recommended_threads)
        
        # Print system information
        print(f"\nSystem Information:")
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Processor: {platform.processor()}")
        print(f"Total CPU cores: {cpu_count}")
        print(f"Using device: {device}")
        print(f"Active threads: {recommended_threads if device.type != 'cuda' else 1}")
        print("\n" + "="*50 + "\n")
            
        # Run the main program
        main()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("\nStack trace:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up
        cv2.destroyAllWindows()
        
        # Clear acceleration cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Clear MPS cache if available
            torch.mps.empty_cache()
            
        print("\nCleanup complete. Program exited.")