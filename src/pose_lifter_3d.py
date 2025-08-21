"""
3D Pose Lifting Module for Hockey Player Analysis
Converts 2D MediaPipe keypoints to 3D coordinates using temporal consistency
and biomechanical constraints for more accurate skeleton visualization.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import math


class PoseLifter3D:
    def __init__(self):
        """Initialize the 3D pose lifter with biomechanical constraints."""
        # Standard human proportions (approximate)
        self.HEAD_HEIGHT_RATIO = 0.13  # Head is ~13% of total height
        self.TORSO_HEIGHT_RATIO = 0.33  # Torso is ~33% of total height
        self.LEG_HEIGHT_RATIO = 0.54   # Legs are ~54% of total height
        
        # Biomechanical constraints
        self.MAX_KNEE_ANGLE = 160  # degrees
        self.MIN_KNEE_ANGLE = 60   # degrees
        self.MAX_ELBOW_ANGLE = 150  # degrees
        self.MIN_ELBOW_ANGLE = 30   # degrees
        
        # Temporal smoothing parameters
        self.SMOOTHING_WINDOW = 5
        self.CONFIDENCE_THRESHOLD = 0.3
        
        # Store pose history for temporal consistency
        self.pose_history = {}
        
    def lift_2d_to_3d(self, 
                       keypoints_2d: List[Dict], 
                       player_id: str,
                       frame_timestamp: float) -> Dict:
        """
        Convert 2D MediaPipe keypoints to 3D coordinates.
        
        Args:
            keypoints_2d: List of 2D keypoints from MediaPipe
            player_id: Unique identifier for the player
            frame_timestamp: Current frame timestamp
            
        Returns:
            Dictionary with 3D keypoints and confidence scores
        """
        if not keypoints_2d:
            return {}
            
        # Initialize player history if needed
        if player_id not in self.pose_history:
            self.pose_history[player_id] = []
            
        # Convert MediaPipe format to our format
        keypoints = self._convert_mediapipe_format(keypoints_2d)
        
        # Apply temporal smoothing
        smoothed_keypoints = self._apply_temporal_smoothing(keypoints, player_id)
        
        # Estimate depth using biomechanical constraints
        keypoints_3d = self._estimate_depth_biomechanical(smoothed_keypoints)
        
        # Apply biomechanical constraints
        keypoints_3d = self._apply_biomechanical_constraints(keypoints_3d)
        
        # Store in history
        self.pose_history[player_id].append({
            'timestamp': frame_timestamp,
            'keypoints': keypoints_3d
        })
        
        # Keep only recent history
        if len(self.pose_history[player_id]) > self.SMOOTHING_WINDOW:
            self.pose_history[player_id].pop(0)
            
        return keypoints_3d
    
    def _convert_mediapipe_format(self, mediapipe_keypoints: List[Dict]) -> List[Dict]:
        """Convert MediaPipe keypoint format to our internal format."""
        keypoints = []
        
        # MediaPipe pose landmarks mapping
        landmark_mapping = {
            0: 'nose',
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        
        for landmark in mediapipe_keypoints:
            if landmark.landmark_id in landmark_mapping:
                keypoints.append({
                    'name': landmark_mapping[landmark.landmark_id],
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': 0.0,  # Will be estimated
                    'confidence': landmark.visibility,
                    'landmark_id': landmark.landmark_id
                })
                
        return keypoints
    
    def _apply_temporal_smoothing(self, keypoints: List[Dict], player_id: str) -> List[Dict]:
        """Apply temporal smoothing to reduce jitter."""
        if not self.pose_history[player_id]:
            return keypoints
            
        smoothed_keypoints = []
        
        for kp in keypoints:
            # Find corresponding keypoints in history
            history_positions = []
            history_confidences = []
            
            for hist_frame in self.pose_history[player_id]:
                hist_kp = next((h for h in hist_frame['keypoints'] 
                               if h['name'] == kp['name']), None)
                if hist_kp and hist_kp['confidence'] > self.CONFIDENCE_THRESHOLD:
                    history_positions.append([hist_kp['x'], hist_kp['y']])
                    history_confidences.append(hist_kp['confidence'])
            
            if history_positions:
                # Weighted average based on confidence
                history_positions = np.array(history_positions)
                history_confidences = np.array(history_confidences)
                
                # Add current position
                current_pos = np.array([[kp['x'], kp['y']]])
                current_conf = np.array([kp['confidence']])
                
                all_positions = np.vstack([history_positions, current_pos])
                all_confidences = np.hstack([history_confidences, current_conf])
                
                # Normalize confidences
                weights = all_confidences / np.sum(all_confidences)
                
                # Calculate weighted average
                smoothed_pos = np.average(all_positions, weights=weights, axis=0)
                
                kp['x'] = float(smoothed_pos[0])
                kp['y'] = float(smoothed_pos[1])
            
            smoothed_keypoints.append(kp)
            
        return smoothed_keypoints
    
    def _estimate_depth_biomechanical(self, keypoints: List[Dict]) -> List[Dict]:
        """Estimate depth using biomechanical constraints and player size."""
        if not keypoints:
            return keypoints
            
        # Estimate player height from keypoints
        player_height = self._estimate_player_height(keypoints)
        
        # Calculate depth based on keypoint relationships
        for kp in keypoints:
            if kp['confidence'] < self.CONFIDENCE_THRESHOLD:
                continue
                
            # Estimate depth based on keypoint type and position
            if 'nose' in kp['name']:
                # Head is typically closest to camera
                kp['z'] = 0.0
            elif 'shoulder' in kp['name']:
                # Shoulders are slightly behind head
                kp['z'] = -0.1 * player_height
            elif 'elbow' in kp['name']:
                # Elbows are further back
                kp['z'] = -0.2 * player_height
            elif 'wrist' in kp['name']:
                # Wrists can vary based on arm position
                kp['z'] = -0.15 * player_height
            elif 'hip' in kp['name']:
                # Hips are behind shoulders
                kp['z'] = -0.05 * player_height
            elif 'knee' in kp['name']:
                # Knees are behind hips
                kp['z'] = -0.1 * player_height
            elif 'ankle' in kp['name'] or 'heel' in kp['name']:
                # Feet are on the ground
                kp['z'] = -0.2 * player_height
            elif 'foot' in kp['name']:
                # Foot index is on the ground
                kp['z'] = -0.2 * player_height
                
        return keypoints
    
    def _estimate_player_height(self, keypoints: List[Dict]) -> float:
        """Estimate player height from keypoint positions."""
        # Find key body points
        nose = next((kp for kp in keypoints if 'nose' in kp['name']), None)
        left_ankle = next((kp for kp in keypoints if 'left_ankle' in kp['name']), None)
        right_ankle = next((kp for kp in keypoints if 'right_ankle' in kp['name']), None)
        
        if nose and (left_ankle or right_ankle):
            ankle = left_ankle if left_ankle else right_ankle
            
            # Calculate height in pixels
            height_pixels = abs(nose['y'] - ankle['y'])
            
            # Convert to meters (approximate)
            # Assuming average NHL player height of 1.85m
            height_meters = (height_pixels / 1000) * 1.85
            
            return max(height_meters, 1.5)  # Minimum reasonable height
            
        return 1.85  # Default NHL player height
    
    def _apply_biomechanical_constraints(self, keypoints: List[Dict]) -> List[Dict]:
        """Apply biomechanical constraints to ensure realistic poses."""
        # Check knee angles
        left_hip = next((kp for kp in keypoints if 'left_hip' in kp['name']), None)
        left_knee = next((kp for kp in keypoints if 'left_knee' in kp['name']), None)
        left_ankle = next((kp for kp in keypoints if 'left_ankle' in kp['name']), None)
        
        if left_hip and left_knee and left_ankle:
            knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
            if not (self.MIN_KNEE_ANGLE <= knee_angle <= self.MAX_KNEE_ANGLE):
                # Adjust knee position to maintain realistic angle
                left_knee = self._adjust_knee_position(left_hip, left_knee, left_ankle)
        
        # Check elbow angles
        left_shoulder = next((kp for kp in keypoints if 'left_shoulder' in kp['name']), None)
        left_elbow = next((kp for kp in keypoints if 'left_elbow' in kp['name']), None)
        left_wrist = next((kp for kp in keypoints if 'left_wrist' in kp['name']), None)
        
        if left_shoulder and left_elbow and left_wrist:
            elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            if not (self.MIN_ELBOW_ANGLE <= elbow_angle <= self.MAX_ELBOW_ANGLE):
                # Adjust elbow position to maintain realistic angle
                left_elbow = self._adjust_elbow_position(left_shoulder, left_elbow, left_wrist)
        
        return keypoints
    
    def _calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """Calculate angle between three points."""
        # Vector 1: point2 to point1
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        # Vector 2: point2 to point3
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle
    
    def _adjust_knee_position(self, hip: Dict, knee: Dict, ankle: Dict) -> Dict:
        """Adjust knee position to maintain realistic knee angle."""
        # This is a simplified adjustment - in practice, you'd want more sophisticated logic
        # For now, we'll just ensure the knee is roughly between hip and ankle
        knee['x'] = (hip['x'] + ankle['x']) / 2
        knee['y'] = (hip['y'] + ankle['y']) / 2
        return knee
    
    def _adjust_elbow_position(self, shoulder: Dict, elbow: Dict, wrist: Dict) -> Dict:
        """Adjust elbow position to maintain realistic elbow angle."""
        # Similar to knee adjustment
        elbow['x'] = (shoulder['x'] + wrist['x']) / 2
        elbow['y'] = (shoulder['y'] + wrist['y']) / 2
        return elbow
    
    def get_pose_quality_score(self, keypoints: List[Dict]) -> float:
        """Calculate overall pose quality score (0-1)."""
        if not keypoints:
            return 0.0
            
        # Calculate average confidence
        confidences = [kp['confidence'] for kp in keypoints if 'confidence' in kp]
        if not confidences:
            return 0.0
            
        avg_confidence = np.mean(confidences)
        
        # Check for missing key body parts
        required_parts = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        missing_parts = sum(1 for part in required_parts 
                           if not any(part in kp['name'] for kp in keypoints))
        
        completeness_score = 1.0 - (missing_parts / len(required_parts))
        
        # Overall quality score
        quality_score = (avg_confidence + completeness_score) / 2
        
        return quality_score
