#!/usr/bin/env python3
"""
Real-time Hockey Analysis Processor
Processes video at full speed with ellipse-based detection visualization.
"""

import cv2
import numpy as np
import os
import argparse
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from player_tracker import PlayerTracker, NumpyEncoder


class RealTimeProcessor:
    """Real-time hockey video processor with ellipse visualization."""
    
    def __init__(self, 
                 detection_model_path: str,
                 orientation_model_path: str,
                 output_dir: str,
                 segmentation_model_path: Optional[str] = None,
                 rink_coordinates_path: Optional[str] = None,
                 rink_image_path: Optional[str] = None):
        """Initialize the real-time processor."""
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize player tracker (using Roboflow API but optimized)
        self.tracker = PlayerTracker(
            orientation_model_path=orientation_model_path,
            segmentation_model_path=segmentation_model_path,
            rink_coordinates_path=rink_coordinates_path,
            output_dir=output_dir
        )
        
        # Define classes to detect and their colors
        self.target_classes = {
            'player': (0, 255, 0),      # Green
            'puck': (0, 255, 255),      # Yellow (BGR format)
            'stick_blade': (255, 0, 255),  # Magenta
            'goalkeeper': (0, 0, 255),   # Red
            'GoalZone': (255, 165, 0)    # Orange
        }
        
        # Processing stats
        self.frame_count = 0
        self.start_time = None
        
    def draw_ellipse(self, frame: np.ndarray, detection: Dict, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw an ellipse around a detection."""
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Calculate ellipse parameters
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Make ellipse slightly larger than bounding box
        axes = (int(width * 0.6), int(height * 0.6))
        
        # Draw filled ellipse with transparency
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, center_y), axes, 0, 0, 360, color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw ellipse outline
        cv2.ellipse(frame, (center_x, center_y), axes, 0, 0, 360, color, 2)
        
        # Add label
        label = detection.get('type', 'unknown').upper()
        if label == 'STICK_BLADE':
            label = 'STICK'
        elif label == 'GOALZONE':
            label = 'GOAL ZONE'
            
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x = center_x - label_size[0] // 2
        label_y = center_y - axes[1] - 10
        
        # Draw label background
        cv2.rectangle(frame, 
                     (label_x - 2, label_y - label_size[1] - 2),
                     (label_x + label_size[0] + 2, label_y + 2),
                     (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, frame_num: int) -> Tuple[np.ndarray, Dict]:
        """Process a single frame and return visualization."""
        
        # Process frame through tracker
        frame_info = self.tracker.process_frame(frame, frame_num)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Filter and draw only target classes
        if frame_info and 'players' in frame_info:
            for detection in frame_info['players']:
                detection_type = detection.get('type', '').lower()
                
                if detection_type in self.target_classes:
                    color = self.target_classes[detection_type]
                    vis_frame = self.draw_ellipse(vis_frame, detection, color)
        
        # Add frame info overlay
        self.add_frame_info(vis_frame, frame_num, frame_info)
        
        return vis_frame, frame_info
    
    def add_frame_info(self, frame: np.ndarray, frame_num: int, frame_info: Dict):
        """Add frame information overlay."""
        height, width = frame.shape[:2]
        
        # Calculate FPS
        if self.start_time is None:
            self.start_time = time.time()
            fps = 0
        else:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Frame info text
        info_lines = [
            f"Frame: {frame_num}",
            f"FPS: {fps:.1f}",
            f"Detections: {len(frame_info.get('players', [])) if frame_info else 0}"
        ]
        
        # Draw info background
        info_height = len(info_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (200, info_height), (0, 0, 0), -1)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            y = 30 + i * 25
            cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw legend
        self.draw_legend(frame)
    
    def draw_legend(self, frame: np.ndarray):
        """Draw color legend for different classes."""
        height, width = frame.shape[:2]
        legend_x = width - 200
        legend_y = 20
        
        # Legend background
        legend_height = len(self.target_classes) * 25 + 20
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10), 
                     (width - 10, legend_y + legend_height), (0, 0, 0), -1)
        
        # Legend title
        cv2.putText(frame, "CLASSES:", (legend_x, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Legend items
        for i, (class_name, color) in enumerate(self.target_classes.items()):
            y = legend_y + 35 + i * 25
            
            # Draw color circle
            cv2.circle(frame, (legend_x + 10, y - 5), 8, color, -1)
            
            # Draw class name
            display_name = class_name.replace('_', ' ').upper()
            cv2.putText(frame, display_name, (legend_x + 25, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_video(self, video_path: str, start_second: int = 0, max_frames: Optional[int] = None):
        """Process video in real-time."""
        
        print(f"Processing video: {video_path}")
        print(f"Target classes: {list(self.target_classes.keys())}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
        
        # Seek to start position
        if start_second > 0:
            start_frame = int(start_second * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Setup output video
        output_path = os.path.join(self.output_dir, "realtime_analysis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        all_frame_data = []
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_num >= max_frames:
                    break
                
                # Process frame
                vis_frame, frame_info = self.process_frame(frame, frame_num)
                
                # Write to output video
                out.write(vis_frame)
                
                # Store frame data
                if frame_info:
                    all_frame_data.append(frame_info)
                
                # Update stats
                self.frame_count += 1
                frame_num += 1
                
                # Progress update
                if frame_num % 30 == 0:  # Every second at 30fps
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    fps_current = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_num} frames, Current FPS: {fps_current:.1f}")
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            
            # Save frame data
            if all_frame_data:
                data_path = os.path.join(self.output_dir, f"realtime_data_{int(time.time())}.json")
                with open(data_path, 'w') as f:
                    json.dump(all_frame_data, f, cls=NumpyEncoder, indent=2)
                print(f"Frame data saved to: {data_path}")
            
            print(f"Real-time analysis video saved to: {output_path}")
            
            # Final stats
            if self.start_time:
                total_time = time.time() - self.start_time
                avg_fps = self.frame_count / total_time
                print(f"Processing complete: {self.frame_count} frames in {total_time:.1f}s")
                print(f"Average processing FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Real-time Hockey Analysis")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--detection-model", type=str, required=True, help="Path to detection model")
    parser.add_argument("--orientation-model", type=str, required=True, help="Path to orientation model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--segmentation-model", type=str, help="Path to segmentation model")
    parser.add_argument("--rink-coordinates", type=str, help="Path to rink coordinates JSON")
    parser.add_argument("--rink-image", type=str, help="Path to rink image")
    parser.add_argument("--start-second", type=int, default=0, help="Start time in seconds")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Create processor
    processor = RealTimeProcessor(
        detection_model_path=args.detection_model,
        orientation_model_path=args.orientation_model,
        output_dir=args.output_dir,
        segmentation_model_path=args.segmentation_model,
        rink_coordinates_path=args.rink_coordinates,
        rink_image_path=args.rink_image
    )
    
    # Process video
    processor.process_video(
        video_path=args.video,
        start_second=args.start_second,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()