#!/usr/bin/env python3
import cv2
import os
import argparse
from typing import List

def extract_frames(video_path: str, frame_times: List[float], output_dir: str):
    """
    Extract frames at specified times from a video.
    
    Args:
        video_path: Path to the video file
        frame_times: List of times (in seconds) to extract frames
        output_dir: Directory to save extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Extract each frame
    for i, time_sec in enumerate(frame_times):
        # Convert time to frame number
        frame_pos = int(time_sec * 1000)  # in milliseconds
        video.set(cv2.CAP_PROP_POS_MSEC, frame_pos)
        
        # Read the frame
        ret, frame = video.read()
        if not ret:
            print(f"Error: Could not read frame at time {time_sec} seconds")
            continue
        
        # Save the frame
        output_path = os.path.join(output_dir, f"frame_{time_sec:.1f}sec.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved frame at {time_sec} seconds to {output_path}")
    
    # Release the video
    video.release()
    print("Frame extraction complete!")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from a video at specified times')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--output-dir', required=True, help='Directory to save frames')
    parser.add_argument('--times', type=float, nargs='+', default=[10.0, 15.0, 20.0, 25.0, 30.0],
                        help='List of times (in seconds) to extract frames')
    
    args = parser.parse_args()
    extract_frames(args.video, args.times, args.output_dir)

if __name__ == "__main__":
    main()
