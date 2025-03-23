import cv2
import numpy as np
import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from player_tracker import PlayerTracker


def process_video(
    video_path: str,
    segmentation_model_path: str,
    detection_model_path: str,
    orientation_model_path: str,
    rink_coordinates_path: str,
    rink_image_path: str,
    output_dir: str,
    start_frame: int = 0,
    end_frame: int = None,
    frame_step: int = 1,
    visualize: bool = True,
    save_tracking_data: bool = True
) -> None:
    """
    Process a video file to track hockey players.
    
    Args:
        video_path: Path to the input video
        segmentation_model_path: Path to the segmentation model
        detection_model_path: Path to the detection model
        orientation_model_path: Path to the orientation model
        rink_coordinates_path: Path to the rink coordinates JSON
        rink_image_path: Path to the rink image for visualization
        output_dir: Directory to save outputs
        start_frame: Frame to start processing from (default: 0)
        end_frame: Frame to end processing at (default: None, process all frames)
        frame_step: Process every nth frame (default: 1)
        visualize: Whether to create visualizations (default: True)
        save_tracking_data: Whether to save tracking data (default: True)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video at {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Adjust end_frame if not specified
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    # Load rink image for visualization
    rink_image = None
    if visualize and rink_image_path:
        rink_image = cv2.imread(rink_image_path)
        if rink_image is None:
            print(f"Warning: Failed to load rink image at {rink_image_path}")
        else:
            # Resize rink image to standard dimensions (1400x600)
            rink_image = cv2.resize(rink_image, (1400, 600))
    
    # Initialize player tracker
    tracker = PlayerTracker(
        segmentation_model_path=segmentation_model_path,
        detection_model_path=detection_model_path,
        orientation_model_path=orientation_model_path,
        rink_coordinates_path=rink_coordinates_path,
        output_dir=output_dir
    )
    
    # Initialize video writers if visualizing
    broadcast_writer = None
    rink_writer = None
    side_by_side_writer = None
    
    if visualize:
        # Create video writers for visualizations
        broadcast_output = os.path.join(output_dir, "broadcast_visualization.mp4")
        rink_output = os.path.join(output_dir, "rink_visualization.mp4")
        side_by_side_output = os.path.join(output_dir, "side_by_side_visualization.mp4")
        
        broadcast_writer = cv2.VideoWriter(
            broadcast_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps / frame_step,
            (width, height)
        )
        
        if rink_image is not None:
            rink_writer = cv2.VideoWriter(
                rink_output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / frame_step,
                (rink_image.shape[1], rink_image.shape[0])
            )
            
            side_by_side_writer = cv2.VideoWriter(
                side_by_side_output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps / frame_step,
                (width + rink_image.shape[1], max(height, rink_image.shape[0]))
            )
    
    # Process frames
    frame_count = 0
    processed_count = 0
    
    # Set video to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    
    # Start timing
    start_time = time.time()
    
    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every frame_step frames
        if (frame_count - start_frame) % frame_step == 0:
            print(f"Processing frame {frame_count}/{end_frame} ({(frame_count - start_frame) / (end_frame - start_frame) * 100:.1f}%)")
            
            # Process frame
            frame_data = tracker.process_frame(frame, frame_count)
            processed_count += 1
            
            # Create visualizations if enabled
            if visualize:
                broadcast_vis, rink_vis = tracker.visualize_frame(frame, frame_data, rink_image)
                
                # Write broadcast visualization
                if broadcast_writer is not None:
                    broadcast_writer.write(broadcast_vis)
                
                # Write rink visualization if successful
                if rink_vis is not None and rink_writer is not None:
                    rink_writer.write(rink_vis)
                
                # Create and write side-by-side visualization
                if side_by_side_writer is not None and rink_vis is not None:
                    # Create a blank canvas for side-by-side visualization
                    side_by_side = np.zeros(
                        (max(height, rink_image.shape[0]), width + rink_image.shape[1], 3),
                        dtype=np.uint8
                    )
                    
                    # Add broadcast visualization
                    side_by_side[:height, :width] = broadcast_vis
                    
                    # Add rink visualization
                    side_by_side[:rink_image.shape[0], width:] = rink_vis
                    
                    # Write side-by-side visualization
                    side_by_side_writer.write(side_by_side)
                
                # Save individual frame visualizations
                if processed_count <= 10:  # Save first 10 frames as images for quick review
                    cv2.imwrite(os.path.join(output_dir, f"broadcast_frame_{frame_count}.jpg"), broadcast_vis)
                    if rink_vis is not None:
                        cv2.imwrite(os.path.join(output_dir, f"rink_frame_{frame_count}.jpg"), rink_vis)
                    if side_by_side_writer is not None and rink_vis is not None:
                        cv2.imwrite(os.path.join(output_dir, f"side_by_side_frame_{frame_count}.jpg"), side_by_side)
        
        frame_count += 1
    
    # Calculate processing time
    total_time = time.time() - start_time
    fps_processing = processed_count / total_time
    
    print(f"Processed {processed_count} frames in {total_time:.2f} seconds ({fps_processing:.2f} fps)")
    
    # Save tracking data if enabled
    if save_tracking_data:
        tracking_output = os.path.join(output_dir, "tracking_data.json")
        tracker.save_tracking_data(tracking_output)
    
    # Release resources
    cap.release()
    if broadcast_writer is not None:
        broadcast_writer.release()
    if rink_writer is not None:
        rink_writer.release()
    if side_by_side_writer is not None:
        side_by_side_writer.release()
    
    print(f"Processing complete. Outputs saved to {output_dir}")


def main():
    """
    Main function to parse arguments and process video.
    """
    parser = argparse.ArgumentParser(description="Process hockey broadcast footage to track players")
    
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--segmentation-model", type=str, required=True, help="Path to segmentation model")
    parser.add_argument("--detection-model", type=str, required=True, help="Path to detection model")
    parser.add_argument("--orientation-model", type=str, required=True, help="Path to orientation model")
    parser.add_argument("--rink-coordinates", type=str, required=True, help="Path to rink coordinates JSON")
    parser.add_argument("--rink-image", type=str, required=True, help="Path to rink image")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame to start processing from")
    parser.add_argument("--end-frame", type=int, default=None, help="Frame to end processing at")
    parser.add_argument("--frame-step", type=int, default=10, help="Process every nth frame")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", help="Disable visualization generation")
    parser.add_argument("--no-save", action="store_false", dest="save_tracking_data", help="Disable saving tracking data")
    
    args = parser.parse_args()
    
    process_video(
        video_path=args.video,
        segmentation_model_path=args.segmentation_model,
        detection_model_path=args.detection_model,
        orientation_model_path=args.orientation_model,
        rink_coordinates_path=args.rink_coordinates,
        rink_image_path=args.rink_image,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        visualize=args.visualize,
        save_tracking_data=args.save_tracking_data
    )


if __name__ == "__main__":
    main()
