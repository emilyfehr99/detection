import cv2
import numpy as np
import os
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
import json

from player_tracker import PlayerTracker


def process_clip(
    video_path: str,
    segmentation_model_path: str,
    detection_model_path: str,
    orientation_model_path: str,
    rink_coordinates_path: str,
    rink_image_path: str,
    output_dir: str,
    start_second: float = 0.0,
    num_seconds: float = 5.0,
    frame_step: int = 5,
):
    """
    Process a short clip from a video to test the player tracking system.
    
    Args:
        video_path: Path to the input video
        segmentation_model_path: Path to the segmentation model
        detection_model_path: Path to the detection model
        orientation_model_path: Path to the orientation model
        rink_coordinates_path: Path to the rink coordinates JSON
        rink_image_path: Path to the rink image for visualization
        output_dir: Directory to save outputs
        start_second: Time in seconds to start processing from
        num_seconds: Number of seconds to process
        frame_step: Process every nth frame
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create frames directory
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # Initialize player tracker
    tracker = PlayerTracker(
        segmentation_model_path=segmentation_model_path,
        detection_model_path=detection_model_path,
        orientation_model_path=orientation_model_path,
        rink_coordinates_path=rink_coordinates_path,
        output_dir=output_dir
    )
    
    # Load rink image for visualization
    rink_image = cv2.imread(rink_image_path)
    if rink_image is None:
        print(f"Warning: Could not load rink image from {rink_image_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video at {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} total frames")
    
    # Calculate start and end frames
    start_frame = int(start_second * fps)
    num_frames = int(num_seconds * fps)
    end_frame = min(start_frame + num_frames, total_frames)
    
    print(f"Processing from frame {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    processed_frames_info = []
    frame_idx = start_frame
    frames_processed = 0
    start_time = time.time()
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        if (frame_idx - start_frame) % frame_step == 0:
            print(f"Processing frame {frame_idx}/{end_frame} ({(frame_idx - start_frame) / (end_frame - start_frame) * 100:.1f}%)")
            
            # Process the frame
            frame_data = tracker.process_frame(frame, frame_idx)
            
            # Create visualizations
            visualizations = tracker.visualize_frame(frame, frame_data, rink_image)
            
            # Save visualizations
            frame_info = {
                "frame_id": frame_idx,
                "timestamp": (frame_idx - start_frame) / fps,
                "num_players": len(frame_data["players"]),
                "homography_success": frame_data["homography_success"]
            }
            
            # Create directory for individual frame if it doesn't exist
            frame_dir = os.path.join(frames_dir, str(frame_idx))
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            
            # Save all visualizations
            for vis_name, vis_img in visualizations.items():
                if vis_img is not None:
                    vis_path = os.path.join(frame_dir, f"{vis_name}.jpg")
                    cv2.imwrite(vis_path, vis_img)
                    frame_info[f"{vis_name}_path"] = os.path.join("frames", str(frame_idx), f"{vis_name}.jpg")
            
            # Create and save side-by-side visualization
            if "broadcast" in visualizations and "overlay" in visualizations:
                # Create side-by-side image
                left_img = visualizations["broadcast"]
                right_img = visualizations["overlay"]
                
                # Resize images to same height
                h1, w1 = left_img.shape[:2]
                h2, w2 = right_img.shape[:2]
                target_height = min(h1, h2)
                
                aspect_ratio1 = w1 / h1
                aspect_ratio2 = w2 / h2
                
                new_w1 = int(target_height * aspect_ratio1)
                new_w2 = int(target_height * aspect_ratio2)
                
                left_img_resized = cv2.resize(left_img, (new_w1, target_height))
                right_img_resized = cv2.resize(right_img, (new_w2, target_height))
                
                # Combine images
                side_by_side = np.zeros((target_height, new_w1 + new_w2, 3), dtype=np.uint8)
                side_by_side[:, :new_w1] = left_img_resized
                side_by_side[:, new_w1:] = right_img_resized
                
                # Add frame information
                cv2.putText(side_by_side, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Save side-by-side image
                side_by_side_path = os.path.join(frame_dir, "side_by_side.jpg")
                cv2.imwrite(side_by_side_path, side_by_side)
                frame_info["side_by_side_path"] = os.path.join("frames", str(frame_idx), "side_by_side.jpg")
            
            processed_frames_info.append(frame_info)
            frames_processed += 1
        
        frame_idx += 1
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    fps_processed = frames_processed / processing_time
    
    print(f"Processed {frames_processed} frames in {processing_time:.2f} seconds ({fps_processed:.2f} fps)")
    
    # Save tracking data
    tracking_output = os.path.join(output_dir, "tracking_data.json")
    tracker.save_tracking_data(tracking_output)
    
    # Create simple HTML visualization
    create_html_visualization(processed_frames_info, output_dir, video_path)
    
    # Release resources
    cap.release()
    
    print(f"Processing complete. Outputs saved to {output_dir}")
    print(f"Open {os.path.join(output_dir, 'visualization.html')} to view results")


def create_html_visualization(frames_info: List[Dict], output_dir: str, video_path: str) -> None:
    """
    Create a simple HTML visualization of the processed frames.
    
    Args:
        frames_info: List of dictionaries containing frame information
        output_dir: Directory to save the HTML file
        video_path: Path to the original video
    """
    html_path = os.path.join(output_dir, "visualization.html")
    
    # Basic HTML template
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hockey Player Tracking Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .frame-container { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }
            .frame-header { background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; }
            .image-container { margin-top: 10px; }
            .viz-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
            img { max-width: 100%; border: 1px solid #ccc; }
            .stats { margin-top: 20px; padding: 10px; background-color: #f8f8f8; }
            button { padding: 10px; margin: 5px; cursor: pointer; }
            .active { background-color: #4CAF50; color: white; }
        </style>
        <script>
            function showVisualization(frameId, visType) {
                // Hide all visualizations for this frame
                var visContainers = document.querySelectorAll('#frame-' + frameId + ' .visualization');
                for (var i = 0; i < visContainers.length; i++) {
                    visContainers[i].style.display = 'none';
                }
                
                // Show the selected visualization
                var selectedVis = document.getElementById(frameId + '-' + visType);
                if (selectedVis) {
                    selectedVis.style.display = 'block';
                }
                
                // Update button states
                var buttons = document.querySelectorAll('#frame-' + frameId + ' button');
                for (var i = 0; i < buttons.length; i++) {
                    buttons[i].classList.remove('active');
                    if (buttons[i].getAttribute('data-vis') === visType) {
                        buttons[i].classList.add('active');
                    }
                }
            }
        </script>
    </head>
    <body>
        <h1>Hockey Player Tracking Visualization</h1>
        <div class="stats">
            <h2>Processing Statistics</h2>
            <p>Video: {video_path}</p>
            <p>Total frames processed: {total_frames}</p>
        </div>
    """
    
    # Add frames to the HTML
    for frame_info in frames_info:
        # Create buttons for different visualizations
        vis_types = []
        buttons_html = ""
        
        if "broadcast_path" in frame_info:
            vis_types.append(("broadcast", "Player Detection"))
        if "segmentation_path" in frame_info:
            vis_types.append(("segmentation", "Segmentation Lines"))
        if "warped_broadcast_path" in frame_info:
            vis_types.append(("warped_broadcast", "Warped Broadcast"))
        if "overlay_path" in frame_info:
            vis_types.append(("overlay", "Overlay on Rink"))
        if "rink_path" in frame_info:
            vis_types.append(("rink", "Players on Rink"))
        if "side_by_side_path" in frame_info:
            vis_types.append(("side_by_side", "Side by Side"))
        
        # Create buttons
        for vis_type, label in vis_types:
            active_class = "active" if vis_type == "side_by_side" else ""
            buttons_html += f'<button class="{active_class}" data-vis="{vis_type}" onclick="showVisualization({frame_info["frame_id"]}, \'{vis_type}\')">{label}</button>'
        
        # Create visualization containers
        vis_containers_html = ""
        for vis_type, _ in vis_types:
            display_style = "block" if vis_type == "side_by_side" else "none"
            if f"{vis_type}_path" in frame_info:
                vis_containers_html += f"""
                <div id="{frame_info['frame_id']}-{vis_type}" class="visualization" style="display: {display_style};">
                    <img src="{frame_info[f'{vis_type}_path']}" alt="{vis_type.capitalize()} Frame {frame_info['frame_id']}">
                </div>
                """
        
        frame_html = f"""
        <div class="frame-container" id="frame-{frame_info['frame_id']}">
            <div class="frame-header">
                <h2>Frame {frame_info['frame_id']} (Time: {frame_info['timestamp']:.2f}s)</h2>
                <p>Players detected: {frame_info['num_players']}</p>
                <p>Homography success: {frame_info['homography_success']}</p>
                <div class="viz-buttons">
                    {buttons_html}
                </div>
            </div>
            <div class="image-container">
                {vis_containers_html}
            </div>
        </div>
        """
        html_content += frame_html
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Format HTML with video path and total frames
    html_content = html_content.format(
        video_path=os.path.basename(video_path),
        total_frames=len(frames_info)
    )
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)


def main():
    """
    Main function to parse arguments and process clip.
    """
    parser = argparse.ArgumentParser(description="Process a short clip from hockey broadcast footage to test player tracking")
    
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--segmentation-model", type=str, required=True, help="Path to segmentation model")
    parser.add_argument("--detection-model", type=str, required=True, help="Path to detection model")
    parser.add_argument("--orientation-model", type=str, required=True, help="Path to orientation model")
    parser.add_argument("--rink-coordinates", type=str, required=True, help="Path to rink coordinates JSON")
    parser.add_argument("--rink-image", type=str, required=True, help="Path to rink image")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--start-second", type=float, default=0.0, help="Time in seconds to start processing from")
    parser.add_argument("--num-seconds", type=float, default=5.0, help="Number of seconds to process")
    parser.add_argument("--frame-step", type=int, default=5, help="Process every nth frame")
    
    args = parser.parse_args()
    
    process_clip(
        video_path=args.video,
        segmentation_model_path=args.segmentation_model,
        detection_model_path=args.detection_model,
        orientation_model_path=args.orientation_model,
        rink_coordinates_path=args.rink_coordinates,
        rink_image_path=args.rink_image,
        output_dir=args.output_dir,
        start_second=args.start_second,
        num_seconds=args.num_seconds,
        frame_step=args.frame_step
    )


if __name__ == "__main__":
    main()
