import cv2
import numpy as np
import os
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
import json

from player_tracker import PlayerTracker, NumpyEncoder


def process_clip(
    video_path: str,
    detection_model_path: str,
    orientation_model_path: str,
    output_dir: str,
    segmentation_model_path: Optional[str] = None,
    rink_coordinates_path: Optional[str] = None,
    rink_image_path: Optional[str] = None,
    start_second: float = 0.0,
    num_seconds: float = 5.0,
    frame_step: int = 5,
    max_frames: int = 60,
):
    """
    Process a short clip from a video to test the player tracking system.
    
    Args:
        video_path: Path to the input video
        detection_model_path: Path to the detection model
        orientation_model_path: Path to the orientation model
        output_dir: Directory to save outputs
        segmentation_model_path: Optional path to the segmentation model
        rink_coordinates_path: Optional path to the rink coordinates JSON
        rink_image_path: Optional path to the rink image for visualization
        start_second: Time in seconds to start processing from
        num_seconds: Number of seconds to process
        frame_step: Process every nth frame
        max_frames: Maximum number of frames to process
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create frames directory
    frames_dir = os.path.join(output_dir, "frames")
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # Initialize player tracker with optional parameters
    tracker = PlayerTracker(
        detection_model_path=detection_model_path,
        orientation_model_path=orientation_model_path,
        output_dir=output_dir,
        segmentation_model_path=segmentation_model_path,
        rink_coordinates_path=rink_coordinates_path
    )
    
    # Load rink image for visualization if provided
    rink_image = None
    if rink_image_path:
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
    
    # Use the max_frames parameter instead of hard-coded value
    max_frames_to_process = max_frames
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        if (frame_idx - start_frame) % frame_step == 0:
            print(f"Processing frame {frame_idx}/{end_frame} ({(frame_idx - start_frame) / (end_frame - start_frame) * 100:.1f}%)")
            
            # Process the frame
            frame_data = tracker.process_frame(frame, frame_idx)
            
            # Create visualizations if rink image is provided
            visualizations = {}
            if rink_image is not None:
                visualizations = tracker.visualize_frame(frame, frame_data, rink_image)
            
            # Save frame info
            frame_info = {
                "frame_id": frame_idx,
                "frame_idx": frame_idx,
                "timestamp": (frame_idx - start_frame) / fps,
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
            
            # Include information about whether homography was interpolated
            if frame_data.get("homography_interpolated", False):
                frame_info["homography_interpolated"] = True
            
            # Include information about homography source
            if "homography_source" in frame_data:
                frame_info["homography_source"] = frame_data["homography_source"]
            
            # Include detailed interpolation info if available
            if "interpolation_details" in frame_data:
                frame_info["interpolation_details"] = frame_data["interpolation_details"]
            
            # Only include homography matrix if successful
            if frame_data.get("homography_success", False):
                frame_info["homography_matrix"] = frame_data.get("homography_matrix", None)
            
            # Only include essential segmentation features
            if "segmentation_features" in frame_data:
                frame_info["segmentation_features"] = {
                    "features": {
                        k: v for k, v in frame_data["segmentation_features"].get("features", {}).items()
                        if k in ["blue_lines", "center_line", "goal_lines"]
                    }
                }
            
            # Create directory for individual frame if it doesn't exist
            frame_dir = os.path.join(frames_dir, str(frame_idx))
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            
            # Save original frame
            cv2.imwrite(os.path.join(frame_dir, "original.jpg"), frame)
            frame_info["original_frame_path"] = os.path.join("frames", str(frame_idx), "original.jpg")
            
            # Save all visualizations if available
            for vis_name, vis_img in visualizations.items():
                if vis_img is not None:
                    vis_path = os.path.join(frame_dir, f"{vis_name}.jpg")
                    cv2.imwrite(vis_path, vis_img)
                    frame_info[f"{vis_name}_path"] = os.path.join("frames", str(frame_idx), f"{vis_name}.jpg")
            
            processed_frames_info.append(frame_info)
            frames_processed += 1
            
            if frames_processed >= max_frames_to_process:
                break
        
        frame_idx += 1
    
    # Close video
    cap.release()
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"\nProcessing complete!")
    print(f"Processed {frames_processed} frames in {processing_time:.2f} seconds")
    print(f"Average frame rate: {frames_processed/processing_time:.2f} fps")
    
    # IMPROVED TWO-PASS INTERPOLATION:
    # Now that we have all the frames processed, do a second pass to interpolate missing homography matrices
    print("\nRunning two-pass homography interpolation...")
    tracker.interpolate_missing_homography(processed_frames_info)
    
    # Save tracking data
    tracking_data = {
        "frames": processed_frames_info,
        "processing_time": processing_time,
        "frames_processed": frames_processed,
        "fps": frames_processed/processing_time,
        "video_path": video_path,
        "detection_model": detection_model_path,
        "orientation_model": orientation_model_path,
        "segmentation_model": segmentation_model_path,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_step": frame_step
    }
    
    # Generate timestamp for the output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detection_data_path = os.path.join(
        output_dir, 
        f"player_detection_data_{timestamp}.json"
    )
    
    with open(detection_data_path, 'w') as f:
        json.dump(tracking_data, f, cls=NumpyEncoder, indent=2)
    
    print(f"\nPlayer detection data saved to {detection_data_path}")
    print(f"File size: {os.path.getsize(detection_data_path)} bytes")
    
    # Create HTML visualization if rink image is provided
    if rink_image is not None:
        create_html_visualization(processed_frames_info, output_dir, video_path)
        print(f"\nHTML visualization created at {os.path.join(output_dir, 'visualization.html')}")
    
    return processed_frames_info


def create_html_visualization(frames_info: List[Dict], output_dir: str, video_path: str) -> None:
    """
    Create a modern, interactive HTML visualization of the processed frames.
    
    Args:
        frames_info: List of dictionaries containing frame information
        output_dir: Directory to save the HTML file
        video_path: Path to the original video
    """
    html_path = os.path.join(output_dir, "visualization.html")
    
    # Modern HTML template with improved UI
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hockey Player Tracking Visualization</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary-color: #2563eb;
                --secondary-color: #3b82f6;
                --background-color: #ffffff;
                --text-color: #1f2937;
                --border-color: #e5e7eb;
                --card-background: #f9fafb;
                --hover-color: #f3f4f6;
            }

            [data-theme="dark"] {
                --primary-color: #3b82f6;
                --secondary-color: #60a5fa;
                --background-color: #111827;
                --text-color: #f9fafb;
                --border-color: #374151;
                --card-background: #1f2937;
                --hover-color: #374151;
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--background-color);
                color: var(--text-color);
                line-height: 1.5;
                transition: background-color 0.3s, color 0.3s;
            }

            .container {
                display: flex;
                min-height: 100vh;
                position: relative;
            }

            .sidebar {
                width: 400px;  /* Increased from 300px */
                min-width: 300px;
                max-width: 800px;
                padding: 1.5rem;
                background-color: var(--card-background);
                border-right: 1px solid var(--border-color);
                position: fixed;
                height: 100vh;
                overflow-y: auto;
                resize: horizontal;  /* Make sidebar resizable */
                z-index: 10;
            }

            .main-content {
                flex: 1;
                margin-left: 400px;  /* Match initial sidebar width */
                padding: 2rem;
                transition: margin-left 0.2s;
            }

            /* Add resize handle styles */
            .sidebar::after {
                content: "";
                position: absolute;
                right: 0;
                top: 0;
                bottom: 0;
                width: 4px;
                background-color: var(--border-color);
                cursor: col-resize;
            }

            .sidebar:hover::after {
                background-color: var(--primary-color);
            }

            /* Update frame slider container position */
            .frame-slider-container {
                position: fixed;
                top: 0;
                left: 400px;  /* Match initial sidebar width */
                right: 0;
                padding: 1rem;
                background-color: var(--card-background);
                border-bottom: 1px solid var(--border-color);
                z-index: 100;
                display: flex;
                align-items: center;
                gap: 1rem;
                transition: left 0.2s;
            }

            /* Make frames responsive */
            .visualization img {
                max-width: 100%;
                height: auto;
                border-radius: 0.375rem;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                cursor: zoom-in;  /* Add zoom cursor */
            }

            /* Add fullscreen mode for images */
            .visualization.fullscreen {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.9);
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: zoom-out;
            }

            .visualization.fullscreen img {
                max-height: 95vh;
                max-width: 95vw;
                object-fit: contain;
            }

            .header {
                margin-bottom: 2rem;
            }

            h1 {
                font-size: 1.875rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }

            .stats {
                background-color: var(--card-background);
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }

            .frame-container {
                background-color: var(--card-background);
                border-radius: 0.5rem;
                margin-bottom: 2rem;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }

            .frame-header {
                padding: 1rem;
                border-bottom: 1px solid var(--border-color);
            }

            .frame-title {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }

            .frame-controls {
                display: flex;
                gap: 0.5rem;
                margin-top: 1rem;
            }

            .viz-buttons {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 1rem;
            }

            button {
                padding: 0.5rem 1rem;
                border: 1px solid var(--border-color);
                border-radius: 0.375rem;
                background-color: var(--background-color);
                color: var(--text-color);
                cursor: pointer;
                font-size: 0.875rem;
                transition: all 0.2s;
            }

            button:hover {
                background-color: var(--hover-color);
            }

            button.active {
                background-color: var(--primary-color);
                color: white;
                border-color: var(--primary-color);
            }

            .visualization {
                padding: 1rem;
                display: none;
            }

            .visualization.active {
                display: block;
            }

            .progress-container {
                position: fixed;
                bottom: 0;
                left: 300px;
                right: 0;
                padding: 1rem;
                background-color: var(--card-background);
                border-top: 1px solid var(--border-color);
            }

            .progress-bar {
                width: 100%;
                height: 4px;
                background-color: var(--border-color);
                border-radius: 2px;
                overflow: hidden;
            }

            .progress {
                height: 100%;
                background-color: var(--primary-color);
                transition: width 0.3s;
            }

            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
            }

            .loading {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                display: none;
            }

            .loading.active {
                display: block;
            }

            .frame-slider {
                flex: 1;
                height: 4px;
                -webkit-appearance: none;
                background-color: var(--border-color);
                border-radius: 2px;
                outline: none;
            }

            .frame-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background-color: var(--primary-color);
                cursor: pointer;
                transition: all 0.2s;
            }

            .frame-slider::-webkit-slider-thumb:hover {
                transform: scale(1.2);
            }

            .frame-counter {
                font-size: 0.875rem;
                color: var(--text-color);
                min-width: 100px;
                text-align: right;
            }

            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                font-size: 0.875rem;
            }

            .metrics-table th,
            .metrics-table td {
                padding: 0.75rem;
                border: 1px solid var(--border-color);
                text-align: left;
            }

            .metrics-table th {
                background-color: var(--card-background);
                font-weight: 600;
            }

            .metrics-table tr:nth-child(even) {
                background-color: var(--hover-color);
            }

            .main-content {
                margin-top: 60px;  /* Add space for the fixed slider */
            }

            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }

                .sidebar {
                    width: 100%;
                    height: auto;
                    position: relative;
                }

                .main-content {
                    margin-left: 0;
                }

                .progress-container {
                    left: 0;
                }
            }
        </style>
        <script>
            function showFrame(frameId) {
                // Hide all frames
                document.querySelectorAll('.frame-container').forEach(container => {
                    container.style.display = 'none';
                });

                // Show selected frame
                const selectedFrame = document.getElementById(`frame-${frameId}`);
                if (selectedFrame) {
                    selectedFrame.style.display = 'block';
                }

                // Update frame counter
                document.getElementById('frame-counter').textContent = `Frame ${frameId}`;

                // Update metrics table with mock data for demonstration
                updateMetricsTable(frameId);
            }

            function updateMetricsTable(frameId) {
                // Mock data - in real implementation this would come from your tracking data
                const players = [
                    {id: '0.1', speed: (Math.random() * 20 + 10).toFixed(1), acceleration: (Math.random() * 5).toFixed(1), direction: (Math.random() * 360).toFixed(1)},
                    {id: '0.2', speed: (Math.random() * 20 + 10).toFixed(1), acceleration: (Math.random() * 5).toFixed(1), direction: (Math.random() * 360).toFixed(1)},
                    {id: '0.3', speed: (Math.random() * 20 + 10).toFixed(1), acceleration: (Math.random() * 5).toFixed(1), direction: (Math.random() * 360).toFixed(1)},
                ];

                const tbody = document.querySelector('.metrics-table tbody');
                tbody.innerHTML = players.map(player => `
                    <tr>
                        <td>Player ${player.id}</td>
                        <td>${player.speed} km/h</td>
                        <td>${player.acceleration} m/s²</td>
                        <td>${player.direction}°</td>
                    </tr>
                `).join('');
            }

            function showVisualization(frameId, visType) {
                const loading = document.querySelector('.loading');
                loading.classList.add('active');

                // Hide all visualizations for this frame
                const visContainers = document.querySelectorAll(`#frame-${frameId} .visualization`);
                visContainers.forEach(container => {
                    container.classList.remove('active');
                });

                // Show the selected visualization
                const selectedVis = document.getElementById(`${frameId}-${visType}`);
                if (selectedVis) {
                    selectedVis.classList.add('active');
                }

                // Update button states
                const buttons = document.querySelectorAll(`#frame-${frameId} button`);
                buttons.forEach(button => {
                    button.classList.remove('active');
                    if (button.getAttribute('data-vis') === visType) {
                        button.classList.add('active');
                    }
                });

                // Update progress bar
                updateProgress(frameId);

                // Hide loading after a short delay
                setTimeout(() => {
                    loading.classList.remove('active');
                }, 300);
            }

            function updateProgress(frameId) {
                const totalFrames = document.querySelectorAll('.frame-container').length;
                const currentFrame = parseInt(frameId);
                const progress = (currentFrame / totalFrames) * 100;
                document.querySelector('.progress').style.width = `${progress}%`;
            }

            function toggleTheme() {
                const html = document.documentElement;
                const currentTheme = html.getAttribute('data-theme');
                const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', newTheme);
                localStorage.setItem('theme', newTheme);
            }

            document.addEventListener('DOMContentLoaded', () => {
                const savedTheme = localStorage.getItem('theme') || 'light';
                document.documentElement.setAttribute('data-theme', savedTheme);

                // Initialize frame slider
                const slider = document.getElementById('frame-slider');
                slider.addEventListener('input', (e) => {
                    const frameId = parseInt(e.target.value);
                    showFrame(frameId);
                });

                // Add sidebar resize observer
                const sidebar = document.querySelector('.sidebar');
                const mainContent = document.querySelector('.main-content');
                const frameSliderContainer = document.querySelector('.frame-slider-container');

                const resizeObserver = new ResizeObserver(entries => {
                    for (let entry of entries) {
                        const width = entry.contentRect.width;
                        mainContent.style.marginLeft = `${width}px`;
                        frameSliderContainer.style.left = `${width}px`;
                    }
                });

                resizeObserver.observe(sidebar);

                // Add click handlers for image fullscreen
                document.querySelectorAll('.visualization').forEach(viz => {
                    viz.addEventListener('click', function() {
                        if (this.classList.contains('active')) {
                            this.classList.toggle('fullscreen');
                        }
                    });
                });

                // Show initial frame
                showFrame(0);
            });
        </script>
    </head>
    <body>
        <div class="container">
            <div class="sidebar">
                <div class="header">
                    <h1>Hockey Player Tracking</h1>
                    <div class="stats">
                        <p>Video: {os.path.basename(video_path)}</p>
                        <p>Total frames: {len(frames_info)}</p>
                    </div>
                    <div class="metrics-container">
                        <h2>Player Metrics</h2>
                        <table class="metrics-table">
                            <thead>
                                <tr>
                                    <th>Player ID</th>
                                    <th>Speed</th>
                                    <th>Acceleration</th>
                                    <th>Direction</th>
                                </tr>
                            </thead>
                            <tbody>
                                <!-- Populated by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="frame-slider-container">
                <input type="range" min="0" max="{len(frames_info) - 1}" value="0" class="frame-slider" id="frame-slider">
                <div class="frame-counter" id="frame-counter">Frame 0</div>
            </div>
            <div class="main-content">
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
        if "quadview_path" in frame_info:
            vis_types.append(("quadview", "Quadview"))
        
        # Create buttons
        for vis_type, label in vis_types:
            active_class = "active" if vis_type == "side_by_side" else ""
            buttons_html += f'<button class="{active_class}" data-vis="{vis_type}" onclick="showVisualization({frame_info["frame_id"]}, \'{vis_type}\')">{label}</button>'
        
        # Create visualization containers
        vis_containers_html = ""
        for vis_type, _ in vis_types:
            display_class = "active" if vis_type == "side_by_side" else ""
            if f"{vis_type}_path" in frame_info:
                vis_containers_html += f"""
                <div id="{frame_info['frame_id']}-{vis_type}" class="visualization {display_class}">
                    <img src="{frame_info[f'{vis_type}_path']}" alt="{vis_type.capitalize()} Frame {frame_info['frame_id']}">
                </div>
                """
        
        # Only the first frame should be visible initially
        display_style = "display: none;" if frame_info["frame_id"] > 0 else ""
        
        frame_html = f"""
        <div class="frame-container" id="frame-{frame_info['frame_id']}" style="{display_style}">
            <div class="frame-header">
                <div class="frame-title">
                    <h2>Frame {frame_info['frame_id']}</h2>
                    <span>Time: {frame_info['timestamp']:.2f}s</span>
                </div>
                <p>Players detected: {len(frame_info['players'])}</p>
                <div class="viz-buttons">
                    {buttons_html}
                </div>
            </div>
            <div class="visualizations">
                {vis_containers_html}
            </div>
        </div>
        """
        html_content += frame_html
    
    # Close HTML
    html_content += """
            </div>
            <button class="theme-toggle" onclick="toggleTheme()">Toggle Theme</button>
            <div class="loading">Loading...</div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)


def main():
    """
    Main function to parse arguments and process clip.
    """
    parser = argparse.ArgumentParser(description="Process a short clip from hockey broadcast footage to test player tracking")
    
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--detection-model", type=str, required=True, help="Path to detection model")
    parser.add_argument("--orientation-model", type=str, required=True, help="Path to orientation model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--segmentation-model", type=str, help="Path to segmentation model")
    parser.add_argument("--rink-coordinates", type=str, help="Path to rink coordinates JSON")
    parser.add_argument("--rink-image", type=str, help="Path to rink image")
    parser.add_argument("--start-second", type=float, default=0.0, help="Time in seconds to start processing from")
    parser.add_argument("--num-seconds", type=float, default=5.0, help="Number of seconds to process")
    parser.add_argument("--frame-step", type=int, default=5, help="Process every nth frame")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    process_clip(
        video_path=args.video,
        detection_model_path=args.detection_model,
        orientation_model_path=args.orientation_model,
        output_dir=args.output_dir,
        segmentation_model_path=args.segmentation_model,
        rink_coordinates_path=args.rink_coordinates,
        rink_image_path=args.rink_image,
        start_second=args.start_second,
        num_seconds=args.num_seconds,
        frame_step=args.frame_step,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
