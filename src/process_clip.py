import cv2
import numpy as np
import os
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
import json
import shutil

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
    
    # Copy rink image to output directory if provided
    if rink_image_path:
        rink_output_path = os.path.join(output_dir, "rink_resized.png")
        shutil.copy2(rink_image_path, rink_output_path)
    
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
    """
    html_path = os.path.join(output_dir, "visualization.html")
    
    # Modern HTML template with improved UI
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hockey Player Tracking Visualization</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                background-color: #f5f5f5;
            }}

            .container {{
                display: flex;
                height: 100vh;
                overflow: hidden;
            }}

            .sidebar {{
                width: 400px;
                min-width: 300px;
                max-width: 800px;
                background-color: white;
                padding: 20px;
                box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
                resize: horizontal;
                overflow: auto;
                z-index: 2;
            }}

            .main-content {{
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}

            .frame-controls {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }}

            .frame-slider-container {{
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-top: 10px;
            }}

            #frame-slider {{
                width: 100%;
                height: 6px;
                -webkit-appearance: none;
                background: #e0e0e0;
                border-radius: 3px;
                outline: none;
                transition: background 0.2s;
            }}

            #frame-slider::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                background: #2196F3;
                border-radius: 50%;
                cursor: pointer;
                transition: all 0.2s;
            }}

            #frame-slider::-webkit-slider-thumb:hover {{
                transform: scale(1.2);
                background: #1976D2;
            }}

            #frame-counter {{
                font-size: 14px;
                color: #666;
                text-align: center;
            }}

            .viz-buttons {{
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }}

            .viz-buttons button {{
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                background: #f0f0f0;
                color: #333;
                cursor: pointer;
                transition: all 0.2s;
            }}

            .viz-buttons button:hover {{
                background: #e0e0e0;
            }}

            .viz-buttons button.active {{
                background: #2196F3;
                color: white;
            }}

            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}

            .metrics-table th,
            .metrics-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}

            .metrics-table th {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #333;
            }}

            .metrics-table tr:hover {{
                background-color: #f5f5f5;
            }}

            .frame-container {{
                display: none;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}

            .visualization {{
                display: none;
                width: 100%;
                height: 100%;
            }}

            .visualization img {{
                width: 100%;
                height: auto;
                display: block;
            }}

            .tracking-data-viz {{
                background-color: #000;
                border-radius: 8px;
                overflow: hidden;
            }}

            .tracking-data-viz canvas {{
                width: 100%;
                height: auto;
                display: block;
            }}

            h1, h2 {{
                color: #333;
                margin-bottom: 20px;
            }}

            h2 {{
                font-size: 1.2em;
                margin-top: 30px;
            }}
        </style>
        <script>
            // Initialize state
            let currentFrame = 0;
            let currentViz = 'tracking';
            let totalFrames = {len(frames_info)};
            
            // Parse frames data
            const framesData = """ + json.dumps(frames_info) + """;

            // Function to draw tracking data on canvas
            function drawTrackingData(frameNum) {
                const frameData = framesData[frameNum];
                if (!frameData || !frameData.players) return;

                const canvas = document.getElementById(`tracking-canvas-${frameNum}`);
                if (!canvas) return;

                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw rink background
                const rinkImg = new Image();
                rinkImg.onload = () => {
                    ctx.drawImage(rinkImg, 0, 0, canvas.width, canvas.height);

                    // Draw player positions
                    frameData.players.forEach(player => {
                        if (player.rink_position) {
                            // Get coordinates from the rink_position object
                            const x = player.rink_position.x;
                            const y = player.rink_position.y;
                            
                            // Draw player dot with glow effect
                            const isHome = player.type === 'home';
                            const color = isHome ? '#ff4444' : '#4444ff';
                            const glowColor = isHome ? '#ff000055' : '#0000ff55';

                            // Glow effect
                            ctx.beginPath();
                            ctx.arc(x, y, 12, 0, 2 * Math.PI);
                            ctx.fillStyle = glowColor;
                            ctx.fill();

                            // Player dot
                            ctx.beginPath();
                            ctx.arc(x, y, 8, 0, 2 * Math.PI);
                            ctx.fillStyle = color;
                            ctx.fill();

                            // Player ID
                            ctx.font = 'bold 14px Inter';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillStyle = 'white';
                            ctx.strokeStyle = 'black';
                            ctx.lineWidth = 3;
                            ctx.strokeText(player.player_id, x, y - 20);
                            ctx.fillText(player.player_id, x, y - 20);
                        }
                    });
                };
                rinkImg.onerror = (err) => {
                    console.error('Error loading rink image:', err);
                    ctx.font = '14px Inter';
                    ctx.fillStyle = 'red';
                    ctx.fillText('Error loading rink image', 10, 30);
                };
                // Use the rink image from the output directory
                rinkImg.src = 'rink_resized.png';
            }

            // Function to show specific frame
            function showFrame(frameNum) {
                if (frameNum < 0 || frameNum >= totalFrames) return;

                currentFrame = frameNum;
                document.getElementById('frame-slider').value = frameNum;
                document.getElementById('frame-counter').textContent = `Frame ${frameNum + 1} of ${totalFrames}`;

                // Hide all frames
                document.querySelectorAll('.frame-container').forEach(container => {
                    container.style.display = 'none';
                });

                // Show selected frame
                const currentContainer = document.getElementById(`frame-${frameNum}`);
                if (currentContainer) {
                    currentContainer.style.display = 'block';

                    // If current visualization is tracking data, draw it
                    if (currentViz === 'tracking_data') {
                        drawTrackingData(frameNum);
                    }
                }

                // Update metrics table
                updateMetricsTable(frameNum);
            }

            // Function to show specific visualization
            function showVisualization(frameId, vizType) {
                currentViz = vizType;

                // Update button states
                const currentContainer = document.getElementById(`frame-${frameId}`);
                if (currentContainer) {
                    // Update button states
                    currentContainer.querySelectorAll('.viz-buttons button').forEach(button => {
                        button.classList.toggle('active', button.dataset.viz === vizType);
                    });

                    // Hide all visualizations
                    currentContainer.querySelectorAll('.visualization').forEach(viz => {
                        viz.style.display = 'none';
                    });

                    // Show selected visualization
                    const vizContainer = document.getElementById(`frame-${frameId}-${vizType}`);
                    if (vizContainer) {
                        vizContainer.style.display = 'block';
                        // Draw tracking data if needed
                        if (vizType === 'tracking_data') {
                            drawTrackingData(frameId);
                        }
                    }
                }
            }

            // Update metrics table with real player data
            function updateMetricsTable(frameNum) {
                const frameData = framesData[frameNum];
                if (!frameData || !frameData.players) return;

                const tbody = document.querySelector('.metrics-table tbody');
                tbody.innerHTML = frameData.players.map(player => `
                    <tr>
                        <td>${player.player_id}</td>
                        <td>${player.speed || 'N/A'} km/h</td>
                        <td>${player.acceleration || 'N/A'} m/s²</td>
                        <td>${player.orientation || 'N/A'}°</td>
                    </tr>
                `).join('');
            }

            // Initialize on page load
            document.addEventListener('DOMContentLoaded', () => {
                // Initialize frame slider
                const slider = document.getElementById('frame-slider');
                slider.addEventListener('input', (e) => {
                    showFrame(parseInt(e.target.value));
                });

                // Add click handlers for visualization buttons
                document.querySelectorAll('.viz-buttons button').forEach(button => {
                    button.addEventListener('click', () => {
                        const frameId = parseInt(button.closest('.frame-container').id.split('-')[1]);
                        showVisualization(frameId, button.dataset.viz);
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
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="main-content">
                <div class="frame-slider-container">
                    <input type="range" min="0" max="{len(frames_info) - 1}" value="0" id="frame-slider">
                    <div class="frame-counter" id="frame-counter">Frame 1 of {len(frames_info)}</div>
                </div>
"""

    # Add frames to the HTML
    for frame_info in frames_info:
        frame_id = frame_info['frame_id']
        
        # Create visualization buttons
        vis_types = []
        if "original_frame_path" in frame_info:
            vis_types.append(("tracking", "Player Detection"))
        if "segmentation_path" in frame_info:
            vis_types.append(("segmentation", "Segmentation"))
        if "rink_overlay_path" in frame_info:
            vis_types.append(("rink", "Rink Overlay"))
        # Add tracking data visualization if we have player positions
        if any(p.get("rink_position") is not None for p in frame_info.get("players", [])):
            vis_types.append(("tracking_data", "Tracking Data"))
        
        buttons_html = "".join([
            f'<button class="{"active" if vtype == "tracking" else ""}" '
            f'data-viz="{vtype}">{label}</button>'
            for vtype, label in vis_types
        ])
        
        # Create visualization containers
        vis_containers = []
        for vis_type, _ in vis_types:
            if vis_type == "tracking_data":
                # Add tracking data visualization container with rink canvas
                vis_containers.append(f"""
                    <div id="frame-{frame_info['frame_id']}-tracking_data" class="visualization tracking-data-viz">
                        <canvas id="tracking-canvas-{frame_info['frame_id']}" width="1400" height="600"></canvas>
                    </div>
                """)
            else:
                # Regular image-based visualization
                path_key = f"{vis_type}_path" if vis_type != "tracking" else "original_frame_path"
                if path_key in frame_info:
                    frame_path = os.path.join(".", frame_info[path_key])
                    vis_containers.append(f"""
                        <div id="frame-{frame_info['frame_id']}-{vis_type}" class="visualization {'active' if vis_type == 'tracking' else ''}">
                            <img src="{frame_path}" alt="{vis_type.capitalize()} Frame {frame_info['frame_id']}">
                        </div>
                    """)
        
        vis_containers_html = "\n".join(vis_containers)
        
        # Create frame container
        frame_html = f"""
                <div class="frame-container {'active' if frame_id == 0 else ''}" id="frame-{frame_id}">
                    <div class="viz-buttons">
                        {buttons_html}
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
