import cv2
import numpy as np
import os
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
import json
import shutil
import math

from player_tracker import PlayerTracker, NumpyEncoder


def calculate_player_metrics(frames_info: List[Dict], fps: float = 30.0) -> List[Dict]:
    """
    Calculate player metrics (speed, acceleration, orientation) for each frame.
    
    Args:
        frames_info: List of frame data dictionaries
        fps: Video frames per second
        
    Returns:
        Updated frames_info with metrics added
    """
    # Track player positions over time
    player_history = {}
    
    for frame_idx, frame_data in enumerate(frames_info):
        if not frame_data.get('players'):
            continue
            
        for player in frame_data['players']:
            player_id = player['player_id']
            
            # Initialize metrics (preserve existing fields)
            if 'speed' not in player:
                player['speed'] = 0.0
            if 'acceleration' not in player:
                player['acceleration'] = 0.0
            if 'orientation' not in player:
                player['orientation'] = 0.0
            
            # Skip if no rink position
            if not player.get('rink_position'):
                continue
                
            current_pos = (
                player['rink_position']['x'],
                player['rink_position']['y']
            )
            
            # Initialize player history if not exists
            if player_id not in player_history:
                player_history[player_id] = []
            
            # Calculate speed and acceleration if we have history
            history = player_history[player_id]
            if history:
                # Calculate speed (pixels per second)
                prev_pos = history[-1]['position']
                time_diff = 1.0 / fps  # Time between frames
                
                # Calculate distance in pixels
                distance = np.sqrt(
                    (current_pos[0] - prev_pos[0])**2 +
                    (current_pos[1] - prev_pos[1])**2
                )
                
                # Convert to km/h (assuming 1 pixel = 0.1 meters)
                speed = (distance * 0.1) / time_diff  # m/s
                speed_kmh = speed * 3.6  # Convert to km/h
                player['speed'] = round(speed_kmh, 2)
                
                # Calculate acceleration if we have at least 2 previous positions
                if len(history) >= 2:
                    prev_speed = history[-1]['speed']  # m/s
                    acceleration = (speed - prev_speed) / time_diff  # m/s²
                    player['acceleration'] = round(acceleration, 2)
                
                # Calculate orientation (angle between current and previous position)
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                orientation = np.degrees(np.arctan2(dy, dx))
                # Normalize to 0-360 range
                orientation = (orientation + 360) % 360
                player['orientation'] = round(orientation, 2)
            
            # Update player history
            history.append({
                'position': current_pos,
                'speed': speed if history else 0.0,
                'frame_idx': frame_idx
            })
            
            # Keep only last 10 frames of history
            if len(history) > 10:
                history.pop(0)
    
    return frames_info


def calculate_moving_averages(frames_info: List[Dict], window_size: int = 5) -> List[Dict]:
    """
    Calculate moving averages for player metrics during the Python processing phase.
    
    Args:
        frames_info: List of frame data dictionaries
        window_size: Size of the moving average window
        
    Returns:
        Updated frames_info with moving averages added
    """
    # Create a dictionary to store player metrics history
    player_metrics = {}
    
    for frame_idx, frame_data in enumerate(frames_info):
        if not frame_data.get('players'):
            continue
            
        for player in frame_data['players']:
            player_id = player['player_id']
            
            # Initialize player metrics history if not exists
            if player_id not in player_metrics:
                player_metrics[player_id] = {
                    'speed': [],
                    'acceleration': [],
                    'orientation': []
                }
            
            # Get current metrics
            current_metrics = {
                'speed': player.get('speed', 0),
                'acceleration': player.get('acceleration', 0),
                'orientation': player.get('orientation', 0)
            }
            
            # Update metrics history
            for metric in ['speed', 'acceleration', 'orientation']:
                player_metrics[player_id][metric].append(current_metrics[metric])
                
                # Keep only the last window_size values
                if len(player_metrics[player_id][metric]) > window_size:
                    player_metrics[player_id][metric].pop(0)
                
                # Calculate moving average
                if len(player_metrics[player_id][metric]) > 0:
                    avg = sum(player_metrics[player_id][metric]) / len(player_metrics[player_id][metric])
                    player[f'{metric}_moving_avg'] = round(avg, 2)
                else:
                    player[f'{metric}_moving_avg'] = 0
    
    return frames_info


def process_clip(
    video_path: str,
    detection_model_path: str,
    orientation_model_path: str,
    output_dir: str,
    segmentation_model_path: Optional[str] = None,
    rink_coordinates_path: Optional[str] = None,
    rink_image_path: Optional[str] = None,
    start_second: float = 0.0,
    num_seconds: float = 0.0,
    frame_step: int = 1,
    max_frames: int = 0,
):
    """
    Process a video for player tracking system analysis.
    
    Args:
        video_path: Path to the input video
        detection_model_path: Path to the detection model
        orientation_model_path: Path to the orientation model
        output_dir: Directory to save outputs
        segmentation_model_path: Optional path to the segmentation model
        rink_coordinates_path: Optional path to the rink coordinates JSON
        rink_image_path: Optional path to the rink image for visualization
        start_second: Time in seconds to start processing from
        num_seconds: Number of seconds to process (0 = entire video)
        frame_step: Process every nth frame (1 = every frame)
        max_frames: Maximum number of frames to process (0 = no limit)
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
    
    # Initialize player tracker
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
    if num_seconds == 0:
        # Process entire video
        end_frame = total_frames
    else:
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
    
    # Limit the number of frames to process
    if max_frames_to_process > 0:
        end_frame = min(end_frame, start_frame + max_frames_to_process)
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every nth frame
        if (frame_idx - start_frame) % frame_step == 0:
            print(f"Processing frame {frame_idx}/{end_frame} ({(frame_idx - start_frame) / (end_frame - start_frame) * 100:.1f}%)")
            
            # Process the frame
            frame_data = tracker.process_frame(frame, frame_idx)
            
            # Calculate metrics for this frame's players using previous frames
            if len(processed_frames_info) > 0:
                last_frame = processed_frames_info[-1]
                for player in frame_data["players"]:
                    # Find this player in the last frame
                    last_player = next((p for p in last_frame["players"] if p["player_id"] == player["player_id"]), None)
                    if last_player and "rink_position" in player and "rink_position" in last_player:
                        # Calculate speed (pixels per second)
                        dt = 1.0 / fps
                        # Handle both tuple and dictionary formats for rink_position
                        if isinstance(player["rink_position"], dict):
                            dx = player["rink_position"]["x"] - last_player["rink_position"]["x"]
                            dy = player["rink_position"]["y"] - last_player["rink_position"]["y"]
                        else:
                            # Fallback for tuple format
                            dx = player["rink_position"][0] - last_player["rink_position"][0]
                            dy = player["rink_position"][1] - last_player["rink_position"][1]
                        speed = math.sqrt(dx * dx + dy * dy) / dt
                        player["speed"] = speed
                        
                        # Calculate acceleration
                        if "speed" in last_player:
                            player["acceleration"] = (speed - last_player["speed"]) / dt
                        else:
                            player["acceleration"] = 0.0
                    else:
                        player["speed"] = 0.0
                        player["acceleration"] = 0.0
            
            # Calculate moving averages for metrics
            window_size = 5
            for player in frame_data["players"]:
                # Find this player's history
                player_history = []
                for past_frame in processed_frames_info[-window_size:]:
                    past_player = next((p for p in past_frame["players"] if p["player_id"] == player["player_id"]), None)
                    if past_player:
                        player_history.append(past_player)
                
                # Calculate moving averages
                if player_history:
                    speed_values = [p["speed"] for p in player_history if "speed" in p]
                    acc_values = [p["acceleration"] for p in player_history if "acceleration" in p]
                    orient_values = [p["orientation"] for p in player_history if "orientation" in p]
                    
                    player["speed_ma"] = sum(speed_values) / len(speed_values) if speed_values else 0.0
                    player["acceleration_ma"] = sum(acc_values) / len(acc_values) if acc_values else 0.0
                    player["orientation_ma"] = sum(orient_values) / len(orient_values) if orient_values else 0.0
                else:
                    player["speed_ma"] = player.get("speed", 0.0)
                    player["acceleration_ma"] = player.get("acceleration", 0.0)
                    player["orientation_ma"] = player.get("orientation", 0.0)
            
            # Create directory for individual frame if it doesn't exist
            frame_dir = os.path.join(frames_dir, str(frame_idx))
            if not os.path.exists(frame_dir):
                os.makedirs(frame_dir)
            
            # Save original frame
            original_path = os.path.join(frame_dir, "original.jpg")
            cv2.imwrite(original_path, frame)
            
            # Create and save player detections visualization using ellipses
            # Convert player data back to detection format for visualization
            detections_for_vis = []
            for player in frame_data["players"]:
                if "bbox" in player:
                    x1, y1, x2, y2 = player["bbox"]
                    # Calculate reference point at bottom center
                    ref_x = (x1 + x2) / 2
                    ref_y = y2
                    
                    detection = {
                        "class": player.get("type", "player"),
                        "reference_point": {
                            "pixel_x": int(ref_x),
                            "pixel_y": int(ref_y)
                        }
                    }
                    detections_for_vis.append(detection)
            
            # Use the PlayerDetector's ellipse visualization method
            detections_vis = tracker.player_detector.visualize_detections(frame, detections_for_vis)
            
            detections_path = os.path.join(frame_dir, "detections.jpg")
            cv2.imwrite(detections_path, detections_vis)
            
            # Create and save tracking visualization if rink image is provided
            tracking_path = None
            if rink_image is not None:
                visualizations = tracker.visualize_frame(frame, frame_data, rink_image)
                if visualizations:
                    tracking_vis = visualizations.get("rink")
                    if tracking_vis is not None:
                        tracking_path = os.path.join(frame_dir, "tracking.jpg")
                        cv2.imwrite(tracking_path, tracking_vis)
            
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
                        "rink_position": p.get("rink_position", None),
                        "speed": p.get("speed", 0.0),
                        "acceleration": p.get("acceleration", 0.0),
                        "orientation": p.get("orientation", 0.0),
                        "speed_ma": p.get("speed_ma", 0.0),
                        "acceleration_ma": p.get("acceleration_ma", 0.0),
                        "orientation_ma": p.get("orientation_ma", 0.0),
                        "team": p.get("team", "Unknown"),
                        "team_confidence": p.get("team_confidence", 0.0),
                        "team_detection_method": p.get("team_detection_method", "unknown"),
                        "roboflow_class": p.get("roboflow_class", "unknown"),
                        "pose_landmarks": p.get("pose_landmarks", None)
                    } for p in frame_data["players"]
                ],
                "homography_success": frame_data.get("homography_success", False),
                "original_frame_path": os.path.join("frames", str(frame_idx), "original.jpg"),
                "detections_path": os.path.join("frames", str(frame_idx), "detections.jpg")
            }
            
            if tracking_path:
                frame_info["tracking_path"] = os.path.join("frames", str(frame_idx), "tracking.jpg")
            
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
            
            processed_frames_info.append(frame_info)
            frames_processed += 1
            
            if max_frames_to_process > 0 and frames_processed >= max_frames_to_process:
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
        create_html_visualization(processed_frames_info, output_dir, rink_image_path, fps)
        print(f"\nHTML visualization created at {os.path.join(output_dir, 'visualization.html')}")
    
    return processed_frames_info


def create_html_visualization(frames_info: List[Dict], output_dir: str, rink_image_path: Optional[str] = None, video_fps: float = 30.0) -> str:
    """
    Create an HTML visualization of the processed frames.
    
    Args:
        frames_info: List of frame data dictionaries
        output_dir: Directory to save output files
        rink_image_path: Optional path to the rink image
        video_fps: Video frames per second for real-time playback
        
    Returns:
        Path to the generated HTML file
    """
    html_path = os.path.join(output_dir, "visualization.html")

    # Ensure skeleton.js is available next to the HTML for browser loading
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_js = os.path.join(current_dir, "skeleton.js")
        if os.path.exists(src_js):
            shutil.copy2(src_js, os.path.join(output_dir, "skeleton.js"))
    except Exception:
        pass
    
    # Convert frames_info to JSON string
    frames_data_json = json.dumps(frames_info, cls=NumpyEncoder)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Player Tracking Visualization - Real-time Playback</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                display: flex;
                gap: 20px;
                max-width: 1800px;
                margin: 0 auto;
            }}
            .main-panel {{
                flex: 2;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .side-panel {{
                flex: 1;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .controls {{
                margin: 20px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .metrics-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .metrics-table th, .metrics-table td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .metrics-table th {{
                background-color: #f8f9fa;
            }}
            .tabs {{
                display: flex;
                gap: 2px;
                margin-bottom: 20px;
            }}
            .tab {{
                padding: 10px 20px;
                background: #e9ecef;
                border: none;
                cursor: pointer;
                border-radius: 4px 4px 0 0;
            }}
            .tab.active {{
                background: #007bff;
                color: white;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            .frame-image {{
                max-width: 100%;
                height: auto;
            }}
            .skeleton-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }}
            #frameSlider {{
                flex-grow: 1;
            }}
            .button-group {{
                display: flex;
                gap: 10px;
                margin: 10px 0;
            }}
            .video-info {{
                background: #f8f9fa;
                padding: 10px 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                margin-bottom: 15px;
                text-align: center;
            }}
            .video-info p {{
                margin: 0;
                color: #495057;
                font-size: 14px;
            }}
            .control-button {{
                padding: 5px 15px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            .control-button:hover {{
                background: #0056b3;
            }}
            #tracking {{
                position: relative;
                width: 1400px;
                height: 600px;
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
            }}
            .player-marker {{
                position: absolute;
                width: 10px;
                height: 10px;
                background-color: blue;
                border-radius: 50%;
                transform: translate(-50%, -50%);
            }}
            .player-label {{
                position: absolute;
                font-size: 12px;
                color: blue;
                transform: translate(10px, -10px);
            }}
            
            .team-info {{
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                margin-top: 20px;
            }}
            .team-distribution, .confidence-analysis, .home-away-distribution {{
                margin-bottom: 20px;
            }}
            .team-stats, .confidence-stats {{
                display: flex;
                gap: 20px;
                margin-top: 10px;
            }}
            .team-stat, .confidence-stat {{
                background: white;
                padding: 10px 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
                text-align: center;
                min-width: 120px;
            }}
            .team-label, .confidence-label {{
                font-weight: bold;
                color: #495057;
                display: block;
                margin-bottom: 5px;
                font-size: 12px;
            }}
            .team-table-container {{
                margin-top: 20px;
            }}
            .team-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            .team-table th, .team-table td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .team-table th {{
                background-color: #e9ecef;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="main-panel">
                <div class="video-info">
                    <p><strong>Video FPS:</strong> {video_fps:.1f} | <strong>Playback Speed:</strong> Real-time</p>
                </div>
                <div class="tabs">
                    <button class="tab active" data-tab="original">Original Frame</button>
                    <button class="tab" data-tab="detections">Player Detections</button>
                    <button class="tab" data-tab="tracking">Player Tracking</button>
                    <button class="tab" data-tab="skeleton">Skeleton</button>
                    <button class="tab" data-tab="team_detection">Team Detection</button>
                </div>
                
                <div id="original" class="tab-content active">
                    <img id="originalFrame" class="frame-image" src="" alt="Original frame">
                </div>
                <div id="detections" class="tab-content">
                    <img id="detectionsFrame" class="frame-image" src="" alt="Player detections">
                </div>
                <div id="tracking" class="tab-content">
                    <div id="trackingContainer" style="width: 1400px; height: 600px; position: relative;">
                        <img id="rinkImage" src="rink_resized.png" style="width: 100%; height: 100%; position: absolute; top: 0; left: 0;">
                        <div id="playerMarkers"></div>
                    </div>
                </div>
                <div id="skeleton" class="tab-content">
                    <div class="controls" style="margin: 0 0 10px 0;">
                        <label for="playerSelect">Select Player:</label>
                        <select id="playerSelect"></select>
                        <button class="control-button" id="skeletonGo">Go</button>
                    </div>
                    <div id="skeletonContainer" style="position: relative; display: inline-block;">
                        <img id="skeletonFrame" class="frame-image" src="" alt="Skeleton frame">
                        <canvas id="skeletonCanvas" class="skeleton-overlay"></canvas>
                    </div>
                </div>
                
                <div id="team_detection" class="tab-content">
                    <div id="teamDetectionContainer">
                        <h3>Team Detection Analysis</h3>
                        <div class="team-info">
                            <div class="team-distribution">
                                <h4>Team Distribution:</h4>
                                <div class="team-stats">
                                    <div class="team-stat">
                                        <span class="team-label">Team A:</span>
                                        <span id="teamACount">-</span>
                                    </div>
                                    <div class="team-stat">
                                        <span class="team-label">Team B:</span>
                                        <span id="teamBCount">-</span>
                                    </div>
                                    <div class="team-stat">
                                        <span class="team-label">Unknown:</span>
                                        <span id="unknownCount">-</span>
                                    </div>
                                </div>
                            </div>
                            <div class="home-away-distribution">
                                <h4>Team Distribution (Home/Away):</h4>
                                <div class="team-stats">
                                    <div class="team-stat">
                                        <span class="team-label">Team A (Home):</span>
                                        <span id="homeCount">-</span>
                                    </div>
                                    <div class="team-stat">
                                        <span class="team-label">Team B (Away):</span>
                                        <span id="awayCount">-</span>
                                    </div>
                                </div>
                            </div>
                            <div class="confidence-analysis">
                                <h4>Detection Confidence:</h4>
                                <div class="confidence-stats">
                                    <div class="confidence-stat">
                                        <span class="confidence-label">High Confidence:</span>
                                        <span id="highConfidenceCount">-</span>
                                    </div>
                                    <div class="confidence-stat">
                                        <span class="confidence-label">Medium Confidence:</span>
                                        <span id="mediumConfidenceCount">-</span>
                                    </div>
                                    <div class="confidence-stat">
                                        <span class="confidence-label">Low Confidence:</span>
                                        <span id="lowConfidenceCount">-</span>
                                    </div>
                                </div>
                            </div>
                            <div class="team-table-container">
                                <h4>Player Team Assignments:</h4>
                                            <table class="team-table">
                <thead>
                    <tr>
                        <th>Player ID</th>
                        <th>Roboflow Class</th>
                        <th>Team</th>
                        <th>Confidence</th>
                        <th>Method</th>
                    </tr>
                </thead>
                <tbody id="teamTableBody">
                </tbody>
            </table>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="controls">
                    <div class="button-group">
                        <button class="control-button" id="prevFrame">◀</button>
                        <button class="control-button" id="playPause">▶</button>
                        <button class="control-button" id="nextFrame">▶</button>
                    </div>
                    <input type="range" id="frameSlider" min="0" max="{len(frames_info)-1}" value="0">
                    <span id="frameNumber">Frame: 0</span>
                </div>
            </div>
            <div class="side-panel">
                <div class="metrics-table-container">
                    <h3>Player Metrics</h3>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Player ID</th>
                                <th>Speed (km/h)</th>
                                <th>Acceleration (m/s²)</th>
                                <th>Orientation (°)</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <script src="skeleton.js"></script>
        <script>
            const framesData = {frames_data_json};
            const frameSlider = document.getElementById('frameSlider');
            const frameNumber = document.getElementById('frameNumber');
            let isPlaying = false;
            let playInterval;
            
            // Tab switching
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.addEventListener('click', () => {{
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    
                    tab.classList.add('active');
                    document.getElementById(tab.dataset.tab).classList.add('active');
                }});
            }});

            function updateFrameImages(frameNum) {{
                const frameData = framesData[frameNum];
                if (!frameData) return;

                // Update frame images
                const originalImg = document.getElementById('originalFrame');
                originalImg.src = frameData.original_frame_path;
                const skeletonImg = document.getElementById('skeletonFrame');
                if (skeletonImg) {{ skeletonImg.src = frameData.original_frame_path; }}
                document.getElementById('detectionsFrame').src = frameData.detections_path || '';
                // After image loads, draw skeleton
                if (typeof drawSkeletonForSelectedPlayer === 'function' && skeletonImg) {{
                    const redraw = () => drawSkeletonForSelectedPlayer(frameNum);
                    if (skeletonImg.complete) {{
                        redraw();
                    }} else {{
                        skeletonImg.onload = redraw;
                    }}
                }}
                
                // Update player markers
                const playerMarkers = document.getElementById('playerMarkers');
                playerMarkers.innerHTML = '';
                
                if (frameData.players) {{
                    frameData.players.forEach(player => {{
                        // Only show markers for actual players, not pucks, goals, etc.
                        if (player.rink_position && player.type && player.type.toLowerCase() === 'player') {{
                            const rx = player.rink_position.x;
                            const ry = player.rink_position.y;

                            const marker = document.createElement('div');
                            marker.className = 'player-marker';
                            marker.style.left = `${{rx}}px`;
                            marker.style.top = `${{ry}}px`;

                            const label = document.createElement('div');
                            label.className = 'player-label';
                            label.textContent = `${{player.player_id}} (${{player.speed_ma}} km/h)`;

                            playerMarkers.appendChild(marker);
                            playerMarkers.appendChild(label);
                            label.style.left = `${{rx}}px`;
                            label.style.top = `${{ry}}px`;
                        }}
                    }});
                }}
            }}
            
            function updateMetricsTable(frameNum) {{
                const frameData = framesData[frameNum];
                if (!frameData || !frameData.players) return;

                const tbody = document.querySelector('.metrics-table tbody');
                tbody.innerHTML = frameData.players
                    .filter(player => player.type && player.type.toLowerCase() === 'player')
                    .map(player => {{
                        return `
                            <tr>
                                <td>${{player.player_id}}</td>
                                <td>${{player.speed_ma}} km/h</td>
                                <td>${{player.acceleration_ma}} m/s²</td>
                                <td>${{player.orientation_ma}}°</td>
                            </tr>
                        `;
                    }}).join('');
            }}
            
            function updateTeamDetection(frameNum) {{
                const frameData = framesData[frameNum];
                if (!frameData || !frameData.players) return;

                // Count team distribution
                let teamACount = 0;
                let teamBCount = 0;
                let unknownCount = 0;
                let homeCount = 0;
                let awayCount = 0;
                let highConfidence = 0;
                let mediumConfidence = 0;
                let lowConfidence = 0;

                frameData.players
                    .filter(player => player.type && player.type.toLowerCase() === 'player')
                    .forEach(player => {{
                        const team = player.team || 'Unknown';
                        const confidence = player.team_confidence || 0;
                        
                        if (team.includes('Team A')) teamACount++;
                        else if (team.includes('Team B')) teamBCount++;
                        else unknownCount++;
                        
                                                    // Count home/away based on team names
                            if (team.toLowerCase().includes('team a')) homeCount++;
                            else if (team.toLowerCase().includes('team b')) awayCount++;
                        
                        if (confidence > 0.8) highConfidence++;
                        else if (confidence > 0.6) mediumConfidence++;
                        else if (confidence > 0.4) lowConfidence++;
                    }});

                // Update team distribution
                document.getElementById('teamACount').textContent = teamACount;
                document.getElementById('teamBCount').textContent = teamBCount;
                document.getElementById('unknownCount').textContent = unknownCount;
                
                // Update home/away counts
                document.getElementById('homeCount').textContent = homeCount;
                document.getElementById('awayCount').textContent = awayCount;
                
                // Update confidence analysis
                document.getElementById('highConfidenceCount').textContent = highConfidence;
                document.getElementById('mediumConfidenceCount').textContent = mediumConfidence;
                document.getElementById('lowConfidenceCount').textContent = lowConfidence;

                // Update team table
                const teamTbody = document.getElementById('teamTableBody');
                teamTbody.innerHTML = frameData.players
                    .filter(player => player.type && player.type.toLowerCase() === 'player')
                    .map(player => {{
                        const team = player.team || 'Unknown';
                        const confidence = player.team_confidence || 0;
                        const method = player.team_detection_method || 'unknown';
                        const roboflowClass = player.roboflow_class || 'unknown';
                        
                        return `
                            <tr>
                                <td>${{player.player_id}}</td>
                                <td>${{roboflowClass}}</td>
                                <td>${{team}}</td>
                                <td>${{(confidence * 100).toFixed(1)}}%</td>
                                <td>${{method}}</td>
                            </tr>
                        `;
                    }}).join('');
            }}

            function updateFrame(frameNum) {{
                frameSlider.value = frameNum;
                frameNumber.textContent = `Frame: ${{frameNum}}`;
                if (typeof updatePlayerSelect === 'function') updatePlayerSelect(frameNum);
                updateFrameImages(frameNum);
                updateMetricsTable(frameNum);
                updateTeamDetection(frameNum);
            }}
            
            // Frame navigation controls
            document.getElementById('prevFrame').addEventListener('click', () => {{
                const newFrame = Math.max(0, parseInt(frameSlider.value) - 1);
                updateFrame(newFrame);
            }});

            document.getElementById('nextFrame').addEventListener('click', () => {{
                const newFrame = Math.min(framesData.length - 1, parseInt(frameSlider.value) + 1);
                updateFrame(newFrame);
            }});

            document.getElementById('playPause').addEventListener('click', () => {{
                const button = document.getElementById('playPause');
                if (isPlaying) {{
                    clearInterval(playInterval);
                    button.textContent = '▶';
                }} else {{
                    playInterval = setInterval(() => {{
                        const newFrame = (parseInt(frameSlider.value) + 1) % framesData.length;
                        updateFrame(newFrame);
                    }}, {int(1000 / video_fps)}); // Real-time playback ({video_fps:.1f} fps)
                    button.textContent = '⏸';
                }}
                isPlaying = !isPlaying;
            }});
            
            frameSlider.addEventListener('input', (e) => {{
                const frameNum = parseInt(e.target.value);
                updateFrame(frameNum);
            }});
            
            // Initialize with first frame
            updateFrame(0);
            const playerSelect = document.getElementById('playerSelect');
            if (playerSelect && typeof drawSkeletonForSelectedPlayer === 'function') {{
                playerSelect.addEventListener('change', () => {{
                    drawSkeletonForSelectedPlayer(parseInt(frameSlider.value));
                }});
            }}
            // Ensure selector is initially populated
            if (typeof updatePlayerSelect === 'function') {{
                updatePlayerSelect(0);
            }}
        </script>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path


def main():
    """
    Main function to parse arguments and process clip.
    """
    parser = argparse.ArgumentParser(description="Process hockey video for player tracking and analysis")
    
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--detection-model", type=str, required=True, help="Path to detection model")
    parser.add_argument("--orientation-model", type=str, required=True, help="Path to orientation model")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--segmentation-model", type=str, help="Path to segmentation model")
    parser.add_argument("--rink-coordinates", type=str, help="Path to rink coordinates JSON")
    parser.add_argument("--rink-image", type=str, help="Path to rink image")
    parser.add_argument("--start-second", type=float, default=0.0, help="Time in seconds to start processing from")
    parser.add_argument("--num-seconds", type=float, default=0.0, help="Number of seconds to process (0 = entire video)")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every nth frame (1 = every frame)")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process (0 = no limit)")
    
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
