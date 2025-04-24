import torch
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import display
from collections import defaultdict
import time
import os
import sys
import json

# Function to extract representative pixel from a bounding box
def get_representative_pixel(frame, bbox):
    """
    Extract a representative pixel from the bounding box
    Located at 1/3 from bottom and 1/2 from left of the box
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    # If box is invalid or too small, return None
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Calculate representative point (1/2 from left, 1/3 from bottom)
    x_rep = int(x1 + (x2 - x1) * 0.5)
    y_rep = int(y2 - (y2 - y1) * 0.33)
    
    # Ensure point is within image bounds
    x_rep = min(max(0, x_rep), frame.shape[1] - 1)
    y_rep = min(max(0, y_rep), frame.shape[0] - 1)
    
    # Get RGB value at this point
    pixel_value = frame[y_rep, x_rep]
    
    return (int(pixel_value[0]), int(pixel_value[1]), int(pixel_value[2]))

# Load the model
model_path = '/Users/ALEX/Desktop/Penn/CMU/Courses/Capstone Project/code/models/detection.pt'
model = YOLO(model_path)

# Set confidence
conf_threshold = 0.35
model.conf = conf_threshold

# We do not care about goal detection right now
exclude_classes = ['goal', 'faceoff']

# Video paths
video_path = '/Users/ALEX/Desktop/Penn/CMU/Courses/Capstone Project/code/stuff/PIT_vs_CHI_2016.mp4'
output_path = '/Users/ALEX/Desktop/Penn/CMU/Courses/Capstone Project/code/stuff/PIT_vs_CHI_2016_output.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate number of frames for 15 seconds
num_frames_15_sec = int(15 * fps)

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Dictionary to store active trackers
# Format: {track_id: {
#   "class": class_name, 
#   "last_seen": frame_number, 
#   "bbox": (x1,y1,x2,y2),
#   "active": True/False,
#   "velocity": (vx, vy),  # pixel movement per frame
#   "history": [(frame_num, bbox), ...],  # last N positions
#   "appearance": appearance_feature_vector,  # For appearance matching
#   "representative_pixels": [(frame_num, (r,g,b)), ...] # Representative pixels for analysis
# }}
all_tracks = {}

# Track history for visualization and analysis
track_history = {}

# Team classification with visual features
team_assignments = {}  # {track_id: "team1" or "team2"}

# Color mapping
team_colors = {
    "team1": (255, 0, 0),    # Red for first team
    "team2": (0, 0, 255),    # Blue for second team
    "referee": (0, 255, 0),  # Green for referee
    "unknown": (255, 255, 0) # Yellow for unassigned
}

# Counter for new track IDs
next_track_id = 1

# Maximum frames to keep a track "alive" when not visible
# Set higher for more persistent tracking (4 seconds)
MAX_DISAPPEARED_FRAMES = int(fps * 4)  

# Maximum history length to store for each track
MAX_HISTORY_LENGTH = 30  # frames

# Minimum tracked frames required to consider a track stable
MIN_HITS_FOR_STABLE_TRACK = 5

# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    box format: [x1, y1, x2, y2]
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

# Function to get distance between box centers
def calculate_center_distance(box1, box2):
    """
    Calculate Euclidean distance between centers of two boxes
    """
    center1 = [(box1[0] + box1[2])/2, (box1[1] + box1[3])/2]
    center2 = [(box2[0] + box2[2])/2, (box2[1] + box2[3])/2]
    
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# Function to calculate box center
def get_box_center(box):
    return [(box[0] + box[2])/2, (box[1] + box[3])/2]

# Function to extract appearance features from a bounding box
# In a real implementation, you would use a more sophisticated feature extractor
def extract_appearance_features(frame, bbox):
    """
    Extract simple appearance features from the bounding box region
    For demonstration, we'll use average color histograms
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Ensure coordinates are within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    # If box is invalid or too small, return None
    if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:
        return None
    
    roi = frame[y1:y2, x1:x2]
    
    # Calculate average RGB values (simple feature)
    # In a real implementation, you'd use a more robust feature
    try:
        hist_r = cv2.calcHist([roi], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [8], [0, 256])
        hist_b = cv2.calcHist([roi], [2], None, [8], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        
        # Combine features
        features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        return features
    except:
        # Fallback if histogram calculation fails
        return None

# Function to calculate appearance similarity between two feature vectors
def calculate_appearance_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    
    # Cosine similarity
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)

# Function to predict next position based on velocity
def predict_next_position(track, frame_diff=1):
    """
    Predict the next position of a track based on its velocity
    """
    if "velocity" not in track or track["velocity"] is None:
        return track["bbox"]
    
    bbox = track["bbox"]
    vx, vy = track["velocity"]
    
    # Apply velocity to predict new position
    predicted_bbox = [
        bbox[0] + vx * frame_diff,
        bbox[1] + vy * frame_diff,
        bbox[2] + vx * frame_diff,
        bbox[3] + vy * frame_diff
    ]
    
    return predicted_bbox

# Function to update track velocity
def update_track_velocity(track, new_bbox, frame_diff=1):
    """
    Update track velocity based on position change
    """
    if not track["history"]:
        return (0, 0)
    
    # Get the last known position
    last_frame, last_bbox = track["history"][-1]
    
    # Calculate time difference (in frames)
    if frame_diff == 0:
        return (0, 0)
    
    # Calculate center points
    last_center = get_box_center(last_bbox)
    new_center = get_box_center(new_bbox)
    
    # Calculate velocity (pixels per frame)
    vx = (new_center[0] - last_center[0]) / frame_diff
    vy = (new_center[1] - last_center[1]) / frame_diff
    
    # Apply smoothing with previous velocity if available
    if "velocity" in track and track["velocity"] is not None:
        prev_vx, prev_vy = track["velocity"]
        alpha = 0.7  # Smoothing factor (0.7 means 70% new, 30% old)
        vx = alpha * vx + (1 - alpha) * prev_vx
        vy = alpha * vy + (1 - alpha) * prev_vy
    
    return (vx, vy)

# Function to assign a player to a team based on position/appearance
def assign_team(bbox, class_name, frame, features=None):
    """
    This function is kept as a placeholder but not used for team assignment
    """
    return "player"

# Function to match new detections with existing tracks
def match_detections_to_tracks(detections, all_tracks, frame_rgb, frame_count):
    """
    Match new detections with existing tracks using multiple cues
    """
    # Step 1: Prepare active tracks and their predictions
    active_track_ids = [tid for tid, t in all_tracks.items() if t["active"]]
    
    if not active_track_ids:
        return {}, detections, []
    
    # Step 2: Calculate matching scores using multiple cues
    matching_scores = []
    
    for det_idx, det in enumerate(detections):
        det_bbox = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
        det_features = extract_appearance_features(frame_rgb, det_bbox)
        
        for track_id in active_track_ids:
            track = all_tracks[track_id]
            
            # Skip if class doesn't match
            if track["class"] != det["name"]:
                continue
            
            # Get predicted position from track
            frames_since_last_seen = frame_count - track["last_seen"]
            predicted_bbox = predict_next_position(track, frames_since_last_seen)
            
            # Calculate IoU score
            iou_score = calculate_iou(det_bbox, predicted_bbox)
            
            # Calculate center distance and convert to a score (higher is better)
            distance = calculate_center_distance(det_bbox, predicted_bbox)
            max_distance = np.sqrt(width**2 + height**2) / 4  # Normalize distance
            distance_score = max(0, 1 - (distance / max_distance))
            
            # Calculate appearance similarity
            appearance_score = 0.5  # Default
            if det_features is not None and "appearance" in track and track["appearance"] is not None:
                appearance_score = calculate_appearance_similarity(det_features, track["appearance"])
            
            # Combine scores (you can adjust weights)
            # Weight IoU higher for recently seen tracks
            iou_weight = max(0.1, min(0.6, 1.0 / (1 + 0.1 * frames_since_last_seen)))
            appearance_weight = 0.3
            distance_weight = 1.0 - iou_weight - appearance_weight
            
            combined_score = (
                iou_weight * iou_score + 
                distance_weight * distance_score + 
                appearance_weight * appearance_score
            )
            
            # Add to matching scores
            matching_scores.append((combined_score, det_idx, track_id))
    
    # Sort matches by score (highest first)
    matching_scores.sort(reverse=True)
    
    # Assign matches greedily
    matched_dets = set()
    matched_tracks = set()
    matched_pairs = {}
    
    for score, det_idx, track_id in matching_scores:
        # Only consider reasonable matches
        if score < 0.1:  # Minimum threshold for matching
            continue
            
        if det_idx not in matched_dets and track_id not in matched_tracks:
            matched_pairs[track_id] = detections[det_idx]
            matched_dets.add(det_idx)
            matched_tracks.add(track_id)
    
    # Get unmatched detections
    unmatched_detections = [det for i, det in enumerate(detections) if i not in matched_dets]
    
    # Get inactive tracks that might be reactivated
    reactivation_candidates = []
    for track_id, track in all_tracks.items():
        if not track["active"] and track_id not in matched_tracks:
            frames_since_last_seen = frame_count - track["last_seen"]
            # Consider for reactivation if not seen for a while but still within threshold
            if frames_since_last_seen <= MAX_DISAPPEARED_FRAMES:
                reactivation_candidates.append(track_id)
    
    return matched_pairs, unmatched_detections, reactivation_candidates

# Function to try matching detections with inactive tracks (for reidentification)
def match_with_inactive_tracks(unmatched_detections, reactivation_candidates, all_tracks, frame_rgb, frame_count):
    """
    Try to match unmatched detections with inactive tracks for re-identification
    """
    if not reactivation_candidates or not unmatched_detections:
        return {}, unmatched_detections
    
    # Calculate matching scores for inactive tracks
    inactive_matching_scores = []
    
    for det_idx, det in enumerate(unmatched_detections):
        det_bbox = [det['xmin'], det['ymin'], det['xmax'], det['ymax']]
        det_features = extract_appearance_features(frame_rgb, det_bbox)
        
        for track_id in reactivation_candidates:
            track = all_tracks[track_id]
            
            # Skip if class doesn't match
            if track["class"] != det["name"]:
                continue
            
            # Get predicted position from track
            frames_since_last_seen = frame_count - track["last_seen"]
            predicted_bbox = predict_next_position(track, frames_since_last_seen)
            
            # Calculate appearance similarity (most important for reidentification)
            appearance_score = 0.0
            if det_features is not None and "appearance" in track and track["appearance"] is not None:
                appearance_score = calculate_appearance_similarity(det_features, track["appearance"])
            
            # Calculate IoU score (less important for long disappearances)
            iou_score = calculate_iou(det_bbox, predicted_bbox)
            
            # Calculate position similarity based on how far the detection is from where we expect the track to be
            distance = calculate_center_distance(det_bbox, predicted_bbox)
            max_distance = np.sqrt(width**2 + height**2) / 4  # Normalize distance
            position_score = max(0, 1 - (distance / max_distance))
            
            # Different weights for reidentification: appearance matters more
            appearance_weight = 0.7
            position_weight = 0.2
            iou_weight = 0.1
            
            combined_score = (
                appearance_weight * appearance_score + 
                position_weight * position_score + 
                iou_weight * iou_score
            )
            
            # Add to matching scores
            inactive_matching_scores.append((combined_score, det_idx, track_id))
    
    # Sort matches by score (highest first)
    inactive_matching_scores.sort(reverse=True)
    
    # Assign matches greedily
    matched_dets = set()
    matched_tracks = set()
    reactivated_tracks = {}
    
    for score, det_idx, track_id in inactive_matching_scores:
        # Higher threshold for reidentification to avoid false positives
        if score < 0.3:  
            continue
            
        if det_idx not in matched_dets and track_id not in matched_tracks:
            reactivated_tracks[track_id] = unmatched_detections[det_idx]
            matched_dets.add(det_idx)
            matched_tracks.add(track_id)
    
    # Get remaining unmatched detections
    remaining_unmatched = [det for i, det in enumerate(unmatched_detections) if i not in matched_dets]
    
    return reactivated_tracks, remaining_unmatched

# Lists to store tracking data
tracking_data = []
frame_count = 0

# Process the first 15 seconds of video
print("Starting video processing...")
start_time = time.time()

for i in range(num_frames_15_sec):
    ret, frame = cap.read()
    
    if not ret:
        print(f"Couldn't read frame {i}. Ending processing.")
        break
    
    # Convert frame to RGB for YOLO model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(frame_rgb)
    
    # Convert detections to standard format
    current_detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            
            if confidence > conf_threshold and class_name.lower() not in exclude_classes:
                current_detections.append({
                    'name': class_name,
                    'confidence': confidence,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'class_id': class_id
                })
    
    # Step 1: Match detections with active tracks
    matched_tracks, unmatched_detections, reactivation_candidates = match_detections_to_tracks(
        current_detections, all_tracks, frame_rgb, frame_count
    )
    
    # Step 2: Try to match remaining detections with inactive tracks (reidentification)
    reactivated_tracks, remaining_unmatched = match_with_inactive_tracks(
        unmatched_detections, reactivation_candidates, all_tracks, frame_rgb, frame_count
    )
    
    # Step 3: Update matched active tracks
    for track_id, detection in matched_tracks.items():
        track = all_tracks[track_id]
        bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        
        # Extract appearance features
        features = extract_appearance_features(frame_rgb, bbox)
        
        # Update velocity
        frames_since_last_seen = frame_count - track["last_seen"]
        new_velocity = update_track_velocity(track, bbox, frames_since_last_seen)
        
        # Extract representative pixel
        rep_pixel = get_representative_pixel(frame_rgb, bbox)
        
        # Extract representative pixel
        rep_pixel = get_representative_pixel(frame_rgb, bbox)
        
        # Update track info
        track.update({
            "bbox": bbox,
            "last_seen": frame_count,
            "active": True,
            "velocity": new_velocity
        })
        
        # Store representative pixel
        if rep_pixel is not None:
            if "representative_pixels" not in track:
                track["representative_pixels"] = []
            track["representative_pixels"].append((frame_count, rep_pixel))
        
        # Store representative pixel
        if rep_pixel is not None:
            if "representative_pixels" not in track:
                track["representative_pixels"] = []
            track["representative_pixels"].append((frame_count, rep_pixel))
        
        # Update appearance features (using moving average)
        if features is not None:
            if "appearance" in track and track["appearance"] is not None:
                # Blend new features with old (70% old, 30% new)
                track["appearance"] = 0.7 * track["appearance"] + 0.3 * features
            else:
                track["appearance"] = features
        
        # Update history
        track["history"].append((frame_count, bbox))
        if len(track["history"]) > MAX_HISTORY_LENGTH:
            track["history"] = track["history"][-MAX_HISTORY_LENGTH:]
        
        # Update track history for visualization
        if track_id not in track_history:
            track_history[track_id] = {
                "frames_visible": [frame_count],
                "class": detection['name'],
                "positions": [bbox]
            }
        else:
            track_history[track_id]["frames_visible"].append(frame_count)
            track_history[track_id]["positions"].append(bbox)
        
        # Store tracking data for analysis
        tracking_data.append({
            'frame': frame_count,
            'track_id': track_id,
            'class': detection['name'],
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3]
        })
    
    # Step 4: Update reactivated tracks
    for track_id, detection in reactivated_tracks.items():
        track = all_tracks[track_id]
        bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        
        # Extract appearance features
        features = extract_appearance_features(frame_rgb, bbox)
        
        # Update velocity (with caution as the track was inactive)
        frames_since_last_seen = frame_count - track["last_seen"]
        new_velocity = update_track_velocity(track, bbox, frames_since_last_seen)
        
        # Update track info
        track.update({
            "bbox": bbox,
            "last_seen": frame_count,
            "active": True,
            "velocity": new_velocity
        })
        
        # Update appearance features (using moving average)
        if features is not None:
            if "appearance" in track and track["appearance"] is not None:
                # Blend new features with old (80% old, 20% new for reactivated tracks)
                track["appearance"] = 0.8 * track["appearance"] + 0.2 * features
            else:
                track["appearance"] = features
        
        # Update history
        track["history"].append((frame_count, bbox))
        if len(track["history"]) > MAX_HISTORY_LENGTH:
            track["history"] = track["history"][-MAX_HISTORY_LENGTH:]
        
        # Update track history for visualization
        if track_id not in track_history:
            track_history[track_id] = {
                "frames_visible": [frame_count],
                "class": detection['name'],
                "positions": [bbox]
            }
        else:
            track_history[track_id]["frames_visible"].append(frame_count)
            track_history[track_id]["positions"].append(bbox)
        
        # Store tracking data
        tracking_data.append({
            'frame': frame_count,
            'track_id': track_id,
            'class': detection['name'],
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3]
        })
        
        print(f"Reactivated track {track_id} after {frames_since_last_seen} frames!")
    
    # Step 5: Initialize new tracks for remaining unmatched detections
    for detection in remaining_unmatched:
        bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
        
        # Extract appearance features
        features = extract_appearance_features(frame_rgb, bbox)
        
        # Create a new track ID
        new_track_id = next_track_id
        next_track_id += 1
        
        # Assign team
        team = assign_team(bbox, detection['name'], frame_rgb, features)
        team_assignments[new_track_id] = team
        
        # Extract representative pixel
        rep_pixel = get_representative_pixel(frame_rgb, bbox)
        
        # Create new track entry
        all_tracks[new_track_id] = {
            "class": detection['name'],
            "last_seen": frame_count,
            "bbox": bbox,
            "active": True,
            "velocity": (0, 0),  # Initial velocity
            "history": [(frame_count, bbox)],
            "appearance": features,
            "representative_pixels": [(frame_count, rep_pixel)] if rep_pixel is not None else []
        }
        
        # Initialize track history
        track_history[new_track_id] = {
            "frames_visible": [frame_count],
            "class": detection['name'],
            "positions": [bbox]
        }
        
        # Store tracking data
        tracking_data.append({
            'frame': frame_count,
            'track_id': new_track_id,
            'class': detection['name'],
            'xmin': bbox[0],
            'ymin': bbox[1],
            'xmax': bbox[2],
            'ymax': bbox[3]
        })
    
    # Step 6: Update status of tracks that weren't seen
    for track_id, track in all_tracks.items():
        frames_since_last_seen = frame_count - track["last_seen"]
        
        # Mark as inactive if not seen in this frame
        if track_id not in matched_tracks and track_id not in reactivated_tracks:
            if frames_since_last_seen > 0:
                track["active"] = False
            
            # If the track wasn't seen for too long, we could remove it completely
            # But for hockey where players might leave and re-enter, we keep them longer
            if frames_since_last_seen > MAX_DISAPPEARED_FRAMES:
                # Instead of removing, we keep the track but mark it as very inactive
                # This allows for later reidentification
                track["very_inactive"] = True
    
    # Create a copy of the frame for visualization
    annotated_frame = frame_rgb.copy()
    
    # Draw bounding boxes and labels for all active tracks
    for track_id, track in all_tracks.items():
        # Only draw active tracks seen in current frame
        if track["last_seen"] == frame_count:
            bbox = track["bbox"]
            class_name = track["class"]
            color = (0, 255, 0)  # Green for all players
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame, 
                (int(bbox[0]), int(bbox[1])), 
                (int(bbox[2]), int(bbox[3])), 
                color, 
                2
            )
            
            # Draw ID and class
            label = f"ID:{track_id} {class_name}"
            cv2.putText(
                annotated_frame, 
                label, 
                (int(bbox[0]), int(bbox[1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
            
            # Draw the representative pixel point
            if "representative_pixels" in track and track["representative_pixels"]:
                # Get latest representative pixel
                _, rep_pixel = track["representative_pixels"][-1]
                
                # Calculate representative point position
                x_rep = int(bbox[0] + (bbox[2] - bbox[0]) * 0.5)
                y_rep = int(bbox[3] - (bbox[3] - bbox[1]) * 0.33)
                
                # Draw a small circle at the representative pixel position
                cv2.circle(
                    annotated_frame,
                    (x_rep, y_rep),
                    3,  # Radius
                    (255, 255, 0),  # Yellow
                    -1  # Filled circle
                )
    
    # Convert annotated frame back to BGR for saving to video
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out.write(annotated_frame_bgr)
    
    # Update frame counter
    frame_count += 1
    
    # Display progress
    if i % 10 == 0:
        elapsed = time.time() - start_time
        fps_processing = i / max(0.1, elapsed)
        print(f"Processed frame {i}/{num_frames_15_sec} ({fps_processing:.1f} FPS)")

# Convert tracking data to DataFrame
tracking_df = pd.DataFrame(tracking_data)

# Create a DataFrame for the representative pixels
rep_pixels_data = []
for track_id, track in all_tracks.items():
    if "representative_pixels" in track and track["representative_pixels"]:
        for frame_num, pixel_value in track["representative_pixels"]:
            rep_pixels_data.append({
                'track_id': track_id,
                'frame': frame_num,
                'class': track["class"],
                'r': pixel_value[0],
                'g': pixel_value[1],
                'b': pixel_value[2]
            })

rep_pixels_df = pd.DataFrame(rep_pixels_data)

# Save representative pixels data to CSV
rep_pixels_csv_path = '/Users/ALEX/Desktop/Penn/CMU/Courses/Capstone Project/code/stuff/representative_pixels.csv'
rep_pixels_df.to_csv(rep_pixels_csv_path, index=False)

# Release resources
cap.release()
out.release()

# Display summary
elapsed = time.time() - start_time
print(f"Processing complete! Total time: {elapsed:.1f} seconds")
print(f"Processed {frame_count} frames ({frame_count/fps:.2f} seconds)")
print(f"Total unique tracks: {next_track_id - 1}")

# Show track statistics
print("\nTracks by class:")
if not tracking_df.empty:
    class_counts = tracking_df.groupby('track_id')['class'].first().value_counts()
    print(class_counts)
    
    # Calculate track durations and continuity
    track_durations = {}
    for track_id, history in track_history.items():
        frames_visible = history["frames_visible"]
        first_frame = min(frames_visible)
        last_frame = max(frames_visible)
        duration_frames = last_frame - first_frame + 1
        duration_seconds = duration_frames / fps
        
        # Calculate how continuous the track was (higher is better)
        continuity = len(frames_visible) / duration_frames
        
        track_durations[track_id] = {
            "class": history["class"],
            "first_frame": first_frame,
            "last_frame": last_frame,
            "duration_seconds": duration_seconds,
            "continuity": continuity,
            "frames_tracked": len(frames_visible)
        }
    
    durations_df = pd.DataFrame.from_dict(track_durations, orient='index')
    
    # Filter out very short tracks
    valid_tracks = durations_df[durations_df['frames_tracked'] > 5]
    
    # Summary statistics for track durations
    print("\nTrack duration statistics (seconds):")
    print(valid_tracks.groupby('class')['duration_seconds'].describe())
    
    print("\nTrack continuity statistics (higher is better):")
    print(valid_tracks.groupby('class')['continuity'].describe())
    
    # Plot track visibility over time
    plt.figure(figsize=(15, 8))
    
    for track_id, history in track_history.items():
        # Skip very short tracks
        if len(history["frames_visible"]) < 5:
            continue
            
        color = (0.0, 0.8, 0.0)  # Green for all players
        
        # Plot visibility
        frames = history["frames_visible"]
        y_values = [track_id] * len(frames)
        
        plt.scatter(frames, y_values, color=color, s=10, alpha=0.7)
        
        # Draw lines to show continuity
        if len(frames) > 1:
            prev_frame = frames[0]
            for frame in frames[1:]:
                if frame - prev_frame <= 5:  # Only connect points if gap is small
                    plt.plot([prev_frame, frame], [track_id, track_id], color=color, alpha=0.3)
                prev_frame = frame
