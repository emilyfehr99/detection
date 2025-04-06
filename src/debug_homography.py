#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Any, Optional

from homography_calculator import HomographyCalculator
from segmentation_processor import SegmentationProcessor


def draw_rink_coordinates(rink_img, coordinates):
    """Draw rink coordinates on the rink image for visualization."""
    img = rink_img.copy()
    
    # Draw destination points (boundary of play area)
    for name, point in coordinates["destination_points"].items():
        x, y = int(point["x"]), int(point["y"])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw additional points
    additional = coordinates["additional_points"]
    
    # Draw blue lines
    blue_lines = additional["blue_lines"]
    for name, point in blue_lines.items():
        x, y = int(point["x"]), int(point["y"])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img, f"blue_{name}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Connect blue line points
    blue_top_left = (int(blue_lines["left_top"]["x"]), int(blue_lines["left_top"]["y"]))
    blue_bottom_left = (int(blue_lines["left_bottom"]["x"]), int(blue_lines["left_bottom"]["y"]))
    blue_top_right = (int(blue_lines["right_top"]["x"]), int(blue_lines["right_top"]["y"]))
    blue_bottom_right = (int(blue_lines["right_bottom"]["x"]), int(blue_lines["right_bottom"]["y"]))
    
    cv2.line(img, blue_top_left, blue_bottom_left, (255, 0, 0), 2)
    cv2.line(img, blue_top_right, blue_bottom_right, (255, 0, 0), 2)
    
    # Draw center line
    center_line = additional["center_line"]
    center_top = (int(center_line["top"]["x"]), int(center_line["top"]["y"]))
    center_bottom = (int(center_line["bottom"]["x"]), int(center_line["bottom"]["y"]))
    cv2.circle(img, center_top, 5, (0, 0, 255), -1)
    cv2.circle(img, center_bottom, 5, (0, 0, 255), -1)
    cv2.line(img, center_top, center_bottom, (0, 0, 255), 2)
    cv2.putText(img, "center_top", (center_top[0] + 5, center_top[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img, "center_bottom", (center_bottom[0] + 5, center_bottom[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw goal lines
    goal_lines = additional["goal_lines"]
    left_top = (int(goal_lines["left_top"]["x"]), int(goal_lines["left_top"]["y"]))
    left_bottom = (int(goal_lines["left_bottom"]["x"]), int(goal_lines["left_bottom"]["y"]))
    right_top = (int(goal_lines["right_top"]["x"]), int(goal_lines["right_top"]["y"]))
    right_bottom = (int(goal_lines["right_bottom"]["x"]), int(goal_lines["right_bottom"]["y"]))
    
    cv2.circle(img, left_top, 5, (255, 0, 255), -1)
    cv2.circle(img, left_bottom, 5, (255, 0, 255), -1)
    cv2.circle(img, right_top, 5, (255, 0, 255), -1)
    cv2.circle(img, right_bottom, 5, (255, 0, 255), -1)
    
    cv2.line(img, left_top, left_bottom, (255, 0, 255), 2)
    cv2.line(img, right_top, right_bottom, (255, 0, 255), 2)
    
    cv2.putText(img, "LEFT GOAL LINE", (left_top[0] - 120, left_top[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(img, "RIGHT GOAL LINE", (right_top[0] - 40, right_top[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw center circle
    center_circle = additional["center_circle"]
    center = (int(center_circle["center"]["x"]), int(center_circle["center"]["y"]))
    radius = int(center_circle["radius"])
    cv2.circle(img, center, 5, (0, 255, 0), -1)
    cv2.circle(img, center, radius, (0, 255, 0), 2)
    cv2.putText(img, "center_circle", (center[0] + 5, center[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw faceoff circles
    faceoff_circles = additional["faceoff_circles"]
    for name, circle in faceoff_circles.items():
        center = (int(circle["center"]["x"]), int(circle["center"]["y"]))
        radius = int(circle["radius"])
        cv2.circle(img, center, 5, (0, 255, 0), -1)
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        cv2.putText(img, f"faceoff_{name}", (center[0] + 5, center[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw zone labels
    cv2.putText(img, "LEFT ZONE", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "NEUTRAL ZONE", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "RIGHT ZONE", (950, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img


def draw_homography_points(frame, source_points, dest_points_info, selected_points):
    """
    Visualize the points used for homography calculation directly on the frame.
    
    Args:
        frame: The broadcast frame to draw on
        source_points: Dictionary of source points {name: (x, y)}
        dest_points_info: Dictionary with info about destination points for labeling
        selected_points: List of point names that were used in homography calculation
    
    Returns:
        Frame with homography points visualized
    """
    vis_frame = frame.copy()
    
    # Draw all detected points in yellow (thin circle)
    for name, point in source_points.items():
        cv2.circle(vis_frame, (int(point[0]), int(point[1])), 7, (0, 255, 255), 1)
        cv2.putText(vis_frame, name, (int(point[0]) + 10, int(point[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw selected points in bright green (thick circle)
    for name in selected_points:
        if name in source_points:
            point = source_points[name]
            cv2.circle(vis_frame, (int(point[0]), int(point[1])), 10, (0, 255, 0), 3)
            cv2.putText(vis_frame, f"{name} → 2D RINK", (int(point[0]) + 10, int(point[1]) + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add a legend
    cv2.putText(vis_frame, "HOMOGRAPHY MAPPING POINTS:", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_frame, "Detected Points", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_frame, "Selected for Homography", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw a line connecting selected points to visualize the mapping
    if len(selected_points) >= 2:
        for i in range(len(selected_points) - 1):
            if (selected_points[i] in source_points and 
                selected_points[i+1] in source_points):
                pt1 = (int(source_points[selected_points[i]][0]), 
                       int(source_points[selected_points[i]][1]))
                pt2 = (int(source_points[selected_points[i+1]][0]), 
                       int(source_points[selected_points[i+1]][1]))
                cv2.line(vis_frame, pt1, pt2, (0, 200, 0), 1, cv2.LINE_AA)
    
    return vis_frame


def create_debug_view(broadcast_frame, rink_img, homography_info):
    """Create a debug visualization showing points used for homography.
    
    Args:
        broadcast_frame: Original broadcast frame
        rink_img: 2D rink image with coordinates
        homography_info: Dictionary with homography debug information
        
    Returns:
        Debug view image
    """
    # Extract relevant information
    source_points = homography_info["source_points"]
    selected_points = homography_info["common_point_names"]
    
    # Create frame with homography points highlighted
    frame_with_points = draw_homography_points(
        broadcast_frame, 
        source_points, 
        {},  # No destination info needed
        selected_points
    )
    
    # Warp the frame using homography matrix
    h_matrix = homography_info["homography_matrix"]
    rink_shape = (rink_img.shape[1], rink_img.shape[0])
    warped_frame = cv2.warpPerspective(broadcast_frame, h_matrix, rink_shape)
    
    # Create a common size for all quadview images
    quadview_h, quadview_w = 600, 800
    
    # Resize all images to same size
    broadcast_resized = cv2.resize(
        frame_with_points, (quadview_w, quadview_h)
    )
    rink_resized = cv2.resize(rink_img, (quadview_w, quadview_h))
    warped_resized = cv2.resize(warped_frame, (quadview_w, quadview_h))
    
    # Create overlay of warped frame on rink
    overlay = rink_resized.copy()
    alpha = 0.5  # Transparency factor
    
    # Convert warped frame to same color space as rink if needed
    w_shape = len(warped_resized.shape)
    r_shape = len(rink_resized.shape)
    
    if w_shape == 3 and r_shape == 2:
        warped_resized = cv2.cvtColor(warped_resized, cv2.COLOR_BGR2GRAY)
    elif r_shape == 3 and w_shape == 2:
        warped_resized = cv2.cvtColor(warped_resized, cv2.COLOR_GRAY2BGR)
    
    # Create the overlay
    cv2.addWeighted(warped_resized, alpha, rink_resized, 1 - alpha, 0, overlay)
    
    # Add titles
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.8
    title_thickness = 2
    white = (255, 255, 255)
    black = (0, 0, 0)
    pos = (20, 30)
    
    cv2.putText(
        broadcast_resized, "Broadcast Frame with Points", 
        pos, title_font, title_scale, white, title_thickness
    )
    
    cv2.putText(
        rink_resized, "2D Rink with Coordinates",
        pos, title_font, title_scale, black, title_thickness
    )
                
    cv2.putText(
        warped_resized, "Warped Broadcast Frame",
        pos, title_font, title_scale, white, title_thickness
    )
                
    cv2.putText(
        overlay, "Warped Frame Overlay on Rink",
        pos, title_font, title_scale, black, title_thickness
    )
    
    # Create the quadview image
    top_row = np.hstack((broadcast_resized, rink_resized))
    bottom_row = np.hstack((warped_resized, overlay))
    quadview = np.vstack((top_row, bottom_row))
    
    return quadview


def process_frame(frame_path, rink_image_path, rink_coordinates_path, output_path):
    """
    Process a single frame to debug homography.
    
    Args:
        frame_path: Path to input broadcast frame
        rink_image_path: Path to 2D rink image
        rink_coordinates_path: Path to rink coordinates JSON
        output_path: Path to save the debug output
    """
    # Load frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Could not load frame from {frame_path}")
        return False
    
    # Load rink image
    rink_img = cv2.imread(rink_image_path)
    if rink_img is None:
        print(f"Error: Could not load rink image from {rink_image_path}")
        return False
    
    # Load rink coordinates
    with open(rink_coordinates_path, 'r') as f:
        rink_coordinates = json.load(f)
    
    # Initialize components
    homography_calculator = HomographyCalculator(
        broadcast_width=frame.shape[1],
        broadcast_height=frame.shape[0],
        rink_coordinates_path=rink_coordinates_path,
    )
    
    segmentation_processor = SegmentationProcessor(
        model_path="models/segmentation.pt",
    )
    
    # Process the frame
    segmentation_result = segmentation_processor.process_frame(frame)
    segmentation_features = segmentation_result
    segmentation_mask = segmentation_result["segmentation_mask"]
    
    # Save the segmentation mask as a separate image
    if segmentation_mask is not None:
        cv2.imwrite(output_path.replace('.jpg', '_segmentation.jpg'), segmentation_mask)
    
    # Print detected features
    print("\nDetected Features:")
    for feature_type, features in segmentation_features["features"].items():
        print(f"{feature_type}: {len(features)} instances")
        for i, feature in enumerate(features):
            if i < 3:  # Print just a few examples to avoid clutter
                print(f"  - {feature}")
    
    # Try to calculate homography
    success, homography_matrix, debug_info = homography_calculator.calculate_homography(
        segmentation_features
    )
    
    if not success:
        print(f"Error: Homography calculation failed - {debug_info.get('reason_for_failure', 'unknown reason')}")
        # Still attempt to create debug view
    else:
        print(f"Homography calculation successful using {len(debug_info['common_point_names'])} points")
    
    # Add the homography matrix to debug info
    debug_info["homography_matrix"] = homography_matrix
    
    # Draw rink with coordinates
    rink_with_coords = draw_rink_coordinates(rink_img, rink_coordinates)
    
    # Create debug view
    debug_view = create_debug_view(frame, rink_with_coords, debug_info)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the debug view
    cv2.imwrite(output_path, debug_view)
    print(f"Debug view saved to {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Debug homography transformation for hockey tracking")
    parser.add_argument("--frame", required=True, help="Path to input broadcast frame")
    parser.add_argument("--rink-image", required=True, help="Path to 2D rink image")
    parser.add_argument("--rink-coordinates", required=True, help="Path to rink coordinates JSON")
    parser.add_argument("--output", required=True, help="Path to save the debug output")
    
    args = parser.parse_args()
    
    process_frame(
        args.frame,
        args.rink_image,
        args.rink_coordinates,
        args.output
    )


if __name__ == "__main__":
    main()
