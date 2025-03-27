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


def create_quadview(broadcast_frame, rink_img, coordinates, homography_matrix):
    """Create a quadview visualization matching the example layout."""
    # Get 2D rink with coordinates overlay
    rink_with_coords = draw_rink_coordinates(rink_img, coordinates)
    
    # Create frame with segmentation lines
    frame_with_lines = broadcast_frame.copy()
    
    # Warp the frame using homography matrix to rink space (1400x600)
    warped_frame = cv2.warpPerspective(
        broadcast_frame, 
        homography_matrix, 
        (rink_img.shape[1], rink_img.shape[0])
    )
    
    # Create a common size for all quadview images
    quadview_h, quadview_w = 600, 800
    
    # Resize broadcast frame and warped frame
    broadcast_resized = cv2.resize(broadcast_frame, (quadview_w, quadview_h))
    warped_resized = cv2.resize(warped_frame, (quadview_w, quadview_h))
    
    # Create overlay in original rink space (1400x600)
    if len(warped_frame.shape) == 2:
        warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(warped_frame, 0.5, rink_with_coords, 0.5, 0)
    
    # Resize rink and overlay images to quadview size
    rink_resized = cv2.resize(rink_with_coords, (quadview_w, quadview_h))
    overlay_resized = cv2.resize(overlay, (quadview_w, quadview_h))
    
    # Add titles
    cv2.putText(broadcast_resized, "Broadcast Frame with Lines", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(rink_resized, "2D Rink with Coordinates", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
    cv2.putText(warped_resized, "Warped Broadcast Frame", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(overlay_resized, "Warped Frame Overlay on Rink", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Create the quadview image
    top_row = np.hstack((broadcast_resized, rink_resized))
    bottom_row = np.hstack((warped_resized, overlay_resized))
    quadview = np.vstack((top_row, bottom_row))
    
    return quadview


def process_tracking_results(tracking_data_path, rink_coordinates_path, rink_image_path, output_dir):
    """Process existing tracking results to generate quadview visualizations."""
    # Load tracking data
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    
    # Load rink coordinates
    with open(rink_coordinates_path, 'r') as f:
        rink_coordinates = json.load(f)
    
    # Load rink image
    rink_img = cv2.imread(rink_image_path)
    if rink_img is None:
        print(f"Error: Could not load rink image {rink_image_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each frame in the tracking data (data is a dict with frame_ids as keys)
    for frame_id, frame_info in tracking_data.items():
        # Get broadcast frame path
        frames_dir = os.path.join(os.path.dirname(tracking_data_path), "frames", frame_id)
        broadcast_path = os.path.join(frames_dir, "broadcast.jpg")
        
        if os.path.exists(broadcast_path):
            broadcast_frame = cv2.imread(broadcast_path)
            
            if broadcast_frame is None:
                print(f"Warning: Could not load broadcast frame for frame {frame_id}")
                continue
            
            # Get homography matrix
            homography_success = frame_info.get("homography_success", False)
            
            if homography_success:
                homography_matrix = np.array(frame_info.get("homography_matrix"))
                
                # Print debug information about the homography points used
                if 'debug_info' in frame_info:
                    debug_info = frame_info['debug_info']
                    if 'common_point_names' in debug_info:
                        print(f"\nHomography points for frame {frame_id}:")
                        print("Common point names:", debug_info['common_point_names'])
                        if 'src_points_used' in debug_info and 'dst_points_used' in debug_info:
                            print("Source points:")
                            for i, point in enumerate(debug_info['src_points_used']):
                                name = debug_info['common_point_names'][i] if i < len(debug_info['common_point_names']) else "unknown"
                                print(f"  {name}: {point}")
                            print("Destination points:")
                            for i, point in enumerate(debug_info['dst_points_used']):
                                name = debug_info['common_point_names'][i] if i < len(debug_info['common_point_names']) else "unknown"
                                print(f"  {name}: {point}")
                
                # Create quadview
                quadview = create_quadview(broadcast_frame, rink_img, rink_coordinates, homography_matrix)
                
                # Save quadview
                quadview_path = os.path.join(output_dir, f"quadview_frame_{frame_id}.jpg")
                cv2.imwrite(quadview_path, quadview)
                print(f"Created quadview for frame {frame_id}: {quadview_path}")
            else:
                print(f"Warning: No homography matrix for frame {frame_id}")
        else:
            print(f"Warning: No broadcast frame found at {broadcast_path}")
    
    print(f"Quadview generation complete. Results saved to {output_dir}")


def generate_quadview(frame, warped_frame, rink_img, draw_rink_coordinates):
    """Generate a quadview visualization.
    
    Args:
        frame: Original broadcast frame
        warped_frame: Warped broadcast frame
        rink_img: 2D rink image
        draw_rink_coordinates: Function to draw coordinates on rink
        
    Returns:
        Quadview visualization image
    """
    # Create quadview visualization
    quadview = create_quadview(frame, warped_frame, rink_img, draw_rink_coordinates)
    
    return quadview


def main():
    parser = argparse.ArgumentParser(description='Generate quadview visualizations from tracking results')
    parser.add_argument('--tracking-data', required=True, help='Path to tracking data JSON')
    parser.add_argument('--rink-image', required=True, help='Path to rink image')
    parser.add_argument('--rink-coordinates', required=True, help='Path to rink coordinates JSON')
    parser.add_argument('--output-dir', required=True, help='Directory to save quadview images')
    
    args = parser.parse_args()
    
    process_tracking_results(
        tracking_data_path=args.tracking_data,
        rink_coordinates_path=args.rink_coordinates,
        rink_image_path=args.rink_image,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
