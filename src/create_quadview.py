#!/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse

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
    
    # Draw diagonal goal line as requested
    cv2.line(img, left_bottom, right_top, (255, 0, 255), 2)
    
    cv2.putText(img, "goal_left_top", (left_top[0] - 100, left_top[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(img, "goal_left_bottom", (left_bottom[0] - 100, left_bottom[1] + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(img, "goal_right_top", (right_top[0] + 5, right_top[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.putText(img, "goal_right_bottom", (right_bottom[0] + 5, right_bottom[1] + 15), 
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


def create_quadview(broadcast_frame, warped_frame, rink_img, draw_rink_coordinates):
    # Create a 2x2 grid of images
    height, width = broadcast_frame.shape[:2]
    
    # Top left: Broadcast frame with lines
    broadcast_resized = cv2.resize(broadcast_frame, (width, height))
    
    # Top right: 2D rink with coordinates
    rink_with_coords = rink_img.copy()
    draw_rink_coordinates(rink_with_coords)
    rink_resized = cv2.resize(rink_with_coords, (width, height))
    
    # Bottom left: Warped broadcast frame
    warped_resized = cv2.resize(warped_frame, (width, height))
    
    # Bottom right: Warped frame overlay on rink
    # First ensure warped frame has same number of channels as rink image
    if len(warped_frame.shape) == 2:
        warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_GRAY2BGR)
    
    # Create overlay in original rink space (1400x600)
    rink_overlay = rink_with_coords.copy()
    overlay = cv2.addWeighted(warped_frame, 0.5, rink_overlay, 0.5, 0)
    
    # Then resize the overlay to match our quadview dimensions
    overlay_resized = cv2.resize(overlay, (width, height))
    
    # Create the 2x2 grid
    top_row = np.hstack((broadcast_resized, rink_resized))
    bottom_row = np.hstack((warped_resized, overlay_resized))
    quadview = np.vstack((top_row, bottom_row))
    
    # Add titles to each quadrant
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    font_color = (255, 255, 255)
    
    # Add title to top left quadrant
    cv2.putText(quadview, 'Broadcast Frame with Lines', 
                (10, 30), font, font_scale, font_color, font_thickness)
    
    # Add title to top right quadrant
    cv2.putText(quadview, '2D Rink with Coordinates',
                (width + 10, 30), font, font_scale, font_color, font_thickness)
    
    # Add title to bottom left quadrant
    cv2.putText(quadview, 'Warped Broadcast Frame',
                (10, height + 30), font, font_scale, font_color, font_thickness)
    
    # Add title to bottom right quadrant
    cv2.putText(quadview, 'Warped Frame Overlay on Rink',
                (width + 10, height + 30), font, font_scale, font_color, font_thickness)
    
    return quadview


def main():
    parser = argparse.ArgumentParser(description='Create quadview visualization from extracted frames')
    parser.add_argument('--input-frame', required=True, help='Path to input frame image')
    parser.add_argument('--rink-image', required=True, help='Path to rink image')
    parser.add_argument('--rink-coordinates', required=True, help='Path to rink coordinates JSON')
    parser.add_argument('--segmentation-model', required=True, help='Path to segmentation model')
    parser.add_argument('--output', required=True, help='Path for output quadview image')
    
    args = parser.parse_args()
    
    # Load the input frame
    frame = cv2.imread(args.input_frame)
    if frame is None:
        print(f"Error: Could not load frame {args.input_frame}")
        return
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Initialize segmentation processor
    segmentation_processor = SegmentationProcessor(
        model_path=args.segmentation_model
    )
    
    # Initialize homography calculator
    homography_calculator = HomographyCalculator(
        rink_coordinates_path=args.rink_coordinates,
        broadcast_width=frame_width,
        broadcast_height=frame_height
    )
    
    # Process the frame to get segmentation features
    segmentation_features = segmentation_processor.process_frame(frame)
    
    # Calculate homography
    success, homography_matrix, debug_info = homography_calculator.calculate_homography(segmentation_features)
    
    # Create frame with segmentation lines and rink features
    frame_with_lines = homography_calculator.draw_visualization(
        frame.copy(), 
        segmentation_features,
        homography_matrix if success else None
    )
    
    # Load and draw coordinates on rink image
    rink_img = cv2.imread(args.rink_image)
    if rink_img is None:
        print(f"Error: Could not load rink image {args.rink_image}")
        return
    
    # Load rink coordinates
    with open(args.rink_coordinates, 'r') as f:
        rink_coordinates = json.load(f)
    
    # Draw coordinates on rink image
    rink_with_coords = draw_rink_coordinates(rink_img, rink_coordinates)
    
    # Create warped frame if homography succeeded
    if success:
        warped_frame = cv2.warpPerspective(
            frame, homography_matrix,
            (rink_img.shape[1], rink_img.shape[0])
        )
    else:
        # Create blank warped frame if homography failed
        warped_frame = np.zeros_like(rink_img)
        cv2.putText(warped_frame, "Homography Failed", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Create quadview visualization
    quadview = create_quadview(frame, warped_frame, rink_with_coords, draw_rink_coordinates)
    
    # Save output
    cv2.imwrite(args.output, quadview)
    print(f"Saved quadview visualization to {args.output}")


if __name__ == "__main__":
    main()
