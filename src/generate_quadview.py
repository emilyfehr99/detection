#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import argparse

def draw_rink_coordinates(rink_img, coordinates):
    """Draw rink coordinates on the rink image for visualization."""
    img = rink_img.copy()
    
    # Draw destination points (boundary of play area)
    for name, point in coordinates["destination_points"].items():
        x, y = int(point["x"]), int(point["y"])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            img,
            name,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )
    
    # Draw additional points
    additional = coordinates["additional_points"]
    
    # Draw blue lines
    blue_lines = additional["blue_lines"]
    for name, point in blue_lines.items():
        x, y = int(point["x"]), int(point["y"])
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(
            img,
            f"blue_{name}",
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )
    
    # Connect blue line points
    blue_top_left = (
        int(blue_lines["left_top"]["x"]),
        int(blue_lines["left_top"]["y"])
    )
    blue_bottom_left = (
        int(blue_lines["left_bottom"]["x"]),
        int(blue_lines["left_bottom"]["y"])
    )
    blue_top_right = (
        int(blue_lines["right_top"]["x"]),
        int(blue_lines["right_top"]["y"])
    )
    blue_bottom_right = (
        int(blue_lines["right_bottom"]["x"]),
        int(blue_lines["right_bottom"]["y"])
    )
    
    cv2.line(img, blue_top_left, blue_bottom_left, (255, 0, 0), 2)
    cv2.line(img, blue_top_right, blue_bottom_right, (255, 0, 0), 2)
    
    # Draw center line
    center_line = additional["center_line"]
    center_top = (
        int(center_line["top"]["x"]),
        int(center_line["top"]["y"])
    )
    center_bottom = (
        int(center_line["bottom"]["x"]),
        int(center_line["bottom"]["y"])
    )
    cv2.circle(img, center_top, 5, (0, 0, 255), -1)
    cv2.circle(img, center_bottom, 5, (0, 0, 255), -1)
    cv2.line(img, center_top, center_bottom, (0, 0, 255), 2)
    cv2.putText(
        img,
        "center_top",
        (center_top[0] + 5, center_top[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )
    cv2.putText(
        img,
        "center_bottom",
        (center_bottom[0] + 5, center_bottom[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )
    
    # Draw goal lines
    goal_lines = additional["goal_lines"]
    left_top = (
        int(goal_lines["left_top"]["x"]),
        int(goal_lines["left_top"]["y"])
    )
    left_bottom = (
        int(goal_lines["left_bottom"]["x"]),
        int(goal_lines["left_bottom"]["y"])
    )
    right_top = (
        int(goal_lines["right_top"]["x"]),
        int(goal_lines["right_top"]["y"])
    )
    right_bottom = (
        int(goal_lines["right_bottom"]["x"]),
        int(goal_lines["right_bottom"]["y"])
    )
    
    cv2.circle(img, left_top, 5, (255, 0, 255), -1)
    cv2.circle(img, left_bottom, 5, (255, 0, 255), -1)
    cv2.circle(img, right_top, 5, (255, 0, 255), -1)
    cv2.circle(img, right_bottom, 5, (255, 0, 255), -1)
    
    cv2.line(img, left_top, left_bottom, (255, 0, 255), 2)
    cv2.line(img, right_top, right_bottom, (255, 0, 255), 2)
    
    cv2.putText(
        img,
        "LEFT GOAL LINE",
        (left_top[0] - 120, left_top[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2
    )
    cv2.putText(
        img,
        "RIGHT GOAL LINE",
        (right_top[0] - 40, right_top[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2
    )
    
    # Draw center circle
    center_circle = additional["center_circle"]
    center = (
        int(center_circle["center"]["x"]),
        int(center_circle["center"]["y"])
    )
    radius = int(center_circle["radius"])
    cv2.circle(img, center, 5, (0, 255, 0), -1)
    cv2.circle(img, center, radius, (0, 255, 0), 2)
    cv2.putText(
        img,
        "center_circle",
        (center[0] + 5, center[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
    
    # Draw faceoff circles
    faceoff_circles = additional["faceoff_circles"]
    for name, circle in faceoff_circles.items():
        center = (
            int(circle["center"]["x"]),
            int(circle["center"]["y"])
        )
        radius = int(circle["radius"])
        cv2.circle(img, center, 5, (0, 255, 0), -1)
        cv2.circle(img, center, radius, (0, 255, 0), 2)
        cv2.putText(
            img,
            f"faceoff_{name}",
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    
    # Draw zone labels
    cv2.putText(
        img,
        "LEFT ZONE",
        (400, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    cv2.putText(
        img,
        "NEUTRAL ZONE",
        (650, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    cv2.putText(
        img,
        "RIGHT ZONE",
        (950, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2
    )
    
    return img


def create_quadview(
        broadcast_frame,
        rink_img,
        coordinates,
        homography_matrix=None,
        segmentation_features=None,
        players=None
):
    """Create a quadview visualization with the following layout:
    Top left: Broadcast footage with segmentation
    Top right: Warped broadcast frame overlaid on 2D rink with labels
    Bottom left: Broadcast footage with detections
    Bottom right: Clean 2D rink with player positions
    """
    # Get 2D rink with coordinates overlay for top right
    rink_with_coords = draw_rink_coordinates(rink_img, coordinates)
    
    # Create a clean rink image for bottom right (without coordinate labels)
    clean_rink = rink_img.copy()
    
    # Create a common size for all quadview images
    quadview_h, quadview_w = 600, 800
    
    # Create the quadview canvas (2x2 grid)
    quadview = np.zeros((quadview_h * 2, quadview_w * 2, 3), dtype=np.uint8)
    
    # Resize broadcast frame
    broadcast_resized = cv2.resize(broadcast_frame, (quadview_w, quadview_h))
    
    # Create segmentation visualization (top left)
    segmentation_vis = broadcast_resized.copy()
    if segmentation_features:
        # Draw segmentation features
        colors = {
            "BlueLine": (255, 0, 0),      # Blue
            "RedCenterLine": (0, 0, 255),  # Red
            "GoalLine": (255, 0, 255),    # Magenta
            "FaceoffCircle": (0, 255, 0)  # Green
        }
        
        for feature_type, features in segmentation_features.items():
            if feature_type in colors:
                color = colors[feature_type]
                for feature in features:
                    if "points" in feature:
                        points = feature["points"]
                        if len(points) >= 2:
                            # Scale points to visualization size
                            scale_x = quadview_w / broadcast_frame.shape[1]
                            scale_y = quadview_h / broadcast_frame.shape[0]
                            
                            # Draw lines between points
                            for i in range(len(points) - 1):
                                pt1 = (
                                    int(points[i]["x"] * scale_x),
                                    int(points[i]["y"] * scale_y)
                                )
                                pt2 = (
                                    int(points[i+1]["x"] * scale_x),
                                    int(points[i+1]["y"] * scale_y)
                                )
                                cv2.line(segmentation_vis, pt1, pt2, color, 2)
                    elif "center" in feature and "radius" in feature:
                        # Draw circles
                        center = feature["center"]
                        radius = feature["radius"]
                        center = (
                            int(center["x"] * scale_x),
                            int(center["y"] * scale_y)
                        )
                        radius = int(radius * scale_x)  # Use x scale for radius
                        cv2.circle(segmentation_vis, center, radius, color, 2)
    
    cv2.putText(
        segmentation_vis,
        "Broadcast Frame with Segmentation",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    quadview[:quadview_h, :quadview_w] = segmentation_vis
    
    # Create warped overlay visualization (top right)
    if homography_matrix is not None:
        # Warp the frame using homography matrix to rink space
        warped_frame = cv2.warpPerspective(
            broadcast_frame,
            homography_matrix,
            (rink_img.shape[1], rink_img.shape[0])
        )
        
        # Create overlay in original rink space
        if len(warped_frame.shape) == 2:
            warped_frame = cv2.cvtColor(warped_frame, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(warped_frame, 0.5, rink_with_coords, 0.5, 0)
        
        # Resize overlay
        overlay_resized = cv2.resize(overlay, (quadview_w, quadview_h))
        cv2.putText(
            overlay_resized,
            "Warped Frame on 2D Rink",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        quadview[:quadview_h, quadview_w:] = overlay_resized
    else:
        # Create blank image with error message if homography failed
        error_img = np.zeros((quadview_h, quadview_w, 3), dtype=np.uint8)
        cv2.putText(
            error_img,
            "Homography Failed",
            (quadview_w//4, quadview_h//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        quadview[:quadview_h, quadview_w:] = error_img
    
    # Create detections visualization (bottom left)
    detections_vis = broadcast_resized.copy()
    if players:
        for player in players:
            if "reference_point" in player:
                pos = player["reference_point"]
                # Scale coordinates to visualization size
                scale_x = quadview_w / broadcast_frame.shape[1]
                scale_y = quadview_h / broadcast_frame.shape[0]
                x = int(pos["pixel_x"] * scale_x)
                y = int(pos["pixel_y"] * scale_y)
                
                # Draw player marker (blue dot)
                cv2.circle(detections_vis, (x, y), 4, (255, 0, 0), -1)
                
                # Draw bounding box if available
                if "bbox" in player:
                    x1, y1, x2, y2 = player["bbox"]
                    x1, y1 = int(x1 * scale_x), int(y1 * scale_y)
                    x2, y2 = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(
                        detections_vis,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )
                
                # Add player ID and orientation if available
                label = player.get("player_id", "")
                if "orientation" in player:
                    label += f" ({player['orientation']}°)"
                if label:
                    cv2.putText(
                        detections_vis,
                        label,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )
    
    cv2.putText(
        detections_vis,
        "Broadcast Frame with Detections",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    quadview[quadview_h:, :quadview_w] = detections_vis
    
    # Create clean rink with players visualization (bottom right)
    clean_rink_resized = cv2.resize(clean_rink, (quadview_w, quadview_h))
    if players:
        for player in players:
            if "rink_position" in player:
                pos = player["rink_position"]
                # Scale coordinates to visualization size
                scale_x = quadview_w / rink_img.shape[1]
                scale_y = quadview_h / rink_img.shape[0]
                x = int(pos["x"] * scale_x)
                y = int(pos["y"] * scale_y)
                
                # Draw player marker (blue dot)
                cv2.circle(clean_rink_resized, (x, y), 4, (255, 0, 0), -1)
                
                # Add player ID and orientation if available
                label = player.get("player_id", "")
                if "orientation" in player:
                    label += f" ({player['orientation']}°)"
                if label:
                    cv2.putText(
                        clean_rink_resized,
                        label,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )
    
    cv2.putText(
        clean_rink_resized,
        "Players on 2D Rink",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    quadview[quadview_h:, quadview_w:] = clean_rink_resized
    
    return quadview


def process_tracking_results(
        tracking_data_path,
        rink_coordinates_path,
        rink_image_path,
        output_dir
):
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
    
    # Process each frame in the tracking data
    for frame_info in tracking_data['frames']:
        # Load original frame
        frame_path = os.path.join(
            os.path.dirname(tracking_data_path),
            frame_info['original_frame_path']
        )
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error: Could not load frame {frame_path}")
            continue
        
        # Extract homography matrix and features
        homography_matrix = frame_info.get('homography_matrix')
        segmentation_features = frame_info.get('segmentation_features', {})
        segmentation_features = segmentation_features.get('features', {})
        players = frame_info.get('players', [])
        
        # Create quadview visualization
        quadview = create_quadview(
            frame, 
            rink_img, 
            rink_coordinates, 
            homography_matrix=homography_matrix,
            segmentation_features=segmentation_features,
            players=players
        )
        
        # Save quadview
        output_path = os.path.join(
            output_dir,
            f"quadview_{frame_info['frame_id']}.jpg"
        )
        cv2.imwrite(output_path, quadview)
        print(f"Saved quadview for frame {frame_info['frame_id']} to {output_path}")


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
