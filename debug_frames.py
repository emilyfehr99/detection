#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import argparse
import logging
import sys

from src.homography_calculator import HomographyCalculator
from src.segmentation_processor import SegmentationProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("debug_frames")

def process_frame(frame_path, frame_number, rink_image_path, rink_coordinates_path, output_dir):
    """
    Process a single frame to debug homography.
    
    Args:
        frame_path: Path to input broadcast frame
        frame_number: Frame number for reporting
        rink_image_path: Path to 2D rink image
        rink_coordinates_path: Path to rink coordinates JSON
        output_dir: Directory to save outputs
    """
    # Load the input frame
    frame = cv2.imread(frame_path)
    if frame is None:
        logger.error(f"Error: Could not load frame {frame_path}")
        return
    
    logger.info(f"Processing frame {frame_number} from {frame_path}")
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    logger.info(f"Frame dimensions: {frame_width}x{frame_height}")
    
    # Initialize segmentation processor
    segmentation_model = "models/segmentation.pt"
    if not os.path.exists(segmentation_model):
        logger.error(f"Segmentation model not found at {segmentation_model}")
        return
        
    segmentation_processor = SegmentationProcessor(
        model_path=segmentation_model
    )
    
    # Initialize homography calculator
    homography_calculator = HomographyCalculator(
        rink_coordinates_path=rink_coordinates_path,
        broadcast_width=frame_width,
        broadcast_height=frame_height
    )
    
    # Process the frame to get segmentation features
    logger.info(f"Running segmentation on frame {frame_number}")
    segmentation_result = segmentation_processor.process_frame(frame)
    
    # Log extracted features
    if "features" in segmentation_result:
        features = segmentation_result["features"]
        for feature_type, items in features.items():
            logger.info(f"Found {len(items)} {feature_type} features")
    
    # Create a visualization with goal line points highlighted
    goal_viz = frame.copy()
    if "GoalLine" in segmentation_result["features"]:
        for gl in segmentation_result["features"]["GoalLine"]:
            if "points" in gl and gl["points"]:
                for i, p in enumerate(gl["points"]):
                    x, y = int(p["x"]), int(p["y"])
                    # Draw a larger, more visible circle for the first and last points
                    if i == 0:  # First point (top)
                        cv2.circle(goal_viz, (x, y), 15, (0, 0, 255), -1)  # Red filled circle
                        cv2.putText(goal_viz, f"Goal Top ({x}, {y})", (x + 20, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    elif i == len(gl["points"]) - 1:  # Last point (bottom)
                        cv2.circle(goal_viz, (x, y), 15, (255, 0, 0), -1)  # Blue filled circle
                        cv2.putText(goal_viz, f"Goal Bottom ({x}, {y})", (x + 20, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    else:
                        cv2.circle(goal_viz, (x, y), 5, (0, 255, 0), -1)  # Green filled circle
    
    # Save the goal line visualization
    goal_viz_path = os.path.join(output_dir, f"frame_{frame_number}_goal_viz.jpg")
    cv2.imwrite(goal_viz_path, goal_viz)
    logger.info(f"Saved goal line visualization to {goal_viz_path}")
    
    # Extract source points
    source_points = homography_calculator.extract_source_points(segmentation_result["features"])
    logger.info(f"Extracted {len(source_points)} source points")
    
    # Get destination points
    dest_points = homography_calculator.get_destination_points()
    logger.info(f"Have {len(dest_points)} destination points")
    
    # Collect matching points
    source_pts = []
    dest_pts = []
    matched_names = []
    
    for name, point in source_points.items():
        if name in dest_points:
            source_pts.append(point)
            dest_pts.append(dest_points[name])
            matched_names.append(name)
    
    logger.info(f"Found {len(matched_names)} matching points: {matched_names}")
    
    if len(matched_names) >= 4:
        # Convert to numpy arrays
        source_pts = np.array(source_pts, dtype=np.float32)
        dest_pts = np.array(dest_pts, dtype=np.float32)
        
        try:
            # Calculate homography matrix using RANSAC
            matrix, mask = cv2.findHomography(
                source_pts, dest_pts, cv2.RANSAC, 5.0
            )

            # Log RANSAC results
            logger.info("RANSAC Results:")
            logger.info(f"Number of points: {len(source_pts)}")
            inliers = 0
            for i, ((sx, sy), (dx, dy), m) in enumerate(zip(source_pts, dest_pts, mask)):
                inlier_status = "INLIER" if m == 1 else "OUTLIER"
                logger.info(f"Point {i} ({matched_names[i]}): Source({sx:.1f}, {sy:.1f}) -> Dest({dx:.1f}, {dy:.1f}) - {inlier_status}")
                if m == 1:
                    inliers += 1
            logger.info(f"Total inliers: {inliers}/{len(mask)}")
            
            # Validate the homography matrix
            valid = homography_calculator.validate_homography(matrix)
            logger.info(f"Homography validation result: {'VALID' if valid else 'INVALID'}")
            
            if valid:
                # Apply homography to create warped frame
                rink_img = cv2.imread(rink_image_path)
                if rink_img is None:
                    logger.error(f"Could not load rink image {rink_image_path}")
                else:
                    warped_frame = cv2.warpPerspective(
                        frame, matrix, (rink_img.shape[1], rink_img.shape[0])
                    )
                    
                    # Save warped frame
                    output_path = os.path.join(output_dir, f"frame_{frame_number}_warped.jpg")
                    cv2.imwrite(output_path, warped_frame)
                    logger.info(f"Saved warped frame to {output_path}")
                    
                    # Create overlay for visualization
                    alpha = 0.5
                    overlay = rink_img.copy()
                    cv2.addWeighted(warped_frame, alpha, rink_img, 1-alpha, 0, overlay)
                    
                    # Save overlay
                    overlay_path = os.path.join(output_dir, f"frame_{frame_number}_overlay.jpg")
                    cv2.imwrite(overlay_path, overlay)
                    logger.info(f"Saved overlay to {overlay_path}")
            
        except Exception as e:
            logger.error(f"Error calculating homography: {str(e)}")
    else:
        logger.error(f"Not enough matching points for homography. Need at least 4, found {len(matched_names)}")
    
    # Create and save visualization with detected features
    vis_frame = homography_calculator.draw_visualization(
        frame.copy(), 
        segmentation_result["features"]
    )
    
    vis_path = os.path.join(output_dir, f"frame_{frame_number}_segmentation.jpg")
    cv2.imwrite(vis_path, vis_frame)
    logger.info(f"Saved feature visualization to {vis_path}")

def main():
    parser = argparse.ArgumentParser(description='Debug homography on specific frames')
    parser.add_argument('--frame520', required=True, help='Path to frame 520')
    parser.add_argument('--frame525', required=True, help='Path to frame 525')
    parser.add_argument('--rink-image', required=True, help='Path to rink image')
    parser.add_argument('--rink-coordinates', required=True, help='Path to rink coordinates JSON')
    parser.add_argument('--output-dir', required=True, help='Directory to save debug output')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process both frames
    process_frame(args.frame520, 520, args.rink_image, args.rink_coordinates, args.output_dir)
    process_frame(args.frame525, 525, args.rink_image, args.rink_coordinates, args.output_dir)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main() 