#!/bin/bash

# Hockey Player Tracking Run Script
# This script simplifies running the player tracking system on a video clip

# Default values
VIDEO_PATH="data/videos/PIT_vs_CHI_2016_2.mp4"
SEGMENTATION_MODEL="models/segmentation.pt"
DETECTION_MODEL="models/detection.pt"
ORIENTATION_MODEL="models/orient.pth"
RINK_COORDINATES="data/rink_coordinates.json"
RINK_IMAGE="data/rink_resized.png"
OUTPUT_DIR="output/tracking_results_$(date +%Y%m%d_%H%M%S)"
START_SECOND=10
NUM_SECONDS=5
FRAME_STEP=5

# Create output directory
mkdir -p $OUTPUT_DIR

# First, resize the rink image (already done in initialize_project.sh, but let's ensure it's updated)
echo "Ensuring rink image is properly sized..."
python src/resize_rink_image.py \
  --input data/nhl-rink.png \
  --output $RINK_IMAGE \
  --width 1400 \
  --height 600

# Run the player tracking on a clip
echo "Running player tracking on clip..."
python src/process_clip.py \
  --video $VIDEO_PATH \
  --segmentation-model $SEGMENTATION_MODEL \
  --detection-model $DETECTION_MODEL \
  --orientation-model $ORIENTATION_MODEL \
  --rink-coordinates $RINK_COORDINATES \
  --rink-image $RINK_IMAGE \
  --output-dir $OUTPUT_DIR \
  --start-second $START_SECOND \
  --num-seconds $NUM_SECONDS \
  --frame-step $FRAME_STEP

echo "Processing complete! Results are in $OUTPUT_DIR"
echo "Open the HTML visualization at $OUTPUT_DIR/visualization.html"
