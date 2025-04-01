#!/bin/bash

# Hockey Player Orientation and Detection Run Script
# This script runs the orientation and detection models on video frames

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --max_frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Default values
VIDEO_PATH="data/videos/PIT_vs_CHI_2016_2.mp4"
DETECTION_MODEL="models/detection.pt"
ORIENTATION_MODEL="models/orient.pth"
OUTPUT_DIR="output/orientation_detection_$(date +%Y%m%d_%H%M%S)"
START_SECOND=0
NUM_SECONDS=15
FRAME_STEP=5
MAX_FRAMES=${MAX_FRAMES:-900}  # Default to 900 frames

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the orientation and detection models on frames
echo "Running orientation and detection models on frames..."
python src/process_clip.py \
  --video $VIDEO_PATH \
  --detection-model $DETECTION_MODEL \
  --orientation-model $ORIENTATION_MODEL \
  --output-dir $OUTPUT_DIR \
  --start-second $START_SECOND \
  --num-seconds $NUM_SECONDS \
  --frame-step $FRAME_STEP \
  --max-frames $MAX_FRAMES

echo "Processing complete! Results are in $OUTPUT_DIR" 