#!/bin/bash

# Hockey Player Tracking Run Script
# This script simplifies running the player tracking system on a video clip

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
VIDEO_PATH="data/videos/CAN-SWE.mp4"
SEGMENTATION_MODEL="models/segmentation.pt"
DETECTION_MODEL="models/detection.pt"
ORIENTATION_MODEL="models/orient.pth"
RINK_COORDINATES="data/rink_coordinates.json"
RINK_IMAGE="data/rink_resized.png"
OUTPUT_DIR="output/tracking_results_$(date +%Y%m%d_%H%M%S)"
START_SECOND=0
NUM_SECONDS=0  # 0 means process entire video
FRAME_STEP=1
MAX_FRAMES=${MAX_FRAMES:-50}  # Default to 50 frames

# Create output directory
mkdir -p $OUTPUT_DIR

# First, resize the rink image (already done in initialize_project.sh, but let's ensure it's updated)
echo "Ensuring rink image is properly sized..."
python3 src/resize_rink_image.py \
  --input data/nhl-rink.png \
  --output $RINK_IMAGE \
  --width 1400 \
  --height 600

# Run the player tracking on a clip
echo "Running player tracking on clip..."
python3 src/process_clip.py \
  --video $VIDEO_PATH \
  --segmentation-model $SEGMENTATION_MODEL \
  --detection-model $DETECTION_MODEL \
  --orientation-model $ORIENTATION_MODEL \
  --rink-coordinates $RINK_COORDINATES \
  --rink-image $RINK_IMAGE \
  --output-dir $OUTPUT_DIR \
  --start-second $START_SECOND \
  --num-seconds $NUM_SECONDS \
  --frame-step $FRAME_STEP \
  --max-frames $MAX_FRAMES

# Wait for the tracking data file to be created and get its name
TRACKING_DATA=$(ls $OUTPUT_DIR/player_detection_data_*.json | head -n 1)

# Generate quadview visualizations from the tracking results
echo "Generating quadview visualizations..."
mkdir -p $OUTPUT_DIR/quadview
python3 src/generate_quadview.py \
  --tracking-data "$TRACKING_DATA" \
  --rink-image $RINK_IMAGE \
  --rink-coordinates $RINK_COORDINATES \
  --output-dir $OUTPUT_DIR/quadview

# Create a special debug quadview for the first frame
FIRST_FRAME="$OUTPUT_DIR/frames/frame_0.jpg"
if [ -f "$FIRST_FRAME" ]; then
  echo "Creating debug quadview for first frame..."
  python3 src/create_quadview.py \
    --input-frame $FIRST_FRAME \
    --rink-image $RINK_IMAGE \
    --rink-coordinates $RINK_COORDINATES \
    --segmentation-model $SEGMENTATION_MODEL \
    --output $OUTPUT_DIR/first_frame_quadview.jpg
fi

echo "Processing complete! Results are in $OUTPUT_DIR"
echo "Open the HTML visualization at $OUTPUT_DIR/visualization.html"
echo "Quadview visualizations are in $OUTPUT_DIR/quadview/"
