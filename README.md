# Hockey Player Tracking System

This system processes hockey broadcast footage to track player positions on a standardized 2D rink visualization. The system integrates three main components:

1. **Rink segmentation** using the segmentation.pt model
2. **Player detection** using the detection.pt model 
3. **Player orientation** detection using the orient.pth model

## Project Structure

- `src/` - Source code files
  - `segmentation_processor.py` - Processes frames to identify rink features
  - `player_detector.py` - Detects players in frames
  - `orientation_detector.py` - Determines player orientation
  - `homography_calculator.py` - Maps broadcast coordinates to rink coordinates
  - `player_tracker.py` - Integrates all components
  - `process_video.py` - Processes full videos
  - `process_clip.py` - Processes short clips (for testing)
  - `resize_rink_image.py` - Utility to resize the rink image
- `models/` - Trained models
- `data/` - Input data
- `output/` - Processing results
- `documentation/` - Additional documentation

## Quick Start

To run the player tracking on a short video clip:

```bash
# Make the run script executable
chmod +x run_tracking.sh

# Run the tracking on a clip
./run_tracking.sh
```

This will:
1. Resize the NHL rink image to 1400x600
2. Process a 5-second clip from the input video
3. Generate visualizations and tracking data
4. Create an HTML visualization for easy review

## Command Line Options

For more control, you can run the scripts directly:

### Processing a Short Clip

```bash
python src/process_clip.py \
  --video [VIDEO_PATH] \
  --segmentation-model [SEGMENTATION_MODEL_PATH] \
  --detection-model [DETECTION_MODEL_PATH] \
  --orientation-model [ORIENTATION_MODEL_PATH] \
  --rink-coordinates [RINK_COORDINATES_PATH] \
  --rink-image [RINK_IMAGE_PATH] \
  --output-dir [OUTPUT_DIR] \
  --start-second [START_TIME] \
  --num-seconds [DURATION] \
  --frame-step [FRAME_STEP]
```

### Processing a Full Video

```bash
python src/process_video.py \
  --video [VIDEO_PATH] \
  --segmentation-model [SEGMENTATION_MODEL_PATH] \
  --detection-model [DETECTION_MODEL_PATH] \
  --orientation-model [ORIENTATION_MODEL_PATH] \
  --rink-coordinates [RINK_COORDINATES_PATH] \
  --rink-image [RINK_IMAGE_PATH] \
  --output-dir [OUTPUT_DIR] \
  --start-frame [START_FRAME] \
  --end-frame [END_FRAME] \
  --frame-step [FRAME_STEP]
```

## Output

The system generates:

1. **Broadcast visualizations**: Original video with player detections and orientations
2. **Rink visualizations**: 2D rink model with mapped player positions
3. **Side-by-side visualizations**: Combined view of both visualizations
4. **Tracking data**: JSON file with all player tracking information
5. **HTML visualization**: Interactive web page to view results (for process_clip.py)

## Processing Pipeline

1. Frames are extracted from the video
2. The segmentation model identifies rink features
3. These features are used to calculate the homography matrix
4. The player detection model identifies players in the frame
5. The orientation model determines player facing direction
6. Player positions are mapped to the 2D rink using the homography matrix
7. Results are visualized and tracking data is saved
