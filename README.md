# Hockey Player Tracking System

This system processes hockey broadcast footage to track player positions on a standardized 2D rink visualization. The system integrates three main components:

1. **Rink segmentation** using the segmentation.pt model
2. **Player detection** using the detection.pt model 
3. **Player orientation** detection using the orient.pth model

## Quick Start

To run the player tracking on a short video clip:

```bash
# Make the run script executable
chmod +x run_tracking.sh

# Run the tracking on a clip (processes 60 frames by default)
./run_tracking.sh

# Process more frames
./run_tracking.sh --max_frames 200

# Process a longer sequence
./run_tracking.sh --max_frames 900
```

This will:
1. Resize the NHL rink image to 1400x600
2. Process frames from the input video
3. Generate visualizations and tracking data
4. Create an HTML visualization for easy review

## Features

### Key Capabilities

- **Rink Feature Detection**: Identifies blue lines, center line, goal lines, and faceoff circles
- **Player Detection**: Locates all players in the broadcast footage
- **Player Orientation**: Determines which direction players are facing
- **Homography Calculation**: Maps between broadcast view and 2D rink coordinates
- **Two-Pass Homography Interpolation**: Ensures smooth camera transitions with accurate tracking
- **Visualization**: Creates quadview visualizations and interactive HTML output

### Recent Improvements

- **Two-Pass Homography Interpolation**: Significantly improves camera transitions by:
  - First identifying frames with successful homography calculation
  - Then properly interpolating between valid matrices for frames where direct calculation fails
  - Providing smooth transitions between different camera views
  - Maintaining tracking accuracy during camera movement

## Project Structure

- `src/` - Source code files
  - `segmentation_processor.py` - Processes frames to identify rink features
  - `player_detector.py` - Detects players in frames
  - `orientation_detector.py` - Determines player orientation
  - `homography_calculator.py` - Maps broadcast coordinates to rink coordinates
  - `player_tracker.py` - Integrates all components and manages homography interpolation
  - `process_video.py` - Processes full videos
  - `process_clip.py` - Processes short clips (for testing)
  - `resize_rink_image.py` - Utility to resize the rink image
  - `generate_quadview.py` - Creates visualizations of tracking results
- `models/` - Trained models
  - `segmentation.pt` - YOLOv8 model for rink feature segmentation
  - `detection.pt` - YOLOv8 model for player detection
  - `orient.pth` - Model for player orientation
- `data/` - Input data
  - `videos/` - Input hockey videos
  - `rink_coordinates.json` - Coordinates of key rink features
  - `nhl-rink.png` - Original rink image
  - `rink_resized.png` - Resized rink image (created by initialize_project.sh)
- `output/` - Processing results
  - `tracking_results_<timestamp>/` - Output directory for each run
    - `frames/` - Extracted video frames
    - `quadview/` - Visualization images
    - `player_detection_data_<timestamp>.json` - Tracking data
    - `visualization.html` - Interactive visualization

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
  --frame-step [FRAME_STEP] \
  --max-frames [MAX_FRAMES]
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

## Output Files

The system generates:

1. **Frame Images**: Individual frames extracted from the video
2. **Quadview Visualizations**: 2x2 grid showing:
   - Original video frame
   - Segmented rink features
   - 2D rink with player positions
   - Combined visualization
3. **JSON Tracking Data**: Complete tracking information including:
   - Frame metadata (timestamp, index)
   - Player detections (position, orientation, confidence)
   - Homography matrices
   - Homography source information (original, fallback, or interpolated)
4. **HTML Visualization**: Interactive web page to view and navigate results

## Processing Pipeline

### Step-by-Step Process

1. **Frame Extraction**: Frames are extracted from the video at specified intervals
2. **Rink Feature Segmentation**: The YOLOv8 segmentation model identifies key rink features
3. **Source Point Extraction**: Points are extracted from segmented features
4. **Camera Zone Detection**: The system detects which part of the rink is visible (left, right, center)
5. **Destination Point Adjustment**: 2D rink coordinates are adjusted based on the camera zone
6. **Homography Calculation**: The transformation matrix is calculated between broadcast and rink views
7. **Player Detection**: Players are identified in the broadcast frame
8. **Orientation Detection**: Player facing direction is determined
9. **Position Mapping**: Player positions are mapped to the 2D rink using the homography matrix
10. **Two-Pass Homography Interpolation**: After all frames are processed, a second pass interpolates homography matrices for frames where direct calculation failed
11. **Visualization**: Results are rendered in various formats
12. **Output Generation**: Data is saved to JSON and HTML output

### Two-Pass Homography Interpolation

A key feature of this system is the two-pass approach to homography calculation:

#### First Pass
During initial frame processing:
- Each frame attempts to calculate a homography matrix directly
- Successful calculations are marked as `homography_source="original"`
- When calculation fails, the system uses the most recent valid matrix as fallback
- These fallback matrices are marked as `homography_source="fallback"`

#### Second Pass
After all frames are processed:
1. The system identifies all frames with original homography matrices
2. For each frame with a fallback matrix:
   - Find the nearest frames with original matrices before and after it
   - Calculate interpolation factor (t) based on temporal position
   - Interpolate: H = (1-t) × H_before + t × H_after
   - Update the frame's homography matrix
   - Mark as `homography_source="interpolated"`

This results in much smoother camera transitions and more accurate player tracking.

## Installation Requirements

### Prerequisites

- Python 3.7+
- CUDA-capable GPU recommended for faster processing

### Dependencies

```bash
# Install required packages
pip install opencv-python numpy torch torchvision ultralytics matplotlib
```

## Troubleshooting

### Common Issues

1. **"No homography matrix" warnings**
   - Ensure the video has clear views of rink features
   - Try increasing the number of frames to process (more frames = more chances for successful homography)
   - Check that the rink_coordinates.json file matches your video

2. **Black or distorted quadview output**
   - Verify that the homography calculation is successful
   - Look for consistent misclassification of rink features
   - Check the logs for interpolation success rate

3. **Slow processing**
   - Processing is GPU-intensive - use a system with CUDA support
   - Reduce the number of frames or increase frame step
   - Close other GPU-intensive applications

4. **Player detection issues**
   - Ensure players are clearly visible in the footage
   - Adjust the confidence threshold in the code if needed
   - Player detections near the edges of the frame may be less reliable

## Future Development

The system is approximately 80% complete, with the following areas for continued development:

1. **Player Identity Persistence**: Maintaining player identity across frames
2. **Team Classification**: Identifying which team each player belongs to
3. **Jersey Number Recognition**: Identifying individual players by jersey number
4. **Performance Optimization**: Improving processing speed for real-time applications
5. **Advanced Analytics**: Implementing heatmaps, speed tracking, and formation analysis

## Contributing

Contributions to the project are welcome! Key areas for improvement include:

1. **Homography Extrapolation**: For end-of-sequence frames without "after" matrices
2. **Feature Classification**: Improving the accuracy of rink feature classification
3. **Player Tracking**: Implementing multi-frame player tracking algorithms
4. **Visualization Enhancements**: Creating more interactive and informative visualizations
5. **Performance Optimization**: Improving processing speed and efficiency

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Ultralytics for YOLOv8 implementation
- PyTorch for deep learning framework
- The hockey analytics community for inspiration and use cases