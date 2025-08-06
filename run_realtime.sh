#!/bin/bash

# Real-time Hockey Analysis Script
# Processes video at full speed with ellipse visualization

echo "🏒 Starting Real-time Hockey Analysis..."
echo "Classes: Player (Green), Puck (Yellow), Stick Blade (Magenta), Goalkeeper (Red), Goal Zone (Orange)"

python3 src/realtime_processor.py \
    --video data/videos/CAN-SWE.mp4 \
    --detection-model models/detection.pt \
    --orientation-model models/orient.pth \
    --output-dir output \
    --segmentation-model models/segmentation.pt \
    --rink-coordinates data/rink_coordinates.json \
    --rink-image data/rink_resized.png \
    --start-second 0 \
    --max-frames 300

echo "✅ Real-time processing complete!"
echo "📹 Output video: output/realtime_analysis.mp4"
echo "📊 Data saved to: output/realtime_data_*.json"