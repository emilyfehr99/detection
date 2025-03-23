#!/bin/bash

# Hockey Player Tracking System - Initialization Script
# This script sets up the project structure and prepares everything for first use

echo "Initializing Hockey Player Tracking System..."

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/videos
mkdir -p models
mkdir -p output
mkdir -p documentation

# Copy essential files from original project if not already present
echo "Copying model files if needed..."
if [ ! -f "models/segmentation.pt" ]; then
  cp ../GoodCodeBuf/models/segmentation.pt models/
fi

if [ ! -f "models/detection.pt" ]; then
  cp ../GoodCodeBuf/models/detection.pt models/
fi

if [ ! -f "models/orient.pth" ]; then
  cp ../GoodCodeBuf/models/orient.pth models/
fi

echo "Copying rink data if needed..."
if [ ! -f "data/rink_coordinates.json" ]; then
  cp ../GoodCodeBuf/data/rink_coordinates.json data/
fi

if [ ! -f "data/nhl-rink.png" ]; then
  cp ../GoodCodeBuf/data/nhl-rink.png data/
fi

# Create a resized rink image
echo "Creating resized rink image..."
python src/resize_rink_image.py \
  --input data/nhl-rink.png \
  --output data/rink_resized.png \
  --width 1400 \
  --height 600

# Create symbolic links to test video if needed
echo "Setting up test video..."
if [ ! -f "data/videos/PIT_vs_CHI_2016_2.mp4" ]; then
  ln -sf ../../GoodCodeBuf/data/videos/PIT_vs_CHI_2016_2.mp4 data/videos/
fi

# Ensure run script is executable
chmod +x run_tracking.sh

echo "Initialization complete! You can now run the tracking system with:"
echo "./run_tracking.sh"
