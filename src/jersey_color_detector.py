#!/usr/bin/env python3
"""
Jersey Color Detector for Team Identification
Detects player jersey colors and assigns teams based on color clustering.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class JerseyColorDetector:
    """
    Detects jersey colors for team identification in hockey video.
    Integrates with existing player tracking system without modifying core functionality.
    """
    
    def __init__(self):
        """Initialize the jersey color detector."""
        # Define common NHL team colors (RGB format)
        self.team_color_definitions = {
            "red": {
                "name": "Team A",
                "colors": [(255, 0, 0), (200, 0, 0), (150, 0, 0)],  # Red variations
                "hsv_range": [(0, 100, 50), (10, 255, 255)]  # Red HSV range
            },
            "blue": {
                "name": "Team B", 
                "colors": [(0, 0, 255), (0, 0, 200), (0, 0, 150)],  # Blue variations
                "hsv_range": [(100, 100, 50), (130, 255, 255)]  # Blue HSV range
            },
            "yellow": {
                "name": "Team A",
                "colors": [(255, 255, 0), (200, 200, 0), (150, 150, 0)],  # Yellow variations
                "hsv_range": [(20, 100, 50), (30, 255, 255)]  # Yellow HSV range
            },
            "green": {
                "name": "Team B",
                "colors": [(0, 255, 0), (0, 200, 0), (0, 150, 0)],  # Green variations
                "hsv_range": [(40, 100, 50), (80, 255, 255)]  # Green HSV range
            },
            "orange": {
                "name": "Team A",
                "colors": [(255, 165, 0), (200, 130, 0), (150, 100, 0)],  # Orange variations
                "hsv_range": [(10, 100, 50), (25, 255, 255)]  # Orange HSV range
            },
            "purple": {
                "name": "Team B",
                "colors": [(128, 0, 128), (100, 0, 100), (75, 0, 75)],  # Purple variations
                "hsv_range": [(130, 100, 50), (160, 255, 255)]  # Purple HSV range
            },
            "brown": {
                "name": "Team A",
                "colors": [(165, 42, 42), (130, 33, 33), (100, 25, 25)],  # Brown variations
                "hsv_range": [(0, 50, 20), (20, 255, 200)]  # Brown HSV range
            },
            "black": {
                "name": "Team B",
                "colors": [(0, 0, 0), (20, 20, 20), (40, 40, 40)],  # Black variations
                "hsv_range": [(0, 0, 0), (180, 255, 50)]  # Black HSV range
            },
            "white": {
                "name": "Team A",
                "colors": [(255, 255, 255), (240, 240, 240), (220, 220, 220)],  # White variations
                "hsv_range": [(0, 0, 200), (180, 30, 255)]  # White HSV range
            }
        }
        
        # Initialize color clustering parameters
        self.n_clusters = 5  # Number of dominant colors to extract
        self.min_color_area = 100  # Minimum area for color analysis
        
    def extract_jersey_region(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract the jersey region from a player bounding box.
        
        Args:
            frame: Input video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Jersey region as numpy array
        """
        try:
            x1, y1, x2, y2 = bbox
            # Ensure bbox values are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Focus on upper body region (jersey area)
            height = y2 - y1
            jersey_start = y1 + int(height * 0.2)  # Start 20% down from top
            jersey_end = y1 + int(height * 0.7)    # End 70% down from top
            
            jersey_region = frame[jersey_start:jersey_end, x1:x2]
            return jersey_region
        except Exception as e:
            logger.warning(f"Error extracting jersey region: {e}")
            try:
                # Fallback to full bbox with integer conversion
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                return frame[y1:y2, x1:x2]
            except:
                # Final fallback - return empty array
                return np.array([])
    
    def extract_dominant_colors(self, image: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Extract dominant colors from an image using K-means clustering.
        
        Args:
            image: Input image
            
        Returns:
            List of dominant RGB colors
        """
        try:
            # Reshape image for clustering
            pixels = image.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Sort by cluster size (most dominant first)
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            
            dominant_colors = [tuple(colors[i]) for i in sorted_indices]
            
            return dominant_colors
            
        except Exception as e:
            logger.error(f"Error in color extraction: {e}")
            return []
    
    def normalize_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Normalize color values for better comparison.
        
        Args:
            color: RGB color tuple
            
        Returns:
            Normalized RGB color tuple
        """
        r, g, b = color
        
        # Normalize to account for lighting variations
        total = r + g + b
        if total > 0:
            r_norm = int((r / total) * 255)
            g_norm = int((g / total) * 255)
            b_norm = int((b / total) * 255)
            return (r_norm, g_norm, b_norm)
        
        return color
    
    def calculate_hsv_similarity(self, color: Tuple[int, int, int], hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> float:
        """
        Calculate HSV similarity between a color and a color range.
        
        Args:
            color: RGB color tuple
            hsv_range: HSV range (min_hsv, max_hsv)
            
        Returns:
            Similarity score (0-1)
        """
        try:
            # Convert RGB to HSV
            color_array = np.array([[color]], dtype=np.uint8)
            hsv_color = cv2.cvtColor(color_array, cv2.COLOR_RGB2HSV)[0][0]
            
            min_hsv, max_hsv = hsv_range
            
            # Check if color falls within HSV range
            if (min_hsv[0] <= hsv_color[0] <= max_hsv[0] and
                min_hsv[1] <= hsv_color[1] <= max_hsv[1] and
                min_hsv[2] <= hsv_color[2] <= max_hsv[2]):
                return 1.0
            
            # Calculate distance to range boundaries
            h_dist = min(abs(hsv_color[0] - min_hsv[0]), abs(hsv_color[0] - max_hsv[0]))
            s_dist = min(abs(hsv_color[1] - min_hsv[1]), abs(hsv_color[1] - max_hsv[1]))
            v_dist = min(abs(hsv_color[2] - min_hsv[2]), abs(hsv_color[2] - max_hsv[2]))
            
            # Normalize distances
            max_h_dist = 180  # Hue range
            max_s_dist = 255  # Saturation range
            max_v_dist = 255  # Value range
            
            total_dist = (h_dist / max_h_dist) + (s_dist / max_s_dist) + (v_dist / max_v_dist)
            similarity = max(0, 1 - (total_dist / 3))
            
            return similarity
            
        except Exception as e:
            logger.warning(f"Error calculating HSV similarity: {e}")
            return 0.0
    
    def match_colors_to_team(self, colors: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """
        Match extracted colors to team definitions.
        
        Args:
            colors: List of RGB colors
            
        Returns:
            Dictionary mapping team names to confidence scores
        """
        team_scores = {}
        
        try:
            for team_name, team_info in self.team_color_definitions.items():
                total_similarity = 0.0
                valid_matches = 0
                
                for color in colors:
                    best_similarity = 0.0
                    
                    # Check RGB similarity
                    for team_color in team_info["colors"]:
                        # Calculate Euclidean distance
                        distance = np.linalg.norm(np.array(color) - np.array(team_color))
                        max_distance = np.sqrt(3 * 255**2)  # Maximum possible distance
                        rgb_similarity = 1 - (distance / max_distance)
                        
                        # Check HSV similarity
                        hsv_similarity = self.calculate_hsv_similarity(color, team_info["hsv_range"])
                        
                        # Combined similarity score (weighted average)
                        combined_similarity = (rgb_similarity * 0.6) + (hsv_similarity * 0.4)
                        
                        if combined_similarity > best_similarity:
                            best_similarity = combined_similarity
                    
                    # Only count matches above threshold
                    if best_similarity > 0.5:
                        total_similarity += best_similarity
                        valid_matches += 1
                
                # Calculate average similarity for this team
                if valid_matches > 0:
                    team_scores[team_info["name"]] = total_similarity / valid_matches
            
            return team_scores
            
        except Exception as e:
            logger.error(f"Error in color matching: {e}")
            return {}
    
    def detect_team(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, any]:
        """
        Detect team for a player based on jersey color.
        
        Args:
            frame: Input video frame
            bbox: Player bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with team information
        """
        try:
            # Extract jersey region
            jersey_region = self.extract_jersey_region(frame, bbox)
            
            if jersey_region.size == 0:
                return {"team": "Unknown", "confidence": 0.0, "method": "no_region"}
            
            # Extract dominant colors
            dominant_colors = self.extract_dominant_colors(jersey_region)
            
            if not dominant_colors:
                return {"team": "Unknown", "confidence": 0.0, "method": "no_colors"}
            
            # Match colors to teams
            team_scores = self.match_colors_to_team(dominant_colors)
            
            if not team_scores:
                return {"team": "Unknown", "confidence": 0.0, "method": "no_matches"}
            
            # Find best team match
            best_team = max(team_scores.items(), key=lambda x: x[1])
            team_name, confidence = best_team
            
            return {
                "team": team_name,
                "confidence": confidence,
                "method": "color_analysis",
                "dominant_colors": dominant_colors[:3],  # Top 3 colors
                "team_scores": team_scores
            }
            
        except Exception as e:
            logger.error(f"Error in team detection: {e}")
            return {"team": "Unknown", "confidence": 0.0, "method": "error"}
    
    def get_team_display_name(self, team_info: Dict[str, any]) -> str:
        """
        Get display name for team detection results.
        
        Args:
            team_info: Team detection results
            
        Returns:
            Display string for team
        """
        if not team_info or team_info.get("team") == "Unknown":
            return "Unknown"
        
        team_name = team_info.get("team", "Unknown")
        confidence = team_info.get("confidence", 0.0)
        
        if confidence > 0.8:
            return f"{team_name} (High)"
        elif confidence > 0.6:
            return f"{team_name} (Medium)"
        elif confidence > 0.4:
            return f"{team_name} (Low)"
        else:
            return f"{team_name} (Very Low)"
