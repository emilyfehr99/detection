import json
import argparse
from pathlib import Path
import pandas as pd


def load_json_data(json_path):
    """Load tracking data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_player_tracking_csv(json_data, output_path):
    """
    Convert player tracking data to CSV format.
    Each row contains: frame_id, timestamp, player_id, x, y, orientation, bbox
    """
    rows = []
    
    # Extract data for each frame and player
    for frame_id, frame_data in json_data.items():
        timestamp = frame_data.get('timestamp', '')
        
        for player in frame_data.get('players', []):
            rink_pos = player.get('rink_position', [None, None])
            bbox = player.get('bbox', [None]*4)
            
            row = {
                'frame_id': frame_data.get('frame_id', frame_id),
                'timestamp': timestamp,
                'player_id': player.get('id', ''),
                'x': rink_pos[0] if rink_pos else None,
                'y': rink_pos[1] if rink_pos else None,
                'orientation': player.get('orientation', None),
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3]
            }
            rows.append(row)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(
        f"Created player tracking CSV at: {output_path}"
    )


def create_homography_csv(json_data, output_path):
    """
    Convert homography data to CSV format.
    Each row contains: frame_id, timestamp, homography_success, homography_matrix
    """
    rows = []
    
    # Extract homography data for each frame
    for frame_id, frame_data in json_data.items():
        row = {
            'frame_id': frame_data.get('frame_id', frame_id),
            'timestamp': frame_data.get('timestamp', ''),
            'homography_success': frame_data.get('homography_success', False)
        }
        
        # Add flattened homography matrix if available
        if 'homography_matrix' in frame_data:
            matrix = frame_data['homography_matrix']
            for i in range(3):
                for j in range(3):
                    row[f'h{i}{j}'] = matrix[i][j] if matrix else None
        
        rows.append(row)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Created homography CSV at: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert player tracking JSON to CSV format'
    )
    parser.add_argument('json_path', help='Path to input JSON file')
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for CSV files'
    )
    parser.add_argument(
        '--type',
        choices=['all', 'tracking', 'homography'],
        default='all',
        help='Type of data to convert to CSV'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load JSON data
    json_data = load_json_data(args.json_path)
    
    # Generate base output filename
    base_name = Path(args.json_path).stem
    
    # Convert data based on specified type
    if args.type in ['all', 'tracking']:
        tracking_path = output_dir / f"{base_name}_tracking.csv"
        create_player_tracking_csv(json_data, tracking_path)
        
    if args.type in ['all', 'homography']:
        homography_path = output_dir / f"{base_name}_homography.csv"
        create_homography_csv(json_data, homography_path)


if __name__ == '__main__':
    main() 