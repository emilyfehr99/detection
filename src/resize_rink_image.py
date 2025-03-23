import cv2
import argparse
import os


def resize_rink_image(input_path: str, output_path: str, width: int = 1400, height: int = 600) -> None:
    """
    Resizes the NHL rink image to the specified dimensions.
    
    Args:
        input_path: Path to the input rink image
        output_path: Path to save the resized image
        width: Target width (default: 1400)
        height: Target height (default: 600)
    """
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load image from {input_path}")
    
    # Resize the image
    resized_img = cv2.resize(img, (width, height))
    
    # Save the resized image
    cv2.imwrite(output_path, resized_img)
    
    print(f"Resized image saved to {output_path}")
    print(f"Original dimensions: {img.shape[1]}x{img.shape[0]}")
    print(f"New dimensions: {width}x{height}")


def main():
    parser = argparse.ArgumentParser(description="Resize NHL rink image to specific dimensions")
    parser.add_argument("--input", type=str, required=True, help="Path to input rink image")
    parser.add_argument("--output", type=str, required=True, help="Path to save resized image")
    parser.add_argument("--width", type=int, default=1400, help="Target width (default: 1400)")
    parser.add_argument("--height", type=int, default=600, help="Target height (default: 600)")
    
    args = parser.parse_args()
    
    resize_rink_image(args.input, args.output, args.width, args.height)


if __name__ == "__main__":
    main()
