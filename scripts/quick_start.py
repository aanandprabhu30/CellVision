"""
Quick Start Script for CellVision
This script helps you test the complete pipeline
"""

import numpy as np
from PIL import Image
import os

def create_sample_image():
    """Create a simple synthetic cell image for testing"""
    print("Creating sample cell image...")

    # Create a 512x512 grayscale image
    img = np.ones((512, 512), dtype=np.uint8) * 200

    # Add some circular "cells"
    centers = [
        (128, 128), (128, 384), (384, 128), (384, 384),
        (256, 256), (150, 256), (362, 256), (256, 150), (256, 362)
    ]

    for cx, cy in centers:
        for i in range(512):
            for j in range(512):
                dist = np.sqrt((i - cy)**2 + (j - cx)**2)
                if dist < 40:
                    # Create circular cells with some intensity variation
                    intensity = int(100 + 50 * (1 - dist/40))
                    img[i, j] = max(0, min(255, intensity))

    # Save the image
    Image.fromarray(img).save('sample_cells.png')
    print("✓ Sample image created: sample_cells.png")
    return 'sample_cells.png'


def test_cellpose():
    """Test CellPose segmentation"""
    print("\n" + "="*50)
    print("STEP 1: Testing CellPose Installation")
    print("="*50)

    from analysis import analyze_microscopy_image

    # Create sample if it doesn't exist
    if not os.path.exists('sample_cells.png'):
        image_path = create_sample_image()
    else:
        image_path = 'sample_cells.png'

    print(f"Analyzing {image_path}...")
    masks, metrics = analyze_microscopy_image(image_path)

    print("\n✓ CellPose Analysis Complete!")
    print(f"   - Detected: {metrics['total_cells']} cells")
    print(f"   - Avg area: {metrics['avg_area']:.1f} pixels²")
    print(f"   - Density: {metrics['density']:.6f} cells/pixel²")
    print(f"   - Circularity: {metrics['avg_circularity']:.2f}")

    return image_path, masks, metrics


def test_gpt4_vision(api_key):
    """Test GPT-4 Vision integration"""
    print("\n" + "="*50)
    print("STEP 2: Testing GPT-4 Vision Integration")
    print("="*50)

    from analysis import generate_analysis_narrative

    # Get the sample analysis
    image_path, masks, metrics = test_cellpose()

    print("Generating AI narrative (this may take 10-20 seconds)...")
    try:
        narrative = generate_analysis_narrative(image_path, masks, metrics, api_key)
        print("\n✓ GPT-4 Vision Analysis Complete!")
        print("\n" + "="*50)
        print("Generated Figure Legend:")
        print("="*50)
        print(narrative)
        print("="*50)
        return True
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - API key doesn't have GPT-4 Vision access")
        print("  - Network connectivity issues")
        return False


def main():
    print("="*50)
    print("CellVision Quick Start Test")
    print("="*50)

    # Check for API key
    api_key = input("\nEnter your OpenAI API key (or press Enter to skip GPT-4 test): ").strip()

    # Test CellPose
    test_cellpose()

    # Test GPT-4 Vision if API key provided
    if api_key:
        success = test_gpt4_vision(api_key)
        if success:
            print("\n✓ All tests passed! CellVision is ready to use.")
            print("\nNext step: Run the web app with:")
            print("  streamlit run app.py")
    else:
        print("\nSkipping GPT-4 Vision test.")
        print("CellPose is working! To test the complete system:")
        print("  1. Get an OpenAI API key")
        print("  2. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
