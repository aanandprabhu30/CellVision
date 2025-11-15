"""
CellVision - Core Analysis Module
Combines CellPose segmentation with GPT-4 Vision for microscopy analysis
"""

from cellpose import models
from skimage import io
from skimage.measure import regionprops_table
import numpy as np
import matplotlib.pyplot as plt
import base64
from openai import OpenAI


# Initialize CellPose model (loaded once, reused for all analyses)
model = None

def get_cellpose_model():
    """Get or initialize CellPose model"""
    global model
    if model is None:
        print("Loading CellPose model...")
        model = models.CellposeModel(gpu=True, model_type='cyto2')
    return model


def analyze_microscopy_image(image_path):
    """
    Complete microscopy image analysis pipeline.

    Args:
        image_path: Path to microscopy image

    Returns:
        masks: Segmentation masks for each cell
        metrics: Dictionary of quantitative measurements
    """
    # 1. Load and preprocess
    img = io.imread(image_path)

    # 2. Cell segmentation with CellPose
    model = get_cellpose_model()
    masks, flows, styles = model.eval(img, diameter=30)

    # 3. Extract quantitative metrics
    cell_count = len(np.unique(masks)) - 1

    # 4. Morphology analysis
    if cell_count > 0:
        props = regionprops_table(
            masks, img,
            properties=['area', 'perimeter', 'eccentricity', 'solidity']
        )

        # 5. Calculate statistics
        areas = props['area']
        perimeters = props['perimeter']

        # Calculate circularity (4π*area/perimeter²)
        circularities = 4 * np.pi * areas / (perimeters ** 2)

        metrics = {
            'total_cells': cell_count,
            'avg_area': float(np.mean(areas)),
            'avg_circularity': float(np.mean(circularities)),
            'density': cell_count / (img.shape[0] * img.shape[1]),
            'size_variation': float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else 0
        }
    else:
        metrics = {
            'total_cells': 0,
            'avg_area': 0,
            'avg_circularity': 0,
            'density': 0,
            'size_variation': 0
        }

    return masks, metrics


def generate_analysis_narrative(image_path, masks, metrics, api_key):
    """
    Generate publication-quality figure legend using GPT-4 Vision.

    Args:
        image_path: Path to original image
        masks: Segmentation masks from CellPose
        metrics: Quantitative metrics dictionary
        api_key: OpenAI API key

    Returns:
        str: Publication-ready figure legend
    """
    # 1. Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(io.imread(image_path))
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(masks, cmap='tab20')
    ax2.set_title(f'Segmented: {metrics["total_cells"]} cells')
    ax2.axis('off')
    plt.savefig('analysis.png', dpi=100, bbox_inches='tight')
    plt.close()

    # 2. Encode image for GPT-4 Vision
    with open('analysis.png', 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')

    # 3. Call GPT-4 Vision with metrics context
    client = OpenAI(api_key=api_key)

    prompt = f"""You are an expert cell biologist. Analyze this microscopy image.

Quantitative metrics detected:
- Cell count: {metrics['total_cells']}
- Average cell area: {metrics['avg_area']:.1f} pixels²
- Cell density: {metrics['density']:.4f} cells/pixel²
- Size variation coefficient: {metrics['size_variation']:.2f}
- Average circularity: {metrics['avg_circularity']:.2f}

Generate a publication-quality figure legend describing:
1. Cell morphology and distribution patterns
2. Notable features or abnormalities
3. Quantitative summary with the provided statistics

Format as: "Figure X: [Comprehensive scientific description]"
Keep it concise but scientifically precise."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }}
            ]
        }],
        max_tokens=500
    )

    return response.choices[0].message.content
