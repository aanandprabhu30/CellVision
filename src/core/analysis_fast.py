"""
CellVision - Fast Analysis Module
Optimized for speed with model caching and simplified processing
"""

from cellpose import models
from skimage import io, exposure, filters
from skimage.measure import regionprops_table
import numpy as np
import matplotlib.pyplot as plt
import base64
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# Global model cache - loads once, reuses forever
_MODEL_CACHE = None


def get_model_cached():
    """Get or create cached CellPose model"""
    global _MODEL_CACHE
    
    if _MODEL_CACHE is None:
        print("📥 Loading CellPose model (first time only)...")
        _MODEL_CACHE = models.Cellpose(gpu=False, model_type='cyto2')
        print("✅ Model loaded and cached!")
    
    return _MODEL_CACHE


def quick_preprocess(img):
    """Fast preprocessing"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    
    # Simple contrast enhancement
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    return img


def calculate_health_fast(area, circularity, solidity):
    """Fast health calculation"""
    # Normalize metrics
    area_score = min(100, (area / 500) * 100) if area < 500 else 100
    circ_score = circularity * 100
    solid_score = solidity * 100
    
    # Weighted average
    health = (area_score * 0.3 + circ_score * 0.4 + solid_score * 0.3)
    
    return max(0, min(100, health))


def classify_morphology_fast(health):
    """Fast morphology classification"""
    if health >= 80:
        return "Healthy"
    elif health >= 60:
        return "Stressed"
    elif health >= 40:
        return "Apoptotic"
    else:
        return "Irregular"


def analyze_fast(image_path, diameter=60):
    """
    Fast analysis pipeline - optimized for speed
    
    Args:
        image_path: Path to image
        diameter: Estimated cell diameter (default 60px)
    
    Returns:
        masks, metrics, cell_data
    """
    print("🔬 Starting fast analysis...")
    
    # 1. Load image
    print("📂 Loading image...")
    img = io.imread(image_path)
    
    # 2. Quick preprocessing
    print("🎨 Preprocessing...")
    img_processed = quick_preprocess(img)
    
    # 3. Get cached model
    model = get_model_cached()
    
    # 4. Segment cells
    print(f"🔍 Segmenting cells (diameter={diameter}px)...")
    masks, flows, styles = model.eval(
        img_processed,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    
    # 5. Count cells
    cell_count = len(np.unique(masks)) - 1
    print(f"✅ Found {cell_count} cells!")
    
    if cell_count == 0:
        return masks, {
            'total_cells': 0,
            'avg_area': 0,
            'avg_health_score': 0,
            'healthy_percentage': 0
        }, []
    
    # 6. Extract properties
    print("📊 Calculating metrics...")
    props = regionprops_table(
        masks,
        img_processed,
        properties=['area', 'perimeter', 'solidity']
    )
    
    # 7. Calculate metrics
    areas = props['area']
    perimeters = props['perimeter']
    solidities = props['solidity']
    
    # Circularity
    circularities = 4 * np.pi * areas / (perimeters ** 2 + 1e-10)
    
    # Health scores
    health_scores = []
    morphologies = []
    
    for i in range(len(areas)):
        health = calculate_health_fast(
            areas[i],
            circularities[i],
            solidities[i]
        )
        health_scores.append(health)
        morphologies.append(classify_morphology_fast(health))
    
    # 8. Compile metrics
    healthy_count = sum(1 for h in health_scores if h >= 75)
    
    metrics = {
        'total_cells': cell_count,
        'avg_area': float(np.mean(areas)),
        'avg_health_score': float(np.mean(health_scores)),
        'healthy_cells': healthy_count,
        'healthy_percentage': (healthy_count / cell_count * 100) if cell_count > 0 else 0,
        'avg_circularity': float(np.mean(circularities)),
        'avg_solidity': float(np.mean(solidities))
    }
    
    # 9. Per-cell data
    cell_data = []
    for i in range(len(areas)):
        cell_data.append({
            'cell_id': i + 1,
            'area': float(areas[i]),
            'health_score': float(health_scores[i]),
            'morphology': morphologies[i],
            'circularity': float(circularities[i]),
            'solidity': float(solidities[i])
        })
    
    print("✅ Analysis complete!")
    return masks, metrics, cell_data


def generate_narrative_fast(image_path, masks, metrics, cell_data, api_key):
    """Generate AI narrative with optimized prompt"""
    
    if not api_key or not api_key.startswith('sk-'):
        return "No API key provided"
    
    try:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original
        img = io.imread(image_path)
        ax1.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax1.set_title('Original')
        ax1.axis('off')
        
        # Segmentation
        ax2.imshow(masks, cmap='nipy_spectral')
        ax2.set_title(f'Segmented: {metrics["total_cells"]} cells')
        ax2.axis('off')
        
        # Save to base64
        plt.tight_layout()
        temp_path = f"temp_viz_{np.random.randint(10000)}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        with open(temp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Optimized prompt
        prompt = f"""Analyze this microscopy image with {metrics['total_cells']} cells detected.

Key Metrics:
- Total cells: {metrics['total_cells']}
- Average health score: {metrics['avg_health_score']:.1f}/100
- Healthy cells: {metrics['healthy_percentage']:.0f}%
- Average area: {metrics['avg_area']:.0f} pixels²

Write a concise, professional figure legend (2-3 sentences) describing:
1. What you see in the image
2. The cell population health status
3. One notable observation

Be specific and scientific."""

        # Call GPT-4
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Latest and best OpenAI model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "low"  # Faster processing
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI narration failed: {str(e)}"
