"""
CellVision - Enhanced Analysis Module
Advanced cell segmentation and morphological analysis with AI narration
"""

from cellpose import models
from skimage import io, exposure, filters, morphology
from skimage.measure import regionprops_table, regionprops
from scipy import ndimage
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
import base64
from openai import OpenAI
import cv2
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


# Global model cache
_model_cache = None


def get_cellpose_model(use_gpu=False):
    """
    Get or initialize CellPose model with GPU fallback.
    
    Args:
        use_gpu: Whether to attempt GPU usage
        
    Returns:
        CellPose model instance
    """
    global _model_cache
    if _model_cache is None:
        print("🔄 Loading CellPose model...")
        try:
            _model_cache = models.CellposeModel(gpu=use_gpu, model_type='cyto2')
            print(f"✅ Model loaded (GPU: {use_gpu})")
        except Exception as e:
            print(f"⚠️  GPU failed, falling back to CPU: {e}")
            _model_cache = models.CellposeModel(gpu=False, model_type='cyto2')
            print("✅ Model loaded (CPU mode)")
    return _model_cache


def preprocess_image(img):
    """
    Advanced image preprocessing for better segmentation.
    
    Args:
        img: Input image (grayscale or RGB)
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)
    
    # Denoise
    img_denoised = filters.gaussian(img_enhanced, sigma=1.0)
    
    # Normalize to 0-255 range
    img_normalized = exposure.rescale_intensity(img_denoised, out_range=(0, 255))
    
    return img_normalized.astype(np.uint8)


def estimate_cell_diameter(img):
    """
    Automatically estimate cell diameter from image.
    
    Args:
        img: Input image
        
    Returns:
        Estimated diameter in pixels
    """
    # Use edge detection to estimate cell size
    edges = filters.sobel(img)
    
    # Find connected components
    labeled = morphology.label(edges > filters.threshold_otsu(edges))
    regions = regionprops(labeled)
    
    if len(regions) > 0:
        # Use median equivalent diameter
        diameters = [r.equivalent_diameter for r in regions if r.area > 50]
        if len(diameters) > 0:
            estimated = np.median(diameters)
            # Clamp to reasonable range
            return max(20, min(100, int(estimated)))
    
    # Default fallback
    return 30


def post_process_masks(masks, min_size=50):
    """
    Clean up segmentation masks.
    
    Args:
        masks: Raw segmentation masks
        min_size: Minimum cell size in pixels
        
    Returns:
        Cleaned masks
    """
    # Remove small objects
    cleaned = morphology.remove_small_objects(masks > 0, min_size=min_size)
    
    # Fill holes
    filled = ndimage.binary_fill_holes(cleaned)
    
    # Re-label
    labeled = morphology.label(filled)
    
    return labeled


def calculate_cell_health_score(props_dict, idx):
    """
    Calculate cell health score based on morphological features.
    
    Args:
        props_dict: Dictionary of region properties
        idx: Index of the cell
        
    Returns:
        Health score (0-100)
    """
    # Extract features
    circularity = 4 * np.pi * props_dict['area'][idx] / (props_dict['perimeter'][idx] ** 2)
    solidity = props_dict['solidity'][idx]
    eccentricity = props_dict['eccentricity'][idx]
    
    # Healthy cells are typically:
    # - Circular (circularity close to 1)
    # - Solid (solidity close to 1)
    # - Not too elongated (eccentricity < 0.8)
    
    circularity_score = min(circularity, 1.0) * 40  # Max 40 points
    solidity_score = solidity * 30  # Max 30 points
    eccentricity_score = (1 - min(eccentricity, 1.0)) * 30  # Max 30 points
    
    total_score = circularity_score + solidity_score + eccentricity_score
    
    return min(100, max(0, total_score))


def classify_cell_morphology(health_score, eccentricity, solidity):
    """
    Classify cell morphology based on features.
    
    Args:
        health_score: Cell health score
        eccentricity: Cell eccentricity
        solidity: Cell solidity
        
    Returns:
        Classification string
    """
    if health_score >= 75:
        return "Healthy"
    elif health_score >= 50:
        if eccentricity > 0.8:
            return "Elongated"
        elif solidity < 0.85:
            return "Irregular"
        else:
            return "Moderate"
    else:
        if solidity < 0.7:
            return "Apoptotic"
        else:
            return "Stressed"


def calculate_spatial_metrics(masks):
    """
    Calculate spatial distribution metrics.
    
    Args:
        masks: Segmentation masks
        
    Returns:
        Dictionary of spatial metrics
    """
    regions = regionprops(masks)
    
    if len(regions) < 2:
        return {
            'avg_nearest_neighbor': 0,
            'clustering_coefficient': 0,
            'spatial_distribution': 'N/A'
        }
    
    # Get centroids
    centroids = np.array([r.centroid for r in regions])
    
    # Calculate distance matrix
    dist_matrix = distance_matrix(centroids, centroids)
    
    # Find nearest neighbor distances (excluding self)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_distances = np.min(dist_matrix, axis=1)
    avg_nearest = np.mean(nearest_distances)
    
    # Clustering analysis using DBSCAN
    clustering = DBSCAN(eps=avg_nearest * 2, min_samples=2).fit(centroids)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    
    # Clustering coefficient
    clustering_coef = n_clusters / len(regions) if len(regions) > 0 else 0
    
    # Determine spatial distribution
    if clustering_coef > 0.5:
        distribution = "Highly Clustered"
    elif clustering_coef > 0.2:
        distribution = "Moderately Clustered"
    else:
        distribution = "Uniformly Distributed"
    
    return {
        'avg_nearest_neighbor': float(avg_nearest),
        'clustering_coefficient': float(clustering_coef),
        'spatial_distribution': distribution,
        'n_clusters': int(n_clusters)
    }


def analyze_microscopy_image(image_path, use_gpu=False, diameter=None):
    """
    Complete enhanced microscopy image analysis pipeline.
    
    Args:
        image_path: Path to microscopy image
        use_gpu: Whether to use GPU acceleration
        diameter: Cell diameter (None for auto-detection)
        
    Returns:
        masks: Segmentation masks for each cell
        metrics: Dictionary of comprehensive measurements
        cell_data: Detailed per-cell analysis
    """
    # 1. Load image
    img = io.imread(image_path)
    
    # 2. Preprocess
    img_processed = preprocess_image(img)
    
    # 3. Estimate diameter if not provided
    if diameter is None:
        diameter = estimate_cell_diameter(img_processed)
        print(f"📏 Estimated cell diameter: {diameter} pixels")
    
    # 4. Cell segmentation with CellPose
    model = get_cellpose_model(use_gpu=use_gpu)
    masks, flows, styles = model.eval(img_processed, diameter=diameter, channels=[0, 0])
    
    # 5. Post-process masks
    masks = post_process_masks(masks, min_size=50)
    
    # 6. Extract quantitative metrics
    cell_count = len(np.unique(masks)) - 1
    
    if cell_count == 0:
        return masks, {
            'total_cells': 0,
            'avg_area': 0,
            'std_area': 0,
            'median_area': 0,
            'avg_circularity': 0,
            'avg_eccentricity': 0,
            'avg_solidity': 0,
            'density': 0,
            'size_variation': 0,
            'avg_health_score': 0,
            'healthy_cells': 0,
            'stressed_cells': 0,
            'apoptotic_cells': 0,
            'healthy_percentage': 0,
            'avg_nearest_neighbor': 0,
            'clustering_coefficient': 0,
            'spatial_distribution': 'N/A'
        }, []
    
    # 7. Morphology analysis
    props = regionprops_table(
        masks, img_processed,
        properties=['area', 'perimeter', 'eccentricity', 'solidity', 
                   'major_axis_length', 'minor_axis_length', 'centroid']
    )
    
    # 8. Calculate advanced metrics
    areas = props['area']
    perimeters = props['perimeter']
    circularities = 4 * np.pi * areas / (perimeters ** 2)
    
    # Calculate health scores for each cell
    health_scores = []
    morphology_classes = []
    
    for i in range(len(areas)):
        health = calculate_cell_health_score(props, i)
        health_scores.append(health)
        
        morph_class = classify_cell_morphology(
            health, 
            props['eccentricity'][i], 
            props['solidity'][i]
        )
        morphology_classes.append(morph_class)
    
    # Count cell types
    healthy_count = sum(1 for h in health_scores if h >= 75)
    stressed_count = sum(1 for h in health_scores if 50 <= h < 75)
    apoptotic_count = sum(1 for h in health_scores if h < 50)
    
    # 9. Spatial analysis
    spatial_metrics = calculate_spatial_metrics(masks)
    
    # 10. Compile comprehensive metrics
    metrics = {
        'total_cells': cell_count,
        'avg_area': float(np.mean(areas)),
        'std_area': float(np.std(areas)),
        'median_area': float(np.median(areas)),
        'avg_circularity': float(np.mean(circularities)),
        'avg_eccentricity': float(np.mean(props['eccentricity'])),
        'avg_solidity': float(np.mean(props['solidity'])),
        'density': cell_count / (img.shape[0] * img.shape[1]),
        'size_variation': float(np.std(areas) / np.mean(areas)) if np.mean(areas) > 0 else 0,
        'avg_health_score': float(np.mean(health_scores)),
        'healthy_cells': healthy_count,
        'stressed_cells': stressed_count,
        'apoptotic_cells': apoptotic_count,
        'healthy_percentage': (healthy_count / cell_count * 100) if cell_count > 0 else 0,
        **spatial_metrics
    }
    
    # 11. Per-cell data
    cell_data = []
    for i in range(len(areas)):
        cell_data.append({
            'cell_id': i + 1,
            'area': float(areas[i]),
            'circularity': float(circularities[i]),
            'eccentricity': float(props['eccentricity'][i]),
            'solidity': float(props['solidity'][i]),
            'health_score': float(health_scores[i]),
            'morphology': morphology_classes[i]
        })
    
    return masks, metrics, cell_data


def generate_analysis_narrative(image_path, masks, metrics, cell_data, api_key):
    """
    Generate enhanced publication-quality figure legend using GPT-4 Vision.
    
    Args:
        image_path: Path to original image
        masks: Segmentation masks from CellPose
        metrics: Quantitative metrics dictionary
        cell_data: Per-cell analysis data
        api_key: OpenAI API key
        
    Returns:
        str: Publication-ready figure legend
    """
    # 1. Create enhanced visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(io.imread(image_path), cmap='gray')
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(masks, cmap='tab20')
    axes[1].set_title(f'Segmented: {metrics["total_cells"]} cells', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Health heatmap
    health_map = np.zeros_like(masks, dtype=float)
    for cell in cell_data:
        health_map[masks == cell['cell_id']] = cell['health_score']
    
    im = axes[2].imshow(health_map, cmap='RdYlGn', vmin=0, vmax=100)
    axes[2].set_title('Cell Health Score', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Health Score (0-100)')
    
    plt.tight_layout()
    plt.savefig('analysis_enhanced.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Encode image for GPT-4 Vision
    with open('analysis_enhanced.png', 'rb') as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 3. Create enhanced prompt with comprehensive context
    prompt = f"""You are an expert cell biologist analyzing microscopy data. Provide a publication-quality figure legend.

QUANTITATIVE ANALYSIS:
- Total cells detected: {metrics['total_cells']}
- Cell density: {metrics['density']:.6f} cells/pixel²
- Average cell area: {metrics['avg_area']:.1f} ± {metrics['std_area']:.1f} pixels²
- Size variation coefficient: {metrics['size_variation']:.2f}

MORPHOLOGICAL FEATURES:
- Average circularity: {metrics['avg_circularity']:.3f} (1.0 = perfect circle)
- Average eccentricity: {metrics['avg_eccentricity']:.3f} (0 = circle, 1 = line)
- Average solidity: {metrics['avg_solidity']:.3f} (convexity measure)

CELL HEALTH ASSESSMENT:
- Average health score: {metrics['avg_health_score']:.1f}/100
- Healthy cells: {metrics['healthy_cells']} ({metrics['healthy_percentage']:.1f}%)
- Stressed cells: {metrics['stressed_cells']}
- Apoptotic/damaged cells: {metrics['apoptotic_cells']}

SPATIAL DISTRIBUTION:
- Pattern: {metrics['spatial_distribution']}
- Average nearest neighbor distance: {metrics['avg_nearest_neighbor']:.1f} pixels
- Clustering coefficient: {metrics['clustering_coefficient']:.3f}

Generate a comprehensive figure legend that:
1. Describes the cell population and morphology
2. Highlights any notable features or abnormalities
3. Interprets the health scores and what they indicate
4. Discusses the spatial distribution pattern
5. Provides quantitative evidence for all observations
6. Uses proper scientific terminology

Format: "Figure X: [Comprehensive scientific description]"
Be concise yet thorough (200-300 words)."""

    # 4. Call GPT-4 Vision
    try:
        client = OpenAI(api_key=api_key)
        
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
            max_tokens=800,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating narrative: {str(e)}\n\nFallback Summary:\nAnalyzed {metrics['total_cells']} cells with average health score of {metrics['avg_health_score']:.1f}/100. Population shows {metrics['spatial_distribution'].lower()} pattern with {metrics['healthy_percentage']:.1f}% healthy cells."
