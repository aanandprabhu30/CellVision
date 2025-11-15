"""
CellVision - PDF Report Generator
Generate publication-quality PDF reports with all analysis results
"""

from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os


class CellVisionReport(FPDF):
    """Custom PDF report class for CellVision"""
    
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 16)
        self.set_text_color(102, 126, 234)
        self.cell(0, 10, 'CellVision Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add chapter title"""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def chapter_body(self, body):
        """Add chapter body text"""
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_metric_box(self, label, value, x, y, width=45, height=20):
        """Add a metric box"""
        self.set_xy(x, y)
        self.set_fill_color(102, 126, 234)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 10)
        self.cell(width, height/2, label, 0, 1, 'C', True)
        
        self.set_xy(x, y + height/2)
        self.set_fill_color(240, 240, 255)
        self.set_text_color(0, 0, 0)
        self.set_font('Arial', 'B', 14)
        self.cell(width, height/2, str(value), 0, 1, 'C', True)


def generate_visualizations(image_path, masks, metrics, cell_data):
    """
    Generate all visualizations for the report.
    
    Returns:
        Dictionary of image paths
    """
    images = {}
    
    # 1. Original image
    img = io.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title('Original Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    images['original'] = 'report_original.png'
    plt.savefig(images['original'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Segmentation
    plt.figure(figsize=(8, 8))
    plt.imshow(masks, cmap='tab20')
    plt.title(f'Segmentation: {metrics["total_cells"]} cells detected', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    images['segmentation'] = 'report_segmentation.png'
    plt.savefig(images['segmentation'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Health heatmap
    health_map = np.zeros_like(masks, dtype=float)
    for cell in cell_data:
        health_map[masks == cell['cell_id']] = cell['health_score']
    
    plt.figure(figsize=(8, 8))
    im = plt.imshow(health_map, cmap='RdYlGn', vmin=0, vmax=100)
    plt.colorbar(im, label='Health Score (0-100)', fraction=0.046, pad=0.04)
    plt.title('Cell Health Heatmap', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    images['health'] = 'report_health.png'
    plt.savefig(images['health'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Health distribution histogram
    health_scores = [cell['health_score'] for cell in cell_data]
    plt.figure(figsize=(10, 6))
    plt.hist(health_scores, bins=20, color='#667eea', alpha=0.7, edgecolor='black')
    plt.xlabel('Health Score', fontsize=12)
    plt.ylabel('Number of Cells', fontsize=12)
    plt.title('Health Score Distribution', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    images['histogram'] = 'report_histogram.png'
    plt.savefig(images['histogram'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Morphology classification
    morphology_counts = {}
    for cell in cell_data:
        morph = cell['morphology']
        morphology_counts[morph] = morphology_counts.get(morph, 0) + 1
    
    plt.figure(figsize=(8, 8))
    colors = ['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    plt.pie(
        morphology_counts.values(),
        labels=morphology_counts.keys(),
        autopct='%1.1f%%',
        colors=colors[:len(morphology_counts)],
        startangle=90
    )
    plt.title('Cell Morphology Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    images['morphology'] = 'report_morphology.png'
    plt.savefig(images['morphology'], dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Feature scatter plot
    areas = [cell['area'] for cell in cell_data]
    circularities = [cell['circularity'] for cell in cell_data]
    health_scores_scatter = [cell['health_score'] for cell in cell_data]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        areas, circularities,
        c=health_scores_scatter,
        s=100,
        cmap='RdYlGn',
        vmin=0, vmax=100,
        alpha=0.6,
        edgecolors='black'
    )
    plt.colorbar(scatter, label='Health Score')
    plt.xlabel('Cell Area (pixels²)', fontsize=12)
    plt.ylabel('Circularity', fontsize=12)
    plt.title('Cell Feature Space', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    images['scatter'] = 'report_scatter.png'
    plt.savefig(images['scatter'], dpi=150, bbox_inches='tight')
    plt.close()
    
    return images


def generate_pdf_report(image_path, masks, metrics, cell_data, narrative=None, output_path=None):
    """
    Generate comprehensive PDF report.
    
    Args:
        image_path: Path to original image
        masks: Segmentation masks
        metrics: Analysis metrics dictionary
        cell_data: Per-cell data list
        narrative: AI-generated narrative (optional)
        output_path: Output PDF path (optional)
        
    Returns:
        Path to generated PDF
    """
    if output_path is None:
        output_path = f"CellVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    images = generate_visualizations(image_path, masks, metrics, cell_data)
    
    # Create PDF
    print("📄 Creating PDF report...")
    pdf = CellVisionReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Cover and Summary
    pdf.add_page()
    
    # Title section
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 20, 'CellVision', 0, 1, 'C')
    
    pdf.set_font('Arial', '', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'AI-Powered Microscopy Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Analysis info
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.cell(0, 8, f'Image: {os.path.basename(image_path)}', 0, 1, 'C')
    pdf.ln(20)
    
    # Key metrics boxes
    pdf.chapter_title('Summary Metrics')
    
    y_pos = pdf.get_y()
    pdf.add_metric_box('Total Cells', metrics['total_cells'], 15, y_pos, 45, 20)
    pdf.add_metric_box('Avg Health', f"{metrics['avg_health_score']:.1f}/100", 65, y_pos, 45, 20)
    pdf.add_metric_box('Healthy %', f"{metrics['healthy_percentage']:.1f}%", 115, y_pos, 45, 20)
    pdf.add_metric_box('Avg Area', f"{metrics['avg_area']:.0f} px²", 165, y_pos, 45, 20)
    
    pdf.ln(25)
    
    # Executive Summary
    pdf.chapter_title('Executive Summary')
    summary = f"""This report presents a comprehensive analysis of {metrics['total_cells']} cells detected in the microscopy image. 
    
The cell population shows an average health score of {metrics['avg_health_score']:.1f}/100, with {metrics['healthy_cells']} healthy cells ({metrics['healthy_percentage']:.1f}%), {metrics['stressed_cells']} stressed cells, and {metrics['apoptotic_cells']} apoptotic/damaged cells.

Spatial analysis reveals a {metrics['spatial_distribution'].lower()} pattern with an average nearest neighbor distance of {metrics['avg_nearest_neighbor']:.1f} pixels and a clustering coefficient of {metrics['clustering_coefficient']:.3f}.

Morphological analysis indicates average circularity of {metrics['avg_circularity']:.3f}, eccentricity of {metrics['avg_eccentricity']:.3f}, and solidity of {metrics['avg_solidity']:.3f}."""
    
    pdf.chapter_body(summary)
    
    # Page 2: Visualizations
    pdf.add_page()
    pdf.chapter_title('Image Analysis')
    
    # Original and segmentation side by side
    pdf.image(images['original'], x=15, y=pdf.get_y(), w=85)
    pdf.image(images['segmentation'], x=110, y=pdf.get_y(), w=85)
    pdf.ln(90)
    
    # Health heatmap
    pdf.image(images['health'], x=55, y=pdf.get_y(), w=100)
    pdf.ln(105)
    
    # Page 3: Statistical Analysis
    pdf.add_page()
    pdf.chapter_title('Statistical Analysis')
    
    # Health distribution
    pdf.image(images['histogram'], x=15, y=pdf.get_y(), w=180)
    pdf.ln(75)
    
    # Morphology pie chart and scatter plot
    pdf.image(images['morphology'], x=15, y=pdf.get_y(), w=85)
    pdf.image(images['scatter'], x=110, y=pdf.get_y(), w=85)
    
    # Page 4: Detailed Metrics
    pdf.add_page()
    pdf.chapter_title('Detailed Metrics')
    
    # Morphological metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Morphological Features', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    morph_metrics = [
        ('Average Cell Area', f"{metrics['avg_area']:.1f} ± {metrics['std_area']:.1f} pixels²"),
        ('Median Cell Area', f"{metrics['median_area']:.1f} pixels²"),
        ('Circularity', f"{metrics['avg_circularity']:.3f}"),
        ('Eccentricity', f"{metrics['avg_eccentricity']:.3f}"),
        ('Solidity', f"{metrics['avg_solidity']:.3f}"),
        ('Size Variation Coefficient', f"{metrics['size_variation']:.3f}"),
    ]
    
    for label, value in morph_metrics:
        pdf.cell(100, 6, label, 0, 0)
        pdf.cell(0, 6, value, 0, 1)
    
    pdf.ln(5)
    
    # Health metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Cell Health Assessment', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    health_metrics = [
        ('Average Health Score', f"{metrics['avg_health_score']:.1f}/100"),
        ('Healthy Cells (≥75)', f"{metrics['healthy_cells']} ({metrics['healthy_percentage']:.1f}%)"),
        ('Stressed Cells (50-75)', f"{metrics['stressed_cells']}"),
        ('Apoptotic Cells (<50)', f"{metrics['apoptotic_cells']}"),
    ]
    
    for label, value in health_metrics:
        pdf.cell(100, 6, label, 0, 0)
        pdf.cell(0, 6, value, 0, 1)
    
    pdf.ln(5)
    
    # Spatial metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Spatial Distribution', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    spatial_metrics = [
        ('Distribution Pattern', metrics['spatial_distribution']),
        ('Avg Nearest Neighbor Distance', f"{metrics['avg_nearest_neighbor']:.1f} pixels"),
        ('Clustering Coefficient', f"{metrics['clustering_coefficient']:.3f}"),
        ('Number of Clusters', f"{metrics.get('n_clusters', 'N/A')}"),
        ('Cell Density', f"{metrics['density']:.6f} cells/pixel²"),
    ]
    
    for label, value in spatial_metrics:
        pdf.cell(100, 6, label, 0, 0)
        pdf.cell(0, 6, str(value), 0, 1)
    
    # Page 5: AI Narrative (if available)
    if narrative:
        pdf.add_page()
        pdf.chapter_title('AI-Generated Figure Legend')
        pdf.chapter_body(narrative)
    
    # Page 6: Methods
    pdf.add_page()
    pdf.chapter_title('Methods')
    
    methods = """Image Analysis Pipeline:

1. Image Preprocessing: Contrast-limited adaptive histogram equalization (CLAHE) was applied to enhance image contrast, followed by Gaussian denoising (σ=1.0) and intensity normalization.

2. Cell Segmentation: Automated cell segmentation was performed using CellPose (Stringer et al., 2021) with the 'cyto2' model. Cell diameter was automatically estimated from the image or manually specified.

3. Post-processing: Small objects (<50 pixels) were removed, and binary holes were filled to improve segmentation quality.

4. Morphological Analysis: For each detected cell, the following features were quantified: area, perimeter, circularity, eccentricity, solidity, and major/minor axis lengths.

5. Health Scoring: A composite health score (0-100) was calculated for each cell based on circularity (40 points), solidity (30 points), and eccentricity (30 points).

6. Spatial Analysis: Nearest neighbor distances were calculated using Euclidean distance between cell centroids. Clustering analysis was performed using DBSCAN algorithm.

7. AI Narration: Publication-quality figure legends were generated using GPT-4 Vision (OpenAI, 2023) with comprehensive quantitative context.

References:
- Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature Methods, 18(1), 100-106.
- OpenAI. (2023). GPT-4 Technical Report. arXiv:2303.08774."""
    
    pdf.chapter_body(methods)
    
    # Save PDF
    pdf.output(output_path)
    
    # Cleanup temporary images
    for img_path in images.values():
        if os.path.exists(img_path):
            os.remove(img_path)
    
    print(f"✅ PDF report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    print("CellVision PDF Report Generator")
    print("This module is designed to be imported and used by the main application.")
