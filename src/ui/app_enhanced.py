"""
CellVision - Enhanced Streamlit Web Application
Professional AI-Powered Microscopy Analysis Interface
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.analysis_enhanced import analyze_microscopy_image, generate_analysis_narrative
from skimage import io
import json


# Page configuration
st.set_page_config(
    page_title="CellVision - AI Microscopy Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def load_demo_images():
    """Load available demo images"""
    # Try multiple possible paths
    possible_paths = [
        Path("data/demo_images"),
        Path("demo_images"),
        Path(__file__).parent.parent.parent / "data" / "demo_images"
    ]
    
    for demo_dir in possible_paths:
        if demo_dir.exists():
            return list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png"))
    return []


def create_health_distribution_chart(cell_data):
    """Create interactive health score distribution chart"""
    if not cell_data:
        return None
    
    health_scores = [cell['health_score'] for cell in cell_data]
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=health_scores,
        nbinsx=20,
        marker_color='rgba(102, 126, 234, 0.7)',
        marker_line_color='rgba(102, 126, 234, 1)',
        marker_line_width=1.5,
        name='Health Score Distribution'
    ))
    
    fig.update_layout(
        title="Cell Health Score Distribution",
        xaxis_title="Health Score (0-100)",
        yaxis_title="Number of Cells",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_morphology_pie_chart(cell_data):
    """Create pie chart of cell morphology classes"""
    if not cell_data:
        return None
    
    morphology_counts = {}
    for cell in cell_data:
        morph = cell['morphology']
        morphology_counts[morph] = morphology_counts.get(morph, 0) + 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(morphology_counts.keys()),
        values=list(morphology_counts.values()),
        hole=0.4,
        marker_colors=['#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    )])
    
    fig.update_layout(
        title="Cell Morphology Classification",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_scatter_plot(cell_data):
    """Create scatter plot of cell features"""
    if not cell_data:
        return None
    
    df = pd.DataFrame(cell_data)
    
    fig = px.scatter(
        df,
        x='area',
        y='circularity',
        color='health_score',
        size='area',
        hover_data=['cell_id', 'morphology'],
        color_continuous_scale='RdYlGn',
        title="Cell Feature Space",
        labels={
            'area': 'Cell Area (pixels²)',
            'circularity': 'Circularity',
            'health_score': 'Health Score'
        }
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    
    return fig


def export_to_csv(metrics, cell_data):
    """Export analysis results to CSV"""
    # Create summary dataframe
    summary_df = pd.DataFrame([metrics])
    
    # Create cell data dataframe
    cells_df = pd.DataFrame(cell_data)
    
    # Combine into CSV string
    csv_buffer = []
    csv_buffer.append("=== SUMMARY METRICS ===\n")
    csv_buffer.append(summary_df.to_csv(index=False))
    csv_buffer.append("\n=== PER-CELL DATA ===\n")
    csv_buffer.append(cells_df.to_csv(index=False))
    
    return "".join(csv_buffer)


def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 CellVision</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Microscopy Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key for AI-powered analysis"
        )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            use_gpu = st.checkbox("Use GPU Acceleration", value=False)
            auto_diameter = st.checkbox("Auto-detect Cell Size", value=True)
            
            if not auto_diameter:
                cell_diameter = st.slider(
                    "Cell Diameter (pixels)",
                    min_value=10,
                    max_value=150,
                    value=30,
                    step=5
                )
            else:
                cell_diameter = None
        
        st.markdown("---")
        
        # Demo images
        st.header("📚 Demo Images")
        demo_images = load_demo_images()
        
        if demo_images:
            demo_names = [img.name for img in demo_images]
            selected_demo = st.selectbox(
                "Load example",
                ["None"] + demo_names
            )
            
            if selected_demo != "None":
                st.session_state['demo_image'] = str([img for img in demo_images if img.name == selected_demo][0])
        
        st.markdown("---")
        
        # About
        st.header("📖 About")
        st.markdown("""
        **CellVision** combines cutting-edge computer vision with AI to provide:
        
        - 🎯 Automated cell segmentation
        - 📊 20+ quantitative metrics
        - 🏥 Cell health scoring
        - 🧬 Morphology classification
        - 📍 Spatial analysis
        - 🤖 AI-powered insights
        
        Transform hours of manual work into seconds of automated analysis.
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📤 Analysis", "📊 Batch Processing", "ℹ️ Help"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a microscopy image",
                type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
                help="Supported formats: PNG, JPEG, TIFF"
            )
            
            # Handle demo image selection
            if 'demo_image' in st.session_state and uploaded_file is None:
                image_path = st.session_state['demo_image']
                st.image(image_path, caption="Demo Image", use_container_width=True)
                del st.session_state['demo_image']  # Clear after use
            elif uploaded_file:
                # Save uploaded file
                temp_path = f"temp_{datetime.now().timestamp()}.png"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_path = temp_path
                
                # Display original
                st.image(image_path, caption="Uploaded Image", use_container_width=True)
            else:
                image_path = None
                st.info("👆 Upload an image or select a demo from the sidebar")
        
        with col2:
            st.subheader("📊 Analysis Results")
            
            if image_path and st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                if not api_key:
                    st.warning("⚠️ Please enter your OpenAI API key in the sidebar for AI-powered narrative generation.")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Segmentation
                    status_text.text("🔍 Segmenting cells...")
                    progress_bar.progress(25)
                    
                    masks, metrics, cell_data = analyze_microscopy_image(
                        image_path,
                        use_gpu=use_gpu,
                        diameter=cell_diameter
                    )
                    
                    progress_bar.progress(50)
                    status_text.text("📈 Calculating advanced metrics...")
                    
                    # Store in session state
                    st.session_state['masks'] = masks
                    st.session_state['metrics'] = metrics
                    st.session_state['cell_data'] = cell_data
                    st.session_state['image_path'] = image_path
                    
                    progress_bar.progress(75)
                    
                    # Step 2: AI Narrative (if API key provided)
                    if api_key:
                        status_text.text("🤖 Generating AI analysis...")
                        narrative = generate_analysis_narrative(
                            image_path, masks, metrics, cell_data, api_key
                        )
                        st.session_state['narrative'] = narrative
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    
                    st.success("🎉 Analysis completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    status_text.text("Analysis failed!")
                    progress_bar.progress(0)
        
        # Display results if available
        if 'metrics' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Quantitative Results")
            
            metrics = st.session_state['metrics']
            cell_data = st.session_state['cell_data']
            
            # Key metrics in columns
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Cells",
                    metrics['total_cells'],
                    help="Number of cells detected"
                )
            
            with col2:
                st.metric(
                    "Avg Health Score",
                    f"{metrics['avg_health_score']:.1f}/100",
                    help="Average cell health score"
                )
            
            with col3:
                st.metric(
                    "Healthy Cells",
                    f"{metrics['healthy_percentage']:.1f}%",
                    help="Percentage of healthy cells (score ≥ 75)"
                )
            
            with col4:
                st.metric(
                    "Avg Area",
                    f"{metrics['avg_area']:.0f} px²",
                    help="Average cell area"
                )
            
            with col5:
                st.metric(
                    "Distribution",
                    metrics['spatial_distribution'],
                    help="Spatial distribution pattern"
                )
            
            # Detailed metrics in expandable sections
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("🔬 Morphological Metrics", expanded=True):
                    st.write(f"**Circularity:** {metrics['avg_circularity']:.3f}")
                    st.write(f"**Eccentricity:** {metrics['avg_eccentricity']:.3f}")
                    st.write(f"**Solidity:** {metrics['avg_solidity']:.3f}")
                    st.write(f"**Size Variation:** {metrics['size_variation']:.3f}")
            
            with col2:
                with st.expander("📍 Spatial Metrics", expanded=True):
                    st.write(f"**Nearest Neighbor:** {metrics['avg_nearest_neighbor']:.1f} px")
                    st.write(f"**Clustering Coef:** {metrics['clustering_coefficient']:.3f}")
                    st.write(f"**Cell Density:** {metrics['density']:.6f} cells/px²")
            
            # Visualizations
            st.markdown("---")
            st.subheader("📈 Interactive Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Health distribution
                health_fig = create_health_distribution_chart(cell_data)
                if health_fig:
                    st.plotly_chart(health_fig, use_container_width=True)
                
                # Morphology pie chart
                morph_fig = create_morphology_pie_chart(cell_data)
                if morph_fig:
                    st.plotly_chart(morph_fig, use_container_width=True)
            
            with viz_col2:
                # Segmentation visualization
                st.write("**Segmentation Result**")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(st.session_state['masks'], cmap='tab20')
                ax.set_title(f"Detected {metrics['total_cells']} cells", fontsize=14, fontweight='bold')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
                # Feature scatter plot
                scatter_fig = create_scatter_plot(cell_data)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
            
            # AI Narrative
            if 'narrative' in st.session_state:
                st.markdown("---")
                st.subheader("🤖 AI-Generated Figure Legend")
                st.info(st.session_state['narrative'])
            
            # Export options
            st.markdown("---")
            st.subheader("💾 Export Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                # Export CSV
                csv_data = export_to_csv(metrics, cell_data)
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_data,
                    file_name=f"cellvision_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with export_col2:
                # Export JSON
                json_data = json.dumps({
                    'metrics': metrics,
                    'cell_data': cell_data
                }, indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"cellvision_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with export_col3:
                # Export narrative
                if 'narrative' in st.session_state:
                    st.download_button(
                        label="📝 Download Legend",
                        data=st.session_state['narrative'],
                        file_name=f"figure_legend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    
    with tab2:
        st.subheader("📊 Batch Processing")
        st.info("🚧 Batch processing feature coming soon! Upload multiple images for parallel analysis.")
        
        # Placeholder for batch processing
        batch_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            accept_multiple_files=True
        )
        
        if batch_files:
            st.write(f"📁 {len(batch_files)} files selected")
            if st.button("🚀 Process Batch"):
                st.warning("Batch processing will be implemented in the next version!")
    
    with tab3:
        st.subheader("ℹ️ How to Use CellVision")
        
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Image**: Click "Browse files" or drag & drop your microscopy image
        2. **Configure**: (Optional) Adjust settings in the sidebar
        3. **Analyze**: Click the "🚀 Analyze Image" button
        4. **Review**: Explore the comprehensive metrics and visualizations
        5. **Export**: Download results in CSV, JSON, or text format
        
        ### Metrics Explained
        
        - **Health Score (0-100)**: Composite score based on cell morphology
          - 75-100: Healthy cells
          - 50-75: Stressed/moderate cells
          - 0-50: Apoptotic/damaged cells
        
        - **Circularity**: How close to a perfect circle (1.0 = perfect circle)
        - **Eccentricity**: Cell elongation (0 = circle, 1 = line)
        - **Solidity**: Convexity measure (1.0 = perfectly convex)
        - **Spatial Distribution**: Clustering pattern analysis
        
        ### Tips for Best Results
        
        - Use high-contrast images with clear cell boundaries
        - Ensure adequate resolution (minimum 512x512 pixels)
        - For fluorescence images, use single-channel images
        - Enable GPU acceleration for faster processing (if available)
        
        ### Troubleshooting
        
        - **Low cell count**: Try adjusting the cell diameter setting
        - **Over-segmentation**: Increase the minimum cell size
        - **Under-segmentation**: Decrease the cell diameter
        
        ### Citation
        
        If you use CellVision in your research, please cite:
        - CellPose: Stringer et al., Nature Methods (2021)
        - OpenAI GPT-4: OpenAI (2023)
        """)


if __name__ == "__main__":
    main()
