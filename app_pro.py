"""
CellVision - Professional UI
Ultra-modern, sleek interface for microscopy analysis
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.analysis_enhanced import analyze_microscopy_image, generate_analysis_narrative
from skimage import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="CellVision Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Ultra-modern CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Modern color scheme */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #1e293b;
        --light: #f8fafc;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Content card */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Header */
    .hero-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Upload area */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 0 2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background: #667eea;
        color: white;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        color: white;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def load_demo_images():
    """Load demo images"""
    paths = [
        Path("data/demo_images"),
        Path(__file__).parent / "data" / "demo_images"
    ]
    for p in paths:
        if p.exists():
            return list(p.glob("*.jpg")) + list(p.glob("*.png"))
    return []


def create_modern_chart(cell_data, chart_type="health"):
    """Create modern, beautiful charts"""
    if not cell_data:
        return None
    
    if chart_type == "health":
        health_scores = [c['health_score'] for c in cell_data]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=health_scores,
            nbinsx=25,
            marker=dict(
                color=health_scores,
                colorscale='RdYlGn',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Health Score: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text="<b>Cell Health Distribution</b>",
                font=dict(size=20, color='#1e293b')
            ),
            xaxis_title="Health Score (0-100)",
            yaxis_title="Number of Cells",
            template="plotly_white",
            height=400,
            plot_bgcolor='rgba(248, 250, 252, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif")
        )
        
        return fig
    
    elif chart_type == "morphology":
        morphology_counts = {}
        for cell in cell_data:
            morph = cell['morphology']
            morphology_counts[morph] = morphology_counts.get(morph, 0) + 1
        
        colors = {
            'Healthy': '#10b981',
            'Stressed': '#f59e0b',
            'Apoptotic': '#ef4444',
            'Elongated': '#8b5cf6',
            'Irregular': '#ec4899'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(morphology_counts.keys()),
            values=list(morphology_counts.values()),
            hole=0.5,
            marker=dict(
                colors=[colors.get(k, '#94a3b8') for k in morphology_counts.keys()],
                line=dict(color='white', width=2)
            ),
            textfont=dict(size=14, color='white', family="Inter"),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=dict(
                text="<b>Cell Morphology Classification</b>",
                font=dict(size=20, color='#1e293b')
            ),
            template="plotly_white",
            height=400,
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig


def main():
    # Hero header
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🔬 CellVision Pro</h1>
        <p class="hero-subtitle">AI-Powered Microscopy Analysis in 30 Seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Microscopy Image")
        
        # Demo images
        demo_images = load_demo_images()
        if demo_images:
            st.markdown("**Quick Start:** Try a demo image")
            demo_names = [img.name for img in demo_images]
            selected_demo = st.selectbox(
                "Select demo",
                ["Upload your own..."] + demo_names,
                label_visibility="collapsed"
            )
            
            if selected_demo != "Upload your own...":
                image_path = str([img for img in demo_images if img.name == selected_demo][0])
                st.image(image_path, use_container_width=True, caption="Demo Image")
            else:
                uploaded_file = st.file_uploader(
                    "Choose image",
                    type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
                    label_visibility="collapsed"
                )
                
                if uploaded_file:
                    temp_path = f"temp_{datetime.now().timestamp()}.png"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    image_path = temp_path
                    st.image(image_path, use_container_width=True, caption="Uploaded Image")
                else:
                    image_path = None
        else:
            uploaded_file = st.file_uploader(
                "Choose image",
                type=['png', 'jpg', 'jpeg', 'tif', 'tiff']
            )
            
            if uploaded_file:
                temp_path = f"temp_{datetime.now().timestamp()}.png"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_path = temp_path
                st.image(image_path, use_container_width=True)
            else:
                image_path = None
        
        # API key
        st.markdown("---")
        api_key = st.text_input(
            "🔑 OpenAI API Key (for AI narration)",
            type="password",
            placeholder="sk-..."
        )
        
        # Analyze button
        st.markdown("---")
        analyze_btn = st.button("🚀 Analyze Image", use_container_width=True, type="primary")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if image_path and analyze_btn:
            progress_bar = st.progress(0)
            status = st.empty()
            
            try:
                status.info("🔍 Analyzing cells...")
                progress_bar.progress(30)
                
                masks, metrics, cell_data = analyze_microscopy_image(
                    image_path,
                    use_gpu=False,
                    diameter=None
                )
                
                progress_bar.progress(70)
                status.info("📈 Calculating metrics...")
                
                st.session_state['masks'] = masks
                st.session_state['metrics'] = metrics
                st.session_state['cell_data'] = cell_data
                st.session_state['image_path'] = image_path
                
                if api_key:
                    status.info("🤖 Generating AI analysis...")
                    narrative = generate_analysis_narrative(
                        image_path, masks, metrics, cell_data, api_key
                    )
                    st.session_state['narrative'] = narrative
                
                progress_bar.progress(100)
                status.success("✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                status.error(f"❌ Error: {str(e)}")
                progress_bar.progress(0)
        
        elif 'metrics' not in st.session_state:
            st.info("👆 Upload an image and click Analyze to get started")
    
    # Results display
    if 'metrics' in st.session_state:
        st.markdown("---")
        st.markdown("## 📈 Quantitative Analysis")
        
        metrics = st.session_state['metrics']
        cell_data = st.session_state['cell_data']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Cells",
                f"{metrics['total_cells']:,}",
                help="Number of cells detected"
            )
        
        with col2:
            health_score = metrics['avg_health_score']
            st.metric(
                "Avg Health",
                f"{health_score:.0f}/100",
                delta=f"{health_score-75:.0f}" if health_score >= 75 else f"{health_score-75:.0f}",
                delta_color="normal" if health_score >= 75 else "inverse",
                help="Average cell health score"
            )
        
        with col3:
            st.metric(
                "Healthy Cells",
                f"{metrics['healthy_percentage']:.0f}%",
                help="Percentage with health ≥ 75"
            )
        
        with col4:
            st.metric(
                "Avg Area",
                f"{metrics['avg_area']:.0f} px²",
                help="Average cell area"
            )
        
        # Visualizations
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔬 Segmentation", "🤖 AI Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                health_fig = create_modern_chart(cell_data, "health")
                if health_fig:
                    st.plotly_chart(health_fig, use_container_width=True)
            
            with col2:
                morph_fig = create_modern_chart(cell_data, "morphology")
                if morph_fig:
                    st.plotly_chart(morph_fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(st.session_state['image_path'], use_container_width=True)
            
            with col2:
                st.markdown(f"**Segmentation ({metrics['total_cells']} cells)**")
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(st.session_state['masks'], cmap='nipy_spectral')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        
        with tab3:
            if 'narrative' in st.session_state:
                st.markdown("### 🤖 AI-Generated Figure Legend")
                st.info(st.session_state['narrative'])
            else:
                st.warning("⚠️ Enter OpenAI API key to generate AI narrative")
        
        # Export
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = pd.DataFrame(cell_data).to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                csv_data,
                f"cellvision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = json.dumps({'metrics': metrics, 'cell_data': cell_data}, indent=2)
            st.download_button(
                "📄 Download JSON",
                json_data,
                f"cellvision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        with col3:
            if 'narrative' in st.session_state:
                st.download_button(
                    "📝 Download Legend",
                    st.session_state['narrative'],
                    f"legend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
