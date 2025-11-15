"""
CellVision - Complete Single-File Application
AI-Powered Microscopy Analysis

Usage: streamlit run cellvision.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import base64
import warnings
warnings.filterwarnings('ignore')

# Imports for analysis
from cellpose import models
from skimage import io, exposure
from skimage.measure import regionprops_table
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CellVision - AI Microscopy Analysis",
    page_icon="🔬",
    layout="wide"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Modern gradient background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* White content card */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Hero header */
    .hero {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .hero h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero p {
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
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
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
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Error */
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
        background-color: #f1f5f9;
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

# Global model cache
_MODEL_CACHE = None

def get_model():
    """Get or create cached CellPose model"""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = models.Cellpose(gpu=False, model_type='cyto2')
    return _MODEL_CACHE


def preprocess_image(img):
    """Preprocess image for better segmentation"""
    if len(img.shape) == 3:
        img = np.mean(img, axis=2).astype(np.uint8)
    
    p2, p98 = np.percentile(img, (2, 98))
    img = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    return img


def calculate_health(area, circularity, solidity):
    """Calculate cell health score (0-100)"""
    # Ideal values
    ideal_area = 500
    ideal_circ = 0.8
    ideal_solid = 0.95
    
    # Normalize and score
    area_score = 100 * np.exp(-((area - ideal_area) / ideal_area) ** 2)
    circ_score = (circularity / ideal_circ) * 100
    solid_score = (solidity / ideal_solid) * 100
    
    # Weighted average
    health = (area_score * 0.3 + circ_score * 0.4 + solid_score * 0.3)
    
    return max(0, min(100, health))


def classify_morphology(health, eccentricity=0.5):
    """Classify cell morphology"""
    if health >= 80:
        return "Healthy"
    elif health >= 60:
        return "Stressed"
    elif health >= 40:
        return "Apoptotic"
    elif eccentricity > 0.8:
        return "Elongated"
    else:
        return "Irregular"


def analyze_image(image_path, diameter=60):
    """
    Analyze microscopy image
    
    Returns: masks, metrics, cell_data
    """
    # Load and preprocess
    img = io.imread(image_path)
    img_processed = preprocess_image(img)
    
    # Segment with CellPose
    model = get_model()
    masks, flows, styles = model.eval(
        img_processed,
        diameter=diameter,
        channels=[0, 0],
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    
    # Count cells
    cell_count = len(np.unique(masks)) - 1
    
    if cell_count == 0:
        return masks, {
            'total_cells': 0,
            'avg_area': 0,
            'avg_health_score': 0,
            'healthy_percentage': 0
        }, []
    
    # Extract properties
    props = regionprops_table(
        masks, img_processed,
        properties=['area', 'perimeter', 'solidity', 'eccentricity']
    )
    
    areas = props['area']
    perimeters = props['perimeter']
    solidities = props['solidity']
    eccentricities = props['eccentricity']
    
    # Calculate circularity
    circularities = 4 * np.pi * areas / (perimeters ** 2 + 1e-10)
    
    # Calculate health scores
    health_scores = []
    morphologies = []
    
    for i in range(len(areas)):
        health = calculate_health(areas[i], circularities[i], solidities[i])
        health_scores.append(health)
        morphologies.append(classify_morphology(health, eccentricities[i]))
    
    # Compile metrics
    healthy_count = sum(1 for h in health_scores if h >= 75)
    
    metrics = {
        'total_cells': cell_count,
        'avg_area': float(np.mean(areas)),
        'avg_health_score': float(np.mean(health_scores)),
        'healthy_cells': healthy_count,
        'healthy_percentage': (healthy_count / cell_count * 100),
        'avg_circularity': float(np.mean(circularities)),
        'avg_solidity': float(np.mean(solidities))
    }
    
    # Per-cell data
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
    
    return masks, metrics, cell_data


def generate_ai_narrative(image_path, masks, metrics, api_key):
    """Generate AI narrative using GPT-4o vision"""
    
    if not api_key or not api_key.startswith('sk-'):
        return "⚠️ No valid API key provided"
    
    try:
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        img = io.imread(image_path)
        ax1.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax1.set_title('Original', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2.imshow(masks, cmap='nipy_spectral')
        ax2.set_title(f'Segmented: {metrics["total_cells"]} cells', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        temp_path = f"temp_viz_{np.random.randint(10000)}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Encode to base64
        with open(temp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        # Create prompt
        prompt = f"""You are an expert cell biologist analyzing microscopy images.

**Image Analysis Results:**
- Total cells detected: {metrics['total_cells']}
- Average health score: {metrics['avg_health_score']:.1f}/100
- Healthy cells (≥75): {metrics['healthy_percentage']:.0f}%
- Average cell area: {metrics['avg_area']:.0f} pixels²
- Average circularity: {metrics['avg_circularity']:.2f}

**Task:** Write a professional, publication-quality figure legend (3-4 sentences) that:
1. Describes what type of cells/tissue you observe
2. Comments on the overall population health and morphology
3. Notes any significant patterns or abnormalities
4. Provides biological interpretation

Be specific, scientific, and insightful. Use proper terminology."""

        # Call GPT-4o
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ AI narration failed: {str(e)}"


# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_health_chart(cell_data):
    """Create health distribution chart"""
    if not cell_data:
        return None
    
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
        hovertemplate='Health: %{x:.0f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="<b>Cell Health Distribution</b>",
        xaxis_title="Health Score (0-100)",
        yaxis_title="Number of Cells",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    return fig


def create_morphology_chart(cell_data):
    """Create morphology pie chart"""
    if not cell_data:
        return None
    
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
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="<b>Cell Morphology Classification</b>",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=True
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Hero header
    st.markdown("""
    <div class="hero">
        <h1>🔬 CellVision</h1>
        <p>AI-Powered Microscopy Analysis in 30 Seconds</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # Demo images
        demo_dir = Path("data/demo_images")
        if demo_dir.exists():
            demo_images = list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.png"))
            if demo_images:
                st.markdown("**Quick Start:** Try a demo")
                demo_names = [img.name for img in demo_images]
                selected = st.selectbox(
                    "Select demo",
                    ["Upload your own..."] + demo_names,
                    label_visibility="collapsed"
                )
                
                if selected != "Upload your own...":
                    image_path = str([img for img in demo_images if img.name == selected][0])
                    st.image(image_path, use_container_width=True)
                else:
                    uploaded = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg', 'tif'])
                    if uploaded:
                        temp_path = f"temp_{datetime.now().timestamp()}.png"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded.getbuffer())
                        image_path = temp_path
                        st.image(image_path, use_container_width=True)
                    else:
                        image_path = None
            else:
                uploaded = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg', 'tif'])
                if uploaded:
                    temp_path = f"temp_{datetime.now().timestamp()}.png"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                    image_path = temp_path
                    st.image(image_path, use_container_width=True)
                else:
                    image_path = None
        else:
            uploaded = st.file_uploader("Choose image", type=['png', 'jpg', 'jpeg', 'tif'])
            if uploaded:
                temp_path = f"temp_{datetime.now().timestamp()}.png"
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                image_path = temp_path
                st.image(image_path, use_container_width=True)
            else:
                image_path = None
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Advanced Settings"):
            diameter = st.slider("Cell diameter (pixels)", 20, 150, 60)
            api_key = st.text_input("OpenAI API Key (for AI narration)", type="password", placeholder="sk-...")
        
        st.markdown("---")
        analyze_btn = st.button("🚀 Analyze Image", type="primary")
    
    with col2:
        st.markdown("### 📊 Results")
        
        if image_path and analyze_btn:
            progress = st.progress(0)
            status = st.empty()
            
            try:
                status.info("🔍 Analyzing cells...")
                progress.progress(30)
                
                masks, metrics, cell_data = analyze_image(image_path, diameter)
                
                progress.progress(70)
                
                st.session_state['masks'] = masks
                st.session_state['metrics'] = metrics
                st.session_state['cell_data'] = cell_data
                st.session_state['image_path'] = image_path
                
                if api_key:
                    status.info("🤖 Generating AI analysis...")
                    narrative = generate_ai_narrative(image_path, masks, metrics, api_key)
                    st.session_state['narrative'] = narrative
                
                progress.progress(100)
                status.success("✅ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                status.error(f"❌ Error: {str(e)}")
                progress.progress(0)
        
        elif 'metrics' not in st.session_state:
            st.info("👆 Upload an image and click Analyze")
    
    # Display results
    if 'metrics' in st.session_state:
        st.markdown("---")
        st.markdown("## 📈 Analysis Results")
        
        metrics = st.session_state['metrics']
        cell_data = st.session_state['cell_data']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cells", f"{metrics['total_cells']:,}")
        
        with col2:
            health = metrics['avg_health_score']
            st.metric("Avg Health", f"{health:.0f}/100", 
                     delta=f"{health-75:.0f}" if health >= 75 else f"{health-75:.0f}")
        
        with col3:
            st.metric("Healthy %", f"{metrics['healthy_percentage']:.0f}%")
        
        with col4:
            st.metric("Avg Area", f"{metrics['avg_area']:.0f} px²")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Charts", "🔬 Segmentation", "🤖 AI Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_health_chart(cell_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_morphology_chart(cell_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
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
                st.markdown("### 🤖 AI-Generated Analysis (GPT-4o)")
                st.info(st.session_state['narrative'])
            else:
                st.warning("⚠️ Enter API key to generate AI analysis")
        
        # Export
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = pd.DataFrame(cell_data).to_csv(index=False)
            st.download_button("📊 CSV", csv, f"cells_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        
        with col2:
            json_data = json.dumps({'metrics': metrics, 'cells': cell_data}, indent=2)
            st.download_button("📄 JSON", json_data, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with col3:
            if 'narrative' in st.session_state:
                st.download_button("📝 Legend", st.session_state['narrative'], 
                                 f"legend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


if __name__ == "__main__":
    main()
