"""
CellVision - Streamlit Web Application
AI-Powered Microscopy Analysis Interface
"""

import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import os
from analysis import analyze_microscopy_image, generate_analysis_narrative


def main():
    # Page config
    st.set_page_config(
        page_title="CellVision",
        page_icon="🔬",
        layout="wide"
    )

    # Header
    st.title("🔬 CellVision: AI-Powered Microscopy Analysis")
    st.markdown("Transform microscopy images into quantitative insights and publication-ready descriptions in seconds.")

    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        CellVision combines:
        - **CellPose**: Cell segmentation
        - **GPT-4 Vision**: Figure legend generation

        Upload an image to get started!
        """)

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a microscopy image",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Supported formats: PNG, JPEG, TIFF"
        )

        if uploaded_file:
            # Save uploaded file
            temp_path = f"temp_{datetime.now().timestamp()}.png"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display original
            st.image(temp_path, caption="Original Image", use_column_width=True)

    with col2:
        st.subheader("📊 Analysis Results")

        if uploaded_file and st.button("🚀 Analyze Image", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Segmentation
                status_text.text("Segmenting cells...")
                progress_bar.progress(33)

                try:
                    masks, metrics = analyze_microscopy_image(temp_path)

                    # Display segmentation
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(masks, cmap='tab20')
                    ax.set_title(f"Detected {metrics['total_cells']} cells")
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()

                    # Step 2: Metrics
                    status_text.text("Calculating metrics...")
                    progress_bar.progress(66)

                    # Display metrics in columns
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Cell Count", metrics['total_cells'])
                    metric_cols[1].metric("Avg Area", f"{metrics['avg_area']:.0f} px²")
                    metric_cols[2].metric("Density", f"{metrics['density']:.4f}")
                    metric_cols[3].metric("Circularity", f"{metrics['avg_circularity']:.2f}")

                    # Step 3: AI Analysis
                    status_text.text("Generating AI analysis...")
                    progress_bar.progress(90)

                    narrative = generate_analysis_narrative(temp_path, masks, metrics, api_key)

                    # Complete
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    # Display narrative
                    st.markdown("### 📝 Publication-Ready Figure Legend")
                    st.info(narrative)

                    # Download button
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=narrative,
                        file_name=f"figure_legend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

                    # Cleanup
                    os.remove(temp_path)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    status_text.text("Analysis failed!")
                    progress_bar.progress(0)


if __name__ == "__main__":
    main()
