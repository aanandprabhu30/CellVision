# CellVision: Microscopy Auto-Analyst with AI Narration

Transform microscopy images into quantitative insights and publication-ready descriptions in 30 seconds.

## Problem

Cell biology researchers spend 5-10 hours per experiment manually analyzing microscopy images. CellVision automates:

- Cell counting and segmentation
- Morphology measurements
- Publication-quality figure legend generation

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get an OpenAI API Key

- Visit <https://platform.openai.com/api-keys>
- Create a new API key
- Save it for later use

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Use the Interface

1. Upload a microscopy image (PNG, JPG, or TIFF)
2. Enter your OpenAI API key in the sidebar
3. Click "Analyze Image"
4. Get instant results:
   - Cell segmentation
   - Quantitative metrics
   - AI-generated figure legend

## Project Structure

```bash
CellVision/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── test_cellpose.py       # Test CellPose installation
├── analysis.py            # Core analysis functions
└── app.py                 # Streamlit web interface
```

## Technical Details

### Core Analysis Pipeline

- **Segmentation**: CellPose for cell detection
- **Metrics**: Area, perimeter, circularity, density
- **AI Narration**: GPT-4 Vision for figure legends

### Success Metrics

- Process images in <30 seconds
- 90% cell count accuracy vs manual
- Support 512x512 to 2048x2048 pixel images

## Resources

- [CellPose Documentation](https://cellpose.readthedocs.io/)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Streamlit Docs](https://docs.streamlit.io/)

## Next Steps

See CellVision.pdf for the complete buildathon guide with:

- Hour-by-hour implementation roadmap
- Testing checklist
- Demo preparation guide
- Team task assignments
