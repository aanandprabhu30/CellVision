# CellVision Quick Start Guide

## ⚠️ SECURITY WARNING
**Never share your OpenAI API key publicly!** If you've exposed your key:
1. Go to https://platform.openai.com/api-keys
2. Revoke the exposed key immediately
3. Create a new key for your use

## Setup Steps

### 1. Activate Virtual Environment
```bash
cd /Users/aanandprabhu/Desktop/CellVision
source venv/bin/activate
```

### 2. Wait for CellPose Model Download (First Time Only)
The first time you use CellPose, it downloads a 1.15GB model file. This is happening in the background right now.

### 3. Test the System
```bash
python quick_start.py
```

When prompted, paste your OpenAI API key (the one you just shared should be rotated first!).

### 4. Run the Web Application
```bash
streamlit run app.py
```

Then:
1. Open the URL shown (usually http://localhost:8501)
2. Enter your OpenAI API key in the sidebar
3. Upload a microscopy image
4. Click "Analyze Image"

## Sample Images

If you don't have microscopy images, the quick_start.py script creates a synthetic one for testing.

For real images, try:
- **LIVECell Dataset**: https://sartorius-research.github.io/LIVECell/
- **BBBC Collection**: https://bbbc.broadinstitute.org/

## Troubleshooting

### "API key invalid"
- Make sure you have a valid OpenAI API key
- Ensure your account has GPT-4 Vision access

### "CellPose model not found"
- Wait for the initial model download to complete
- Check your internet connection

### "Module not found" errors
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## What's Next?

Once everything works:
1. Try different microscopy images
2. Adjust CellPose parameters in `analysis.py` (diameter, etc.)
3. Customize the GPT-4 Vision prompt for your specific use case
4. Add batch processing for multiple images

See the full buildathon guide in `CellVision.pdf` for advanced features and deployment.
