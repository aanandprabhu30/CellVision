# 🚀 CellVision Quick Start Guide

**Get up and running in 5 minutes!**

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- OpenAI API key (for AI narration)
- (Optional) Google Colab Pro for training

---

## ⚡ 5-Minute Setup

### Step 1: Clone Repository (30 seconds)

```bash
git clone https://github.com/aanandprabhu30/CellVision.git
cd CellVision
```

### Step 2: Create Virtual Environment (1 minute)

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate     # On Windows
```

### Step 3: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

**What gets installed:**
- cellpose (cell segmentation)
- streamlit (web interface)
- openai (AI narration)
- plotly (interactive charts)
- scikit-image, opencv-python (image processing)
- pandas, numpy (data analysis)
- fpdf2 (PDF reports)

### Step 4: Get OpenAI API Key (1 minute)

```bash
# 1. Go to: https://platform.openai.com/api-keys
# 2. Click "Create new secret key"
# 3. Copy the key (starts with sk-...)
# 4. Create .env file:

cp .env.example .env

# 5. Edit .env and add your key:
# OPENAI_API_KEY=sk-your-key-here
```

### Step 5: Run the App! (30 seconds)

```bash
streamlit run app_enhanced.py
```

**Browser opens automatically at http://localhost:8501**

---

## 🎯 First Analysis (2 minutes)

### Option A: Use Demo Images

1. **Open the app** (should be at http://localhost:8501)
2. **In the sidebar**, scroll to "📚 Demo Images"
3. **Select** any demo image (e.g., "hela_cells_fluorescence.jpg")
4. **Click** "🚀 Analyze Image"
5. **Wait** ~30 seconds
6. **Explore** results:
   - Cell count and metrics
   - Health score distribution
   - Interactive visualizations
   - AI-generated legend

### Option B: Upload Your Own Image

1. **Click** "Browse files" in main area
2. **Select** your microscopy image (PNG, JPG, TIFF)
3. **Enter** OpenAI API key in sidebar (if not in .env)
4. **Click** "🚀 Analyze Image"
5. **View** comprehensive results

---

## 📊 Understanding the Results

### Top Metrics Bar
- **Total Cells**: Number of cells detected
- **Avg Health Score**: 0-100 scale (75+ = healthy)
- **Healthy Cells %**: Percentage in good condition
- **Avg Area**: Average cell size in pixels²
- **Distribution**: Spatial pattern (clustered/uniform)

### Visualizations
1. **Segmentation**: Color-coded cell boundaries
2. **Health Distribution**: Histogram of health scores
3. **Morphology Pie Chart**: Cell type breakdown
4. **Feature Scatter**: Area vs circularity plot

### Export Options
- **📊 CSV**: Detailed per-cell data
- **📄 JSON**: Structured analysis results
- **📝 TXT**: AI-generated figure legend

---

## 🎓 Training Your Own Model (Optional)

**Time Required: 2-3 hours**
**Cost: $10 (Colab Pro subscription)**

### Step 1: Get Colab Pro

```
1. Go to: https://colab.research.google.com/
2. Click "Upgrade to Colab Pro" ($10/month)
3. Subscribe and confirm
```

### Step 2: Upload Notebook

```
1. In Colab: File → Upload notebook
2. Select: train_on_colab.ipynb (from your CellVision folder)
3. Wait for upload to complete
```

### Step 3: Enable GPU

```
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. GPU type → T4 (or better)
4. Save
```

### Step 4: Run All Cells

```
1. Runtime → Run all
2. Approve permissions if asked
3. Wait for dataset download (~10 minutes)
4. Training starts automatically (~2 hours)
5. Watch progress bars
```

### Step 5: Download Trained Model

```
1. Last cell downloads cellvision_model.zip
2. Extract the zip file
3. Copy cellvision_finetuned.pth to CellVision/models/
4. Restart app - it automatically uses the fine-tuned model!
```

---

## 📥 Where to Get Training Data

### LIVECell Dataset (Recommended)

**Automatic (in Colab notebook):**
- Just run the cells - data downloads automatically!

**Manual Download:**
```bash
# Option 1: Direct download
wget https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/images.zip

# Option 2: From website
# Visit: https://sartorius-research.github.io/LIVECell/
# Click "Download" button
# Extract to /content/data/livecell/ in Colab
```

**What You Get:**
- 1.6 million cells
- 8 cancer cell types (A549, MCF7, HeLa, etc.)
- 5,239 high-quality images
- Expert-validated annotations

### BBBC Datasets (Alternative)

**Website:** https://bbbc.broadinstitute.org/

**Recommended:**
- **BBBC038**: Nuclei in histology (670 images)
  - URL: https://bbbc.broadinstitute.org/BBBC038
- **BBBC039**: Nuclei segmentation (200 images)
  - URL: https://bbbc.broadinstitute.org/BBBC039

**Download:**
```bash
1. Visit the URL above
2. Click "Download images" and "Download masks"
3. Extract to /content/data/bbbc/ in Colab
```

---

## 🐳 Docker Deployment (Alternative)

**If you prefer Docker:**

```bash
# 1. Set API key
export OPENAI_API_KEY="your_key_here"

# 2. Run with Docker Compose
docker-compose up -d

# 3. Access app
# Open: http://localhost:8501

# 4. View logs
docker-compose logs -f

# 5. Stop app
docker-compose down
```

---

## 🔧 Common Issues & Solutions

### Issue: "streamlit: command not found"
```bash
# Solution: Activate virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### Issue: "ModuleNotFoundError: No module named 'cellpose'"
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: "OpenAI API error: Invalid API key"
```bash
# Solution: Check your API key
# 1. Verify key at: https://platform.openai.com/api-keys
# 2. Make sure it's in .env file or entered in sidebar
# 3. Key should start with: sk-
```

### Issue: "GPU not available" in Colab
```bash
# Solution: Enable GPU
# Runtime → Change runtime type → Hardware accelerator → GPU
```

### Issue: "Port 8501 already in use"
```bash
# Solution: Use different port
streamlit run app_enhanced.py --server.port=8502
```

### Issue: "CellPose model download fails"
```bash
# Solution: Pre-download model
python3 -c "from cellpose import models; models.CellposeModel(model_type='cyto2')"
```

---

## 📁 File Overview

**Main Files to Use:**
- `app_enhanced.py` - **RUN THIS** (enhanced UI)
- `train_on_colab.ipynb` - Upload to Colab for training

**Data:**
- `demo_images/` - 5 real microscopy images (already included!)
- `models/` - Place your fine-tuned model here

**Documentation:**
- `README.md` - Main documentation
- `DEPLOYMENT.md` - Cloud deployment guide
- `MODEL_ENHANCEMENT_GUIDE.md` - Advanced model strategies

---

## 🎬 Demo Preparation

### For Hackathon/Presentation:

```bash
# 1. Test everything
streamlit run app_enhanced.py

# 2. Load each demo image and verify:
# - Segmentation works
# - Metrics display correctly
# - Visualizations render
# - AI legend generates

# 3. Practice 90-second pitch:
# - Problem (15s)
# - Solution demo (45s)
# - Technical depth (15s)
# - Impact (15s)

# 4. Prepare backup:
# - Take screenshots of results
# - Export sample PDF report
# - Have slides ready
```

---

## 🎯 Next Steps

### Beginner Path:
1. ✅ Run app with demo images
2. ✅ Upload your own images
3. ✅ Explore all features
4. ⏳ Read documentation

### Advanced Path:
1. ✅ Train model on Colab
2. ✅ Validate on multiple datasets
3. ✅ Deploy with Docker
4. ✅ Customize for your cell types

### Hackathon Path:
1. ✅ Test all demo images
2. ✅ Practice pitch
3. ✅ Prepare backup materials
4. ✅ WIN! 🏆

---

## 📞 Need Help?

- **Documentation**: See README.md
- **Training**: See train_on_colab.ipynb
- **Deployment**: See DEPLOYMENT.md
- **Models**: See MODEL_ENHANCEMENT_GUIDE.md
- **Issues**: Open issue on GitHub

---

## ✅ Quick Checklist

Before demo/hackathon:

- [ ] App runs successfully
- [ ] All 5 demo images tested
- [ ] OpenAI API key configured
- [ ] All visualizations work
- [ ] Export functions tested
- [ ] (Optional) Model trained on Colab
- [ ] 90-second pitch practiced
- [ ] Backup screenshots ready

---

## 🎊 You're Ready!

**Congratulations! You now have a hackathon-winning microscopy analysis platform.**

**Commands to remember:**
```bash
# Run app
streamlit run app_enhanced.py

# Train model (in Colab)
# Upload: train_on_colab.ipynb

# Deploy with Docker
docker-compose up -d
```

**Now go analyze some cells! 🔬✨**

---

<div align="center">
  <strong>Questions? Check README.md or open an issue on GitHub</strong>
</div>
