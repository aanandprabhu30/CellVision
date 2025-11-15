# CellVision Deployment Guide

Complete guide for deploying CellVision locally, on cloud, or for hackathon demos.

---

## 🚀 Quick Start (Local Development)

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aanandprabhu30/CellVision.git
cd CellVision

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the application
streamlit run app_enhanced.py
```

The application will open in your browser at `http://localhost:8501`

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# 2. Build and run
docker-compose up -d

# 3. Access the application
# Open http://localhost:8501 in your browser

# 4. View logs
docker-compose logs -f

# 5. Stop the application
docker-compose down
```

### Using Docker Directly

```bash
# 1. Build the image
docker build -t cellvision:latest .

# 2. Run the container
docker run -d \
  -p 8501:8501 \
  -e OPENAI_API_KEY="your_api_key_here" \
  -v $(pwd)/demo_images:/app/demo_images:ro \
  --name cellvision \
  cellvision:latest

# 3. Access at http://localhost:8501
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud (Easiest)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set `app_enhanced.py` as the main file
5. Add `OPENAI_API_KEY` in Secrets management
6. Deploy!

**Secrets format** (in Streamlit Cloud dashboard):
```toml
OPENAI_API_KEY = "your_api_key_here"
```

### Heroku

```bash
# 1. Create Procfile
echo "web: streamlit run app_enhanced.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# 2. Create Heroku app
heroku create your-cellvision-app

# 3. Set environment variables
heroku config:set OPENAI_API_KEY="your_api_key_here"

# 4. Deploy
git push heroku main

# 5. Open app
heroku open
```

### AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker ubuntu

# 4. Clone and deploy
git clone https://github.com/aanandprabhu30/CellVision.git
cd CellVision
export OPENAI_API_KEY="your_api_key_here"
docker-compose up -d

# 5. Configure security group to allow port 8501
# Access at http://your-instance-ip:8501
```

### Google Cloud Run

```bash
# 1. Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cellvision

# 2. Deploy to Cloud Run
gcloud run deploy cellvision \
  --image gcr.io/YOUR_PROJECT_ID/cellvision \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY="your_api_key_here"

# 3. Access the provided URL
```

---

## 🎬 Hackathon Demo Setup

### Pre-Demo Checklist

- [ ] Test with all demo images
- [ ] Verify OpenAI API key has sufficient credits
- [ ] Clear browser cache
- [ ] Close unnecessary applications
- [ ] Test internet connection
- [ ] Have backup screenshots ready
- [ ] Practice 90-second pitch

### Demo Mode Setup

```bash
# 1. Run locally for best performance
streamlit run app_enhanced.py

# 2. Pre-load demo images
# Demo images are in demo_images/ folder

# 3. Test each demo scenario:
# - HeLa cells (fluorescence)
# - A549 lung cancer cells
# - Dense cell culture
# - Phase contrast imaging

# 4. Have results pre-cached if possible
```

### Demo Script (90 seconds)

**0-15s: Hook**
> "Researchers waste 30% of their time manually counting cells. One image takes 2 hours to analyze. Watch this..."

**15-30s: Upload & Analyze**
> "Upload real cancer cell image... Click analyze... [Wait for results]"

**30-60s: Show Results**
> "234 cells detected instantly. Health scores calculated. Spatial patterns analyzed. AI-generated publication-ready legend."

**60-75s: Highlight Features**
> "20+ metrics. Interactive visualizations. Export to PDF. What took 2 hours now takes 30 seconds."

**75-90s: Impact**
> "Democratizes expert-level analysis. Any researcher can produce Nature-quality results. Seeking clinical partners for validation."

### Backup Plan

If live demo fails:
1. Show pre-recorded video
2. Use pre-generated screenshots
3. Show PDF report example
4. Explain architecture with slides

---

## 🔧 Troubleshooting

### Common Issues

**Issue: CellPose model download fails**
```bash
# Solution: Pre-download models
python3 -c "from cellpose import models; models.CellposeModel(model_type='cyto2')"
```

**Issue: GPU not detected**
```bash
# Solution: Install CUDA toolkit or disable GPU
# In app, set use_gpu=False in sidebar settings
```

**Issue: Out of memory**
```bash
# Solution: Reduce image size or use CPU mode
# Resize images to max 2048x2048 pixels
```

**Issue: Streamlit port already in use**
```bash
# Solution: Use different port
streamlit run app_enhanced.py --server.port=8502
```

**Issue: OpenAI API rate limit**
```bash
# Solution: Add retry logic or use cached results
# Check API usage at platform.openai.com
```

---

## 📊 Performance Optimization

### For Faster Processing

1. **Use GPU**: Enable GPU acceleration in settings (10x faster)
2. **Reduce Image Size**: Resize to 1024x1024 for faster processing
3. **Cache Results**: Results are stored in session state
4. **Pre-load Model**: Model loads once and stays in memory

### For Production

1. **Use Redis for Caching**: Cache analysis results
2. **Load Balancing**: Deploy multiple instances behind nginx
3. **CDN for Images**: Use CloudFront or similar for demo images
4. **Database**: Store results in PostgreSQL for history

---

## 🔐 Security Best Practices

### API Key Management

```bash
# NEVER commit .env file
echo ".env" >> .gitignore

# Use environment variables
export OPENAI_API_KEY="your_key"

# For production, use secrets management:
# - AWS Secrets Manager
# - Google Secret Manager
# - HashiCorp Vault
```

### Production Security

1. Enable HTTPS (use Let's Encrypt)
2. Add authentication (Streamlit supports OAuth)
3. Rate limiting (use nginx or API gateway)
4. Input validation (check file types and sizes)
5. Regular security updates

---

## 📈 Monitoring

### Health Checks

```bash
# Check if app is running
curl http://localhost:8501/_stcore/health

# View logs
docker-compose logs -f cellvision

# Monitor resource usage
docker stats cellvision
```

### Metrics to Track

- Response time per analysis
- Number of analyses per day
- API usage and costs
- Error rates
- User engagement

---

## 🆘 Support

### Getting Help

- **Documentation**: See README.md and QUICKSTART.md
- **Issues**: Open issue on GitHub
- **Community**: Join discussions on GitHub

### Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

See LICENSE file for details.

---

## 🎯 Next Steps After Deployment

1. **Test thoroughly** with various cell types
2. **Gather feedback** from users
3. **Monitor performance** and optimize
4. **Add features** based on user needs
5. **Scale infrastructure** as usage grows

---

**Ready to deploy? Start with the Quick Start section above!** 🚀
