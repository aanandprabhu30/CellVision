"""
Test CellPose installation
"""
from cellpose import models
import numpy as np

print("Testing CellPose installation...")

# Initialize model (use CPU for reliability)
print("Loading CellPose model...")
model = models.CellposeModel(gpu=False, model_type='cyto2')

# Create a simple test image
print("Creating test image...")
test_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

# Test segmentation
print("Running segmentation...")
masks, flows, styles, diams = model.eval(test_img, diameter=30)

print(f"✓ Success! Detected {len(np.unique(masks))-1} cells")
print("CellPose is working correctly!")
