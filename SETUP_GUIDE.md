# Setup Guide for Vector Quantization Educational Repository

This guide helps you set up the repository to run the educational examples and use the vector quantization implementations.

## 🐍 Python Environment Setup

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv vq_env

# Activate it
source vq_env/bin/activate  # On Linux/Mac
# OR
vq_env\Scripts\activate     # On Windows

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn tqdm Pillow tensorboard
```

### Option 2: Using Conda

```bash
# Create conda environment
conda create -n vq_env python=3.9

# Activate it
conda activate vq_env

# Install PyTorch
conda install pytorch torchvision -c pytorch

# Install other dependencies
pip install matplotlib scikit-learn tqdm Pillow tensorboard
```

### Option 3: System Installation (if allowed)

```bash
# Install dependencies system-wide (use with caution)
pip3 install --user torch torchvision numpy matplotlib scikit-learn tqdm Pillow tensorboard
```

## 📦 Package Installation

### Option 1: Development Installation (Recommended)

```bash
cd vector-quantization-educational
pip install -e .
```

### Option 2: Manual Path Setup

If you can't install the package, add this to your Python scripts:

```python
import sys
import os
from pathlib import Path

# Add the source directory to Python path
repo_root = Path(__file__).parent.parent  # Adjust if needed
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

# Now you can import
from vector_quantization import VectorQuantizer, VQVAE, RQVAE, RQKMeans
```

### Option 3: Using the Setup Script

```bash
python3 install_local.py
```

## 🧪 Testing Your Setup

Run the quick test to verify everything works:

```bash
python3 quick_test.py
```

Expected output:
```
🎓 Vector Quantization Educational Package - Quick Test
============================================================
🧪 Testing basic imports...
✅ VectorQuantizer imported successfully
✅ VQVAE imported successfully
✅ RQVAE imported successfully
✅ RQKMeans imported successfully

🔬 Testing basic functionality...
  Testing VectorQuantizer...
    ✅ VQ output shape: torch.Size([1, 2, 2, 4]), loss: 0.xxxx, perplexity: x.xx
  ...
🎉 All tests passed! Your setup is working correctly.
```

## 🎓 Running Examples

Once setup is complete, try the educational examples:

```bash
cd examples

# Basic vector quantization concepts
python3 basic_vq_demo.py

# VQ-VAE image reconstruction
python3 vqvae_image_demo.py

# Compare VQ-VAE vs RQ-VAE
python3 rqvae_comparison.py

# RQ-K-means clustering
python3 rq_kmeans_demo.py
```

## 🚨 Troubleshooting

### Import Error: "No module named 'vector_quantization'"

**Solution 1:** Install the package
```bash
pip install -e .
```

**Solution 2:** Add path manually (add to your script)
```python
import sys
sys.path.append('path/to/vector-quantization-educational/src')
```

### Import Error: "No module named 'torch'"

**Solution:** Install PyTorch
```bash
pip install torch torchvision
```

### Permission Error: "externally-managed-environment"

**Solution:** Use virtual environment
```bash
python3 -m venv vq_env
source vq_env/bin/activate
pip install torch torchvision numpy matplotlib scikit-learn
```

### Display Issues with Matplotlib

**Solution:** For headless systems or Jupyter notebooks
```python
import matplotlib
matplotlib.use('Agg')  # Use before importing pyplot
import matplotlib.pyplot as plt
```

## 🎯 Quick Start Example

Once setup is complete, try this simple example:

```python
import torch
import sys
from pathlib import Path

# Setup path (if not installed)
sys.path.insert(0, str(Path.cwd() / "src"))

# Import and use
from vector_quantization import VectorQuantizer

# Create quantizer
vq = VectorQuantizer(num_embeddings=64, embedding_dim=8)

# Sample data
x = torch.randn(4, 4, 4, 8)  # batch=4, h=4, w=4, dim=8

# Apply quantization
quantized, loss, perplexity = vq(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {quantized.shape}")
print(f"VQ Loss: {loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
```

## 📚 Learning Path

1. **Start here:** `python3 examples/basic_vq_demo.py`
2. **Then:** `python3 examples/vqvae_image_demo.py`
3. **Advanced:** `python3 examples/rqvae_comparison.py`
4. **Clustering:** `python3 examples/rq_kmeans_demo.py`

## 💡 Tips

- All examples create visualizations in `examples/outputs/`
- Examples work with synthetic data (no external datasets needed)
- Each example includes detailed educational comments
- Modify parameters in examples to see different behaviors
- Check the README.md for mathematical background

## 🆘 Need Help?

1. Check this setup guide first
2. Run `python3 quick_test.py` to diagnose issues
3. Look at the error messages carefully
4. Make sure all dependencies are installed
5. Try the troubleshooting section above