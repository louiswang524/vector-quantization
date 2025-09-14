# Vector Quantization Educational Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)

A comprehensive educational implementation of Vector Quantization techniques, including VQ-VAE, RQ-VAE, and RQ-K-means, designed to help students and researchers understand these fundamental concepts in discrete representation learning.

## 🎓 Educational Objectives

This repository is designed to teach you:
- **Vector Quantization fundamentals** and the straight-through estimator
- **VQ-VAE architecture** and discrete latent representations
- **Residual Quantization principles** and hierarchical discrete representations
- **RQ-K-means clustering** and multi-stage quantization
- **Practical implementation details** with extensive code comments
- **Real-world applications** in compression and representation learning

## 📋 Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Techniques Implemented](#techniques-implemented)
- [Examples and Tutorials](#examples-and-tutorials)
- [API Reference](#api-reference)
- [Educational Resources](#educational-resources)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Vector Quantization (VQ) is a fundamental technique in signal processing and machine learning that maps continuous vectors to discrete representations. This repository implements several key variants:

### 🏗️ Architecture Overview

```
Input Data → Encoder → Vector Quantizer → Decoder → Reconstructed Data
                           ↓
                    Discrete Codes
```

### 🎯 Key Features

- **Educational Focus**: Extensive comments explaining theory and implementation
- **Multiple Techniques**: VQ, VQ-VAE, RQ-VAE, and RQ-K-means
- **Interactive Examples**: Hands-on demonstrations with visualizations
- **Performance Analysis**: Comprehensive comparison and analysis tools
- **Research-Ready**: Clean, modular code suitable for research extensions

## 📐 Mathematical Background

### Vector Quantization (VQ)

Vector Quantization discretizes continuous vectors by mapping them to the nearest entry in a learned codebook:

```
q(z) = argmin_k ||z - e_k||²
```

Where:
- `z` is the input vector
- `e_k` are the codebook entries
- `q(z)` is the quantized representation

### VQ-VAE (Vector Quantized Variational Autoencoder)

VQ-VAE combines variational autoencoders with vector quantization:

**Loss Function:**
```
L = ||x - D(q(E(x)))||² + ||sg[E(x)] - q(E(x))||² + β||E(x) - sg[q(E(x))]||²
```

Where:
- `E(x)` is the encoder output
- `D(·)` is the decoder
- `q(·)` is the vector quantizer
- `sg[·]` is the stop-gradient operator
- `β` is the commitment cost

### RQ-VAE (Residual Quantized VAE)

RQ-VAE applies quantization iteratively to residuals:

```
r₁ = E(x), q₁ = VQ₁(r₁)
r₂ = r₁ - q₁, q₂ = VQ₂(r₂)
...
z_q = q₁ + q₂ + ... + qₙ
```

### RQ-K-means (Residual Quantized K-means)

Extends K-means with residual quantization:

```
Stage 1: C₁ = K-means(X)
Stage 2: C₂ = K-means(X - X_quantized₁)
...
Final: X_approx = Σᵢ Cᵢ[assignments_i]
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy 1.21+
- Matplotlib 3.5+
- scikit-learn 1.1+

### Install from Source

```bash
git clone https://github.com/yourusername/vector-quantization-educational.git
cd vector-quantization-educational
pip install -r requirements.txt
pip install -e .
```

### Dependencies

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm Pillow tensorboard
```

## ⚡ Quick Start

### Basic Vector Quantization

```python
import torch
from vector_quantization import VectorQuantizer

# Create vector quantizer
vq_layer = VectorQuantizer(
    num_embeddings=512,    # Codebook size
    embedding_dim=64,      # Vector dimension
    commitment_cost=0.25   # Commitment loss weight
)

# Sample input (batch, height, width, channels)
x = torch.randn(32, 8, 8, 64)

# Apply quantization
quantized, vq_loss, perplexity = vq_layer(x)
print(f"VQ Loss: {vq_loss:.4f}, Perplexity: {perplexity:.2f}")
```

### VQ-VAE for Images

```python
from vector_quantization import VQVAE

# Create VQ-VAE model
model = VQVAE(
    in_channels=3,
    embedding_dim=64,
    num_embeddings=512,
    hidden_dims=[128, 256]
)

# Sample images (batch, channels, height, width)
images = torch.randn(16, 3, 32, 32)

# Forward pass
outputs = model(images)
reconstructed = outputs['reconstructed']
vq_loss = outputs['vq_loss']
print(f"Reconstruction shape: {reconstructed.shape}")
```

### RQ-VAE with Multiple Quantization Levels

```python
from vector_quantization import RQVAE

# Create RQ-VAE with 4 quantization levels
model = RQVAE(
    in_channels=3,
    embedding_dim=64,
    num_embeddings=512,
    num_quantizers=4  # 4 hierarchical levels
)

# Forward pass
outputs = model(images)
print(f"Quantization levels: {len(outputs['quantized_list'])}")
print(f"Perplexity per level: {outputs['perplexity_list']}")
```

### RQ-K-means Clustering

```python
from vector_quantization import RQKMeans
import numpy as np

# Generate sample data
X = np.random.randn(1000, 10)  # 1000 points, 10 dimensions

# Create and fit RQ-K-means
rq_kmeans = RQKMeans(
    n_clusters=64,    # Clusters per stage
    n_stages=4,       # Number of stages
    verbose=True
)

# Fit and transform
X_quantized = rq_kmeans.fit_transform(X)
print(f"Reconstruction error: {rq_kmeans.calculate_reconstruction_error(X):.6f}")
```

## 🛠️ Techniques Implemented

### 1. Vector Quantization (`VectorQuantizer`)

**Core Implementation:**
- Exponential Moving Average (EMA) for codebook updates
- Straight-through estimator for gradient computation
- Commitment loss for encoder stabilization
- Codebook utilization analysis

**Key Features:**
- Configurable codebook size and embedding dimension
- Multiple initialization strategies
- Comprehensive metrics (perplexity, utilization, etc.)

### 2. VQ-VAE (`VQVAE`)

**Architecture Components:**
- **Encoder**: Convolutional layers with residual blocks
- **Vector Quantizer**: Core VQ layer with discrete bottleneck
- **Decoder**: Transposed convolutions for reconstruction

**Training Objective:**
- Reconstruction loss (MSE)
- Vector quantization loss
- Commitment loss

### 3. RQ-VAE (`RQVAE`)

**Hierarchical Quantization:**
- Multiple VQ layers applied sequentially
- Each layer quantizes residuals from previous stages
- Improved reconstruction quality
- Analysis tools for hierarchical representations

### 4. RQ-K-means (`RQKMeans`)

**Multi-Stage Clustering:**
- Iterative K-means on residuals
- Configurable number of stages and clusters
- Comprehensive quality analysis
- Comparison with standard K-means

## 📚 Examples and Tutorials

### Interactive Demonstrations

1. **Basic VQ Demo** (`examples/basic_vq_demo.py`)
   - Fundamental vector quantization concepts
   - Codebook size effects
   - Training dynamics visualization
   - 2D data examples with plots

2. **VQ-VAE Image Demo** (`examples/vqvae_image_demo.py`)
   - Complete VQ-VAE training pipeline
   - Synthetic image generation
   - Reconstruction quality analysis
   - Discrete code visualization

3. **RQ-VAE Comparison** (`examples/rqvae_comparison.py`)
   - VQ-VAE vs RQ-VAE comparison
   - Hierarchical quantization analysis
   - Performance benchmarking
   - Multi-level reconstruction visualization

4. **RQ-K-means Demo** (`examples/rq_kmeans_demo.py`)
   - Comprehensive RQ-K-means analysis
   - Multiple dataset comparisons
   - Parameter sensitivity study
   - Stage-by-stage error reduction

### Running Examples

```bash
cd examples

# Basic concepts
python basic_vq_demo.py

# Image applications
python vqvae_image_demo.py

# Advanced comparisons
python rqvae_comparison.py
python rq_kmeans_demo.py
```

## 📖 API Reference

### VectorQuantizer

```python
VectorQuantizer(
    num_embeddings: int,        # Codebook size
    embedding_dim: int,         # Vector dimension
    commitment_cost: float = 0.25,  # Commitment loss weight
    decay: float = 0.99,        # EMA decay rate
    epsilon: float = 1e-5       # Numerical stability
)
```

**Methods:**
- `forward(inputs)` → `(quantized, loss, perplexity)`
- `get_codebook_entry(indices)` → `codebook_vectors`
- `get_distance_matrix(inputs)` → `distances`

### VQVAE

```python
VQVAE(
    in_channels: int = 3,
    embedding_dim: int = 64,
    num_embeddings: int = 512,
    hidden_dims: List[int] = [128, 256],
    num_residual_layers: int = 2,
    residual_hidden_dim: int = 32,
    commitment_cost: float = 0.25
)
```

**Methods:**
- `forward(x)` → `{'reconstructed', 'vq_loss', 'perplexity', ...}`
- `encode(x)` → `continuous_latents`
- `decode(z)` → `reconstructed_images`
- `encode_to_indices(x)` → `discrete_codes`
- `decode_from_indices(indices)` → `reconstructed_images`

### RQVAE

```python
RQVAE(
    in_channels: int = 3,
    embedding_dim: int = 64,
    num_embeddings: int = 512,
    num_quantizers: int = 4,
    # ... other parameters same as VQVAE
    shared_codebook: bool = False
)
```

**Methods:**
- `forward(x)` → `{'reconstructed', 'rq_loss', 'perplexity_list', ...}`
- `get_codes(x)` → `List[discrete_codes_per_level]`
- `analyze_quantization_levels(x)` → `detailed_analysis_dict`

### RQKMeans

```python
RQKMeans(
    n_clusters: int = 256,
    n_stages: int = 4,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
    verbose: bool = False
)
```

**Methods:**
- `fit(X)` → `self`
- `transform(X)` → `quantized_data`
- `fit_transform(X)` → `quantized_data`
- `get_codes(X)` → `List[codes_per_stage]`
- `analyze_quantization_quality(X)` → `quality_metrics_dict`

## 🎯 Educational Resources

### Key Concepts Explained

1. **Straight-Through Estimator**
   - Problem: VQ operation is non-differentiable
   - Solution: Copy gradients from output to input
   - Implementation in `vector_quantization.py:85-88`

2. **Codebook Learning**
   - Exponential Moving Average updates
   - Prevents codebook collapse
   - Implementation in `vector_quantization.py:120-135`

3. **Commitment Loss**
   - Ensures encoder commits to codebook vectors
   - Prevents encoder drift
   - Mathematical formulation and code explanation

4. **Residual Quantization**
   - Progressive error reduction
   - Hierarchical representation learning
   - Stage-by-stage analysis in examples

### Paper References

- **VQ-VAE**: [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) (van den Oord et al., 2017)
- **VQ-VAE-2**: [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446) (Razavi et al., 2019)
- **Residual VQ**: Various papers on residual/hierarchical quantization
- **Product Quantization**: [Product Quantization for Nearest Neighbor Search](https://hal.inria.fr/inria-00514462/) (Jégou et al., 2011)

### Learning Path

1. **Start with Basic VQ** (`examples/basic_vq_demo.py`)
   - Understand core concepts
   - See codebook learning in action
   - Experiment with parameters

2. **Move to VQ-VAE** (`examples/vqvae_image_demo.py`)
   - Learn autoencoder integration
   - Understand reconstruction vs compression trade-off
   - Analyze discrete representations

3. **Explore RQ-VAE** (`examples/rqvae_comparison.py`)
   - See hierarchical quantization benefits
   - Compare with standard VQ-VAE
   - Understand multi-level representations

4. **Master RQ-K-means** (`examples/rq_kmeans_demo.py`)
   - Apply concepts to clustering
   - Understand parameter effects
   - See real-world applications

## 🔬 Research Extensions

This codebase is designed to support research extensions:

### Possible Extensions

1. **New Quantization Schemes**
   - Product quantization
   - Gumbel quantization
   - Learned quantization

2. **Architecture Improvements**
   - Transformer-based encoders/decoders
   - Multi-scale architectures
   - Conditional quantization

3. **Applications**
   - Audio processing
   - Natural language processing
   - Video compression
   - Generative modeling

### Adding New Techniques

The modular design makes it easy to add new quantization methods:

```python
# Example: Custom quantizer
class CustomQuantizer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Initialize your quantizer

    def forward(self, x):
        # Implement your quantization logic
        return quantized, loss, metrics
```

## 📊 Performance Analysis

### Benchmarking Tools

Each implementation includes comprehensive analysis tools:

- **Reconstruction Quality**: MSE, PSNR, SSIM metrics
- **Codebook Utilization**: Perplexity, usage histograms
- **Compression Efficiency**: Bits per pixel, compression ratios
- **Training Dynamics**: Loss curves, convergence analysis

### Visualization Tools

Extensive plotting capabilities for educational purposes:

- 2D data visualization with quantization boundaries
- Training curve analysis
- Hierarchical reconstruction progression
- Codebook evolution over training
- Parameter sensitivity studies

## 🤝 Contributing

We welcome contributions to improve the educational value of this repository!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Add your improvements** (new examples, better explanations, bug fixes)
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

### Contribution Guidelines

- **Educational Focus**: Prioritize clarity and learning over performance
- **Code Comments**: Extensive documentation explaining the "why"
- **Examples**: Include runnable examples for new features
- **Testing**: Ensure all examples run correctly
- **Documentation**: Update README and docstrings

### Areas for Contribution

- Additional quantization techniques
- More diverse datasets and examples
- Improved visualizations
- Performance optimizations
- Bug fixes and improvements
- Educational content and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original VQ-VAE authors for the foundational work
- PyTorch community for the excellent deep learning framework
- Educational AI community for inspiration and feedback
- All contributors who help improve this resource

## 📞 Contact

For questions, suggestions, or educational discussions:

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Research**: Cite this repository if it helps your research

---

**Happy Learning! 🎓**

*This repository is designed to be your comprehensive guide to understanding vector quantization. Start with the basics, experiment with the examples, and build your intuition through hands-on coding!*