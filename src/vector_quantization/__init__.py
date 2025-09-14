"""
Vector Quantization Educational Package

This package provides educational implementations of various vector quantization techniques:
- Basic Vector Quantization (VQ)
- Vector Quantized Variational Autoencoder (VQ-VAE)
- Residual Quantized VAE (RQ-VAE)
- Residual Quantized K-means (RQ-K-means)

All implementations include detailed comments explaining the theory and implementation details.
"""

from .vector_quantization import VectorQuantizer
from .vq_vae import VQVAE, VQVAEEncoder, VQVAEDecoder
from .rq_vae import RQVAE, ResidualVectorQuantizer
from .rq_kmeans import RQKMeans

__version__ = "1.0.0"
__all__ = [
    "VectorQuantizer",
    "VQVAE",
    "VQVAEEncoder",
    "VQVAEDecoder",
    "RQVAE",
    "ResidualVectorQuantizer",
    "RQKMeans"
]