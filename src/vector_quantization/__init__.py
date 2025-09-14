"""
Vector Quantization Educational Package

This package provides educational implementations of representation learning techniques:

Foundation Models:
- AutoEncoder: Basic autoencoder for learning compressed representations
- VAE: Variational Autoencoder with probabilistic latent space

Vector Quantization Techniques:
- Basic Vector Quantization (VQ)
- Vector Quantized Variational Autoencoder (VQ-VAE)
- Residual Quantized VAE (RQ-VAE)
- Residual Quantized K-means (RQ-K-means)

All implementations include detailed comments explaining the theory and implementation details.
"""

from .autoencoder import AutoEncoder, AutoEncoderEncoder, AutoEncoderDecoder
from .vae import VAE, VAEEncoder, VAEDecoder
from .vector_quantization import VectorQuantizer
from .vq_vae import VQVAE, VQVAEEncoder, VQVAEDecoder
from .rq_vae import RQVAE, ResidualVectorQuantizer
from .rq_kmeans import RQKMeans

__version__ = "1.1.0"
__all__ = [
    # Foundation models
    "AutoEncoder",
    "AutoEncoderEncoder",
    "AutoEncoderDecoder",
    "VAE",
    "VAEEncoder",
    "VAEDecoder",
    # Vector quantization models
    "VectorQuantizer",
    "VQVAE",
    "VQVAEEncoder",
    "VQVAEDecoder",
    "RQVAE",
    "ResidualVectorQuantizer",
    "RQKMeans"
]