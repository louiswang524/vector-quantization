"""
Vector Quantized Variational Autoencoder (VQ-VAE) Implementation

VQ-VAE is a type of variational autoencoder that learns discrete latent representations
instead of continuous ones. This is achieved by using vector quantization in the bottleneck.

Key Components:
1. Encoder: Maps input to continuous latent space
2. Vector Quantizer: Discretizes the continuous representations
3. Decoder: Reconstructs input from discrete codes

Mathematical Framework:
- Input: x ∈ ℝ^(H×W×C)
- Encoder: E(x) = z_e ∈ ℝ^(h×w×d)
- Quantizer: Q(z_e) = z_q where z_q[i,j] = arg min_k ||z_e[i,j] - e_k||²
- Decoder: D(z_q) = x̂ ∈ ℝ^(H×W×C)

Loss Function:
L = ||x - x̂||² + ||sg[z_e] - z_q||² + β||z_e - sg[z_q]||²
where sg[·] is the stop-gradient operator

Paper Reference: "Neural Discrete Representation Learning" (van den Oord et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Any
from .vector_quantization import VectorQuantizer


class VQVAEEncoder(nn.Module):
    """
    VQ-VAE Encoder Network

    The encoder transforms input images into continuous latent representations
    that will later be quantized. It uses convolutional layers with residual
    connections to learn hierarchical features.

    Architecture:
    - Initial convolution to project to hidden dimensions
    - Series of residual blocks for feature extraction
    - Final convolution to project to embedding dimension
    - Each residual block contains two convolutions with ReLU activation

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB)
        hidden_dims (List[int]): Hidden dimensions for each layer
        embedding_dim (int): Output embedding dimension for quantization
        num_residual_layers (int): Number of residual blocks
        residual_hidden_dim (int): Hidden dimension within residual blocks
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [128, 256],
        embedding_dim: int = 64,
        num_residual_layers: int = 2,
        residual_hidden_dim: int = 32
    ):
        super(VQVAEEncoder, self).__init__()

        # Store configuration
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim

        # Build encoder layers
        modules = []

        # Initial convolution: project input to first hidden dimension
        # This layer captures basic features and sets up the feature extraction pipeline
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        )

        # Downsampling layers: progressively reduce spatial dimensions
        # Each layer doubles the number of channels while halving spatial resolution
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )

        # Add residual layers for better feature representation
        # Residual connections help with gradient flow and feature learning
        for _ in range(num_residual_layers):
            modules.append(ResidualBlock(hidden_dims[-1], residual_hidden_dim))

        # Final projection to embedding dimension
        # This maps the learned features to the space used for quantization
        modules.append(
            nn.Sequential(
                nn.Conv2d(hidden_dims[-1], embedding_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through encoder

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Encoded features of shape (batch, embedding_dim, h, w)
            where h and w are reduced by factor 2^(num_downsampling_layers)
        """
        return self.encoder(x)


class VQVAEDecoder(nn.Module):
    """
    VQ-VAE Decoder Network

    The decoder reconstructs images from quantized latent codes.
    It uses transposed convolutions (upsampling) with residual connections
    to progressively increase spatial resolution while reducing channels.

    Architecture mirrors the encoder in reverse:
    - Initial projection from embedding dimension
    - Series of residual blocks
    - Upsampling layers with transposed convolutions
    - Final convolution to output channels

    Args:
        embedding_dim (int): Input embedding dimension from quantizer
        hidden_dims (List[int]): Hidden dimensions (reversed from encoder)
        out_channels (int): Number of output channels (e.g., 3 for RGB)
        num_residual_layers (int): Number of residual blocks
        residual_hidden_dim (int): Hidden dimension within residual blocks
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [256, 128],
        out_channels: int = 3,
        num_residual_layers: int = 2,
        residual_hidden_dim: int = 32
    ):
        super(VQVAEDecoder, self).__init__()

        # Store configuration
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels

        # Build decoder layers
        modules = []

        # Initial projection from embedding dimension to first hidden dimension
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=1),
                nn.ReLU(inplace=True)
            )
        )

        # Add residual layers for better feature processing
        for _ in range(num_residual_layers):
            modules.append(ResidualBlock(hidden_dims[0], residual_hidden_dim))

        # Upsampling layers: progressively increase spatial dimensions
        # Each layer halves the number of channels while doubling spatial resolution
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.ReLU(inplace=True)
                )
            )

        # Final upsampling to output resolution and channels
        modules.append(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through decoder

        Args:
            x: Quantized features of shape (batch, embedding_dim, h, w)

        Returns:
            Reconstructed images of shape (batch, out_channels, height, width)
        """
        return self.decoder(x)


class ResidualBlock(nn.Module):
    """
    Residual Block for VQ-VAE

    A residual block consists of two convolutions with a skip connection.
    This design helps with gradient flow and allows training deeper networks.

    Architecture:
    Input -> Conv -> ReLU -> Conv -> Add with Input -> ReLU -> Output

    The first convolution changes the channel dimension, the second reverts it back.
    This creates a bottleneck that encourages efficient feature learning.

    Args:
        in_channels (int): Number of input channels
        hidden_channels (int): Number of hidden channels (bottleneck dimension)
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super(ResidualBlock, self).__init__()

        # Two convolutions with bottleneck architecture
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection

        Args:
            x: Input tensor

        Returns:
            Output tensor with same shape as input
        """
        residual = x

        # First convolution with ReLU activation
        out = F.relu(self.conv1(x), inplace=True)

        # Second convolution (no activation here)
        out = self.conv2(out)

        # Add residual connection and apply final ReLU
        out = F.relu(out + residual, inplace=True)

        return out


class VQVAE(nn.Module):
    """
    Complete VQ-VAE Model

    This class combines the encoder, vector quantizer, and decoder into a complete
    autoencoder model that learns discrete latent representations.

    The training objective consists of three terms:
    1. Reconstruction loss: ||x - D(Q(E(x)))||²
    2. VQ loss: ||sg[E(x)] - Q(E(x))||² + β||E(x) - sg[Q(E(x))]||²
    3. Commitment loss: Ensures encoder commits to codebook vectors

    Key Features:
    - Discrete latent space for interpretable representations
    - Straight-through gradient estimation for end-to-end training
    - Exponential moving average for stable codebook learning
    - Perplexity monitoring for codebook utilization analysis

    Args:
        in_channels (int): Number of input image channels
        embedding_dim (int): Dimension of quantized embeddings
        num_embeddings (int): Size of the discrete codebook
        hidden_dims (List[int]): Hidden dimensions for encoder/decoder
        num_residual_layers (int): Number of residual blocks
        residual_hidden_dim (int): Hidden dimension in residual blocks
        commitment_cost (float): Weight for commitment loss
        decay (float): EMA decay rate for codebook updates
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        hidden_dims: List[int] = [128, 256],
        num_residual_layers: int = 2,
        residual_hidden_dim: int = 32,
        commitment_cost: float = 0.25,
        decay: float = 0.99
    ):
        super(VQVAE, self).__init__()

        # Store hyperparameters
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Initialize encoder: x -> z_e
        self.encoder = VQVAEEncoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

        # Initialize vector quantizer: z_e -> z_q
        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay
        )

        # Initialize decoder: z_q -> x̂
        self.decoder = VQVAEDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=in_channels,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass through complete VQ-VAE

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Dictionary containing:
            - 'reconstructed': Reconstructed images
            - 'vq_loss': Vector quantization loss
            - 'perplexity': Codebook utilization measure
            - 'encodings': Continuous encoder outputs
            - 'quantized': Quantized representations
        """
        # Encode input to continuous latent space
        # Shape: (batch, embedding_dim, h, w)
        z_e = self.encoder(x)

        # Apply vector quantization
        # z_q has same shape as z_e but with discrete values from codebook
        z_q, vq_loss, perplexity = self.vq_layer(z_e)

        # Decode quantized representations back to image space
        # Shape: (batch, in_channels, height, width)
        x_recon = self.decoder(z_q)

        return {
            'reconstructed': x_recon,
            'vq_loss': vq_loss,
            'perplexity': perplexity,
            'encodings': z_e,
            'quantized': z_q
        }

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to continuous latent representations

        Args:
            x: Input images

        Returns:
            Continuous latent codes
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representations to images

        Args:
            z: Latent codes (continuous or quantized)

        Returns:
            Reconstructed images
        """
        return self.decoder(z)

    def quantize(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply vector quantization to continuous representations

        Args:
            z: Continuous latent codes

        Returns:
            Tuple of (quantized codes, vq loss, perplexity)
        """
        return self.vq_layer(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Complete reconstruction: encode -> quantize -> decode

        Args:
            x: Input images

        Returns:
            Reconstructed images
        """
        z_e = self.encode(x)
        z_q, _, _ = self.quantize(z_e)
        return self.decode(z_q)

    def get_codebook(self) -> Tensor:
        """
        Get the learned codebook vectors

        Returns:
            Codebook tensor of shape (num_embeddings, embedding_dim)
        """
        return self.vq_layer.embedding.weight.data

    def encode_to_indices(self, x: Tensor) -> Tensor:
        """
        Encode input to discrete codebook indices

        This is useful for compression and discrete latent analysis.

        Args:
            x: Input images

        Returns:
            Discrete indices of shape (batch, h, w)
        """
        z_e = self.encode(x)
        flat_z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Calculate distances to codebook
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True)
                    + torch.sum(self.vq_layer.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.vq_layer.embedding.weight.t()))

        # Get indices of nearest codebook vectors
        indices = torch.argmin(distances, dim=1)

        # Reshape back to spatial dimensions
        return indices.view(x.shape[0], z_e.shape[2], z_e.shape[3])

    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """
        Decode images from discrete codebook indices

        Args:
            indices: Codebook indices of shape (batch, h, w)

        Returns:
            Reconstructed images
        """
        # Get codebook vectors for indices
        z_q = self.vq_layer.get_codebook_entry(indices.view(-1))

        # Reshape to proper dimensions
        z_q = z_q.view(indices.shape[0], indices.shape[1], indices.shape[2], self.embedding_dim)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return self.decode(z_q)