"""
Residual Quantized VAE (RQ-VAE) Implementation

RQ-VAE extends VQ-VAE by using multiple quantization layers in a residual manner.
Instead of quantizing the entire vector at once, RQ-VAE quantizes the residual
between the input and the previous quantization step iteratively.

Key Innovations:
1. Multi-level quantization: Multiple VQ layers applied sequentially
2. Residual quantization: Each layer quantizes the residual from previous layers
3. Improved reconstruction quality: Better approximation of continuous distributions
4. Hierarchical discrete representations: Different levels capture different details

Mathematical Framework:
Given input z₀ (encoder output):
- r₁ = z₀, q₁ = VQ₁(r₁), r₂ = r₁ - q₁
- q₂ = VQ₂(r₂), r₃ = r₂ - q₂
- ...
- qₙ = VQₙ(rₙ)
- Final quantized: z_q = q₁ + q₂ + ... + qₙ

This approach allows for:
- Better approximation of the original continuous vector
- Hierarchical representation learning
- Improved codebook utilization across different scales

Paper Reference: "Improved Vector Quantized Diffusion Models" and related works
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Any
from .vector_quantization import VectorQuantizer
from .vq_vae import VQVAEEncoder, VQVAEDecoder


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer

    This module implements residual quantization by applying multiple
    VQ layers sequentially, where each layer quantizes the residual
    from all previous quantization steps.

    The key insight is that instead of trying to quantize a continuous
    vector directly with a single codebook, we can iteratively quantize
    the "errors" or "residuals" left by previous quantization steps.

    Process:
    1. Start with original vector r₁ = z
    2. Quantize: q₁ = VQ₁(r₁)
    3. Compute residual: r₂ = r₁ - q₁
    4. Quantize residual: q₂ = VQ₂(r₂)
    5. Repeat for n_q layers
    6. Final result: z_q = q₁ + q₂ + ... + qₙ

    Args:
        num_quantizers (int): Number of quantization layers
        num_embeddings (int): Size of each codebook
        embedding_dim (int): Dimension of embeddings
        commitment_cost (float): Weight for commitment loss
        decay (float): EMA decay for codebook updates
        shared_codebook (bool): Whether to share codebook across layers
    """

    def __init__(
        self,
        num_quantizers: int = 4,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        shared_codebook: bool = False
    ):
        super(ResidualVectorQuantizer, self).__init__()

        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.shared_codebook = shared_codebook

        # Create quantizer layers
        if shared_codebook:
            # All quantizers share the same codebook
            # This reduces parameters but may limit expressiveness
            base_quantizer = VectorQuantizer(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay
            )
            self.quantizers = nn.ModuleList([base_quantizer for _ in range(num_quantizers)])
        else:
            # Each quantizer has its own codebook
            # This allows each level to specialize for different types of residuals
            self.quantizers = nn.ModuleList([
                VectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                    decay=decay
                ) for _ in range(num_quantizers)
            ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Forward pass through residual quantization

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            quantized: Final quantized representation (sum of all levels)
            total_loss: Combined loss from all quantization levels
            quantized_list: List of quantized vectors from each level
            perplexity_list: List of perplexity values from each level
        """
        quantized_list = []
        perplexity_list = []
        loss_list = []

        # Start with the original input as the first residual
        residual = x

        # Apply quantization layers sequentially
        for i, quantizer in enumerate(self.quantizers):
            # Quantize the current residual
            quantized_residual, vq_loss, perplexity = quantizer(residual)

            # Store results
            quantized_list.append(quantized_residual)
            perplexity_list.append(perplexity)
            loss_list.append(vq_loss)

            # Update residual for next iteration
            # The residual becomes what's left after this quantization step
            residual = residual - quantized_residual

        # Final quantized representation is the sum of all quantization levels
        # This reconstructs the original input as a sum of discrete components
        quantized = torch.stack(quantized_list, dim=0).sum(dim=0)

        # Total loss is the mean of all quantization losses
        # Each layer contributes equally to the training objective
        total_loss = torch.stack(loss_list).mean()

        return quantized, total_loss, quantized_list, perplexity_list

    def get_codes_and_reconstruct(self, x: Tensor) -> Tuple[List[Tensor], Tensor]:
        """
        Get discrete codes and reconstruct from them

        This method is useful for analysis and compression applications.

        Args:
            x: Input tensor

        Returns:
            codes_list: List of discrete codes from each quantization level
            reconstructed: Reconstructed tensor from discrete codes
        """
        codes_list = []
        residual = x

        # Extract discrete codes from each level
        for quantizer in self.quantizers:
            # Get distances and find nearest codes
            flat_residual = residual.view(-1, self.embedding_dim)
            distances = (torch.sum(flat_residual**2, dim=1, keepdim=True)
                        + torch.sum(quantizer.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_residual, quantizer.embedding.weight.t()))

            codes = torch.argmin(distances, dim=1)
            codes = codes.view(*residual.shape[:-1])
            codes_list.append(codes)

            # Get quantized version and update residual
            quantized_residual = quantizer.get_codebook_entry(codes.view(-1))
            quantized_residual = quantized_residual.view(*residual.shape)
            residual = residual - quantized_residual

        # Reconstruct from codes
        reconstructed = torch.zeros_like(x)
        for i, codes in enumerate(codes_list):
            quantized = self.quantizers[i].get_codebook_entry(codes.view(-1))
            quantized = quantized.view(*x.shape)
            reconstructed = reconstructed + quantized

        return codes_list, reconstructed

    def reconstruct_from_codes(self, codes_list: List[Tensor], original_shape: Tuple[int, ...]) -> Tensor:
        """
        Reconstruct tensor from list of discrete codes

        Args:
            codes_list: List of discrete codes from each level
            original_shape: Target shape for reconstruction

        Returns:
            Reconstructed tensor
        """
        reconstructed = torch.zeros(*original_shape, device=codes_list[0].device)

        for i, codes in enumerate(codes_list):
            quantized = self.quantizers[i].get_codebook_entry(codes.view(-1))
            quantized = quantized.view(*original_shape)
            reconstructed = reconstructed + quantized

        return reconstructed

    def get_codebook_usage(self, x: Tensor) -> List[Tensor]:
        """
        Analyze codebook usage across all quantization levels

        Args:
            x: Input tensor

        Returns:
            List of codebook usage histograms for each level
        """
        usage_list = []
        residual = x

        for quantizer in self.quantizers:
            flat_residual = residual.view(-1, self.embedding_dim)
            distances = (torch.sum(flat_residual**2, dim=1, keepdim=True)
                        + torch.sum(quantizer.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_residual, quantizer.embedding.weight.t()))

            codes = torch.argmin(distances, dim=1)

            # Count usage of each codebook entry
            usage = torch.bincount(codes, minlength=self.num_embeddings)
            usage_list.append(usage)

            # Update residual
            quantized_residual = quantizer.get_codebook_entry(codes)
            quantized_residual = quantized_residual.view(*residual.shape)
            residual = residual - quantized_residual

        return usage_list


class RQVAE(nn.Module):
    """
    Complete Residual Quantized VAE Model

    RQVAE combines an encoder, residual vector quantizer, and decoder
    to learn hierarchical discrete representations. The key advantage
    over standard VQ-VAE is better reconstruction quality through
    multi-level quantization.

    Architecture:
    Input -> Encoder -> Residual VQ -> Decoder -> Output

    The residual quantization provides several benefits:
    1. Better approximation of continuous distributions
    2. Hierarchical representation learning
    3. Improved reconstruction fidelity
    4. More efficient use of discrete codes

    Training Objective:
    L = ||x - x̂||² + Σᵢ VQ_loss_i
    where VQ_loss_i includes both commitment and codebook losses for level i

    Args:
        in_channels (int): Number of input channels
        embedding_dim (int): Dimension of quantized embeddings
        num_embeddings (int): Size of each codebook
        num_quantizers (int): Number of quantization levels
        hidden_dims (List[int]): Hidden dimensions for encoder/decoder
        num_residual_layers (int): Number of residual blocks
        residual_hidden_dim (int): Hidden dimension in residual blocks
        commitment_cost (float): Weight for commitment loss
        decay (float): EMA decay for codebook updates
        shared_codebook (bool): Whether to share codebook across levels
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        num_quantizers: int = 4,
        hidden_dims: List[int] = [128, 256],
        num_residual_layers: int = 2,
        residual_hidden_dim: int = 32,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        shared_codebook: bool = False
    ):
        super(RQVAE, self).__init__()

        # Store hyperparameters
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_quantizers = num_quantizers

        # Initialize encoder: x -> z_e
        # Uses the same architecture as VQ-VAE for fair comparison
        self.encoder = VQVAEEncoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

        # Initialize residual vector quantizer: z_e -> z_q
        self.rq_layer = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            shared_codebook=shared_codebook
        )

        # Initialize decoder: z_q -> x̂
        self.decoder = VQVAEDecoder(
            embedding_dim=embedding_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=in_channels,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim
        )

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """
        Forward pass through complete RQ-VAE

        Args:
            x: Input images of shape (batch, channels, height, width)

        Returns:
            Dictionary containing:
            - 'reconstructed': Reconstructed images
            - 'rq_loss': Residual quantization loss
            - 'perplexity_list': List of perplexity values for each level
            - 'encodings': Continuous encoder outputs
            - 'quantized': Final quantized representations
            - 'quantized_list': Quantized outputs from each level
        """
        # Encode input to continuous latent space
        z_e = self.encoder(x)

        # Apply residual vector quantization
        # This produces a hierarchical discrete representation
        z_q, rq_loss, quantized_list, perplexity_list = self.rq_layer(z_e)

        # Decode quantized representations back to image space
        x_recon = self.decoder(z_q)

        return {
            'reconstructed': x_recon,
            'rq_loss': rq_loss,
            'perplexity_list': perplexity_list,
            'encodings': z_e,
            'quantized': z_q,
            'quantized_list': quantized_list
        }

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to continuous latent representations"""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representations to images"""
        return self.decoder(z)

    def quantize(self, z: Tensor) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """Apply residual vector quantization"""
        return self.rq_layer(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        """Complete reconstruction: encode -> quantize -> decode"""
        z_e = self.encode(x)
        z_q, _, _, _ = self.quantize(z_e)
        return self.decode(z_q)

    def get_codes(self, x: Tensor) -> List[Tensor]:
        """
        Extract discrete codes from input

        Args:
            x: Input images

        Returns:
            List of discrete codes for each quantization level
        """
        z_e = self.encode(x)
        codes_list, _ = self.rq_layer.get_codes_and_reconstruct(z_e)
        return codes_list

    def reconstruct_from_codes(self, codes_list: List[Tensor], spatial_shape: Tuple[int, int]) -> Tensor:
        """
        Reconstruct images from discrete codes

        Args:
            codes_list: List of discrete codes for each level
            spatial_shape: Spatial dimensions (height, width) of latent space

        Returns:
            Reconstructed images
        """
        # Reconstruct quantized representation from codes
        batch_size = codes_list[0].shape[0]
        z_shape = (batch_size, spatial_shape[0], spatial_shape[1], self.embedding_dim)
        z_q = self.rq_layer.reconstruct_from_codes(codes_list, z_shape)

        return self.decode(z_q)

    def analyze_quantization_levels(self, x: Tensor) -> Dict[str, Any]:
        """
        Analyze the contribution of each quantization level

        Args:
            x: Input images

        Returns:
            Dictionary with detailed analysis of each quantization level
        """
        z_e = self.encode(x)
        z_q, rq_loss, quantized_list, perplexity_list = self.quantize(z_e)

        # Analyze reconstruction quality at each level
        reconstruction_errors = []
        cumulative_reconstructions = []

        cumulative_quantized = torch.zeros_like(z_e)
        for i, q_level in enumerate(quantized_list):
            cumulative_quantized = cumulative_quantized + q_level
            recon = self.decode(cumulative_quantized)
            cumulative_reconstructions.append(recon)

            # Calculate reconstruction error
            error = F.mse_loss(recon, x)
            reconstruction_errors.append(error.item())

        # Analyze codebook usage
        usage_stats = self.rq_layer.get_codebook_usage(z_e)

        return {
            'reconstruction_errors': reconstruction_errors,
            'cumulative_reconstructions': cumulative_reconstructions,
            'perplexity_list': [p.item() for p in perplexity_list],
            'codebook_usage': [usage.cpu().numpy() for usage in usage_stats],
            'total_loss': rq_loss.item()
        }

    def get_compression_ratio(self) -> float:
        """
        Calculate the compression ratio achieved by the model

        Returns:
            Compression ratio (original bits per pixel / compressed bits per pixel)
        """
        # Original: 8 bits per pixel for RGB (24 bits total)
        original_bpp = 8 * 3  # 24 bits per pixel

        # Compressed: log2(codebook_size) bits per code, num_quantizers codes per spatial location
        import math
        bits_per_code = math.log2(self.num_embeddings)
        compressed_bpp = bits_per_code * self.num_quantizers

        return original_bpp / compressed_bpp