"""
Vector Quantization (VQ) Implementation

Vector Quantization is a fundamental technique in signal processing and machine learning
that maps high-dimensional continuous vectors to a finite set of discrete vectors
(called codewords or codebook entries).

Key Concepts:
1. Codebook: A learnable dictionary of vectors (embeddings)
2. Quantization: Finding the nearest codebook vector for each input
3. Straight-through Estimator: Allows gradients to flow through the discrete quantization step

Mathematical Formulation:
- Given input z of shape (batch, height, width, channels)
- Codebook C of shape (codebook_size, embedding_dim)
- Quantized output q = argmin_c ||z - c||² for each spatial location

The VQ operation is non-differentiable, so we use the straight-through estimator:
- Forward: q = codebook[argmin_c ||z - c||²]
- Backward: ∇z = ∇q (copy gradients from quantized to input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantization Layer

    This module implements the core vector quantization operation used in VQ-VAE
    and other discrete representation learning methods.

    The quantization process works as follows:
    1. Flatten input tensor to (batch*height*width, channels)
    2. Compute distances to all codebook vectors
    3. Find nearest codebook vector for each input vector
    4. Replace input vectors with their nearest codebook vectors
    5. Apply straight-through estimator for gradient computation

    Args:
        num_embeddings (int): Size of the codebook (number of discrete codes)
        embedding_dim (int): Dimensionality of each code vector
        commitment_cost (float): Weight for commitment loss (default: 0.25)
        decay (float): Exponential moving average decay for codebook updates (default: 0.99)
        epsilon (float): Small constant for numerical stability (default: 1e-5)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super(VectorQuantizer, self).__init__()

        # Store hyperparameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize the codebook with random vectors
        # Shape: (num_embeddings, embedding_dim)
        # Each row represents one code vector in the vocabulary
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

        # For exponential moving average updates of codebook
        # These buffers track usage statistics and moving averages
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of Vector Quantization

        Args:
            inputs: Input tensor of shape (batch, height, width, channels) or
                   (batch, sequence_length, channels)

        Returns:
            quantized: Quantized tensor with same shape as input
            loss: VQ loss (commitment loss + codebook loss)
            perplexity: Measure of codebook utilization

        The quantization process:
        1. Reshape input to (batch_size * spatial_dims, embedding_dim)
        2. Calculate distances to all codebook vectors
        3. Find nearest neighbors using argmin
        4. Retrieve quantized vectors from codebook
        5. Apply straight-through estimator
        6. Calculate VQ losses for training
        """
        # Convert inputs to float32 for numerical stability
        input_shape = inputs.shape

        # Flatten input: (batch, height, width, channels) -> (batch*height*width, channels)
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate L2 distances between input vectors and codebook vectors
        # Using the identity: ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Find the closest codebook vector for each input vector
        # encoding_indices: (batch*height*width,) containing indices of nearest codes
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Create one-hot encodings for the selected codes
        # Shape: (batch*height*width, num_embeddings)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Retrieve the quantized vectors from the codebook
        # This is the discrete/quantized representation
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Update codebook using exponential moving average (only during training)
        if self.training:
            self._update_codebook(flat_input, encodings)

        # Calculate Vector Quantization losses

        # 1. Commitment loss: Encourages encoder output to stay close to chosen codebook vector
        #    This prevents the encoder from "wandering away" from the codebook
        commitment_loss = F.mse_loss(quantized.detach(), inputs)

        # 2. Codebook loss: Encourages codebook vectors to stay close to encoder outputs
        #    This is handled by the EMA updates during training
        #    In standard VQ-VAE, this would be F.mse_loss(quantized, inputs.detach())
        codebook_loss = F.mse_loss(quantized, inputs.detach())

        # Total VQ loss combines both terms
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Apply straight-through estimator
        # Forward: use quantized values
        # Backward: copy gradients from quantized to inputs
        quantized = inputs + (quantized - inputs).detach()

        # Calculate perplexity as a measure of codebook utilization
        # Higher perplexity means more codebook vectors are being used
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, vq_loss, perplexity

    def _update_codebook(self, flat_input: Tensor, encodings: Tensor):
        """
        Update codebook vectors using exponential moving average

        This update rule helps stabilize training by smoothly updating
        the codebook vectors based on the moving average of assigned vectors.

        Args:
            flat_input: Flattened input vectors (batch*spatial, embedding_dim)
            encodings: One-hot encoding matrix (batch*spatial, num_embeddings)
        """
        # Count how many times each codebook vector was selected
        cluster_size = torch.sum(encodings, 0)

        # Calculate the sum of input vectors assigned to each codebook entry
        dw = torch.matmul(encodings.t(), flat_input)

        # Update moving averages
        self._ema_cluster_size = self.decay * self._ema_cluster_size + (1 - self.decay) * cluster_size
        self._ema_w = self.decay * self._ema_w + (1 - self.decay) * dw

        # Calculate updated codebook vectors
        n = torch.sum(self._ema_cluster_size.data)
        cluster_size = (self._ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

        normalized_cluster_size = cluster_size.unsqueeze(1)
        self.embedding.weight.data = self._ema_w / normalized_cluster_size

    def get_codebook_entry(self, indices: Tensor) -> Tensor:
        """
        Retrieve codebook vectors by their indices

        This is useful for reconstructing from discrete codes
        or analyzing the learned codebook.

        Args:
            indices: Tensor of codebook indices

        Returns:
            Corresponding codebook vectors
        """
        return self.embedding(indices)

    def get_distance_matrix(self, inputs: Tensor) -> Tensor:
        """
        Calculate distance matrix between inputs and all codebook vectors

        Useful for analysis and debugging.

        Args:
            inputs: Input tensor

        Returns:
            Distance matrix of shape (batch*spatial, num_embeddings)
        """
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        return distances