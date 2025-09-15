"""
AutoEncoder Implementation with Educational Focus

AutoEncoders are fundamental neural network architectures that learn efficient
representations of data by compressing input into a lower-dimensional latent
space and then reconstructing the original input from this compressed representation.

Key Concepts:
1. Encoder: Maps input to latent representation (dimensionality reduction)
2. Decoder: Reconstructs input from latent representation (dimensionality expansion)
3. Bottleneck: The compressed latent space representation
4. Reconstruction Loss: Measures how well the output matches the input

Mathematical Framework:
- Encoder: z = f_encoder(x), where z ∈ ℝ^d (d < input_dim)
- Decoder: x̂ = f_decoder(z)
- Loss: L = ||x - x̂||² (reconstruction error)

Applications:
- Dimensionality reduction
- Data compression
- Denoising
- Feature learning
- Anomaly detection
- Foundation for more advanced models (VAE, VQ-VAE)

Educational Value:
This implementation focuses on clarity and understanding rather than
state-of-the-art performance, making it perfect for learning the fundamentals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Any


class AutoEncoderEncoder(nn.Module):
    """
    Encoder Network for AutoEncoder

    The encoder progressively reduces the spatial dimensions while increasing
    the number of channels, ultimately producing a compact latent representation.

    Architecture Pattern:
    Input → Conv + ReLU → Conv + ReLU → ... → Flatten → FC → Latent Code

    This creates a bottleneck that forces the network to learn a compressed
    representation of the input data.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images)
        hidden_dims (List[int]): Number of channels in each convolutional layer
        latent_dim (int): Dimension of the latent space (bottleneck)
        input_size (int): Size of input images (assumed square)
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [32, 64, 128, 256],
        latent_dim: int = 128,
        input_size: int = 32
    ):
        super(AutoEncoderEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Build convolutional encoder layers
        # Each layer reduces spatial dimensions by 2x (due to stride=2)
        # and increases the number of feature channels
        modules = []

        # First convolution: input channels → first hidden dimension
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),  # Normalize for stable training
                nn.ReLU(inplace=True)
            )
        )

        # Progressive convolutions: each reduces spatial size, increases channels
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )

        self.conv_layers = nn.Sequential(*modules)

        # Calculate the size after all convolutions
        # Each conv layer with stride=2 halves the spatial dimensions
        self.final_spatial_size = input_size // (2 ** len(hidden_dims))
        self.conv_output_size = hidden_dims[-1] * self.final_spatial_size * self.final_spatial_size

        # Validate that the spatial size is reasonable
        if self.final_spatial_size < 1:
            raise ValueError(f"Too many downsampling layers for input size {input_size}. "
                           f"With {len(hidden_dims)} layers, final size would be {self.final_spatial_size}. "
                           f"Reduce the number of hidden layers or increase input size.")

        # Final fully connected layer to produce latent representation
        # This creates the bottleneck that forces compression
        self.fc_latent = nn.Linear(self.conv_output_size, latent_dim)

        print(f"Encoder: {input_size}x{input_size}x{in_channels} → "
              f"{self.final_spatial_size}x{self.final_spatial_size}x{hidden_dims[-1]} → "
              f"{latent_dim}D latent")

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Latent representation of shape (batch, latent_dim)
        """
        # Apply convolutional layers
        # This progressively extracts higher-level features while reducing spatial size
        h = self.conv_layers(x)

        # Flatten to prepare for fully connected layer
        # Shape: (batch, channels, height, width) → (batch, channels*height*width)
        h = h.view(h.size(0), -1)

        # Map to latent space
        # This is where the compression happens - from conv_output_size to latent_dim
        latent = self.fc_latent(h)

        return latent


class AutoEncoderDecoder(nn.Module):
    """
    Decoder Network for AutoEncoder

    The decoder reverses the encoder process, taking the compact latent
    representation and reconstructing the original input.

    Architecture Pattern:
    Latent Code → FC → Reshape → TransposedConv + ReLU → ... → Output

    The decoder must learn to "decompress" the latent representation back
    to the original data space while preserving as much information as possible.

    Args:
        latent_dim (int): Dimension of the latent space
        hidden_dims (List[int]): Number of channels (reversed from encoder)
        out_channels (int): Number of output channels
        output_size (int): Size of output images (assumed square)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64, 32],
        out_channels: int = 3,
        output_size: int = 32
    ):
        super(AutoEncoderDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.output_size = output_size

        # Calculate starting spatial size for reconstruction
        # This should match the encoder's final spatial size
        # Number of stride-2 conv layers = len(hidden_dims) (not len-1)
        self.start_spatial_size = output_size // (2 ** len(hidden_dims))
        self.fc_input_size = hidden_dims[0] * self.start_spatial_size * self.start_spatial_size

        # Validate dimensions
        if self.start_spatial_size < 1:
            raise ValueError(f"Too many upsampling layers for output size {output_size}. "
                           f"With {len(hidden_dims)} layers, start size would be {self.start_spatial_size}. "
                           f"Reduce the number of hidden layers or increase output size.")

        # First fully connected layer: expand latent to feature map
        # This reverses the encoder's final compression step
        self.fc_decode = nn.Linear(latent_dim, self.fc_input_size)

        # Build transposed convolutional decoder layers
        # Each layer doubles spatial dimensions (due to stride=2)
        # and decreases the number of feature channels
        modules = []

        # Progressive transposed convolutions: each increases spatial size, decreases channels
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1  # Ensures exact size doubling
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )

        # Final layer: produce output with correct number of channels
        # No ReLU here since we want unrestricted output values
        modules.append(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        )

        self.deconv_layers = nn.Sequential(*modules)

        print(f"Decoder: {latent_dim}D latent → "
              f"{self.start_spatial_size}x{self.start_spatial_size}x{hidden_dims[0]} → "
              f"{output_size}x{output_size}x{out_channels}")

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to output

        Args:
            z: Latent representation of shape (batch, latent_dim)

        Returns:
            Reconstructed output of shape (batch, out_channels, height, width)
        """
        # Expand latent to feature map size
        # This reverses the encoder's final compression
        h = self.fc_decode(z)

        # Reshape to spatial dimensions
        # Shape: (batch, fc_input_size) → (batch, channels, height, width)
        h = h.view(h.size(0), self.hidden_dims[0], self.start_spatial_size, self.start_spatial_size)

        # Apply transposed convolutional layers
        # This progressively increases spatial size while decreasing channels
        output = self.deconv_layers(h)

        return output


class AutoEncoder(nn.Module):
    """
    Complete AutoEncoder Model

    AutoEncoders learn to compress data into a lower-dimensional representation
    and then reconstruct the original data from this compressed form. This forces
    the model to learn the most important features of the data.

    The training process:
    1. Input data is encoded to latent space (compression)
    2. Latent representation is decoded back to original space (decompression)
    3. Reconstruction loss measures how well the output matches the input
    4. Backpropagation updates weights to minimize reconstruction error

    Key Benefits:
    - Learns meaningful data representations automatically
    - Can compress data while preserving important information
    - Provides interpretable latent space for analysis
    - Foundation for more advanced techniques (VAE, VQ-VAE)

    Training Objective:
    L = ||x - decoder(encoder(x))||²

    Where x is the input and the goal is to minimize reconstruction error.

    Args:
        in_channels (int): Number of input channels
        latent_dim (int): Dimension of latent space (compression level)
        hidden_dims (List[int]): Architecture specification
        input_size (int): Input image size (assumed square)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: List[int] = [32, 64, 128, 256],
        input_size: int = 32
    ):
        super(AutoEncoder, self).__init__()

        # Store hyperparameters
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.input_size = input_size

        # Initialize encoder: x → z
        self.encoder = AutoEncoderEncoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            input_size=input_size
        )

        # Initialize decoder: z → x̂
        # Note: decoder hidden_dims are reversed to mirror encoder
        self.decoder = AutoEncoderDecoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=in_channels,
            output_size=input_size
        )

        # Calculate compression ratio
        input_size_total = in_channels * input_size * input_size
        compression_ratio = input_size_total / latent_dim

        print(f"\nAutoEncoder Summary:")
        print(f"Input size: {in_channels} × {input_size} × {input_size} = {input_size_total:,} values")
        print(f"Latent size: {latent_dim} values")
        print(f"Compression ratio: {compression_ratio:.1f}:1")
        print(f"Information bottleneck: {latent_dim/input_size_total:.1%} of original size")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Complete forward pass through autoencoder

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Dictionary containing:
            - 'reconstructed': Reconstructed output
            - 'latent': Latent representation
            - 'loss': Reconstruction loss (MSE)
        """
        # Encode: compress input to latent representation
        # This step learns to extract the most important features
        latent = self.encoder(x)

        # Decode: reconstruct input from latent representation
        # This step learns to generate realistic outputs from compressed features
        reconstructed = self.decoder(latent)

        # Calculate reconstruction loss
        # Mean Squared Error measures pixel-wise reconstruction quality
        reconstruction_loss = F.mse_loss(reconstructed, x)

        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'loss': reconstruction_loss
        }

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to latent representation

        Args:
            x: Input tensor

        Returns:
            Latent representation
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation to output

        Args:
            z: Latent representation

        Returns:
            Reconstructed output
        """
        return self.decoder(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        """
        Complete encode-decode process

        Args:
            x: Input tensor

        Returns:
            Reconstructed output
        """
        latent = self.encode(x)
        return self.decode(latent)

    def get_latent_representation(self, x: Tensor) -> Tensor:
        """
        Extract latent representations for analysis

        This is useful for:
        - Visualizing learned representations
        - Clustering in latent space
        - Transfer learning
        - Anomaly detection

        Args:
            x: Input tensor

        Returns:
            Latent representations
        """
        with torch.no_grad():
            return self.encode(x)

    def interpolate_in_latent_space(self, x1: Tensor, x2: Tensor, num_steps: int = 10) -> Tensor:
        """
        Interpolate between two inputs in latent space

        This demonstrates the smoothness of the learned latent space.
        Good autoencoders should produce meaningful interpolations.

        Args:
            x1: First input tensor
            x2: Second input tensor
            num_steps: Number of interpolation steps

        Returns:
            Tensor of interpolated reconstructions
        """
        with torch.no_grad():
            # Encode both inputs
            z1 = self.encode(x1)
            z2 = self.encode(x2)

            # Create interpolation weights
            weights = torch.linspace(0, 1, num_steps, device=x1.device).view(-1, 1)

            # Interpolate in latent space
            # z_interp = (1 - α) * z1 + α * z2 for α ∈ [0, 1]
            z_interp = (1 - weights) * z1.unsqueeze(0) + weights * z2.unsqueeze(0)

            # Decode interpolated latents
            reconstructed = self.decode(z_interp.view(-1, self.latent_dim))

            return reconstructed

    def calculate_reconstruction_error(self, x: Tensor) -> Dict[str, float]:
        """
        Calculate various reconstruction error metrics

        Args:
            x: Input tensor

        Returns:
            Dictionary of error metrics
        """
        with torch.no_grad():
            reconstructed = self.reconstruct(x)

            # Mean Squared Error
            mse = F.mse_loss(reconstructed, x).item()

            # Mean Absolute Error
            mae = F.l1_loss(reconstructed, x).item()

            # Peak Signal-to-Noise Ratio (higher is better)
            psnr = -10 * torch.log10(torch.mean((x - reconstructed) ** 2)).item()

            return {
                'mse': mse,
                'mae': mae,
                'psnr': psnr,
                'rmse': mse ** 0.5
            }

    def analyze_latent_space(self, dataloader) -> Dict[str, Any]:
        """
        Analyze properties of the learned latent space

        Args:
            dataloader: DataLoader with input data

        Returns:
            Dictionary with analysis results
        """
        latents = []

        with torch.no_grad():
            for batch, _ in dataloader:
                latent = self.encode(batch)
                latents.append(latent)

                if len(latents) * batch.size(0) >= 1000:  # Limit for analysis
                    break

        latents = torch.cat(latents, dim=0)

        # Statistical analysis
        mean = torch.mean(latents, dim=0)
        std = torch.std(latents, dim=0)

        # Activation statistics
        active_dimensions = torch.sum(torch.abs(latents) > 0.1, dim=0) / latents.size(0)

        return {
            'mean': mean,
            'std': std,
            'active_dimensions': active_dimensions,
            'effective_dim': torch.sum(active_dimensions > 0.1).item(),
            'latent_utilization': torch.mean(active_dimensions).item()
        }