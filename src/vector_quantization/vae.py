"""
Variational AutoEncoder (VAE) Implementation with Educational Focus

Variational AutoEncoders extend traditional autoencoders by learning probabilistic
latent representations instead of deterministic ones. This enables generation of
new data samples and provides a principled approach to unsupervised learning.

Key Innovations over Standard AutoEncoders:
1. Probabilistic Encoder: Outputs mean and variance instead of point estimates
2. Reparameterization Trick: Enables backpropagation through stochastic sampling
3. Regularization: KL divergence encourages latent space to match prior distribution
4. Generative Capability: Can sample from learned latent distribution

Mathematical Framework:
- Encoder: q_φ(z|x) = N(μ(x), σ²(x))
- Decoder: p_θ(x|z)
- Prior: p(z) = N(0, I)
- ELBO: L = 𝔼[log p_θ(x|z)] - KL[q_φ(z|x)||p(z)]
- Reparameterization: z = μ + σ ⊙ ε, where ε ~ N(0, I)

The VAE Loss Function:
L = Reconstruction Loss + β × KL Divergence Loss
- Reconstruction Loss: ||x - x̂||² (like autoencoder)
- KL Loss: Regularizes latent space to match standard normal prior
- β: Controls trade-off between reconstruction and regularization

Educational Value:
This implementation emphasizes understanding the mathematical foundations
and intuitions behind variational inference and generative modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Any
import math


class VAEEncoder(nn.Module):
    """
    Probabilistic Encoder for VAE

    Unlike deterministic autoencoders, VAE encoders output parameters of a
    probability distribution (mean and log-variance) rather than point estimates.

    Key Differences from AutoEncoder Encoder:
    1. Outputs μ and log(σ²) instead of single latent vector
    2. Uses log-variance for numerical stability
    3. Same architecture but different output layer

    The probabilistic formulation enables:
    - Uncertainty quantification in latent space
    - Sampling different latents for same input
    - Principled regularization through KL divergence

    Args:
        in_channels (int): Number of input channels
        hidden_dims (List[int]): Convolutional layer dimensions
        latent_dim (int): Dimension of latent space
        input_size (int): Input image size
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: List[int] = [32, 64, 128, 256],
        latent_dim: int = 128,
        input_size: int = 32
    ):
        super(VAEEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Build convolutional feature extractor (same as AutoEncoder)
        modules = []

        # First convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU(inplace=True)
            )
        )

        # Progressive convolutions
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )

        self.conv_layers = nn.Sequential(*modules)

        # Calculate flattened size
        self.final_spatial_size = input_size // (2 ** len(hidden_dims))
        self.conv_output_size = hidden_dims[-1] * self.final_spatial_size * self.final_spatial_size

        # Validate that the spatial size is reasonable
        if self.final_spatial_size < 1:
            raise ValueError(f"Too many downsampling layers for input size {input_size}. "
                           f"With {len(hidden_dims)} layers, final size would be {self.final_spatial_size}. "
                           f"Reduce the number of hidden layers or increase input size.")

        # VAE-specific: separate linear layers for mean and log-variance
        # This is the key difference from deterministic autoencoders
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)      # Mean parameters μ
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)  # Log-variance log(σ²)

        print(f"VAE Encoder: {input_size}x{input_size}x{in_channels} → "
              f"μ,log(σ²) ∈ ℝ^{latent_dim}")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution parameters

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of (mu, logvar) where:
            - mu: Mean of latent distribution (batch, latent_dim)
            - logvar: Log-variance of latent distribution (batch, latent_dim)
        """
        # Extract features using convolutional layers
        h = self.conv_layers(x)

        # Flatten for fully connected layers
        h = h.view(h.size(0), -1)

        # Compute distribution parameters
        # μ: Mean can be any real number
        mu = self.fc_mu(h)

        # log(σ²): Log-variance for numerical stability
        # Using log-variance instead of variance prevents numerical issues
        # and ensures σ² > 0 since σ² = exp(log(σ²))
        logvar = self.fc_logvar(h)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    Decoder Network for VAE

    The VAE decoder is identical to the autoencoder decoder, as it maps
    from latent space back to data space. The probabilistic nature is
    handled in the encoder and loss function.

    Args:
        latent_dim (int): Dimension of latent space
        hidden_dims (List[int]): Decoder layer dimensions (reversed from encoder)
        out_channels (int): Number of output channels
        output_size (int): Output image size
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: List[int] = [256, 128, 64, 32],
        out_channels: int = 3,
        output_size: int = 32
    ):
        super(VAEDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.output_size = output_size

        # Calculate spatial dimensions
        # Number of stride-2 conv layers = len(hidden_dims) (not len-1)
        self.start_spatial_size = output_size // (2 ** len(hidden_dims))
        self.fc_input_size = hidden_dims[0] * self.start_spatial_size * self.start_spatial_size

        # Validate dimensions
        if self.start_spatial_size < 1:
            raise ValueError(f"Too many upsampling layers for output size {output_size}. "
                           f"With {len(hidden_dims)} layers, start size would be {self.start_spatial_size}. "
                           f"Reduce the number of hidden layers or increase output size.")

        # Fully connected layer: latent → feature map
        self.fc_decode = nn.Linear(latent_dim, self.fc_input_size)

        # Transposed convolutional layers
        modules = []

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i+1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.ReLU(inplace=True)
                )
            )

        # Final layer to output channels
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

        print(f"VAE Decoder: ℝ^{latent_dim} → {output_size}x{output_size}x{out_channels}")

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent samples to reconstructed output

        Args:
            z: Latent samples of shape (batch, latent_dim)

        Returns:
            Reconstructed output of shape (batch, out_channels, height, width)
        """
        # Expand to feature map
        h = self.fc_decode(z)

        # Reshape to spatial dimensions
        h = h.view(h.size(0), self.hidden_dims[0], self.start_spatial_size, self.start_spatial_size)

        # Generate output through transposed convolutions
        output = self.deconv_layers(h)

        return output


class VAE(nn.Module):
    """
    Complete Variational AutoEncoder Model

    VAE combines probabilistic encoding, the reparameterization trick, and
    regularized training to learn meaningful latent representations that
    can generate new data samples.

    Key Components:
    1. Probabilistic Encoder: q_φ(z|x) = N(μ_φ(x), σ_φ²(x))
    2. Reparameterization Trick: z = μ + σ ⊙ ε, where ε ~ N(0,I)
    3. Decoder: p_θ(x|z)
    4. Regularized Loss: Reconstruction + β × KL Divergence

    Training Process:
    1. Encoder produces μ and σ² for input x
    2. Sample z using reparameterization trick (enables backprop)
    3. Decoder reconstructs x̂ from z
    4. Compute reconstruction loss ||x - x̂||²
    5. Compute KL loss KL[q_φ(z|x)||N(0,I)]
    6. Total loss = reconstruction + β × KL

    The β parameter (β-VAE):
    - β = 1: Standard VAE (theoretically motivated)
    - β > 1: Emphasizes disentanglement over reconstruction
    - β < 1: Emphasizes reconstruction over regularization

    Args:
        in_channels (int): Number of input channels
        latent_dim (int): Dimension of latent space
        hidden_dims (List[int]): Network architecture
        input_size (int): Input image size
        beta (float): Weight for KL divergence loss
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        hidden_dims: List[int] = [32, 64, 128, 256],
        input_size: int = 32,
        beta: float = 1.0
    ):
        super(VAE, self).__init__()

        # Store hyperparameters
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.beta = beta

        # Initialize encoder: x → (μ, σ²)
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            input_size=input_size
        )

        # Initialize decoder: z → x̂
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=in_channels,
            output_size=input_size
        )

        print(f"\nVAE Summary:")
        print(f"Latent dimension: {latent_dim}")
        print(f"β parameter: {beta}")
        print(f"Prior: p(z) = N(0, I_{latent_dim})")
        print(f"Posterior: q(z|x) = N(μ_φ(x), diag(σ_φ²(x)))")

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization Trick

        This is the key innovation that enables backpropagation through
        stochastic sampling. Instead of sampling z directly from N(μ,σ²),
        we sample ε from N(0,I) and compute z = μ + σ⊙ε.

        Why this works:
        - z ~ N(μ,σ²) has the same distribution as μ + σ⊙ε where ε ~ N(0,I)
        - Gradients can flow through μ and σ (deterministic)
        - Stochasticity is isolated in ε (no gradients needed)

        Args:
            mu: Mean tensor (batch, latent_dim)
            logvar: Log-variance tensor (batch, latent_dim)

        Returns:
            Reparameterized samples z (batch, latent_dim)
        """
        if self.training:
            # Standard deviation: σ = exp(0.5 × log(σ²)) = exp(0.5 × logvar)
            std = torch.exp(0.5 * logvar)

            # Sample random noise ε ~ N(0, I)
            eps = torch.randn_like(std)

            # Reparameterized sample: z = μ + σ ⊙ ε
            return mu + eps * std
        else:
            # During evaluation, use mean (no randomness)
            return mu

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Complete forward pass through VAE

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Dictionary containing:
            - 'reconstructed': Reconstructed output
            - 'mu': Latent means
            - 'logvar': Latent log-variances
            - 'z': Sampled latent codes
            - 'reconstruction_loss': Reconstruction term
            - 'kl_loss': KL divergence term
            - 'total_loss': Combined VAE loss
        """
        # Encode: x → (μ, log(σ²))
        mu, logvar = self.encoder(x)

        # Reparameterize: sample z from N(μ, σ²)
        z = self.reparameterize(mu, logvar)

        # Decode: z → x̂
        reconstructed = self.decoder(z)

        # Compute losses
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='mean')

        # KL divergence: KL[q_φ(z|x)||p(z)] where p(z) = N(0,I)
        # KL[N(μ,σ²)||N(0,I)] = 0.5 × Σ[μ² + σ² - log(σ²) - 1]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total VAE loss (Evidence Lower Bound - ELBO)
        total_loss = reconstruction_loss + self.beta * kl_loss

        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to latent distribution parameters

        Args:
            x: Input tensor

        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent samples to output

        Args:
            z: Latent samples

        Returns:
            Decoded output
        """
        return self.decoder(z)

    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> Tensor:
        """
        Generate new samples from the learned distribution

        This demonstrates the generative capability of VAEs. By sampling
        from the prior p(z) = N(0,I) and decoding, we generate new data.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device

        with torch.no_grad():
            # Sample from prior distribution p(z) = N(0, I)
            z = torch.randn(num_samples, self.latent_dim, device=device)

            # Decode to generate new samples
            samples = self.decode(z)

        return samples

    def interpolate(self, x1: Tensor, x2: Tensor, num_steps: int = 10) -> Tensor:
        """
        Interpolate between two inputs in latent space

        VAE interpolations are often smoother than autoencoder interpolations
        due to the regularized latent space structure.

        Args:
            x1: First input
            x2: Second input
            num_steps: Number of interpolation steps

        Returns:
            Interpolated reconstructions
        """
        with torch.no_grad():
            # Encode to get latent means (ignore variance for interpolation)
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)

            # Linear interpolation weights
            weights = torch.linspace(0, 1, num_steps, device=x1.device).view(-1, 1)

            # Interpolate in latent space
            z_interp = (1 - weights) * mu1.unsqueeze(0) + weights * mu2.unsqueeze(0)

            # Decode interpolated latents
            interpolations = self.decode(z_interp.view(-1, self.latent_dim))

        return interpolations

    def reconstruct(self, x: Tensor, use_mean: bool = True) -> Tensor:
        """
        Reconstruct input (encode then decode)

        Args:
            x: Input tensor
            use_mean: If True, use latent mean; if False, sample from distribution

        Returns:
            Reconstructed output
        """
        with torch.no_grad():
            mu, logvar = self.encode(x)

            if use_mean:
                z = mu  # Use mean for deterministic reconstruction
            else:
                z = self.reparameterize(mu, logvar)  # Sample for stochastic reconstruction

            return self.decode(z)

    def compute_elbo(self, x: Tensor) -> Dict[str, float]:
        """
        Compute Evidence Lower BOund (ELBO) components

        ELBO = 𝔼[log p(x|z)] - KL[q(z|x)||p(z)]

        Maximizing ELBO is equivalent to minimizing the negative log likelihood
        of the data under the VAE model.

        Args:
            x: Input data

        Returns:
            Dictionary with ELBO components
        """
        with torch.no_grad():
            outputs = self.forward(x)

            # ELBO components (note: we minimize negative ELBO)
            log_likelihood = -outputs['reconstruction_loss'].item()  # log p(x|z)
            kl_divergence = outputs['kl_loss'].item()  # KL[q(z|x)||p(z)]
            elbo = log_likelihood - kl_divergence

            return {
                'elbo': elbo,
                'log_likelihood': log_likelihood,
                'kl_divergence': kl_divergence,
                'reconstruction_loss': outputs['reconstruction_loss'].item(),
                'total_loss': outputs['total_loss'].item()
            }

    def analyze_latent_space(self, dataloader) -> Dict[str, Any]:
        """
        Analyze the learned latent space properties

        Args:
            dataloader: DataLoader with input data

        Returns:
            Analysis results
        """
        all_mu = []
        all_logvar = []

        with torch.no_grad():
            for batch, _ in dataloader:
                mu, logvar = self.encode(batch)
                all_mu.append(mu)
                all_logvar.append(logvar)

                if len(all_mu) * batch.size(0) >= 1000:  # Limit for analysis
                    break

        all_mu = torch.cat(all_mu, dim=0)
        all_logvar = torch.cat(all_logvar, dim=0)

        # Analyze posterior statistics
        mu_mean = torch.mean(all_mu, dim=0)
        mu_std = torch.std(all_mu, dim=0)

        # Average posterior variance
        avg_posterior_var = torch.mean(torch.exp(all_logvar), dim=0)

        # KL divergence per dimension
        kl_per_dim = 0.5 * (mu_mean**2 + avg_posterior_var - torch.log(avg_posterior_var) - 1)

        # Active dimensions (dimensions with significant variation)
        active_dims = torch.sum(mu_std > 0.1, dim=0).float()

        return {
            'posterior_mean': mu_mean,
            'posterior_std': mu_std,
            'avg_posterior_variance': avg_posterior_var,
            'kl_per_dimension': kl_per_dim,
            'active_dimensions': active_dims.item(),
            'total_kl': torch.sum(kl_per_dim).item(),
            'effective_latent_dim': active_dims.item()
        }

    def set_beta(self, beta: float):
        """
        Update the β parameter for β-VAE experiments

        Args:
            beta: New β value
        """
        self.beta = beta
        print(f"Updated β to {beta}")