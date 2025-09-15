"""
Fixed VAE Demo

This demo shows the corrected VAE implementation that resolves
the tensor size mismatch error, and demonstrates the key features
of Variational AutoEncoders including generation and interpolation.

The VAE had the same dimension mismatch issue as the AutoEncoder,
which has now been fixed by correcting the decoder's spatial size calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vector_quantization import VAE


def test_fixed_vae_configuration():
    """Test the VAE configuration that was previously causing errors"""
    print("🧪 Testing Fixed VAE Configuration")
    print("=" * 40)

    try:
        # This configuration was causing the tensor size mismatch
        vae = VAE(
            in_channels=3,
            latent_dim=64,
            hidden_dims=[32, 64],
            input_size=32,
            beta=1.0  # KL weight (β-VAE)
        )

        # Sample images
        images = torch.randn(16, 3, 32, 32)

        # Forward pass - this should now work!
        outputs = vae(images)

        print(f"✅ SUCCESS! No tensor size mismatch.")
        print(f"Input shape: {images.shape}")
        print(f"Reconstructed shape: {outputs['reconstructed'].shape}")
        print(f"Total loss: {outputs['total_loss']:.4f}")
        print(f"Reconstruction loss: {outputs['reconstruction_loss']:.4f}")
        print(f"KL divergence: {outputs['kl_loss']:.4f}")

        # Test generation capability
        new_samples = vae.sample(num_samples=8)
        print(f"Generated samples shape: {new_samples.shape}")

        return True, vae

    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None


def demonstrate_vae_features(vae):
    """Demonstrate key VAE features: encoding, decoding, and generation"""
    print("\n🎯 VAE Key Features Demonstration")
    print("=" * 40)

    # Create sample data
    sample_images = torch.randn(8, 3, 32, 32)

    print("1. Encoding (deterministic representation):")
    mu, logvar = vae.encode(sample_images)
    print(f"   Mean (μ) shape: {mu.shape}")
    print(f"   Log-variance shape: {logvar.shape}")
    print(f"   Mean range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"   Log-var range: [{logvar.min():.3f}, {logvar.max():.3f}]")

    print("\n2. Reparameterization (stochastic sampling):")
    z = vae.reparameterize(mu, logvar)
    print(f"   Sampled latent shape: {z.shape}")
    print(f"   Latent range: [{z.min():.3f}, {z.max():.3f}]")

    print("\n3. Decoding (reconstruction):")
    reconstructed = vae.decode(z)
    print(f"   Reconstructed shape: {reconstructed.shape}")

    print("\n4. Generation (sampling from prior):")
    generated = vae.sample(num_samples=4)
    print(f"   Generated samples shape: {generated.shape}")

    print("\n5. Latent space interpolation:")
    # Encode two different images
    img1 = torch.randn(1, 3, 32, 32)
    img2 = torch.randn(1, 3, 32, 32)

    mu1, logvar1 = vae.encode(img1)
    mu2, logvar2 = vae.encode(img2)

    # Interpolate between the means
    interpolations = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = alpha * mu1 + (1 - alpha) * mu2
        img_interp = vae.decode(z_interp)
        interpolations.append(img_interp)

    print(f"   Created {len(interpolations)} interpolation steps")
    print(f"   Each interpolation shape: {interpolations[0].shape}")


def compare_vae_vs_autoencoder():
    """Compare VAE with regular AutoEncoder to highlight differences"""
    print("\n🔍 VAE vs AutoEncoder Comparison")
    print("=" * 40)

    from vector_quantization import AutoEncoder

    # Same configuration for both
    config = {
        "in_channels": 3,
        "latent_dim": 64,
        "hidden_dims": [32, 64],
        "input_size": 32
    }

    # Create both models
    vae = VAE(**config, beta=1.0)
    ae = AutoEncoder(**config)

    # Sample data
    test_images = torch.randn(4, 3, 32, 32)

    print("AutoEncoder:")
    ae_outputs = ae(test_images)
    print(f"   Latent shape: {ae_outputs['latent'].shape}")
    print(f"   Loss: {ae_outputs['loss']:.4f}")
    print(f"   Loss type: Reconstruction only")

    print("\nVAE:")
    vae_outputs = vae(test_images)
    print(f"   Mean shape: {vae_outputs['mu'].shape}")
    print(f"   Logvar shape: {vae_outputs['logvar'].shape}")
    print(f"   Total loss: {vae_outputs['total_loss']:.4f}")
    print(f"   Reconstruction: {vae_outputs['reconstruction_loss']:.4f}")
    print(f"   KL divergence: {vae_outputs['kl_loss']:.4f}")

    print("\nKey Differences:")
    print("   • AutoEncoder: Deterministic latent codes")
    print("   • VAE: Probabilistic latent distribution (μ, σ²)")
    print("   • AutoEncoder: Reconstruction loss only")
    print("   • VAE: Reconstruction + KL regularization")
    print("   • AutoEncoder: Cannot generate new samples")
    print("   • VAE: Can generate by sampling from prior N(0,I)")


def demonstrate_beta_vae():
    """Demonstrate β-VAE for disentanglement"""
    print("\n🎛️ β-VAE for Disentanglement")
    print("=" * 35)

    beta_values = [0.1, 1.0, 4.0, 10.0]

    print("Effect of β (KL weight) on VAE behavior:")
    print("β controls trade-off between reconstruction and regularization")

    for beta in beta_values:
        vae = VAE(
            in_channels=3,
            latent_dim=64,
            hidden_dims=[32, 64],
            input_size=32,
            beta=beta
        )

        # Test with sample data
        test_data = torch.randn(4, 3, 32, 32)
        outputs = vae(test_data)

        print(f"\nβ = {beta}:")
        print(f"   Total loss: {outputs['total_loss']:.4f}")
        print(f"   Reconstruction: {outputs['reconstruction_loss']:.4f}")
        print(f"   KL × β: {outputs['kl_loss']:.4f}")
        print(f"   Raw KL: {outputs['kl_loss'] / beta:.4f}")

        if beta < 1.0:
            print("   → Emphasizes reconstruction (sharper images)")
        elif beta > 1.0:
            print("   → Emphasizes regularization (better disentanglement)")
        else:
            print("   → Standard VAE balance")


def training_example():
    """Show a simple training example with the fixed VAE"""
    print("\n🏋️ Simple VAE Training Example")
    print("=" * 35)

    # Create VAE
    vae = VAE(
        in_channels=3,
        latent_dim=32,
        hidden_dims=[64, 128],
        input_size=64,
        beta=1.0
    )

    # Simple synthetic data generator
    def create_simple_patterns(batch_size=16, size=64):
        """Create simple geometric patterns for training"""
        data = torch.zeros(batch_size, 3, size, size)

        for i in range(batch_size):
            # Random color
            color = torch.rand(3) * 0.8 + 0.2  # Avoid very dark colors

            # Random pattern type
            pattern_type = torch.randint(0, 3, (1,)).item()

            if pattern_type == 0:  # Circle
                center_x, center_y = size // 2, size // 2
                radius = torch.randint(10, 25, (1,)).item()
                y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
                mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
                data[i, :, mask] = color.view(3, 1)

            elif pattern_type == 1:  # Rectangle
                x1 = torch.randint(5, 20, (1,)).item()
                y1 = torch.randint(5, 20, (1,)).item()
                x2 = x1 + torch.randint(20, 40, (1,)).item()
                y2 = y1 + torch.randint(20, 40, (1,)).item()
                data[i, :, y1:min(y2, size), x1:min(x2, size)] = color.view(3, 1, 1)

            else:  # Diagonal stripes
                for j in range(0, size, 8):
                    if j + 4 < size:
                        data[i, :, j:j+4, :] = color.view(3, 1, 1)

        return data

    # Training setup
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    num_epochs = 5

    print("Training VAE on synthetic patterns...")

    for epoch in range(num_epochs):
        # Generate batch
        data = create_simple_patterns(batch_size=16, size=64)

        # Forward pass
        outputs = vae(data)
        total_loss = outputs['total_loss']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"   Total: {outputs['total_loss']:.4f}")
        print(f"   Recon: {outputs['reconstruction_loss']:.4f}")
        print(f"   KL: {outputs['kl_loss']:.4f}")

    print("✅ Training completed!")

    # Test generation after training
    with torch.no_grad():
        generated = vae.sample(num_samples=4)
        print(f"Generated samples shape: {generated.shape}")


def main():
    """Run all VAE demonstrations"""
    print("🔧 VAE Fix and Feature Demonstration")
    print("=" * 45)

    # Test the fix
    success, vae = test_fixed_vae_configuration()

    if success and vae:
        # Demonstrate VAE features
        demonstrate_vae_features(vae)

        # Compare with AutoEncoder
        compare_vae_vs_autoencoder()

        # Show β-VAE effects
        demonstrate_beta_vae()

        # Training example
        training_example()

        print("\n🎉 All VAE demonstrations completed successfully!")
        print("\nKey takeaways:")
        print("• VAE tensor size mismatch has been fixed")
        print("• VAE provides probabilistic latent representations")
        print("• β parameter controls reconstruction vs. regularization trade-off")
        print("• VAE enables generation of new samples")
        print("• Latent space interpolation creates smooth transitions")

    else:
        print("\n❌ VAE fix verification failed.")


if __name__ == "__main__":
    main()