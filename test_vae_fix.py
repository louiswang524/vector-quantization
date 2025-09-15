"""
Test VAE Fix

This script tests that the VAE tensor size mismatch has been fixed.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_vae_problematic_config():
    """Test the VAE configuration that was causing the tensor size mismatch"""
    print("🧪 Testing VAE with Previously Problematic Configuration")
    print("=" * 60)

    try:
        from vector_quantization import VAE

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
        print(f"Input shape: {images.shape}")

        # Forward pass - this should now work!
        outputs = vae(images)

        print(f"✅ SUCCESS! No tensor size mismatch.")
        print(f"Total loss: {outputs['total_loss']:.4f}")
        print(f"Reconstruction: {outputs['reconstruction_loss']:.4f}")
        print(f"KL divergence: {outputs['kl_loss']:.4f}")

        # Test generation
        new_samples = vae.sample(num_samples=8)
        print(f"Generated samples shape: {new_samples.shape}")

        # Test other methods
        mu, logvar = vae.encode(images)
        print(f"Encoded mu shape: {mu.shape}")
        print(f"Encoded logvar shape: {logvar.shape}")

        decoded = vae.decode(mu)
        print(f"Decoded shape: {decoded.shape}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vae_dimension_consistency():
    """Test VAE with various configurations to ensure dimension consistency"""
    print("\n🔧 Testing VAE Dimension Consistency")
    print("=" * 45)

    test_configs = [
        {"input_size": 32, "hidden_dims": [32, 64], "latent_dim": 64},
        {"input_size": 64, "hidden_dims": [64, 128], "latent_dim": 128},
        {"input_size": 64, "hidden_dims": [32, 64, 128], "latent_dim": 64},
        {"input_size": 128, "hidden_dims": [64, 128, 256], "latent_dim": 256},
    ]

    all_passed = True

    for i, config in enumerate(test_configs, 1):
        print(f"\nTest {i}: {config}")

        try:
            from vector_quantization import VAE

            vae = VAE(
                in_channels=3,
                latent_dim=config["latent_dim"],
                hidden_dims=config["hidden_dims"],
                input_size=config["input_size"],
                beta=1.0
            )

            # Test with sample data
            input_size = config["input_size"]
            test_data = torch.randn(4, 3, input_size, input_size)

            outputs = vae(test_data)
            samples = vae.sample(num_samples=2)

            print(f"  ✅ Input {test_data.shape} → Output {outputs['reconstructed'].shape}")
            print(f"  ✅ Generated samples: {samples.shape}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("🔬 VAE Fix Test")
    print("=" * 20)

    # Test the specific problematic configuration
    main_test = test_vae_problematic_config()

    # Test various configurations
    consistency_test = test_vae_dimension_consistency()

    print("\n" + "=" * 50)
    if main_test and consistency_test:
        print("🎉 All VAE tests PASSED!")
        print("The VAE tensor size mismatch has been fixed.")
        print("\nUsers can now successfully run:")
        print("vae = VAE(in_channels=3, latent_dim=64, hidden_dims=[32, 64], input_size=32)")
        print("outputs = vae(images)  # Works without error!")
    else:
        print("❌ Some VAE tests FAILED.")
        print("Further investigation needed.")