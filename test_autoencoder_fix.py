"""
Test script to verify AutoEncoder dimension fix
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_autoencoder():
    """Test the AutoEncoder with the problematic configuration"""
    print("Testing AutoEncoder with fixed dimensions...")

    try:
        from vector_quantization import AutoEncoder

        # Create autoencoder with the configuration from the user's example
        model = AutoEncoder(
            in_channels=3,        # RGB images
            latent_dim=128,       # Compressed representation size
            hidden_dims=[32, 64], # Network architecture - this was causing the issue
            input_size=32         # 32x32 images
        )

        # Sample images
        images = torch.randn(16, 3, 32, 32)
        print(f"Input shape: {images.shape}")

        # Forward pass
        outputs = model(images)
        reconstructed = outputs['reconstructed']
        latent_codes = outputs['latent']
        loss = outputs['loss']

        print(f"Latent shape: {latent_codes.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstruction loss: {loss:.4f}")

        # Verify shapes are correct
        assert images.shape == reconstructed.shape, f"Shape mismatch: {images.shape} vs {reconstructed.shape}"
        assert latent_codes.shape == (16, 128), f"Latent shape wrong: {latent_codes.shape}"

        print("✅ AutoEncoder test PASSED!")
        return True

    except Exception as e:
        print(f"❌ AutoEncoder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dimension_calculations():
    """Test the dimension calculation logic directly"""
    print("\nTesting dimension calculations...")

    input_size = 32
    hidden_dims = [32, 64]

    # Encoder: number of stride-2 layers = len(hidden_dims)
    # Each stride-2 layer halves the dimension
    encoder_output_size = input_size // (2 ** len(hidden_dims))
    print(f"Encoder output size: {input_size} // (2^{len(hidden_dims)}) = {encoder_output_size}")

    # Decoder: should start from encoder_output_size and upscale back to input_size
    decoder_start_size = encoder_output_size
    decoder_layers = len(hidden_dims)  # Should match encoder
    final_size = decoder_start_size * (2 ** decoder_layers)
    print(f"Decoder final size: {decoder_start_size} * (2^{decoder_layers}) = {final_size}")

    print(f"Match: {input_size == final_size}")

    return input_size == final_size

if __name__ == "__main__":
    print("🧪 Testing AutoEncoder Fix")
    print("=" * 50)

    # Test dimension calculations
    calc_ok = test_dimension_calculations()

    # Test actual AutoEncoder
    ae_ok = test_autoencoder()

    print("\n" + "=" * 50)
    if calc_ok and ae_ok:
        print("🎉 All tests PASSED! AutoEncoder fix is working.")
    else:
        print("❌ Some tests FAILED. Need further investigation.")