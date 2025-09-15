"""
Fixed AutoEncoder Demo

This demo shows the corrected AutoEncoder implementation that resolves
the tensor size mismatch error. It also demonstrates how to choose
appropriate architecture configurations.

The original error:
"RuntimeError: The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 3"

Was caused by mismatched spatial dimensions between encoder and decoder.
This has been fixed by correcting the decoder's starting spatial size calculation.
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

from vector_quantization import AutoEncoder


def test_problematic_configuration():
    """Test the configuration that was causing the original error"""
    print("🧪 Testing Previously Problematic Configuration")
    print("=" * 55)

    try:
        # This configuration was causing the tensor size mismatch
        model = AutoEncoder(
            in_channels=3,        # RGB images
            latent_dim=128,       # Compressed representation size
            hidden_dims=[32, 64], # Network architecture
            input_size=32         # 32x32 images
        )

        # Sample images
        images = torch.randn(16, 3, 32, 32)

        # Forward pass - this should now work!
        outputs = model(images)
        reconstructed = outputs['reconstructed']
        latent_codes = outputs['latent']
        loss = outputs['loss']

        print(f"✅ SUCCESS! No tensor size mismatch.")
        print(f"Input shape: {images.shape}")
        print(f"Latent shape: {latent_codes.shape}")
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Reconstruction loss: {loss:.4f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def demonstrate_architecture_guidelines():
    """Demonstrate guidelines for choosing AutoEncoder architectures"""
    print("\n🏗️ AutoEncoder Architecture Guidelines")
    print("=" * 45)

    configurations = [
        {
            "name": "Small Network",
            "config": {"input_size": 32, "hidden_dims": [32, 64], "latent_dim": 64},
            "description": "Good for small images, fast training"
        },
        {
            "name": "Medium Network",
            "config": {"input_size": 64, "hidden_dims": [64, 128, 256], "latent_dim": 128},
            "description": "Balanced performance and capacity"
        },
        {
            "name": "Large Network",
            "config": {"input_size": 128, "hidden_dims": [64, 128, 256, 512], "latent_dim": 256},
            "description": "High capacity for complex data"
        },
        {
            "name": "Deep Compression",
            "config": {"input_size": 256, "hidden_dims": [32, 64, 128], "latent_dim": 32},
            "description": "Aggressive compression"
        }
    ]

    for config_info in configurations:
        print(f"\n📐 {config_info['name']}:")
        print(f"   {config_info['description']}")

        config = config_info["config"]

        # Calculate compression ratio
        input_size_total = 3 * config["input_size"] ** 2
        compression_ratio = input_size_total / config["latent_dim"]

        # Calculate spatial reduction
        num_layers = len(config["hidden_dims"])
        final_spatial_size = config["input_size"] // (2 ** num_layers)

        print(f"   Input: 3×{config['input_size']}×{config['input_size']} = {input_size_total:,} values")
        print(f"   Latent: {config['latent_dim']} values")
        print(f"   Compression: {compression_ratio:.1f}:1")
        print(f"   Spatial: {config['input_size']}×{config['input_size']} → {final_spatial_size}×{final_spatial_size}")

        # Test if configuration is valid
        try:
            model = AutoEncoder(**config, in_channels=3)
            print(f"   Status: ✅ Valid configuration")
        except Exception as e:
            print(f"   Status: ❌ Invalid - {e}")


def demonstrate_training_example():
    """Show a simple training example with the fixed AutoEncoder"""
    print("\n🎯 Simple Training Example")
    print("=" * 30)

    # Create model
    model = AutoEncoder(
        in_channels=3,
        latent_dim=128,
        hidden_dims=[64, 128],
        input_size=64
    )

    # Create synthetic data (colored squares)
    def create_colored_squares(batch_size=32, size=64):
        """Create simple synthetic data with colored squares"""
        data = torch.zeros(batch_size, 3, size, size)

        for i in range(batch_size):
            # Random color
            color = torch.rand(3)

            # Random square position and size
            square_size = torch.randint(10, 30, (1,)).item()
            x_pos = torch.randint(0, size - square_size, (1,)).item()
            y_pos = torch.randint(0, size - square_size, (1,)).item()

            # Fill square with color
            data[i, :, y_pos:y_pos+square_size, x_pos:x_pos+square_size] = color.view(3, 1, 1)

        return data

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5

    print("Training AutoEncoder on synthetic colored squares...")

    for epoch in range(num_epochs):
        # Generate batch of data
        data = create_colored_squares(batch_size=16, size=64)

        # Forward pass
        outputs = model(data)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    print("✅ Training completed successfully!")

    # Test reconstruction
    with torch.no_grad():
        test_data = create_colored_squares(batch_size=4, size=64)
        outputs = model(test_data)

        print(f"Final reconstruction loss: {outputs['loss']:.4f}")
        print(f"Latent codes shape: {outputs['latent'].shape}")


def show_dimension_calculation_helper():
    """Show how to calculate dimensions for custom configurations"""
    print("\n🧮 Dimension Calculation Helper")
    print("=" * 35)

    print("To design your own AutoEncoder configuration:")
    print("1. Choose input size (must be power of 2 for best results)")
    print("2. Choose number of layers (hidden_dims length)")
    print("3. Verify: final_spatial_size = input_size // (2^num_layers) >= 1")
    print("4. Choose latent_dim based on desired compression ratio")

    print("\nExamples:")
    test_cases = [
        (32, 2),   # input_size=32, num_layers=2
        (64, 3),   # input_size=64, num_layers=3
        (128, 4),  # input_size=128, num_layers=4
        (256, 5),  # input_size=256, num_layers=5
    ]

    for input_size, num_layers in test_cases:
        final_size = input_size // (2 ** num_layers)
        status = "✅" if final_size >= 1 else "❌"
        print(f"  Input {input_size}, Layers {num_layers}: {input_size} → {final_size} {status}")


def main():
    """Run all demonstrations"""
    print("🔧 AutoEncoder Fix Demonstration")
    print("=" * 40)

    # Test the problematic configuration
    success = test_problematic_configuration()

    if success:
        # Show architecture guidelines
        demonstrate_architecture_guidelines()

        # Show training example
        demonstrate_training_example()

        # Show dimension helper
        show_dimension_calculation_helper()

        print("\n🎉 All tests completed successfully!")
        print("The AutoEncoder tensor size mismatch has been fixed.")
        print("\nKey takeaways:")
        print("• The decoder now correctly calculates starting dimensions")
        print("• Added validation to prevent invalid configurations")
        print("• Choose architectures where final_spatial_size >= 1")
        print("• Consider compression ratio vs. reconstruction quality trade-off")

    else:
        print("\n❌ The fix did not work. Please check the implementation.")


if __name__ == "__main__":
    main()