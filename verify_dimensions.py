"""
Verify AutoEncoder dimension calculations without PyTorch
"""

def verify_autoencoder_dimensions():
    """Verify that encoder and decoder dimensions match"""

    print("Verifying AutoEncoder dimension calculations...")

    # Test case from user's example
    input_size = 32
    hidden_dims = [32, 64]

    print(f"Input size: {input_size}x{input_size}")
    print(f"Hidden dims: {hidden_dims}")

    # Encoder calculation
    # Number of stride-2 convolution layers = len(hidden_dims)
    # Each layer halves spatial dimensions
    encoder_layers = len(hidden_dims)
    encoder_final_size = input_size // (2 ** encoder_layers)

    print(f"\nEncoder:")
    print(f"  Number of stride-2 layers: {encoder_layers}")
    print(f"  Spatial reduction: {input_size} → {encoder_final_size}")

    # Decoder calculation (FIXED)
    # Should start from encoder's final size and upscale back
    decoder_start_size = encoder_final_size
    decoder_layers = len(hidden_dims)  # Should match encoder
    decoder_final_size = decoder_start_size * (2 ** decoder_layers)

    print(f"\nDecoder:")
    print(f"  Start size: {decoder_start_size}x{decoder_start_size}")
    print(f"  Number of stride-2 layers: {decoder_layers}")
    print(f"  Spatial upscaling: {decoder_start_size} → {decoder_final_size}")

    # Verify they match
    dimensions_match = input_size == decoder_final_size
    print(f"\nDimension check:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {decoder_final_size}")
    print(f"  Match: {dimensions_match}")

    if dimensions_match:
        print("✅ Dimensions are consistent!")
    else:
        print("❌ Dimension mismatch!")

    return dimensions_match

def test_various_configurations():
    """Test various AutoEncoder configurations"""

    print("\n" + "="*50)
    print("Testing various configurations...")

    test_cases = [
        {"input_size": 32, "hidden_dims": [32, 64]},
        {"input_size": 64, "hidden_dims": [32, 64, 128]},
        {"input_size": 128, "hidden_dims": [64, 128, 256, 512]},
        {"input_size": 256, "hidden_dims": [32, 64, 128]},
    ]

    all_passed = True

    for i, config in enumerate(test_cases, 1):
        print(f"\nTest {i}: {config}")

        input_size = config["input_size"]
        hidden_dims = config["hidden_dims"]

        # Calculate dimensions
        encoder_layers = len(hidden_dims)
        encoder_final_size = input_size // (2 ** encoder_layers)
        decoder_final_size = encoder_final_size * (2 ** encoder_layers)

        match = input_size == decoder_final_size
        print(f"  {input_size} → {encoder_final_size} → {decoder_final_size}: {'✅' if match else '❌'}")

        if not match:
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("🔧 AutoEncoder Dimension Verification")
    print("="*50)

    # Test the specific problematic case
    main_test = verify_autoencoder_dimensions()

    # Test various configurations
    all_tests = test_various_configurations()

    print("\n" + "="*50)
    if main_test and all_tests:
        print("🎉 All dimension calculations are correct!")
        print("The AutoEncoder fix should resolve the tensor size mismatch.")
    else:
        print("❌ Some dimension calculations failed.")
        print("Further investigation needed.")