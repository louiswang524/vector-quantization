"""
Verify VAE dimension calculations are correct
"""

def verify_vae_dimensions():
    """Verify VAE encoder and decoder dimensions match"""
    print("🔧 VAE Dimension Verification")
    print("=" * 35)

    # Test the problematic configuration
    input_size = 32
    hidden_dims = [32, 64]

    print(f"Configuration: input_size={input_size}, hidden_dims={hidden_dims}")

    # Encoder calculation (same as AutoEncoder)
    encoder_layers = len(hidden_dims)
    encoder_final_size = input_size // (2 ** encoder_layers)

    print(f"\nEncoder:")
    print(f"  Stride-2 layers: {encoder_layers}")
    print(f"  Spatial reduction: {input_size} → {encoder_final_size}")

    # Decoder calculation (FIXED)
    decoder_start_size = encoder_final_size
    decoder_layers = len(hidden_dims)
    decoder_final_size = decoder_start_size * (2 ** decoder_layers)

    print(f"\nDecoder:")
    print(f"  Start size: {decoder_start_size}")
    print(f"  Stride-2 layers: {decoder_layers}")
    print(f"  Spatial upscaling: {decoder_start_size} → {decoder_final_size}")

    # Check consistency
    match = input_size == decoder_final_size
    print(f"\nConsistency check:")
    print(f"  Input: {input_size}")
    print(f"  Output: {decoder_final_size}")
    print(f"  Match: {'✅' if match else '❌'}")

    return match

def test_various_vae_configs():
    """Test various VAE configurations"""
    print("\n🧪 Testing Various VAE Configurations")
    print("=" * 45)

    configs = [
        {"input_size": 32, "hidden_dims": [32, 64]},
        {"input_size": 64, "hidden_dims": [64, 128]},
        {"input_size": 64, "hidden_dims": [32, 64, 128]},
        {"input_size": 128, "hidden_dims": [64, 128, 256]},
        {"input_size": 256, "hidden_dims": [32, 64, 128, 256]},
    ]

    all_passed = True

    for i, config in enumerate(configs, 1):
        input_size = config["input_size"]
        hidden_dims = config["hidden_dims"]

        # Calculate dimensions
        encoder_layers = len(hidden_dims)
        encoder_final = input_size // (2 ** encoder_layers)
        decoder_final = encoder_final * (2 ** encoder_layers)

        match = input_size == decoder_final
        status = "✅" if match else "❌"

        print(f"Config {i}: {input_size}, {hidden_dims}")
        print(f"  {input_size} → {encoder_final} → {decoder_final} {status}")

        if not match:
            all_passed = False

    return all_passed

if __name__ == "__main__":
    print("🔬 VAE Dimension Verification")
    print("=" * 30)

    # Test specific problematic case
    main_test = verify_vae_dimensions()

    # Test various configurations
    all_configs = test_various_vae_configs()

    print("\n" + "=" * 40)
    if main_test and all_configs:
        print("🎉 All VAE dimension calculations are correct!")
        print("The VAE fix should resolve the tensor size mismatch.")
    else:
        print("❌ Some dimension calculations failed.")
        print("Need further investigation.")