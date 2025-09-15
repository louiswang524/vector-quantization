"""
Installation Test Script

This script tests that the package can be imported correctly after installation.
Run this from outside the source directory to verify installation.
"""

def test_imports():
    """Test importing the main modules"""
    print("🧪 Testing package imports after installation...")

    try:
        import vector_quantization
        from vector_quantization import VectorQuantizer
        print("✅ VectorQuantizer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import VectorQuantizer: {e}")
        return False

    try:
        from vector_quantization import VQVAE, RQVAE, RQKMeans
        print("✅ VQ models imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import VQ models: {e}")
        return False

    try:
        from vector_quantization import VQPipeline, PipelineConfig
        print("✅ Pipeline system imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import pipeline system: {e}")
        return False

    try:
        from vector_quantization import ModalityType, EncoderType, VQMethodType
        print("✅ Configuration enums imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import configuration enums: {e}")
        return False

    try:
        from vector_quantization import EncoderFactory
        print("✅ Encoder factory imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import encoder factory: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality without PyTorch (config only)"""
    print("\n🔧 Testing basic configuration functionality...")

    try:
        from vector_quantization import PipelineConfig, ModalityType, EncoderType, VQMethodType

        # Test configuration creation
        config = PipelineConfig(
            modality=ModalityType.IMAGE,
            encoder_type=EncoderType.CNN,
            vq_method=VQMethodType.BASIC_VQ,
            input_dim=(3, 64, 64),
            embedding_dim=256,
            num_embeddings=1024,
            encoder_config={'hidden_dims': [64, 128, 256]}
        )

        print("✅ Configuration created successfully")
        print(f"   Modality: {config.modality.value}")
        print(f"   Encoder: {config.encoder_type.value}")
        print(f"   VQ Method: {config.vq_method.value}")

        # Test serialization
        config_dict = config.to_dict()
        restored_config = PipelineConfig.from_dict(config_dict)
        print("✅ Configuration serialization works")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    print("🎓 Vector Quantization Package Installation Test")
    print("=" * 50)

    imports_ok = test_imports()
    config_ok = test_basic_functionality()

    print("\n" + "=" * 50)
    if imports_ok and config_ok:
        print("🎉 Installation test PASSED!")
        print("\nThe package is correctly installed and ready to use.")
        print("\nNext steps:")
        print("1. Install PyTorch if not already installed:")
        print("   pip install torch torchvision")
        print("2. Try the examples:")
        print("   python -m vector_quantization.examples.basic_vq_demo")
        print("3. Check the documentation in README.md")
    else:
        print("❌ Installation test FAILED!")
        print("\nTroubleshooting:")
        print("1. Make sure you installed the package:")
        print("   pip install -e .")
        print("2. Check your Python environment")
        print("3. Verify you're not in the source directory when testing")

if __name__ == "__main__":
    main()