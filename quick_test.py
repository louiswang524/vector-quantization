"""
Quick Test Script for Vector Quantization Package

This script tests that all modules can be imported and basic functionality works.
Run this to verify your installation is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.absolute()
src_path = repo_root / "src"
sys.path.insert(0, str(src_path))

def test_basic_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing basic imports...")

    try:
        from vector_quantization import VectorQuantizer
        print("✅ VectorQuantizer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import VectorQuantizer: {e}")
        return False

    try:
        from vector_quantization import VQVAE
        print("✅ VQVAE imported successfully")
    except Exception as e:
        print(f"❌ Failed to import VQVAE: {e}")
        return False

    try:
        from vector_quantization import RQVAE
        print("✅ RQVAE imported successfully")
    except Exception as e:
        print(f"❌ Failed to import RQVAE: {e}")
        return False

    try:
        from vector_quantization import RQKMeans
        print("✅ RQKMeans imported successfully")
    except Exception as e:
        print(f"❌ Failed to import RQKMeans: {e}")
        return False

    return True

def test_basic_functionality():
    """Test basic functionality of each module"""
    print("\n🔬 Testing basic functionality...")

    try:
        import torch
        import numpy as np
        from vector_quantization import VectorQuantizer, VQVAE, RQVAE, RQKMeans

        # Test VectorQuantizer
        print("  Testing VectorQuantizer...")
        vq = VectorQuantizer(num_embeddings=8, embedding_dim=4)
        x = torch.randn(1, 2, 2, 4)  # batch=1, h=2, w=2, dim=4
        quantized, loss, perplexity = vq(x)
        print(f"    ✅ VQ output shape: {quantized.shape}, loss: {loss:.4f}, perplexity: {perplexity:.2f}")

        # Test VQVAE
        print("  Testing VQVAE...")
        vqvae = VQVAE(in_channels=3, embedding_dim=16, num_embeddings=32, hidden_dims=[32, 64])
        x = torch.randn(2, 3, 16, 16)  # batch=2, channels=3, 16x16 images
        outputs = vqvae(x)
        print(f"    ✅ VQ-VAE reconstruction shape: {outputs['reconstructed'].shape}")

        # Test RQVAE
        print("  Testing RQVAE...")
        rqvae = RQVAE(in_channels=3, embedding_dim=16, num_embeddings=32, num_quantizers=2, hidden_dims=[32, 64])
        outputs = rqvae(x)
        print(f"    ✅ RQ-VAE reconstruction shape: {outputs['reconstructed'].shape}, levels: {len(outputs['quantized_list'])}")

        # Test RQKMeans
        print("  Testing RQKMeans...")
        X = np.random.randn(100, 8)  # 100 samples, 8 dimensions
        rq_kmeans = RQKMeans(n_clusters=4, n_stages=2, verbose=False)
        X_reconstructed = rq_kmeans.fit_transform(X)
        error = rq_kmeans.calculate_reconstruction_error(X)
        print(f"    ✅ RQ-K-means reconstruction error: {error:.6f}")

        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎓 Vector Quantization Educational Package - Quick Test")
    print("=" * 60)

    # Test imports
    imports_ok = test_basic_imports()

    if not imports_ok:
        print("\n❌ Import tests failed. Check your setup.")
        return

    # Test functionality
    functionality_ok = test_basic_functionality()

    print("\n" + "=" * 60)
    if imports_ok and functionality_ok:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\nYou can now:")
        print("1. Run the examples: cd examples && python basic_vq_demo.py")
        print("2. Import modules in your own code")
        print("3. Start learning about vector quantization!")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()