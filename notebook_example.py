"""
Notebook-Friendly Example for Vector Quantization

This example is designed to work in Jupyter notebooks or Python scripts
without requiring package installation. Just run this cell-by-cell.
"""

# Step 1: Setup (run this first)
import sys
from pathlib import Path

# Add source to path
current_dir = Path.cwd()
src_path = current_dir / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Added {src_path} to Python path")

# Step 2: Check if dependencies are available
try:
    import torch
    print("✅ PyTorch available")
except ImportError:
    print("❌ PyTorch not available. Install with: pip install torch")
    print("   For this example, we'll continue anyway...")

try:
    import numpy as np
    print("✅ NumPy available")
except ImportError:
    print("❌ NumPy not available. Install with: pip install numpy")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib available")
except ImportError:
    print("❌ Matplotlib not available. Install with: pip install matplotlib")

# Step 3: Import our modules
try:
    from vector_quantization import VectorQuantizer
    print("✅ VectorQuantizer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import VectorQuantizer: {e}")
    print("   Make sure you're running from the repository root directory")

try:
    from vector_quantization import VQVAE, RQVAE, RQKMeans
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import some modules: {e}")

# Step 4: Simple usage examples (if PyTorch is available)
if 'torch' in globals():
    print("\n" + "="*50)
    print("🎓 SIMPLE USAGE EXAMPLES")
    print("="*50)

    # Example 1: Basic Vector Quantization
    print("\n1. Basic Vector Quantization:")
    try:
        # Create a simple VQ layer
        vq = VectorQuantizer(num_embeddings=16, embedding_dim=8)

        # Create some random data
        # Shape: (batch=2, height=4, width=4, channels=8)
        x = torch.randn(2, 4, 4, 8)

        # Apply quantization
        quantized, vq_loss, perplexity = vq(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {quantized.shape}")
        print(f"  VQ Loss: {vq_loss.item():.4f}")
        print(f"  Perplexity: {perplexity.item():.2f} / {vq.num_embeddings}")
        print(f"  Codebook utilization: {perplexity.item()/vq.num_embeddings:.1%}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Example 2: VQ-VAE for images
    print("\n2. VQ-VAE for Images:")
    try:
        # Create VQ-VAE model
        model = VQVAE(
            in_channels=3,           # RGB images
            embedding_dim=32,        # Smaller for demo
            num_embeddings=64,       # Smaller codebook for demo
            hidden_dims=[32, 64]     # Smaller network for demo
        )

        # Create fake RGB images
        images = torch.randn(4, 3, 32, 32)  # 4 images, 32x32 RGB

        # Forward pass
        with torch.no_grad():  # No gradients for demo
            outputs = model(images)

        reconstructed = outputs['reconstructed']
        vq_loss = outputs['vq_loss']
        perplexity = outputs['perplexity']

        print(f"  Input images shape: {images.shape}")
        print(f"  Reconstructed shape: {reconstructed.shape}")
        print(f"  VQ Loss: {vq_loss.item():.4f}")
        print(f"  Perplexity: {perplexity.item():.2f}")

        # Calculate reconstruction error
        mse = torch.mean((images - reconstructed)**2)
        print(f"  Reconstruction MSE: {mse.item():.4f}")

    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Example 3: RQ-VAE with hierarchical quantization
    print("\n3. RQ-VAE (Residual Quantized VAE):")
    try:
        # Create RQ-VAE with multiple quantization levels
        rq_model = RQVAE(
            in_channels=3,
            embedding_dim=32,
            num_embeddings=64,
            num_quantizers=3,        # 3 hierarchical levels
            hidden_dims=[32, 64]
        )

        # Use the same fake images
        with torch.no_grad():
            rq_outputs = rq_model(images)

        rq_reconstructed = rq_outputs['reconstructed']
        rq_loss = rq_outputs['rq_loss']
        perplexity_list = rq_outputs['perplexity_list']

        print(f"  Input images shape: {images.shape}")
        print(f"  Reconstructed shape: {rq_reconstructed.shape}")
        print(f"  RQ Loss: {rq_loss.item():.4f}")
        print(f"  Perplexity per level: {[p.item() for p in perplexity_list]}")

        # Compare with VQ-VAE
        rq_mse = torch.mean((images - rq_reconstructed)**2)
        print(f"  RQ-VAE reconstruction MSE: {rq_mse.item():.4f}")
        if 'mse' in locals():
            improvement = (mse.item() - rq_mse.item()) / mse.item() * 100
            print(f"  Improvement over VQ-VAE: {improvement:.1f}%")

    except Exception as e:
        print(f"  ❌ Error: {e}")

# Step 5: RQ-K-means example (NumPy only)
if 'np' in globals():
    print("\n4. RQ-K-means Clustering:")
    try:
        # Create some sample data
        X = np.random.randn(200, 10)  # 200 samples, 10 dimensions

        # Add some structure to the data
        X[:50] += [2, 2, 0, 0, 0, 0, 0, 0, 0, 0]    # Cluster 1
        X[50:100] += [-2, -2, 0, 0, 0, 0, 0, 0, 0, 0]  # Cluster 2
        X[100:150] += [0, 0, 2, 2, 0, 0, 0, 0, 0, 0]   # Cluster 3

        # Create RQ-K-means model
        rq_kmeans = RQKMeans(
            n_clusters=8,    # 8 clusters per stage
            n_stages=3,      # 3 stages
            verbose=False    # Quiet for demo
        )

        # Fit and transform
        X_reconstructed = rq_kmeans.fit_transform(X)

        # Calculate error
        error = rq_kmeans.calculate_reconstruction_error(X)
        compression_ratio = rq_kmeans.get_compression_ratio(X.shape[1])

        print(f"  Data shape: {X.shape}")
        print(f"  Reconstructed shape: {X_reconstructed.shape}")
        print(f"  Reconstruction error: {error:.6f}")
        print(f"  Compression ratio: {compression_ratio:.1f}:1")

        # Compare with standard K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        X_kmeans_reconstructed = kmeans.cluster_centers_[kmeans_labels]
        kmeans_error = np.mean((X - X_kmeans_reconstructed)**2)

        print(f"  Standard K-means error: {kmeans_error:.6f}")
        improvement = (kmeans_error - error) / kmeans_error * 100
        print(f"  RQ-K-means improvement: {improvement:.1f}%")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "="*50)
print("🎉 DEMO COMPLETE!")
print("="*50)
print("\nNext steps:")
print("1. Install dependencies: pip install torch numpy matplotlib scikit-learn")
print("2. Try the full examples: cd examples && python basic_vq_demo.py")
print("3. Read the SETUP_GUIDE.md for detailed instructions")
print("4. Explore the code in src/vector_quantization/")

# Step 6: Create a simple visualization if matplotlib is available
if 'plt' in globals() and 'np' in globals():
    try:
        print("\n📊 Creating a simple visualization...")

        # Generate 2D data for visualization
        np.random.seed(42)
        data_2d = np.random.randn(100, 2)
        data_2d[:30] += [2, 2]    # Cluster 1
        data_2d[30:60] += [-2, -2]  # Cluster 2
        data_2d[60:] += [0, 3]    # Cluster 3

        # Apply RQ-K-means
        rq_kmeans_2d = RQKMeans(n_clusters=4, n_stages=2, verbose=False)
        data_2d_reconstructed = rq_kmeans_2d.fit_transform(data_2d)

        # Create plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)
        plt.title('Original 2D Data')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(data_2d_reconstructed[:, 0], data_2d_reconstructed[:, 1], alpha=0.7, color='orange')
        plt.title('RQ-K-means Reconstructed')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_file = current_dir / "demo_output.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✅ Visualization saved to: {output_file}")

    except Exception as e:
        print(f"❌ Visualization failed: {e}")

print(f"\n💡 Tip: You're currently in directory: {Path.cwd()}")
print("   Make sure this is the repository root for imports to work correctly.")