"""
VQ-VAE Image Reconstruction Demonstration

This example demonstrates how VQ-VAE works on image data. It shows:
1. Training a VQ-VAE on simple synthetic images
2. The reconstruction process and quality
3. Analysis of the learned discrete codes
4. Comparison with standard VAE (conceptual)
5. Codebook utilization and visualization

Educational Goals:
- Understand VQ-VAE architecture and components
- See how discrete representations work for images
- Learn about the trade-offs in discrete latent spaces
- Visualize the quantization process in 2D latent space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_quantization import VQVAE


def generate_synthetic_images(n_images=1000, image_size=32):
    """
    Generate synthetic images for demonstration

    Creates simple geometric patterns that are easy to understand
    and visualize, perfect for educational purposes.

    Args:
        n_images: Number of images to generate
        image_size: Size of square images

    Returns:
        Tensor of images (n_images, 3, image_size, image_size)
    """
    print(f"Generating {n_images} synthetic images of size {image_size}x{image_size}...")

    images = torch.zeros(n_images, 3, image_size, image_size)

    for i in range(n_images):
        # Create different types of patterns
        pattern_type = i % 4

        if pattern_type == 0:  # Horizontal stripes
            color = torch.rand(3) * 0.8 + 0.1  # Avoid pure black/white
            stripe_width = np.random.randint(2, 6)
            for y in range(0, image_size, stripe_width * 2):
                images[i, :, y:y+stripe_width, :] = color.unsqueeze(-1)

        elif pattern_type == 1:  # Vertical stripes
            color = torch.rand(3) * 0.8 + 0.1
            stripe_width = np.random.randint(2, 6)
            for x in range(0, image_size, stripe_width * 2):
                images[i, :, :, x:x+stripe_width] = color.unsqueeze(-1).unsqueeze(-1)

        elif pattern_type == 2:  # Circles
            center_x, center_y = np.random.randint(8, image_size-8, 2)
            radius = np.random.randint(4, 12)
            color = torch.rand(3) * 0.8 + 0.1

            # Create circle mask
            y, x = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2

            for c in range(3):
                images[i, c][mask] = color[c]

        elif pattern_type == 3:  # Rectangles
            x1, y1 = np.random.randint(4, image_size//2, 2)
            x2, y2 = np.random.randint(image_size//2, image_size-4, 2)
            color = torch.rand(3) * 0.8 + 0.1

            images[i, :, y1:y2, x1:x2] = color.unsqueeze(-1).unsqueeze(-1)

    print(f"Generated {n_images} images with 4 pattern types:")
    print("  - Horizontal stripes")
    print("  - Vertical stripes")
    print("  - Circles")
    print("  - Rectangles")

    return images


def create_data_loader(images, batch_size=32, train_split=0.8):
    """
    Create train/test data loaders

    Args:
        images: Generated images
        batch_size: Batch size for training
        train_split: Fraction of data to use for training

    Returns:
        train_loader, test_loader
    """
    n_train = int(len(images) * train_split)
    train_images = images[:n_train]
    test_images = images[n_train:]

    train_dataset = TensorDataset(train_images, train_images)  # Input = Target for autoencoder
    test_dataset = TensorDataset(test_images, test_images)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Created data loaders:")
    print(f"  Training samples: {len(train_images)}")
    print(f"  Test samples: {len(test_images)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, test_loader


def train_vqvae(model, train_loader, test_loader, num_epochs=50, learning_rate=1e-3):
    """
    Train the VQ-VAE model

    Args:
        model: VQ-VAE model to train
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization

    Returns:
        Training history dictionary
    """
    print(f"\n🚀 Training VQ-VAE for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training history
    history = {
        'train_recon_loss': [],
        'train_vq_loss': [],
        'train_total_loss': [],
        'test_recon_loss': [],
        'test_vq_loss': [],
        'test_total_loss': [],
        'perplexity': []
    }

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Training phase
        train_recon_loss = 0
        train_vq_loss = 0
        train_samples = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            reconstructed = outputs['reconstructed']
            vq_loss = outputs['vq_loss']

            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, images)

            # Total loss
            total_loss = recon_loss + vq_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            batch_size = images.size(0)
            train_recon_loss += recon_loss.item() * batch_size
            train_vq_loss += vq_loss.item() * batch_size
            train_samples += batch_size

        # Average training losses
        train_recon_loss /= train_samples
        train_vq_loss /= train_samples
        train_total_loss = train_recon_loss + train_vq_loss

        # Validation phase
        model.eval()
        test_recon_loss = 0
        test_vq_loss = 0
        test_samples = 0
        total_perplexity = 0

        with torch.no_grad():
            for images, _ in test_loader:
                outputs = model(images)
                reconstructed = outputs['reconstructed']
                vq_loss = outputs['vq_loss']
                perplexity = outputs['perplexity']

                recon_loss = F.mse_loss(reconstructed, images)

                batch_size = images.size(0)
                test_recon_loss += recon_loss.item() * batch_size
                test_vq_loss += vq_loss.item() * batch_size
                total_perplexity += perplexity.item() * batch_size
                test_samples += batch_size

        # Average test losses
        test_recon_loss /= test_samples
        test_vq_loss /= test_samples
        test_total_loss = test_recon_loss + test_vq_loss
        avg_perplexity = total_perplexity / test_samples

        # Store history
        history['train_recon_loss'].append(train_recon_loss)
        history['train_vq_loss'].append(train_vq_loss)
        history['train_total_loss'].append(train_total_loss)
        history['test_recon_loss'].append(test_recon_loss)
        history['test_vq_loss'].append(test_vq_loss)
        history['test_total_loss'].append(test_total_loss)
        history['perplexity'].append(avg_perplexity)

        # Update learning rate
        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_total_loss:.4f} "
                  f"(Recon: {train_recon_loss:.4f}, VQ: {train_vq_loss:.4f}), "
                  f"Test Loss = {test_total_loss:.4f}, "
                  f"Perplexity = {avg_perplexity:.1f}")

        model.train()

    print("✅ Training completed!")
    return history


def visualize_reconstructions(model, test_loader, n_examples=8):
    """
    Visualize original vs reconstructed images

    Args:
        model: Trained VQ-VAE model
        test_loader: Test data loader
        n_examples: Number of examples to show
    """
    print(f"\n🎨 Visualizing {n_examples} reconstruction examples...")

    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:n_examples]

        # Get reconstructions
        outputs = model(test_images)
        reconstructions = outputs['reconstructed']

        # Create visualization
        fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 2, 4))
        fig.suptitle('VQ-VAE Image Reconstructions', fontsize=16)

        for i in range(n_examples):
            # Original image
            orig_img = test_images[i].permute(1, 2, 0).clamp(0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # Reconstructed image
            recon_img = reconstructions[i].permute(1, 2, 0).clamp(0, 1)
            axes[1, i].imshow(recon_img)

            # Calculate MSE for this image
            mse = F.mse_loss(reconstructions[i], test_images[i]).item()
            axes[1, i].set_title(f'Recon {i+1}\n(MSE: {mse:.4f})')
            axes[1, i].axis('off')

        plt.tight_layout()

        # Save visualization
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'vqvae_reconstructions.png'), dpi=300, bbox_inches='tight')
        print(f"Reconstructions saved to: {os.path.join(output_dir, 'vqvae_reconstructions.png')}")

        plt.show()


def analyze_discrete_codes(model, test_loader):
    """
    Analyze the discrete codes learned by VQ-VAE

    Args:
        model: Trained VQ-VAE model
        test_loader: Test data loader
    """
    print("\n🔍 Analyzing learned discrete codes...")

    model.eval()
    all_codes = []
    all_images = []

    with torch.no_grad():
        for images, _ in test_loader:
            # Get discrete codes
            codes = model.encode_to_indices(images)
            all_codes.append(codes)
            all_images.append(images)

            if len(all_codes) * images.size(0) >= 100:  # Limit for analysis
                break

    # Combine all codes and images
    all_codes = torch.cat(all_codes, dim=0)[:100]
    all_images = torch.cat(all_images, dim=0)[:100]

    print(f"Analyzing {len(all_codes)} samples...")
    print(f"Code tensor shape: {all_codes.shape}")
    print(f"Spatial resolution: {all_codes.shape[1]}x{all_codes.shape[2]}")

    # Analyze code distribution
    unique_codes, counts = torch.unique(all_codes, return_counts=True)
    print(f"Number of unique codes used: {len(unique_codes)}")
    print(f"Total possible codes: {model.num_embeddings}")
    print(f"Code utilization: {len(unique_codes)/model.num_embeddings:.1%}")

    # Visualize code usage
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VQ-VAE Discrete Code Analysis', fontsize=16)

    # Code distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(all_codes.flatten().numpy(), bins=min(50, len(unique_codes)), alpha=0.7)
    ax1.set_title('Distribution of Discrete Codes')
    ax1.set_xlabel('Code Index')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Example code maps for a few images
    for i in range(3):
        row = (i + 1) // 2
        col = (i + 1) % 2
        if row >= 2:
            break

        ax = axes[row, col]

        # Show the discrete code map
        code_map = all_codes[i].numpy()
        im = ax.imshow(code_map, cmap='tab20', vmin=0, vmax=model.num_embeddings-1)
        ax.set_title(f'Discrete Code Map {i+1}')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Code Index')

    plt.tight_layout()

    # Save code analysis
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'vqvae_code_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Code analysis saved to: {os.path.join(output_dir, 'vqvae_code_analysis.png')}")

    plt.show()

    return all_codes, unique_codes, counts


def visualize_training_curves(history):
    """
    Visualize training and validation curves

    Args:
        history: Training history dictionary
    """
    print("\n📊 Creating training curves visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('VQ-VAE Training History', fontsize=16)

    epochs = range(1, len(history['train_total_loss']) + 1)

    # Total loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_total_loss'], label='Train', color='blue')
    ax1.plot(epochs, history['test_total_loss'], label='Test', color='red')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reconstruction loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_recon_loss'], label='Train', color='blue')
    ax2.plot(epochs, history['test_recon_loss'], label='Test', color='red')
    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # VQ loss
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['train_vq_loss'], label='Train', color='blue')
    ax3.plot(epochs, history['test_vq_loss'], label='Test', color='red')
    ax3.set_title('Vector Quantization Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('VQ Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Perplexity
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['perplexity'], color='green')
    ax4.set_title('Codebook Perplexity')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Perplexity')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save training curves
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'vqvae_training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {os.path.join(output_dir, 'vqvae_training_curves.png')}")

    plt.show()


def demonstrate_compression_capabilities(model, test_loader):
    """
    Demonstrate the compression capabilities of VQ-VAE

    Args:
        model: Trained VQ-VAE model
        test_loader: Test data loader
    """
    print("\n💾 Analyzing compression capabilities...")

    # Get some test images
    test_images, _ = next(iter(test_loader))
    test_images = test_images[:16]  # Use 16 images

    model.eval()
    with torch.no_grad():
        # Get discrete codes
        codes = model.encode_to_indices(test_images)
        print(f"Original image size: {test_images.shape}")
        print(f"Discrete codes shape: {codes.shape}")

        # Calculate bits per pixel
        original_bpp = 8 * 3  # 8 bits per channel, 3 channels
        import math
        compressed_bpp = math.log2(model.num_embeddings) * codes.numel() / (test_images.shape[2] * test_images.shape[3])
        compression_ratio = original_bpp / compressed_bpp

        print(f"Original bits per pixel: {original_bpp}")
        print(f"Compressed bits per pixel: {compressed_bpp:.2f}")
        print(f"Compression ratio: {compression_ratio:.1f}:1")

        # Reconstruct from codes
        reconstructed = model.decode_from_indices(codes)

        # Calculate reconstruction quality
        mse = F.mse_loss(reconstructed, test_images).item()
        psnr = -10 * math.log10(mse)

        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Reconstruction PSNR: {psnr:.2f} dB")

        # Show compression example
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f'VQ-VAE Compression (Ratio: {compression_ratio:.1f}:1, PSNR: {psnr:.1f} dB)', fontsize=14)

        # Original
        orig_img = test_images[0].permute(1, 2, 0).clamp(0, 1)
        axes[0].imshow(orig_img)
        axes[0].set_title(f'Original\n({original_bpp} bpp)')
        axes[0].axis('off')

        # Discrete codes
        axes[1].imshow(codes[0].numpy(), cmap='tab20')
        axes[1].set_title(f'Discrete Codes\n({compressed_bpp:.1f} bpp)')
        axes[1].axis('off')

        # Reconstructed
        recon_img = reconstructed[0].permute(1, 2, 0).clamp(0, 1)
        axes[2].imshow(recon_img)
        axes[2].set_title(f'Reconstructed\n(MSE: {F.mse_loss(reconstructed[0], test_images[0]).item():.4f})')
        axes[2].axis('off')

        plt.tight_layout()

        # Save compression demo
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        plt.savefig(os.path.join(output_dir, 'vqvae_compression_demo.png'), dpi=300, bbox_inches='tight')
        print(f"Compression demo saved to: {os.path.join(output_dir, 'vqvae_compression_demo.png')}")

        plt.show()


def main():
    """
    Run the complete VQ-VAE demonstration
    """
    print("🎓 VQ-VAE IMAGE RECONSTRUCTION EDUCATIONAL DEMO")
    print("This demo teaches you how VQ-VAE works with image data")
    print("through hands-on training and analysis.\n")

    try:
        # Configuration
        image_size = 32
        n_images = 800
        batch_size = 32
        num_epochs = 30
        embedding_dim = 64
        num_embeddings = 128

        print("Configuration:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Number of images: {n_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training epochs: {num_epochs}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Codebook size: {num_embeddings}")

        # Generate data
        images = generate_synthetic_images(n_images=n_images, image_size=image_size)
        train_loader, test_loader = create_data_loader(images, batch_size=batch_size)

        # Create model
        print(f"\n🏗️ Creating VQ-VAE model...")
        model = VQVAE(
            in_channels=3,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=[64, 128],  # Smaller for our simple images
            num_residual_layers=2,
            residual_hidden_dim=32
        )

        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Train model
        history = train_vqvae(model, train_loader, test_loader, num_epochs=num_epochs)

        # Visualizations and analysis
        visualize_training_curves(history)
        visualize_reconstructions(model, test_loader)
        analyze_discrete_codes(model, test_loader)
        demonstrate_compression_capabilities(model, test_loader)

        print("\n" + "=" * 60)
        print("VQ-VAE DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. VQ-VAE learns discrete representations of images")
        print("2. The encoder-quantizer-decoder pipeline enables compression")
        print("3. Discrete codes capture semantic information")
        print("4. Trade-off between codebook size and reconstruction quality")
        print("5. Perplexity indicates how well the codebook is utilized")

        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        print(f"\n📁 All output files saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Make sure you have all required packages installed.")
        raise


if __name__ == "__main__":
    main()