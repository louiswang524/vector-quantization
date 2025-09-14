"""
RQ-VAE vs VQ-VAE Comparison Demo

This example compares Residual Quantized VAE (RQ-VAE) with standard VQ-VAE
to demonstrate the benefits of multi-level quantization. It shows:
1. Side-by-side training of both models
2. Reconstruction quality comparison
3. Analysis of hierarchical representations in RQ-VAE
4. Compression efficiency comparison
5. Codebook utilization analysis

Educational Goals:
- Understand the difference between VQ-VAE and RQ-VAE
- See how residual quantization improves reconstruction
- Learn about hierarchical discrete representations
- Analyze the trade-offs in multi-level quantization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_quantization import VQVAE, RQVAE


def generate_complex_images(n_images=600, image_size=32):
    """
    Generate more complex synthetic images to better demonstrate
    the advantages of residual quantization.

    Args:
        n_images: Number of images to generate
        image_size: Size of square images

    Returns:
        Tensor of images (n_images, 3, image_size, image_size)
    """
    print(f"Generating {n_images} complex synthetic images...")

    images = torch.zeros(n_images, 3, image_size, image_size)

    for i in range(n_images):
        # Create more complex patterns that benefit from hierarchical quantization
        pattern_type = i % 6

        if pattern_type == 0:  # Gradient background with shapes
            # Create gradient background
            x = torch.linspace(0, 1, image_size)
            y = torch.linspace(0, 1, image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')

            base_color = torch.rand(3)
            for c in range(3):
                images[i, c] = base_color[c] * (0.3 + 0.7 * (X + Y) / 2)

            # Add circle
            center_x, center_y = np.random.randint(8, image_size-8, 2)
            radius = np.random.randint(3, 8)
            circle_color = torch.rand(3) * 0.8 + 0.1

            y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2

            for c in range(3):
                images[i, c][mask] = circle_color[c]

        elif pattern_type == 1:  # Multiple overlapping circles
            # Random background
            bg_color = torch.rand(3) * 0.3
            for c in range(3):
                images[i, c].fill_(bg_color[c])

            # Add multiple circles
            n_circles = np.random.randint(2, 4)
            for _ in range(n_circles):
                center_x, center_y = np.random.randint(6, image_size-6, 2)
                radius = np.random.randint(4, 10)
                circle_color = torch.rand(3) * 0.7 + 0.2

                y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2

                for c in range(3):
                    images[i, c][mask] = circle_color[c]

        elif pattern_type == 2:  # Textured stripes
            stripe_width = np.random.randint(3, 8)
            base_color = torch.rand(3)
            noise_color = torch.rand(3)

            for y in range(0, image_size, stripe_width * 2):
                # Base stripe
                for c in range(3):
                    images[i, c, y:y+stripe_width, :] = base_color[c]

                # Add texture noise
                noise = torch.randn(stripe_width, image_size) * 0.1
                for c in range(3):
                    if y + stripe_width <= image_size:
                        images[i, c, y:y+stripe_width, :] += noise * noise_color[c]

        elif pattern_type == 3:  # Checkerboard with variations
            check_size = np.random.randint(4, 8)
            color1 = torch.rand(3) * 0.6
            color2 = torch.rand(3) * 0.6 + 0.4

            for y in range(0, image_size, check_size):
                for x in range(0, image_size, check_size):
                    if ((y // check_size) + (x // check_size)) % 2 == 0:
                        color = color1
                    else:
                        color = color2

                    # Add some variation to each square
                    variation = torch.randn(3) * 0.05
                    final_color = torch.clamp(color + variation, 0, 1)

                    for c in range(3):
                        images[i, c, y:min(y+check_size, image_size),
                              x:min(x+check_size, image_size)] = final_color[c]

        elif pattern_type == 4:  # Radial patterns
            center_x, center_y = image_size // 2, image_size // 2
            max_radius = image_size // 2
            base_color = torch.rand(3)

            y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            distances = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            angles = torch.atan2(y_grid - center_y, x_grid - center_x)

            # Create radial pattern
            pattern = torch.sin(distances * np.pi / max_radius * 3) * torch.cos(angles * 4)
            pattern = (pattern + 1) / 2  # Normalize to [0, 1]

            for c in range(3):
                images[i, c] = base_color[c] * 0.5 + pattern * 0.5

        elif pattern_type == 5:  # Complex overlapping shapes
            # Background gradient
            x = torch.linspace(0, 1, image_size)
            y = torch.linspace(0, 1, image_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')

            bg_color = torch.rand(3) * 0.4
            for c in range(3):
                images[i, c] = bg_color[c] * (0.5 + 0.5 * torch.sin(X * np.pi) * torch.cos(Y * np.pi))

            # Add rectangle
            x1, y1 = np.random.randint(4, image_size//2, 2)
            x2, y2 = np.random.randint(image_size//2, image_size-4, 2)
            rect_color = torch.rand(3) * 0.6 + 0.3

            for c in range(3):
                images[i, c, y1:y2, x1:x2] = rect_color[c]

            # Add circle on top
            center_x, center_y = np.random.randint(8, image_size-8, 2)
            radius = np.random.randint(5, 12)
            circle_color = torch.rand(3) * 0.8 + 0.1

            y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2

            for c in range(3):
                images[i, c][mask] = circle_color[c]

    print(f"Generated complex images with 6 pattern types:")
    print("  - Gradient backgrounds with shapes")
    print("  - Multiple overlapping circles")
    print("  - Textured stripes")
    print("  - Checkerboard variations")
    print("  - Radial patterns")
    print("  - Complex overlapping shapes")

    return images


def train_model(model, train_loader, test_loader, model_name, num_epochs=40):
    """
    Train a model (VQ-VAE or RQ-VAE)

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        model_name: Name for logging
        num_epochs: Number of epochs

    Returns:
        Training history and final test metrics
    """
    print(f"\n🚀 Training {model_name}...")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    history = {
        'train_loss': [],
        'test_loss': [],
        'test_recon_loss': [],
        'test_vq_loss': [],
        'perplexity': []
    }

    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_samples = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(images)
            reconstructed = outputs['reconstructed']

            # Handle different loss key names
            if 'vq_loss' in outputs:
                vq_loss = outputs['vq_loss']
            elif 'rq_loss' in outputs:
                vq_loss = outputs['rq_loss']
            else:
                raise KeyError("Expected 'vq_loss' or 'rq_loss' in model outputs")

            recon_loss = F.mse_loss(reconstructed, images)
            total_loss = recon_loss + vq_loss

            total_loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += total_loss.item() * batch_size
            train_samples += batch_size

        train_loss /= train_samples

        # Testing
        model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_vq_loss = 0
        test_samples = 0
        total_perplexity = 0

        with torch.no_grad():
            for images, _ in test_loader:
                outputs = model(images)
                reconstructed = outputs['reconstructed']

                if 'vq_loss' in outputs:
                    vq_loss = outputs['vq_loss']
                elif 'rq_loss' in outputs:
                    vq_loss = outputs['rq_loss']

                # Handle perplexity (might be list for RQ-VAE)
                if 'perplexity' in outputs:
                    perplexity = outputs['perplexity']
                    if isinstance(perplexity, list):
                        perplexity = sum(perplexity) / len(perplexity)  # Average for RQ-VAE
                elif 'perplexity_list' in outputs:
                    perplexity_list = outputs['perplexity_list']
                    perplexity = sum(perplexity_list) / len(perplexity_list)
                else:
                    perplexity = torch.tensor(0.0)

                recon_loss = F.mse_loss(reconstructed, images)
                total_loss = recon_loss + vq_loss

                batch_size = images.size(0)
                test_loss += total_loss.item() * batch_size
                test_recon_loss += recon_loss.item() * batch_size
                test_vq_loss += vq_loss.item() * batch_size
                total_perplexity += perplexity.item() * batch_size
                test_samples += batch_size

        test_loss /= test_samples
        test_recon_loss /= test_samples
        test_vq_loss /= test_samples
        avg_perplexity = total_perplexity / test_samples

        # Store history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_recon_loss'].append(test_recon_loss)
        history['test_vq_loss'].append(test_vq_loss)
        history['perplexity'].append(avg_perplexity)

        # Update learning rate
        scheduler.step()

        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss

        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1:3d}: "
                  f"Train = {train_loss:.4f}, "
                  f"Test = {test_loss:.4f}, "
                  f"Recon = {test_recon_loss:.4f}, "
                  f"VQ = {test_vq_loss:.4f}, "
                  f"Perp = {avg_perplexity:.1f}, "
                  f"Time = {elapsed:.1f}s")

    print(f"  ✅ {model_name} training completed! Best test loss: {best_test_loss:.4f}")

    return history, {
        'best_test_loss': best_test_loss,
        'final_recon_loss': test_recon_loss,
        'final_vq_loss': test_vq_loss,
        'final_perplexity': avg_perplexity
    }


def compare_reconstructions(vqvae_model, rqvae_model, test_loader, n_examples=6):
    """
    Compare reconstructions from both models side by side
    """
    print(f"\n🎨 Comparing reconstructions from both models...")

    vqvae_model.eval()
    rqvae_model.eval()

    with torch.no_grad():
        # Get test images
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:n_examples]

        # Get reconstructions
        vqvae_outputs = vqvae_model(test_images)
        rqvae_outputs = rqvae_model(test_images)

        vqvae_recons = vqvae_outputs['reconstructed']
        rqvae_recons = rqvae_outputs['reconstructed']

        # Calculate MSE for each image
        vqvae_mse = [F.mse_loss(vqvae_recons[i], test_images[i]).item() for i in range(n_examples)]
        rqvae_mse = [F.mse_loss(rqvae_recons[i], test_images[i]).item() for i in range(n_examples)]

        # Create comparison visualization
        fig, axes = plt.subplots(3, n_examples, figsize=(n_examples * 2.5, 7))
        fig.suptitle('VQ-VAE vs RQ-VAE Reconstruction Comparison', fontsize=16)

        for i in range(n_examples):
            # Original
            orig_img = test_images[i].permute(1, 2, 0).clamp(0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # VQ-VAE reconstruction
            vqvae_img = vqvae_recons[i].permute(1, 2, 0).clamp(0, 1)
            axes[1, i].imshow(vqvae_img)
            axes[1, i].set_title(f'VQ-VAE\n(MSE: {vqvae_mse[i]:.4f})')
            axes[1, i].axis('off')

            # RQ-VAE reconstruction
            rqvae_img = rqvae_recons[i].permute(1, 2, 0).clamp(0, 1)
            axes[2, i].imshow(rqvae_img)
            axes[2, i].set_title(f'RQ-VAE\n(MSE: {rqvae_mse[i]:.4f})')
            axes[2, i].axis('off')

        plt.tight_layout()

        # Save comparison
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'vqvae_vs_rqvae_reconstructions.png'),
                   dpi=300, bbox_inches='tight')
        print(f"Reconstruction comparison saved to outputs/")

        plt.show()

        # Print statistics
        avg_vqvae_mse = sum(vqvae_mse) / len(vqvae_mse)
        avg_rqvae_mse = sum(rqvae_mse) / len(rqvae_mse)
        improvement = (avg_vqvae_mse - avg_rqvae_mse) / avg_vqvae_mse * 100

        print(f"Average MSE - VQ-VAE: {avg_vqvae_mse:.4f}, RQ-VAE: {avg_rqvae_mse:.4f}")
        print(f"RQ-VAE improvement: {improvement:.1f}% lower MSE")

        return vqvae_mse, rqvae_mse


def analyze_rqvae_hierarchy(rqvae_model, test_loader):
    """
    Analyze the hierarchical quantization in RQ-VAE
    """
    print("\n🔍 Analyzing RQ-VAE hierarchical quantization...")

    rqvae_model.eval()
    with torch.no_grad():
        # Get a test batch
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:4]  # Use 4 examples

        # Get detailed outputs
        outputs = rqvae_model(test_images)

        if 'quantized_list' in outputs:
            quantized_list = outputs['quantized_list']
        else:
            print("Warning: quantized_list not found in outputs")
            return

        n_levels = len(quantized_list)
        print(f"RQ-VAE uses {n_levels} quantization levels")

        # Analyze reconstruction quality at each level
        reconstruction_errors = []
        cumulative_reconstructions = []

        cumulative_quantized = torch.zeros_like(quantized_list[0])

        for i, q_level in enumerate(quantized_list):
            cumulative_quantized = cumulative_quantized + q_level
            recon = rqvae_model.decode(cumulative_quantized)
            cumulative_reconstructions.append(recon)

            # Calculate error
            error = F.mse_loss(recon, test_images).item()
            reconstruction_errors.append(error)
            print(f"  Level {i+1}: MSE = {error:.4f}")

        # Visualize hierarchical reconstruction
        fig, axes = plt.subplots(n_levels + 1, 4, figsize=(10, (n_levels + 1) * 2))
        fig.suptitle('RQ-VAE Hierarchical Reconstruction Process', fontsize=14)

        for img_idx in range(4):
            # Original
            orig_img = test_images[img_idx].permute(1, 2, 0).clamp(0, 1)
            axes[0, img_idx].imshow(orig_img)
            axes[0, img_idx].set_title(f'Original {img_idx+1}')
            axes[0, img_idx].axis('off')

            # Each level
            for level in range(n_levels):
                recon_img = cumulative_reconstructions[level][img_idx].permute(1, 2, 0).clamp(0, 1)
                axes[level + 1, img_idx].imshow(recon_img)
                axes[level + 1, img_idx].set_title(f'Level {level+1}\nMSE: {reconstruction_errors[level]:.4f}')
                axes[level + 1, img_idx].axis('off')

        plt.tight_layout()

        # Save hierarchy analysis
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        plt.savefig(os.path.join(output_dir, 'rqvae_hierarchy_analysis.png'),
                   dpi=300, bbox_inches='tight')
        print(f"Hierarchy analysis saved to outputs/")

        plt.show()

        return reconstruction_errors


def compare_training_curves(vqvae_history, rqvae_history):
    """
    Compare training curves from both models
    """
    print("\n📊 Comparing training curves...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('VQ-VAE vs RQ-VAE Training Comparison', fontsize=16)

    epochs = range(1, len(vqvae_history['train_loss']) + 1)

    # Total loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, vqvae_history['test_loss'], label='VQ-VAE', color='blue')
    ax1.plot(epochs, rqvae_history['test_loss'], label='RQ-VAE', color='red')
    ax1.set_title('Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Reconstruction loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, vqvae_history['test_recon_loss'], label='VQ-VAE', color='blue')
    ax2.plot(epochs, rqvae_history['test_recon_loss'], label='RQ-VAE', color='red')
    ax2.set_title('Reconstruction Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # VQ/RQ loss
    ax3 = axes[1, 0]
    ax3.plot(epochs, vqvae_history['test_vq_loss'], label='VQ-VAE', color='blue')
    ax3.plot(epochs, rqvae_history['test_vq_loss'], label='RQ-VAE', color='red')
    ax3.set_title('Quantization Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('VQ/RQ Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Perplexity
    ax4 = axes[1, 1]
    ax4.plot(epochs, vqvae_history['perplexity'], label='VQ-VAE', color='blue')
    ax4.plot(epochs, rqvae_history['perplexity'], label='RQ-VAE', color='red')
    ax4.set_title('Perplexity')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Perplexity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save training comparison
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'training_curves_comparison.png'),
               dpi=300, bbox_inches='tight')
    print(f"Training curves comparison saved to outputs/")

    plt.show()


def main():
    """
    Run the complete VQ-VAE vs RQ-VAE comparison
    """
    print("🎓 VQ-VAE vs RQ-VAE COMPARISON EDUCATIONAL DEMO")
    print("This demo compares standard VQ-VAE with Residual Quantized VAE")
    print("to show the benefits of hierarchical quantization.\n")

    try:
        # Configuration
        image_size = 32
        n_images = 600
        batch_size = 32
        num_epochs = 35
        embedding_dim = 64
        num_embeddings = 128
        num_quantizers = 4  # For RQ-VAE

        print("Configuration:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Number of images: {n_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training epochs: {num_epochs}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Codebook size: {num_embeddings}")
        print(f"  RQ-VAE quantization levels: {num_quantizers}")

        # Generate complex data
        images = generate_complex_images(n_images=n_images, image_size=image_size)

        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        n_train = int(len(images) * 0.8)
        train_images = images[:n_train]
        test_images = images[n_train:]

        train_dataset = TensorDataset(train_images, train_images)
        test_dataset = TensorDataset(test_images, test_images)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nData prepared: {len(train_images)} train, {len(test_images)} test samples")

        # Create models
        print(f"\n🏗️ Creating models...")

        vqvae_model = VQVAE(
            in_channels=3,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=[64, 128],
            num_residual_layers=2,
            residual_hidden_dim=32
        )

        rqvae_model = RQVAE(
            in_channels=3,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            num_quantizers=num_quantizers,
            hidden_dims=[64, 128],
            num_residual_layers=2,
            residual_hidden_dim=32
        )

        vqvae_params = sum(p.numel() for p in vqvae_model.parameters())
        rqvae_params = sum(p.numel() for p in rqvae_model.parameters())

        print(f"VQ-VAE parameters: {vqvae_params:,}")
        print(f"RQ-VAE parameters: {rqvae_params:,}")
        print(f"Parameter ratio: {rqvae_params/vqvae_params:.2f}x")

        # Train both models
        vqvae_history, vqvae_metrics = train_model(
            vqvae_model, train_loader, test_loader, "VQ-VAE", num_epochs
        )

        rqvae_history, rqvae_metrics = train_model(
            rqvae_model, train_loader, test_loader, "RQ-VAE", num_epochs
        )

        # Compare results
        print(f"\n📊 FINAL COMPARISON:")
        print(f"VQ-VAE  - Recon Loss: {vqvae_metrics['final_recon_loss']:.4f}, "
              f"VQ Loss: {vqvae_metrics['final_vq_loss']:.4f}, "
              f"Perplexity: {vqvae_metrics['final_perplexity']:.1f}")
        print(f"RQ-VAE  - Recon Loss: {rqvae_metrics['final_recon_loss']:.4f}, "
              f"RQ Loss: {rqvae_metrics['final_vq_loss']:.4f}, "
              f"Perplexity: {rqvae_metrics['final_perplexity']:.1f}")

        improvement = (vqvae_metrics['final_recon_loss'] - rqvae_metrics['final_recon_loss']) / vqvae_metrics['final_recon_loss'] * 100
        print(f"RQ-VAE Improvement: {improvement:.1f}% lower reconstruction loss")

        # Detailed analysis
        compare_training_curves(vqvae_history, rqvae_history)
        compare_reconstructions(vqvae_model, rqvae_model, test_loader)
        analyze_rqvae_hierarchy(rqvae_model, test_loader)

        print("\n" + "=" * 60)
        print("VQ-VAE vs RQ-VAE COMPARISON COMPLETED! 🎉")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. RQ-VAE achieves better reconstruction quality than VQ-VAE")
        print("2. Hierarchical quantization captures details at multiple scales")
        print("3. Each quantization level reduces the residual error")
        print("4. RQ-VAE uses more parameters but provides better compression quality")
        print("5. The trade-off between model complexity and performance is favorable")

        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        print(f"\n📁 All output files saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()