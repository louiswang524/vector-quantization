"""
AutoEncoder Educational Demonstration

This example demonstrates the fundamentals of autoencoders and shows:
1. How autoencoders compress and reconstruct data
2. The effect of different latent dimensions
3. Latent space visualization and analysis
4. Interpolation in latent space
5. Comparison with different compression levels

Educational Goals:
- Understand the encoder-decoder architecture
- See how the bottleneck forces compression
- Learn about reconstruction vs compression trade-offs
- Visualize learned latent representations
- Understand latent space interpolation
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

from vector_quantization import AutoEncoder


def generate_diverse_images(n_images=800, image_size=32):
    """
    Generate diverse synthetic images to test autoencoder capabilities

    Creates various patterns that test different aspects of compression:
    - Simple geometric shapes (easy to compress)
    - Complex textures (harder to compress)
    - Gradients (require smooth representations)
    - Mixed patterns (test generalization)

    Args:
        n_images: Number of images to generate
        image_size: Size of square images

    Returns:
        Tensor of images (n_images, 3, image_size, image_size)
    """
    print(f"Generating {n_images} diverse synthetic images...")

    images = torch.zeros(n_images, 3, image_size, image_size)

    for i in range(n_images):
        pattern_type = i % 8

        if pattern_type == 0:  # Solid colors with geometric shapes
            base_color = torch.rand(3) * 0.8 + 0.1
            for c in range(3):
                images[i, c].fill_(base_color[c])

            # Add a rectangle
            x1, y1 = np.random.randint(2, image_size//3, 2)
            x2, y2 = x1 + np.random.randint(4, image_size//2), y1 + np.random.randint(4, image_size//2)
            shape_color = torch.rand(3)
            for c in range(3):
                images[i, c, y1:min(y2, image_size), x1:min(x2, image_size)] = shape_color[c]

        elif pattern_type == 1:  # Gradients
            # Linear gradients
            direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
            color1 = torch.rand(3)
            color2 = torch.rand(3)

            if direction == 'horizontal':
                for x in range(image_size):
                    weight = x / (image_size - 1)
                    color = (1 - weight) * color1 + weight * color2
                    for c in range(3):
                        images[i, c, :, x] = color[c]
            elif direction == 'vertical':
                for y in range(image_size):
                    weight = y / (image_size - 1)
                    color = (1 - weight) * color1 + weight * color2
                    for c in range(3):
                        images[i, c, y, :] = color[c]

        elif pattern_type == 2:  # Circles and ellipses
            bg_color = torch.rand(3) * 0.3
            for c in range(3):
                images[i, c].fill_(bg_color[c])

            # Add multiple circles
            n_circles = np.random.randint(1, 4)
            for _ in range(n_circles):
                center_x, center_y = np.random.randint(5, image_size-5, 2)
                radius_x = np.random.randint(3, 8)
                radius_y = np.random.randint(3, 8)
                circle_color = torch.rand(3) * 0.8 + 0.1

                y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                mask = ((x_grid - center_x)/radius_x)**2 + ((y_grid - center_y)/radius_y)**2 <= 1

                for c in range(3):
                    images[i, c][mask] = circle_color[c]

        elif pattern_type == 3:  # Noise patterns
            # Structured noise
            base_noise = torch.randn(image_size, image_size) * 0.3
            base_color = torch.rand(3) * 0.5 + 0.25

            for c in range(3):
                color_noise = torch.randn(image_size, image_size) * 0.1
                images[i, c] = torch.clamp(base_color[c] + base_noise + color_noise, 0, 1)

        elif pattern_type == 4:  # Stripes with texture
            stripe_width = np.random.randint(2, 8)
            color1 = torch.rand(3) * 0.6
            color2 = torch.rand(3) * 0.6 + 0.4

            for y in range(0, image_size, stripe_width * 2):
                # Base stripe
                for c in range(3):
                    images[i, c, y:min(y+stripe_width, image_size), :] = color1[c]
                    if y + stripe_width < image_size:
                        images[i, c, y+stripe_width:min(y+2*stripe_width, image_size), :] = color2[c]

                # Add texture
                texture = torch.randn(min(stripe_width, image_size-y), image_size) * 0.05
                for c in range(3):
                    if y < image_size:
                        images[i, c, y:min(y+stripe_width, image_size), :] += texture[:min(stripe_width, image_size-y), :]

        elif pattern_type == 5:  # Radial patterns
            center_x, center_y = image_size // 2, image_size // 2
            max_radius = image_size // 2

            y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            distances = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
            angles = torch.atan2(y_grid - center_y, x_grid - center_x)

            # Radial gradient with angular variation
            radial_pattern = torch.cos(distances * np.pi / max_radius * 2)
            angular_pattern = torch.sin(angles * np.random.randint(2, 8))
            combined_pattern = (radial_pattern * angular_pattern + 1) / 2

            base_color = torch.rand(3)
            for c in range(3):
                images[i, c] = base_color[c] * 0.3 + combined_pattern * 0.7

        elif pattern_type == 6:  # Checkerboard variations
            check_size = np.random.randint(2, 6)
            color1 = torch.rand(3) * 0.4
            color2 = torch.rand(3) * 0.4 + 0.6

            for y in range(0, image_size, check_size):
                for x in range(0, image_size, check_size):
                    if ((y // check_size) + (x // check_size)) % 2 == 0:
                        color = color1 + torch.randn(3) * 0.05
                    else:
                        color = color2 + torch.randn(3) * 0.05

                    color = torch.clamp(color, 0, 1)
                    for c in range(3):
                        images[i, c, y:min(y+check_size, image_size),
                              x:min(x+check_size, image_size)] = color[c]

        elif pattern_type == 7:  # Complex mixed patterns
            # Background gradient
            x_coords = torch.linspace(0, 1, image_size)
            y_coords = torch.linspace(0, 1, image_size)
            X, Y = torch.meshgrid(x_coords, y_coords, indexing='ij')

            bg_pattern = torch.sin(X * np.pi) * torch.cos(Y * np.pi * 2) * 0.3 + 0.5
            bg_color = torch.rand(3)

            for c in range(3):
                images[i, c] = bg_color[c] * bg_pattern

            # Add geometric shapes
            n_shapes = np.random.randint(1, 3)
            for _ in range(n_shapes):
                if np.random.choice([True, False]):
                    # Circle
                    center_x, center_y = np.random.randint(4, image_size-4, 2)
                    radius = np.random.randint(3, 8)
                    y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
                    mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= radius**2
                else:
                    # Rectangle
                    x1, y1 = np.random.randint(2, image_size//2, 2)
                    x2, y2 = x1 + np.random.randint(3, image_size//3), y1 + np.random.randint(3, image_size//3)
                    mask = torch.zeros(image_size, image_size, dtype=torch.bool)
                    mask[y1:min(y2, image_size), x1:min(x2, image_size)] = True

                shape_color = torch.rand(3) * 0.6 + 0.2
                for c in range(3):
                    images[i, c][mask] = shape_color[c]

    print(f"Generated diverse images with 8 pattern types:")
    print("  1. Solid colors with geometric shapes")
    print("  2. Linear gradients")
    print("  3. Circles and ellipses")
    print("  4. Noise patterns")
    print("  5. Textured stripes")
    print("  6. Radial patterns")
    print("  7. Checkerboard variations")
    print("  8. Complex mixed patterns")

    return images


def train_autoencoder(model, train_loader, test_loader, num_epochs=50, learning_rate=1e-3):
    """
    Train the autoencoder model

    Args:
        model: AutoEncoder model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate

    Returns:
        Training history
    """
    print(f"\n🚀 Training AutoEncoder for {num_epochs} epochs...")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    history = {
        'train_loss': [],
        'test_loss': [],
        'train_mse': [],
        'test_mse': []
    }

    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_samples = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(images)
            loss = outputs['loss']

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

        train_loss /= train_samples

        # Testing phase
        model.eval()
        test_loss = 0
        test_samples = 0

        with torch.no_grad():
            for images, _ in test_loader:
                outputs = model(images)
                loss = outputs['loss']

                batch_size = images.size(0)
                test_loss += loss.item() * batch_size
                test_samples += batch_size

        test_loss /= test_samples

        # Store history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_mse'].append(train_loss)  # For AE, loss is MSE
        history['test_mse'].append(test_loss)

        # Update learning rate
        scheduler.step()

        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss

        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1:3d}: "
                  f"Train Loss = {train_loss:.6f}, "
                  f"Test Loss = {test_loss:.6f}, "
                  f"Time = {elapsed:.1f}s")

    print(f"  ✅ Training completed! Best test loss: {best_test_loss:.6f}")
    return history


def compare_latent_dimensions():
    """
    Compare autoencoders with different latent dimensions
    """
    print("\n" + "=" * 60)
    print("LATENT DIMENSION COMPARISON")
    print("=" * 60)

    # Generate test data
    test_images = generate_diverse_images(n_images=200, image_size=32)
    test_dataset = TensorDataset(test_images, test_images)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Test different latent dimensions
    latent_dims = [16, 32, 64, 128, 256]
    results = {}

    print(f"Testing latent dimensions: {latent_dims}")
    print("-" * 50)

    for latent_dim in latent_dims:
        print(f"Testing latent dimension {latent_dim}:")

        # Create model
        model = AutoEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128],
            input_size=32
        )

        # Quick training (fewer epochs for comparison)
        train_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        history = train_autoencoder(model, train_loader, test_loader, num_epochs=20, learning_rate=2e-3)

        # Analyze reconstruction quality
        model.eval()
        total_error = 0
        with torch.no_grad():
            for images, _ in test_loader:
                outputs = model(images)
                total_error += outputs['loss'].item() * images.size(0)

        avg_error = total_error / len(test_images)

        # Calculate compression ratio
        input_size = 3 * 32 * 32
        compression_ratio = input_size / latent_dim

        results[latent_dim] = {
            'avg_error': avg_error,
            'compression_ratio': compression_ratio,
            'final_train_loss': history['train_loss'][-1],
            'final_test_loss': history['test_loss'][-1]
        }

        print(f"  Average reconstruction error: {avg_error:.6f}")
        print(f"  Compression ratio: {compression_ratio:.1f}:1")
        print(f"  Final test loss: {history['test_loss'][-1]:.6f}")
        print()

    # Summary
    print("Latent Dimension Comparison Summary:")
    print("Dim   | Compression | Test Loss  | Reconstruction Error")
    print("------|-------------|------------|--------------------")
    for dim in latent_dims:
        r = results[dim]
        print(f"{dim:4d}  | {r['compression_ratio']:8.1f}:1 | {r['final_test_loss']:8.6f} | {r['avg_error']:8.6f}")

    return results


def visualize_reconstructions_and_latent_space(model, test_loader, n_examples=8):
    """
    Visualize reconstruction quality and latent space properties
    """
    print(f"\n🎨 Visualizing reconstructions and latent space...")

    model.eval()
    with torch.no_grad():
        # Get test batch
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:n_examples]

        # Get reconstructions and latent codes
        outputs = model(test_images)
        reconstructed = outputs['reconstructed']
        latent_codes = outputs['latent']

        # Calculate reconstruction errors
        errors = [F.mse_loss(reconstructed[i], test_images[i]).item() for i in range(n_examples)]

        # Create visualization
        fig, axes = plt.subplots(3, n_examples, figsize=(n_examples * 2, 6))
        fig.suptitle('AutoEncoder Reconstruction Analysis', fontsize=16)

        for i in range(n_examples):
            # Original image
            orig_img = test_images[i].permute(1, 2, 0).clamp(0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # Reconstructed image
            recon_img = reconstructed[i].permute(1, 2, 0).clamp(0, 1)
            axes[1, i].imshow(recon_img)
            axes[1, i].set_title(f'Reconstructed\nMSE: {errors[i]:.4f}')
            axes[1, i].axis('off')

            # Latent representation (first 16 dimensions as heatmap)
            latent_vis = latent_codes[i][:16].unsqueeze(0)  # Take first 16 dims
            im = axes[2, i].imshow(latent_vis, cmap='RdBu', aspect='auto')
            axes[2, i].set_title(f'Latent (16D)')
            axes[2, i].set_xticks(range(0, 16, 4))

        plt.tight_layout()

        # Save visualization
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'autoencoder_reconstructions.png'),
                   dpi=300, bbox_inches='tight')
        print(f"Reconstruction analysis saved to outputs/")

        plt.show()

        # Print statistics
        avg_error = sum(errors) / len(errors)
        print(f"Average reconstruction error: {avg_error:.6f}")

        return errors


def demonstrate_latent_interpolation(model, test_loader):
    """
    Demonstrate smooth interpolation in latent space
    """
    print(f"\n🔄 Demonstrating latent space interpolation...")

    model.eval()
    with torch.no_grad():
        # Get two different test images
        test_images, _ = next(iter(test_loader))
        img1, img2 = test_images[0], test_images[1]

        # Perform interpolation
        num_steps = 10
        interpolations = model.interpolate_in_latent_space(
            img1.unsqueeze(0), img2.unsqueeze(0), num_steps
        )

        # Visualize interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
        fig.suptitle('Latent Space Interpolation', fontsize=14)

        for i in range(num_steps):
            interp_img = interpolations[i].permute(1, 2, 0).clamp(0, 1)
            axes[i].imshow(interp_img)
            axes[i].set_title(f'α={i/(num_steps-1):.1f}')
            axes[i].axis('off')

        plt.tight_layout()

        # Save interpolation
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        plt.savefig(os.path.join(output_dir, 'autoencoder_interpolation.png'),
                   dpi=300, bbox_inches='tight')
        print(f"Latent interpolation saved to outputs/")

        plt.show()


def analyze_compression_vs_quality(results):
    """
    Analyze the trade-off between compression and reconstruction quality
    """
    print(f"\n📊 Analyzing compression vs quality trade-off...")

    latent_dims = list(results.keys())
    compression_ratios = [results[d]['compression_ratio'] for d in latent_dims]
    reconstruction_errors = [results[d]['avg_error'] for d in latent_dims]

    # Create trade-off plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('AutoEncoder: Compression vs Quality Analysis', fontsize=16)

    # Compression ratio vs error
    ax1.plot(compression_ratios, reconstruction_errors, 'bo-', markersize=8)
    ax1.set_xlabel('Compression Ratio (input_size : latent_size)')
    ax1.set_ylabel('Reconstruction Error (MSE)')
    ax1.set_title('Quality vs Compression Trade-off')
    ax1.grid(True, alpha=0.3)

    # Annotate points
    for i, dim in enumerate(latent_dims):
        ax1.annotate(f'{dim}D', (compression_ratios[i], reconstruction_errors[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Latent dimension vs error
    ax2.semilogx(latent_dims, reconstruction_errors, 'ro-', markersize=8)
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Reconstruction Error (MSE)')
    ax2.set_title('Error vs Latent Dimension')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save analysis
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'autoencoder_compression_analysis.png'),
               dpi=300, bbox_inches='tight')
    print(f"Compression analysis saved to outputs/")

    plt.show()


def demonstrate_latent_space_analysis(model, test_loader):
    """
    Analyze the learned latent space properties
    """
    print(f"\n🔬 Analyzing learned latent space...")

    # Analyze latent space
    analysis = model.analyze_latent_space(test_loader)

    print(f"Latent space analysis:")
    print(f"  Effective dimensions: {analysis['effective_dim']:.0f} / {model.latent_dim}")
    print(f"  Latent utilization: {analysis['latent_utilization']:.1%}")

    # Visualize latent statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Latent Space Analysis', fontsize=16)

    # Mean activation
    axes[0].bar(range(len(analysis['mean'])), analysis['mean'].numpy())
    axes[0].set_title('Latent Dimension Means')
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Mean Activation')
    axes[0].grid(True, alpha=0.3)

    # Standard deviation
    axes[1].bar(range(len(analysis['std'])), analysis['std'].numpy())
    axes[1].set_title('Latent Dimension Standard Deviations')
    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].grid(True, alpha=0.3)

    # Active dimensions
    axes[2].bar(range(len(analysis['active_dimensions'])), analysis['active_dimensions'].numpy())
    axes[2].set_title('Active Dimensions (Usage > 10%)')
    axes[2].set_xlabel('Latent Dimension')
    axes[2].set_ylabel('Usage Ratio')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save analysis
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'autoencoder_latent_analysis.png'),
               dpi=300, bbox_inches='tight')
    print(f"Latent space analysis saved to outputs/")

    plt.show()


def main():
    """
    Run the complete AutoEncoder educational demonstration
    """
    print("🎓 AUTOENCODER EDUCATIONAL DEMONSTRATION")
    print("This demo teaches the fundamentals of autoencoders")
    print("through hands-on examples and analysis.\n")

    try:
        # Configuration
        image_size = 32
        n_images = 600
        batch_size = 32
        num_epochs = 40
        latent_dim = 64

        print("Configuration:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Number of images: {n_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training epochs: {num_epochs}")
        print(f"  Latent dimension: {latent_dim}")

        # Generate diverse training data
        images = generate_diverse_images(n_images=n_images, image_size=image_size)

        # Create data loaders
        n_train = int(len(images) * 0.8)
        train_images = images[:n_train]
        test_images = images[n_train:]

        train_dataset = TensorDataset(train_images, train_images)
        test_dataset = TensorDataset(test_images, test_images)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nData prepared: {len(train_images)} train, {len(test_images)} test samples")

        # Create and train main model
        print(f"\n🏗️ Creating AutoEncoder...")
        model = AutoEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128],
            input_size=image_size
        )

        # Train model
        history = train_autoencoder(model, train_loader, test_loader, num_epochs)

        # Demonstrations and analysis
        visualize_reconstructions_and_latent_space(model, test_loader)
        demonstrate_latent_interpolation(model, test_loader)
        demonstrate_latent_space_analysis(model, test_loader)

        # Compare different latent dimensions
        latent_comparison = compare_latent_dimensions()
        analyze_compression_vs_quality(latent_comparison)

        print("\n" + "=" * 60)
        print("AUTOENCODER DEMONSTRATION COMPLETED! 🎉")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. AutoEncoders learn compressed representations through bottlenecks")
        print("2. Smaller latent dimensions = higher compression but lower quality")
        print("3. Latent space enables smooth interpolation between data points")
        print("4. Not all latent dimensions may be actively used")
        print("5. Trade-off between reconstruction quality and compression ratio")

        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        print(f"\n📁 All visualizations saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()