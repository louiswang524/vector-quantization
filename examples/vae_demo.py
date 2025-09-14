"""
Variational AutoEncoder (VAE) Educational Demonstration

This example demonstrates the key concepts of Variational AutoEncoders:
1. Probabilistic encoding vs deterministic encoding
2. The reparameterization trick and its importance
3. KL divergence regularization and its effects
4. Generative capabilities (sampling new data)
5. β-VAE and disentanglement
6. Comparison with standard AutoEncoders

Educational Goals:
- Understand probabilistic latent representations
- Learn about the reparameterization trick
- See how KL regularization affects the latent space
- Experience VAE's generative capabilities
- Compare VAE with standard autoencoders
- Understand the ELBO objective
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

from vector_quantization import VAE, AutoEncoder


def generate_structured_images(n_images=800, image_size=32):
    """
    Generate images with clear structure for VAE analysis

    Creates images that should enable disentangled representations:
    - Position variations
    - Shape variations
    - Color variations
    - Size variations

    This structured data helps demonstrate VAE's ability to learn
    meaningful latent representations.

    Args:
        n_images: Number of images to generate
        image_size: Size of square images

    Returns:
        Tensor of images and labels for analysis
    """
    print(f"Generating {n_images} structured images for VAE training...")

    images = torch.zeros(n_images, 3, image_size, image_size)
    labels = []  # For analysis purposes

    for i in range(n_images):
        # Randomly choose shape type
        shape_type = np.random.choice(['circle', 'square', 'triangle'])

        # Random position (but not too close to edges)
        center_x = np.random.randint(8, image_size - 8)
        center_y = np.random.randint(8, image_size - 8)

        # Random size
        size = np.random.randint(3, 8)

        # Random color
        color = torch.rand(3) * 0.8 + 0.1  # Avoid very dark/bright colors

        # Random background color (different from shape)
        bg_color = torch.rand(3) * 0.3

        # Fill background
        for c in range(3):
            images[i, c].fill_(bg_color[c])

        # Create shape
        if shape_type == 'circle':
            y_grid, x_grid = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
            mask = ((x_grid - center_x)**2 + (y_grid - center_y)**2) <= size**2

        elif shape_type == 'square':
            mask = torch.zeros(image_size, image_size, dtype=torch.bool)
            y1, y2 = max(0, center_y - size), min(image_size, center_y + size)
            x1, x2 = max(0, center_x - size), min(image_size, center_x + size)
            mask[y1:y2, x1:x2] = True

        elif shape_type == 'triangle':
            mask = torch.zeros(image_size, image_size, dtype=torch.bool)
            for dy in range(-size, size + 1):
                for dx in range(-abs(dy), abs(dy) + 1):
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < image_size and 0 <= x < image_size:
                        mask[y, x] = True

        # Apply color to shape
        for c in range(3):
            images[i, c][mask] = color[c]

        # Store label for analysis
        labels.append({
            'shape': shape_type,
            'center_x': center_x,
            'center_y': center_y,
            'size': size,
            'color': color.numpy()
        })

    print(f"Generated structured images with:")
    print(f"  - 3 shape types: circle, square, triangle")
    print(f"  - Position range: 8 to {image_size-8}")
    print(f"  - Size range: 3 to 7")
    print(f"  - Random colors and backgrounds")

    return images, labels


def train_vae(model, train_loader, test_loader, num_epochs=50, learning_rate=1e-3):
    """
    Train the VAE model

    Args:
        model: VAE model to train
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate

    Returns:
        Training history
    """
    print(f"\n🚀 Training VAE for {num_epochs} epochs...")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'test_total_loss': [],
        'test_recon_loss': [],
        'test_kl_loss': []
    }

    best_test_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_total = train_recon = train_kl = 0
        train_samples = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(images)
            loss = outputs['total_loss']

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            train_total += outputs['total_loss'].item() * batch_size
            train_recon += outputs['reconstruction_loss'].item() * batch_size
            train_kl += outputs['kl_loss'].item() * batch_size
            train_samples += batch_size

        # Average training losses
        train_total /= train_samples
        train_recon /= train_samples
        train_kl /= train_samples

        # Testing phase
        model.eval()
        test_total = test_recon = test_kl = 0
        test_samples = 0

        with torch.no_grad():
            for images, _ in test_loader:
                outputs = model(images)

                batch_size = images.size(0)
                test_total += outputs['total_loss'].item() * batch_size
                test_recon += outputs['reconstruction_loss'].item() * batch_size
                test_kl += outputs['kl_loss'].item() * batch_size
                test_samples += batch_size

        # Average test losses
        test_total /= test_samples
        test_recon /= test_samples
        test_kl /= test_samples

        # Store history
        history['train_total_loss'].append(train_total)
        history['train_recon_loss'].append(train_recon)
        history['train_kl_loss'].append(train_kl)
        history['test_total_loss'].append(test_total)
        history['test_recon_loss'].append(test_recon)
        history['test_kl_loss'].append(test_kl)

        # Update learning rate
        scheduler.step()

        # Track best model
        if test_total < best_test_loss:
            best_test_loss = test_total

        # Print progress
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1:3d}: "
                  f"Total = {test_total:.4f} "
                  f"(Recon: {test_recon:.4f}, KL: {test_kl:.4f}), "
                  f"Time = {elapsed:.1f}s")

    print(f"  ✅ Training completed! Best test loss: {best_test_loss:.4f}")
    return history


def compare_vae_vs_autoencoder(vae_model, ae_model, test_loader, n_examples=6):
    """
    Compare VAE and AutoEncoder reconstructions
    """
    print(f"\n🔍 Comparing VAE vs AutoEncoder reconstructions...")

    vae_model.eval()
    ae_model.eval()

    with torch.no_grad():
        # Get test images
        test_images, _ = next(iter(test_loader))
        test_images = test_images[:n_examples]

        # VAE reconstructions (using mean for deterministic comparison)
        vae_outputs = vae_model(test_images)
        vae_recons = vae_outputs['reconstructed']

        # AutoEncoder reconstructions
        ae_outputs = ae_model(test_images)
        ae_recons = ae_outputs['reconstructed']

        # Calculate errors
        vae_errors = [F.mse_loss(vae_recons[i], test_images[i]).item() for i in range(n_examples)]
        ae_errors = [F.mse_loss(ae_recons[i], test_images[i]).item() for i in range(n_examples)]

        # Visualize comparison
        fig, axes = plt.subplots(3, n_examples, figsize=(n_examples * 2.5, 7))
        fig.suptitle('VAE vs AutoEncoder Reconstruction Comparison', fontsize=16)

        for i in range(n_examples):
            # Original
            orig_img = test_images[i].permute(1, 2, 0).clamp(0, 1)
            axes[0, i].imshow(orig_img)
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')

            # AutoEncoder reconstruction
            ae_img = ae_recons[i].permute(1, 2, 0).clamp(0, 1)
            axes[1, i].imshow(ae_img)
            axes[1, i].set_title(f'AutoEncoder\nMSE: {ae_errors[i]:.4f}')
            axes[1, i].axis('off')

            # VAE reconstruction
            vae_img = vae_recons[i].permute(1, 2, 0).clamp(0, 1)
            axes[2, i].imshow(vae_img)
            axes[2, i].set_title(f'VAE\nMSE: {vae_errors[i]:.4f}')
            axes[2, i].axis('off')

        plt.tight_layout()

        # Save comparison
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'vae_vs_autoencoder_comparison.png'),
                   dpi=300, bbox_inches='tight')
        print(f"VAE vs AE comparison saved to outputs/")

        plt.show()

        # Statistics
        avg_vae_error = sum(vae_errors) / len(vae_errors)
        avg_ae_error = sum(ae_errors) / len(ae_errors)

        print(f"Average reconstruction errors:")
        print(f"  AutoEncoder: {avg_ae_error:.6f}")
        print(f"  VAE: {avg_vae_error:.6f}")
        print(f"  Difference: {avg_vae_error - avg_ae_error:.6f} (VAE typically higher due to regularization)")


def demonstrate_vae_generation(model, num_samples=12):
    """
    Demonstrate VAE's generative capabilities
    """
    print(f"\n✨ Demonstrating VAE generation capabilities...")

    model.eval()

    # Generate new samples
    generated_samples = model.sample(num_samples)

    # Visualize generated samples
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    fig.suptitle('VAE Generated Samples (from random noise)', fontsize=16)

    for i in range(num_samples):
        row, col = i // 4, i % 4

        gen_img = generated_samples[i].permute(1, 2, 0).clamp(0, 1)
        axes[row, col].imshow(gen_img)
        axes[row, col].set_title(f'Generated {i+1}')
        axes[row, col].axis('off')

    plt.tight_layout()

    # Save generated samples
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'vae_generated_samples.png'),
               dpi=300, bbox_inches='tight')
    print(f"Generated samples saved to outputs/")

    plt.show()

    print(f"Generated {num_samples} new samples from random latent codes")
    print("These samples demonstrate VAE's ability to generate novel data")


def analyze_latent_space_structure(model, test_loader):
    """
    Analyze the structure of VAE's latent space
    """
    print(f"\n🔬 Analyzing VAE latent space structure...")

    # Comprehensive latent space analysis
    analysis = model.analyze_latent_space(test_loader)

    print(f"VAE Latent Space Analysis:")
    print(f"  Active dimensions: {analysis['active_dimensions']:.0f} / {model.latent_dim}")
    print(f"  Total KL divergence: {analysis['total_kl']:.4f}")
    print(f"  Average posterior variance: {torch.mean(analysis['avg_posterior_variance']):.4f}")

    # Visualize latent space properties
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('VAE Latent Space Analysis', fontsize=16)

    # Posterior means
    axes[0, 0].bar(range(len(analysis['posterior_mean'])), analysis['posterior_mean'].numpy())
    axes[0, 0].set_title('Posterior Means')
    axes[0, 0].set_xlabel('Latent Dimension')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].grid(True, alpha=0.3)

    # Posterior standard deviations
    axes[0, 1].bar(range(len(analysis['posterior_std'])), analysis['posterior_std'].numpy())
    axes[0, 1].set_title('Posterior Standard Deviations')
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].grid(True, alpha=0.3)

    # KL divergence per dimension
    axes[1, 0].bar(range(len(analysis['kl_per_dimension'])), analysis['kl_per_dimension'].numpy())
    axes[1, 0].set_title('KL Divergence per Dimension')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].grid(True, alpha=0.3)

    # Average posterior variance
    axes[1, 1].bar(range(len(analysis['avg_posterior_variance'])), analysis['avg_posterior_variance'].numpy())
    axes[1, 1].set_title('Average Posterior Variance')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Variance')
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Prior variance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save analysis
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'vae_latent_space_analysis.png'),
               dpi=300, bbox_inches='tight')
    print(f"Latent space analysis saved to outputs/")

    plt.show()

    return analysis


def demonstrate_beta_vae_effects(test_loader, latent_dim=32, image_size=32):
    """
    Demonstrate the effects of different β values in β-VAE
    """
    print(f"\n⚙️ Demonstrating β-VAE effects...")

    # Test different β values
    beta_values = [0.1, 1.0, 4.0]
    models = {}

    print(f"Training β-VAE models with β = {beta_values}")

    for beta in beta_values:
        print(f"\nTraining β-VAE with β = {beta}:")

        # Create model with specific β
        model = VAE(
            in_channels=3,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128],
            input_size=image_size,
            beta=beta
        )

        # Quick training (fewer epochs for comparison)
        history = train_vae(model, test_loader, test_loader, num_epochs=20, learning_rate=2e-3)
        models[beta] = model

    # Compare results
    print(f"\nComparing β-VAE results:")

    with torch.no_grad():
        test_images, _ = next(iter(test_loader))
        test_batch = test_images[:4]

        fig, axes = plt.subplots(len(beta_values), 4, figsize=(12, len(beta_values) * 2.5))
        fig.suptitle('β-VAE Comparison: Effect of β on Reconstructions', fontsize=16)

        for i, beta in enumerate(beta_values):
            model = models[beta]
            model.eval()

            outputs = model(test_batch)
            reconstructed = outputs['reconstructed']

            for j in range(4):
                if len(beta_values) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                if j == 0:
                    # Show original in first column
                    img = test_batch[j].permute(1, 2, 0).clamp(0, 1)
                    ax.set_ylabel(f'β = {beta}', fontsize=12)
                else:
                    # Show reconstruction
                    img = reconstructed[j].permute(1, 2, 0).clamp(0, 1)

                ax.imshow(img)
                if i == 0:  # Top row
                    if j == 0:
                        ax.set_title('Original')
                    else:
                        ax.set_title(f'Reconstruction {j}')
                ax.axis('off')

    plt.tight_layout()

    # Save β-VAE comparison
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'beta_vae_comparison.png'),
               dpi=300, bbox_inches='tight')
    print(f"β-VAE comparison saved to outputs/")

    plt.show()

    # Print analysis
    print(f"\nβ-VAE Analysis:")
    print(f"  β = 0.1: Lower regularization → Better reconstruction, less structured latent space")
    print(f"  β = 1.0: Standard VAE → Balanced reconstruction and regularization")
    print(f"  β = 4.0: Higher regularization → More structured latent space, potentially lower quality")

    return models


def visualize_training_curves(history):
    """
    Visualize VAE training curves showing loss components
    """
    print(f"\n📊 Visualizing VAE training curves...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('VAE Training Curves', fontsize=16)

    epochs = range(1, len(history['train_total_loss']) + 1)

    # Total loss
    axes[0].plot(epochs, history['train_total_loss'], label='Train', color='blue')
    axes[0].plot(epochs, history['test_total_loss'], label='Test', color='red')
    axes[0].set_title('Total Loss (ELBO)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon_loss'], label='Train', color='blue')
    axes[1].plot(epochs, history['test_recon_loss'], label='Test', color='red')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # KL loss
    axes[2].plot(epochs, history['train_kl_loss'], label='Train', color='blue')
    axes[2].plot(epochs, history['test_kl_loss'], label='Test', color='red')
    axes[2].set_title('KL Divergence Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save training curves
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'vae_training_curves.png'),
               dpi=300, bbox_inches='tight')
    print(f"Training curves saved to outputs/")

    plt.show()


def demonstrate_latent_interpolation(model, test_loader):
    """
    Demonstrate smooth interpolation in VAE latent space
    """
    print(f"\n🔄 Demonstrating VAE latent space interpolation...")

    model.eval()
    with torch.no_grad():
        # Get two test images
        test_images, _ = next(iter(test_loader))
        img1, img2 = test_images[0], test_images[1]

        # VAE interpolation
        num_steps = 10
        interpolations = model.interpolate(img1.unsqueeze(0), img2.unsqueeze(0), num_steps)

        # Visualize
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
        fig.suptitle('VAE Latent Space Interpolation', fontsize=14)

        for i in range(num_steps):
            interp_img = interpolations[i].permute(1, 2, 0).clamp(0, 1)
            axes[i].imshow(interp_img)
            axes[i].set_title(f'α={i/(num_steps-1):.1f}')
            axes[i].axis('off')

        plt.tight_layout()

        # Save interpolation
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        plt.savefig(os.path.join(output_dir, 'vae_interpolation.png'),
                   dpi=300, bbox_inches='tight')
        print(f"VAE interpolation saved to outputs/")

        plt.show()


def main():
    """
    Run the complete VAE educational demonstration
    """
    print("🎓 VARIATIONAL AUTOENCODER (VAE) EDUCATIONAL DEMO")
    print("This demo teaches the fundamentals of VAEs through")
    print("hands-on examples and comparisons with AutoEncoders.\n")

    try:
        # Configuration
        image_size = 32
        n_images = 600
        batch_size = 32
        num_epochs = 40
        latent_dim = 32
        beta = 1.0

        print("Configuration:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Number of images: {n_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training epochs: {num_epochs}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  β parameter: {beta}")

        # Generate structured data
        images, labels = generate_structured_images(n_images=n_images, image_size=image_size)

        # Create data loaders
        n_train = int(len(images) * 0.8)
        train_images = images[:n_train]
        test_images = images[n_train:]

        train_dataset = TensorDataset(train_images, train_images)
        test_dataset = TensorDataset(test_images, test_images)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nData prepared: {len(train_images)} train, {len(test_images)} test samples")

        # Create models
        print(f"\n🏗️ Creating VAE and AutoEncoder models...")

        vae_model = VAE(
            in_channels=3,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128],
            input_size=image_size,
            beta=beta
        )

        ae_model = AutoEncoder(
            in_channels=3,
            latent_dim=latent_dim,
            hidden_dims=[32, 64, 128],
            input_size=image_size
        )

        # Train VAE
        print(f"\n🚀 Training VAE...")
        vae_history = train_vae(vae_model, train_loader, test_loader, num_epochs)

        # Quick train AutoEncoder for comparison
        print(f"\n🚀 Training AutoEncoder for comparison...")
        from examples.autoencoder_demo import train_autoencoder
        ae_history = train_autoencoder(ae_model, train_loader, test_loader, num_epochs=25, learning_rate=1e-3)

        # Demonstrations and analysis
        visualize_training_curves(vae_history)
        compare_vae_vs_autoencoder(vae_model, ae_model, test_loader)
        demonstrate_vae_generation(vae_model)
        demonstrate_latent_interpolation(vae_model, test_loader)
        analyze_latent_space_structure(vae_model, test_loader)

        # Advanced demonstrations
        demonstrate_beta_vae_effects(train_loader, latent_dim=24, image_size=image_size)

        print("\n" + "=" * 60)
        print("VAE DEMONSTRATION COMPLETED! 🎉")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. VAEs learn probabilistic latent representations (μ, σ²)")
        print("2. Reparameterization trick enables backpropagation through sampling")
        print("3. KL divergence regularizes latent space to match prior N(0,I)")
        print("4. VAEs can generate new samples by sampling from prior")
        print("5. β parameter controls reconstruction vs regularization trade-off")
        print("6. VAE reconstructions may be blurrier due to regularization")
        print("7. VAE latent spaces are more structured and interpretable")

        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        print(f"\n📁 All visualizations saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()