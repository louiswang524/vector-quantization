"""
Basic Vector Quantization Demonstration

This example shows the fundamental concepts of vector quantization using
the VectorQuantizer class. It demonstrates:
1. How vector quantization works on simple 2D data
2. The effect of different codebook sizes
3. Visualization of quantization results
4. Analysis of codebook utilization

Educational Goals:
- Understand the basic VQ operation
- See how codebook size affects approximation quality
- Visualize the quantization process
- Learn about the straight-through estimator
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_quantization import VectorQuantizer


def generate_sample_data(n_samples=1000, n_features=2, noise_std=0.1):
    """
    Generate sample 2D data for demonstration

    Creates data with multiple clusters to show how VQ learns to represent
    different regions of the input space.

    Args:
        n_samples: Number of data points to generate
        n_features: Number of features (should be 2 for visualization)
        noise_std: Standard deviation of noise to add

    Returns:
        Data tensor of shape (n_samples, n_features)
    """
    # Create three distinct clusters
    cluster_centers = torch.tensor([
        [-2.0, -2.0],
        [2.0, 2.0],
        [-2.0, 2.0],
        [2.0, -2.0]
    ])

    # Generate points around each cluster
    data_points = []
    samples_per_cluster = n_samples // len(cluster_centers)

    for center in cluster_centers:
        # Generate points around this cluster center
        cluster_points = torch.randn(samples_per_cluster, n_features) * noise_std + center
        data_points.append(cluster_points)

    # Combine all points
    data = torch.cat(data_points, dim=0)

    # Add some random points to make it more interesting
    remaining_samples = n_samples - len(data)
    if remaining_samples > 0:
        random_points = torch.randn(remaining_samples, n_features) * 2
        data = torch.cat([data, random_points], dim=0)

    # Shuffle the data
    indices = torch.randperm(len(data))
    data = data[indices]

    return data


def demonstrate_basic_vq():
    """
    Demonstrate basic vector quantization functionality
    """
    print("=" * 60)
    print("BASIC VECTOR QUANTIZATION DEMONSTRATION")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample data...")
    data = generate_sample_data(n_samples=500, n_features=2, noise_std=0.3)
    print(f"   Generated {len(data)} data points with {data.shape[1]} features")
    print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]")

    # Create vector quantizer
    print("\n2. Creating Vector Quantizer...")
    num_embeddings = 16  # Size of the codebook
    embedding_dim = 2    # Must match data dimensionality

    vq_layer = VectorQuantizer(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        commitment_cost=0.25
    )

    print(f"   Codebook size: {num_embeddings}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Initial codebook range: [{vq_layer.embedding.weight.min():.3f}, {vq_layer.embedding.weight.max():.3f}]")

    # Reshape data for VQ layer (expects 4D input for spatial dimensions)
    # We'll treat each data point as a 1x1 spatial location
    data_reshaped = data.unsqueeze(0).unsqueeze(0)  # (1, 1, n_samples, n_features)

    print("\n3. Applying Vector Quantization...")

    # Forward pass through VQ layer
    quantized, vq_loss, perplexity = vq_layer(data_reshaped)

    # Reshape back to original format
    quantized_data = quantized.squeeze(0).squeeze(0)  # (n_samples, n_features)

    print(f"   VQ Loss: {vq_loss.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.2f} (out of {num_embeddings})")

    # Calculate reconstruction error
    mse = torch.mean((data - quantized_data)**2)
    print(f"   Mean Squared Error: {mse.item():.4f}")

    # Analyze codebook usage
    print("\n4. Analyzing Codebook Usage...")
    distances = vq_layer.get_distance_matrix(data_reshaped)
    distances = distances.squeeze()

    # Find which codebook entry each point was assigned to
    assignments = torch.argmin(distances, dim=1)
    unique_assignments, counts = torch.unique(assignments, return_counts=True)

    print(f"   Used {len(unique_assignments)} out of {num_embeddings} codebook entries")
    print(f"   Usage distribution:")
    for i, (code_idx, count) in enumerate(zip(unique_assignments, counts)):
        if i < 5:  # Show first 5
            print(f"     Code {code_idx.item()}: {count.item()} points ({count.item()/len(data)*100:.1f}%)")
        elif i == 5 and len(unique_assignments) > 5:
            print(f"     ... and {len(unique_assignments)-5} more codes")
            break

    return data, quantized_data, vq_layer, assignments


def compare_codebook_sizes():
    """
    Compare the effect of different codebook sizes
    """
    print("\n" + "=" * 60)
    print("CODEBOOK SIZE COMPARISON")
    print("=" * 60)

    # Generate test data
    data = generate_sample_data(n_samples=300, n_features=2, noise_std=0.2)
    data_reshaped = data.unsqueeze(0).unsqueeze(0)

    # Test different codebook sizes
    codebook_sizes = [4, 8, 16, 32, 64]
    results = {}

    print(f"\nTesting codebook sizes: {codebook_sizes}")
    print("-" * 50)

    for size in codebook_sizes:
        print(f"Codebook size {size}:")

        # Create VQ layer
        vq_layer = VectorQuantizer(
            num_embeddings=size,
            embedding_dim=2,
            commitment_cost=0.25
        )

        # Apply quantization
        quantized, vq_loss, perplexity = vq_layer(data_reshaped)
        quantized_data = quantized.squeeze(0).squeeze(0)

        # Calculate metrics
        mse = torch.mean((data - quantized_data)**2).item()
        perplexity_ratio = perplexity.item() / size

        results[size] = {
            'mse': mse,
            'perplexity': perplexity.item(),
            'perplexity_ratio': perplexity_ratio,
            'vq_loss': vq_loss.item()
        }

        print(f"  MSE: {mse:.4f}")
        print(f"  Perplexity: {perplexity.item():.1f}/{size} ({perplexity_ratio:.1%})")
        print(f"  VQ Loss: {vq_loss.item():.4f}")
        print()

    # Summary
    print("Summary:")
    print("Size  | MSE    | Perplexity | Usage%")
    print("------|--------|------------|-------")
    for size in codebook_sizes:
        r = results[size]
        print(f"{size:4d}  | {r['mse']:.4f} | {r['perplexity']:6.1f}     | {r['perplexity_ratio']:5.1%}")

    return results


def visualize_quantization_results(data, quantized_data, vq_layer, assignments):
    """
    Create visualizations of the quantization process
    """
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Vector Quantization Demonstration', fontsize=16)

    # Original data
    ax1 = axes[0, 0]
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
    ax1.set_title('Original Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)

    # Quantized data with color coding by assignment
    ax2 = axes[0, 1]
    scatter = ax2.scatter(quantized_data[:, 0], quantized_data[:, 1],
                         c=assignments.numpy(), cmap='tab20', alpha=0.7, s=20)
    ax2.set_title('Quantized Data (Colored by Code Assignment)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Codebook Index')

    # Codebook visualization
    ax3 = axes[1, 0]
    codebook = vq_layer.embedding.weight.data
    ax3.scatter(codebook[:, 0], codebook[:, 1], c='red', s=100, marker='x', linewidth=3)
    ax3.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, c='lightblue')
    ax3.set_title('Learned Codebook (Red X) vs Original Data')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.grid(True, alpha=0.3)
    ax3.legend(['Codebook Vectors', 'Original Data'])

    # Reconstruction error visualization
    ax4 = axes[1, 1]
    errors = torch.sum((data - quantized_data)**2, dim=1).numpy()
    scatter = ax4.scatter(data[:, 0], data[:, 1], c=errors, cmap='Reds', alpha=0.7, s=20)
    ax4.set_title('Reconstruction Error per Point')
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    ax4.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Squared Error')

    plt.tight_layout()

    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'basic_vq_demo.png'), dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {os.path.join(output_dir, 'basic_vq_demo.png')}")

    # Show the plot
    plt.show()


def demonstrate_training_dynamics():
    """
    Show how the codebook evolves during training
    """
    print("\n" + "=" * 60)
    print("TRAINING DYNAMICS DEMONSTRATION")
    print("=" * 60)

    # Generate training data
    data = generate_sample_data(n_samples=200, n_features=2, noise_std=0.2)
    data_reshaped = data.unsqueeze(0).unsqueeze(0)

    # Create VQ layer
    vq_layer = VectorQuantizer(
        num_embeddings=8,
        embedding_dim=2,
        commitment_cost=0.25,
        decay=0.99  # EMA decay for codebook updates
    )

    # Training loop
    num_epochs = 100
    losses = []
    perplexities = []
    codebook_snapshots = []

    print(f"Training for {num_epochs} epochs...")

    # Set to training mode
    vq_layer.train()

    for epoch in range(num_epochs):
        # Forward pass
        quantized, vq_loss, perplexity = vq_layer(data_reshaped)

        # Store metrics
        losses.append(vq_loss.item())
        perplexities.append(perplexity.item())

        # Store codebook snapshots at certain epochs
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            codebook_snapshots.append((epoch, vq_layer.embedding.weight.data.clone()))

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {vq_loss.item():.4f}, Perplexity = {perplexity.item():.2f}")

    print("Training completed!")

    # Visualize training dynamics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Vector Quantization Training Dynamics', fontsize=16)

    # Loss and perplexity curves
    ax1 = axes[0, 0]
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('VQ Loss')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(perplexities)
    ax2.set_title('Codebook Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.grid(True, alpha=0.3)

    # Codebook evolution
    for i, (epoch, codebook) in enumerate(codebook_snapshots[:4]):  # Show first 4 snapshots
        row = i // 2
        col = (i % 2) + 1 if row == 0 else i % 2
        if row == 1 and col >= 2:
            break

        ax = axes[row, col] if row == 0 else axes[1, col]
        ax.scatter(data[:, 0], data[:, 1], alpha=0.3, s=10, c='lightblue')
        ax.scatter(codebook[:, 0], codebook[:, 1], c='red', s=100, marker='x', linewidth=3)
        ax.set_title(f'Codebook at Epoch {epoch}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save training dynamics plot
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'vq_training_dynamics.png'), dpi=300, bbox_inches='tight')
    print(f"Training dynamics plot saved to: {os.path.join(output_dir, 'vq_training_dynamics.png')}")

    plt.show()

    return losses, perplexities, codebook_snapshots


def main():
    """
    Run all demonstrations
    """
    print("🎓 VECTOR QUANTIZATION EDUCATIONAL DEMO")
    print("This demo will teach you the fundamentals of Vector Quantization")
    print("and show you how it works with interactive examples.\n")

    try:
        # Basic demonstration
        data, quantized_data, vq_layer, assignments = demonstrate_basic_vq()

        # Codebook size comparison
        results = compare_codebook_sizes()

        # Visualizations
        visualize_quantization_results(data, quantized_data, vq_layer, assignments)

        # Training dynamics
        demonstrate_training_dynamics()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. Vector Quantization maps continuous vectors to discrete codes")
        print("2. Larger codebooks give better reconstruction but use more memory")
        print("3. The codebook learns to represent the data distribution")
        print("4. Perplexity measures how well the codebook is utilized")
        print("5. Training uses exponential moving averages for stable learning")

        print(f"\nOutput files saved to: {os.path.join(os.path.dirname(__file__), 'outputs')}")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Make sure you have matplotlib installed: pip install matplotlib")
        raise


if __name__ == "__main__":
    main()