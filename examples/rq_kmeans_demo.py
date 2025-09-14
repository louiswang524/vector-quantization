"""
RQ-K-means Educational Demonstration

This example demonstrates Residual Quantized K-means (RQ-K-means) clustering
and compares it with standard K-means. It shows:
1. How RQ-K-means works on various datasets
2. Comparison with standard K-means clustering
3. Analysis of residual quantization stages
4. Visualization of quantization hierarchy
5. Compression and approximation quality analysis

Educational Goals:
- Understand residual quantization principles in clustering
- See how multi-stage quantization improves approximation
- Learn about the trade-offs in RQ-K-means
- Visualize the iterative residual reduction process
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
import time

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_quantization import RQKMeans


def generate_test_datasets():
    """
    Generate various test datasets to demonstrate RQ-K-means
    on different types of data distributions.

    Returns:
        Dictionary of datasets with names and data arrays
    """
    print("Generating test datasets...")

    datasets = {}

    # 1. Simple Gaussian blobs
    X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8,
                           center_box=(-10, 10), random_state=42)
    datasets['Gaussian Blobs'] = X_blobs

    # 2. Circles (non-linear structure)
    X_circles, _ = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)
    datasets['Circles'] = X_circles

    # 3. Moons (curved structure)
    X_moons, _ = make_moons(n_samples=300, noise=0.15, random_state=42)
    datasets['Moons'] = X_moons

    # 4. Mixed distribution (combines different patterns)
    np.random.seed(42)
    cluster1 = np.random.multivariate_normal([-3, -3], [[1, 0.5], [0.5, 1]], 100)
    cluster2 = np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], 100)
    cluster3 = np.random.multivariate_normal([0, 4], [[2, 0], [0, 0.5]], 100)
    X_mixed = np.vstack([cluster1, cluster2, cluster3])
    datasets['Mixed Distribution'] = X_mixed

    # 5. High-dimensional data (project to 2D for visualization)
    X_high_dim = np.random.randn(400, 10)  # 10-dimensional
    # Add structure by creating clusters in high-dimensional space
    centers_high_dim = np.random.randn(5, 10) * 3
    labels = np.random.choice(5, 400)
    for i in range(400):
        X_high_dim[i] += centers_high_dim[labels[i]]

    datasets['High Dimensional'] = X_high_dim

    # 6. Grid pattern
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    xx, yy = np.meshgrid(x, y)
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    # Add some noise
    X_grid += np.random.normal(0, 0.2, X_grid.shape)
    datasets['Grid Pattern'] = X_grid

    print(f"Generated {len(datasets)} test datasets:")
    for name, data in datasets.items():
        print(f"  - {name}: {data.shape[0]} samples, {data.shape[1]} dimensions")

    return datasets


def compare_kmeans_vs_rqkmeans(X, dataset_name, n_clusters=16, n_stages=4):
    """
    Compare standard K-means with RQ-K-means on a given dataset

    Args:
        X: Input data array
        dataset_name: Name of the dataset for display
        n_clusters: Number of clusters per stage
        n_stages: Number of RQ stages

    Returns:
        Dictionary with comparison results
    """
    print(f"\n🔍 Analyzing dataset: {dataset_name}")
    print(f"Data shape: {X.shape}")
    print(f"K-means clusters: {n_clusters}, RQ-K-means stages: {n_stages}")

    # Standard K-means
    print("  Running standard K-means...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    X_kmeans_reconstructed = kmeans.cluster_centers_[kmeans_labels]
    kmeans_time = time.time() - start_time
    kmeans_mse = np.mean((X - X_kmeans_reconstructed)**2)

    # RQ-K-means
    print("  Running RQ-K-means...")
    start_time = time.time()
    rq_kmeans = RQKMeans(
        n_clusters=n_clusters,
        n_stages=n_stages,
        random_state=42,
        verbose=False
    )
    X_rq_reconstructed = rq_kmeans.fit_transform(X)
    rq_time = time.time() - start_time
    rq_mse = rq_kmeans.calculate_reconstruction_error(X)

    # Get compression ratios
    original_bits = X.shape[1] * 32  # 32-bit floats
    kmeans_bits = np.log2(n_clusters)
    rq_bits = np.log2(n_clusters) * n_stages

    kmeans_ratio = original_bits / kmeans_bits
    rq_ratio = original_bits / rq_bits

    print(f"  Standard K-means: MSE = {kmeans_mse:.6f}, Time = {kmeans_time:.3f}s")
    print(f"  RQ-K-means:       MSE = {rq_mse:.6f}, Time = {rq_time:.3f}s")
    print(f"  Improvement:      {(kmeans_mse - rq_mse)/kmeans_mse*100:.1f}% lower MSE")
    print(f"  Compression:      K-means {kmeans_ratio:.1f}:1, RQ-K-means {rq_ratio:.1f}:1")

    return {
        'dataset_name': dataset_name,
        'X': X,
        'kmeans_mse': kmeans_mse,
        'rq_mse': rq_mse,
        'kmeans_time': kmeans_time,
        'rq_time': rq_time,
        'X_kmeans_reconstructed': X_kmeans_reconstructed,
        'X_rq_reconstructed': X_rq_reconstructed,
        'kmeans_labels': kmeans_labels,
        'rq_kmeans': rq_kmeans,
        'kmeans_ratio': kmeans_ratio,
        'rq_ratio': rq_ratio
    }


def visualize_2d_comparison(results, max_datasets=4):
    """
    Visualize comparison results for 2D datasets
    """
    print("\n🎨 Creating 2D visualization comparison...")

    # Filter for 2D datasets only
    results_2d = [r for r in results if r['X'].shape[1] == 2][:max_datasets]

    if not results_2d:
        print("No 2D datasets available for visualization")
        return

    n_datasets = len(results_2d)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(12, n_datasets * 3))
    fig.suptitle('K-means vs RQ-K-means Comparison (2D Data)', fontsize=16)

    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results_2d):
        X = result['X']
        X_kmeans = result['X_kmeans_reconstructed']
        X_rq = result['X_rq_reconstructed']

        # Original data
        axes[i, 0].scatter(X[:, 0], X[:, 1], alpha=0.6, s=20, c='blue')
        axes[i, 0].set_title(f"{result['dataset_name']}\nOriginal Data")
        axes[i, 0].grid(True, alpha=0.3)

        # K-means reconstruction
        axes[i, 1].scatter(X_kmeans[:, 0], X_kmeans[:, 1], alpha=0.6, s=20,
                          c=result['kmeans_labels'], cmap='tab20')
        axes[i, 1].set_title(f"K-means Reconstruction\nMSE: {result['kmeans_mse']:.4f}")
        axes[i, 1].grid(True, alpha=0.3)

        # RQ-K-means reconstruction
        # Color by first stage assignment for visualization
        first_stage_codes = result['rq_kmeans'].get_codes(X)[0]
        axes[i, 2].scatter(X_rq[:, 0], X_rq[:, 1], alpha=0.6, s=20,
                          c=first_stage_codes, cmap='tab20')
        axes[i, 2].set_title(f"RQ-K-means Reconstruction\nMSE: {result['rq_mse']:.4f}")
        axes[i, 2].grid(True, alpha=0.3)

        # Set consistent axis limits
        all_data = np.vstack([X, X_kmeans, X_rq])
        x_margin = (all_data[:, 0].max() - all_data[:, 0].min()) * 0.1
        y_margin = (all_data[:, 1].max() - all_data[:, 1].min()) * 0.1

        for ax in axes[i, :]:
            ax.set_xlim(all_data[:, 0].min() - x_margin, all_data[:, 0].max() + x_margin)
            ax.set_ylim(all_data[:, 1].min() - y_margin, all_data[:, 1].max() + y_margin)

    plt.tight_layout()

    # Save visualization
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rq_kmeans_2d_comparison.png'),
               dpi=300, bbox_inches='tight')
    print(f"2D comparison saved to outputs/")

    plt.show()


def analyze_rqkmeans_stages(rq_kmeans, X, sample_points=5):
    """
    Analyze how RQ-K-means progressively reduces error through stages
    """
    print(f"\n🔬 Analyzing RQ-K-means stage progression...")

    # Analyze overall quality progression
    quality_analysis = rq_kmeans.analyze_quantization_quality(X)

    print("Stage-by-stage error reduction:")
    for i, error in enumerate(quality_analysis['cumulative_errors']):
        print(f"  After stage {i+1}: MSE = {error:.6f}")

    print(f"Final reconstruction error: {quality_analysis['final_reconstruction_error']:.6f}")
    print(f"Compression ratio: {quality_analysis['compression_ratio']:.1f}:1")
    print(f"Total clusters used: {quality_analysis['total_clusters_used']}")

    # Analyze specific data points
    print(f"\nHierarchical breakdown for {sample_points} sample points:")

    sample_indices = np.random.choice(len(X), sample_points, replace=False)

    for idx in sample_indices[:3]:  # Show first 3 for brevity
        point = X[idx]
        hierarchy = rq_kmeans.get_quantization_hierarchy(point)

        print(f"\n  Point {idx}: {point}")
        print(f"  Original: {hierarchy['original']}")

        for stage_info in hierarchy['stages']:
            stage = stage_info['stage']
            cluster = stage_info['cluster_assignment']
            quantized = stage_info['quantized_vector']
            print(f"    Stage {stage+1}: Cluster {cluster}, Quantized: {quantized}")

        final_recon = hierarchy['cumulative_reconstruction'][-1]
        error = np.sum((point - final_recon)**2)
        print(f"  Final reconstruction: {final_recon}")
        print(f"  Reconstruction error: {error:.6f}")

    # Visualize stage progression
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RQ-K-means Stage Analysis', fontsize=16)

    # Error reduction curve
    ax1 = axes[0, 0]
    stages = range(1, len(quality_analysis['cumulative_errors']) + 1)
    ax1.plot(stages, quality_analysis['cumulative_errors'], 'bo-')
    ax1.set_title('Cumulative Reconstruction Error')
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)

    # Stage contributions
    ax2 = axes[0, 1]
    ax2.bar(stages, quality_analysis['stage_contributions'])
    ax2.set_title('Stage Contributions (L2 Norm)')
    ax2.set_xlabel('Stage')
    ax2.set_ylabel('Contribution')
    ax2.grid(True, alpha=0.3)

    # Cluster utilization
    ax3 = axes[1, 0]
    ax3.bar(stages, quality_analysis['cluster_utilization'])
    ax3.set_title('Cluster Utilization per Stage')
    ax3.set_xlabel('Stage')
    ax3.set_ylabel('Utilization Ratio')
    ax3.grid(True, alpha=0.3)

    # Stage inertias
    ax4 = axes[1, 1]
    ax4.bar(stages, quality_analysis['stage_inertias'])
    ax4.set_title('K-means Inertia per Stage')
    ax4.set_xlabel('Stage')
    ax4.set_ylabel('Inertia')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save stage analysis
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'rq_kmeans_stage_analysis.png'),
               dpi=300, bbox_inches='tight')
    print(f"Stage analysis saved to outputs/")

    plt.show()

    return quality_analysis


def demonstrate_parameter_effects():
    """
    Demonstrate the effect of different RQ-K-means parameters
    """
    print("\n⚙️ Analyzing parameter effects...")

    # Generate test data
    X_test, _ = make_blobs(n_samples=400, centers=6, cluster_std=1.2, random_state=42)

    # Test different numbers of stages
    stages_to_test = [1, 2, 3, 4, 6, 8]
    stage_results = []

    print("Testing different numbers of stages:")
    for n_stages in stages_to_test:
        rq_kmeans = RQKMeans(n_clusters=16, n_stages=n_stages, random_state=42, verbose=False)
        mse = rq_kmeans.fit(X_test).calculate_reconstruction_error(X_test)
        compression_ratio = rq_kmeans.get_compression_ratio(X_test.shape[1])

        stage_results.append((n_stages, mse, compression_ratio))
        print(f"  {n_stages} stages: MSE = {mse:.6f}, Compression = {compression_ratio:.1f}:1")

    # Test different numbers of clusters per stage
    clusters_to_test = [8, 16, 32, 64, 128]
    cluster_results = []

    print("\nTesting different numbers of clusters per stage:")
    for n_clusters in clusters_to_test:
        rq_kmeans = RQKMeans(n_clusters=n_clusters, n_stages=4, random_state=42, verbose=False)
        mse = rq_kmeans.fit(X_test).calculate_reconstruction_error(X_test)
        compression_ratio = rq_kmeans.get_compression_ratio(X_test.shape[1])

        cluster_results.append((n_clusters, mse, compression_ratio))
        print(f"  {n_clusters} clusters: MSE = {mse:.6f}, Compression = {compression_ratio:.1f}:1")

    # Visualize parameter effects
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RQ-K-means Parameter Effects', fontsize=16)

    # Stages effect
    stages, stage_mse, stage_compression = zip(*stage_results)
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(stages, stage_mse, 'bo-', label='MSE', color='blue')
    line2 = ax1_twin.plot(stages, stage_compression, 'rs-', label='Compression Ratio', color='red')

    ax1.set_xlabel('Number of Stages')
    ax1.set_ylabel('Reconstruction MSE', color='blue')
    ax1_twin.set_ylabel('Compression Ratio', color='red')
    ax1.set_title('Effect of Number of Stages')
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Clusters effect
    clusters, cluster_mse, cluster_compression = zip(*cluster_results)
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    line1 = ax2.semilogx(clusters, cluster_mse, 'bo-', label='MSE', color='blue')
    line2 = ax2_twin.semilogx(clusters, cluster_compression, 'rs-', label='Compression Ratio', color='red')

    ax2.set_xlabel('Number of Clusters per Stage')
    ax2.set_ylabel('Reconstruction MSE', color='blue')
    ax2_twin.set_ylabel('Compression Ratio', color='red')
    ax2.set_title('Effect of Clusters per Stage')
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    # Save parameter effects
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'rq_kmeans_parameter_effects.png'),
               dpi=300, bbox_inches='tight')
    print(f"Parameter effects analysis saved to outputs/")

    plt.show()

    return stage_results, cluster_results


def create_summary_report(all_results):
    """
    Create a comprehensive summary report of all experiments
    """
    print("\n📋 Creating comprehensive summary report...")

    # Calculate overall statistics
    kmeans_mse_values = [r['kmeans_mse'] for r in all_results]
    rq_mse_values = [r['rq_mse'] for r in all_results]
    improvements = [(km - rq)/km * 100 for km, rq in zip(kmeans_mse_values, rq_mse_values)]

    avg_kmeans_mse = np.mean(kmeans_mse_values)
    avg_rq_mse = np.mean(rq_mse_values)
    avg_improvement = np.mean(improvements)

    print(f"\n📊 OVERALL RESULTS SUMMARY:")
    print(f"Average K-means MSE:       {avg_kmeans_mse:.6f}")
    print(f"Average RQ-K-means MSE:    {avg_rq_mse:.6f}")
    print(f"Average improvement:       {avg_improvement:.1f}%")
    print(f"Best improvement:          {max(improvements):.1f}% ({all_results[np.argmax(improvements)]['dataset_name']})")
    print(f"Worst improvement:         {min(improvements):.1f}% ({all_results[np.argmin(improvements)]['dataset_name']})")

    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('RQ-K-means vs K-means: Comprehensive Summary', fontsize=16)

    # MSE comparison
    ax1 = axes[0, 0]
    dataset_names = [r['dataset_name'] for r in all_results]
    x_pos = np.arange(len(dataset_names))

    bar_width = 0.35
    ax1.bar(x_pos - bar_width/2, kmeans_mse_values, bar_width, label='K-means', alpha=0.8)
    ax1.bar(x_pos + bar_width/2, rq_mse_values, bar_width, label='RQ-K-means', alpha=0.8)

    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('MSE')
    ax1.set_title('Reconstruction Error Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvement percentage
    ax2 = axes[0, 1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(x_pos, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('RQ-K-means Improvement over K-means')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Compression ratio comparison
    ax3 = axes[1, 0]
    kmeans_ratios = [r['kmeans_ratio'] for r in all_results]
    rq_ratios = [r['rq_ratio'] for r in all_results]

    ax3.bar(x_pos - bar_width/2, kmeans_ratios, bar_width, label='K-means', alpha=0.8)
    ax3.bar(x_pos + bar_width/2, rq_ratios, bar_width, label='RQ-K-means', alpha=0.8)

    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Compression Ratio')
    ax3.set_title('Compression Ratio Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Quality vs Compression trade-off
    ax4 = axes[1, 1]
    scatter = ax4.scatter(rq_ratios, rq_mse_values, c=improvements, cmap='RdYlGn', s=100, alpha=0.7)
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('RQ-K-means MSE')
    ax4.set_title('Quality vs Compression Trade-off')
    ax4.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Improvement (%)')

    # Annotate points
    for i, (ratio, mse, name) in enumerate(zip(rq_ratios, rq_mse_values, dataset_names)):
        ax4.annotate(name, (ratio, mse), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()

    # Save summary report
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    plt.savefig(os.path.join(output_dir, 'rq_kmeans_summary_report.png'),
               dpi=300, bbox_inches='tight')
    print(f"Summary report saved to outputs/")

    plt.show()

    return {
        'avg_kmeans_mse': avg_kmeans_mse,
        'avg_rq_mse': avg_rq_mse,
        'avg_improvement': avg_improvement,
        'improvements': improvements
    }


def main():
    """
    Run the complete RQ-K-means educational demonstration
    """
    print("🎓 RQ-K-MEANS EDUCATIONAL DEMONSTRATION")
    print("This demo teaches you about Residual Quantized K-means")
    print("and shows its advantages over standard K-means clustering.\n")

    try:
        # Configuration
        n_clusters = 16
        n_stages = 4

        print("Configuration:")
        print(f"  Clusters per stage: {n_clusters}")
        print(f"  Number of stages: {n_stages}")

        # Generate test datasets
        datasets = generate_test_datasets()

        # Run comparisons on all datasets
        all_results = []
        for name, X in datasets.items():
            result = compare_kmeans_vs_rqkmeans(X, name, n_clusters, n_stages)
            all_results.append(result)

        # Visualizations and analysis
        visualize_2d_comparison(all_results)

        # Detailed analysis on first 2D dataset
        first_2d_result = next((r for r in all_results if r['X'].shape[1] == 2), None)
        if first_2d_result:
            analyze_rqkmeans_stages(first_2d_result['rq_kmeans'], first_2d_result['X'])

        # Parameter effects analysis
        demonstrate_parameter_effects()

        # Comprehensive summary
        summary = create_summary_report(all_results)

        print("\n" + "=" * 60)
        print("RQ-K-MEANS DEMONSTRATION COMPLETED! 🎉")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. RQ-K-means improves approximation quality over standard K-means")
        print("2. Multi-stage quantization captures finer details progressively")
        print("3. Each stage reduces residual error from previous stages")
        print("4. Trade-off between compression ratio and reconstruction quality")
        print("5. Works well on various data distributions and dimensionalities")
        print(f"6. Average improvement: {summary['avg_improvement']:.1f}% lower MSE")

        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        print(f"\n📁 All output files saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()