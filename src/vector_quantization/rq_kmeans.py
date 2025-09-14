"""
Residual Quantized K-means (RQ-K-means) Implementation

RQ-K-means extends traditional K-means clustering by applying residual quantization
principles to vector quantization. Instead of finding a single centroid for each
data point, RQ-K-means iteratively finds multiple centroids that together
approximate the original data point.

Key Concepts:
1. Multi-stage clustering: Apply K-means iteratively on residuals
2. Residual reduction: Each stage reduces the approximation error
3. Additive quantization: Final representation is sum of all centroids
4. Improved approximation: Better reconstruction than single-stage K-means

Mathematical Framework:
Given data points X = {x₁, x₂, ..., xₙ}:
Stage 1: r₁ᵢ = xᵢ, find centroids C₁, assign each r₁ᵢ to nearest c₁ⱼ
Stage 2: r₂ᵢ = r₁ᵢ - c₁ⱼ, find centroids C₂, assign each r₂ᵢ to nearest c₂ₖ
...
Stage m: rₘᵢ = rₘ₋₁ᵢ - cₘ₋₁ₗ, find centroids Cₘ

Final approximation: x̂ᵢ = c₁ⱼ + c₂ₖ + ... + cₘₚ

Applications:
- Vector compression with controllable quality/size trade-off
- Feature quantization for machine learning models
- Image compression and representation learning
- Approximate nearest neighbor search acceleration

Paper References:
- "Residual Vector Quantization" literature
- "Product Quantization for Nearest Neighbor Search" (Jégou et al.)
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Any, Union
from sklearn.cluster import KMeans
import warnings


class RQKMeans:
    """
    Residual Quantized K-means Clustering

    This class implements multi-stage K-means clustering where each stage
    quantizes the residual error from previous stages. This approach
    provides better approximation quality compared to standard K-means.

    The algorithm works as follows:
    1. Apply K-means to original data, get first set of centroids
    2. Compute residuals (data - assigned centroids)
    3. Apply K-means to residuals, get second set of centroids
    4. Repeat for specified number of stages
    5. Final approximation is sum of centroids from all stages

    This creates a hierarchical quantization where:
    - First stage captures major patterns/clusters
    - Subsequent stages capture finer details and corrections
    - Each stage reduces the quantization error

    Args:
        n_clusters (int): Number of clusters (centroids) per stage
        n_stages (int): Number of residual quantization stages
        max_iter (int): Maximum iterations for each K-means stage
        tol (float): Tolerance for K-means convergence
        init_method (str): Initialization method ('k-means++', 'random')
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print progress information
    """

    def __init__(
        self,
        n_clusters: int = 256,
        n_stages: int = 4,
        max_iter: int = 300,
        tol: float = 1e-4,
        init_method: str = 'k-means++',
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.n_clusters = n_clusters
        self.n_stages = n_stages
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        self.verbose = verbose

        # Storage for learned centroids from each stage
        # Each element is a (n_clusters, feature_dim) array
        self.centroids_list: List[np.ndarray] = []

        # Storage for K-means models from each stage (for analysis)
        self.kmeans_models: List[KMeans] = []

        # Flag to track if model has been fitted
        self.is_fitted = False

        # Statistics for monitoring training progress
        self.stage_errors: List[float] = []
        self.stage_inertias: List[float] = []

    def fit(self, X: Union[np.ndarray, Tensor]) -> 'RQKMeans':
        """
        Fit RQ-K-means model to data

        This method iteratively applies K-means to the data and its residuals,
        building up a hierarchical quantization scheme.

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            self: Fitted RQKMeans instance

        Process:
        1. For each stage i:
           a. Apply K-means to current residuals
           b. Store centroids from this stage
           c. Compute new residuals for next stage
           d. Track quantization error improvement
        """
        # Convert to numpy if tensor
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X = np.asarray(X)
        n_samples, n_features = X.shape

        if self.verbose:
            print(f"Fitting RQ-K-means with {self.n_stages} stages, {self.n_clusters} clusters per stage")
            print(f"Input data shape: {X.shape}")

        # Initialize storage
        self.centroids_list = []
        self.kmeans_models = []
        self.stage_errors = []
        self.stage_inertias = []

        # Start with original data as the first residual
        current_residual = X.copy()

        # Apply K-means iteratively to residuals
        for stage in range(self.n_stages):
            if self.verbose:
                print(f"\nStage {stage + 1}/{self.n_stages}")
                print(f"Residual data range: [{current_residual.min():.4f}, {current_residual.max():.4f}]")
                print(f"Residual RMS: {np.sqrt(np.mean(current_residual**2)):.4f}")

            # Apply K-means to current residuals
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init=self.init_method,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_init=10  # Multiple initializations for robustness
            )

            # Fit K-means to current residuals
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress sklearn warnings
                cluster_assignments = kmeans.fit_predict(current_residual)

            # Store centroids and model
            self.centroids_list.append(kmeans.cluster_centers_.copy())
            self.kmeans_models.append(kmeans)

            # Calculate stage statistics
            stage_inertia = kmeans.inertia_
            self.stage_inertias.append(stage_inertia)

            # Get quantized vectors for this stage
            stage_quantized = kmeans.cluster_centers_[cluster_assignments]

            # Calculate reconstruction error for this stage
            stage_error = np.mean(np.sum((current_residual - stage_quantized)**2, axis=1))
            self.stage_errors.append(stage_error)

            if self.verbose:
                print(f"Stage inertia (within-cluster sum of squares): {stage_inertia:.4f}")
                print(f"Stage reconstruction error: {stage_error:.4f}")
                print(f"Used clusters: {len(np.unique(cluster_assignments))}/{self.n_clusters}")

            # Update residuals for next stage
            # The residual becomes what's left after this quantization step
            current_residual = current_residual - stage_quantized

        self.is_fitted = True

        if self.verbose:
            total_error = self.calculate_reconstruction_error(X)
            compression_ratio = self.get_compression_ratio(n_features)
            print(f"\nTraining completed!")
            print(f"Total reconstruction error: {total_error:.6f}")
            print(f"Compression ratio: {compression_ratio:.2f}:1")

        return self

    def predict(self, X: Union[np.ndarray, Tensor]) -> List[np.ndarray]:
        """
        Predict cluster assignments for each stage

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            List of cluster assignments for each stage
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X = np.asarray(X)

        assignments_list = []
        current_residual = X.copy()

        # For each stage, predict assignments and update residuals
        for stage in range(self.n_stages):
            # Predict cluster assignments for current residuals
            assignments = self.kmeans_models[stage].predict(current_residual)
            assignments_list.append(assignments)

            # Get quantized vectors and update residuals
            stage_quantized = self.centroids_list[stage][assignments]
            current_residual = current_residual - stage_quantized

        return assignments_list

    def transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Transform data to quantized representation

        Args:
            X: Input data of shape (n_samples, n_features)

        Returns:
            Quantized data with same shape as input
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X = np.asarray(X)
        n_samples = X.shape[0]

        # Initialize reconstruction
        reconstructed = np.zeros_like(X)
        current_residual = X.copy()

        # Add contribution from each stage
        for stage in range(self.n_stages):
            # Get cluster assignments for current residuals
            assignments = self.kmeans_models[stage].predict(current_residual)

            # Get quantized vectors for this stage
            stage_quantized = self.centroids_list[stage][assignments]

            # Add to final reconstruction
            reconstructed += stage_quantized

            # Update residuals for next stage
            current_residual = current_residual - stage_quantized

        return reconstructed

    def fit_transform(self, X: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Fit model and transform data in one step

        Args:
            X: Input data

        Returns:
            Quantized data
        """
        return self.fit(X).transform(X)

    def get_codes(self, X: Union[np.ndarray, Tensor]) -> List[np.ndarray]:
        """
        Get discrete codes for each stage

        Args:
            X: Input data

        Returns:
            List of discrete codes for each stage
        """
        return self.predict(X)

    def reconstruct_from_codes(self, codes_list: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct data from discrete codes

        Args:
            codes_list: List of cluster assignments for each stage

        Returns:
            Reconstructed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before reconstruction")

        # Get shape from first code array
        n_samples = len(codes_list[0])
        n_features = self.centroids_list[0].shape[1]

        # Initialize reconstruction
        reconstructed = np.zeros((n_samples, n_features))

        # Add contribution from each stage
        for stage, codes in enumerate(codes_list):
            stage_quantized = self.centroids_list[stage][codes]
            reconstructed += stage_quantized

        return reconstructed

    def calculate_reconstruction_error(self, X: Union[np.ndarray, Tensor]) -> float:
        """
        Calculate mean squared reconstruction error

        Args:
            X: Original data

        Returns:
            Mean squared error between original and reconstructed data
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X_reconstructed = self.transform(X)
        mse = np.mean((X - X_reconstructed)**2)
        return mse

    def get_compression_ratio(self, n_features: int) -> float:
        """
        Calculate compression ratio

        Args:
            n_features: Number of features in original data

        Returns:
            Compression ratio (original size / compressed size)
        """
        # Original: each feature stored as float (32 bits)
        original_bits = n_features * 32

        # Compressed: each stage uses log2(n_clusters) bits
        import math
        bits_per_stage = math.log2(self.n_clusters)
        compressed_bits = self.n_stages * bits_per_stage

        return original_bits / compressed_bits

    def analyze_quantization_quality(self, X: Union[np.ndarray, Tensor]) -> Dict[str, Any]:
        """
        Comprehensive analysis of quantization quality

        Args:
            X: Original data

        Returns:
            Dictionary with detailed quality metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        X = np.asarray(X)

        # Analyze reconstruction quality at each stage
        cumulative_error = []
        stage_contributions = []

        current_residual = X.copy()
        cumulative_reconstruction = np.zeros_like(X)

        for stage in range(self.n_stages):
            # Get assignments and quantized vectors for this stage
            assignments = self.kmeans_models[stage].predict(current_residual)
            stage_quantized = self.centroids_list[stage][assignments]

            # Update cumulative reconstruction
            cumulative_reconstruction += stage_quantized

            # Calculate cumulative error
            error = np.mean((X - cumulative_reconstruction)**2)
            cumulative_error.append(error)

            # Calculate contribution of this stage
            stage_contribution = np.mean(np.sum(stage_quantized**2, axis=1))
            stage_contributions.append(stage_contribution)

            # Update residual
            current_residual = current_residual - stage_quantized

        # Analyze cluster utilization
        cluster_utilization = []
        for stage in range(self.n_stages):
            assignments = self.kmeans_models[stage].predict(X if stage == 0 else current_residual)
            unique_clusters = len(np.unique(assignments))
            utilization = unique_clusters / self.n_clusters
            cluster_utilization.append(utilization)

        return {
            'cumulative_errors': cumulative_error,
            'stage_contributions': stage_contributions,
            'stage_inertias': self.stage_inertias,
            'cluster_utilization': cluster_utilization,
            'final_reconstruction_error': cumulative_error[-1],
            'compression_ratio': self.get_compression_ratio(X.shape[1]),
            'total_clusters_used': sum(int(u * self.n_clusters) for u in cluster_utilization)
        }

    def visualize_centroids(self, stage: int = 0) -> np.ndarray:
        """
        Get centroids for visualization

        Args:
            stage: Which quantization stage to visualize

        Returns:
            Centroid vectors for the specified stage
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization")

        if stage >= self.n_stages:
            raise ValueError(f"Stage {stage} not available. Model has {self.n_stages} stages.")

        return self.centroids_list[stage].copy()

    def get_quantization_hierarchy(self, x: Union[np.ndarray, Tensor]) -> Dict[str, Any]:
        """
        Analyze the quantization hierarchy for a single data point

        Args:
            x: Single data point of shape (n_features,)

        Returns:
            Dictionary with hierarchical breakdown
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analysis")

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x).reshape(1, -1)  # Ensure 2D shape

        hierarchy = {
            'original': x[0].copy(),
            'stages': [],
            'residuals': [],
            'cumulative_reconstruction': []
        }

        current_residual = x.copy()
        cumulative_reconstruction = np.zeros_like(x)

        for stage in range(self.n_stages):
            # Get cluster assignment
            assignment = self.kmeans_models[stage].predict(current_residual)[0]

            # Get quantized vector
            quantized = self.centroids_list[stage][assignment]

            # Update cumulative reconstruction
            cumulative_reconstruction += quantized.reshape(1, -1)

            # Store stage information
            hierarchy['stages'].append({
                'stage': stage,
                'cluster_assignment': assignment,
                'quantized_vector': quantized.copy(),
                'residual_before': current_residual[0].copy(),
            })

            hierarchy['cumulative_reconstruction'].append(cumulative_reconstruction[0].copy())

            # Update residual
            current_residual = current_residual - quantized.reshape(1, -1)
            hierarchy['residuals'].append(current_residual[0].copy())

        return hierarchy

    def save_model(self, filepath: str):
        """Save fitted model to file"""
        import pickle

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        model_data = {
            'centroids_list': self.centroids_list,
            'kmeans_models': self.kmeans_models,
            'n_clusters': self.n_clusters,
            'n_stages': self.n_stages,
            'stage_errors': self.stage_errors,
            'stage_inertias': self.stage_inertias,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load fitted model from file"""
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.centroids_list = model_data['centroids_list']
        self.kmeans_models = model_data['kmeans_models']
        self.n_clusters = model_data['n_clusters']
        self.n_stages = model_data['n_stages']
        self.stage_errors = model_data['stage_errors']
        self.stage_inertias = model_data['stage_inertias']
        self.is_fitted = model_data['is_fitted']