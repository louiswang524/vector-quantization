"""
Unified Vector Quantization Pipeline

This module provides a configurable pipeline for generating semantic IDs and codebooks
from different modalities (image, text, video) using various encoder architectures
and vector quantization methods.

Pipeline Architecture:
Input Data → Configurable Encoder → Vector Quantizer → Semantic IDs + Codebook

Key Features:
- Modular encoder selection (image/text/video encoders)
- Multiple VQ method options (VQ, RQ-VAE, RQ-K-means)
- Unified configuration system
- Semantic ID generation and codebook management
- Cross-modal compatibility

Educational Goals:
- Demonstrate how VQ works across different modalities
- Show the flexibility of VQ systems in practice
- Provide a framework for research and experimentation
- Illustrate semantic representation learning
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json


class ModalityType(Enum):
    """Supported input modalities"""
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"


class EncoderType(Enum):
    """Available encoder architectures"""
    # Image encoders
    CNN = "cnn"
    RESNET = "resnet"
    VISION_TRANSFORMER = "vit"

    # Text encoders
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    BERT_LIKE = "bert_like"

    # Video encoders
    CNN_3D = "cnn_3d"
    VIDEO_TRANSFORMER = "video_transformer"


class VQMethodType(Enum):
    """Available vector quantization methods"""
    BASIC_VQ = "basic_vq"
    VQ_VAE = "vq_vae"
    RQ_VAE = "rq_vae"
    RQ_KMEANS = "rq_kmeans"


@dataclass
class PipelineConfig:
    """
    Configuration class for the VQ Pipeline

    This class defines all the parameters needed to configure
    the pipeline components and training process.
    """
    # Modality and architecture selection
    modality: ModalityType
    encoder_type: EncoderType
    vq_method: VQMethodType

    # Input specifications
    input_dim: Union[Tuple[int, ...], int]  # Input dimensions (varies by modality)

    # Encoder parameters
    encoder_config: Dict[str, Any]

    # Vector quantization parameters
    embedding_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25

    # VQ method specific parameters
    vq_config: Dict[str, Any] = None

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32

    def __post_init__(self):
        """Initialize default configurations"""
        if self.vq_config is None:
            self.vq_config = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'modality': self.modality.value,
            'encoder_type': self.encoder_type.value,
            'vq_method': self.vq_method.value,
            'input_dim': self.input_dim,
            'encoder_config': self.encoder_config,
            'embedding_dim': self.embedding_dim,
            'num_embeddings': self.num_embeddings,
            'commitment_cost': self.commitment_cost,
            'vq_config': self.vq_config,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        return cls(
            modality=ModalityType(config_dict['modality']),
            encoder_type=EncoderType(config_dict['encoder_type']),
            vq_method=VQMethodType(config_dict['vq_method']),
            input_dim=config_dict['input_dim'],
            encoder_config=config_dict['encoder_config'],
            embedding_dim=config_dict.get('embedding_dim', 256),
            num_embeddings=config_dict.get('num_embeddings', 1024),
            commitment_cost=config_dict.get('commitment_cost', 0.25),
            vq_config=config_dict.get('vq_config', {}),
            learning_rate=config_dict.get('learning_rate', 1e-4),
            batch_size=config_dict.get('batch_size', 32)
        )

    def save(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoders

    This defines the interface that all encoders must implement,
    ensuring consistency across different modalities and architectures.
    """

    def __init__(self, input_dim: Union[Tuple[int, ...], int], embedding_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to embedding space

        Args:
            x: Input tensor (format depends on modality)

        Returns:
            Encoded embeddings of shape (batch, embedding_dim) or
            (batch, seq_len, embedding_dim) for sequence data
        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, ...]:
        """Return the output shape (excluding batch dimension)"""
        pass


class SemanticTokenizer:
    """
    Manages the conversion between embeddings and semantic IDs

    This class handles:
    - Converting embeddings to discrete semantic IDs
    - Maintaining the codebook mapping
    - Providing utilities for analysis and visualization
    """

    def __init__(self, vq_method, num_embeddings: int, embedding_dim: int):
        self.vq_method = vq_method
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def encode_to_ids(self, embeddings: Tensor) -> Tensor:
        """
        Convert embeddings to semantic IDs

        Args:
            embeddings: Input embeddings

        Returns:
            Semantic IDs (discrete tokens)
        """
        if hasattr(self.vq_method, 'encode_to_indices'):
            return self.vq_method.encode_to_indices(embeddings)
        elif hasattr(self.vq_method, 'get_codes'):
            codes = self.vq_method.get_codes(embeddings)
            if isinstance(codes, list):
                # For multi-stage methods like RQ-VAE
                return codes
            return codes
        else:
            # For basic VQ methods
            with torch.no_grad():
                # Assume embeddings need to be reshaped for VQ
                batch_size = embeddings.size(0)
                flat_embeddings = embeddings.view(batch_size, 1, 1, -1)
                _, _, _ = self.vq_method(flat_embeddings)

                # Calculate distances and get indices
                distances = self.vq_method.get_distance_matrix(flat_embeddings)
                return torch.argmin(distances, dim=-1).squeeze()

    def decode_from_ids(self, semantic_ids: Union[Tensor, List[Tensor]],
                       output_shape: Optional[Tuple[int, ...]] = None) -> Tensor:
        """
        Convert semantic IDs back to embeddings

        Args:
            semantic_ids: Discrete semantic IDs
            output_shape: Target output shape (if needed)

        Returns:
            Reconstructed embeddings
        """
        if hasattr(self.vq_method, 'decode_from_indices'):
            return self.vq_method.decode_from_indices(semantic_ids)
        elif hasattr(self.vq_method, 'reconstruct_from_codes'):
            if isinstance(semantic_ids, list):
                return self.vq_method.reconstruct_from_codes(semantic_ids, output_shape)
            else:
                return self.vq_method.reconstruct_from_codes([semantic_ids], output_shape)
        else:
            # For basic VQ methods, use embedding lookup
            if isinstance(semantic_ids, list):
                semantic_ids = semantic_ids[0]  # Take first level

            codebook_entries = self.vq_method.get_codebook_entry(semantic_ids.flatten())
            if output_shape:
                return codebook_entries.view(semantic_ids.size(0), *output_shape)
            return codebook_entries.view(semantic_ids.size(0), -1)

    def get_codebook(self) -> Tensor:
        """Get the learned codebook"""
        if hasattr(self.vq_method, 'get_codebook'):
            return self.vq_method.get_codebook()
        elif hasattr(self.vq_method, 'embedding'):
            return self.vq_method.embedding.weight.data
        else:
            raise NotImplementedError("Codebook access not implemented for this VQ method")

    def analyze_usage(self, semantic_ids: Union[Tensor, List[Tensor]]) -> Dict[str, Any]:
        """
        Analyze codebook usage statistics

        Args:
            semantic_ids: Collection of semantic IDs from dataset

        Returns:
            Dictionary with usage statistics
        """
        if isinstance(semantic_ids, list) and len(semantic_ids) > 0 and isinstance(semantic_ids[0], Tensor):
            # Multi-stage quantization
            all_ids = []
            for level_ids in semantic_ids:
                all_ids.extend(level_ids.flatten().tolist())
            semantic_ids = torch.tensor(all_ids)
        else:
            semantic_ids = semantic_ids.flatten()

        unique_ids, counts = torch.unique(semantic_ids, return_counts=True)

        return {
            'total_tokens': len(semantic_ids),
            'unique_tokens': len(unique_ids),
            'usage_ratio': len(unique_ids) / self.num_embeddings,
            'most_frequent': unique_ids[torch.argmax(counts)].item(),
            'least_frequent': unique_ids[torch.argmin(counts)].item(),
            'usage_distribution': counts.float() / len(semantic_ids),
            'entropy': -torch.sum(counts.float() / len(semantic_ids) *
                                torch.log(counts.float() / len(semantic_ids) + 1e-10)).item()
        }


class VQPipeline(nn.Module):
    """
    Main Vector Quantization Pipeline

    This class orchestrates the entire pipeline from input data to semantic IDs:
    1. Input data is passed through the configured encoder
    2. Encoder output is quantized using the selected VQ method
    3. Semantic IDs and codebooks are generated and managed

    The pipeline supports:
    - Multiple input modalities (image, text, video)
    - Various encoder architectures
    - Different VQ methods
    - Unified training and inference interface
    """

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

        # Build encoder based on configuration
        self.encoder = self._build_encoder()

        # Build vector quantizer based on configuration
        self.vq_method = self._build_vq_method()

        # Initialize semantic tokenizer
        self.tokenizer = SemanticTokenizer(
            self.vq_method,
            config.num_embeddings,
            config.embedding_dim
        )

        # Store training metrics
        self.training_history = []

    def _build_encoder(self) -> BaseEncoder:
        """Build encoder based on configuration"""
        from .encoders import EncoderFactory
        return EncoderFactory.create_encoder(
            encoder_type=self.config.encoder_type,
            input_dim=self.config.input_dim,
            embedding_dim=self.config.embedding_dim,
            config=self.config.encoder_config
        )

    def _build_vq_method(self):
        """Build VQ method based on configuration"""
        from .vector_quantization import VectorQuantizer
        from .vq_vae import VQVAE
        from .rq_vae import RQVAE
        from .rq_kmeans import RQKMeans

        if self.config.vq_method == VQMethodType.BASIC_VQ:
            return VectorQuantizer(
                num_embeddings=self.config.num_embeddings,
                embedding_dim=self.config.embedding_dim,
                commitment_cost=self.config.commitment_cost
            )

        elif self.config.vq_method == VQMethodType.VQ_VAE:
            # For VQ-VAE, we need encoder/decoder architecture
            # Here we use the embedding_dim as the latent dimension
            return VQVAE(
                in_channels=self.config.embedding_dim,  # Treat embeddings as channels
                embedding_dim=self.config.embedding_dim,
                num_embeddings=self.config.num_embeddings,
                hidden_dims=self.config.vq_config.get('hidden_dims', [128, 256]),
                commitment_cost=self.config.commitment_cost
            )

        elif self.config.vq_method == VQMethodType.RQ_VAE:
            return RQVAE(
                in_channels=self.config.embedding_dim,
                embedding_dim=self.config.embedding_dim,
                num_embeddings=self.config.num_embeddings,
                num_quantizers=self.config.vq_config.get('num_quantizers', 4),
                hidden_dims=self.config.vq_config.get('hidden_dims', [128, 256]),
                commitment_cost=self.config.commitment_cost
            )

        elif self.config.vq_method == VQMethodType.RQ_KMEANS:
            return RQKMeans(
                n_clusters=self.config.num_embeddings,
                n_stages=self.config.vq_config.get('n_stages', 4),
                random_state=42
            )

        else:
            raise ValueError(f"Unknown VQ method: {self.config.vq_method}")

    def forward(self, x: Tensor) -> Dict[str, Any]:
        """
        Forward pass through the complete pipeline

        Args:
            x: Input data tensor

        Returns:
            Dictionary containing:
            - 'embeddings': Encoded embeddings
            - 'quantized': Quantized embeddings
            - 'semantic_ids': Discrete semantic IDs
            - 'vq_loss': Vector quantization loss (if applicable)
            - 'reconstruction': Reconstructed input (if applicable)
        """
        # Step 1: Encode input to embeddings
        embeddings = self.encoder(x)

        # Step 2: Apply vector quantization
        if self.config.vq_method in [VQMethodType.BASIC_VQ]:
            # Handle basic VQ case
            batch_size = embeddings.size(0)
            if len(embeddings.shape) == 2:  # (batch, features)
                # Reshape for VQ layer expectation: (batch, height, width, channels)
                embeddings_vq = embeddings.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, features)
            else:
                embeddings_vq = embeddings

            quantized, vq_loss, perplexity = self.vq_method(embeddings_vq)

            # Get semantic IDs
            semantic_ids = self.tokenizer.encode_to_ids(embeddings_vq)

            return {
                'embeddings': embeddings,
                'quantized': quantized.view(batch_size, -1),
                'semantic_ids': semantic_ids,
                'vq_loss': vq_loss,
                'perplexity': perplexity
            }

        elif self.config.vq_method == VQMethodType.RQ_KMEANS:
            # RQ-K-means works with numpy, so convert
            embeddings_np = embeddings.detach().cpu().numpy()
            quantized_np = self.vq_method.transform(embeddings_np)
            quantized = torch.from_numpy(quantized_np).to(embeddings.device)

            # Get semantic IDs
            semantic_ids = self.vq_method.get_codes(embeddings_np)

            return {
                'embeddings': embeddings,
                'quantized': quantized,
                'semantic_ids': semantic_ids,
                'reconstruction_error': self.vq_method.calculate_reconstruction_error(embeddings_np)
            }

        else:
            # VQ-VAE and RQ-VAE cases
            batch_size = embeddings.size(0)

            # Reshape embeddings to image-like format for VAE models
            if len(embeddings.shape) == 2:
                # Convert (batch, features) to (batch, channels, height, width)
                height = width = int((embeddings.size(1) // self.config.embedding_dim) ** 0.5)
                if height * width * self.config.embedding_dim != embeddings.size(1):
                    # If not perfect square, pad or use 1D representation
                    height = width = 1
                    channels = embeddings.size(1)
                else:
                    channels = self.config.embedding_dim

                embeddings_reshaped = embeddings.view(batch_size, channels, height, width)
            else:
                embeddings_reshaped = embeddings

            # Forward through VQ-VAE/RQ-VAE
            outputs = self.vq_method(embeddings_reshaped)

            # Extract semantic IDs
            if 'encodings' in outputs:
                semantic_ids = self.tokenizer.encode_to_ids(outputs['encodings'])
            else:
                semantic_ids = self.tokenizer.encode_to_ids(embeddings_reshaped)

            return {
                'embeddings': embeddings,
                'quantized': outputs.get('quantized', outputs.get('reconstructed')),
                'semantic_ids': semantic_ids,
                'vq_loss': outputs.get('vq_loss', outputs.get('rq_loss', 0)),
                'reconstruction': outputs.get('reconstructed'),
                'perplexity': outputs.get('perplexity', outputs.get('perplexity_list'))
            }

    def encode_to_semantic_ids(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        """
        Encode input directly to semantic IDs

        Args:
            x: Input data

        Returns:
            Semantic IDs (tokens)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['semantic_ids']

    def decode_from_semantic_ids(self, semantic_ids: Union[Tensor, List[Tensor]],
                                target_shape: Optional[Tuple[int, ...]] = None) -> Tensor:
        """
        Decode semantic IDs back to embeddings

        Args:
            semantic_ids: Semantic IDs to decode
            target_shape: Target output shape

        Returns:
            Decoded embeddings
        """
        with torch.no_grad():
            if target_shape is None:
                target_shape = (self.config.embedding_dim,)
            return self.tokenizer.decode_from_ids(semantic_ids, target_shape)

    def get_codebook(self) -> Tensor:
        """Get the learned codebook"""
        return self.tokenizer.get_codebook()

    def analyze_semantic_usage(self, dataloader) -> Dict[str, Any]:
        """
        Analyze semantic ID usage across a dataset

        Args:
            dataloader: DataLoader with input data

        Returns:
            Usage analysis results
        """
        all_semantic_ids = []

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                semantic_ids = self.encode_to_semantic_ids(x)

                if isinstance(semantic_ids, list):
                    all_semantic_ids.extend(semantic_ids)
                else:
                    all_semantic_ids.append(semantic_ids)

        # Combine all semantic IDs
        if isinstance(all_semantic_ids[0], list):
            # Multi-stage quantization
            combined_ids = []
            for i in range(len(all_semantic_ids[0])):
                level_ids = torch.cat([ids[i] for ids in all_semantic_ids], dim=0)
                combined_ids.append(level_ids)
            analysis_input = combined_ids
        else:
            analysis_input = torch.cat(all_semantic_ids, dim=0)

        return self.tokenizer.analyze_usage(analysis_input)

    def save_pipeline(self, filepath: str):
        """
        Save complete pipeline (config + weights)

        Args:
            filepath: Path to save pipeline
        """
        save_dict = {
            'config': self.config.to_dict(),
            'model_state_dict': self.state_dict(),
            'training_history': self.training_history
        }
        torch.save(save_dict, filepath)

    @classmethod
    def load_pipeline(cls, filepath: str, device: Optional[torch.device] = None) -> 'VQPipeline':
        """
        Load complete pipeline from file

        Args:
            filepath: Path to pipeline file
            device: Device to load model on

        Returns:
            Loaded pipeline
        """
        if device is None:
            device = torch.device('cpu')

        save_dict = torch.load(filepath, map_location=device)
        config = PipelineConfig.from_dict(save_dict['config'])

        pipeline = cls(config)
        pipeline.load_state_dict(save_dict['model_state_dict'])
        pipeline.training_history = save_dict.get('training_history', [])

        return pipeline

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'config': self.config.to_dict(),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_type': self.config.encoder_type.value,
            'vq_method': self.config.vq_method.value,
            'modality': self.config.modality.value,
            'embedding_dim': self.config.embedding_dim,
            'num_embeddings': self.config.num_embeddings
        }