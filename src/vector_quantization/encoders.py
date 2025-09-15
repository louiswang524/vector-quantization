"""
Configurable Encoders for Vector Quantization Pipeline

This module provides various encoder architectures for different modalities:
- Image encoders: CNN, ResNet, Vision Transformer
- Text encoders: LSTM, Transformer, BERT-like
- Video encoders: 3D CNN, Video Transformer

All encoders implement the BaseEncoder interface for consistency and interchangeability.

Educational Goals:
- Demonstrate different architectural approaches to representation learning
- Show how various encoders can be integrated with VQ methods
- Provide building blocks for multimodal research
- Illustrate the relationship between encoder choice and VQ performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Any, Union
import math
from .pipeline import BaseEncoder, EncoderType


class ImageCNNEncoder(BaseEncoder):
    """
    Convolutional Neural Network Encoder for Images

    A standard CNN architecture with progressive downsampling,
    suitable for learning hierarchical visual features.

    Architecture:
    - Multiple convolutional blocks with batch normalization
    - Progressive spatial downsampling
    - Global average pooling
    - Final projection to embedding dimension

    Args:
        input_dim: Image dimensions (channels, height, width)
        embedding_dim: Output embedding dimension
        config: Additional configuration parameters
    """

    def __init__(self, input_dim: Tuple[int, int, int], embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        channels, height, width = input_dim
        hidden_dims = config.get('hidden_dims', [64, 128, 256, 512])
        kernel_size = config.get('kernel_size', 3)
        use_residual = config.get('use_residual', False)

        self.layers = nn.ModuleList()

        # Input convolution
        self.layers.append(nn.Sequential(
            nn.Conv2d(channels, hidden_dims[0], kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        ))

        # Progressive downsampling layers
        for i in range(len(hidden_dims) - 1):
            if use_residual:
                self.layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i + 1]))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(inplace=True)
                ))

        # Global average pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(hidden_dims[-1], embedding_dim)

        print(f"CNN Encoder: {channels}×{height}×{width} → {embedding_dim}D")

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Final projection
        x = self.projection(x)

        return x

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class ResidualBlock(nn.Module):
    """Residual block for CNN architectures"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        return F.relu(out)


class ImageResNetEncoder(BaseEncoder):
    """
    ResNet-style Encoder for Images

    Implements a ResNet-like architecture with skip connections
    for better gradient flow and feature learning.

    Args:
        input_dim: Image dimensions (channels, height, width)
        embedding_dim: Output embedding dimension
        config: Configuration including num_layers, width_multiplier
    """

    def __init__(self, input_dim: Tuple[int, int, int], embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        channels, height, width = input_dim
        num_layers = config.get('num_layers', 4)
        width_multiplier = config.get('width_multiplier', 1)

        # Base width for ResNet
        base_width = int(64 * width_multiplier)

        # Initial convolution
        self.conv1 = nn.Conv2d(channels, base_width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layers = nn.ModuleList()
        current_channels = base_width

        for i in range(num_layers):
            out_channels = base_width * (2 ** i)
            stride = 1 if i == 0 else 2

            self.layers.append(self._make_layer(current_channels, out_channels, stride))
            current_channels = out_channels

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, embedding_dim)

        print(f"ResNet Encoder: {channels}×{height}×{width} → {embedding_dim}D")

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Create a ResNet layer with residual blocks"""
        return ResidualBlock(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class VisionTransformerEncoder(BaseEncoder):
    """
    Vision Transformer (ViT) Encoder for Images

    Implements a simplified Vision Transformer that splits images into patches,
    applies transformer layers, and produces embeddings.

    Key Components:
    - Patch embedding layer
    - Positional encoding
    - Multi-head self-attention layers
    - MLP blocks
    - Classification token

    Args:
        input_dim: Image dimensions (channels, height, width)
        embedding_dim: Output embedding dimension
        config: ViT-specific configuration
    """

    def __init__(self, input_dim: Tuple[int, int, int], embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        channels, height, width = input_dim
        patch_size = config.get('patch_size', 16)
        num_layers = config.get('num_layers', 6)
        num_heads = config.get('num_heads', 8)
        mlp_ratio = config.get('mlp_ratio', 4.0)
        dropout = config.get('dropout', 0.1)

        assert height % patch_size == 0 and width % patch_size == 0, \
            "Image dimensions must be divisible by patch size"

        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        patch_dim = channels * patch_size * patch_size

        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, embedding_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, int(embedding_dim * mlp_ratio), dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        print(f"ViT Encoder: {channels}×{height}×{width} → patches:{self.num_patches} → {embedding_dim}D")

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # Split image into patches
        patches = self._extract_patches(x)  # (batch, num_patches, patch_dim)

        # Embed patches
        patch_embeddings = self.patch_embedding(patches)  # (batch, num_patches, embedding_dim)

        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # Add positional encoding
        embeddings = embeddings + self.pos_embedding[:, :embeddings.size(1)]

        # Apply transformer layers
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)

        # Normalize and return classification token
        embeddings = self.norm(embeddings)
        return embeddings[:, 0]  # Return CLS token

    def _extract_patches(self, x: Tensor) -> Tensor:
        """Extract patches from input image"""
        batch_size, channels, height, width = x.shape
        patch_size = self.patch_size

        # Reshape to patches
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, self.num_patches, -1)

        return patches

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and MLP"""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Multi-head self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class TextLSTMEncoder(BaseEncoder):
    """
    LSTM-based Text Encoder

    Uses bidirectional LSTM layers to encode sequential text data,
    with optional attention mechanism for better representation.

    Args:
        input_dim: Vocabulary size or input feature dimension
        embedding_dim: Output embedding dimension
        config: LSTM-specific configuration
    """

    def __init__(self, input_dim: int, embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        vocab_size = input_dim if isinstance(input_dim, int) else input_dim[0]
        embed_dim = config.get('embed_dim', 256)
        hidden_dim = config.get('hidden_dim', 512)
        num_layers = config.get('num_layers', 2)
        dropout = config.get('dropout', 0.1)
        use_attention = config.get('use_attention', False)

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Attention mechanism (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim * 2, 8, dropout=dropout, batch_first=True)

        # Output projection
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        self.projection = nn.Linear(lstm_output_dim, embedding_dim)

        print(f"LSTM Text Encoder: vocab:{vocab_size} → {embedding_dim}D")

    def forward(self, x: Tensor) -> Tensor:
        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch, seq_len, hidden_dim * 2)

        # Apply attention if configured
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use mean pooling over sequence
            pooled = torch.mean(attn_out, dim=1)
        else:
            # Use last hidden state (concatenated forward and backward)
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_dim * 2)

        # Project to embedding dimension
        output = self.projection(pooled)

        return output

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class TextTransformerEncoder(BaseEncoder):
    """
    Transformer-based Text Encoder

    Uses transformer architecture with multi-head self-attention
    for encoding sequential text data.

    Args:
        input_dim: Vocabulary size
        embedding_dim: Output embedding dimension
        config: Transformer-specific configuration
    """

    def __init__(self, input_dim: int, embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        vocab_size = input_dim if isinstance(input_dim, int) else input_dim[0]
        d_model = config.get('d_model', 512)
        num_heads = config.get('num_heads', 8)
        num_layers = config.get('num_layers', 6)
        d_ff = config.get('d_ff', 2048)
        max_seq_len = config.get('max_seq_len', 512)
        dropout = config.get('dropout', 0.1)

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.projection = nn.Linear(d_model, embedding_dim)

        print(f"Transformer Text Encoder: vocab:{vocab_size} → {embedding_dim}D")

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.shape

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # Embed tokens and positions
        token_embeddings = self.token_embedding(x)
        pos_embeddings = self.pos_embedding(positions)
        embeddings = token_embeddings + pos_embeddings

        # Apply transformer
        transformed = self.transformer(embeddings)

        # Global average pooling across sequence
        pooled = torch.mean(transformed, dim=1)

        # Project to embedding dimension
        output = self.projection(pooled)

        return output

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class BERTLikeEncoder(BaseEncoder):
    """
    BERT-like Text Encoder

    Implements a BERT-style encoder with bidirectional attention,
    masked language modeling capabilities, and CLS token for classification.

    Key Components:
    - Token embeddings with segment and position encodings
    - Bidirectional transformer layers
    - CLS token for sequence representation
    - Layer normalization and residual connections

    Args:
        input_dim: Vocabulary size
        embedding_dim: Output embedding dimension
        config: BERT-specific configuration
    """

    def __init__(self, input_dim: int, embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        vocab_size = input_dim if isinstance(input_dim, int) else input_dim[0]
        d_model = config.get('d_model', 768)
        num_layers = config.get('num_layers', 12)
        num_heads = config.get('num_heads', 12)
        intermediate_size = config.get('intermediate_size', 3072)
        max_position_embeddings = config.get('max_position_embeddings', 512)
        dropout = config.get('dropout', 0.1)
        layer_norm_eps = config.get('layer_norm_eps', 1e-12)

        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.embeddings_layernorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.embeddings_dropout = nn.Dropout(dropout)

        # BERT Encoder layers
        self.layers = nn.ModuleList([
            BERTLayer(d_model, num_heads, intermediate_size, dropout, layer_norm_eps)
            for _ in range(num_layers)
        ])

        # Pooler for CLS token
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()

        # Final projection to embedding dimension
        self.projection = nn.Linear(d_model, embedding_dim)

        print(f"BERT-like Text Encoder: vocab:{vocab_size} → {embedding_dim}D")

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through BERT-like encoder

        Args:
            x: Input token ids (batch, seq_len)
            attention_mask: Optional attention mask (batch, seq_len)

        Returns:
            CLS token representation (batch, embedding_dim)
        """
        batch_size, seq_length = x.shape

        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Get embeddings
        token_embeddings = self.token_embeddings(x)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.embeddings_layernorm(embeddings)
        embeddings = self.embeddings_dropout(embeddings)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(x, dtype=torch.float)

        # Extend attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Apply BERT layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask)

        # Pool CLS token (first token)
        cls_token = hidden_states[:, 0]
        pooled_output = self.pooler_activation(self.pooler(cls_token))

        # Project to embedding dimension
        output = self.projection(pooled_output)

        return output

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class BERTLayer(nn.Module):
    """Single BERT transformer layer"""

    def __init__(self, d_model: int, num_heads: int, intermediate_size: int,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()

        # Multi-head attention
        self.attention = BERTSelfAttention(d_model, num_heads, dropout)
        self.attention_output = nn.Linear(d_model, d_model)
        self.attention_layernorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(dropout)

        # Feed forward network
        self.intermediate = nn.Linear(d_model, intermediate_size)
        self.intermediate_activation = nn.GELU()
        self.output = nn.Linear(intermediate_size, d_model)
        self.output_layernorm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # Self attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layernorm(attention_output + hidden_states)

        # Feed forward
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_activation(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layernorm(layer_output + attention_output)

        return layer_output


class BERTSelfAttention(nn.Module):
    """BERT self-attention mechanism"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.attention_head_size = d_model // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size

        # Linear projections
        self.query = nn.Linear(d_model, self.all_head_size)
        self.key = nn.Linear(d_model, self.all_head_size)
        self.value = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """Transpose tensor for multi-head attention"""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # Linear projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        attention_scores = attention_scores + attention_mask

        # Attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Reshape back to original dimension
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class Video3DCNNEncoder(BaseEncoder):
    """
    3D Convolutional Encoder for Videos

    Uses 3D convolutions to capture spatiotemporal features in video data.
    Suitable for short video clips and action recognition.

    Args:
        input_dim: Video dimensions (channels, frames, height, width)
        embedding_dim: Output embedding dimension
        config: 3D CNN configuration
    """

    def __init__(self, input_dim: Tuple[int, int, int, int], embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        channels, frames, height, width = input_dim
        hidden_dims = config.get('hidden_dims', [64, 128, 256, 512])
        kernel_size = config.get('kernel_size', (3, 3, 3))
        temporal_stride = config.get('temporal_stride', 2)

        self.layers = nn.ModuleList()

        # Initial 3D convolution
        self.layers.append(nn.Sequential(
            nn.Conv3d(channels, hidden_dims[0], kernel_size, stride=(temporal_stride, 2, 2), padding=1),
            nn.BatchNorm3d(hidden_dims[0]),
            nn.ReLU(inplace=True)
        ))

        # Progressive 3D convolutions
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv3d(hidden_dims[i], hidden_dims[i + 1], kernel_size,
                         stride=(temporal_stride, 2, 2), padding=1),
                nn.BatchNorm3d(hidden_dims[i + 1]),
                nn.ReLU(inplace=True)
            ))

        # Global spatiotemporal pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Linear(hidden_dims[-1], embedding_dim)

        print(f"3D CNN Video Encoder: {channels}×{frames}×{height}×{width} → {embedding_dim}D")

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)

        # Global pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Final projection
        x = self.projection(x)

        return x

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class VideoTransformerEncoder(BaseEncoder):
    """
    Video Transformer Encoder

    Treats video frames as sequences and applies transformer layers
    to learn temporal relationships between frames.

    Args:
        input_dim: Video dimensions (channels, frames, height, width)
        embedding_dim: Output embedding dimension
        config: Video transformer configuration
    """

    def __init__(self, input_dim: Tuple[int, int, int, int], embedding_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, embedding_dim)

        channels, frames, height, width = input_dim
        patch_size = config.get('patch_size', 16)
        num_layers = config.get('num_layers', 6)
        num_heads = config.get('num_heads', 8)
        dropout = config.get('dropout', 0.1)

        # Frame-level processing (treat each frame as ViT)
        self.frame_encoder = VisionTransformerEncoder(
            (channels, height, width),
            embedding_dim,
            config
        )

        # Temporal transformer for frame sequence
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, frames, embedding_dim))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers)

        # Final projection
        self.final_norm = nn.LayerNorm(embedding_dim)

        print(f"Video Transformer: {channels}×{frames}×{height}×{width} → {embedding_dim}D")

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = x.shape

        # Process each frame independently
        frame_features = []
        for t in range(frames):
            frame = x[:, :, t, :, :]  # (batch, channels, height, width)
            frame_feat = self.frame_encoder(frame)  # (batch, embedding_dim)
            frame_features.append(frame_feat)

        # Stack frame features
        frame_sequence = torch.stack(frame_features, dim=1)  # (batch, frames, embedding_dim)

        # Add temporal positional encoding
        frame_sequence = frame_sequence + self.temporal_pos_embedding

        # Apply temporal transformer
        temporal_features = self.temporal_transformer(frame_sequence)

        # Global temporal pooling
        video_features = torch.mean(temporal_features, dim=1)

        # Final normalization
        output = self.final_norm(video_features)

        return output

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.embedding_dim,)


class EncoderFactory:
    """
    Factory class for creating encoders based on configuration

    This factory pattern allows for easy encoder selection and configuration,
    making the pipeline flexible and extensible.
    """

    @staticmethod
    def create_encoder(encoder_type: EncoderType, input_dim: Union[Tuple[int, ...], int],
                      embedding_dim: int, config: Dict[str, Any]) -> BaseEncoder:
        """
        Create an encoder based on the specified type and configuration

        Args:
            encoder_type: Type of encoder to create
            input_dim: Input dimensions (format depends on encoder type)
            embedding_dim: Output embedding dimension
            config: Encoder-specific configuration

        Returns:
            Configured encoder instance
        """

        if encoder_type == EncoderType.CNN:
            if not isinstance(input_dim, tuple) or len(input_dim) != 3:
                raise ValueError("CNN encoder requires input_dim as (channels, height, width)")
            return ImageCNNEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.RESNET:
            if not isinstance(input_dim, tuple) or len(input_dim) != 3:
                raise ValueError("ResNet encoder requires input_dim as (channels, height, width)")
            return ImageResNetEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.VISION_TRANSFORMER:
            if not isinstance(input_dim, tuple) or len(input_dim) != 3:
                raise ValueError("ViT encoder requires input_dim as (channels, height, width)")
            return VisionTransformerEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.LSTM:
            if isinstance(input_dim, tuple):
                input_dim = input_dim[0]  # Take vocabulary size
            return TextLSTMEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.TRANSFORMER:
            if isinstance(input_dim, tuple):
                input_dim = input_dim[0]  # Take vocabulary size
            return TextTransformerEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.BERT_LIKE:
            if isinstance(input_dim, tuple):
                input_dim = input_dim[0]  # Take vocabulary size
            return BERTLikeEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.CNN_3D:
            if not isinstance(input_dim, tuple) or len(input_dim) != 4:
                raise ValueError("3D CNN encoder requires input_dim as (channels, frames, height, width)")
            return Video3DCNNEncoder(input_dim, embedding_dim, config)

        elif encoder_type == EncoderType.VIDEO_TRANSFORMER:
            if not isinstance(input_dim, tuple) or len(input_dim) != 4:
                raise ValueError("Video Transformer requires input_dim as (channels, frames, height, width)")
            return VideoTransformerEncoder(input_dim, embedding_dim, config)

        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    @staticmethod
    def get_encoder_info(encoder_type: EncoderType) -> Dict[str, Any]:
        """Get information about a specific encoder type"""

        encoder_info = {
            EncoderType.CNN: {
                'name': 'Convolutional Neural Network',
                'modality': 'image',
                'input_format': '(channels, height, width)',
                'description': 'Standard CNN with progressive downsampling',
                'config_options': ['hidden_dims', 'kernel_size', 'use_residual']
            },
            EncoderType.RESNET: {
                'name': 'Residual Neural Network',
                'modality': 'image',
                'input_format': '(channels, height, width)',
                'description': 'ResNet architecture with skip connections',
                'config_options': ['num_layers', 'width_multiplier']
            },
            EncoderType.VISION_TRANSFORMER: {
                'name': 'Vision Transformer',
                'modality': 'image',
                'input_format': '(channels, height, width)',
                'description': 'Transformer architecture for images using patches',
                'config_options': ['patch_size', 'num_layers', 'num_heads', 'mlp_ratio', 'dropout']
            },
            EncoderType.LSTM: {
                'name': 'Long Short-Term Memory',
                'modality': 'text',
                'input_format': 'vocab_size (int)',
                'description': 'Bidirectional LSTM for sequential text',
                'config_options': ['embed_dim', 'hidden_dim', 'num_layers', 'dropout', 'use_attention']
            },
            EncoderType.TRANSFORMER: {
                'name': 'Transformer',
                'modality': 'text',
                'input_format': 'vocab_size (int)',
                'description': 'Transformer encoder for text sequences',
                'config_options': ['d_model', 'num_heads', 'num_layers', 'd_ff', 'max_seq_len', 'dropout']
            },
            EncoderType.BERT_LIKE: {
                'name': 'BERT-like',
                'modality': 'text',
                'input_format': 'vocab_size (int)',
                'description': 'BERT-style bidirectional transformer with CLS token',
                'config_options': ['d_model', 'num_heads', 'num_layers', 'intermediate_size', 'max_position_embeddings', 'dropout', 'layer_norm_eps']
            },
            EncoderType.CNN_3D: {
                'name': '3D Convolutional Neural Network',
                'modality': 'video',
                'input_format': '(channels, frames, height, width)',
                'description': '3D CNN for spatiotemporal feature extraction',
                'config_options': ['hidden_dims', 'kernel_size', 'temporal_stride']
            },
            EncoderType.VIDEO_TRANSFORMER: {
                'name': 'Video Transformer',
                'modality': 'video',
                'input_format': '(channels, frames, height, width)',
                'description': 'Transformer for video sequence modeling',
                'config_options': ['patch_size', 'num_layers', 'num_heads', 'dropout']
            }
        }

        return encoder_info.get(encoder_type, {})

    @staticmethod
    def list_available_encoders() -> Dict[str, List[EncoderType]]:
        """List all available encoders grouped by modality"""
        return {
            'image': [EncoderType.CNN, EncoderType.RESNET, EncoderType.VISION_TRANSFORMER],
            'text': [EncoderType.LSTM, EncoderType.TRANSFORMER, EncoderType.BERT_LIKE],
            'video': [EncoderType.CNN_3D, EncoderType.VIDEO_TRANSFORMER]
        }