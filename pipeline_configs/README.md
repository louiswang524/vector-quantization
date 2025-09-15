# Vector Quantization Pipeline Configuration Templates

This directory contains pre-configured templates for different encoder and VQ method combinations across various modalities.

## Available Templates

### Image Processing

#### CNN-based Encoders
- `image_cnn_basicvq.json` - Simple CNN + Basic VQ for basic image features
- `image_cnn_vqvae.json` - CNN + VQ-VAE for image reconstruction
- `image_cnn_rqvae.json` - CNN + RQ-VAE for hierarchical image semantics

#### ResNet-based Encoders
- `image_resnet_basicvq.json` - ResNet + Basic VQ for robust features
- `image_resnet_vqvae.json` - ResNet + VQ-VAE for quality reconstruction
- `image_resnet_rqvae.json` - ResNet + RQ-VAE for deep image understanding

#### Vision Transformer (ViT)
- `image_vit_basicvq.json` - ViT + Basic VQ for patch-based features
- `image_vit_vqvae.json` - ViT + VQ-VAE for transformer-based reconstruction
- `image_vit_rqvae.json` - ViT + RQ-VAE for fine-grained visual semantics

### Text Processing

#### LSTM-based Encoders
- `text_lstm_basicvq.json` - LSTM + Basic VQ for sequential features
- `text_lstm_vqvae.json` - LSTM + VQ-VAE for text reconstruction
- `text_lstm_rqkmeans.json` - LSTM + RQ-K-means for hierarchical clustering

#### Transformer-based Encoders
- `text_transformer_basicvq.json` - Transformer + Basic VQ for attention features
- `text_transformer_vqvae.json` - Transformer + VQ-VAE for semantic reconstruction
- `text_transformer_rqkmeans.json` - Transformer + RQ-K-means for multi-level semantics

#### BERT-like Encoders
- `text_bert_basicvq.json` - BERT + Basic VQ for contextual features
- `text_bert_vqvae.json` - BERT + VQ-VAE for contextual reconstruction
- `text_bert_rqvae.json` - BERT + RQ-VAE for hierarchical context

### Video Processing

#### 3D CNN-based Encoders
- `video_3dcnn_basicvq.json` - 3D CNN + Basic VQ for spatiotemporal features
- `video_3dcnn_vqvae.json` - 3D CNN + VQ-VAE for video reconstruction
- `video_3dcnn_rqvae.json` - 3D CNN + RQ-VAE for hierarchical video semantics

#### Video Transformer
- `video_transformer_basicvq.json` - Video Transformer + Basic VQ
- `video_transformer_vqvae.json` - Video Transformer + VQ-VAE
- `video_transformer_rqvae.json` - Video Transformer + RQ-VAE

## Using Configuration Templates

### Method 1: Load from File
```python
from vector_quantization import PipelineConfig, VQPipeline

# Load a pre-configured template
config = PipelineConfig.load("pipeline_configs/image_vit_rqvae.json")
pipeline = VQPipeline(config)
```

### Method 2: Modify Template
```python
import json
from vector_quantization import PipelineConfig, VQPipeline

# Load and modify template
with open("pipeline_configs/image_cnn_basicvq.json", 'r') as f:
    config_dict = json.load(f)

# Modify parameters
config_dict['embedding_dim'] = 512
config_dict['num_embeddings'] = 2048
config_dict['encoder_config']['hidden_dims'] = [128, 256, 512]

# Create config and pipeline
config = PipelineConfig.from_dict(config_dict)
pipeline = VQPipeline(config)
```

### Method 3: Create Custom Configuration
```python
from vector_quantization import (
    PipelineConfig, VQPipeline,
    ModalityType, EncoderType, VQMethodType
)

config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.VISION_TRANSFORMER,
    vq_method=VQMethodType.RQ_VAE,
    input_dim=(3, 224, 224),  # RGB 224x224 images
    embedding_dim=768,
    num_embeddings=8192,
    encoder_config={
        'patch_size': 16,
        'num_layers': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1
    },
    vq_config={
        'num_quantizers': 8,
        'hidden_dims': [256, 512]
    }
)

pipeline = VQPipeline(config)
```

## Configuration Parameters

### Common Parameters
- `modality`: Input data type (image/text/video)
- `encoder_type`: Architecture type (cnn/resnet/vit/lstm/transformer/bert_like/cnn_3d/video_transformer)
- `vq_method`: Quantization method (basic_vq/vq_vae/rq_vae/rq_kmeans)
- `input_dim`: Input data dimensions
- `embedding_dim`: Encoder output dimension
- `num_embeddings`: Codebook size
- `commitment_cost`: VQ commitment loss weight
- `learning_rate`: Training learning rate
- `batch_size`: Training batch size

### Encoder-Specific Parameters

#### CNN Encoders
- `hidden_dims`: Channel dimensions for each layer
- `kernel_size`: Convolution kernel size
- `use_residual`: Whether to use residual connections

#### ResNet Encoders
- `num_layers`: Number of residual layers
- `width_multiplier`: Channel width multiplier

#### Vision Transformer
- `patch_size`: Image patch size
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `mlp_ratio`: MLP expansion ratio
- `dropout`: Dropout rate

#### LSTM Encoders
- `embed_dim`: Token embedding dimension
- `hidden_dim`: LSTM hidden dimension
- `num_layers`: Number of LSTM layers
- `use_attention`: Whether to use attention mechanism

#### Transformer Encoders
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `d_ff`: Feed-forward dimension
- `max_seq_len`: Maximum sequence length

#### BERT-like Encoders
- `d_model`: Model dimension
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `intermediate_size`: Feed-forward hidden size
- `max_position_embeddings`: Maximum position embeddings

#### 3D CNN Encoders
- `hidden_dims`: Channel dimensions for each layer
- `kernel_size`: 3D convolution kernel size
- `temporal_stride`: Temporal downsampling stride

### VQ Method Parameters

#### VQ-VAE
- `hidden_dims`: Encoder/decoder hidden dimensions

#### RQ-VAE
- `num_quantizers`: Number of quantization levels
- `hidden_dims`: Encoder/decoder hidden dimensions
- `shared_codebook`: Whether to share codebook across levels

#### RQ-K-means
- `n_stages`: Number of clustering stages
- `max_iter`: Maximum iterations per stage
- `tol`: Convergence tolerance

## Best Practices

### For Images
- Use larger `embedding_dim` (512-1024) for high-resolution images
- Consider ViT for patch-based analysis
- Use RQ-VAE for hierarchical visual understanding

### For Text
- Match `embedding_dim` to your language model dimension
- Use BERT-like encoders for contextual understanding
- Consider RQ-K-means for hierarchical text clustering

### For Videos
- Use smaller spatial resolution to manage memory
- Video Transformer for temporal modeling
- RQ-VAE for multi-level temporal semantics

### Performance Tuning
- Start with smaller models for experimentation
- Increase `num_embeddings` for more diverse representations
- Adjust `commitment_cost` (0.1-0.5) based on training stability
- Use appropriate `batch_size` for your hardware

## Troubleshooting

### Memory Issues
- Reduce `batch_size`
- Use smaller `embedding_dim`
- Reduce input resolution

### Training Instability
- Adjust `commitment_cost`
- Reduce `learning_rate`
- Use gradient clipping

### Poor Reconstruction
- Increase `num_embeddings`
- Use larger `embedding_dim`
- Try different VQ methods (VQ-VAE vs RQ-VAE)

### Low Codebook Utilization
- Reduce `num_embeddings`
- Adjust `commitment_cost`
- Check data diversity