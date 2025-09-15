# Vector Quantization Pipeline System

## 🎯 Overview

This repository now includes a comprehensive, configurable pipeline system that allows users to easily configure different combinations of:

- **Image, Text, and Video Encoders** from open-source pretrained models
- **Multiple Vector Quantization Methods** for semantic ID and codebook generation
- **Unified Configuration System** for easy experimentation
- **Cross-modal Semantic Representation Learning**

## 🏗️ Pipeline Architecture

```
Input Data → Configurable Encoder → Vector Quantizer → Semantic IDs + Codebook
    ↓              ↓                       ↓                    ↓
[Image/Text/    [CNN/ResNet/ViT/      [Basic VQ/           [Discrete
 Video]          LSTM/Transformer/     VQ-VAE/              Tokens]
                 BERT/3D CNN/          RQ-VAE/
                 Video Transformer]    RQ-K-means]
```

## 📊 Supported Configurations

### Image Encoders
- **CNN**: Standard convolutional architecture with progressive downsampling
- **ResNet**: Residual networks with skip connections for robust feature learning
- **Vision Transformer (ViT)**: Patch-based transformer for fine-grained image understanding

### Text Encoders
- **LSTM**: Bidirectional LSTM with optional attention for sequential modeling
- **Transformer**: Standard transformer encoder with self-attention
- **BERT-like**: Bidirectional transformer with CLS token for contextual understanding

### Video Encoders
- **3D CNN**: Spatiotemporal convolutions for video feature extraction
- **Video Transformer**: Frame-wise ViT with temporal attention for video understanding

### Vector Quantization Methods
- **Basic VQ**: Standard vector quantization with codebook learning
- **VQ-VAE**: Vector quantized variational autoencoder with reconstruction capability
- **RQ-VAE**: Residual quantization for hierarchical semantic representations
- **RQ-K-means**: Multi-stage K-means clustering for hierarchical discrete codes

## 🚀 Quick Start

### 1. Basic Image Pipeline
```python
from vector_quantization import VQPipeline, PipelineConfig, ModalityType, EncoderType, VQMethodType

# Configure pipeline
config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.VISION_TRANSFORMER,
    vq_method=VQMethodType.RQ_VAE,
    input_dim=(3, 64, 64),  # RGB 64x64 images
    embedding_dim=256,
    num_embeddings=1024,
    encoder_config={
        'patch_size': 8,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1
    },
    vq_config={
        'num_quantizers': 4
    }
)

# Initialize and use pipeline
pipeline = VQPipeline(config)
images = torch.randn(8, 3, 64, 64)
outputs = pipeline(images)

# Extract semantic IDs
semantic_ids = pipeline.encode_to_semantic_ids(images)
codebook = pipeline.get_codebook()
```

### 2. Text Processing Pipeline
```python
config = PipelineConfig(
    modality=ModalityType.TEXT,
    encoder_type=EncoderType.BERT_LIKE,
    vq_method=VQMethodType.VQ_VAE,
    input_dim=10000,  # Vocabulary size
    embedding_dim=256,
    num_embeddings=1024,
    encoder_config={
        'd_model': 768,
        'num_layers': 6,
        'num_heads': 8
    }
)

pipeline = VQPipeline(config)
text_tokens = torch.randint(0, 10000, (8, 128))
outputs = pipeline(text_tokens)
```

### 3. Video Analysis Pipeline
```python
config = PipelineConfig(
    modality=ModalityType.VIDEO,
    encoder_type=EncoderType.VIDEO_TRANSFORMER,
    vq_method=VQMethodType.RQ_VAE,
    input_dim=(3, 16, 32, 32),  # RGB, 16 frames, 32x32
    embedding_dim=256,
    num_embeddings=1024,
    encoder_config={
        'patch_size': 8,
        'num_layers': 6
    },
    vq_config={
        'num_quantizers': 4
    }
)

pipeline = VQPipeline(config)
videos = torch.randn(4, 3, 16, 32, 32)
outputs = pipeline(videos)
```

## 📁 Files and Structure

### Core Pipeline Files
- `src/vector_quantization/pipeline.py` - Main pipeline orchestration and configuration
- `src/vector_quantization/encoders.py` - All encoder implementations
- `examples/pipeline_demo.py` - Comprehensive demonstration of all configurations

### Configuration Templates
- `pipeline_configs/` - Pre-configured templates for common use cases
- `pipeline_configs/README.md` - Detailed configuration guide
- Template examples:
  - `image_vit_rqvae.json` - Vision Transformer + RQ-VAE for images
  - `text_bert_vqvae.json` - BERT + VQ-VAE for text
  - `video_3dcnn_rqvae.json` - 3D CNN + RQ-VAE for videos

### Documentation and Examples
- `PIPELINE_GUIDE.md` - This comprehensive guide
- `pipeline_usage_examples.md` - Code examples and usage patterns
- `pipeline_architectures.md` - Visual guide to pipeline architectures
- `examples/pipeline_demo.py` - Full demonstration script

## 🎛️ Configuration System

### Loading from Templates
```python
# Load pre-configured template
config = PipelineConfig.load("pipeline_configs/image_vit_rqvae.json")
pipeline = VQPipeline(config)
```

### Custom Configuration
```python
config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.CNN,
    vq_method=VQMethodType.BASIC_VQ,
    input_dim=(3, 224, 224),
    embedding_dim=512,
    num_embeddings=2048,
    encoder_config={
        'hidden_dims': [64, 128, 256, 512],
        'use_residual': True
    }
)
```

### Saving and Loading Pipelines
```python
# Save configuration only
config.save("my_config.json")

# Save complete pipeline (config + trained weights)
pipeline.save_pipeline("my_pipeline.pth")

# Load complete pipeline
loaded_pipeline = VQPipeline.load_pipeline("my_pipeline.pth")
```

## 🔧 Encoder Configuration Options

### CNN Encoders
```python
encoder_config = {
    'hidden_dims': [64, 128, 256, 512],  # Channel progression
    'kernel_size': 3,                    # Convolution kernel size
    'use_residual': False                # Use residual connections
}
```

### Vision Transformer
```python
encoder_config = {
    'patch_size': 16,         # Image patch size
    'num_layers': 12,         # Number of transformer layers
    'num_heads': 12,          # Number of attention heads
    'mlp_ratio': 4.0,         # MLP expansion ratio
    'dropout': 0.1            # Dropout rate
}
```

### BERT-like Text Encoder
```python
encoder_config = {
    'd_model': 768,                      # Model dimension
    'num_layers': 12,                    # Number of layers
    'num_heads': 12,                     # Attention heads
    'intermediate_size': 3072,           # Feed-forward size
    'max_position_embeddings': 512,     # Max sequence length
    'dropout': 0.1                      # Dropout rate
}
```

### Video Transformer
```python
encoder_config = {
    'patch_size': 16,         # Spatial patch size
    'num_layers': 6,          # Number of layers
    'num_heads': 8,           # Attention heads
    'dropout': 0.1            # Dropout rate
}
```

## 🎯 Vector Quantization Configuration

### Basic VQ
```python
vq_config = {}  # Uses default parameters
```

### VQ-VAE
```python
vq_config = {
    'hidden_dims': [128, 256]  # Encoder/decoder architecture
}
```

### RQ-VAE (Residual Quantization)
```python
vq_config = {
    'num_quantizers': 4,               # Number of quantization levels
    'hidden_dims': [128, 256],         # Architecture
    'shared_codebook': False           # Whether to share codebooks
}
```

### RQ-K-means
```python
vq_config = {
    'n_stages': 4,            # Number of clustering stages
    'max_iter': 300,          # Maximum iterations per stage
    'tol': 1e-4              # Convergence tolerance
}
```

## 📊 Semantic Analysis

### Basic Usage Analysis
```python
# Analyze semantic ID usage across dataset
usage_stats = pipeline.analyze_semantic_usage(dataloader)

print(f"Unique tokens: {usage_stats['unique_tokens']}")
print(f"Usage ratio: {usage_stats['usage_ratio']:.3f}")
print(f"Token entropy: {usage_stats['entropy']:.3f}")
```

### Codebook Management
```python
# Get learned codebook
codebook = pipeline.get_codebook()
print(f"Codebook shape: {codebook.shape}")

# Convert between embeddings and semantic IDs
semantic_ids = pipeline.encode_to_semantic_ids(data)
reconstructed = pipeline.decode_from_semantic_ids(semantic_ids)
```

## 🎨 Cross-Modal Applications

### Image-Text Semantic Matching
```python
# Configure both pipelines with same semantic space
image_config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.VISION_TRANSFORMER,
    embedding_dim=512,
    num_embeddings=2048,
    # ... other config
)

text_config = PipelineConfig(
    modality=ModalityType.TEXT,
    encoder_type=EncoderType.BERT_LIKE,
    embedding_dim=512,          # Same embedding dimension
    num_embeddings=2048,        # Same codebook size
    # ... other config
)

image_pipeline = VQPipeline(image_config)
text_pipeline = VQPipeline(text_config)

# Extract semantic representations
image_semantics = image_pipeline.encode_to_semantic_ids(images)
text_semantics = text_pipeline.encode_to_semantic_ids(text)

# Compare semantic representations for matching
```

## 🧪 Running the Demo

```bash
# Run comprehensive demo (requires PyTorch)
python examples/pipeline_demo.py

# This will:
# 1. Test all encoder-VQ combinations
# 2. Generate semantic IDs and analyze usage
# 3. Create configuration templates
# 4. Generate usage examples and documentation
```

## 🎓 Educational Value

This pipeline system demonstrates:

1. **Modular Design**: How to build flexible, configurable ML systems
2. **Encoder Variety**: Different approaches to representation learning across modalities
3. **Quantization Methods**: Various ways to create discrete semantic representations
4. **Cross-Modal Learning**: How to create unified semantic spaces across modalities
5. **Configuration Management**: Best practices for managing complex ML configurations
6. **Semantic Analysis**: How to analyze and understand learned discrete representations

## 🔬 Research Applications

### Multimodal Representation Learning
- Create shared semantic spaces across image, text, and video
- Study cross-modal semantic alignment
- Develop unified multimodal models

### Discrete Representation Analysis
- Compare different quantization strategies
- Analyze semantic granularity vs. reconstruction quality
- Study codebook utilization patterns

### Architecture Comparison
- Benchmark different encoder architectures
- Study the impact of encoder choice on semantic quality
- Evaluate scalability across modalities

## 🚀 Future Extensions

The modular design makes it easy to add:
- New encoder architectures (e.g., CLIP, DALL-E style encoders)
- Additional VQ methods (e.g., Gumbel quantization, product quantization)
- More modalities (e.g., audio, medical images)
- Advanced semantic analysis tools
- Cross-modal training strategies

## 📝 Summary

This pipeline system provides a comprehensive, educational, and research-ready framework for exploring vector quantization across multiple modalities. Users can:

✅ **Easily configure** different encoder-VQ combinations
✅ **Generate semantic IDs** and codebooks from any supported modality
✅ **Analyze semantic representations** with built-in tools
✅ **Save and load** complete pipeline configurations
✅ **Extend the system** with new encoders and VQ methods
✅ **Learn about** modern representation learning techniques

The system is designed to be both educational and practical, providing clear examples while maintaining the flexibility needed for serious research applications.