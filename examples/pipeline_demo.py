"""
Vector Quantization Pipeline Demo

This comprehensive demo shows how to configure and use the VQ Pipeline with:
- Different modalities (image, text, video)
- Various encoder architectures
- Multiple vector quantization methods
- Semantic ID generation and codebook analysis

The pipeline provides a unified interface for:
1. Configuring encoder types (CNN, ResNet, ViT, LSTM, Transformer, BERT, 3D CNN, Video Transformer)
2. Selecting VQ methods (Basic VQ, VQ-VAE, RQ-VAE, RQ-K-means)
3. Generating semantic IDs and managing codebooks
4. Cross-modal semantic representation learning

Educational Goals:
- Demonstrate modular pipeline design
- Show encoder-VQ method combinations
- Illustrate semantic tokenization across modalities
- Provide templates for research and experimentation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vector_quantization import (
    VQPipeline, PipelineConfig, SemanticTokenizer,
    ModalityType, EncoderType, VQMethodType, EncoderFactory
)


def create_sample_data(modality: ModalityType, batch_size: int = 8) -> torch.Tensor:
    """
    Create sample data for different modalities

    Args:
        modality: Type of data to generate
        batch_size: Number of samples

    Returns:
        Sample data tensor
    """
    if modality == ModalityType.IMAGE:
        # RGB images: (batch, channels, height, width)
        return torch.randn(batch_size, 3, 64, 64)

    elif modality == ModalityType.TEXT:
        # Token sequences: (batch, sequence_length)
        vocab_size = 10000
        seq_length = 128
        return torch.randint(0, vocab_size, (batch_size, seq_length))

    elif modality == ModalityType.VIDEO:
        # Video data: (batch, channels, frames, height, width)
        return torch.randn(batch_size, 3, 16, 32, 32)

    else:
        raise ValueError(f"Unknown modality: {modality}")


def create_image_pipeline_configs() -> List[Dict[str, Any]]:
    """Create example configurations for image modality"""

    base_config = {
        'modality': ModalityType.IMAGE,
        'input_dim': (3, 64, 64),  # RGB 64x64 images
        'embedding_dim': 256,
        'num_embeddings': 1024,
        'commitment_cost': 0.25,
        'learning_rate': 1e-4,
        'batch_size': 32
    }

    configs = []

    # 1. CNN + Basic VQ
    configs.append({
        **base_config,
        'name': 'Image_CNN_BasicVQ',
        'encoder_type': EncoderType.CNN,
        'vq_method': VQMethodType.BASIC_VQ,
        'encoder_config': {
            'hidden_dims': [64, 128, 256],
            'kernel_size': 3,
            'use_residual': False
        },
        'vq_config': {}
    })

    # 2. ResNet + VQ-VAE
    configs.append({
        **base_config,
        'name': 'Image_ResNet_VQVAE',
        'encoder_type': EncoderType.RESNET,
        'vq_method': VQMethodType.VQ_VAE,
        'encoder_config': {
            'num_layers': 4,
            'width_multiplier': 1
        },
        'vq_config': {
            'hidden_dims': [128, 256]
        }
    })

    # 3. Vision Transformer + RQ-VAE
    configs.append({
        **base_config,
        'name': 'Image_ViT_RQVAE',
        'encoder_type': EncoderType.VISION_TRANSFORMER,
        'vq_method': VQMethodType.RQ_VAE,
        'encoder_config': {
            'patch_size': 8,
            'num_layers': 6,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout': 0.1
        },
        'vq_config': {
            'num_quantizers': 4,
            'hidden_dims': [128, 256]
        }
    })

    return configs


def create_text_pipeline_configs() -> List[Dict[str, Any]]:
    """Create example configurations for text modality"""

    base_config = {
        'modality': ModalityType.TEXT,
        'input_dim': 10000,  # Vocabulary size
        'embedding_dim': 256,
        'num_embeddings': 1024,
        'commitment_cost': 0.25,
        'learning_rate': 1e-4,
        'batch_size': 32
    }

    configs = []

    # 1. LSTM + Basic VQ
    configs.append({
        **base_config,
        'name': 'Text_LSTM_BasicVQ',
        'encoder_type': EncoderType.LSTM,
        'vq_method': VQMethodType.BASIC_VQ,
        'encoder_config': {
            'embed_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.1,
            'use_attention': True
        },
        'vq_config': {}
    })

    # 2. Transformer + RQ-K-means
    configs.append({
        **base_config,
        'name': 'Text_Transformer_RQKMeans',
        'encoder_type': EncoderType.TRANSFORMER,
        'vq_method': VQMethodType.RQ_KMEANS,
        'encoder_config': {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 512,
            'dropout': 0.1
        },
        'vq_config': {
            'n_stages': 4
        }
    })

    # 3. BERT-like + VQ-VAE
    configs.append({
        **base_config,
        'name': 'Text_BERT_VQVAE',
        'encoder_type': EncoderType.BERT_LIKE,
        'vq_method': VQMethodType.VQ_VAE,
        'encoder_config': {
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
            'max_position_embeddings': 512,
            'dropout': 0.1
        },
        'vq_config': {
            'hidden_dims': [128, 256]
        }
    })

    return configs


def create_video_pipeline_configs() -> List[Dict[str, Any]]:
    """Create example configurations for video modality"""

    base_config = {
        'modality': ModalityType.VIDEO,
        'input_dim': (3, 16, 32, 32),  # RGB, 16 frames, 32x32 resolution
        'embedding_dim': 256,
        'num_embeddings': 1024,
        'commitment_cost': 0.25,
        'learning_rate': 1e-4,
        'batch_size': 16
    }

    configs = []

    # 1. 3D CNN + Basic VQ
    configs.append({
        **base_config,
        'name': 'Video_3DCNN_BasicVQ',
        'encoder_type': EncoderType.CNN_3D,
        'vq_method': VQMethodType.BASIC_VQ,
        'encoder_config': {
            'hidden_dims': [64, 128, 256],
            'kernel_size': (3, 3, 3),
            'temporal_stride': 2
        },
        'vq_config': {}
    })

    # 2. Video Transformer + RQ-VAE
    configs.append({
        **base_config,
        'name': 'Video_Transformer_RQVAE',
        'encoder_type': EncoderType.VIDEO_TRANSFORMER,
        'vq_method': VQMethodType.RQ_VAE,
        'encoder_config': {
            'patch_size': 8,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1
        },
        'vq_config': {
            'num_quantizers': 4,
            'hidden_dims': [128, 256]
        }
    })

    return configs


def demonstrate_pipeline(config_dict: Dict[str, Any]) -> Tuple[VQPipeline, Dict[str, Any]]:
    """
    Demonstrate a single pipeline configuration

    Args:
        config_dict: Configuration dictionary

    Returns:
        Tuple of (pipeline, results)
    """
    print(f"\n{'='*60}")
    print(f"🎯 Testing: {config_dict['name']}")
    print(f"{'='*60}")

    # Create pipeline configuration
    config = PipelineConfig(
        modality=config_dict['modality'],
        encoder_type=config_dict['encoder_type'],
        vq_method=config_dict['vq_method'],
        input_dim=config_dict['input_dim'],
        encoder_config=config_dict['encoder_config'],
        embedding_dim=config_dict['embedding_dim'],
        num_embeddings=config_dict['num_embeddings'],
        commitment_cost=config_dict['commitment_cost'],
        vq_config=config_dict['vq_config'],
        learning_rate=config_dict['learning_rate'],
        batch_size=config_dict['batch_size']
    )

    # Initialize pipeline
    print(f"🔧 Initializing pipeline...")
    print(f"   Modality: {config.modality.value}")
    print(f"   Encoder: {config.encoder_type.value}")
    print(f"   VQ Method: {config.vq_method.value}")

    pipeline = VQPipeline(config)

    # Get model info
    model_info = pipeline.get_model_info()
    print(f"   Parameters: {model_info['total_parameters']:,}")

    # Create sample data
    sample_data = create_sample_data(config.modality, batch_size=4)
    print(f"   Input shape: {list(sample_data.shape)}")

    # Test forward pass
    pipeline.eval()
    with torch.no_grad():
        try:
            outputs = pipeline(sample_data)

            print(f"✅ Forward pass successful!")
            print(f"   Embeddings shape: {list(outputs['embeddings'].shape)}")
            print(f"   Quantized shape: {list(outputs['quantized'].shape)}")

            # Print semantic IDs info
            semantic_ids = outputs['semantic_ids']
            if isinstance(semantic_ids, list):
                print(f"   Semantic IDs: {len(semantic_ids)} levels")
                for i, ids in enumerate(semantic_ids):
                    print(f"     Level {i}: {list(ids.shape)}")
            else:
                print(f"   Semantic IDs shape: {list(semantic_ids.shape)}")

            # Print loss information
            if 'vq_loss' in outputs:
                print(f"   VQ Loss: {outputs['vq_loss']:.4f}")
            if 'perplexity' in outputs:
                perplexity = outputs['perplexity']
                if isinstance(perplexity, list):
                    print(f"   Perplexity: {[f'{p:.2f}' for p in perplexity]}")
                else:
                    print(f"   Perplexity: {perplexity:.2f}")

            # Test semantic ID encoding/decoding
            semantic_ids_only = pipeline.encode_to_semantic_ids(sample_data)
            print(f"   Semantic ID extraction: ✅")

            # Test codebook access
            codebook = pipeline.get_codebook()
            print(f"   Codebook shape: {list(codebook.shape)}")

            return pipeline, outputs

        except Exception as e:
            print(f"❌ Error during forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            return pipeline, {}


def analyze_semantic_usage(pipeline: VQPipeline, sample_data: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze semantic ID usage patterns

    Args:
        pipeline: Trained pipeline
        sample_data: Sample data for analysis

    Returns:
        Usage analysis results
    """
    print(f"\n📊 Analyzing semantic usage...")

    # Generate multiple batches of semantic IDs
    all_semantic_ids = []
    num_batches = 5

    pipeline.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            # Create new sample data for each batch
            batch_data = create_sample_data(pipeline.config.modality, batch_size=8)
            semantic_ids = pipeline.encode_to_semantic_ids(batch_data)

            if isinstance(semantic_ids, list):
                if not all_semantic_ids:
                    all_semantic_ids = [[] for _ in range(len(semantic_ids))]
                for i, level_ids in enumerate(semantic_ids):
                    all_semantic_ids[i].append(level_ids)
            else:
                all_semantic_ids.append(semantic_ids)

    # Combine semantic IDs
    if isinstance(all_semantic_ids[0], list):
        # Multi-level quantization
        combined_ids = []
        for i in range(len(all_semantic_ids)):
            level_ids = torch.cat(all_semantic_ids[i], dim=0)
            combined_ids.append(level_ids)
        analysis_input = combined_ids
    else:
        analysis_input = torch.cat(all_semantic_ids, dim=0)

    # Analyze usage
    usage_stats = pipeline.tokenizer.analyze_usage(analysis_input)

    print(f"   Total tokens: {usage_stats['total_tokens']}")
    print(f"   Unique tokens: {usage_stats['unique_tokens']}")
    print(f"   Usage ratio: {usage_stats['usage_ratio']:.3f}")
    print(f"   Token entropy: {usage_stats['entropy']:.3f}")

    return usage_stats


def save_configuration_templates():
    """Save example configurations as templates"""
    print(f"\n💾 Saving configuration templates...")

    output_dir = Path("pipeline_configs")
    output_dir.mkdir(exist_ok=True)

    # Get all configurations
    all_configs = []
    all_configs.extend(create_image_pipeline_configs())
    all_configs.extend(create_text_pipeline_configs())
    all_configs.extend(create_video_pipeline_configs())

    # Save individual configs
    for config_dict in all_configs:
        filename = f"{config_dict['name'].lower()}.json"
        filepath = output_dir / filename

        # Convert enums to strings for JSON serialization
        json_config = {}
        for key, value in config_dict.items():
            if hasattr(value, 'value'):  # Enum
                json_config[key] = value.value
            else:
                json_config[key] = value

        with open(filepath, 'w') as f:
            json.dump(json_config, f, indent=2)

        print(f"   Saved: {filename}")

    # Save combined template file
    template_file = output_dir / "all_templates.json"
    with open(template_file, 'w') as f:
        json_configs = []
        for config_dict in all_configs:
            json_config = {}
            for key, value in config_dict.items():
                if hasattr(value, 'value'):  # Enum
                    json_config[key] = value.value
                else:
                    json_config[key] = value
            json_configs.append(json_config)

        json.dump(json_configs, f, indent=2)

    print(f"   Combined template: all_templates.json")


def create_usage_examples():
    """Create code examples for common usage patterns"""

    usage_examples = """
# Vector Quantization Pipeline Usage Examples

## 1. Basic Image Pipeline with CNN + VQ

```python
import torch
from vector_quantization import VQPipeline, PipelineConfig, ModalityType, EncoderType, VQMethodType

# Configure pipeline
config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.CNN,
    vq_method=VQMethodType.BASIC_VQ,
    input_dim=(3, 64, 64),  # RGB 64x64 images
    embedding_dim=256,
    num_embeddings=1024,
    encoder_config={
        'hidden_dims': [64, 128, 256],
        'kernel_size': 3,
        'use_residual': False
    }
)

# Initialize pipeline
pipeline = VQPipeline(config)

# Process images
images = torch.randn(8, 3, 64, 64)  # Batch of 8 images
outputs = pipeline(images)

# Extract semantic IDs
semantic_ids = pipeline.encode_to_semantic_ids(images)
print(f"Semantic IDs shape: {semantic_ids.shape}")

# Get codebook
codebook = pipeline.get_codebook()
print(f"Codebook shape: {codebook.shape}")
```

## 2. Text Pipeline with BERT + VQ-VAE

```python
# Configure text pipeline
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
        'num_heads': 8,
        'intermediate_size': 2048,
        'dropout': 0.1
    },
    vq_config={
        'hidden_dims': [128, 256]
    }
)

pipeline = VQPipeline(config)

# Process text (token sequences)
text_tokens = torch.randint(0, 10000, (8, 128))  # Batch of 8 sequences
outputs = pipeline(text_tokens)

print(f"VQ Loss: {outputs['vq_loss']:.4f}")
print(f"Perplexity: {outputs['perplexity']:.2f}")
```

## 3. Video Pipeline with 3D CNN + RQ-VAE

```python
# Configure video pipeline
config = PipelineConfig(
    modality=ModalityType.VIDEO,
    encoder_type=EncoderType.CNN_3D,
    vq_method=VQMethodType.RQ_VAE,
    input_dim=(3, 16, 32, 32),  # RGB, 16 frames, 32x32
    embedding_dim=256,
    num_embeddings=512,
    encoder_config={
        'hidden_dims': [64, 128, 256],
        'temporal_stride': 2
    },
    vq_config={
        'num_quantizers': 4
    }
)

pipeline = VQPipeline(config)

# Process videos
videos = torch.randn(4, 3, 16, 32, 32)  # Batch of 4 videos
outputs = pipeline(videos)

# RQ-VAE provides multi-level semantic IDs
semantic_ids = outputs['semantic_ids']
print(f"Number of quantization levels: {len(semantic_ids)}")
for i, level_ids in enumerate(semantic_ids):
    print(f"Level {i} IDs shape: {level_ids.shape}")
```

## 4. Pipeline Configuration from File

```python
# Save configuration
config.save("my_pipeline_config.json")

# Load configuration
loaded_config = PipelineConfig.load("my_pipeline_config.json")
pipeline = VQPipeline(loaded_config)

# Save complete pipeline (config + weights)
pipeline.save_pipeline("my_trained_pipeline.pth")

# Load complete pipeline
loaded_pipeline = VQPipeline.load_pipeline("my_trained_pipeline.pth")
```

## 5. Semantic Analysis

```python
# Analyze semantic usage across dataset
from torch.utils.data import DataLoader

# Assuming you have a dataloader with your data
dataloader = DataLoader(your_dataset, batch_size=32)

# Analyze semantic ID usage
usage_stats = pipeline.analyze_semantic_usage(dataloader)

print(f"Unique tokens used: {usage_stats['unique_tokens']}")
print(f"Usage ratio: {usage_stats['usage_ratio']:.3f}")
print(f"Token entropy: {usage_stats['entropy']:.3f}")
```

## 6. Cross-Modal Applications

```python
# Example: Image-Text matching using semantic IDs

# Configure image pipeline
image_config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.VISION_TRANSFORMER,
    vq_method=VQMethodType.BASIC_VQ,
    input_dim=(3, 64, 64),
    embedding_dim=512,
    num_embeddings=2048
)

# Configure text pipeline
text_config = PipelineConfig(
    modality=ModalityType.TEXT,
    encoder_type=EncoderType.TRANSFORMER,
    vq_method=VQMethodType.BASIC_VQ,
    input_dim=10000,
    embedding_dim=512,
    num_embeddings=2048  # Same codebook size for compatibility
)

image_pipeline = VQPipeline(image_config)
text_pipeline = VQPipeline(text_config)

# Process both modalities
image_semantic_ids = image_pipeline.encode_to_semantic_ids(images)
text_semantic_ids = text_pipeline.encode_to_semantic_ids(text_tokens)

# Compare semantic representations
# (Implementation depends on your specific matching strategy)
```
"""

    with open("pipeline_usage_examples.md", 'w') as f:
        f.write(usage_examples)

    print(f"📝 Usage examples saved to: pipeline_usage_examples.md")


def main():
    """Main demonstration function"""
    print("🎓 Vector Quantization Pipeline Demonstration")
    print("=" * 60)
    print()
    print("This demo showcases the configurable VQ pipeline with:")
    print("• Multiple modalities: Image, Text, Video")
    print("• Various encoders: CNN, ResNet, ViT, LSTM, Transformer, BERT, 3D CNN")
    print("• Different VQ methods: Basic VQ, VQ-VAE, RQ-VAE, RQ-K-means")
    print("• Semantic ID generation and codebook analysis")
    print()

    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Get all configurations
    all_configs = []
    all_configs.extend(create_image_pipeline_configs())
    all_configs.extend(create_text_pipeline_configs())
    all_configs.extend(create_video_pipeline_configs())

    results = []

    # Demonstrate each configuration
    for config_dict in all_configs:
        try:
            pipeline, outputs = demonstrate_pipeline(config_dict)

            if outputs:  # If successful
                # Analyze semantic usage
                sample_data = create_sample_data(config_dict['modality'], batch_size=8)
                usage_stats = analyze_semantic_usage(pipeline, sample_data)

                results.append({
                    'name': config_dict['name'],
                    'config': config_dict,
                    'success': True,
                    'usage_stats': usage_stats,
                    'model_info': pipeline.get_model_info()
                })
            else:
                results.append({
                    'name': config_dict['name'],
                    'config': config_dict,
                    'success': False
                })

        except Exception as e:
            print(f"❌ Failed to test {config_dict['name']}: {str(e)}")
            results.append({
                'name': config_dict['name'],
                'config': config_dict,
                'success': False,
                'error': str(e)
            })

    # Print summary
    print(f"\n🎯 SUMMARY")
    print("=" * 60)
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"Successful configurations: {successful}/{total}")
    print()

    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['name']}")
        if result['success'] and 'usage_stats' in result:
            stats = result['usage_stats']
            print(f"    Unique tokens: {stats['unique_tokens']}/{stats['total_tokens']}")
            print(f"    Usage ratio: {stats['usage_ratio']:.3f}")

    # Save configuration templates
    save_configuration_templates()

    # Create usage examples
    create_usage_examples()

    print(f"\n🎉 Demo completed! Check the following files:")
    print(f"   📁 pipeline_configs/ - Configuration templates")
    print(f"   📄 pipeline_usage_examples.md - Code examples")

    # Create a simple visualization of pipeline architectures
    create_architecture_visualization()


def create_architecture_visualization():
    """Create a simple text-based visualization of pipeline architectures"""

    visualization = """
# Vector Quantization Pipeline Architectures

## Pipeline Flow
```
Input Data → Encoder → Embeddings → Vector Quantizer → Semantic IDs + Codebook
                                        ↓
                                   Quantized Embeddings
```

## Modality-Specific Pipelines

### Image Pipelines
```
Image (3×H×W) → CNN/ResNet/ViT → Embeddings (N×D) → VQ → Semantic IDs
```

**Encoder Options:**
- CNN: Convolutional layers + Global pooling
- ResNet: Residual blocks + Skip connections
- ViT: Patch embedding + Transformer layers

### Text Pipelines
```
Tokens (N×L) → LSTM/Transformer/BERT → Embeddings (N×D) → VQ → Semantic IDs
```

**Encoder Options:**
- LSTM: Bidirectional LSTM + Attention (optional)
- Transformer: Self-attention + Position encoding
- BERT: Bidirectional transformer + CLS token

### Video Pipelines
```
Video (3×T×H×W) → 3D CNN/Video Transformer → Embeddings (N×D) → VQ → Semantic IDs
```

**Encoder Options:**
- 3D CNN: Spatiotemporal convolutions + 3D pooling
- Video Transformer: Frame-wise ViT + Temporal attention

## Vector Quantization Methods

### Basic VQ
```
Embeddings → Nearest Neighbor Search → Codebook Index → Quantized Embedding
```

### VQ-VAE
```
Embeddings → VQ Layer → Quantized → Decoder → Reconstruction
                ↓
           Semantic IDs
```

### RQ-VAE (Residual Quantization)
```
Embeddings → VQ₁ → Residual₁ → VQ₂ → Residual₂ → ... → VQₙ
                ↓              ↓                      ↓
           Semantic IDs₁  Semantic IDs₂         Semantic IDsₙ
```

### RQ-K-means
```
Data → K-means₁ → Residual₁ → K-means₂ → ... → K-meansₙ
         ↓                     ↓                  ↓
    Cluster IDs₁         Cluster IDs₂      Cluster IDsₙ
```

## Configuration Matrix

| Modality | Encoder | VQ Method | Use Case |
|----------|---------|-----------|----------|
| Image | CNN | Basic VQ | Simple image features |
| Image | ResNet | VQ-VAE | Robust image representations |
| Image | ViT | RQ-VAE | Fine-grained image semantics |
| Text | LSTM | Basic VQ | Sequential text features |
| Text | Transformer | RQ-K-means | Hierarchical text semantics |
| Text | BERT | VQ-VAE | Contextual text representations |
| Video | 3D CNN | Basic VQ | Spatiotemporal features |
| Video | Video Transformer | RQ-VAE | Complex video understanding |

## Semantic ID Properties

- **Basic VQ**: Single-level discrete tokens
- **VQ-VAE**: Single-level with reconstruction capability
- **RQ-VAE**: Multi-level hierarchical tokens
- **RQ-K-means**: Multi-stage clustering codes

The choice of encoder and VQ method affects:
- Semantic granularity
- Reconstruction quality
- Computational complexity
- Interpretability of representations
"""

    with open("pipeline_architectures.md", 'w') as f:
        f.write(visualization)

    print(f"   📄 pipeline_architectures.md - Architecture guide")


if __name__ == "__main__":
    main()