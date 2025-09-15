# Installation Guide

## Quick Installation

### Method 1: Direct from GitHub (Recommended)
```bash
pip install git+https://github.com/louiswang524/vector-quantization.git
```

### Method 2: Clone and Install
```bash
git clone https://github.com/louiswang524/vector-quantization.git
cd vector-quantization
pip install -e .
```

## Verify Installation

After installation, test that everything works:

```bash
# Method 1: Use the built-in test command
vq-test

# Method 2: Quick import test
python -c "from vector_quantization import VQPipeline; print('✅ Installation successful!')"

# Method 3: Test specific components
python -c "
from vector_quantization import VectorQuantizer, VQVAE, RQVAE, RQKMeans
from vector_quantization import PipelineConfig, ModalityType, EncoderType, VQMethodType
print('✅ All components imported successfully!')
"
```

## Common Installation Issues

### Issue 1: "ModuleNotFoundError: No module named 'vector_quantization'"

**Cause**: Package not properly installed or installed in wrong environment.

**Solutions**:
1. **Check your Python environment**:
   ```bash
   which python
   which pip
   ```

2. **Reinstall the package**:
   ```bash
   pip uninstall vector-quantization-educational
   pip install git+https://github.com/louiswang524/vector-quantization.git
   ```

3. **Use virtual environment** (recommended):
   ```bash
   python -m venv vq_env
   source vq_env/bin/activate  # On Windows: vq_env\Scripts\activate
   pip install git+https://github.com/louiswang524/vector-quantization.git
   ```

4. **Check if installed in correct location**:
   ```bash
   pip list | grep vector-quantization
   ```

### Issue 2: "No module named 'torch'"

**Cause**: PyTorch not installed.

**Solution**:
```bash
# Install PyTorch (choose appropriate version for your system)
pip install torch torchvision

# Or install with the package
pip install git+https://github.com/louiswang524/vector-quantization.git
```

### Issue 3: Permission errors

**Cause**: Installing to system Python without proper permissions.

**Solutions**:
1. **Use virtual environment** (recommended):
   ```bash
   python -m venv vq_env
   source vq_env/bin/activate
   pip install git+https://github.com/louiswang524/vector-quantization.git
   ```

2. **User installation**:
   ```bash
   pip install --user git+https://github.com/louiswang524/vector-quantization.git
   ```

### Issue 4: "externally-managed-environment" error

**Cause**: System protection on newer Linux distributions.

**Solution**: Use virtual environment:
```bash
python -m venv vq_env
source vq_env/bin/activate
pip install git+https://github.com/louiswang524/vector-quantization.git
```

### Issue 5: Import works in source directory but not elsewhere

**Cause**: Python is finding the source code instead of the installed package.

**Solutions**:
1. **Test outside source directory**:
   ```bash
   cd ~  # Go to home directory
   python -c "from vector_quantization import VQPipeline; print('✅ Works!')"
   ```

2. **Reinstall properly**:
   ```bash
   cd /path/to/vector-quantization
   pip install -e .
   ```

## Development Installation

For development and contributing:

```bash
git clone https://github.com/louiswang524/vector-quantization.git
cd vector-quantization
pip install -e ".[dev]"
```

This installs additional development dependencies for testing and linting.

## Platform-Specific Instructions

### Windows
```bash
# Use PowerShell or Command Prompt
python -m venv vq_env
vq_env\Scripts\activate
pip install git+https://github.com/louiswang524/vector-quantization.git
```

### macOS
```bash
# Install Xcode command line tools if needed
xcode-select --install

python3 -m venv vq_env
source vq_env/bin/activate
pip install git+https://github.com/louiswang524/vector-quantization.git
```

### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-venv python3-pip

python3 -m venv vq_env
source vq_env/bin/activate
pip install git+https://github.com/louiswang524/vector-quantization.git
```

## Docker Installation

For a completely isolated environment:

```dockerfile
FROM python:3.9-slim

RUN pip install git+https://github.com/louiswang524/vector-quantization.git

CMD ["python", "-c", "from vector_quantization import VQPipeline; print('✅ Docker installation successful!')"]
```

## Conda Installation

If you prefer conda:

```bash
conda create -n vq_env python=3.9
conda activate vq_env
pip install git+https://github.com/louiswang524/vector-quantization.git
```

## Testing Your Installation

### Basic Test
```python
from vector_quantization import VQPipeline, PipelineConfig
from vector_quantization import ModalityType, EncoderType, VQMethodType

# Create a simple configuration
config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.CNN,
    vq_method=VQMethodType.BASIC_VQ,
    input_dim=(3, 64, 64),
    embedding_dim=256,
    num_embeddings=1024,
    encoder_config={'hidden_dims': [64, 128]}
)

print("✅ Configuration created successfully!")
print(f"Pipeline: {config.modality.value} + {config.encoder_type.value} + {config.vq_method.value}")
```

### Full Test with PyTorch
```python
import torch
from vector_quantization import VQPipeline, PipelineConfig
from vector_quantization import ModalityType, EncoderType, VQMethodType

# Create configuration
config = PipelineConfig(
    modality=ModalityType.IMAGE,
    encoder_type=EncoderType.CNN,
    vq_method=VQMethodType.BASIC_VQ,
    input_dim=(3, 64, 64),
    embedding_dim=256,
    num_embeddings=1024,
    encoder_config={'hidden_dims': [64, 128]}
)

# Initialize pipeline
pipeline = VQPipeline(config)

# Test with sample data
sample_images = torch.randn(4, 3, 64, 64)
outputs = pipeline(sample_images)

print("✅ Full pipeline test successful!")
print(f"Input shape: {sample_images.shape}")
print(f"Output semantic IDs shape: {outputs['semantic_ids'].shape}")
```

## Getting Help

If you still have issues:

1. **Check the GitHub Issues**: https://github.com/louiswang524/vector-quantization/issues
2. **Run the diagnostic command**: `vq-test`
3. **Provide the following information when reporting issues**:
   - Python version: `python --version`
   - Pip version: `pip --version`
   - Operating system
   - Error message (full traceback)
   - Installation method used

## Uninstallation

To remove the package:

```bash
pip uninstall vector-quantization-educational
```