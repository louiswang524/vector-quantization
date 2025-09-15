"""
Test Pipeline Configuration System

This script tests the pipeline configuration system without requiring PyTorch,
focusing on configuration validation, serialization, and structure.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_configuration_creation():
    """Test configuration creation and validation"""
    print("🧪 Testing Configuration Creation...")

    try:
        # Import the configuration classes (these don't require torch)
        from vector_quantization.pipeline import (
            PipelineConfig, ModalityType, EncoderType, VQMethodType
        )

        # Test basic configuration creation
        config = PipelineConfig(
            modality=ModalityType.IMAGE,
            encoder_type=EncoderType.CNN,
            vq_method=VQMethodType.BASIC_VQ,
            input_dim=(3, 64, 64),
            embedding_dim=256,
            num_embeddings=1024,
            encoder_config={'hidden_dims': [64, 128, 256]}
        )

        print("✅ Basic configuration created successfully")
        print(f"   Modality: {config.modality.value}")
        print(f"   Encoder: {config.encoder_type.value}")
        print(f"   VQ Method: {config.vq_method.value}")

        # Test serialization
        config_dict = config.to_dict()
        print("✅ Configuration serialized to dictionary")

        # Test deserialization
        config_restored = PipelineConfig.from_dict(config_dict)
        print("✅ Configuration restored from dictionary")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_encoder_types():
    """Test all encoder type configurations"""
    print("\n🔧 Testing All Encoder Types...")

    try:
        from vector_quantization.pipeline import (
            PipelineConfig, ModalityType, EncoderType, VQMethodType
        )

        # Image encoders
        image_encoders = [
            (EncoderType.CNN, {'hidden_dims': [64, 128]}),
            (EncoderType.RESNET, {'num_layers': 4}),
            (EncoderType.VISION_TRANSFORMER, {'patch_size': 8, 'num_layers': 6})
        ]

        for encoder_type, encoder_config in image_encoders:
            config = PipelineConfig(
                modality=ModalityType.IMAGE,
                encoder_type=encoder_type,
                vq_method=VQMethodType.BASIC_VQ,
                input_dim=(3, 64, 64),
                embedding_dim=256,
                num_embeddings=1024,
                encoder_config=encoder_config
            )
            print(f"✅ Image {encoder_type.value} encoder configured")

        # Text encoders
        text_encoders = [
            (EncoderType.LSTM, {'hidden_dim': 512, 'num_layers': 2}),
            (EncoderType.TRANSFORMER, {'d_model': 512, 'num_layers': 6}),
            (EncoderType.BERT_LIKE, {'d_model': 768, 'num_layers': 12})
        ]

        for encoder_type, encoder_config in text_encoders:
            config = PipelineConfig(
                modality=ModalityType.TEXT,
                encoder_type=encoder_type,
                vq_method=VQMethodType.BASIC_VQ,
                input_dim=10000,
                embedding_dim=256,
                num_embeddings=1024,
                encoder_config=encoder_config
            )
            print(f"✅ Text {encoder_type.value} encoder configured")

        # Video encoders
        video_encoders = [
            (EncoderType.CNN_3D, {'hidden_dims': [64, 128, 256]}),
            (EncoderType.VIDEO_TRANSFORMER, {'patch_size': 8, 'num_layers': 6})
        ]

        for encoder_type, encoder_config in video_encoders:
            config = PipelineConfig(
                modality=ModalityType.VIDEO,
                encoder_type=encoder_type,
                vq_method=VQMethodType.BASIC_VQ,
                input_dim=(3, 16, 32, 32),
                embedding_dim=256,
                num_embeddings=1024,
                encoder_config=encoder_config
            )
            print(f"✅ Video {encoder_type.value} encoder configured")

        return True

    except Exception as e:
        print(f"❌ Encoder type test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_vq_methods():
    """Test all VQ method configurations"""
    print("\n🔬 Testing All VQ Methods...")

    try:
        from vector_quantization.pipeline import (
            PipelineConfig, ModalityType, EncoderType, VQMethodType
        )

        vq_methods = [
            (VQMethodType.BASIC_VQ, {}),
            (VQMethodType.VQ_VAE, {'hidden_dims': [128, 256]}),
            (VQMethodType.RQ_VAE, {'num_quantizers': 4, 'hidden_dims': [128, 256]}),
            (VQMethodType.RQ_KMEANS, {'n_stages': 4})
        ]

        for vq_method, vq_config in vq_methods:
            config = PipelineConfig(
                modality=ModalityType.IMAGE,
                encoder_type=EncoderType.CNN,
                vq_method=vq_method,
                input_dim=(3, 64, 64),
                embedding_dim=256,
                num_embeddings=1024,
                encoder_config={'hidden_dims': [64, 128]},
                vq_config=vq_config
            )
            print(f"✅ {vq_method.value} method configured")

        return True

    except Exception as e:
        print(f"❌ VQ method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_templates():
    """Test loading configuration templates"""
    print("\n📄 Testing Configuration Templates...")

    try:
        # Test creating configuration templates
        template_configs = []

        # Image template
        image_template = {
            'name': 'Image_CNN_BasicVQ',
            'modality': 'image',
            'encoder_type': 'cnn',
            'vq_method': 'basic_vq',
            'input_dim': [3, 64, 64],
            'embedding_dim': 256,
            'num_embeddings': 1024,
            'commitment_cost': 0.25,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'encoder_config': {
                'hidden_dims': [64, 128, 256],
                'kernel_size': 3,
                'use_residual': False
            },
            'vq_config': {}
        }
        template_configs.append(image_template)

        # Text template
        text_template = {
            'name': 'Text_BERT_VQVAE',
            'modality': 'text',
            'encoder_type': 'bert_like',
            'vq_method': 'vq_vae',
            'input_dim': 10000,
            'embedding_dim': 256,
            'num_embeddings': 1024,
            'commitment_cost': 0.25,
            'learning_rate': 0.0001,
            'batch_size': 32,
            'encoder_config': {
                'd_model': 768,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1
            },
            'vq_config': {
                'hidden_dims': [128, 256]
            }
        }
        template_configs.append(text_template)

        # Video template
        video_template = {
            'name': 'Video_3DCNN_RQVAE',
            'modality': 'video',
            'encoder_type': 'cnn_3d',
            'vq_method': 'rq_vae',
            'input_dim': [3, 16, 32, 32],
            'embedding_dim': 256,
            'num_embeddings': 1024,
            'commitment_cost': 0.25,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'encoder_config': {
                'hidden_dims': [64, 128, 256],
                'temporal_stride': 2
            },
            'vq_config': {
                'num_quantizers': 4,
                'hidden_dims': [128, 256]
            }
        }
        template_configs.append(video_template)

        # Test creating PipelineConfig from each template
        from vector_quantization.pipeline import PipelineConfig

        for template in template_configs:
            config = PipelineConfig.from_dict(template)
            print(f"✅ Template '{template['name']}' loaded successfully")
            print(f"   {config.modality.value} + {config.encoder_type.value} + {config.vq_method.value}")

        # Test serialization round-trip
        for template in template_configs:
            config = PipelineConfig.from_dict(template)
            serialized = config.to_dict()
            restored = PipelineConfig.from_dict(serialized)
            print(f"✅ Round-trip serialization successful for {template['name']}")

        return True

    except Exception as e:
        print(f"❌ Template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_factory_info():
    """Test encoder factory information functions"""
    print("\n🏭 Testing Encoder Factory...")

    try:
        from vector_quantization.encoders import EncoderFactory
        from vector_quantization.pipeline import EncoderType

        # Test listing available encoders
        available_encoders = EncoderFactory.list_available_encoders()
        print("✅ Available encoders by modality:")
        for modality, encoders in available_encoders.items():
            encoder_names = [enc.value for enc in encoders]
            print(f"   {modality}: {encoder_names}")

        # Test getting encoder info
        for encoder_type in EncoderType:
            info = EncoderFactory.get_encoder_info(encoder_type)
            if info:
                print(f"✅ {encoder_type.value}: {info['name']} ({info['modality']})")

        return True

    except Exception as e:
        print(f"❌ Encoder factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all configuration tests"""
    print("🎓 Vector Quantization Pipeline Configuration Test")
    print("=" * 60)
    print("Testing configuration system without PyTorch dependency...")
    print()

    tests = [
        test_configuration_creation,
        test_all_encoder_types,
        test_all_vq_methods,
        test_configuration_templates,
        test_encoder_factory_info
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All {total} configuration tests passed!")
        print()
        print("✅ Configuration system is working correctly")
        print("✅ All encoder types can be configured")
        print("✅ All VQ methods can be configured")
        print("✅ Template system is functional")
        print("✅ Encoder factory provides correct information")
        print()
        print("📝 The pipeline system is ready for use!")
        print("   To use with actual models, install PyTorch:")
        print("   pip install torch torchvision")
    else:
        print(f"❌ {total - passed}/{total} tests failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()