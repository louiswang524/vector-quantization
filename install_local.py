#!/usr/bin/env python3
"""
Local Installation Script for Vector Quantization Educational Package

This script helps set up the package for local development and testing
without requiring system-wide installation.
"""

import sys
import os
from pathlib import Path

def setup_local_package():
    """
    Add the source directory to Python path for local development
    """
    # Get the repository root directory
    repo_root = Path(__file__).parent.absolute()
    src_path = repo_root / "src"

    print(f"Repository root: {repo_root}")
    print(f"Source path: {src_path}")

    # Check if source directory exists
    if not src_path.exists():
        print("❌ Error: src directory not found!")
        return False

    # Check if the vector_quantization module exists
    module_path = src_path / "vector_quantization"
    if not module_path.exists():
        print("❌ Error: vector_quantization module not found!")
        return False

    # Add to Python path
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
        print(f"✅ Added {src_str} to Python path")
    else:
        print(f"ℹ️ {src_str} already in Python path")

    # Test import
    try:
        from vector_quantization import VectorQuantizer, VQVAE, RQVAE, RQKMeans
        print("✅ Successfully imported all modules!")
        print("Available classes:")
        print("  - VectorQuantizer: Basic vector quantization")
        print("  - VQVAE: Vector Quantized Variational Autoencoder")
        print("  - RQVAE: Residual Quantized VAE")
        print("  - RQKMeans: Residual Quantized K-means")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def create_setup_snippet():
    """
    Create a code snippet that users can add to their notebooks/scripts
    """
    repo_root = Path(__file__).parent.absolute()
    src_path = repo_root / "src"

    snippet = f'''
# Add this to the top of your Python script or notebook:
import sys
import os
sys.path.insert(0, r"{src_path}")

# Now you can import the modules:
from vector_quantization import VectorQuantizer, VQVAE, RQVAE, RQKMeans
'''

    snippet_file = repo_root / "setup_snippet.py"
    with open(snippet_file, 'w') as f:
        f.write(snippet.strip())

    print(f"✅ Created setup snippet at: {snippet_file}")
    print("You can copy-paste this code into your scripts or notebooks.")

if __name__ == "__main__":
    print("🔧 Setting up Vector Quantization Educational Package locally...")
    print("=" * 60)

    success = setup_local_package()

    if success:
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("\nYou can now use the package in this Python session.")
        print("For other sessions, either:")
        print("1. Run this script again, or")
        print("2. Use the setup snippet created below")

        create_setup_snippet()

    else:
        print("\n" + "=" * 60)
        print("❌ Setup failed. Please check the error messages above.")

    print("\nFor examples, run:")
    print("  cd examples")
    print("  python basic_vq_demo.py")