"""
Setup script for Vector Quantization Educational Repository
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vector-quantization-educational",
    version="1.2.0",
    author="Educational Repository",
    author_email="",
    description="Educational implementation of Vector Quantization, VQ-VAE, RQ-VAE, and RQ-K-means with configurable pipeline system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/louiswang524/vector-quantization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="vector-quantization, vq-vae, machine-learning, representation-learning, educational",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.1.0",
        "tqdm>=4.64.0",
        "Pillow>=9.0.0",
        "tensorboard>=2.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "vq-test=vector_quantization.test_installation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)