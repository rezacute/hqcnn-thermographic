#!/usr/bin/env python3
"""
Thermo Classifier Setup
=======================
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="thermo-classifier",
    version="1.0.0",
    author="QuantiLog Research Team",
    author_email="contact@quantilog.org",
    description="Hybrid quantum-classical CNN for thermal image classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantilog/hqcnn-thermographic",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "typer>=0.9.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "python-multipart>=0.0.6",
        "jinja2>=3.1.0",
        "plotly>=5.18.0",
        "orjson>=3.9.0",
        "h5py>=3.10.0",
        "joblib>=1.3.0",
        "onnx>=1.15.0",
        "onnxruntime-gpu>=1.16.0",
    ],
    extras_require={
        "quantum": [
            "cuda-quantum",
            "qiskit>=0.45.0",
            "qiskit-machine-learning>=0.7.0",
            "pennylane>=0.34.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "thermo-classifier=thermo_classifier_cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
