"""Pytest configuration"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device():
    """Get compute device"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def sample_image():
    """Create sample thermal image"""
    import numpy as np
    return np.random.randint(0, 256, (224, 224), dtype=np.uint8)


@pytest.fixture
def sample_batch():
    """Create sample batch"""
    return torch.randn(4, 1, 224, 224)
