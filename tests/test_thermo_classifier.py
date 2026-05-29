"""Test Suite for Thermo Classifier"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thermo_classifier.models.hq_cnn import HQCNN, QuantumFeatureLayer, HybridBlock
from thermo_classifier.models.classifier import ThermalClassifier
from thermo_classifier.data.dataset import ThermalDataset
from thermo_classifier.data.transforms import ThermalTransforms
from thermo_classifier.utils.preprocessing import ThermalPreprocessor


class TestQuantumLayers:
    """Test quantum layer components"""
    
    def test_quantum_feature_layer(self):
        """Test quantum feature layer"""
        layer = QuantumFeatureLayer(n_qubits=4, n_layers=2, n_features=32)
        x = torch.randn(2, 32)
        out = layer(x)
        assert out.shape == (2, 4)
    
    def test_hybrid_block(self):
        """Test hybrid block"""
        block = HybridBlock(64, 128, use_quantum=False)
        x = torch.randn(2, 64, 32, 32)
        out = block(x)
        assert out.shape == (2, 128, 32, 32)


class TestHQCNN:
    """Test HQ-CNN model"""
    
    def test_hqcnn_medical(self):
        """Test medical domain model"""
        model = HQCNN(in_channels=1, num_classes=5, domain='medical', use_quantum=False)
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, 5)
    
    def test_hqcnn_industrial(self):
        """Test industrial domain model"""
        model = HQCNN(in_channels=1, num_classes=6, domain='industrial', use_quantum=False)
        x = torch.randn(2, 1, 224, 224)
        out = model(x)
        assert out.shape == (2, 6)


class TestThermalClassifier:
    """Test thermal classifier"""
    
    def test_classifier_init(self):
        """Test classifier initialization"""
        clf = ThermalClassifier(domain='medical', use_quantum=False)
        assert clf.domain == 'medical'
        assert clf.model is not None
    
    def test_classifier_predict(self):
        """Test prediction"""
        clf = ThermalClassifier(domain='medical', use_quantum=False)
        # Random image
        x = torch.randn(1, 224, 224)
        result = clf.predict(x)
        assert 'predicted_class' in result
        assert 'confidence' in result


class TestPreprocessing:
    """Test preprocessing utilities"""
    
    def test_normalize_thermal(self):
        """Test thermal normalization"""
        img = np.random.randn(100, 100) * 50 + 128
        normalized = ThermalPreprocessor.normalize_thermal(img, 'minmax')
        assert normalized.min() >= 0 and normalized.max() <= 1
    
    def test_detect_hotspots(self):
        """Test hotspot detection"""
        img = np.zeros((100, 100))
        img[40:60, 40:60] = 1.0  # Hotspot
        mask, hotspots = ThermalPreprocessor.detect_hotspots(img, 0.5)
        assert mask.sum() > 0


class TestTransforms:
    """Test data transforms"""
    
    def test_train_transforms(self):
        """Test training transforms"""
        transform = ThermalTransforms.get_train_transforms((224, 224))
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        transformed = transform(image=img)
        assert 'image' in transformed
    
    def test_val_transforms(self):
        """Test validation transforms"""
        transform = ThermalTransforms.get_val_transforms((224, 224))
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        transformed = transform(image=img)
        assert 'image' in transformed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
