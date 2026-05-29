"""
Models Package - Neural Network Architectures
==============================================
Contains hybrid quantum-classical CNN models for thermal image classification.
"""

from thermo_classifier.models.hq_cnn import HQCNN
from thermo_classifier.models.quantum_layer import QuantumFeatureLayer
from thermo_classifier.models.classifier import ThermalClassifier

__all__ = [
    "HQCNN",
    "QuantumFeatureLayer", 
    "ThermalClassifier",
]
