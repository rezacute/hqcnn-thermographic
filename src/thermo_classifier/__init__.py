"""
Thermo Classifier - Thermal Image Classification Package
==========================================================
A hybrid quantum-classical CNN framework for thermographic image analysis.

Domains:
    - Medical: Breast cancer detection, inflammation analysis, vascular imaging
    - Industrial: Electrical fault detection, building inspection, machinery monitoring
"""

__version__ = "1.0.0"
__author__ = "QuantiLog Research Team"

from thermo_classifier import models, utils, data, api
from thermo_classifier.models.classifier import ThermalClassifier

__all__ = [
    "ThermalClassifier",
    "models",
    "utils", 
    "data",
    "api",
]
