"""
Inference Module
=================
Model inference for thermal image classification.
"""

import torch
from pathlib import Path
from typing import Union, Dict, List
import numpy as np
from PIL import Image

from thermo_classifier.models.classifier import ThermalClassifier
from thermo_classifier.data.transforms import ThermalTransforms


class Predictor:
    """Inference predictor for thermal images"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        domain: str = 'medical',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.domain = domain
        self.device = device
        self.transform = ThermalTransforms.get_inference_transforms()
        
        # Load model
        self.classifier = ThermalClassifier(
            domain=domain,
            model_type='hq-cnn',
            use_quantum=True,
            device=device,
            checkpoint=model_path
        )
    
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """Predict on a single image"""
        # Load and preprocess
        image = np.array(Image.open(image_path).convert('L'))
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)
        
        # Predict
        result = self.classifier.predict(image_tensor.squeeze(0))
        
        return result
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict]:
        """Predict on multiple images"""
        results = []
        for img_path in image_paths:
            results.append(self.predict(img_path))
        return results
    
    def predict_from_array(self, image: np.ndarray) -> Dict:
        """Predict from numpy array"""
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # Add channel dim
        
        transformed = self.transform(image=image.squeeze())
        image_tensor = transformed['image']
        
        return self.classifier.predict(image_tensor)
