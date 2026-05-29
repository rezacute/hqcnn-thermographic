"""
Thermal Classifier Wrapper
===========================
High-level interface for thermal image classification.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Union
import numpy as np

from thermo_classifier.models.hq_cnn import HQCNN, ClassicalCNN


class ThermalClassifier:
    """High-level thermal image classifier
    
    Provides unified interface for both quantum-enhanced and classical models.
    
    Args:
        domain: 'medical' or 'industrial'
        model_type: 'hq-cnn' or 'classical'
        use_quantum: Whether to use quantum enhancement
        device: 'cuda' or 'cpu'
        checkpoint: Optional path to model checkpoint
    """
    
    DOMAIN_CLASSES = {
        'medical': ['normal', 'benign', 'malignant'],
        'industrial': ['normal', 'overheating', 'electrical_fault', 'structural_defect', 'thermal_bridge', 'moisture']
    }
    
    def __init__(
        self,
        domain: str = 'medical',
        model_type: str = 'hq-cnn',
        use_quantum: bool = True,
        device: Optional[str] = None,
        checkpoint: Optional[Union[str, Path]] = None
    ):
        self.domain = domain
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get number of classes for domain
        num_classes = len(self.DOMAIN_CLASSES[domain])
        
        # Initialize model
        if model_type == 'hq-cnn':
            self.model = HQCNN(
                in_channels=1,
                num_classes=num_classes,
                domain=domain,
                use_quantum=use_quantum
            )
        else:
            self.model = ClassicalCNN(
                in_channels=1,
                num_classes=num_classes,
                domain=domain
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint if provided
        if checkpoint:
            self.load(checkpoint)
    
    def predict(
        self, 
        image: Union[np.ndarray, torch.Tensor],
        return_probs: bool = True
    ) -> Dict:
        """Predict class for a single image
        
        Args:
            image: Input image (H, W) or (1, H, W) or (B, 1, H, W)
            return_probs: Whether to return class probabilities
            
        Returns:
            Dictionary with predictions, probabilities, and class labels
        """
        # Preprocess
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif image.dim() == 3:
            image = image.unsqueeze(1)  # (B, 1, H, W)
        
        image = image.to(self.device).float()
        
        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Predict
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)
        
        # Get top predictions
        top_probs, top_indices = probs.max(dim=1)
        
        classes = self.DOMAIN_CLASSES[self.domain]
        
        results = {
            'predicted_class': classes[top_indices.item()],
            'confidence': top_probs.item()
        }
        
        if return_probs:
            results['probabilities'] = {
                cls: prob.item() 
                for cls, prob in zip(classes, probs[0])
            }
            results['all_predictions'] = [
                (cls, prob.item()) 
                for cls, prob in zip(classes, probs[0])
            ]
        
        return results
    
    def predict_batch(self, images: torch.Tensor) -> List[Dict]:
        """Predict on a batch of images"""
        images = images.to(self.device)
        with torch.no_grad():
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
        
        classes = self.DOMAIN_CLASSES[self.domain]
        results = []
        
        for i in range(len(images)):
            top_prob, top_idx = probs[i].max(0)
            results.append({
                'predicted_class': classes[top_idx.item()],
                'confidence': top_prob.item(),
                'probabilities': {
                    cls: prob.item() 
                    for cls, prob in zip(classes, probs[i])
                }
            })
        
        return results
    
    def save(self, path: Union[str, Path]):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'domain': self.domain,
            'model_type': self.model_type,
        }, path)
    
    def load(self, path: Union[str, Path]):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Load pretrained model"""
        # This would load from a model hub in production
        raise NotImplementedError("Pretrained models coming soon")
