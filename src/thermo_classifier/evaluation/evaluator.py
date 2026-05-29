"""
Evaluation Module
==================
Model evaluation and metrics computation.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from thermo_classifier.models.classifier import ThermalClassifier
from thermo_classifier.data.dataset import ThermalDataset
from thermo_classifier.data.loader import ThermalDataLoader


class Evaluator:
    """Model evaluator for thermal image classification"""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        domain: str = 'medical',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.domain = domain
        self.device = device
        
        # Load model
        self.classifier = ThermalClassifier(
            domain=domain,
            model_type='hq-cnn',
            use_quantum=True,
            device=device,
            checkpoint=model_path
        )
    
    def evaluate(self, data_path: Union[str, Path]) -> Dict:
        """Evaluate model on test dataset"""
        
        # Create test loader
        test_loader = ThermalDataLoader.create(
            data_path=data_path,
            domain=self.domain,
            split='test',
            batch_size=32,
            shuffle=False
        )
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        self.classifier.model.eval()
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                
                # Get predictions
                logits = self.classifier.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return metrics
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict:
        """Compute evaluation metrics"""
        
        classes = ThermalClassifier.DOMAIN_CLASSES[self.domain]
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Per-class precision, recall, F1
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=classes, 
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class': {
                cls: {
                    'precision': per_class_precision[i],
                    'recall': per_class_recall[i],
                    'f1': per_class_f1[i]
                }
                for i, cls in enumerate(classes)
            },
            'classification_report': report
        }
    
    def evaluate_single(self, image_path: Union[str, Path]) -> Dict:
        """Evaluate a single image"""
        from PIL import Image
        
        # Load and preprocess
        image = np.array(Image.open(image_path).convert('L'))
        
        return self.classifier.predict(image)
    
    def cross_validate(
        self, 
        data_path: Union[str, Path], 
        n_folds: int = 5
    ) -> Dict:
        """Perform cross-validation"""
        # This would implement k-fold cross-validation
        raise NotImplementedError("Cross-validation coming soon")
