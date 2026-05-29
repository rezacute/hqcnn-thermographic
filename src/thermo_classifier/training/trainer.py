"""
Training Module
================
Training loop and optimization for thermal classifiers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Any
import json
from tqdm import tqdm
import numpy as np

from thermo_classifier.models.hq_cnn import HQCNN
from thermo_classifier.models.classifier import ThermalClassifier
from thermo_classifier.data.dataset import ThermalDataset
from thermo_classifier.data.loader import ThermalDataLoader


class Trainer:
    """Trainer for thermal image classification models"""
    
    def __init__(
        self,
        model: str = 'hq-cnn',
        domain: str = 'medical',
        quantum: bool = True,
        qubits: int = 8,
        device: str = 'cuda',
        lr: float = 1e-4,
        batch_size: int = 32,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ):
        self.model_name = model
        self.domain = domain
        self.quantum = quantum
        self.qubits = qubits
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        # Will be set during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
    def _build_model(self, num_classes: int):
        """Build model architecture"""
        if self.model_name == 'hq-cnn':
            self.model = HQCNN(
                in_channels=1,
                num_classes=num_classes,
                domain=self.domain,
                use_quantum=self.quantum
            )
        else:
            from thermo_classifier.models.hq_cnn import ClassicalCNN
            self.model = ClassicalCNN(
                in_channels=1,
                num_classes=num_classes,
                domain=self.domain
            )
        
        self.model.to(self.device)
        
    def _build_optimizer(self):
        """Build optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train(
        self,
        dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = 'outputs'
    ):
        """Train the model
        
        Args:
            dataset: Training dataset
            val_dataset: Optional validation dataset
            output_dir: Directory to save checkpoints
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_classes = len(dataset.classes) if hasattr(dataset, 'classes') else 5
        
        # Build model
        self._build_model(num_classes)
        self._build_optimizer()
        
        # Create data loaders
        train_loader = ThermalDataLoader.create(
            data_path='dummy',  # Already have dataset
            domain=self.domain,
            split='train',
            batch_size=self.batch_size
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(dataset, desc=f'Epoch {epoch+1}/{self.epochs}')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/train_total:.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_acc = 100. * train_correct / train_total
            
            # Validation
            if val_dataset:
                val_acc = self._validate(val_dataset)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save(output_dir / 'best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save(output_dir / f'model_epoch_{epoch+1}.pt')
        
        print(f"Training complete. Best val accuracy: {best_val_acc:.2f}%")
    
    def _validate(self, val_dataset: Dataset) -> float:
        """Validate on dataset"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_dataset:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def save(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'domain': self.domain,
            'model_name': self.model_name,
            'quantum': self.quantum,
        }, path)
    
    def load(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self._build_model(checkpoint.get('num_classes', 5))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._build_optimizer()
