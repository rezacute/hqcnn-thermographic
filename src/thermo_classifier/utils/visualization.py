"""
Visualization Utilities
========================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
from pathlib import Path
import cv2


class ThermalVisualizer:
    """Visualization utilities for thermal images and results"""
    
    @staticmethod
    def plot_thermal_comparison(
        original: np.ndarray,
        processed: np.ndarray,
        colormap: str = 'jet',
        save_path: Optional[Path] = None
    ):
        """Plot original vs processed thermal image"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        axes[1].imshow(processed, cmap=colormap)
        axes[1].set_title('Processed')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        save_path: Optional[Path] = None,
        normalize: bool = True
    ):
        """Plot confusion matrix"""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_training_history(
        history: Dict,
        save_path: Optional[Path] = None
    ):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.get('train_loss', []), label='Train')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(history.get('train_acc', []), label='Train')
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_prediction_heatmap(
        image: np.ndarray,
        predictions: Dict[str, float],
        save_path: Optional[Path] = None
    ):
        """Plot image with prediction probabilities"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Show thermal image
        im = ax.imshow(image, cmap='jet')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Temperature')
        
        # Add prediction text
        pred_text = '\n'.join([
            f"{k}: {v:.2%}" 
            for k, v in sorted(predictions.items(), key=lambda x: -x[1])
        ])
        
        ax.text(
            0.02, 0.98, pred_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def create_classification_report_image(
        metrics: Dict,
        save_path: Optional[Path] = None
    ):
        """Create visualization of classification metrics"""
        classes = list(metrics.get('per_class', {}).keys())
        
        precision = [metrics['per_class'][c]['precision'] for c in classes]
        recall = [metrics['per_class'][c]['recall'] for c in classes]
        f1 = [metrics['per_class'][c]['f1'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1 Score')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
