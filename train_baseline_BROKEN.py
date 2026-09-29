#!/usr/bin/env python3
"""
Baseline ResNet Classifier for Breast Thermography
==================================================
Uses pre-trained ResNet50 for transfer learning on thermal images.
Includes t-SNE visualization, F1 tracking, and training graphs.
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# Dataset class
class ThermalDataset(torch.utils.data.Dataset):
    """Thermal image dataset"""
    
    CLASSES = ['benign', 'malignant', 'normal']
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        for cls_idx, cls_name in enumerate(self.CLASSES):
            cls_dir = self.root_dir / cls_name
            if cls_dir.exists():
                for img_path in cls_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), cls_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224):
    """Get data transforms for train/val/test"""
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(model_name='resnet50', num_classes=3, pretrained=True):
    """Create model with pre-trained weights
    
    Supported models:
    - resnet18, resnet34, resnet50, resnet101, resnet152
    - efficientnet_b0, efficientnet_b1, efficientnet_b2
    - vgg16, vgg19
    - densenet121, densenet169
    - mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
    """
    
    model_fn = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
    }
    
    if model_name not in model_fn:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_fn.keys())}")
    
    # Get model constructor
    if pretrained:
        weights = f'{model_name.upper()}_Weights.IMAGENET1K_V1' if 'efficientnet' not in model_name else f'{model_name.upper()}_Weights.IMAGENET1K_V1'
        try:
            model = model_fn[model_name](weights=models.__dict__[weights.replace('_', '_')]())
        except:
            model = model_fn[model_name](pretrained=True)
    else:
        model = model_fn[model_name](pretrained=False)
    
    # Modify classifier based on model type
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif 'efficientnet' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif 'vgg' in model_name:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif 'densenet' in model_name:
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif 'mobilenet' in model_name:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    return model


def get_feature_extractor(model):
    """Get feature extractor (before final FC layer)"""
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    return feature_extractor


def extract_features(model, dataloader, device):
    """Extract features from the model"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, target_labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu().numpy())
            labels.append(target_labels.numpy())
    
    return np.vstack(features), np.concatenate(labels)


def plot_tsne(features, labels, classes, output_path):
    """Plot t-SNE visualization"""
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    # Color palette
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, cls in enumerate(classes):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=colors[i], label=cls, alpha=0.7, s=100, edgecolors='white')
    
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title('t-SNE Visualization of Feature Embeddings', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to: {output_path}")


def plot_training_history(history, output_dir):
    """Plot training and validation graphs"""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training vs Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    plt.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Training vs Validation Metrics', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_dir}/")
    print(f"  - loss_curve.png")
    print(f"  - metrics_curve.png")


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    train_loss = running_loss / len(dataloader)
    train_acc = 100. * correct / total
    train_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return train_loss, train_acc, train_f1


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return val_loss, val_acc, val_f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='ResNet Baseline for Breast Thermography')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                               'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                               'vgg16', 'vgg19', 'densenet121', 'densenet169',
                               'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large'],
                       help='Model architecture (default: resnet50)')
    parser.add_argument('--output', type=str, default='outputs/baseline_resnet', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'model.pth'
    
    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"Available GPUs: {num_gpus}")
            if args.gpu >= num_gpus:
                print(f"Warning: GPU {args.gpu} not available, using GPU 0")
                args.gpu = 0
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    
    # Transforms
    train_transform, val_transform = get_transforms(args.image_size)
    
    # Datasets
    print(f"\nLoading dataset from: {args.data}")
    train_dataset = ThermalDataset(args.data, split='train', transform=train_transform)
    val_dataset = ThermalDataset(args.data, split='val', transform=val_transform)
    test_dataset = ThermalDataset(args.data, split='test', transform=val_transform)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    print(f"\nCreating {args.model} model...")
    model = create_model(model_name=args.model, num_classes=3, pretrained=args.pretrained)
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Training loop
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs (early stopping patience: {args.patience})...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, F1: {train_f1:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
            }, model_path)
            print(f"  -> Saved best model (val acc: {val_acc:.2f}%, F1: {val_f1:.2f}%)")
        else:
            patience_counter += 1
            print(f"  -> No improvement ({patience_counter}/{args.patience})")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation F1: {best_val_f1:.2f}%")
    
    # Save training history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    plot_training_history(history, output_dir)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1, preds, labels = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1 Score: {test_f1:.2f}%")
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(labels, preds, target_names=ThermalDataset.CLASSES))
    
    # Confusion matrix
    plot_confusion_matrix(labels, preds, ThermalDataset.CLASSES, output_dir / 'confusion_matrix.png')
    
    # t-SNE visualization
    print("\nGenerating t-SNE visualization...")
    feature_extractor = get_feature_extractor(model)
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # Extract features from test set
    test_features, test_labels = extract_features(feature_extractor, test_loader, device)
    test_features = test_features.reshape(test_features.shape[0], -1)
    
    plot_tsne(test_features, test_labels, ThermalDataset.CLASSES, output_dir / 'tsne.png')
    
    # Summary
    print("\n" + "="*60)
    print("OUTPUT FILES")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  - model.pth              (trained model)")
    print(f"  - history.json          (training history)")
    print(f"  - loss_curve.png        (training/validation loss)")
    print(f"  - metrics_curve.png     (accuracy & F1 curves)")
    print(f"  - confusion_matrix.png (confusion matrix)")
    print(f"  - tsne.png              (t-SNE visualization)")


if __name__ == '__main__':
    main()
