#!/usr/bin/env python3
"""
Hybrid Quantum-Classical CNN (HQ-CNN) for Breast Thermography Classification
================================================================================
Uses quantum-inspired parameterized circuits with classical simulation for training.
This approach mimics quantum computing behavior while being practical for training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Quantum-Inspired Layer (Parametrized Quantum Circuit Simulation)
# =============================================================================

class QuantumInspiredLayer(nn.Module):
    """
    Quantum-inspired layer that simulates parameterized quantum circuits.
    Uses trainable rotational transformations that mimic quantum gate operations.
    """
    
    def __init__(self, n_qubits=8, n_layers=2, input_dim=512, n_heads=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Project input to qubit dimension
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Light self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=n_qubits, 
            num_heads=n_heads, 
            dropout=0.05,
            batch_first=True
        )
        
        # Light entanglement
        self.entangle_weights = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits) * 0.1)
        
        # Simple output projection
        self.output_proj = nn.Sequential(
            nn.Linear(n_qubits * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        x = torch.tanh(self.input_projection(x))
        x_attn = x.unsqueeze(1)
        
        for layer in range(self.n_layers):
            attn_out, _ = self.attention(x_attn, x_attn, x_attn)
            x_attn = x_attn + attn_out * 0.3
            entangle_matrix = torch.tanh(self.entangle_weights[layer])
            entangled = torch.matmul(x_attn.squeeze(1), entangle_matrix)
            x = x + entangled * 0.03
        
        x = x_attn.squeeze(1)
        features = torch.cat([x, torch.abs(x)], dim=1)
        out = self.output_proj(features)
        return out


class HybridQuantumCNN(nn.Module):
    """Hybrid Quantum-Classical CNN architecture"""
    
    def __init__(self, num_classes=3, backbone='resnet18', use_quantum=True, n_qubits=8, 
                 gaussian_noise=True, n_quantum_layers=2, dropout_rate=0.5):
        super().__init__()
        self.use_quantum = use_quantum
        self.gaussian_noise = gaussian_noise
        self.n_quantum_layers = n_quantum_layers
        
        # Classical backbone (feature extractor)
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        if use_quantum:
            # Quantum-inspired layer
            self.quantum_layer = QuantumInspiredLayer(
                n_qubits=n_qubits,
                n_layers=n_quantum_layers,
                input_dim=feature_dim
            )
            # Combine classical + quantum features
            combined_dim = feature_dim + 32
        else:
            combined_dim = feature_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract classical features
        classical_features = self.backbone(x)
        classical_features = classical_features.view(classical_features.size(0), -1)
        
        # Add Gaussian noise during training
        if self.training and self.gaussian_noise:
            noise = torch.randn_like(classical_features) * 0.01
            classical_features = classical_features + noise
        
        if self.use_quantum:
            # Get quantum-enhanced features
            quantum_features = self.quantum_layer(classical_features)
            # Concatenate
            combined = torch.cat([classical_features, quantum_features], dim=1)
        else:
            combined = classical_features
        
        # Classify
        output = self.classifier(combined)
        return output


# =============================================================================
# Data Loading
# =============================================================================

class ThermographyDataset(torch.utils.data.Dataset):
    """Breast Thermography Dataset"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = ['benign', 'malignant', 'normal']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train'):
    """Get data transforms with augmentation for training"""
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='macro') * 100
    
    return running_loss / total, 100. * correct / total, f1


# =============================================================================
# Main Training
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/breast-thermography',
                       help='Dataset path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--no-quantum', action='store_true',
                       help='Disable quantum layer (classical only)')
    parser.add_argument('--n-qubits', type=int, default=8)
    parser.add_argument('--n-quantum-layers', type=int, default=2,
                       help='Number of quantum layers')
    parser.add_argument('--gaussian-noise', action='store_true', default=True,
                       help='Add Gaussian noise layer')
    parser.add_argument('--attention-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay')
    parser.add_argument('--output', type=str, default='outputs/hq_cnn')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load datasets
    train_dataset = ThermographyDataset(args.data, 'train', get_transforms('train'))
    val_dataset = ThermographyDataset(args.data, 'val', get_transforms('val'))
    test_dataset = ThermographyDataset(args.data, 'test', get_transforms('test'))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"\nDataset: {args.data}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = HybridQuantumCNN(
        num_classes=3,
        backbone=args.backbone,
        use_quantum=not args.no_quantum,
        n_qubits=args.n_qubits,
        gaussian_noise=args.gaussian_noise,
        n_quantum_layers=args.n_quantum_layers,
        dropout_rate=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Hybrid Quantum-CNN ({args.backbone})")
    print(f"Quantum enabled: {not args.no_quantum}")
    print(f"Total parameters: {total_params:,}")
    
    # Class weights for imbalanced data
    class_counts = [50, 17, 52]  # benign, malignant, normal
    class_weights = torch.tensor([max(class_counts) / c for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Different learning rates for backbone vs head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # Warmup + Cosine decay scheduler
    def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return min_lr / args.lr + (1 - min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = warmup_cosine_scheduler(optimizer, warmup_epochs=5, total_epochs=args.epochs)
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train: {train_acc:.1f}% | "
              f"Val: {val_acc:.1f}% | F1: {val_f1:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.output}/model.pth")
            print(f"  -> Saved best model (val acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f"{args.output}/model.pth"))
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1 Score: {test_f1:.2f}%")
    
    # Save history
    with open(f"{args.output}/history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    return test_acc, test_f1


if __name__ == '__main__':
    main()
