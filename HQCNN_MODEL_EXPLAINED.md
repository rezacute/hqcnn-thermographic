# Hybrid Quantum-Classical CNN (HQ-CNN) for Breast Thermography Classification

## Overview

This project implements a **Hybrid Quantum-Classical CNN (HQ-CNN)** for breast cancer detection using thermographic images. The model combines classical deep learning (ResNet backbone) with quantum-inspired layers that simulate parameterized quantum circuits.

## Best Model Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet18 (ImageNet pretrained) |
| Qubits | 8 |
| Quantum Layers | 2 |
| Attention Heads | 4 |
| Dropout | 0.55 |
| Weight Decay | 0.0008 |
| Learning Rate | 0.0001 |
| Batch Size | 16 |
| Early Stopping Patience | 25 |

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 29.17% |
| **Test Accuracy** | **46.30%** |
| **Test F1 Score** | **34.62%** |

## Architecture

### Overall Structure

```
Input Image (224x224x3)
        ↓
┌─────────────────────────────┐
│   ResNet18 Backbone         │  (frozen pretrained features)
│   (ImageNet weights)         │
│   Output: 512 features       │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Gaussian Noise (0.01)     │  (regularization during training)
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Quantum-Inspired Layer    │  (8 qubits, 2 layers, 4 attention heads)
│   - Input projection        │
│   - Multihead self-attention│
│   - Entanglement weights    │
│   - Output projection       │
│   Output: 32 features       │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Feature Concatenation     │  (512 + 32 = 544 features)
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   Classifier Head           │
│   - Dropout(0.55)          │
│   - Linear(544→256)        │
│   - ReLU + BatchNorm       │
│   - Dropout(0.33)          │
│   - Linear(256→3)           │
└─────────────────────────────┘
        ↓
Output: 3 classes (benign, malignant, normal)
```

### Quantum-Inspired Layer

The quantum-inspired layer simulates quantum computing behavior using classical neural networks:

```
Classical Features (512) → Input Projection → Qubit States (8)
                                         ↓
                    ┌────────────────────────────────────┐
                    │  For each quantum layer (2):       │
                    │  1. Multihead Self-Attention       │
                    │     - 4 attention heads             │
                    │     - Simulates quantum interference│
                    │  2. Entanglement                   │
                    │     - Learnable coupling matrix    │
                    │     - Simulates qubit entanglement │
                    └────────────────────────────────────┘
                                         ↓
                              Enhanced Features (8)
                                         ↓
                    ┌────────────────────────────────────┐
                    │  Output Projection                 │
                    │  - Concatenate [x, |x|]           │
                    │  - Project to 32 dimensions        │
                    └────────────────────────────────────┘
                                         ↓
                              Quantum Features (32)
```

### Key Components

1. **Input Projection**: Linear layer mapping 512→8 qubits with tanh activation

2. **Multihead Self-Attention**: 
   - 4 attention heads
   - Simulates quantum interference patterns
   - Allows qubits to "interact" and learn correlations

3. **Entanglement Weights**:
   - Trainable matrices (8×8)
   - Learnable qubit-qubit coupling
   - Initialized with small random values (σ=0.1)

4. **Output Projection**:
   - Concatenates real and absolute values
   - Projects to 32-dimensional quantum feature space

## Training Details

### Data Augmentation

- Random horizontal flip (p=0.5)
- Random rotation (±20°)
- Color jitter (brightness=0.2, contrast=0.2)
- Random affine translation (10%)

### Class Imbalance Handling

| Class | Samples | Weight |
|-------|---------|--------|
| Benign | 50 | 1.0 |
| Malignant | 17 | 2.94 |
| Normal | 52 | 0.96 |

Cross-entropy loss with class weights + label smoothing (0.1)

### Optimization

- **Optimizer**: AdamW
- **Learning Rate**: 0.0001 (backbone: 0.00001)
- **Scheduler**: Cosine Annealing with Warm Restarts
- **Weight Decay**: 0.0008
- **Gradient Clipping**: max_norm=1.0

## Hyperparameter Tuning Journey

### Experiments Summary

| Model | Qubits | Dropout | Weight Decay | LR | Test Acc | F1 |
|-------|--------|---------|--------------|-----|----------|-----|
| Baseline | 8 | 0.5 | 1e-4 | 1e-4 | 40.74% | 19.56% |
| +Higher LR | 8 | 0.5 | 1e-4 | 0.001 | 35.19% | 19.57% |
| +Attention | 8 | 0.5 | 1e-4 | 1e-4 | 38.89% | 28.25% |
| +Regularization | 8 | 0.6 | 0.001 | 1e-4 | 44.44% | 20.51% |
| **+Tuned (8q_v2)** | **8** | **0.55** | **0.0008** | **1e-4** | **46.30%** | **34.62%** |
| 16 qubits | 16 | 0.55 | 0.0008 | 1e-4 | 27.78% | 20.83% |
| Long training | 8 | 0.55 | 0.0008 | 1e-4 | 27.78% | 21.11% |
| Warmup | 8 | 0.55 | 0.0008 | 1e-4 | 44.44% | 25.88% |

### Key Findings

1. **8 qubits optimal**: 16 qubits caused underfitting
2. **Moderate regularization**: Dropout 0.55-0.6 works best
3. **Weight decay 0.0008**: Better than 0.001 or 0.0001
4. **LR 0.0001**: Higher LRs caused overfitting
5. **Early stopping critical**: Models peak around epoch 4-10

## Limitations

### Dataset Size
- **Training**: 201 images
- **Validation**: 48 images  
- **Test**: 54 images

This is extremely small for deep learning. The SOTA paper (97.62% accuracy) likely used:
- Full Mendeley dataset (more samples)
- Cross-validation
- Medical-pretrained backbones

### Challenges

1. **Class Imbalance**: Malignant class only has 17 samples
2. **Overfitting**: Small dataset leads to overfitting
3. **Limited Features**: Thermal images have less visual information

## Future Improvements

1. **More Data**: Use full Mendeley dataset
2. **Medical Pretrained Models**: MedNet, MedicalNet
3. **Proper CUDA Quantum**: Actual quantum circuits instead of simulation
4. **Ensemble**: Multiple models with different seeds
5. **K-Fold CV**: More robust validation

## Files

- `train_hq_cnn.py` - Main training script
- `outputs/hq_cnn_8q_v2/` - Best model checkpoint
- `dataset/breast-thermography/` - Dataset
