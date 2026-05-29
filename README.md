# HQ-CNN: Hybrid Quantum-Classical CNN for Breast Thermography Classification

> **Comparative Analysis of Quantum-Inspired, Parameterized Quantum Circuit, and Classical Approaches for Breast Thermography Classification Using CUDA Quantum**
>
> Riza Alaudin Syah, Haza Nuzly Bin Abdul Hamed — *Universiti Teknologi Malaysia*

The first hybrid quantum-classical CNN applied to breast infrared thermography, and the first three-way comparison of classical, quantum-inspired, and real parameterized quantum circuit (PQC) approaches on a medical imaging task. Evaluated on the [DMR-IR benchmark dataset](https://visual.ic.uff.br/dmi) (1,522 breast thermograms).

## Key Results

| Model | Test Accuracy | F1 Score | AUC | Training Time |
|-------|:------------:|:--------:|:---:|:------------:|
| Classical ResNet18 | **92.08%** | 92.03% | 0.97 | 5 min |
| HQ-CNN (Quantum-Inspired) | 91.25% | 91.18% | 0.96 | 10 min |
| CUDA Quantum PQC (8-qubit) | 90.42% | 90.38% | 0.94 | 30 min |

All three models are **statistically equivalent** within a 1.66% margin on 240 test images (within the 95% confidence interval of ±3.4%).

## Contributions

1. **First hybrid quantum CNN for breast infrared thermography** on the DMR-IR benchmark
2. **First three-way comparison** of classical, quantum-inspired, and real PQC approaches sharing the same backbone and classifier
3. **Three critical PQC training fixes** yielding a cumulative **54.49% accuracy improvement** (from 35.93% to 90.42%):
   - **π/4 encoding scaling** — prevents rotation saturation and barren plateaus
   - **BatchNorm** — resolves feature scale mismatch between quantum measurements ([-1,1]) and classical features
   - **Calibrated dropout** — regularization tuned for quantum parameter counts
4. **Qubit-scaling analysis** showing optimal expressivity at 8 qubits (~23 training samples per quantum parameter)
5. **Evidence that quantum-inspired layers match real PQCs** at 3× lower computational cost

## Architecture

All three variants share a common **ResNet18 backbone** (pretrained on ImageNet) and **classifier head**:

```
Input (224×224 thermal image)
    │
    ▼
┌─────────────────────┐
│  ResNet18 Backbone   │  → h_backbone ∈ R^512
│  (pretrained)        │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────────────────────┐
│Classical│  │  Quantum Layer          │  → h_quantum ∈ R^32
│(skip)  │  │  (Quantum-Inspired OR   │
│        │  │   CUDA Quantum PQC)     │
└───┬────┘  └──────────┬─────────────┘
    │                  │
    └────────┬─────────┘
             ▼
    ┌─────────────────┐
    │  Classifier Head │  Linear(544→256) → ReLU → BN → Dropout → Linear(256→2)
    │  y = f([h_bb;    │
    │       h_quantum])│
    └─────────────────┘
             │
             ▼
      Normal / Malignant
```

### Quantum-Inspired Layer (HQ-CNN)
Projects 512-dim features to 8 simulated qubits via tanh, applies **4-head self-attention** (×2 layers) with learnable coupling matrices, and outputs `[x; |x|]` projected to 32 dimensions.

### CUDA Quantum PQC
Encodes features as **π/4 · tanh(Wh + b)** rotation angles on 8 qubits, applies **2 variational layers** (RY-RZ single-qubit rotations + CNOT entangling ladder with ring closure), measures **15 observables** (8 Pauli-Z + 7 ZZ correlations), and projects through Linear(15→32) + BatchNorm.

## Critical PQC Training Fixes

The initial CUDA Quantum PQC achieved only **35.93%** test accuracy. Three fixes were essential:

| Fix Applied | Test Accuracy | Δ |
|-------------|:------------:|:--:|
| Baseline (broken) | 35.93% | — |
| + Encoding (π/4) + BatchNorm | 81.81% | +45.88% |
| + Regularization (dropout 0.6) | 87.92% | +6.11% |
| + 8 qubits (from 4) | 90.42% | +2.50% |
| **Total improvement** | | **+54.49%** |

## Dataset

**DMR-IR** (Database for Mastology Research with Infrared Images) — 1,522 breast infrared thermograms from the Visual Lab, Universidade Federal Fluminense, Brazil.

| Split | Images |
|-------|:------:|
| Train | 1,090 |
| Validation | 192 |
| Test | 240 |

**Classes:** Normal, Malignant (binary classification)

**Preprocessing:** Resize to 224×224, ImageNet normalization, standard augmentation (flip, rotation, color jitter, affine). No breast segmentation or GAN augmentation — ensuring a fair controlled comparison.

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate hqcnn-thermographic

# Install package
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Train the HQ-CNN model
thermo-classifier train --domain medical --data /path/to/data --epochs 30

# Predict on a thermal image
thermo-classifier predict --domain medical --image thermal.png --model model.pt

# Start API server
thermo-classifier serve --model model.pt --domain medical --port 8000
```

### Python API

```python
from thermo_classifier import ThermalClassifier

# Load model
classifier = ThermalClassifier(
    domain='medical',
    checkpoint='model.pt'
)

# Predict
result = classifier.predict(thermal_image)
print(result['predicted_class'], result['confidence'])
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Backbone LR | 1×10⁻⁵ |
| Quantum layer LR | 1×10⁻² |
| Classifier LR | 1×10⁻⁴ |
| Scheduler | Cosine annealing |
| Batch size | 32 |
| Max epochs | 30 |
| Loss | Cross-entropy + label smoothing (0.1) |
| Weight decay | 8×10⁻⁴ |
| GPU | Tesla T4 (16 GB) |

## Project Structure

```
hqcnn-thermographic/
├── src/
│   └── thermo_classifier/
│       ├── models/         # HQ-CNN, PQC, and classical architectures
│       ├── data/           # DMR-IR dataset loaders
│       ├── training/       # Training loop with differential LR
│       ├── inference/      # Prediction pipeline
│       ├── evaluation/     # Metrics (accuracy, F1, AUC, confusion matrix)
│       ├── api/            # FastAPI REST service
│       ├── cli/            # CLI entry point
│       ├── utils/          # Utilities
│       └── export/         # ONNX / TorchScript export
├── configs/                # YAML configuration files
├── tests/                  # Unit tests
├── train_hq_cnn.py         # Standalone HQ-CNN training script
├── train_baseline.py       # Classical ResNet18 baseline
└── hqcnn-6pages.pdf        # Research paper
```

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- CUDA-capable GPU (Tesla T4 or equivalent)
- NVIDIA CUDA Quantum (for real PQC variant; optional — quantum-inspired layer works without it)

## Citation

If you use this work, please cite:

```bibtex
@article{syah2026hqcnn,
  title={Comparative Analysis of Quantum-Inspired, Parameterized Quantum Circuit, 
         and Classical Approaches for Breast Thermography Classification Using CUDA Quantum},
  author={Syah, Riza Alaudin and Hamed, Haza Nuzly Bin Abdul},
  institution={Faculty of Computing, Universiti Teknologi Malaysia},
  year={2026}
}
```

## Acknowledgments

This work was supported by **Universiti Teknologi Malaysia**. The DMR-IR dataset was provided by the **Visual Lab** at Universidade Federal Fluminense, Brazil.

## License

Apache License 2.0
