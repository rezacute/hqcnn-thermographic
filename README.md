# HQ-CNN: Hybrid Quantum-Classical CNN for Breast Thermography Classification

> **Comparative Analysis of Quantum-Inspired, Parameterized Quantum Circuit, and Classical Approaches for Breast Thermography Classification Using CUDA Quantum**
>
> Riza Alaudin Syah, Haza Nuzly Bin Abdul Hamed — *Universiti Teknologi Malaysia*
> DR. Irwan A. Kautsar — *Universitas Muhammadiyah Sidoarjo, Indonesia*

A controlled three-way comparison of classical, quantum-inspired, and real parameterized quantum circuit (PQC) approaches on breast infrared thermography, evaluated on the [DMR-IR benchmark](https://visual.ic.uff.br/dmi) (1,522 thermograms, 56 patients).

## Key Results

| Variant | Test Accuracy | AUC | Sens | Spec |
|---------|:------------:|:---:|:----:|:----:|
| Classical ResNet18 | **77.0 ± 4.9%** | 0.957 | 69% | 100% |
| Quantum-Inspired (QI) | 76.8 ± 8.4% | 0.934 | 69% | 100% |
| PQC n=6 | 76.9 ± 4.3% | 0.948 | — | — |
| PQC n=8 | 71.2 ± 6.5% | 0.964 | 61% | 100% |
| MLP-twin (classical control) | 71.3 ± 2.7% | 0.910 | 62% | 99% |
| PQC no-CNOT (ablation) | 72.7 ± 4.6% | 0.950 | 64% | 100% |

All variants evaluated on **5 seeds**, patient-level test split of **482 images** (360 malignant, 122 normal).

**Main findings:**
- Classical and QI are statistically tied (Δ = +0.17 pp, McNemar p = 0.85)
- PQC n=8 is statistically indistinguishable from its MLP-twin replacement (Δ = −0.08 pp, p = 0.95) — the circuit provides no measurable benefit over a classical tanh map
- Entanglement provides no benefit: PQC ≈ PQC-noCNOT (p = 0.18)
- Optimal qubit count is n = 6, not n = 8
- Smaller encoding scale (π/64) outperforms the canonical π/4 by 7.1 pp — the PQC in this regime reduces to a classical linear map

## Architecture

All variants share a **ResNet18 backbone** (ImageNet pretrained, fine-tuned) outputting a 512-dim GAP feature vector and a shared classifier head: `Linear(544→256) → ReLU → BN → Dropout → Linear(256→2)`.

```
Input (224×224 thermal image)
    │
    ▼
┌─────────────────────────┐
│  ResNet18 Backbone        │  → h ∈ R^512
│  (ImageNet pretrained)    │
└───────────┬───────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌──────────┐  ┌──────────────────────────────┐
│ Classical │  │  Quantum Branch               │
│ (skip)   │  │  tanh projection → PQC/MHA   │
└────┬─────┘  └──────────────┬───────────────┘
     │                        │
     └────────────┬───────────┘
                  ▼
       ┌──────────────────────┐
       │  Classifier Head      │  Linear(544→256) → ReLU → BN → Dropout → Linear(256→2)
       └──────────────────────┘
                  │
                  ▼
          Normal / Malignant
```

**Six evaluated variants:**
- `classical` — GAP feature → classifier (no quantum branch)
- `qi` — GAP → tanh → 2-layer 4-head self-attention → Linear → 32
- `pqc` — GAP → tanh → n-qubit PQC (RY/RZ + CNOT ring) → 2n−1 expectations → Linear → BN → 32
- `pqc_noent` — same as pqc, no CNOTs (product-state ablation)
- `mlp_twin` — same architecture as pqc but circuit replaced by classical `tanh(A·φ + a)` map
- `mha` — 2-layer 4-head multi-head self-attention over spatial feature map

## Dataset

**DMR-IR** (Database for Mastology Research with Infrared Images, Silva et al. 2014) — 1,522 breast infrared thermograms from 56 patients.

Patient-level split (zero overlap between splits):

| Split | Malignant pts | Normal pts | Total pts | Images |
|-------|:-----------:|:---------:|:---------:|:------:|
| Train | 20 | 11 | **31** | 840 |
| Validation | 5 | 2 | **7** | 200 |
| Test | 15 | 3 | **18** | 482 |

Images resized to 224×224, ImageNet normalization, augmentation: horizontal flip (p=0.5), rotation ±20°, brightness/contrast 0.2, translation 10%. No breast segmentation or synthetic augmentation.

## PQC Implementation

The PQC is exactly simulated (statevector, no shots, no noise) via PyTorch autograd. Verified against dense NumPy matrices, Qiskit Statevector, and CUDA-Q — all 12 checks pass with maximum error < 10⁻¹².

```
Encoding:       φ = α · tanh(W·h + b)          (α = encoding scale)
Variational:   2 layers of RY(θ₁)·RZ(θ₂) per qubit + CNOT ladder (ring closure)
Measurement:   ⟨Z_k⟩ (k=0..n-1) + ⟨Z_k·Z_{k+1}⟩ (k=0..n-2)  →  2n−1 features
```

## Installation

```bash
conda env create -f environment.yml
conda activate hqcnn-thermographic
pip install -e .
```

## Quick Start

```bash
# Train classical baseline
python train.py --data dataset/dmrir2 --variant classical --seed 0 --epochs 30

# Train PQC (8 qubits, π/4 encoding)
python train.py --data dataset/dmrir2 --variant pqc --n-qubits 8 --enc-scale pi/4 --seed 0

# Train quantum-inspired
python train.py --data dataset/dmrir2 --variant qi --seed 0

# Run full grid (all variants × 5 seeds)
python run_grid.py --data dataset/dmrir2 --runs runs --grid full --seeds 5

# Aggregate into paper tables and figures
python aggregate.py --runs runs --out results
```

## Project Structure

```
hqcnn-thermographic/
├── train.py              # Single-run training script
├── run_grid.py           # Full grid runner (all variants × seeds)
├── aggregate.py          # Aggregate results into tables + figures
├── pqc_torch.py          # PQC statevector simulator (verified < 1e-12)
├── models.py             # All 6 model variants
├── data.py               # Patient-level dataset loader
├── verify_simulator.py   # Simulator verification suite
├── paper/                # LaTeX paper + figures
│   ├── main.tex
│   └── visualization_*.jpeg
├── results/              # Generated tables and figures
│   ├── numbers.tex       # Paper-ready \\def values
│   ├── tab_main.tex
│   └── predictions_seed*.csv
└── runs/                 # All run outputs (metrics.json, predictions.csv)
```

## Requirements

- Python 3.9+
- PyTorch with CUDA
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (for metrics and ROC/CM)
- CUDA-capable GPU (RTX / T4 / A100)

## Citation

```bibtex
@article{syah2026hqcnn,
  title={Comparative Analysis of Quantum-Inspired, Parameterized Quantum Circuit,
         and Classical Approaches for Breast Thermography Classification},
  author={Syah, Riza Alaudin and Hamed, Haza Nuzly Bin Abdul and
          Kautsar, Irwan Alnarus},
  institution={Universiti Teknologi Malaysia},
  year={2026}
}
```

## Acknowledgments

This work was supported by **Universiti Teknologi Malaysia**. The DMR-IR dataset was provided by the **Visual Lab** at Universidade Federal Fluminense, Brazil.

## License

Apache License 2.0
