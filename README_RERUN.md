# Controlled rerun: classical vs quantum-inspired vs simulated PQC

This folder replaces `train_baseline.py`, `train_hq_cnn.py` and `train_cuda_quantum.py` for the paper.
All three variants (plus two controls) are trained **the same way**; only the branch differs.

| File | What it does |
|---|---|
| `pqc_torch.py` | Exact statevector simulation of the quantum branch in PyTorch (batched, GPU, exact gradients) |
| `verify_simulator.py` | Checks the simulator against dense matrices, Qiskit and CUDA-Q, and autograd against the parameter-shift rule |
| `models.py` | Backbone + one shared head + five branches: `classical`, `qi`, `pqc`, `pqc_noent`, `mlp_twin` |
| `data.py` | Loads your existing `train/val/test` folders once, caches 224x224 pixels, applies the paper's augmentation |
| `train.py` | One run → `runs/<name>/` with history, test predictions, metrics, branch ablations, circuit diagnostics |
| `run_grid.py` | Every configuration x every seed, one job per GPU, resumable |
| `aggregate.py` | Tables, figures, statistics and `numbers.tex` for the paper |
| `benchmark.py` | Inference latency and throughput (and CUDA-Q timing with `--cudaq`) |
| `leakage_audit.py` | Checks whether test images are near-copies of training images (same patient) |
| `circuit_analysis.py` | Data-free circuit analysis (expressibility, entanglement, gradient variance), already run |
| `compute_stats.py` | McNemar, Newcombe, DeLong from a predictions CSV |
| `expressibility_ci.py` | Frame-potential ratio with bootstrap intervals (already run: `expressibility_ci.json`) |
| `make_circuit_figure.py` | Fig. 3 of the paper from the two JSON files |
| `AUDIT.md`, `audit/verify_old_code.py` | What was wrong with the old code and draft, and a script that reproduces it |

## What changed compared with the old scripts

- The PQC is actually simulated. The old `train_cuda_quantum.py` never called its circuit (it returned random numbers).
- One head, loss, optimiser, schedule, augmentation and seed set for all variants. The old baseline had its own head, loss and learning rate.
- Binary output (normal / malignant) with no leftover class weights from the 201-image dataset.
- Two controls that isolate the circuit: `pqc_noent` (same circuit, no CNOTs) and `mlp_twin` (circuit replaced by `tanh(A·phi + a)`).
- The "quantum-inspired" layer is kept exactly as in `train_hq_cnn.py` (2 heads, as the old code actually built it), so its result is comparable with the old one.

## Steps

Needs Python 3.10+, `torch`, `torchvision`, `numpy`, `scipy`, `matplotlib`, `pillow`. Optional: `cudaq`, `qiskit` (only for the verification script and the CUDA-Q timing).

```bash
# 0. check the simulator on your machine (about 1 minute)
python verify_simulator.py

# 1. leakage check on the split you used (seconds)
python leakage_audit.py --data /path/to/DATA --out results
#    DATA/{train,val,test}/{normal,malignant}/  (use --negative/--positive if your folder names differ)
#    If many test images sit within a few bits of a training image, the split is per image, not per patient.
#    Tell me before training: the results would then need a patient-level split.

# 2. the full grid: 14 configurations x 5 seeds = 70 runs
python run_grid.py --data /path/to/DATA --grid full --seeds 0,1,2,3,4 --gpus 0,1,2,3
#    about 3-5 min per run on a T4 -> roughly 1-1.5 h on four T4s. Resumable: just re-run the command.
#    For a first look: --grid quick --seeds 0,1,2

# 3. inference timing on the T4 (and CUDA-Q timing if cudaq is installed)
python benchmark.py --device cuda --cudaq --out results/bench_t4.json

# 4. tables, figures and numbers for the paper
python aggregate.py --runs runs --out results --bench results/bench_t4.json --leakage results/leakage_report.json
```

Then send me the `results/` folder (a few MB: tables, figures, `summary.json`, `numbers.tex`, prediction CSVs).
To fill the paper yourself: copy `results/` next to the `.tex` file and recompile. Every table and in-text
number marked **pending** fills in automatically.

## Settings (identical for every variant; defaults in `train.py`)

| Setting | Value |
|---|---|
| Backbone | ResNet18, ImageNet weights, fine-tuned |
| Head | Dropout(0.5) → Linear(d→256) → ReLU → BatchNorm → Dropout(0.3) → Linear(256→2) |
| Optimiser | AdamW, lr backbone 1e-5, branch 1e-3, head 1e-4, weight decay 1e-4 (none on rotation angles) |
| Schedule | cosine annealing over 30 epochs, early stopping patience 10 on validation accuracy |
| Batch size | 32 (last partial batch dropped) |
| Loss | cross-entropy, label smoothing 0.1, no class weights |
| Augmentation | horizontal flip 0.5, rotation ±20°, brightness/contrast 0.2, translation 10% |
| Mixed precision | backbone only (fp16 on GPU); the quantum branch always runs in complex64 |
| PQC | n = 8, 2 layers, encoding phi = (pi/4)·tanh(W·h + b), readout 2n−1 values, Linear → 32 → BatchNorm |
| Seeds | 0, 1, 2, 3, 4 (same seeds for every variant) |

If you change a setting, change it for every variant and tell me, so the paper's settings table matches.
