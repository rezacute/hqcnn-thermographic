# Audit of the HQ-CNN thermography results (29 Sep 2026)

What was checked: the public repo `rezacute/hqcnn-thermographic` (commit b7939f1), `RESEARCH_REPORT.md`
and the 6-page IEEE draft. Every statement about the old code can be reproduced with
`python audit/verify_old_code.py /path/to/hqcnn-thermographic`.

## 1. Old code

| File | Finding |
|---|---|
| `train_cuda_quantum.py` | The circuit is defined but never executed. In training every image gets the same fixed random "measurements"; at test time they are fresh random noise. The custom backward returns a constant π/2 for the first 8 parameters, whatever the loss. The encoding is computed and then not used. The script that the report credits with 90.42% (`train_cuda_quantum_fixed.py`) is not in the repo, and neither are logs, predictions or checkpoints (`outputs/`, `logs/`, `*.pth` are git-ignored). |
| `train_hq_cnn.py` (QI layer) | Built with 2 attention heads (the `--attention-heads` flag is ignored). Attention over a single token is exactly a linear map, so its query and key weights get no gradient. The "entanglement" matrices are added to a variable that is then overwritten, so they never affect the output. 272 of 7,688 parameters have no effect. The layer is a small MLP with an 8-unit tanh bottleneck. |
| `train_baseline.py` | Differs from the hybrid scripts in its head (dropout + one linear layer), loss (plain cross-entropy), learning-rate policy (one rate), weight decay and augmentation. It defaults to ResNet50 and reads only `.jpg` files. |
| all three | 3-class output for a 2-class task. The hybrid scripts also use class weights left over from the 201-image dataset. |

## 2. Research report

- The split adds up to 1,493, not 1,522 (1,090 + 163 + 240).
- The 8-qubit result is one fold; the 4-qubit result is a 3-fold mean. On the same fold the gain is 6.25 points, not 8.61.
- The 12-qubit runs used a shortened budget, so they cannot show overfitting or support "8 qubits is optimal".
- Validation accuracy is 99.39% (162/163) in all three folds while test accuracy is 77–84%. The validation set probably overlaps with training.
- The encoding fix and the BatchNorm fix are both credited with +45% from the same starting point. They were applied together, so their separate effects were never measured.
- "0% saturated after the fix" is true by construction: with π/4 scaling no angle can reach π/2.
- The QI layer description (4 heads, "sin-based measurement") does not match the code.

## 3. The 6-page draft written before this audit

- Table VII (ablation) and most of Table VI (latency, memory) contained invented numbers.
- Wrong authors or venues in several references:
  - BCDGAN is by Veerlapalli & Dutta, Sci. Rep. 15:19665 (2025).
  - The 99.33% U-Net paper is Mohamed et al., PLoS ONE 17:e0262349 (2022).
  - The npj review is Gupta et al., npj Digit. Med. 8:237 (2025).
  - DMR-IR should be cited as Silva et al. (2014).
  - A few further entries were unverified, and eight references were never cited in the text.
- "sin(θ/2) ≥ 0.38 across the encoding range" is wrong: it is at most 0.38, and 0 at θ = 0.
- "1,440 circuit evaluations per sample" is wrong: one simulation gives all 15 readouts, so parameter shift needs 2 × 48 = 96.
- The fixes table had two rows swapped (8 qubits vs dropout).
- The confusion matrix labelled 92.08% contains 217/240 = 90.42%.
- "First hybrid quantum CNN for breast thermography" is contradicted by arXiv:2604.16953.
- "Statistically equivalent" rested on a single-proportion interval, which does not test equivalence.
- The four result figures could not be traced to any code or log, and the training curves look synthetic.

## 4. What replaces it

- `hqcnn_rerun/`: one controlled pipeline. All variants are trained identically, the PQC is actually simulated (checked against Qiskit and CUDA-Q), per-image predictions are saved, and it produces the statistics, the qubit sweep, the fix ablations and the inference timing.
- `circuit_analysis.json` and `expressibility_ci.json`: data-free circuit results for 2–16 qubits (final).
- `paper_v2/main.tex`: a draft whose tables and numbers fill in from the rerun's `results/` folder.
- Recommended: update the public README, which still repeats the unsupported claims.
