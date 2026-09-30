#!/usr/bin/env python3
"""
train.py - one controlled training run.

    python train.py --data DATA --variant pqc --n-qubits 8 --seed 0

Writes RUN_DIR/ with
    config.json            every setting, library versions, GPU name
    history.json           per-epoch train/val loss and accuracy (real training curves)
    metrics.json           test metrics, branch ablations, circuit diagnostics, timing
    test_predictions.csv   image path, true label, P(malignant), prediction  (for McNemar / DeLong)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CachedImages, build_cache
from models import VARIANTS, HybridNet
from pqc_torch import PQCBranch, simulate_state

SCALES = {"pi": math.pi, "pi/2": math.pi / 2, "pi/4": math.pi / 4, "pi/8": math.pi / 8}


# ---------------------------------------------------------------- metrics (no sklearn needed)
def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    xs = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def binary_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp else float("nan")
    return dict(n=int(len(y)), correct=tp + tn, accuracy=(tp + tn) / len(y), sensitivity=sens,
                specificity=spec, balanced_accuracy=(sens + spec) / 2, f1_malignant=f1_pos,
                f1_macro=(f1_pos + f1_neg) / 2, auc=auc_score(y, prob), tp=tp, tn=tn, fp=fp, fn=fn)


# ---------------------------------------------------------------- helpers
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_name(a) -> str:
    if a.variant == "classical":
        return f"classical_seed{a.seed}"
    if a.variant == "qi":
        return f"qi_n{a.n_qubits}_seed{a.seed}"
    scale = a.enc_scale.replace("/", "")
    return f"{a.variant}_n{a.n_qubits}_{scale}_{'bn' if not a.no_bn else 'nobn'}_seed{a.seed}"


@torch.no_grad()
def collect(model, loader, device, amp):
    """Backbone features, branch outputs, logits and labels for a whole split (eval mode)."""
    model.eval()
    H, Q, Y, I = [], [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            h, q = model.features(x)
        H.append(h.float()); Y.append(y); I.append(idx)
        if q is not None:
            Q.append(q.float())
    H = torch.cat(H); Y = torch.cat(Y).numpy(); I = torch.cat(I).numpy()
    Q = torch.cat(Q) if Q else None
    logits = model.classify(H, Q)
    return H, Q, logits, Y, I


def evaluate(model, loader, device, amp, criterion):
    H, Q, logits, y, idx = collect(model, loader, device, amp)
    loss = criterion(logits, torch.as_tensor(y, device=logits.device)).item()
    prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    return loss, binary_metrics(y, prob, pred), prob, pred, y, idx, (H, Q)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--variant", choices=VARIANTS, default="pqc")
    ap.add_argument("--n-qubits", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--enc-scale", choices=list(SCALES), default="pi/4")
    ap.add_argument("--no-bn", action="store_true", help="drop the BatchNorm after the circuit readout")
    ap.add_argument("--pqc-backend", choices=["torch", "cudaq"], default="torch",
                    help="cudaq: run every circuit (forward and parameter-shift backward) with CUDA Quantum")
    ap.add_argument("--cudaq-target", default=None,
                    help="CUDA-Q target (default: nvidia on a GPU machine, qpp-cpu otherwise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-branch", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--negative", default="normal")
    ap.add_argument("--positive", default="malignant")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--permutations", type=int, default=20)
    a = ap.parse_args()

    out = Path(a.runs) / run_name(a)
    if (out / "metrics.json").exists():
        print(f"[skip] {out} already finished"); return
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    amp = device.type == "cuda" and not a.no_amp
    torch.backends.cudnn.benchmark = False
    seed_everything(a.seed)

    data = build_cache(a.data, a.negative, a.positive, a.image_size, a.cache_dir)
    gen = torch.Generator().manual_seed(a.seed)
    kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    train_loader = DataLoader(CachedImages(data["train"], True), a.batch_size, shuffle=True, drop_last=True,
                              generator=gen, **kw)
    plain = lambda split: DataLoader(CachedImages(data[split], False), 128, shuffle=False, **kw)
    train_eval_loader, val_loader, test_loader = plain("train"), plain("val"), plain("test")

    seed_everything(a.seed)          # same seed for every variant (data order and augmentation are identical)
    if a.pqc_backend == "cudaq" and a.cudaq_target is None:
        a.cudaq_target = "nvidia" if device.type == "cuda" else "qpp-cpu"
    model = HybridNet(a.variant, a.n_qubits, a.n_layers, SCALES[a.enc_scale], not a.no_bn, a.dropout,
                      pretrained=not a.no_pretrained,
                      pqc_backend=a.pqc_backend if a.variant in ("pqc", "pqc_noent") else "torch",
                      cudaq_target=a.cudaq_target).to(device)
    groups = model.param_groups(a.lr_backbone, a.lr_branch, a.lr_head, a.weight_decay)
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    config = dict(vars(a), run=out.name, enc_scale_value=SCALES[a.enc_scale],
                  circuit_backend=(a.pqc_backend if a.variant in ("pqc", "pqc_noent") else None),
                  parameters=model.count_parameters(), param_groups=[g["name"] for g in groups],
                  torch=torch.__version__, python=platform.python_version(),
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                  split_sizes={s: int(len(d["labels"])) for s, d in data.items()},
                  split_positives={s: int(d["labels"].sum()) for s, d in data.items()})
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run] {out.name}  params={config['parameters']}  device={config['device_name']}")

    history, best, best_key, stale = [], None, None, 0
    t_start = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, tot, correct, loss_sum = time.time(), 0, 0, 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                h, q = model.features(x)
            logits = model.classify(h, q)
            loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(y); tot += len(y); correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        val_loss, val_m, *_ = evaluate(model, val_loader, device, amp, criterion)
        rec = dict(epoch=epoch, train_loss=loss_sum / tot, train_acc=correct / tot, val_loss=val_loss,
                   val_acc=val_m["accuracy"], val_auc=val_m["auc"], seconds=time.time() - t0,
                   lr_head=opt.param_groups[1]["lr"])
        history.append(rec)
        key = (val_m["accuracy"], -val_loss)
        if best_key is None or key > best_key:
            best_key, stale, best_epoch = key, 0, epoch
            best = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        print(f"  epoch {epoch:2d}  train {rec['train_loss']:.4f}/{rec['train_acc']:.4f}  "
              f"val {val_loss:.4f}/{val_m['accuracy']:.4f}  auc {val_m['auc']:.4f}  {rec['seconds']:.1f}s", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if stale >= a.patience:
            break
    train_seconds = time.time() - t_start

    model.load_state_dict(best)
    test_loss, test_m, prob, pred, y, idx, (H, Q) = evaluate(model, test_loader, device, amp, criterion)
    paths = data["test"]["paths"]
    with open(out / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image", "y_true", "prob_malignant", "pred"])
        for i, t, p, d in zip(idx, y, prob, pred):
            w.writerow([paths[i], int(t), f"{p:.6f}", int(d)])

    metrics = dict(run=out.name, best_epoch=best_epoch, epochs_run=len(history), test_loss=test_loss,
                   test=test_m, train_seconds=train_seconds,
                   seconds_per_epoch=float(np.mean([h["seconds"] for h in history])))
    if a.pqc_backend == "cudaq" and isinstance(model.branch, PQCBranch):
        import re, cudaq
        from pqc_cudaq import _RUNNERS
        version = re.search(r"\d+\.\d+(\.\d+)?", str(getattr(cudaq, "__version__", "")))
        metrics["cudaq"] = dict(target=a.cudaq_target, version=version.group(0) if version else "unknown",
                                circuits_executed=int(sum(r.circuits for r in _RUNNERS.values())))

    # ------------------------------------------------ does the branch matter? (test-time ablation)
    if Q is not None:
        _, Qtr, _, _, _ = collect(model, train_eval_loader, device, amp)
        mean_q = Qtr.mean(0, keepdim=True)
        yt = torch.as_tensor(y, device=H.device)
        acc = lambda logits: (logits.argmax(1) == yt).float().mean().item()
        g = torch.Generator(device="cpu").manual_seed(1000 + a.seed)
        perm_accs = [acc(model.classify(H, Q[torch.randperm(len(Q), generator=g).to(Q.device)]))
                     for _ in range(a.permutations)]
        metrics["branch_ablation"] = dict(
            full=test_m["accuracy"],
            mean_replaced=acc(model.classify(H, mean_q.expand_as(Q))),
            permuted_mean=float(np.mean(perm_accs)), permuted_sd=float(np.std(perm_accs, ddof=1)),
            branch_output_sd=float(Q.std(0).mean().item()))

    # ------------------------------------------------ what the circuit actually does on test images
    if isinstance(model.branch, PQCBranch):
        br = model.branch
        with torch.no_grad():
            u = torch.tanh(br.proj(H.float()))
            meas = br.measurements(H)
            diag = dict(mean_abs_tanh=float(u.abs().mean()), frac_abs_tanh_gt_0_9=float((u.abs() > 0.9).float().mean()),
                        readout_variance_across_images=float(meas.var(0).mean()))
            if br.n <= 12:
                psi = simulate_state(br.angles(H), br.theta, br.theta_final, br.entangle, torch.complex64)
                pur = torch.zeros(len(psi), device=psi.device)
                for k in range(br.n):
                    p = psi.reshape(len(psi), 1 << k, 2, 1 << (br.n - k - 1))
                    rho = torch.einsum("baic,bajc->bij", p, p.conj())
                    pur += (rho.abs() ** 2).sum((1, 2))
                diag["meyer_wallach_mean"] = float((2 * (1 - pur / br.n)).mean())
                gram = (psi @ psi.conj().T).abs() ** 2
                off = gram[~torch.eye(len(psi), dtype=torch.bool, device=gram.device)]
                diag["mean_pairwise_fidelity"] = float(off.mean())
        metrics["circuit_diagnostics"] = diag

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if a.save_checkpoint:
        torch.save(best, out / "best.pt")
    print(f"[done] {out.name}: test acc {test_m['accuracy']:.4f}  auc {test_m['auc']:.4f}  "
          f"({train_seconds / 60:.1f} min)")


if __name__ == "__main__":
    main()
