#!/usr/bin/env python3
"""
aggregate.py - turn finished runs into the paper's tables, figures and statistics.

    python aggregate.py --runs runs --out results

Statistics
  * per configuration: mean, SD and t-based 95% CI of test accuracy over seeds, plus
    sensitivity, specificity, macro-F1, AUC and training time
  * per pair of variants (same seeds, same 240 test images):
      - exact McNemar test for every seed (Wilson/Newcombe via compute_stats.py)
      - mean paired accuracy difference over seeds and a paired t-test
      - hierarchical bootstrap CI of the difference (resampling seeds AND test images)
      - equivalence: the smallest margin d for which the 90% bootstrap CI lies inside
        (-d, +d)  (two one-sided tests at alpha = 0.05)
  * qubit sweep, fix ablations and branch ablations

Outputs in OUT/: summary.json, tab_*.tex, fig_*.pdf/.png, predictions_seed<k>.csv
(the CSVs are in the format `python compute_stats.py predictions --csv ...` expects).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from compute_stats import mcnemar_exact, newcombe_paired

LABELS = {
    "classical": "Classical (no branch)",
    "qi_n8": "Quantum-inspired (QI)",
    "pqc_n8_pi4_bn": "PQC, 8 qubits",
    "pqc_noent_n8_pi4_bn": "PQC without CNOTs",
    "mlp_twin_n8_pi4_bn": "PQC circuit $\\to$ MLP",
}
SHORT = {"classical": "Classical", "qi_n8": "QI", "pqc_n8_pi4_bn": "PQC",
         "pqc_noent_n8_pi4_bn": "PQC-noCNOT", "mlp_twin_n8_pi4_bn": "MLP-twin"}
TICK = {"classical": "Classical", "qi_n8": "QI", "pqc_n8_pi4_bn": "PQC",
        "pqc_noent_n8_pi4_bn": "PQC\nno CNOT", "mlp_twin_n8_pi4_bn": "PQC circuit\n→ MLP"}
CORE = list(LABELS)
PAIRS = [("classical", "qi_n8"), ("classical", "pqc_n8_pi4_bn"), ("qi_n8", "pqc_n8_pi4_bn"),
         ("pqc_n8_pi4_bn", "pqc_noent_n8_pi4_bn"), ("pqc_n8_pi4_bn", "mlp_twin_n8_pi4_bn")]
FIXES = [("pi4", "bn"), ("pi", "bn"), ("pi4", "nobn"), ("pi", "nobn")]
PQC_RE = re.compile(r"^pqc_n(\d+)_(pi4|pi2|pi8|pi)_(bn|nobn)$")


def sweep_keys(keys):
    """[(n, key)] for the qubit sweep: plain PQC, pi/4 encoding, BatchNorm."""
    return sorted((int(m.group(1)), k) for k in keys
                  if (m := PQC_RE.match(k)) and m.group(2) == "pi4" and m.group(3) == "bn")


# ------------------------------------------------------------------ loading
def load_runs(root: Path) -> dict:
    groups = defaultdict(dict)
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not (d / "metrics.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        met = json.loads((d / "metrics.json").read_text())
        hist = json.loads((d / "history.json").read_text())
        with open(d / "test_predictions.csv") as fh:
            preds = {r["image"]: (int(r["y_true"]), float(r["prob_malignant"]), int(r["pred"]))
                     for r in csv.DictReader(fh)}
        key = cfg["run"].rsplit("_seed", 1)[0]
        groups[key][int(cfg["seed"])] = dict(cfg=cfg, met=met, hist=hist, preds=preds)
    return groups


def mean_ci(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), (float("nan"), float("nan"))
    sd = float(x.std(ddof=1))
    h = stats.t.ppf(0.975, len(x) - 1) * sd / math.sqrt(len(x))
    return m, sd, (m - h, m + h)


def summarize(group: dict) -> dict:
    seeds = sorted(group)
    t = lambda k: [group[s]["met"]["test"][k] for s in seeds]
    out = dict(seeds=seeds, n_seeds=len(seeds))
    for k in ("accuracy", "sensitivity", "specificity", "f1_macro", "auc", "balanced_accuracy"):
        m, sd, ci = mean_ci(t(k))
        out[k] = dict(mean=m, sd=sd, ci95=ci, values=t(k))
    out["train_minutes"] = float(np.mean([group[s]["met"]["train_seconds"] / 60 for s in seeds]))
    out["best_epoch_mean"] = float(np.mean([group[s]["met"]["best_epoch"] for s in seeds]))
    out["parameters"] = group[seeds[0]]["cfg"]["parameters"]
    abl = [group[s]["met"].get("branch_ablation") for s in seeds]
    if all(abl):
        out["branch_ablation"] = {k: float(np.mean([a[k] for a in abl]))
                                  for k in ("full", "mean_replaced", "permuted_mean", "branch_output_sd")}
    diag = [group[s]["met"].get("circuit_diagnostics") for s in seeds]
    if all(diag):
        out["circuit_diagnostics"] = {k: float(np.mean([d[k] for d in diag])) for k in diag[0]}
    return out


# ------------------------------------------------------------------ paired statistics
def correctness_matrix(group: dict, seeds, images):
    return np.array([[group[s]["preds"][im][2] == group[s]["preds"][im][0] for im in images] for s in seeds],
                    dtype=float)


def paired(gA: dict, gB: dict, n_boot=10000, seed=0) -> dict | None:
    seeds = sorted(set(gA) & set(gB))
    if not seeds:
        return None
    images = sorted(set.intersection(*[set(gA[s]["preds"]) for s in seeds], *[set(gB[s]["preds"]) for s in seeds]))
    CA, CB = correctness_matrix(gA, seeds, images), correctness_matrix(gB, seeds, images)
    S, N = CA.shape
    per_seed = []
    for i, s in enumerate(seeds):
        a_, b_ = CA[i].astype(bool), CB[i].astype(bool)
        a, b, c, d = int((a_ & b_).sum()), int((a_ & ~b_).sum()), int((~a_ & b_).sum()), int((~a_ & ~b_).sum())
        lo, hi = newcombe_paired(a, b, c, d)
        per_seed.append(dict(seed=s, b=b, c=c, diff=(b - c) / N, p_exact=mcnemar_exact(b, c), ci95=(lo, hi)))
    diffs = CA.mean(1) - CB.mean(1)
    if S >= 2 and diffs.std(ddof=1) > 0:
        tstat = diffs.mean() / (diffs.std(ddof=1) / math.sqrt(S))
        p_t = float(2 * stats.t.sf(abs(tstat), S - 1))
    else:
        tstat, p_t = float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=n_boot).astype(float)       # image resampling (paired)
    accA, accB = W @ CA.T / N, W @ CB.T / N                                      # (B, S)
    sidx = rng.integers(0, S, size=(n_boot, S))                                  # seed resampling
    boot = np.take_along_axis(accA - accB, sidx, axis=1).mean(1)
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    lo90, hi90 = np.percentile(boot, [5, 95])
    ps = [r["p_exact"] for r in per_seed]
    return dict(seeds=seeds, n_images=N, mean_diff=float(diffs.mean()), paired_t=float(tstat), p_paired_t=p_t,
                boot_ci95=(float(lo95), float(hi95)), boot_ci90=(float(lo90), float(hi90)),
                equivalence_margin=float(max(abs(lo90), abs(hi90))),
                mcnemar_median_p=float(np.median(ps)), mcnemar_min_p=float(min(ps)), mcnemar_max_p=float(max(ps)),
                seeds_p_below_05=int(sum(p < 0.05 for p in ps)), per_seed=per_seed)


# ------------------------------------------------------------------ figures
def fig_main(summ: dict, out: Path):
    import matplotlib.pyplot as plt
    import figstyle as fs
    fs.apply()
    keys = [k for k in CORE if k in summ]
    fig, ax = plt.subplots(figsize=(fs.COLUMN_IN, 1.9), constrained_layout=True)
    rng = np.random.default_rng(0)
    allv = []
    for i, k in enumerate(keys):
        v = 100 * np.asarray(summ[k]["accuracy"]["values"])
        allv += list(v)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color=fs.MUTED, zorder=2, linewidths=0)
        m, (lo, hi) = 100 * summ[k]["accuracy"]["mean"], summ[k]["accuracy"]["ci95"]
        if not math.isnan(lo):
            lo, hi = max(0.0, 100 * lo), min(100.0, 100 * hi)
            ax.plot([i + 0.25, i + 0.25], [lo, hi], color=fs.BLUE, lw=1.3)
            allv += [lo, hi]
        ax.plot(i + 0.25, m, "o", color=fs.BLUE, ms=4.5, mec="white", mew=1)
        fs.end_label(ax, i + 0.25, m, f"{m:.1f}", color=fs.INK2, dx=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([TICK[k] for k in keys])
    ax.set_ylabel("Test accuracy (%)")
    pad = max(1.0, 0.08 * (max(allv) - min(allv)))
    ax.set_ylim(max(0, min(allv) - pad), min(100, max(allv) + pad))
    ax.set_xlim(-0.5, len(keys) - 0.3)
    fig.savefig(out / "fig_main_comparison.pdf"); fig.savefig(out / "fig_main_comparison.png", dpi=300)
    plt.close(fig)


def fig_qubits(summ: dict, out: Path):
    import matplotlib.pyplot as plt
    import figstyle as fs
    ns = [n for n, _ in sweep_keys(summ)]
    if len(ns) < 2:
        return
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COLUMN_IN, 1.9), constrained_layout=True)
    if "classical" in summ:
        c = summ["classical"]["accuracy"]
        lo, hi = c["ci95"]
        if not math.isnan(lo):
            ax.axhspan(100 * lo, 100 * hi, color=fs.GRID, lw=0, zorder=0)
        ax.axhline(100 * c["mean"], color=fs.MUTED, lw=0.9, zorder=1)
        top = 100 * (hi if not math.isnan(hi) else c["mean"])
        ax.annotate("classical baseline", (ns[-1], top), xytext=(0, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=6.5, color=fs.MUTED)
    means, los, his = [], [], []
    for n in ns:
        a = summ[f"pqc_n{n}_pi4_bn"]["accuracy"]
        v = 100 * np.asarray(a["values"])
        ax.scatter(np.full(len(v), n) + np.random.default_rng(n).uniform(-0.25, 0.25, len(v)), v, s=8,
                   color=fs.MUTED, linewidths=0, zorder=2)
        means.append(100 * a["mean"]); los.append(100 * a["ci95"][0]); his.append(100 * a["ci95"][1])
    ax.fill_between(ns, los, his, color=fs.BLUE, alpha=0.12, lw=0, zorder=1)
    ax.plot(ns, means, "o-", color=fs.BLUE, mec="white", mew=1, zorder=3)
    ax.set_xticks(ns)
    ax.set_xlabel("Qubits $n$")
    ax.set_ylabel("Test accuracy (%)")
    fig.savefig(out / "fig_qubit_sweep.pdf"); fig.savefig(out / "fig_qubit_sweep.png", dpi=300)
    plt.close(fig)


def fig_curves(groups: dict, out: Path):
    import matplotlib.pyplot as plt
    import figstyle as fs
    keys = [k for k in ("classical", "qi_n8", "pqc_n8_pi4_bn") if k in groups]
    if not keys:
        return
    fs.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(fs.COLUMN_IN, 1.6), constrained_layout=True)
    for k, col in zip(keys, fs.SERIES):
        for ax, field, scale in ((a1, "val_acc", 100), (a2, "train_loss", 1)):
            L = max(len(r["hist"]) for r in groups[k].values())
            m, s, e = [], [], []
            for ep in range(L):
                v = [scale * r["hist"][ep][field] for r in groups[k].values() if len(r["hist"]) > ep]
                if len(v) >= 2:
                    m.append(np.mean(v)); s.append(np.std(v, ddof=1)); e.append(ep + 1)
            m, s = np.asarray(m), np.asarray(s)
            ax.fill_between(e, m - s, m + s, color=col, alpha=0.12, lw=0)
            ax.plot(e, m, color=col, label=SHORT[k])
    from matplotlib.ticker import MaxNLocator
    a1.set_title("(a) Validation accuracy (%)"); a2.set_title("(b) Training loss")
    for ax in (a1, a2):
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    a2.legend(loc="upper right", handlelength=1.4)
    fig.savefig(out / "fig_training_curves.pdf"); fig.savefig(out / "fig_training_curves.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ LaTeX
def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    v = round(100 * x, 1)
    return f"{0.0 if v == 0 else v:.1f}"


def spts(x):
    """signed percentage points, no negative zero"""
    v = round(100 * x, 1)
    return f"{0.0 if v == 0 else v:+.1f}"


def write_tables(summ, pairs, groups, out: Path):
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Variant & Accuracy (\%) & 95\% CI & Sens. & Spec. & AUC & Min \\", r"\midrule"]
    for k in CORE:
        if k in summ:
            s = summ[k]
            lo, hi = s["accuracy"]["ci95"]
            lines.append(f"{LABELS[k]} & {pct(s['accuracy']['mean'])} $\\pm$ {pct(s['accuracy']['sd'])} & "
                         f"[{pct(lo)}, {pct(hi)}] & {pct(s['sensitivity']['mean'])} & {pct(s['specificity']['mean'])} & "
                         f"{s['auc']['mean']:.3f} & {s['train_minutes']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_main.tex").write_text("\n".join(lines))

    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Pair (A $-$ B) & $\Delta$ (pts) & 95\% CI & McNemar $p$ & $p<.05$ & Margin \\", r"\midrule"]
    for (ka, kb), r in pairs.items():
        lines.append(f"{SHORT[ka]} $-$ {SHORT[kb]} & {spts(r['mean_diff'])} & "
                     f"[{spts(r['boot_ci95'][0])}, {spts(r['boot_ci95'][1])}] & "
                     f"{r['mcnemar_median_p']:.2f} ({r['mcnemar_min_p']:.2f}--{r['mcnemar_max_p']:.2f}) & "
                     f"{r['seeds_p_below_05']}/{len(r['seeds'])} & {100 * r['equivalence_margin']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_pairs.tex").write_text("\n".join(lines))

    qk = sweep_keys(summ)
    if qk:
        lines = [r"\begin{tabular}{rrcc}", r"\toprule", r"$n$ & Circuit params & Accuracy (\%) & $\Delta$ vs classical \\",
                 r"\midrule"]
        for n, k in qk:
            s = summ[k]
            d = pairs_one(groups, "classical", k)
            dtxt = (f"{spts(-d['mean_diff'])} [{spts(-d['boot_ci95'][1])}, {spts(-d['boot_ci95'][0])}]"
                    if d else "--")
            lines.append(f"{n} & {s['parameters']['circuit']} & {pct(s['accuracy']['mean'])} $\\pm$ "
                         f"{pct(s['accuracy']['sd'])} & {dtxt} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "tab_qubits.tex").write_text("\n".join(lines))

    ref = "pqc_n8_pi4_bn"
    if ref in summ and any(f"pqc_n8_{e}_{b}" in summ for e, b in FIXES[1:]):
        lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Encoding & BatchNorm & Accuracy (\%) & $\Delta$ vs $\pi/4$+BN \\",
                 r"\midrule"]
        for e, b in FIXES:
            k = f"pqc_n8_{e}_{b}"
            if k not in summ:
                continue
            s = summ[k]
            d = pairs_one(groups, k, ref) if k != ref else None
            dtxt = f"{spts(d['mean_diff'])} [{spts(d['boot_ci95'][0])}, {spts(d['boot_ci95'][1])}]" if d else "--"
            enc = r"$\pi/4$" if e == "pi4" else r"$\pi$"
            lines.append(f"{enc} & {'yes' if b == 'bn' else 'no'} & "
                         f"{pct(s['accuracy']['mean'])} $\\pm$ {pct(s['accuracy']['sd'])} & {dtxt} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "tab_fixes.tex").write_text("\n".join(lines))

    rows = [k for k in CORE if k in summ and "branch_ablation" in summ[k]]
    if rows:
        lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Variant & Full & Branch $\to$ mean & Branch permuted \\", r"\midrule"]
        for k in rows:
            b = summ[k]["branch_ablation"]
            lines.append(f"{SHORT[k]} & {pct(b['full'])} & {pct(b['mean_replaced'])} & {pct(b['permuted_mean'])} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / "tab_branch_ablation.tex").write_text("\n".join(lines))


_PAIR_CACHE = {}


def pairs_one(groups, ka, kb):
    if (ka, kb) not in _PAIR_CACHE:
        _PAIR_CACHE[(ka, kb)] = paired(groups[ka], groups[kb]) if ka in groups and kb in groups else None
    return _PAIR_CACHE[(ka, kb)]


def write_prediction_csvs(groups, out: Path):
    keys = [k for k in CORE if k in groups]
    seeds = sorted(set.intersection(*[set(groups[k]) for k in keys])) if keys else []
    for s in seeds:
        images = sorted(set.intersection(*[set(groups[k][s]["preds"]) for k in keys]))
        with open(out / f"predictions_seed{s}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_id", "y_true"] + [f"pred_{SHORT[k]}" for k in keys] + [f"score_{SHORT[k]}" for k in keys])
            for im in images:
                y = groups[keys[0]][s]["preds"][im][0]
                w.writerow([im, y] + [groups[k][s]["preds"][im][2] for k in keys]
                           + [f"{groups[k][s]['preds'][im][1]:.6f}" for k in keys])


# ------------------------------------------------------------------ in-text numbers for the paper
def write_numbers(summ, groups, out: Path, bench_path=None, leak_path=None):
    """results/numbers.tex: \def macros the paper uses in its text (missing ones show as 'pending')."""
    num = {}
    acc = lambda k: f"{pct(summ[k]['accuracy']['mean'])}\\,$\\pm$\\,{pct(summ[k]['accuracy']['sd'])}"
    names = dict(classical="Classical", qi_n8="QI", pqc_n8_pi4_bn="PQC", pqc_noent_n8_pi4_bn="NoCNOT",
                 mlp_twin_n8_pi4_bn="Twin")
    for k, nm in names.items():
        if k in summ:
            num[f"Acc{nm}"] = acc(k)
            num[f"AUC{nm}"] = f"{summ[k]['auc']['mean']:.3f}"
            num[f"Min{nm}"] = f"{summ[k]['train_minutes']:.1f}"
            ab = summ[k].get("branch_ablation")
            if ab:
                num[f"Abl{nm}Mean"] = pct(ab["mean_replaced"]); num[f"Abl{nm}Perm"] = pct(ab["permuted_mean"])
                num[f"Abl{nm}Change"] = spts(ab["permuted_mean"] - ab["full"])   # shuffled minus intact
    for (ka, kb) in PAIRS + [("classical", "pqc_noent_n8_pi4_bn"), ("classical", "mlp_twin_n8_pi4_bn")]:
        r = pairs_one(groups, ka, kb)
        if not r:
            continue
        tag = names[ka] + names[kb]
        num[f"Diff{tag}"] = spts(r["mean_diff"])
        num[f"CI{tag}"] = f"[{spts(r['boot_ci95'][0])},\\,{spts(r['boot_ci95'][1])}]"
        num[f"Margin{tag}"] = f"{100 * r['equivalence_margin']:.1f}"
        num[f"P{tag}"] = f"{r['mcnemar_median_p']:.2f}"
        num[f"Sig{tag}"] = f"{r['seeds_p_below_05']}/{len(r['seeds'])}"
    sk = sweep_keys(summ)
    if sk:
        best = max(sk, key=lambda t: summ[t[1]]["accuracy"]["mean"])
        worst = min(sk, key=lambda t: summ[t[1]]["accuracy"]["mean"])
        num.update(SweepBestN=str(best[0]), SweepBestAcc=acc(best[1]), SweepWorstN=str(worst[0]),
                   SweepWorstAcc=acc(worst[1]), SweepRange=f"{sk[0][0]}--{sk[-1][0]}")
    for e, b, nm in (("pi", "bn", "Pi"), ("pi4", "nobn", "NoBN"), ("pi", "nobn", "PiNoBN")):
        k = f"pqc_n8_{e}_{b}"
        if k in summ:
            num[f"AccPQC{nm}"] = acc(k)
            r = pairs_one(groups, k, "pqc_n8_pi4_bn")
            if r:
                num[f"Diff{nm}"] = spts(r["mean_diff"])
                num[f"CI{nm}"] = f"[{spts(r['boot_ci95'][0])},\\,{spts(r['boot_ci95'][1])}]"
    d = summ.get("pqc_n8_pi4_bn", {}).get("circuit_diagnostics")
    if d:
        num["DiagTanhSat"] = pct(d["frac_abs_tanh_gt_0_9"])
        num["DiagReadoutVar"] = f"{d['readout_variance_across_images']:.3f}"
        if "meyer_wallach_mean" in d:
            num["DiagMW"] = f"{d['meyer_wallach_mean']:.2f}"; num["DiagFid"] = f"{d['mean_pairwise_fidelity']:.2f}"
    anyk = next(iter(summ))
    cfg = groups[anyk][summ[anyk]["seeds"][0]]["cfg"]
    num.update(NTrain=f"{cfg['split_sizes']['train']:,}", NVal=f"{cfg['split_sizes']['val']:,}",
               NTest=f"{cfg['split_sizes']['test']:,}", NPosTest=str(cfg["split_positives"]["test"]),
               NSeeds=str(max(v["n_seeds"] for v in summ.values())),
               EpochsBest=f"{np.mean([v['best_epoch_mean'] for v in summ.values()]):.0f}")
    if leak_path and Path(leak_path).exists():
        lk = json.loads(Path(leak_path).read_text())
        t = lk["test_nearest_train_distance"]
        for key, nm in (("n_within_16_bits", "LeakWithinSixteen"), ("n_within_8_bits", "LeakWithinEight"),
                        ("n_identical_hash", "LeakIdentical"), ("n_below_unrelated_p1", "LeakBelowRandom")):
            if key in t:
                num[nm] = str(t[key])
        num["LeakTestN"] = str(lk["sizes"]["test"])
    if bench_path and Path(bench_path).exists():
        b = json.loads(Path(bench_path).read_text())
        for v, nm in (("classical", "Classical"), ("qi", "QI"), ("pqc", "PQC"), ("pqc_noent", "NoCNOT"), ("mlp_twin", "Twin")):
            if v in b["models"]:
                num[f"Lat{nm}"] = f"{b['models'][v]['batch1']['median_ms']:.1f}"
                num[f"Thr{nm}"] = f"{b['models'][v]['batch32']['images_per_s']:.0f}"
        bo = b.get("branch_only", {})
        for n, nm in (("8", "Eight"), ("16", "Sixteen")):
            if n in bo:
                num[f"Branch{nm}"] = f"{1000 * bo[n]['batch32']['per_image_ms']:.0f}"   # microseconds per image
        num["BenchDevice"] = "".join("\\" + ch if ch in "_&%#$" else ch for ch in str(b["info"]["device_name"]))
        cq = b.get("cudaq", {})
        if "8" in cq:
            num["CudaqEight"] = f"{cq['8']['per_image_ms']:.1f}"
        tex = Path(bench_path).with_suffix(".tex")
        if tex.exists():
            (out / "tab_inference.tex").write_text(tex.read_text())
    lines = ["% generated by aggregate.py - do not edit"] + [f"\\def\\{k}{{{v}}}" for k, v in sorted(num.items())]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    return num


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--bench", default=None, help="benchmark JSON from benchmark.py (GPU run)")
    ap.add_argument("--leakage", default=None, help="leakage_report.json from leakage_audit.py")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    groups = load_runs(Path(a.runs))
    if not groups:
        raise SystemExit(f"no finished runs in {a.runs}")
    summ = {k: summarize(g) for k, g in groups.items()}
    pairs = {p: pairs_one(groups, *p) for p in PAIRS if p[0] in groups and p[1] in groups}
    for _, k in sweep_keys(summ):
        if "classical" in groups:
            pairs_one(groups, "classical", k)
    write_tables(summ, pairs, groups, out)
    write_prediction_csvs(groups, out)
    nums = write_numbers(summ, groups, out, a.bench, a.leakage)
    if not a.no_figures:
        fig_main(summ, out); fig_qubits(summ, out); fig_curves(groups, out)
    dump = dict(summary=summ, pairs={f"{x}|{y}": v for (x, y), v in _PAIR_CACHE.items() if v})
    (out / "summary.json").write_text(json.dumps(dump, indent=2, default=float))
    print(f"{sum(len(g) for g in groups.values())} runs in {len(groups)} configurations -> {out}/  "
          f"({len(nums)} numbers in numbers.tex)")
    for k in sorted(summ):
        s = summ[k]["accuracy"]
        print(f"  {k:24s} n={summ[k]['n_seeds']}  acc {100 * s['mean']:.2f} +- {100 * s['sd']:.2f}")
    for (ka, kb), r in pairs.items():
        print(f"  {ka} - {kb}: {100 * r['mean_diff']:+.2f} pts, 95% CI [{100 * r['boot_ci95'][0]:+.2f}, "
              f"{100 * r['boot_ci95'][1]:+.2f}], McNemar median p {r['mcnemar_median_p']:.3f}, "
              f"equivalence margin {100 * r['equivalence_margin']:.2f} pts")


if __name__ == "__main__":
    main()
