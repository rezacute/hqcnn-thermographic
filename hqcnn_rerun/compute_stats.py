#!/usr/bin/env python3
"""
compute_stats.py - statistical comparison of the three breast-thermography models.

Three sub-commands:

  bounds       What can be said from the reported accuracies alone (no per-image
               predictions needed). Because McNemar's test depends on how the two
               models' errors overlap, it reports the full range of possible
               p-values and confidence intervals over every possible overlap.
               This is what the paper currently reports.

                 python compute_stats.py bounds --n 240 \
                     --correct Classical=221 QI=219 PQC=217

  predictions  Exact analysis from per-image predictions on the SAME test set.
               CSV columns (header required):
                 image_id, y_true, pred_<model>..., [score_<model>...]
               y_true / pred_* are 0 (normal) or 1 (malignant); score_* is the
               predicted probability of malignant (optional, enables AUC + DeLong).

                 python compute_stats.py predictions --csv test_predictions.csv

  folds        Per-fold accuracies for the qubit-scaling study (paired t-test).

                 python compute_stats.py folds --n 240 \
                     --fold q4=202,185,202 q8=217,213,210 q12=186,193,202

Every sub-command prints plain-text results and a LaTeX table you can paste
into the paper.

Methods
  * Wilson score interval for single accuracies (Wilson 1927).
  * Exact (conditional binomial) McNemar test, plus mid-p (Fagerland et al. 2013).
  * Newcombe hybrid-score interval (method 10) for the paired difference in
    accuracy (Newcombe 1998); Bonett-Price interval as a cross-check.
  * Equivalence (TOST, Schuirmann 1987): the smallest margin for which
    equivalence holds at alpha = 0.05 is the largest |limit| of the 90% CI.
  * AUC with DeLong variance and paired DeLong test (DeLong et al. 1988),
    using the fast algorithm of Sun & Xu (2014).

Only numpy and scipy are required.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass

import numpy as np
from scipy import stats

Z95 = stats.norm.ppf(0.975)
Z90 = stats.norm.ppf(0.95)


# ----------------------------------------------------------------------------
# Single proportion
# ----------------------------------------------------------------------------
def wilson(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion x/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ----------------------------------------------------------------------------
# Paired proportions (McNemar)
# ----------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact conditional McNemar p-value (doubling the smaller tail)."""
    nd = b + c
    if nd == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * stats.binom.cdf(k, nd, 0.5)
    return min(1.0, p)


def mcnemar_midp(b: int, c: int) -> float:
    """Two-sided mid-p McNemar (Fagerland, Lydersen & Laake 2013)."""
    nd = b + c
    if nd == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * (stats.binom.cdf(k, nd, 0.5) - 0.5 * stats.binom.pmf(k, nd, 0.5))
    return min(1.0, p)


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = Z95) -> tuple[float, float]:
    """
    Newcombe (1998) method 10: hybrid score interval for p1 - p2 with paired data.

    a = both correct, b = model 1 correct & model 2 wrong,
    c = model 1 wrong & model 2 correct, d = both wrong.
    p1 = (a + b)/n is model 1's accuracy, p2 = (a + c)/n is model 2's.
    """
    n = a + b + c + d
    p1 = (a + b) / n
    p2 = (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    e, f, g, h = a + b, c + d, a + c, b + d
    denom = e * f * g * h
    phi = 0.0 if denom == 0 else (a * d - b * c) / math.sqrt(denom)
    theta = p1 - p2
    delta = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    eps = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (theta - delta, theta + eps)


def bonett_price(b: int, c: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Bonett & Price (2012) adjusted Wald interval for a paired difference."""
    p12 = (b + 1) / (n + 2)
    p21 = (c + 1) / (n + 2)
    theta = p12 - p21
    se = math.sqrt(max(0.0, (p12 + p21 - theta ** 2) / (n + 2)))
    return (theta - z * se, theta + z * se)


@dataclass
class PairResult:
    a: int
    b: int
    c: int
    d: int
    p_exact: float
    p_midp: float
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    ci95_bp: tuple[float, float]

    @property
    def tost_margin(self) -> float:
        """Smallest equivalence margin (as a proportion) supported at alpha=0.05."""
        return max(abs(self.ci90[0]), abs(self.ci90[1]))


def analyse_pair(a: int, b: int, c: int, d: int) -> PairResult:
    n = a + b + c + d
    return PairResult(
        a, b, c, d,
        p_exact=mcnemar_exact(b, c),
        p_midp=mcnemar_midp(b, c),
        ci95=newcombe_paired(a, b, c, d, Z95),
        ci90=newcombe_paired(a, b, c, d, Z90),
        ci95_bp=bonett_price(b, c, n, Z95),
    )


def all_overlaps(correct1: int, correct2: int, n: int) -> list[PairResult]:
    """
    Every 2x2 table consistent with the two marginal accuracies.
    k = number of test images that BOTH models get wrong.
    """
    w1, w2 = n - correct1, n - correct2
    out = []
    for k in range(0, min(w1, w2) + 1):
        b = w2 - k          # model 1 right, model 2 wrong
        c = w1 - k          # model 1 wrong, model 2 right
        d = k
        a = n - b - c - d
        if min(a, b, c, d) < 0:
            continue
        out.append(analyse_pair(a, b, c, d))
    return out


# ----------------------------------------------------------------------------
# AUC + DeLong (fast algorithm, Sun & Xu 2014)
# ----------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(scores: np.ndarray, y: np.ndarray):
    """
    scores: (k, N) predicted scores for k models on the same N images.
    y: (N,) with 1 = positive (malignant).
    Returns (auc (k,), covariance (k, k)).
    """
    scores = np.atleast_2d(scores)
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.concatenate([pos, neg], axis=1)])
    auc = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    cov = sx / m + sy / n
    return auc, np.atleast_2d(cov)


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def fmt_p(p: float) -> str:
    if p >= 0.995:
        return "1.00"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ----------------------------------------------------------------------------
# Sub-command: bounds
# ----------------------------------------------------------------------------
def cmd_bounds(args):
    n = args.n
    models = []
    for item in args.correct:
        name, val = item.split("=")
        models.append((name, int(val)))

    print(f"Test set: n = {n}\n")
    print("Single-model accuracy (Wilson 95% CI)")
    for name, x in models:
        lo, hi = wilson(x, n)
        print(f"  {name:10s} {x:4d}/{n}  {pct(x / n, 2)}%  [{pct(lo)}, {pct(hi)}]")

    rows = []
    print("\nPairwise comparisons over every possible overlap of errors")
    for (n1, x1), (n2, x2) in itertools.combinations(models, 2):
        res = all_overlaps(x1, x2, n)
        pe = [r.p_exact for r in res]
        pm = [r.p_midp for r in res]
        lo95 = min(r.ci95[0] for r in res)
        hi95 = max(r.ci95[1] for r in res)
        lo95_bp = min(r.ci95_bp[0] for r in res)
        hi95_bp = max(r.ci95_bp[1] for r in res)
        tost = [r.tost_margin for r in res]
        nd = [r.b + r.c for r in res]
        print(f"\n  {n1} vs {n2}: difference = {x1 - x2:+d} images = {pct((x1 - x2) / n, 2)} points")
        print(f"    discordant pairs b+c range     : {min(nd)} .. {max(nd)}")
        print(f"    exact McNemar p range          : {min(pe):.4f} .. {max(pe):.4f}")
        print(f"    mid-p McNemar p range          : {min(pm):.4f} .. {max(pm):.4f}")
        print(f"    Newcombe 95% CI envelope (pts) : [{pct(lo95, 2)}, {pct(hi95, 2)}]")
        print(f"    Bonett-Price 95% CI envelope   : [{pct(lo95_bp, 2)}, {pct(hi95_bp, 2)}]")
        print(f"    smallest TOST margin (pts)     : {pct(min(tost), 2)} (best case) .. {pct(max(tost), 2)} (worst case)")
        rows.append(dict(pair=f"{n1} vs {n2}", diff=x1 - x2, nd=(min(nd), max(nd)),
                         p=(min(pe), max(pe)), ci=(lo95, hi95), tost=(min(tost), max(tost))))

    print("\n% ---- LaTeX (bounds) ----")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"\textbf{Pair} & $\Delta$ & \textbf{McNemar $p$} & \textbf{95\% CI (pts)} & \textbf{Margin} \\")
    print(r"\midrule")
    for r in rows:
        print(f"{r['pair']} & ${r['diff']:+d}$ & {fmt_p(r['p'][0])}--{fmt_p(r['p'][1])} & "
              f"$[{pct(r['ci'][0])},\\,{pct(r['ci'][1])}]$ & {pct(r['tost'][1])} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ----------------------------------------------------------------------------
# Sub-command: predictions
# ----------------------------------------------------------------------------
def _read_predictions(path: str):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        sys.exit("empty CSV")
    cols = rows[0].keys()
    if "y_true" not in cols:
        sys.exit("CSV needs a y_true column")
    y = np.array([int(r["y_true"]) for r in rows])
    preds = {c[5:]: np.array([int(r[c]) for r in rows]) for c in cols if c.startswith("pred_")}
    scores = {c[6:]: np.array([float(r[c]) for r in rows]) for c in cols if c.startswith("score_")}
    if not preds:
        sys.exit("CSV needs pred_<model> columns")
    return y, preds, scores


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if tn else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, sens=sens, spec=spec,
                f1_malignant=f1_pos, f1_macro=(f1_pos + f1_neg) / 2)


def cmd_predictions(args):
    y, preds, scores = _read_predictions(args.csv)
    n = len(y)
    print(f"Test set: n = {n} ({int(y.sum())} malignant, {int((1 - y).sum())} normal)\n")
    table = []
    for name, p in preds.items():
        m = _binary_metrics(y, p)
        x = int((p == y).sum())
        lo, hi = wilson(x, n)
        slo, shi = wilson(m["tp"], m["tp"] + m["fn"])
        plo, phi_ = wilson(m["tn"], m["tn"] + m["fp"])
        print(f"{name}: acc {pct(x / n, 2)}% [{pct(lo)}, {pct(hi)}]  "
              f"sens {pct(m['sens'])}% [{pct(slo)}, {pct(shi)}]  "
              f"spec {pct(m['spec'])}% [{pct(plo)}, {pct(phi_)}]  "
              f"F1(malignant) {pct(m['f1_malignant'], 2)}  F1(macro) {pct(m['f1_macro'], 2)}  "
              f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")
        table.append((name, x, lo, hi, m))

    if scores:
        names = [k for k in preds if k in scores]
        if names:
            auc, cov = delong(np.vstack([scores[k] for k in names]), y)
            print("\nAUC (DeLong 95% CI)")
            for i, k in enumerate(names):
                se = math.sqrt(cov[i, i])
                print(f"  {k}: {auc[i]:.3f} [{max(0.0, auc[i] - Z95 * se):.3f}, {min(1.0, auc[i] + Z95 * se):.3f}]")
            for i, j in itertools.combinations(range(len(names)), 2):
                var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
                zst = (auc[i] - auc[j]) / math.sqrt(var) if var > 0 else 0.0
                pv = 2 * stats.norm.sf(abs(zst))
                print(f"  DeLong {names[i]} vs {names[j]}: dAUC={auc[i] - auc[j]:+.3f}, p={pv:.3f}")

    print("\nPairwise McNemar (exact) and Newcombe 95% CI for the accuracy difference")
    lines = []
    for (n1, p1), (n2, p2) in itertools.combinations(preds.items(), 2):
        c1, c2 = p1 == y, p2 == y
        a = int((c1 & c2).sum())
        b = int((c1 & ~c2).sum())
        c = int((~c1 & c2).sum())
        d = int((~c1 & ~c2).sum())
        r = analyse_pair(a, b, c, d)
        print(f"  {n1} vs {n2}: b={b} c={c}  exact p={r.p_exact:.4f}  mid-p={r.p_midp:.4f}  "
              f"diff={pct((b - c) / n, 2)} pts  95% CI [{pct(r.ci95[0], 2)}, {pct(r.ci95[1], 2)}]  "
              f"TOST margin={pct(r.tost_margin, 2)} pts")
        lines.append(f"{n1} vs {n2} & {b} & {c} & {fmt_p(r.p_exact)} & "
                     f"$[{pct(r.ci95[0])},\\,{pct(r.ci95[1])}]$ & {pct(r.tost_margin)} \\\\")

    print("\n% ---- LaTeX (exact, from predictions) ----")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"\textbf{Pair} & $b$ & $c$ & \textbf{McNemar $p$} & \textbf{95\% CI (pts)} & \textbf{Margin} \\")
    print(r"\midrule")
    for ln in lines:
        print(ln)
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ----------------------------------------------------------------------------
# Sub-command: folds
# ----------------------------------------------------------------------------
def cmd_folds(args):
    n = args.n
    folds = {}
    for item in args.fold:
        name, vals = item.split("=")
        folds[name] = np.array([int(v) for v in vals.split(",")], dtype=float)
    print(f"Per-fold correct counts out of n = {n}")
    for k, v in folds.items():
        acc = 100 * v / n
        print(f"  {k}: {', '.join(f'{a:.2f}' for a in acc)}  mean {acc.mean():.2f}  SD {acc.std(ddof=1):.2f}")
    for (k1, v1), (k2, v2) in itertools.combinations(folds.items(), 2):
        d = 100 * (v1 - v2) / n
        if len(d) > 1 and d.std(ddof=1) > 0:
            t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
            p = 2 * stats.t.sf(abs(t), len(d) - 1)
        else:
            t, p = float("nan"), float("nan")
        wins = int((d > 0).sum())
        print(f"  {k1} - {k2}: mean {d.mean():+.2f} pts, paired t = {t:.2f}, df = {len(d) - 1}, p = {p:.3f}, "
              f"{k1} better in {wins}/{len(d)} folds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bounds", help="bounds from aggregate accuracies")
    b.add_argument("--n", type=int, required=True)
    b.add_argument("--correct", nargs="+", required=True, help="NAME=count ...")
    b.set_defaults(func=cmd_bounds)

    p = sub.add_parser("predictions", help="exact analysis from a predictions CSV")
    p.add_argument("--csv", required=True)
    p.set_defaults(func=cmd_predictions)

    f = sub.add_parser("folds", help="per-fold comparison")
    f.add_argument("--n", type=int, required=True)
    f.add_argument("--fold", nargs="+", required=True, help="NAME=c1,c2,c3 ...")
    f.set_defaults(func=cmd_folds)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
