#!/usr/bin/env python3
"""
make_circuit_figure.py - figure and LaTeX table from circuit_analysis.json.

    python make_circuit_figure.py --json circuit_analysis.json --out paper_figures/
Produces fig_circuit_analysis.pdf/.png and tab_circuit_analysis.tex.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle as fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="circuit_analysis.json")
    ap.add_argument("--out", default="paper_figures")
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())["per_n"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ns = sorted(int(k) for k in res)
    get = lambda f: [f(res[str(n)]) for n in ns]

    fs.apply()
    fig, axes = plt.subplots(2, 2, figsize=(fs.COLUMN_IN, 3.0), constrained_layout=True)
    (a1, a2), (a3, a4) = axes

    # (a) expressibility relative to Haar
    ci_path = Path(a.json).with_name("expressibility_ci.json")
    if ci_path.exists():                      # 10,000 pairs with bootstrap 95% CI
        ci = json.loads(ci_path.read_text())
        ns_ci = [n for n in ns if str(n) in ci]
        r2 = [ci[str(n)]["r2"] for n in ns_ci]
        lo = [ci[str(n)]["ci95"][0] for n in ns_ci]; hi = [ci[str(n)]["ci95"][1] for n in ns_ci]
        a1.fill_between(ns_ci, lo, hi, color=fs.INK2, alpha=0.15, lw=0)
        a1.plot(ns_ci, r2, "o-", color=fs.INK2)
        r2 = hi
    else:
        r2 = get(lambda r: r["expressibility"]["frame_ratio_2"])
        a1.plot(ns, r2, "o-", color=fs.INK2)
    a1.axhline(1.0, color=fs.MUTED, lw=0.8)
    fs.end_label(a1, ns[-1], 1.0, "random states = 1", color=fs.MUTED, dx=0, dy=-6, ha="right")
    a1.set_ylim(0.85, max(r2) * 1.06)
    a1.set_title("(a) Expressibility")
    a1.set_ylabel("Frame-potential ratio $R_2$")

    # (b) gradient variance, random parameters
    gl = get(lambda r: r["gradient"]["var_local_Z0"])
    gg = get(lambda r: r["gradient"]["var_global_parity"])
    a2.semilogy(ns, gl, "o-", color=fs.INK2)
    a2.semilogy(ns, gg, "s--", color=fs.MUTED)
    fs.end_label(a2, ns[-1], gl[-1], r"$\langle Z_0\rangle$ readout", dx=-2, dy=7, ha="right")
    fs.end_label(a2, ns[-1], gg[-1], "global parity", dx=-2, dy=7, ha="right")
    a2.set_title("(b) Gradient variance")
    a2.set_ylabel(r"Var$[\partial\langle O\rangle/\partial\theta]$")

    # (c) entanglement at initialisation, pi/4 vs pi, random-parameter reference
    q4 = get(lambda r: r["entanglement_init_pi4"]["mean"])
    qp = get(lambda r: r["entanglement_init_pi"]["mean"])
    qr = get(lambda r: r["entanglement_random"]["mean"])
    a3.plot(ns, qr, "-", color=fs.MUTED, lw=0.9)
    a3.plot(ns, qp, "s-", color=fs.ORANGE, label=r"encoding $\pi$")
    a3.plot(ns, q4, "o-", color=fs.BLUE, label=r"encoding $\pi/4$")
    fs.end_label(a3, ns[-1], qr[-1], "random angles", color=fs.MUTED, dx=-2, dy=5, ha="right")
    a3.set_ylim(0, 1.0)
    a3.set_title("(c) Entanglement at start")
    a3.set_ylabel("Meyer-Wallach $Q$")
    a3.legend(loc="lower right", handlelength=1.6)

    # (d) readout variation across inputs at initialisation
    v4 = get(lambda r: r["features_init_pi4"]["mean_readout_variance"])
    vp = get(lambda r: r["features_init_pi"]["mean_readout_variance"])
    a4.semilogy(ns, vp, "s-", color=fs.ORANGE, label=r"encoding $\pi$")
    a4.semilogy(ns, v4, "o-", color=fs.BLUE, label=r"encoding $\pi/4$")
    a4.set_title("(d) Readout spread at start")
    a4.set_ylabel("Variance across inputs")
    a4.legend(loc="upper right", handlelength=1.6)

    for ax in axes.flat:
        ax.set_xticks(ns)
        ax.set_xlabel("Qubits $n$")
    fig.savefig(out / "fig_circuit_analysis.pdf")
    fig.savefig(out / "fig_circuit_analysis.png", dpi=300)

    ci = json.loads(Path(a.json).with_name("expressibility_ci.json").read_text()) \
        if Path(a.json).with_name("expressibility_ci.json").exists() else {}
    rows = []
    for n in ns:
        r = res[str(n)]
        if str(n) in ci:
            r["expressibility"]["frame_ratio_2"] = ci[str(n)]["r2"]
        rows.append(f"{n} & {r['circuit_params']} & {r['expressibility']['kl']:.3f} & "
                    f"{r['expressibility']['frame_ratio_2']:.2f} & {r['entanglement_random']['mean']:.2f} & "
                    f"{r['gradient']['var_local_Z0']:.1e} & {r['gradient']['var_global_parity']:.1e} & "
                    f"{r['light_cone']['mean_weight']:.1f} \\\\")
    tex = "\n".join([r"\begin{tabular}{rrrrrrrr}", r"\toprule",
                     r"$n$ & Params & KL & $R_2$ & $Q$ & Var$_{Z_0}$ & Var$_{\mathrm{par}}$ & $\bar w$ \\",
                     r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    (out / "tab_circuit_analysis.tex").write_text(tex.replace("e-0", "e-"))
    print(f"wrote {out}/fig_circuit_analysis.pdf/.png and tab_circuit_analysis.tex")


if __name__ == "__main__":
    main()
