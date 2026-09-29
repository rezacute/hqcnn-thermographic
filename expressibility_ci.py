#!/usr/bin/env python3
"""
expressibility_ci.py - frame-potential ratio R2 with bootstrap confidence intervals.

R2 = E[F^2] / E_Haar[F^2] for fidelities F between random-parameter states of the
2-layer ansatz. At large n the estimate rests on rare high-fidelity pairs, so it is
reported with a 95% bootstrap interval over pairs.   Writes expressibility_ci.json.
"""
import argparse
import json
import math

import numpy as np
import torch

from circuit_analysis import chunks, rand_flat, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", default="2,4,6,8,10,12,14,16")
    ap.add_argument("--pairs", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="expressibility_ci.json")
    a = ap.parse_args()
    out = {}
    for n in [int(v) for v in a.qubits.split(",")]:
        gen = torch.Generator().manual_seed(a.seed + n)
        chunk = max(8, min(512, (1 << 22) // (1 << n)))
        fids = []
        for B in chunks(a.pairs, chunk):
            s1 = states(rand_flat(B, n, 0.0, 2 * math.pi, 0.0, gen), n)
            s2 = states(rand_flat(B, n, 0.0, 2 * math.pi, 0.0, gen), n)
            fids.append((torch.sum(s1.conj() * s2, dim=1).abs() ** 2).double())
        F = torch.cat(fids).numpy()
        N = 1 << n
        haar2 = 2.0 / (N * (N + 1))
        rng = np.random.default_rng(a.seed)
        boot = np.array([(F[rng.integers(0, len(F), len(F))] ** 2).mean() / haar2 for _ in range(a.boot)])
        r2 = float((F ** 2).mean() / haar2)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[str(n)] = dict(r2=r2, ci95=[float(lo), float(hi)], pairs=a.pairs, mean_F_times_N=float(F.mean() * N))
        print(f"n={n:2d}  R2={r2:.3f}  95% CI [{lo:.3f}, {hi:.3f}]", flush=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
