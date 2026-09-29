#!/usr/bin/env python3
"""
circuit_analysis.py - data-free analysis of the HQ-CNN circuit (reviewer point C).

For n = 2, 4, ..., 16 qubits and the 2-layer ansatz in pqc_torch.py:

  A. Expressibility (Sim, Johnson & Aspuru-Guzik 2019): KL divergence between the
     distribution of state fidelities F = |<psi(t)|psi(t')>|^2 for random parameters
     t, t' ~ U[0, 2pi) and the Haar distribution (N-1)(1-F)^(N-2), 75 bins.
     Also the second frame-potential ratio R2 = E[F^2] / E_Haar[F^2] (1 = Haar-like).
  B. Entangling capability: mean Meyer-Wallach Q over random parameters, and Q in the
     regime the model starts training in (circuit angles ~ U[-0.1, 0.1], encoding
     angles ~ U[-s, s] for s = pi/4 and s = pi).
  C. Gradient variance: Var over random parameters of d<O>/d(theta) for the first
     RY angle on qubit 0 (exact, parameter-shift), for a local readout O = Z_0 and the
     global parity Z_0 Z_1 ... Z_{n-1}.
  D. Feature concentration at initialisation: variance across inputs of the 2n-1
     readout values when encoding angles are ~ U[-s, s], s = pi/4 vs pi.
  E. Light cone at theta = 0: the Z-string each readout maps to through the two CNOT
     rings (exact Pauli propagation over GF(2)); at theta = 0 every readout equals a
     product of cos(phi_j) over that string.

Writes circuit_analysis.json.  Runtime on 2 CPU cores: about 10-15 minutes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from pqc_torch import expectations, ring_cnot_pairs, simulate_state, unpack

L = 2
BINS = 75


def chunks(total, size):
    for start in range(0, total, size):
        yield min(size, total - start)


def rand_flat(B, n, circ_low, circ_high, enc_scale, gen):
    P = n + 2 * n * L + n
    flat = torch.empty(B, P, dtype=torch.float32)
    flat[:, :n] = (torch.rand(B, n, generator=gen) * 2 - 1) * enc_scale
    flat[:, n:] = circ_low + (circ_high - circ_low) * torch.rand(B, P - n, generator=gen)
    return flat


def states(flat, n):
    return simulate_state(*unpack(flat, n, L), cdtype=torch.complex64)


# ------------------------------------------------------------------ A. expressibility
def haar_bin_logprob(n):
    """log of the Haar probability mass in each of BINS equal fidelity bins, stable for large N."""
    N = 1 << n
    edges = np.linspace(0.0, 1.0, BINS + 1)
    out = np.empty(BINS)
    for i in range(BINS):
        a, b = edges[i], edges[i + 1]
        la = (N - 1) * math.log1p(-a) if a < 1 else -math.inf
        lb = (N - 1) * math.log1p(-b) if b < 1 else -math.inf
        # log((1-a)^(N-1) - (1-b)^(N-1)) = la + log(1 - exp(lb - la))
        out[i] = la + math.log1p(-math.exp(lb - la)) if lb > -math.inf else la
    return out


def expressibility(n, pairs, gen, chunk):
    fids = []
    for B in chunks(pairs, chunk):
        a = states(rand_flat(B, n, 0.0, 2 * math.pi, 0.0, gen), n)
        b = states(rand_flat(B, n, 0.0, 2 * math.pi, 0.0, gen), n)
        fids.append((torch.sum(a.conj() * b, dim=1).abs() ** 2).double())
    F = torch.cat(fids).numpy()
    counts, _ = np.histogram(F, bins=BINS, range=(0.0, 1.0))
    p = counts / counts.sum()
    logq = haar_bin_logprob(n)
    mask = p > 0
    kl = float(np.sum(p[mask] * (np.log(p[mask]) - logq[mask])))
    N = 1 << n
    r1 = float(F.mean() * N)
    r2 = float((F ** 2).mean() / (2.0 / (N * (N + 1))))
    return dict(kl=kl, frame_ratio_1=r1, frame_ratio_2=r2, mean_fidelity=float(F.mean()),
                haar_mean_fidelity=1.0 / N, pairs=int(pairs))


# ------------------------------------------------------------------ B. Meyer-Wallach
def meyer_wallach(psi, n):
    B = psi.shape[0]
    purity = torch.zeros(B, dtype=torch.float64)
    for k in range(n):
        p = psi.reshape(B, 1 << k, 2, 1 << (n - k - 1))
        rho = torch.einsum("baic,bajc->bij", p, p.conj())
        purity += (rho.abs() ** 2).sum(dim=(1, 2)).double()
    return 2.0 * (1.0 - purity / n)


def entangling(n, samples, gen, chunk, circ_low, circ_high, enc_scale):
    qs = []
    for B in chunks(samples, chunk):
        qs.append(meyer_wallach(states(rand_flat(B, n, circ_low, circ_high, enc_scale, gen), n), n))
    q = torch.cat(qs).numpy()
    return dict(mean=float(q.mean()), sd=float(q.std(ddof=1)), samples=int(samples))


# ------------------------------------------------------------------ C. gradient variance
def parity_signs(n, device):
    x = torch.arange(1 << n, device=device)
    pc = torch.zeros_like(x)
    for k in range(n):
        pc += (x >> k) & 1
    return (1 - 2 * (pc % 2)).float()


def gradient_variance(n, samples, gen, chunk, param_index):
    g_local, g_global = [], []
    for B in chunks(samples, chunk):
        flat = rand_flat(B, n, 0.0, 2 * math.pi, 0.0, gen)
        plus, minus = flat.clone(), flat.clone()
        plus[:, param_index] += math.pi / 2
        minus[:, param_index] -= math.pi / 2
        sp, sm = states(plus, n), states(minus, n)
        pp, pm = sp.real ** 2 + sp.imag ** 2, sm.real ** 2 + sm.imag ** 2
        g_local.append(0.5 * (expectations(sp, n)[:, 0] - expectations(sm, n)[:, 0]).double())
        par = parity_signs(n, pp.device)
        g_global.append(0.5 * ((pp @ par) - (pm @ par)).double())
    gl, gg = torch.cat(g_local).numpy(), torch.cat(g_global).numpy()
    return dict(var_local_Z0=float(gl.var(ddof=1)), var_global_parity=float(gg.var(ddof=1)),
                samples=int(samples))


# ------------------------------------------------------------------ D. feature concentration
def feature_concentration(n, draws, inputs, gen, chunk, enc_scale):
    """Variance across inputs of each readout, averaged over readouts and random initial circuits."""
    per_draw = []
    for _ in range(draws):
        theta = (torch.rand(L, n, 2, generator=gen) * 2 - 1) * 0.1
        theta_f = (torch.rand(n, generator=gen) * 2 - 1) * 0.1
        feats = []
        for B in chunks(inputs, chunk):
            phi = (torch.rand(B, n, generator=gen) * 2 - 1) * enc_scale
            psi = simulate_state(phi, theta, theta_f, cdtype=torch.complex64)
            feats.append(expectations(psi, n).double())
        f = torch.cat(feats)
        per_draw.append(f.var(dim=0, unbiased=True).mean().item())
    return dict(mean_readout_variance=float(np.mean(per_draw)), sd_over_draws=float(np.std(per_draw, ddof=1)),
                draws=int(draws), inputs=int(inputs))


# ------------------------------------------------------------------ E. light cone at theta = 0
def conjugate_through_ring(z, n):
    """Heisenberg picture: Z-string z (0/1 vector) conjugated by one CNOT ring (last gate first)."""
    z = z.copy()
    for c, t in reversed(ring_cnot_pairs(n)):
        if z[t]:
            z[c] ^= 1
    return z


def light_cone(n):
    weights = []
    for k in range(n):
        z = np.zeros(n, dtype=np.int64); z[k] = 1
        weights.append(int(conjugate_through_ring(conjugate_through_ring(z, n), n).sum()))
    for k in range(n - 1):
        z = np.zeros(n, dtype=np.int64); z[k] = z[k + 1] = 1
        weights.append(int(conjugate_through_ring(conjugate_through_ring(z, n), n).sum()))
    return weights


def analytic_readout_variance(weights, s):
    """At theta = 0, readout = prod_{j in S} cos(phi_j) with phi_j ~ U[-s, s] independent."""
    m1 = math.sin(s) / s
    m2 = 0.5 + math.sin(2 * s) / (4 * s)
    return float(np.mean([m2 ** w - m1 ** (2 * w) for w in weights]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", default="2,4,6,8,10,12,14,16")
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--mw-samples", type=int, default=1000)
    ap.add_argument("--grad-samples", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--inputs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="circuit_analysis.json")
    ap.add_argument("--merge", action="store_true", help="add to an existing output file instead of replacing it")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    results = dict(meta=dict(layers=L, bins=BINS, seed=args.seed, pairs=args.pairs,
                             mw_samples=args.mw_samples, grad_samples=args.grad_samples,
                             draws=args.draws, inputs=args.inputs,
                             torch=torch.__version__), per_n={})
    if args.merge and Path(args.out).exists():
        results["per_n"] = json.loads(Path(args.out).read_text())["per_n"]
    for n in [int(v) for v in args.qubits.split(",")]:
        t0 = time.time()
        gen = torch.Generator().manual_seed(args.seed + n)
        chunk = max(8, min(512, (1 << 22) // (1 << n)))
        r = dict(n=n, circuit_params=2 * n * L + n, cnots=2 * len(ring_cnot_pairs(n)))
        r["expressibility"] = expressibility(n, args.pairs, gen, chunk)
        r["entanglement_random"] = entangling(n, args.mw_samples, gen, chunk, 0.0, 2 * math.pi, 0.0)
        r["entanglement_init_pi4"] = entangling(n, args.mw_samples, gen, chunk, -0.1, 0.1, math.pi / 4)
        r["entanglement_init_pi"] = entangling(n, args.mw_samples, gen, chunk, -0.1, 0.1, math.pi)
        r["gradient"] = gradient_variance(n, args.grad_samples, gen, chunk, param_index=n)  # first-layer RY, qubit 0
        r["features_init_pi4"] = feature_concentration(n, args.draws, args.inputs, gen, chunk, math.pi / 4)
        r["features_init_pi"] = feature_concentration(n, args.draws, args.inputs, gen, chunk, math.pi)
        w = light_cone(n)
        r["light_cone"] = dict(weights=w, mean_weight=float(np.mean(w)),
                               analytic_var_theta0_pi4=analytic_readout_variance(w, math.pi / 4),
                               analytic_var_theta0_pi=analytic_readout_variance(w, math.pi))
        r["seconds"] = round(time.time() - t0, 1)
        results["per_n"][str(n)] = r
        e = r["expressibility"]
        print(f"n={n:2d}  KL={e['kl']:.3f}  R2={e['frame_ratio_2']:.3g}  "
              f"MW(rand)={r['entanglement_random']['mean']:.3f}  MW(init,pi/4)={r['entanglement_init_pi4']['mean']:.4f}  "
              f"MW(init,pi)={r['entanglement_init_pi']['mean']:.4f}  "
              f"Var dZ0={r['gradient']['var_local_Z0']:.3g}  Var dParity={r['gradient']['var_global_parity']:.3g}  "
              f"featVar pi/4={r['features_init_pi4']['mean_readout_variance']:.3g}  "
              f"pi={r['features_init_pi']['mean_readout_variance']:.3g}  "
              f"lightcone={np.mean(w):.2f}  ({r['seconds']} s)", flush=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
