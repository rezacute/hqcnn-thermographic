"""
pqc_cudaq.py - train the PQC branch with CUDA Quantum (CUDA-Q).

Forward:  every image's circuit is executed by cudaq.observe (kernel in cudaq_kernel.py);
          one broadcast call per batch returns the 2n-1 readouts per image.
Backward: exact gradients by the parameter-shift rule, with every shifted circuit also
          executed by CUDA-Q:  dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2,
          valid because each angle sits in exactly one RY or RZ gate.
          The encoding RY(phi_k) is followed directly by the first-layer RY on the same
          qubit, so RY(t) RY(phi) = RY(t + phi) and dE/dphi_k equals the per-image
          derivative for that first-layer angle: no extra circuits are needed for phi.

Circuits per training step (batch B, n qubits, L layers):  B * (1 + 2 * (2nL + n)),
i.e. 2,592 for B = 32, n = 8, L = 2.  Checked against pqc_torch.py in verify_simulator.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_RUNNERS: dict = {}
_TARGET_SET: list = []


class CudaqRunner:
    """Executes batches of the HQ-CNN circuit on one CUDA-Q target and returns readouts."""

    def __init__(self, n: int, layers: int, entangle: bool, target: str | None):
        import cudaq
        from cudaq_kernel import hqcnn_kernel, readout_terms
        if target and not _TARGET_SET:
            cudaq.set_target(target)
            _TARGET_SET.append(target)
        self.cudaq, self.kernel = cudaq, hqcnn_kernel
        self.n, self.layers, self.entangle = n, layers, int(bool(entangle))
        self.terms, self.ham = readout_terms(n)
        self.circuits = 0          # running count of executed circuits

    def expect(self, phi: np.ndarray, theta: np.ndarray, theta_final: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """phi (R, n), theta (R, 2nL), theta_final (R, n) -> readouts (R, 2n-1)."""
        R = phi.shape[0]
        out = np.empty((R, len(self.terms)))
        for s in range(0, R, chunk):
            e = min(R, s + chunk)
            r = e - s
            res = self.cudaq.observe(self.kernel, self.ham, [self.n] * r, [self.layers] * r, [self.entangle] * r,
                                     phi[s:e].tolist(), theta[s:e].tolist(), theta_final[s:e].tolist())
            if not isinstance(res, (list, tuple)):
                res = [res]
            out[s:e] = [[x.expectation(t) for t in self.terms] for x in res]
        self.circuits += R
        return out


def get_runner(n: int, layers: int, entangle: bool, target: str | None) -> CudaqRunner:
    key = (n, layers, bool(entangle), target)
    if key not in _RUNNERS:
        _RUNNERS[key] = CudaqRunner(n, layers, entangle, target)
    return _RUNNERS[key]


class _CudaqCircuit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, phi, theta, theta_final, runner):
        B = phi.shape[0]
        P = phi.detach().double().cpu().numpy()
        T = theta.detach().double().cpu().numpy().reshape(-1)
        F = theta_final.detach().double().cpu().numpy()
        E = runner.expect(P, np.tile(T, (B, 1)), np.tile(F, (B, 1)))
        ctx.save_for_backward(phi.detach(), theta.detach(), theta_final.detach())
        ctx.runner = runner
        return torch.as_tensor(E, dtype=phi.dtype, device=phi.device)

    @staticmethod
    def backward(ctx, grad_out):
        phi, theta, theta_final = ctx.saved_tensors
        runner = ctx.runner
        B, n = phi.shape
        P0 = phi.double().cpu().numpy()
        T0 = theta.double().cpu().numpy().reshape(-1)
        F0 = theta_final.double().cpu().numpy()
        nt, nf = T0.size, F0.size
        npar = nt + nf
        rows = B * 2 * npar                                    # order: image, parameter, sign (+, -)
        PH = np.repeat(P0, 2 * npar, axis=0)
        TT = np.tile(T0, (rows, 1))
        FF = np.tile(F0, (rows, 1))
        idx = np.arange(rows)
        p = (idx // 2) % npar
        shift = np.where(idx % 2 == 0, math.pi / 2, -math.pi / 2)
        in_t = p < nt
        TT[idx[in_t], p[in_t]] += shift[in_t]
        FF[idx[~in_t], p[~in_t] - nt] += shift[~in_t]
        E = runner.expect(PH, TT, FF).reshape(B, npar, 2, -1)
        dE = 0.5 * (E[:, :, 0, :] - E[:, :, 1, :])                   # (B, npar, 2n-1)
        g = grad_out.detach().double().cpu().numpy()                  # (B, 2n-1)
        per_image = np.einsum("bpm,bm->bp", dE, g)                    # (B, npar)
        grad_theta = per_image[:, :nt].sum(0).reshape(tuple(theta.shape))
        grad_final = per_image[:, nt:].sum(0)
        first_ry = np.arange(n) * 2                                   # layer-0 RY angle of qubit k in flat theta
        grad_phi = per_image[:, first_ry]
        as_t = lambda a, like: torch.as_tensor(np.ascontiguousarray(a), dtype=like.dtype, device=like.device)
        return as_t(grad_phi, phi), as_t(grad_theta, theta), as_t(grad_final, theta_final), None


def cudaq_expectations(phi: torch.Tensor, theta: torch.Tensor, theta_final: torch.Tensor,
                       entangle: bool = True, target: str | None = None) -> torch.Tensor:
    """Drop-in replacement for pqc_torch.simulate(): readouts (B, 2n-1), differentiable."""
    n_layers, n = theta.shape[0], theta.shape[1]
    if n_layers < 1:
        raise ValueError("the phi-gradient shortcut needs at least one variational layer")
    return _CudaqCircuit.apply(phi, theta, theta_final, get_runner(n, n_layers, entangle, target))


def circuits_per_step(batch: int, n: int, layers: int = 2) -> int:
    return batch * (1 + 2 * (2 * n * layers + n))
