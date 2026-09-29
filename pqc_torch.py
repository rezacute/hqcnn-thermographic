"""
pqc_torch.py - exact, batched statevector simulation of the HQ-CNN quantum branch.

The circuit (n qubits, L variational layers; L = 2 in the paper):

    |0...0>  -- RY(phi_k) on every qubit k            data encoding, phi = s * tanh(W h + b)
             for l = 1..L:
                 RY(theta[l,k,0]) then RZ(theta[l,k,1]) on every qubit k
                 CNOT(k, k+1) for k = 0..n-2, then CNOT(n-1, 0)      (ring)
             RY(theta_final[k]) on every qubit k

    readout: <Z_k> for k = 0..n-1 and <Z_k Z_{k+1}> for k = 0..n-2   (2n - 1 values)

Trainable circuit parameters: 2nL + n  (40 for n = 8, L = 2).

The simulation is exact (no shots, no noise). Every operation is differentiable, so
ordinary autograd gives exact gradients. For these gates the autograd gradient equals
the parameter-shift gradient to machine precision (checked in verify_simulator.py), so
training needs no shifted circuits and no finite differences.

Convention: qubit 0 is the most significant bit of the basis index.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn

__all__ = [
    "ring_cnot_pairs", "encode_product_state", "simulate_state", "expectations",
    "simulate", "parameter_shift_gradient", "PQCBranch", "MLPTwinBranch",
    "n_circuit_params",
]


# ---------------------------------------------------------------------------
# Fixed parts of the circuit
# ---------------------------------------------------------------------------
def ring_cnot_pairs(n: int) -> list[tuple[int, int]]:
    """CNOT(k, k+1) for k = 0..n-2 followed by the ring closure CNOT(n-1, 0)."""
    if n < 2:
        return []
    return [(k, k + 1) for k in range(n - 1)] + [(n - 1, 0)]


def n_circuit_params(n: int, n_layers: int = 2) -> int:
    return 2 * n * n_layers + n


@lru_cache(maxsize=None)
def _ring_permutation_cpu(n: int) -> torch.Tensor:
    """The CNOT ring is a permutation of basis states: new_state = state[:, perm]."""
    N = 1 << n
    x = torch.arange(N, dtype=torch.long)
    f = x.clone()
    for c, t in ring_cnot_pairs(n):
        cbit = (f >> (n - 1 - c)) & 1
        f = f ^ (cbit << (n - 1 - t))
    inv = torch.empty_like(f)
    inv[f] = x                     # psi'[f(x)] = psi[x]  =>  psi'[j] = psi[f^-1(j)]
    return inv


@lru_cache(maxsize=None)
def _readout_signs_cpu(n: int) -> torch.Tensor:
    """(2^n, 2n-1) matrix of eigenvalues: columns Z_0..Z_{n-1}, then Z_k Z_{k+1}."""
    N = 1 << n
    x = torch.arange(N, dtype=torch.long)
    z = torch.stack([1 - 2 * ((x >> (n - 1 - k)) & 1) for k in range(n)], dim=1)
    cols = [z[:, k] for k in range(n)] + [z[:, k] * z[:, k + 1] for k in range(n - 1)]
    return torch.stack(cols, dim=1).to(torch.float64)


_DEVICE_CACHE: dict = {}


def _cached(fn, n, device, dtype=None):
    key = (fn.__name__, n, str(device), dtype)
    if key not in _DEVICE_CACHE:
        t = fn(n).to(device)
        _DEVICE_CACHE[key] = t if dtype is None else t.to(dtype)
    return _DEVICE_CACHE[key]


# ---------------------------------------------------------------------------
# State preparation and gates
# ---------------------------------------------------------------------------
def encode_product_state(phi: torch.Tensor) -> torch.Tensor:
    """phi (B, n) real -> (B, 2^n) real amplitudes of  RY(phi_0)|0> x ... x RY(phi_{n-1})|0>."""
    B, n = phi.shape
    c, s = torch.cos(phi / 2), torch.sin(phi / 2)
    psi = torch.ones(B, 1, dtype=phi.dtype, device=phi.device)
    for k in range(n):
        v = torch.stack([c[:, k], s[:, k]], dim=1)              # (B, 2)
        psi = (psi.unsqueeze(2) * v.unsqueeze(1)).reshape(B, -1)
    return psi


def _apply_1q(psi, n, k, u00, u01, u10, u11):
    """Apply a (possibly per-sample) 2x2 gate to qubit k.  u.. have shape (B,)."""
    B = psi.shape[0]
    p = psi.reshape(B, 1 << k, 2, 1 << (n - k - 1))
    a, b = p[:, :, 0, :], p[:, :, 1, :]
    u00, u01, u10, u11 = (u.reshape(-1, 1, 1) for u in (u00, u01, u10, u11))
    return torch.stack([u00 * a + u01 * b, u10 * a + u11 * b], dim=2).reshape(B, -1)


def _expi(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(torch.cos(x), torch.sin(x))


def simulate_state(phi: torch.Tensor, theta: torch.Tensor, theta_final: torch.Tensor,
                   entangle: bool = True, cdtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    phi         (B, n)          encoding angles
    theta       (L, n, 2)       shared variational angles, or (B, L, n, 2) per sample
    theta_final (n,)            shared final RY angles,    or (B, n) per sample
    entangle    False removes every CNOT (product-state ablation)
    returns     (B, 2^n) complex state vector
    """
    rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
    phi = phi.to(rdtype)
    B, n = phi.shape
    if theta.dim() == 3:
        theta = theta.unsqueeze(0).expand(B, *theta.shape)
    if theta_final.dim() == 1:
        theta_final = theta_final.unsqueeze(0).expand(B, -1)
    theta, theta_final = theta.to(rdtype), theta_final.to(rdtype)

    psi = encode_product_state(phi).to(cdtype)
    perm = _cached(_ring_permutation_cpu, n, psi.device) if (entangle and n > 1) else None

    for layer in range(theta.shape[1]):
        t1, t2 = theta[:, layer, :, 0], theta[:, layer, :, 1]            # (B, n)
        c, s = torch.cos(t1 / 2), torch.sin(t1 / 2)
        em, ep = _expi(-t2 / 2), _expi(t2 / 2)
        for k in range(n):                                             # RZ(t2) @ RY(t1)
            psi = _apply_1q(psi, n, k, em[:, k] * c[:, k], -em[:, k] * s[:, k],
                            ep[:, k] * s[:, k], ep[:, k] * c[:, k])
        if perm is not None:
            psi = psi.index_select(1, perm)

    c, s = torch.cos(theta_final / 2), torch.sin(theta_final / 2)
    for k in range(n):
        psi = _apply_1q(psi, n, k, c[:, k], -s[:, k], s[:, k], c[:, k])
    return psi


def expectations(psi: torch.Tensor, n: int) -> torch.Tensor:
    """(B, 2^n) state -> (B, 2n-1) values of <Z_k> and <Z_k Z_{k+1}>."""
    probs = psi.real ** 2 + psi.imag ** 2
    signs = _cached(_readout_signs_cpu, n, psi.device, probs.dtype)
    return probs @ signs


def simulate(phi, theta, theta_final, entangle=True, cdtype=torch.complex64):
    """Expectation values (B, 2n-1) of the circuit above."""
    return expectations(simulate_state(phi, theta, theta_final, entangle, cdtype), phi.shape[1])


# ---------------------------------------------------------------------------
# Parameter-shift rule (used for verification and the gradient-variance study)
# ---------------------------------------------------------------------------
def parameter_shift_gradient(fn, params: torch.Tensor, index: int | None = None) -> torch.Tensor:
    """
    Exact gradient of fn by the two-term shift rule, valid because every parameter
    sits in exactly one RY or RZ gate (generator eigenvalues +-1/2).

    fn      maps per-sample flat parameters (B, P) to per-sample values (B,) or (B, m)
    index   if given, differentiate only with respect to that parameter
    returns (B, P[, m]) or (B[, m]) when index is given
    """
    idx = range(params.shape[1]) if index is None else [index]
    grads = []
    with torch.no_grad():
        for j in idx:
            plus, minus = params.clone(), params.clone()
            plus[:, j] += math.pi / 2
            minus[:, j] -= math.pi / 2
            grads.append(0.5 * (fn(plus) - fn(minus)))
    return grads[0] if index is not None else torch.stack(grads, dim=1)


def unpack(flat: torch.Tensor, n: int, n_layers: int = 2):
    """Flat per-sample vector [phi (n) | theta (L*n*2) | theta_final (n)] -> the three tensors."""
    B = flat.shape[0]
    phi = flat[:, :n]
    theta = flat[:, n:n + 2 * n * n_layers].reshape(B, n_layers, n, 2)
    theta_final = flat[:, n + 2 * n * n_layers:]
    return phi, theta, theta_final


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------
class PQCBranch(nn.Module):
    """
    ResNet feature h (512) -> phi = enc_scale * tanh(W h + b) (n angles) -> circuit above
    -> 2n-1 expectation values -> Linear(2n-1 -> out_dim) -> BatchNorm1d(out_dim).

    enc_scale, batchnorm and entangle are exposed so the paper's "training fixes"
    (pi -> pi/4 encoding, BatchNorm) and the role of entanglement can be ablated on a
    circuit that is actually simulated.
    """

    def __init__(self, in_dim: int = 512, n_qubits: int = 8, n_layers: int = 2, out_dim: int = 32,
                 enc_scale: float = math.pi / 4, batchnorm: bool = True, entangle: bool = True,
                 init_range: float = 0.1, cdtype: torch.dtype = torch.complex64):
        super().__init__()
        self.n, self.n_layers = n_qubits, n_layers
        self.enc_scale, self.entangle, self.cdtype = float(enc_scale), entangle, cdtype
        self.proj = nn.Linear(in_dim, n_qubits)
        self.theta = nn.Parameter(torch.empty(n_layers, n_qubits, 2).uniform_(-init_range, init_range))
        self.theta_final = nn.Parameter(torch.empty(n_qubits).uniform_(-init_range, init_range))
        self.readout = nn.Linear(2 * n_qubits - 1, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity()

    def angles(self, h: torch.Tensor) -> torch.Tensor:
        return self.enc_scale * torch.tanh(self.proj(h.to(self.proj.weight.dtype)))

    def measurements(self, h: torch.Tensor) -> torch.Tensor:
        return simulate(self.angles(h), self.theta, self.theta_final, self.entangle, self.cdtype)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.bn(self.readout(self.measurements(h).to(self.readout.weight.dtype)))

    def circuit_parameters(self):
        return [self.theta, self.theta_final]


class MLPTwinBranch(nn.Module):
    """
    The PQC branch with only the circuit swapped for a classical map
    g(phi) = tanh(A phi + a), A: n -> 2n-1. Same projection, encoding, readout and
    BatchNorm as PQCBranch, so any accuracy difference is attributable to the circuit.
    """

    def __init__(self, in_dim: int = 512, n_qubits: int = 8, out_dim: int = 32,
                 enc_scale: float = math.pi / 4, batchnorm: bool = True):
        super().__init__()
        self.n, self.enc_scale = n_qubits, float(enc_scale)
        self.proj = nn.Linear(in_dim, n_qubits)
        self.mix = nn.Linear(n_qubits, 2 * n_qubits - 1)
        self.readout = nn.Linear(2 * n_qubits - 1, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if batchnorm else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        phi = self.enc_scale * torch.tanh(self.proj(h.to(self.proj.weight.dtype)))
        return self.bn(self.readout(torch.tanh(self.mix(phi))))
