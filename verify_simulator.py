#!/usr/bin/env python3
"""
verify_simulator.py - independent checks of pqc_torch.py.

  1. Amplitudes vs a dense NumPy reference built from explicit 2^n x 2^n matrices (n = 1..6).
  2. Expectation values vs Qiskit's Statevector (n = 2..10), if qiskit is installed.
  3. Expectation values vs CUDA-Q's CPU simulator (n = 2..10), if cudaq is installed.
  4. Autograd gradients vs the parameter-shift rule, for every circuit and encoding angle.
  5. torch.autograd.gradcheck in double precision.
  6. Product-state ablation: without CNOTs, <Z_k Z_k+1> = <Z_k><Z_k+1>.
  7. PQCBranch bookkeeping: 5n circuit parameters, shared-angle gradient = sum over images.
  8. The CUDA-Q training back end (pqc_cudaq.py: cudaq.observe forward, parameter-shift
     backward) gives the same readouts and gradients as the PyTorch simulator.

Exit code is non-zero if any check fails.  Run:  python verify_simulator.py
"""
import math
import sys

import numpy as np
import torch

try:  # load torchvision before CUDA-Q (see cudaq_kernel.py)
    import torchvision  # noqa: F401
except Exception:
    pass

from pqc_torch import (PQCBranch, n_circuit_params, parameter_shift_gradient, ring_cnot_pairs,
                       simulate, simulate_state, unpack)

torch.manual_seed(1234)
rng = np.random.default_rng(1234)
L = 2
failures = []


def check(name, err, tol):
    ok = err <= tol
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:62s} max error {err:.2e} (tol {tol:.0e})")
    if not ok:
        failures.append(name)


def random_params(n, B=1, dtype=torch.float64):
    return torch.rand(B, n + 2 * n * L + n, dtype=dtype) * 2 * math.pi - math.pi


# ---------------------------------------------------------------- 1. dense NumPy reference
def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def on(q, g, n):
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, g if k == q else np.eye(2))
    return out


def cnot(c, t, n):
    N = 1 << n
    U = np.zeros((N, N))
    for x in range(N):
        y = x ^ ((((x >> (n - 1 - c)) & 1)) << (n - 1 - t))
        U[y, x] = 1.0
    return U


def reference_state(phi, theta, theta_final, n, entangle=True):
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for k in range(n):
        psi = on(k, ry(phi[k]), n) @ psi
    for layer in range(L):
        for k in range(n):
            psi = on(k, ry(theta[layer, k, 0]), n) @ psi
            psi = on(k, rz(theta[layer, k, 1]), n) @ psi
        if entangle:
            for c, t in ring_cnot_pairs(n):
                psi = cnot(c, t, n) @ psi
    for k in range(n):
        psi = on(k, ry(theta_final[k]), n) @ psi
    return psi


print("1. Amplitudes vs dense NumPy matrices")
for n in range(1, 7):
    for entangle in (True, False):
        flat = random_params(n, B=3)
        phi, th, thf = unpack(flat, n, L)
        got = simulate_state(phi, th, thf, entangle=entangle, cdtype=torch.complex128).numpy()
        ref = np.stack([reference_state(phi[b].numpy(), th[b].numpy(), thf[b].numpy(), n, entangle)
                        for b in range(3)])
        check(f"n={n} entangle={entangle}", np.abs(got - ref).max(), 1e-12)


# ---------------------------------------------------------------- 2. Qiskit
print("\n2. Expectation values vs Qiskit Statevector")
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    for n in range(2, 11):
        flat = random_params(n, B=1)
        phi, th, thf = (x[0].numpy() for x in unpack(flat, n, L))
        qc = QuantumCircuit(n)
        for k in range(n):
            qc.ry(phi[k], k)
        for layer in range(L):
            for k in range(n):
                qc.ry(th[layer, k, 0], k)
                qc.rz(th[layer, k, 1], k)
            for c, t in ring_cnot_pairs(n):
                qc.cx(c, t)
        for k in range(n):
            qc.ry(thf[k], k)
        sv = Statevector(qc)
        ops = [SparsePauliOp.from_sparse_list([("Z", [k], 1.0)], n) for k in range(n)]
        ops += [SparsePauliOp.from_sparse_list([("ZZ", [k, k + 1], 1.0)], n) for k in range(n - 1)]
        ref = np.array([sv.expectation_value(o).real for o in ops])
        got = simulate(*unpack(flat, n, L), cdtype=torch.complex128)[0].numpy()
        check(f"n={n}  (2n-1 = {2 * n - 1} observables)", np.abs(got - ref).max(), 1e-10)
except ImportError:
    print("  qiskit not installed - skipped")


# ---------------------------------------------------------------- 3. CUDA-Q
print("\n3. Expectation values vs CUDA-Q (qpp-cpu simulator)")
try:
    import cudaq
    from cudaq import spin

    cudaq.set_target("qpp-cpu")

    @cudaq.kernel
    def hqcnn_kernel(n: int, layers: int, phi: list[float], theta: list[float], theta_final: list[float]):
        q = cudaq.qvector(n)
        for k in range(n):
            ry(phi[k], q[k])
        for layer in range(layers):
            for k in range(n):
                ry(theta[(layer * n + k) * 2], q[k])
                rz(theta[(layer * n + k) * 2 + 1], q[k])
            for k in range(n - 1):
                x.ctrl(q[k], q[k + 1])
            x.ctrl(q[n - 1], q[0])
        for k in range(n):
            ry(theta_final[k], q[k])

    for n in range(2, 11):
        flat = random_params(n, B=1)
        phi, th, thf = (x[0] for x in unpack(flat, n, L))
        args = (n, L, phi.tolist(), th.reshape(-1).tolist(), thf.tolist())
        ops = [spin.z(k) for k in range(n)] + [spin.z(k) * spin.z(k + 1) for k in range(n - 1)]
        ref = np.array([cudaq.observe(hqcnn_kernel, o, *args).expectation() for o in ops])
        got = simulate(*unpack(flat, n, L), cdtype=torch.complex128)[0].numpy()
        check(f"n={n}", np.abs(got - ref).max(), 1e-8)
except Exception as exc:  # ImportError or runtime errors from a missing target
    print(f"  cudaq unavailable ({type(exc).__name__}: {exc}) - skipped")


# ---------------------------------------------------------------- 4. autograd vs shift rule
print("\n4. Autograd vs parameter-shift gradients (all encoding + circuit angles)")
for n in (1, 2, 3, 5, 8):
    B = 4
    flat = random_params(n, B=B)
    fn = lambda f, n=n: simulate(*unpack(f, n, L), cdtype=torch.complex128)   # (B, m)
    shift = parameter_shift_gradient(fn, flat)                                   # (B, P, m)
    f = flat.clone().requires_grad_(True)
    out = fn(f)
    auto = torch.stack([torch.autograd.grad(out[:, j].sum(), f, retain_graph=True)[0]
                        for j in range(out.shape[1])], dim=2)
    check(f"n={n}  ({flat.shape[1]} angles x {out.shape[1]} observables)", (auto - shift).abs().max().item(), 1e-12)


# ---------------------------------------------------------------- 5. gradcheck
print("\n5. torch.autograd.gradcheck (float64)")
n = 3
flat = random_params(n, B=2).requires_grad_(True)
ok = torch.autograd.gradcheck(lambda f: simulate(*unpack(f, n, L), cdtype=torch.complex128), (flat,),
                              eps=1e-6, atol=1e-8)
check("gradcheck n=3", 0.0 if ok else 1.0, 0.0)


# ---------------------------------------------------------------- 6. product-state ablation
print("\n6. No-CNOT ablation factorises")
for n in (2, 4, 8):
    flat = random_params(n, B=5)
    e = simulate(*unpack(flat, n, L), entangle=False, cdtype=torch.complex128)
    z, zz = e[:, :n], e[:, n:]
    check(f"n={n}  <ZZ> = <Z><Z>", (zz - z[:, :-1] * z[:, 1:]).abs().max().item(), 1e-12)


# ---------------------------------------------------------------- 7. module bookkeeping
print("\n7. PQCBranch parameter counts and shared-parameter gradients")
for n in (2, 4, 8, 12, 16):
    m = PQCBranch(n_qubits=n)
    got = sum(p.numel() for p in m.circuit_parameters())
    check(f"n={n}: circuit parameters = 5n = {n_circuit_params(n)}", abs(got - n_circuit_params(n)), 0)
m = PQCBranch(n_qubits=4, cdtype=torch.complex128).double()
h = torch.randn(6, 512, dtype=torch.float64)
out = m.measurements(h).sum()
g_auto = torch.autograd.grad(out, m.theta)[0]
phi = m.angles(h).detach()
flat = torch.cat([phi, m.theta.detach().reshape(1, -1).expand(6, -1),
                  m.theta_final.detach().reshape(1, -1).expand(6, -1)], dim=1)
g_shift = parameter_shift_gradient(lambda f: simulate(*unpack(f, 4, L), cdtype=torch.complex128).sum(1), flat)
g_shift = g_shift[:, 4:4 + 16].sum(0).reshape(2, 4, 2)          # shared parameter = sum over samples
check("shared theta gradient = sum of per-sample shift gradients", (g_auto - g_shift).abs().max().item(), 1e-10)

# ---------------------------------------------------------------- 8. CUDA-Q training backend
print("\n8. CUDA-Q training backend (forward via cudaq.observe, parameter-shift backward) vs PyTorch")
try:
    import cudaq  # noqa: F401
    target = "nvidia" if torch.cuda.is_available() else "qpp-cpu"
    tol = 1e-5 if target == "nvidia" else 1e-10          # the GPU target runs in single precision
    for n in (1, 2, 3, 5):
        for entangle in (True, False):
            torch.manual_seed(n)
            ref = PQCBranch(n_qubits=n, entangle=entangle, cdtype=torch.complex128).double()
            cq = PQCBranch(n_qubits=n, entangle=entangle, backend="cudaq", cudaq_target=target).double()
            cq.load_state_dict(ref.state_dict())
            h = torch.randn(3, 512, dtype=torch.float64)
            w = torch.randn(3, 2 * n - 1, dtype=torch.float64)          # random upstream gradient
            outs, grads = [], []
            for m in (ref, cq):
                m.zero_grad()
                e = m.measurements(h)
                (e * w).sum().backward()
                outs.append(e.detach())
                grads.append(torch.cat([m.theta.grad.reshape(-1), m.theta_final.grad, m.proj.weight.grad.reshape(-1)]))
            check(f"n={n} entangle={entangle}: readouts", (outs[0] - outs[1]).abs().max().item(), tol)
            check(f"n={n} entangle={entangle}: gradients (angles + encoding layer)",
                  (grads[0] - grads[1]).abs().max().item(), tol)
except Exception as exc:
    print(f"  cudaq unavailable ({type(exc).__name__}: {exc}) - skipped")

print("\nALL CHECKS PASSED" if not failures else f"\n{len(failures)} CHECK(S) FAILED: {failures}")
sys.exit(1 if failures else 0)
