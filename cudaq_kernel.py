"""
cudaq_kernel.py - the HQ-CNN circuit as a CUDA-Q kernel.

Same circuit as pqc_torch.py:  RY(phi) on every qubit, then for each layer RY(theta) and
RZ(theta) on every qubit followed by the CNOT ring (0,1),(1,2),...,(n-1,0), then a final RY.
`entangle = 0` removes every CNOT (the product-state control).

Imported only when the CUDA-Q backend is requested, so the rest of the pipeline runs
without CUDA-Q installed.
"""
import sys

if "cudaq" not in sys.modules:
    # Load torchvision (which loads Triton's LLVM) before CUDA-Q. The opposite order segfaults
    # in triton/knobs.py with some builds (seen with torch 2.14 + CUDA-Q 0.16). If CUDA-Q is
    # already loaded, importing torchvision now would crash, so it is left alone.
    try:
        import torchvision  # noqa: F401
    except Exception:
        pass

import cudaq  # noqa: E402
from cudaq import spin  # noqa: E402


@cudaq.kernel
def hqcnn_kernel(n: int, layers: int, entangle: int, phi: list[float], theta: list[float],
                 theta_final: list[float]):
    q = cudaq.qvector(n)
    for k in range(n):
        ry(phi[k], q[k])
    for layer in range(layers):
        for k in range(n):
            ry(theta[(layer * n + k) * 2], q[k])
            rz(theta[(layer * n + k) * 2 + 1], q[k])
        if entangle == 1:
            if n > 1:
                for k in range(n - 1):
                    x.ctrl(q[k], q[k + 1])
                x.ctrl(q[n - 1], q[0])
    for k in range(n):
        ry(theta_final[k], q[k])


def readout_terms(n: int):
    """<Z_k> for k = 0..n-1, then <Z_k Z_{k+1}> for k = 0..n-2 (same order as pqc_torch)."""
    terms = [spin.z(k) for k in range(n)] + [spin.z(k) * spin.z(k + 1) for k in range(n - 1)]
    ham = terms[0]
    for t in terms[1:]:
        ham = ham + t
    return terms, ham
