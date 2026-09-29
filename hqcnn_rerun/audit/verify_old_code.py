"""Run the old repo's own layer code to confirm what it computes.  Usage: python verify_old_code.py /path/to/hqcnn-thermographic"""
import sys, types, importlib.util, numpy as np, torch
REPO = sys.argv[1] if len(sys.argv) > 1 else "hqcnn-thermographic"

# ---------- 1. Quantum-inspired layer (train_hq_cnn.py) ----------
spec = importlib.util.spec_from_file_location("hq", f"{REPO}/train_hq_cnn.py")
hq = importlib.util.module_from_spec(spec); spec.loader.exec_module(hq)
torch.manual_seed(0)
layer = hq.QuantumInspiredLayer(n_qubits=8, n_layers=2, input_dim=512)   # how HybridQuantumCNN builds it
print("QI attention heads actually used:", layer.attention.num_heads)
h = torch.randn(64, 512)
out = layer(h); out.pow(2).sum().backward()
g = layer.entangle_weights.grad
print("entangle_weights gradient:", "None (not in the computation graph)" if g is None else g.abs().sum().item())
print("grad norm of attention Q/K rows (single-token softmax):", layer.attention.in_proj_weight.grad[:16].abs().sum().item())
# closed form: single-token attention == affine map W_O (W_V z + b_V) + b_O applied twice with 0.3 residual
layer.eval()
with torch.no_grad():
    z = torch.tanh(layer.input_projection(h))
    Wv = layer.attention.in_proj_weight[16:24]; bv = layer.attention.in_proj_bias[16:24]
    Wo = layer.attention.out_proj.weight; bo = layer.attention.out_proj.bias
    for _ in range(2):
        z = z + 0.3 * ((z @ Wv.T + bv) @ Wo.T + bo)
    ref = layer.output_proj(torch.cat([z, z.abs()], 1))
    print("max |layer(h) - closed form| (eval):", (layer(h) - ref).abs().max().item())
n_all = sum(p.numel() for p in layer.parameters())
print("QI layer parameters:", n_all, "| dead (entangle + Q/K):", layer.entangle_weights.numel() + 2*8*8 + 2*8)

# ---------- 2. CUDA-Q layer (train_cuda_quantum.py), with a stub cudaq so it imports ----------
cq = types.ModuleType("cudaq")
cq.set_target = lambda *a, **k: None
cq.kernel = lambda f: f
class _S:
    def __mul__(self, o): return self
cq.spin = types.SimpleNamespace(z=lambda i: _S())
sys.modules["cudaq"] = cq
spec = importlib.util.spec_from_file_location("cq", f"{REPO}/train_cuda_quantum.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.manual_seed(0)
ql = m.CUDAQuantumLayer(input_dim=512, n_qubits=8)
xa, xb = torch.randn(16, 512), torch.randn(16, 512) * 5 + 3   # two unrelated batches
ql.train()
qa, qb = m.CUDAQuantumFunction.apply(xa, ql.quantum_params, 8), m.CUDAQuantumFunction.apply(xb, ql.quantum_params, 8)
print("\nPQC train mode: 'measurements' identical for unrelated inputs:", torch.equal(qa, qb))
ql.eval()
with torch.no_grad():
    e1, e2 = ql(xa), ql(xa)
print("PQC eval mode: same input gives same output:", torch.allclose(e1, e2))
# gradients returned by the custom backward
p = ql.quantum_params.detach().clone().requires_grad_(True)
xg = xa.clone().requires_grad_(True)
y = m.CUDAQuantumFunction.apply(xg, p, 8); (y * torch.randn_like(y)).sum().backward()
print("PQC 'parameter-shift' grads, first 10 params:", np.round(p.grad[:10].numpy(), 4))
print("PQC param count:", ql.quantum_params.numel(), "| circuit kernel called anywhere:",
      "quantum_circuit_kernel(" in open(f"{REPO}/train_cuda_quantum.py").read().split("def quantum_circuit_kernel")[1])
