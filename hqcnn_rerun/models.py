"""
models.py - one network, five interchangeable branches, one shared head.

    image -> ResNet18 (ImageNet) -> h in R^512 --+-------------------------------+--> head -> 2 logits
                                                 |                               |
                                                 +-> branch(h) in R^32 (optional)+

    head = Dropout(p) -> Linear(d, 256) -> ReLU -> BatchNorm1d(256) -> Dropout(0.6 p) -> Linear(256, 2)
    d = 512 (classical) or 544 (any branch)

Variants
    classical   no branch
    qi          the "quantum-inspired" layer exactly as in train_hq_cnn.py (see QIBranch)
    pqc         exactly simulated parameterized quantum circuit (pqc_torch.PQCBranch)
    pqc_noent   the same circuit with every CNOT removed (product states only)
    mlp_twin    the PQC branch with only the circuit replaced by tanh(A phi + a)

Everything outside the branch - backbone, head, loss, optimiser, schedule, augmentation,
data order and seeds - is identical across variants.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from pqc_torch import MLPTwinBranch, PQCBranch

VARIANTS = ("classical", "qi", "pqc", "pqc_noent", "mlp_twin")


class QIBranch(nn.Module):
    """
    Verbatim copy of QuantumInspiredLayer from train_hq_cnn.py (constructed the way
    HybridQuantumCNN constructs it, so n_heads = 2). Kept unchanged so the rerun
    reproduces the model behind the reported 91.25%.

    What it computes (verified numerically in verify_repo_layers.py):
      z0 = tanh(W_in h + b_in)                           (8 values)
      attention over a single token: the softmax weight is always 1, so the block
      reduces to the affine map  g(z) = W_O (W_V z + b_V) + b_O ; the query and key
      weights receive no gradient
      z1 = z0 + 0.3 g(z0),  z2 = z1 + 0.3 g(z1)          (same g both times)
      the "entanglement" matrices are added to a variable that is then overwritten,
      so they never affect the output and receive no gradient
      out = W_2 ReLU(W_1 [z2, |z2|] + b_1) + b_2         (Dropout 0.1 inside, 32 values)
    """

    def __init__(self, n_qubits=8, n_layers=2, input_dim=512, n_heads=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_projection = nn.Linear(input_dim, n_qubits)
        self.attention = nn.MultiheadAttention(embed_dim=n_qubits, num_heads=n_heads,
                                               dropout=0.05, batch_first=True)
        self.entangle_weights = nn.Parameter(torch.randn(n_layers, n_qubits, n_qubits) * 0.1)
        self.output_proj = nn.Sequential(nn.Linear(n_qubits * 2, 64), nn.ReLU(), nn.Dropout(0.1),
                                         nn.Linear(64, 32))

    def forward(self, x):
        x = torch.tanh(self.input_projection(x))
        x_attn = x.unsqueeze(1)
        for layer in range(self.n_layers):
            attn_out, _ = self.attention(x_attn, x_attn, x_attn)
            x_attn = x_attn + attn_out * 0.3
            entangle_matrix = torch.tanh(self.entangle_weights[layer])
            entangled = torch.matmul(x_attn.squeeze(1), entangle_matrix)
            x = x + entangled * 0.03
        x = x_attn.squeeze(1)
        features = torch.cat([x, torch.abs(x)], dim=1)
        return self.output_proj(features)


def make_head(d_in: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, 256), nn.ReLU(), nn.BatchNorm1d(256),
                         nn.Dropout(dropout * 0.6), nn.Linear(256, 2))


def make_backbone(pretrained: bool = True) -> nn.Module:
    from torchvision import models
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    net.fc = nn.Identity()
    return net


class HybridNet(nn.Module):
    def __init__(self, variant="classical", n_qubits=8, n_layers=2, enc_scale=math.pi / 4,
                 batchnorm=True, dropout=0.5, pretrained=True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.backbone = make_backbone(pretrained)
        if variant == "classical":
            self.branch = None
        elif variant == "qi":
            self.branch = QIBranch(n_qubits=n_qubits, n_layers=n_layers, input_dim=512)
        elif variant in ("pqc", "pqc_noent"):
            self.branch = PQCBranch(512, n_qubits, n_layers, 32, enc_scale, batchnorm,
                                    entangle=(variant == "pqc"))
        else:
            self.branch = MLPTwinBranch(512, n_qubits, 32, enc_scale, batchnorm)
        self.head = make_head(512 + (0 if self.branch is None else 32), dropout)

    def features(self, x):
        """Backbone features (float32) and branch output (or None)."""
        h = self.backbone(x).float()
        if self.branch is None:
            return h, None
        with torch.autocast(device_type=h.device.type, enabled=False):
            q = self.branch(h)
        return h, q

    def classify(self, h, q):
        z = h if q is None else torch.cat([h, q.float()], dim=1)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.head(z.float())

    def forward(self, x):
        return self.classify(*self.features(x))

    def param_groups(self, lr_backbone, lr_branch, lr_head, weight_decay):
        circuit = set()
        if isinstance(self.branch, PQCBranch):
            circuit = {id(p) for p in self.branch.circuit_parameters()}
        groups = [dict(params=list(self.backbone.parameters()), lr=lr_backbone, weight_decay=weight_decay,
                       name="backbone"),
                  dict(params=list(self.head.parameters()), lr=lr_head, weight_decay=weight_decay, name="head")]
        if self.branch is not None:
            other = [p for p in self.branch.parameters() if id(p) not in circuit]
            groups.append(dict(params=other, lr=lr_branch, weight_decay=weight_decay, name="branch"))
            if circuit:   # rotation angles are periodic: no weight decay
                groups.append(dict(params=[p for p in self.branch.parameters() if id(p) in circuit],
                                   lr=lr_branch, weight_decay=0.0, name="circuit"))
        return groups

    def count_parameters(self) -> dict:
        count = lambda m: sum(p.numel() for p in m.parameters()) if m is not None else 0
        out = dict(total=count(self), backbone=count(self.backbone), head=count(self.head),
                   branch=count(self.branch), circuit=0)
        if isinstance(self.branch, PQCBranch):
            out["circuit"] = sum(p.numel() for p in self.branch.circuit_parameters())
        if isinstance(self.branch, QIBranch):
            dead = self.branch.entangle_weights.numel() + 2 * self.branch.n_qubits ** 2 + 2 * self.branch.n_qubits
            out["branch_effective"] = out["branch"] - dead
        return out
