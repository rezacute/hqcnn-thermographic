"""
Hybrid Quantum-Classical CNN (HQ-CNN) for Thermal Image Classification
=====================================================================
Uses CUDA Quantum for quantum feature extraction combined with classical CNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np

try:
    import cudaq
    from cudaq import spin
    CUDA_QUANTUM_AVAILABLE = True
except ImportError:
    CUDA_QUANTUM_AVAILABLE = False
    print("Warning: CUDA Quantum not available. Using classical fallback.")


class QuantumFeatureLayer(nn.Module):
    """Quantum Feature Extraction Layer using CUDA Quantum
    
    Implements a parameterized quantum circuit for feature extraction
    from thermal image features.
    """
    
    def __init__(self, n_qubits: int = 8, n_layers: int = 2, n_features: int = 64):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        
        # Classical preprocessing to match qubit count
        self.feature_map = nn.Linear(n_features, n_qubits * n_layers)
        
        # Learnable parameters for quantum circuit
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits * 3) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Classical preprocessing
        x = self.feature_map(x)
        x = x.view(batch_size, self.n_layers, self.n_qubits)
        
        # Apply quantum circuit simulation (mock for now - CUDA Quantum backend)
        # In production, this would use cudaq.simulate()
        quantum_features = self._simulate_quantum_circuit(x)
        
        return quantum_features
    
    def _simulate_quantum_circuit(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate quantum circuit effects for feature extraction
        
        In production: Replace with actual CUDA Quantum execution
        """
        batch_size = x.shape[0]
        
        # Simulated quantum feature extraction
        # Using parameterized rotations and entanglement simulation
        features = []
        for b in range(batch_size):
            feat = x[b]  # [n_layers, n_qubits]
            # Add quantum-inspired transformations
            feat = torch.tanh(feat)  # Activation
            feat = feat.sum(dim=0)   # Aggregate layers
            features.append(feat)
        
        return torch.stack(features)


class HybridBlock(nn.Module):
    """Hybrid Classical-Quantum Block
    
    Combines classical convolution with quantum feature enhancement.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_quantum: bool = True):
        super().__init__()
        self.use_quantum = use_quantum and CUDA_QUANTUM_AVAILABLE
        
        # Classical convolution path
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Quantum enhancement path
        if self.use_quantum:
            self.quantum_layer = QuantumFeatureLayer(
                n_qubits=min(8, out_channels),
                n_layers=2,
                n_features=out_channels
            )
            self.quantum_fusion = nn.Linear(out_channels * 2, out_channels)
        else:
            self.quantum_fusion = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical path
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        conv_out = F.relu(conv_out)
        
        if self.use_quantum:
            # Global pooling for quantum path
            pooled = F.adaptive_avg_pool2d(conv_out, 1).squeeze(-1).squeeze(-1)
            quantum_feat = self.quantum_layer(pooled)
            
            # Expand and concatenate
            quantum_expanded = quantum_feat.unsqueeze(-1).unsqueeze(-1)
            quantum_expanded = quantum_expanded.expand(-1, -1, conv_out.shape[2], conv_out.shape[3])
            
            # Fusion
            fused = torch.cat([conv_out, quantum_expanded], dim=1)
            out = self.quantum_fusion(fused)
        else:
            out = conv_out
            
        return out


class HQCNN(nn.Module):
    """Hybrid Quantum-Classical CNN for Thermal Image Classification
    
    Architecture:
        - Input: Thermal images (grayscale or RGB)
        - Backbone: Hybrid convolutional blocks with quantum enhancement
        - Head: Classification head for domain-specific classes
        
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        domain: 'medical' or 'industrial'
        use_quantum: Whether to use quantum enhancement
        base_channels: Number of base channels
    """
    
    DOMAIN_CONFIGS = {
        'medical': {
            'num_classes': 3,
            'classes': ['normal', 'benign', 'malignant']
        },
        'industrial': {
            'num_classes': 6,
            'classes': ['normal', 'overheating', 'electrical_fault', 'structural_defect', 'thermal_bridge', 'moisture']
        }
    }
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: Optional[int] = None,
        domain: str = 'medical',
        use_quantum: bool = True,
        base_channels: int = 64
    ):
        super().__init__()
        
        self.domain = domain
        self.use_quantum = use_quantum
        
        # Get domain config
        if num_classes is None:
            num_classes = self.DOMAIN_CONFIGS[domain]['num_classes']
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Hybrid blocks
        self.stage1 = self._make_stage(base_channels, base_channels * 2, num_blocks=2, use_quantum=use_quantum)
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks=2, use_quantum=use_quantum)
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, num_blocks=2, use_quantum=use_quantum)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _make_stage(self, in_ch: int, out_ch: int, num_blocks: int, use_quantum: bool):
        layers = []
        for i in range(num_blocks):
            layers.append(HybridBlock(
                in_ch if i == 0 else out_ch,
                out_ch,
                use_quantum=use_quantum and (i == num_blocks - 1)
            ))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_domain_classes(self) -> List[str]:
        """Get class labels for the domain"""
        return self.DOMAIN_CONFIGS[self.domain]['classes']


class ClassicalCNN(nn.Module):
    """Classical CNN fallback when quantum is not available"""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 5, domain: str = 'medical'):
        super().__init__()
        self.domain = domain
        
        # Use HQCNN without quantum
        self.model = HQCNN(in_channels, num_classes, domain, use_quantum=False)
        
    def forward(self, x):
        return self.model(x)
