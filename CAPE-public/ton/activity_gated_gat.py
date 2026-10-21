                       
pass
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

DEVICE_STRONG = ((4, 5), (5, 4), (2, 3), (3, 2), (4, 6), (6, 4))
DEVICE_WEAK = ((0, 1), (1, 0), (0, 2), (2, 0))

FRONT_KEYS = (
    'device_projectors', 'share_logit', 'act_temp',
    'prior_weight', 'dynamic_prior_gate', 'gat1', 'n1',
)


def build_device_prior(num_nodes: int) -> torch.Tensor:
    pass
    pw = torch.zeros((num_nodes, num_nodes))
    for i, j in DEVICE_STRONG:
        if i < num_nodes and j < num_nodes:
            pw[i, j] = 1.0
    for i, j in DEVICE_WEAK:
        if i < num_nodes and j < num_nodes:
            pw[i, j] = 0.2
    pw[pw == 0] = 0.1
    return pw


class ActivityGatedGAT(nn.Module):
    pass

    def __init__(
        self,
        device_dims: Sequence[int],
        d_model: int = 16,
        gat_heads: int = 1,
        dropout: float = 0.1,
        flatten: bool = True,
    ):
        super().__init__()
        self.device_dims = list(device_dims)
        self.num_devices = len(self.device_dims)
        self.d_model = d_model
        self.flatten = flatten
        self.out_dim = d_model * self.num_devices

        self.device_projectors = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU())
            for dim in self.device_dims
        ])
        self.share_logit = nn.Parameter(torch.tensor(-2.0))
        self.act_temp = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('prior_weight', build_device_prior(self.num_devices))
        self.dynamic_prior_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.gat1 = GATv2Conv(
            d_model, d_model, heads=gat_heads, concat=False,
            dropout=dropout, add_self_loops=False, edge_dim=1)
        self.n1 = nn.LayerNorm(d_model)

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.share_logit)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        pass
        feats, split_idx = [], 0
        for proj, dim in zip(self.device_projectors, self.device_dims):
            feats.append(proj(x[:, :, split_idx:split_idx + dim]))
            split_idx += dim
        return torch.stack(feats, dim=2)

    def calibrate_cross_protocol(self, device_feats: torch.Tensor) -> torch.Tensor:
        pass
        act = device_feats.norm(dim=-1, keepdim=True)
        temp = self.act_temp.clamp(0.2, 5.0)
        w = torch.softmax(act / temp, dim=2)
        shared = (device_feats * w).sum(dim=2, keepdim=True)
        gamma = torch.sigmoid(self.share_logit)
        return (1.0 - gamma) * device_feats + gamma * shared

    def compute_dynamic_edges(self, pooled: torch.Tensor):
        B, N, _ = pooled.shape
        device = pooled.device
        ei = torch.tensor(
            [[i, j] for i in range(N) for j in range(N) if i != j],
            dtype=torch.long, device=device).t().contiguous()
        normed = F.normalize(pooled, p=2, dim=-1)
        sim = torch.matmul(normed, normed.transpose(1, 2))
        dyn = torch.stack([sim[:, i, j] for i, j in zip(ei[0], ei[1])], dim=-1)
        fi, fj = pooled[:, ei[0]], pooled[:, ei[1]]
        alpha = self.dynamic_prior_gate(torch.cat([fi, fj], -1)).squeeze(-1)
        prior = self.prior_weight[ei[0], ei[1]].view(1, -1)
        agree = ((fi * fj).sum(-1) / (fi.norm(dim=-1) * fj.norm(dim=-1) + 1e-6)).clamp(0, 1)
        ew = ((alpha * prior + (1 - alpha) * dyn) * (0.5 + 0.5 * agree)).reshape(-1, 1)
        off = torch.arange(B, device=device) * N
        eib = (ei.unsqueeze(1) + off.view(1, -1, 1)).reshape(2, -1)
        return eib, ew

    def align_nodes(self, h: torch.Tensor) -> torch.Tensor:
        pass
        B, T, N, D = h.shape
        h_prime = self.calibrate_cross_protocol(h)
        pooled = h_prime.mean(1)
        edge_index, edge_weights = self.compute_dynamic_edges(pooled)
        g = pooled.reshape(-1, D)
        g = self.n1(g + self.gat1(g, edge_index, edge_attr=edge_weights))
        return h_prime + g.reshape(B, 1, N, D)

    def forward(self, x: torch.Tensor, flatten: Optional[bool] = None) -> torch.Tensor:
        z = self.align_nodes(self.project(x))
        use_flat = self.flatten if flatten is None else flatten
        if use_flat:
            return z.reshape(z.size(0), z.size(1), -1)
        return z

    def load_from_ultralite(
        self,
        ckpt_or_state: Union[str, Dict],
        strict: bool = True,
    ) -> Tuple[list, list]:
        pass
        if isinstance(ckpt_or_state, str):
            raw = torch.load(ckpt_or_state, map_location='cpu', weights_only=False)
        else:
            raw = ckpt_or_state
        sd = raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else raw
        sub = {k: v for k, v in sd.items() if k.split('.', 1)[0] in FRONT_KEYS}
        return self.load_state_dict(sub, strict=strict)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == '__main__':
    dims = [3, 2, 2, 2, 2, 3, 3]
    x = torch.randn(2, 16, sum(dims))
    m = ActivityGatedGAT(dims, d_model=16)
    z = m(x)
    z_node = m(x, flatten=False)
    print('Z_flat', tuple(z.shape), 'Z_node', tuple(z_node.shape),
          f'γ={float(m.gamma):.3f}', f'params={n_params(m)}')
