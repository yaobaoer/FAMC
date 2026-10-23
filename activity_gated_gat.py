                       
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


FRONT_KEYS = (
    'device_projectors', 'share_logit', 'act_temp',
       
)




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


    def align_nodes(self, h):
        return self.calibrate_cross_protocol(h)

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
