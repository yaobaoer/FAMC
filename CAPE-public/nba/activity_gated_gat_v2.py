                       
pass
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

DEVICE_STRONG = ((4, 5), (5, 4), (2, 3), (3, 2), (4, 6), (6, 4))
DEVICE_WEAK = ((0, 1), (1, 0), (0, 2), (2, 0))

FRONT_KEYS = (
    'device_projectors', 'act_temp', 'prior_weight',
    'dynamic_prior_gate', 'gat1', 'n1', 'dim_gate', 'domain_clf',
)


def build_device_prior(num_nodes: int) -> torch.Tensor:
    pass
    pw = torch.zeros((num_nodes, num_nodes))
    if num_nodes >= 7:
        for i, j in DEVICE_STRONG:
            if i < num_nodes and j < num_nodes:
                pw[i, j] = 1.0
        for i, j in DEVICE_WEAK:
            if i < num_nodes and j < num_nodes:
                pw[i, j] = 0.2
        pw[pw == 0] = 0.1
                              
        if num_nodes > 7:
            pw[7:, :] = 0.5
            pw[:, 7:] = 0.5
            pw.fill_diagonal_(0.0)
        return pw
    pw = torch.full((num_nodes, num_nodes), 0.5)
    pw.fill_diagonal_(0.0)
    return pw


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, alpha)


class ActivityGatedGAT(nn.Module):
    pass

    def __init__(
        self,
        device_dims: Sequence[int],
        d_model: int = 16,
        gat_heads: int = 1,
        dropout: float = 0.1,
        flatten: bool = True,
        num_protocols: Optional[int] = None,
        axis_residual: bool = False,
    ):
        super().__init__()
        self.device_dims = list(device_dims)
        self.num_devices = len(self.device_dims)
        self.d_model = d_model
        self.flatten = flatten
        self.axis_residual = bool(axis_residual)
        self.out_dim = d_model * self.num_devices
        self.num_protocols = num_protocols or self.num_devices
        self._raw_valid = None
        self._raw_feat = None
        self.pass_through = None

        self.device_projectors = nn.ModuleList()
        for dim in self.device_dims:
            if self.axis_residual:
                if int(dim) != int(d_model):
                    raise ValueError(f'axis residual needs d_model == dim, got {d_model} != {dim}')
                last = nn.Linear(dim, dim)
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
                self.device_projectors.append(nn.Sequential(
                    nn.Linear(dim, dim), nn.GELU(), last))
            elif int(dim) >= 64:
                self.device_projectors.append(nn.Sequential(
                    nn.Linear(dim, d_model * 2), nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model), nn.GELU()))
            else:
                self.device_projectors.append(nn.Sequential(
                    nn.Linear(dim, d_model), nn.LayerNorm(d_model), nn.GELU()))
        self.act_temp = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('prior_weight', build_device_prior(self.num_devices))
        self.dynamic_prior_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, 1), nn.Sigmoid())
        self.use_gat = True
        self.gat1 = GATv2Conv(
            d_model, d_model, heads=gat_heads, concat=False,
            dropout=dropout, add_self_loops=False, edge_dim=1)
        self.n1 = nn.LayerNorm(d_model)
        self.gate_target = 0.3
        self.dim_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid())
        self.set_gate_target(0.3)
        self.domain_clf = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, self.num_protocols))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        feats, raws, valids, split_idx = [], [], [], 0
        for proj, dim in zip(self.device_projectors, self.device_dims):
            chunk = x[:, :, split_idx:split_idx + dim]
            raws.append(chunk)
            valids.append(chunk.detach().norm(dim=-1) > 1e-6)
            if self.axis_residual:
                feats.append(chunk + proj(chunk))
            else:
                feats.append(proj(chunk))
            split_idx += dim
        self._raw_valid = torch.stack(valids, dim=2) if valids else None
        self._raw_feat = torch.stack(raws, dim=2) if self.axis_residual else None
        return torch.stack(feats, dim=2)

    def node_mask(self, h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pass
        raw = self._raw_valid
        if raw is not None and raw.shape[:3] == h.shape[:3] and raw.device == h.device:
            return raw
        return h.norm(dim=-1) > eps

    def calibrate_cross_protocol(self, h: torch.Tensor):
        pass
        B, T, N, D = h.shape
        valid = self.node_mask(h)           
        vote = valid
        pt = getattr(self, 'pass_through', None)
        if pt:
            vote = valid.clone()
            vote[:, :, list(pt)] = False
        n_valid = vote.float().sum(dim=2, keepdim=True).clamp_min(1.0)           
        h_masked = h * vote.unsqueeze(-1)
        mean_all = h_masked.sum(dim=2, keepdim=True) / n_valid.unsqueeze(-1)
                                
        mean_others = (mean_all * n_valid.unsqueeze(-1) - h_masked) / (n_valid - vote.float()).unsqueeze(-1).clamp_min(1.0)
        consistency = F.cosine_similarity(h, mean_others, dim=-1, eps=1e-6)
        consistency = consistency.masked_fill(~vote, -1e9)
        temp = self.act_temp.clamp(0.2, 5.0)
        w = torch.softmax(consistency.unsqueeze(-1) / temp, dim=2)
        w = w * vote.unsqueeze(-1)        
        shared = (h * w).sum(dim=2, keepdim=True)
        gate = self.dim_gate(h) * vote.unsqueeze(-1)
        h_prime = (1.0 - gate) * h + gate * shared
        h_prime = h_prime * vote.unsqueeze(-1)
        if pt and self._raw_feat is not None:
            idx = list(pt)
            h_prime[:, :, idx] = self._raw_feat[:, :, idx]
        return h_prime, shared, gate

    def set_gate_target(self, target: float) -> float:
        pass
        t = float(min(max(target, 1e-4), 1.0 - 1e-4))
        self.gate_target = t
        last = self.dim_gate[2]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, math.log(t / (1.0 - t)))
        return t

    def gate_reg_loss(self, gate: torch.Tensor) -> torch.Tensor:
        pass
        mean_pen = (gate.mean() - self.gate_target) ** 2
        sat_pen = (-torch.log(1.0 - gate.clamp(max=1.0 - 1e-4))).mean()
        return mean_pen + 0.05 * sat_pen

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

    def align_nodes(self, h: torch.Tensor):
        B, T, N, D = h.shape
        h_prime, shared, gate = self.calibrate_cross_protocol(h)
        if not getattr(self, 'use_gat', True):
            return h_prime, h_prime, shared, gate
        valid = self.node_mask(h).any(dim=1)                  
        pooled = h_prime.mean(1)
                                        
        pooled = pooled * valid.unsqueeze(-1).float()
        edge_index, edge_weights = self.compute_dynamic_edges(pooled)
                    
        src, dst = edge_index[0] % N, edge_index[1] % N
        b_idx = edge_index[0] // N
        ok = valid[b_idx, src] & valid[b_idx, dst]
        edge_weights = edge_weights * ok.float().unsqueeze(-1)
        g = pooled.reshape(-1, D)
        g = self.n1(g + self.gat1(g, edge_index, edge_attr=edge_weights))
        g = g.reshape(B, N, D) * valid.unsqueeze(-1).float()
        z = h_prime + g.reshape(B, 1, N, D)
        return z, h_prime, shared, gate

    def forward(
        self,
        x: torch.Tensor,
        flatten: Optional[bool] = None,
        return_features: bool = False,
    ):
        z, h_prime, shared, gate = self.align_nodes(self.project(x))
        use_flat = self.flatten if flatten is None else flatten
        z_out = z.reshape(z.size(0), z.size(1), -1) if use_flat else z
        if return_features:
            return z_out, {'h_prime': h_prime, 'shared': shared, 'gate': gate}
        return z_out

    def cross_protocol_contrastive_loss(
        self,
        h_prime: torch.Tensor,
        attack_labels: torch.Tensor,
        temperature: float = 0.1,
        protocol_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = h_prime.mean(dim=1)
        B, N, D = feat.shape
                             
        node_ok = feat.norm(dim=-1) > 1e-6         
        feat_flat = feat.reshape(-1, D)
        attack_expand = attack_labels.unsqueeze(1).expand(-1, N).reshape(-1)
        if protocol_ids is None:
            protocol_ids = torch.arange(N, device=feat.device)
        proto_expand = protocol_ids.unsqueeze(0).expand(B, -1).reshape(-1)
        ok_flat = node_ok.reshape(-1)

        feat_norm = F.normalize(feat_flat, dim=-1)
        sim = (feat_norm @ feat_norm.t()) / temperature
        same_attack = attack_expand.unsqueeze(0) == attack_expand.unsqueeze(1)
        diff_proto = proto_expand.unsqueeze(0) != proto_expand.unsqueeze(1)
        both_ok = ok_flat.unsqueeze(0) & ok_flat.unsqueeze(1)
        pos_mask = same_attack & diff_proto & both_ok
        pos_mask.fill_diagonal_(False)

        sim = sim - sim.max(dim=-1, keepdim=True).values
        exp_sim = torch.exp(sim) * both_ok.float()
        denom = exp_sim.sum(dim=-1) - torch.diagonal(exp_sim)
        numer = (exp_sim * pos_mask.float()).sum(dim=-1)
        has_pos = pos_mask.any(dim=-1) & ok_flat
        if not bool(has_pos.any()):
            return feat_flat.sum() * 0.0
        return -torch.log(numer[has_pos] / (denom[has_pos] + 1e-8) + 1e-8).mean()

    def domain_adversarial_loss(
        self,
        h_prime: torch.Tensor,
        alpha: float = 1.0,
        protocol_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = h_prime.mean(dim=1)
        B, N, D = feat.shape
        feat_flat = feat.reshape(-1, D)
        if protocol_ids is None:
            protocol_ids = torch.arange(N, device=feat.device)
        labels = protocol_ids.unsqueeze(0).expand(B, -1).reshape(-1)
        valid = self.node_mask(h_prime).any(dim=1).reshape(-1)
        reversed_feat = grad_reverse(feat_flat, alpha)
        logits = self.domain_clf(reversed_feat)
        if not bool(valid.any()):
            return feat_flat.sum() * 0.0
        return F.cross_entropy(logits[valid], labels[valid])

    def load_from_ultralite(
        self,
        ckpt_or_state: Union[str, Dict],
        strict: bool = False,
    ) -> Tuple[list, list]:
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
    x = torch.randn(4, 16, sum(dims))
    attack_labels = torch.tensor([0, 1, 0, 1])
    m = ActivityGatedGAT(dims, d_model=16)
    z, feats = m(x, return_features=True)
    loss_task = z.mean()
    loss_contra = m.cross_protocol_contrastive_loss(feats['h_prime'], attack_labels)
    loss_domain = m.domain_adversarial_loss(feats['h_prime'], alpha=0.5)
    loss_gate = m.gate_reg_loss(feats['gate'])
    loss = loss_task + 0.1 * loss_contra + 0.05 * loss_domain + 1.0 * loss_gate
    loss.backward()
    print('Z_flat', tuple(z.shape), 'gate', f"{feats['gate'].mean().item():.3f}",
          'contra', f'{loss_contra.item():.4f}', 'domain', f'{loss_domain.item():.4f}',
          'gate_reg', f'{loss_gate.item():.4f}', 'params', n_params(m))
