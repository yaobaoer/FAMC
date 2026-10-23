                       
from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _chunk_sizes(n: int, m: int) -> List[int]:
    base, rem = divmod(int(n), int(m))
    return [base + (1 if i < rem else 0) for i in range(m)]


class TFusionBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1, n_modal: int = 3):
        super().__init__()
        self.n_modal = int(n_modal)
        sizes = _chunk_sizes(d_in, self.n_modal)
        self.sizes = sizes
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(s, d_hidden), nn.GELU(), nn.LayerNorm(d_hidden))
            if s > 0 else nn.Identity()
            for s in sizes
        ])
                                                                     
        self.pkt_conv = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1, groups=1)
        self.flow_pool_k = 5
        self.q = nn.Linear(d_hidden, d_hidden)
        self.k = nn.Linear(d_hidden, d_hidden)
        self.v = nn.Linear(d_hidden, d_hidden)
        self.fuse = nn.Sequential(
            nn.Linear(d_hidden * self.n_modal, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
        )
        self.out_dim = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                    
        parts, s = [], 0
        for i, sz in enumerate(self.sizes):
            if sz == 0:
                parts.append(torch.zeros(x.size(0), x.size(1), self.out_dim, device=x.device, dtype=x.dtype))
                continue
            h = self.proj[i](x[:, :, s:s + sz])
            s += sz
            if i == 0:
                h = self.pkt_conv(h.transpose(1, 2)).transpose(1, 2)
            elif i == 1:
                                                   
                k = min(self.flow_pool_k, h.size(1))
                pooled = F.avg_pool1d(h.transpose(1, 2), kernel_size=k, stride=1, padding=k // 2)
                h = pooled[:, :, :h.size(1)].transpose(1, 2)
            else:
                host = h.mean(1, keepdim=True).expand_as(h)
                h = h + 0.5 * host
            parts.append(h)
                                                                     
        stacked = torch.stack(parts, dim=2)             
        b, t, m, d = stacked.shape
        tok = stacked.reshape(b * t, m, d)
        q, k, v = self.q(tok), self.k(tok), self.v(tok)
        attn = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)
        fused = (attn @ v).reshape(b, t, m * d)
        return self.fuse(fused)


class TipsoAttnBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1,
                 nhead: int = 4, decay_lambda: float = 0.15):
        super().__init__()
        self.proj = nn.Identity() if d_in == d_hidden else nn.Linear(d_in, d_hidden)
        self.nhead = int(nhead)
        assert d_hidden % self.nhead == 0
        self.d_head = d_hidden // self.nhead
        self.qkv = nn.Linear(d_hidden, d_hidden * 3)
        self.out = nn.Linear(d_hidden, d_hidden)
        self.ff = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hidden * 2, d_hidden),
        )
        self.n1 = nn.LayerNorm(d_hidden)
        self.n2 = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(dropout)
        self.decay_lambda = float(decay_lambda)
        self.out_dim = d_hidden
        self.register_buffer('_decay', torch.empty(0), persistent=False)

    def _decay_bias(self, t: int, device, dtype) -> torch.Tensor:
                                                     
        idx = torch.arange(t, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float()
        return (-self.decay_lambda * dist).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        b, t, d = h.shape
        qkv = self.qkv(h).reshape(b, t, 3, self.nhead, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)              
        q, k, v = qkv[0], qkv[1], qkv[2]
                                      
        bias = self._decay_bias(t, h.device, h.dtype)         
        scale = 1.0 / math.sqrt(self.d_head)
        outs = []
        chunk = 32 if t > 64 else t
        for i in range(0, t, chunk):
            qi = q[:, :, i:i + chunk]              
            logits = (qi @ k.transpose(-1, -2)) * scale + bias[i:i + chunk]
            outs.append(torch.softmax(logits, dim=-1) @ v)
        y = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, t, d)
        h = self.n1(h + self.drop(self.out(y)))
        return self.n2(h + self.drop(self.ff(h)))


class CoLDBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1, n_part: int = 3):
        super().__init__()
        self.n_part = int(n_part)
        self.sizes = _chunk_sizes(d_in, self.n_part)
        max_s = max(self.sizes) if self.sizes else 1
        self.pad_to = max_s
        self.enc = nn.Sequential(
            nn.Linear(max_s, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )
        self.agg = nn.Sequential(
            nn.Linear(d_hidden * self.n_part, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
        )
                                                            
        self.reorder = nn.Parameter(torch.arange(d_in).float())
        self.out_dim = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                                
        order = torch.argsort(self.reorder)
        x = x.index_select(-1, order)
        parts, s = [], 0
        for sz in self.sizes:
            chunk = x[:, :, s:s + sz]
            s += sz
            if sz < self.pad_to:
                chunk = F.pad(chunk, (0, self.pad_to - sz))
            parts.append(self.enc(chunk))
        return self.agg(torch.cat(parts, dim=-1))


class RapierBiGRUBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        hid = max(8, d_hidden // 2)
        self.gru = nn.GRU(d_in, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hid * 2, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.n = nn.LayerNorm(d_hidden)
        self.out_dim = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.n(self.drop(self.proj(y)))


class HyperVisionGNNBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1, n_nodes: int = 7):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.d_node = max(4, d_hidden // 2)
        self.to_nodes = nn.Linear(d_in, self.n_nodes * self.d_node)
        self.w1 = nn.Linear(self.d_node * 2, self.d_node)
        self.w2 = nn.Linear(self.d_node * 2, self.d_node)
        self.readout = nn.Sequential(
            nn.Linear(self.n_nodes * self.d_node, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = d_hidden

    def _msg(self, h: torch.Tensor, w: nn.Linear) -> torch.Tensor:
                      
                              
        hn = F.normalize(h, dim=-1)
        a = torch.softmax(hn @ hn.transpose(-1, -2), dim=-1)             
        neigh = a @ h
        return self.drop(F.gelu(w(torch.cat([h, neigh], dim=-1))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.to_nodes(x).view(b, t, self.n_nodes, self.d_node)
        h = self._msg(h, self.w1)
        h = self._msg(h, self.w2)
        return self.readout(h.reshape(b, t, -1))


class FeCoBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_dev = 9 if int(d_in) % 9 == 0 else 1
        self.slot = int(d_in) // self.n_dev
        h = int(d_hidden)
        self.net = nn.Sequential(
            nn.Linear(self.slot, h * 2),
            nn.ReLU(),
            nn.Linear(h * 2, h * 4),
            nn.ReLU(),
            nn.Linear(h * 4, h * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h * 2, h),
        )
        self.out_dim = h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = x.view(b, t, self.n_dev, self.slot)
        return self.net(h).mean(2)


class AOCIDSBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_dev = 9 if int(d_in) % 9 == 0 else 1
        self.slot = int(d_in) // self.n_dev
        mid = int(d_hidden) * 2
        self.enc = nn.Sequential(
            nn.Linear(self.slot, mid),
            nn.ReLU(),
            nn.Linear(mid, d_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, self.slot),
        )
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_hidden + self.slot, d_hidden),
        )
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = x.view(b, t, self.n_dev, self.slot)
        z = self.enc(h)
        rec = self.dec(z)
        return self.out(torch.cat([z, rec], dim=-1)).mean(2)


class ReCDABackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        h = int(d_hidden)
        self.enc = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.proj = nn.Linear(h, h)
        self.to_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h))
        self.out_dim = h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_out(self.encode(x))


class FSNIDBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, 50, num_layers=1, batch_first=True)
        self.to_out = nn.Sequential(nn.Dropout(dropout), nn.Linear(50, d_hidden))
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        return self.to_out(y)


PAPER_BACKBONES = {
    'tfusion': ('tFusion', TFusionBackbone),
    'tipsogan': ('TIPSO-Attn', TipsoAttnBackbone),
    'cold': ('CoLD', CoLDBackbone),
    'rapier': ('RAPIER-BiGRU', RapierBiGRUBackbone),
    'hypervision': ('HyperVision-GNN', HyperVisionGNNBackbone),
    'feco': ('FeCo', FeCoBackbone),
    'aocids': ('AOC-IDS', AOCIDSBackbone),
    'recda': ('ReCDA', ReCDABackbone),
    'fsnid': ('FSNID', FSNIDBackbone),
}


def build_paper_backbone(kind: str, d_in: int, d_hidden: int = 32, dropout: float = 0.1) -> nn.Module:
    kind = str(kind).lower()
    if kind not in PAPER_BACKBONES:
        raise ValueError(f'unknown paper backbone {kind}, expect {list(PAPER_BACKBONES)}')
    _, cls = PAPER_BACKBONES[kind]
    return cls(d_in, d_hidden=d_hidden, dropout=dropout)


def smoke_all(d_in: int = 17, t: int = 8, d_hidden: int = 32):
    x = torch.randn(2, t, d_in)
    for k in PAPER_BACKBONES:
        m = build_paper_backbone(k, d_in, d_hidden=d_hidden)
        y = m(x)
        assert y.shape == (2, t, d_hidden), (k, y.shape)
        n = sum(p.numel() for p in m.parameters())
        print(f'{k:12s} -> {tuple(y.shape)}  params={n}')


if __name__ == '__main__':
    smoke_all()
