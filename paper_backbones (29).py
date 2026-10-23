                       
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


PAPER_BACKBONES = {
    'tfusion': ('tFusion', TFusionBackbone),
    'tipsogan': ('TIPSO-Attn', TipsoAttnBackbone),
    'cold': ('CoLD', CoLDBackbone),
    'rapier': ('RAPIER-BiGRU', RapierBiGRUBackbone),
    'hypervision': ('HyperVision-GNN', HyperVisionGNNBackbone),
}


class PerFLIDBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        del dropout
        self.lstm = nn.LSTM(d_in, 15, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(15, 20),
            nn.ReLU(),
            nn.Linear(20, 60),
            nn.ReLU(),
            nn.Linear(60, 80),
            nn.ReLU(),
            nn.Linear(80, 90),
            nn.ReLU(),
            nn.Linear(90, d_hidden),
        )
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.fc(h)


                         
_NORMAL = 3


def _cosine_crc(rep: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    pass
    z = F.normalize(rep, dim=-1)
    pos = z[y == _NORMAL]
    neg = z[y != _NORMAL]
    n = pos.size(0)
    if n < 2 or neg.size(0) < 1:
        return rep.new_zeros(())
    pp = pos @ pos.T / tau
    pn = pos @ neg.T / tau
    eye = torch.eye(n, device=rep.device, dtype=torch.bool)
    pp = pp.masked_fill(eye, -1e9)
    log_pos = torch.logsumexp(pp, dim=-1)
    log_den = torch.logsumexp(torch.stack([log_pos, torch.logsumexp(pn, dim=-1)], 0), 0)
    return -(log_pos - log_den).mean()


def _feco_pair_loss(rep: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    pass
    z = F.normalize(rep, dim=-1)
    pos = z[y == _NORMAL]
    neg = z[y != _NORMAL]
    n = pos.size(0)
    if n < 2 or neg.size(0) < 1:
        return rep.new_zeros(())
    pp = pos @ pos.T / tau
    pn = pos @ neg.T / tau
    neg_term = torch.logsumexp(pn, dim=-1, keepdim=True)
    log_den = torch.logsumexp(torch.stack([pp, neg_term.expand_as(pp)], 0), 0)
    loss = -(pp - log_den)
    eye = torch.eye(n, device=rep.device, dtype=torch.bool)
    loss = loss.masked_fill(eye, 0)
    return loss.sum() / (n * (n - 1))


class AOCBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        del dropout
        hid = 64
        self.enc1 = nn.Linear(d_in, hid)
        self.enc2 = nn.Linear(hid, d_hidden)
        self.dec1 = nn.Linear(d_hidden, hid)
        self.dec2 = nn.Linear(hid, d_in)
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc2(F.relu(self.enc1(x)))
        dh = F.relu(self.dec1(z))
        self._enc = z
        self._dec = dh
        self._rec = self.dec2(dh)
        return z

    def aoc_loss(self, y: torch.Tensor) -> torch.Tensor:
        return _cosine_crc(self._enc.mean(1), y) + _cosine_crc(self._dec.mean(1), y)


class FeCoBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        del dropout
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.proj = nn.Linear(128, d_hidden)
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.net(x)
        self._v = v
        return self.proj(v)

    def feco_loss(self, y: torch.Tensor) -> torch.Tensor:
        return _feco_pair_loss(self._v.mean(1), y)


class FSNIDBackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        del dropout
        self.lstm = nn.LSTM(d_in, 50, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, d_hidden),
        )
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.fc(h)


class ReCDABackbone(nn.Module):
    pass

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        del dropout
        self.enc = nn.Sequential(
            nn.Linear(d_in, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.proj_h = nn.Linear(16, 16)
        self.proj = nn.Linear(16, d_hidden)
        self.sigma = 0.6
        self.tau = 0.1
        self.out_dim = int(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        self._in = x
        self._lat = h.mean(1)
        return self.proj(h)

    def recda_loss(self) -> torch.Tensor:
        x = self._in
        b = x.size(0)
        if b < 2:
            return x.new_zeros(())
        xr = x.roll(1, 0).detach()
        m = torch.bernoulli(x.new_full((b, 1, x.size(-1)), self.sigma))
        xt = m * xr + (1.0 - m) * x
        z = F.normalize(self.proj_h(self._lat), dim=-1)
        zt = F.normalize(self.proj_h(self.enc(xt).mean(1)), dim=-1)
        logits = z @ zt.transpose(0, 1) / self.tau
        target = torch.arange(b, device=x.device)
        return F.cross_entropy(logits, target)


PAPER_BACKBONES['perflid'] = ('PerFLID', PerFLIDBackbone)
PAPER_BACKBONES['aoc'] = ('AOC-IDS', AOCBackbone)
PAPER_BACKBONES['feco'] = ('FeCo', FeCoBackbone)
PAPER_BACKBONES['fsnid'] = ('FSNID', FSNIDBackbone)
PAPER_BACKBONES['recda'] = ('ReCDA', ReCDABackbone)


def build_paper_backbone(kind: str, d_in: int, d_hidden: int = 32, dropout: float = 0.1,
                         device_dims=None, front_dim: int = 0) -> nn.Module:
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
