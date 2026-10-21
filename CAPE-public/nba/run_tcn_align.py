                       
pass
from __future__ import annotations

import os
import sys
import math
import random
import shutil
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from activity_gated_gat_v2 import ActivityGatedGAT as FrontV2

DATA_CSV = os.path.join(_ROOT, 'nba_iot_11class.csv')
                                              
NODEWISE = True
_TAG = 'tcn_node' if NODEWISE else 'tcn_align'
OUT_DIR = os.path.join(_ROOT, f'{_TAG}_ckpts')
LODO_DIR = os.path.join(_ROOT, f'{_TAG}_lodo_ckpts')

DEVICE_NAMES = (
    'danmini', 'ecobee', 'ennio', 'philips',
    'pt737e', 'pt838', 'samsung', 'xcs1002', 'xcs1003',
)

                                 
PUBLIC_FEATS = None                              

WINDOW, TRAIN_STEP, EVAL_STEP = 64, 4, 8
MAX_WINDOWS = 12000
MAX_ROWS_PER_CLASS = 15000
BATCH, EPOCHS, LR, PATIENCE = 64, 40, 1e-3, 8
D_MODEL, D_HIDDEN = 32, 64
SEED = 42
EVAL_SEEDS = (42, 123, 2024)

                                                         
V2_HPARAM = dict(lam_contra=0.5, lam_domain=0.2, lam_gate=0.3,
                 contra_temp=0.1, gate_target=0.3)
USE_GAT = False                         

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def all_feat_cols(df):
    skip = {'device', 'device_full', 'final_type', 'final_label', 'label_encoded'}
    return [c for c in df.columns if c not in skip]


def public_feat_cols(df):
    pass
    feats = all_feat_cols(df)
    return [c for c in feats if not c.startswith('HH_jit_')]


def build_aligned(df, feat_cols, device_names=DEVICE_NAMES,
                  max_rows_per_class=MAX_ROWS_PER_CLASS, seed=42):
    n_dev, d = len(device_names), len(feat_cols)
    xs, ys = [], []
    rng = np.random.RandomState(seed)
    for lab in sorted(df['label_encoded'].unique().tolist()):
        streams = []
        for name in device_names:
            sub = df[(df['device'] == name) & (df['label_encoded'] == lab)]
            streams.append(None if len(sub) == 0 else
                           sub[feat_cols].to_numpy(np.float32, copy=True))
        lens = [len(s) for s in streams if s is not None]
        if not lens:
            continue
        n = min(min(lens), max_rows_per_class)
        block = np.zeros((n, n_dev * d), np.float32)
        for di, s in enumerate(streams):
            if s is None:
                continue
            take = np.linspace(0, len(s) - 1, n).astype(np.int64)
            j = rng.randint(-max(len(s) // max(n, 1), 1),
                            max(len(s) // max(n, 1), 1) + 1, size=n)
            take = np.clip(take + j, 0, len(s) - 1)
            block[:, di * d:(di + 1) * d] = s[take]
        xs.append(block)
        ys.append(np.full(n, int(lab), np.int64))
    X = np.concatenate(xs); y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def device_offsets(dims):
    offs, s = [], 0
    for d in dims:
        offs.append(s); s += d
    return offs


def zero_devices(X, dims, ids):
    out = np.array(X, copy=True)
    offs = device_offsets(dims)
    for p in ids:
        s = offs[int(p)]
        out[..., s:s + dims[int(p)]] = 0.0
    return out


class ArrayWinDS(Dataset):
    def __init__(self, X, y, dims, window=64, step=4, max_windows=None, seed=42):
        self.device_dims = list(dims)
        starts = list(range(0, len(X) - window + 1, step))
        if max_windows and len(starts) > max_windows:
            rng = np.random.RandomState(seed)
            starts = sorted(rng.choice(starts, max_windows, replace=False).tolist())
        n, C = len(starts), X.shape[1]
        samples = np.empty((n, window, C), np.float32)
        labels = np.empty(n, np.int64)
        for i, s in enumerate(starts):
            samples[i] = X[s:s + window]
            labels[i] = int(y[s + window - 1])
        self.samples, self.labels = samples, labels
        print(f'windows={n} dims={dims} step={step}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.as_tensor(self.samples[i]), torch.tensor(self.labels[i], dtype=torch.long)


class DualPool(nn.Module):
    def __init__(self, d, n_cls):
        super().__init__()
        self.attn = nn.Linear(d, 1)
        self.clf = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, n_cls))

    def forward(self, seq):
        w = torch.softmax(self.attn(seq).squeeze(-1), -1)
        a = (seq * w.unsqueeze(-1)).sum(1)
        mu = seq.mean(1)
        return self.clf(torch.cat([a, mu], -1))


class TCNBlock(nn.Module):
    def __init__(self, cin, cout, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Identity() if cin == cout else nn.Conv1d(cin, cout, 1)
        self.n = nn.LayerNorm(cout)

    def forward(self, x):
        h = F.pad(x.transpose(1, 2), (self.pad, 0))
        h = self.drop(F.gelu(self.conv(h)))
        y = (h + self.res(x.transpose(1, 2))).transpose(1, 2)
        return self.n(y)


class TCNBackbone(nn.Module):
    def __init__(self, d_in, d_h=64, dropout=0.1):
        super().__init__()
        self.b1 = TCNBlock(d_in, d_h, kernel=3, dilation=1, dropout=dropout)
        self.b2 = TCNBlock(d_h, d_h, kernel=3, dilation=2, dropout=dropout)
        self.out_dim = d_h

    def forward(self, x):
        return self.b2(self.b1(x))


class PlugTCN(nn.Module):
    def __init__(self, dims, n_cls, use_front=True):
        super().__init__()
        self.front = None
        self.n_nodes = len(dims)
        self.node_raw_dim = int(dims[0])
        self.nodewise = bool(NODEWISE)
        if use_front:
            self.front = FrontV2(
                dims, d_model=D_MODEL, gat_heads=1, dropout=0.1,
                flatten=not self.nodewise)
            self.front.use_gat = USE_GAT
            self.front.set_gate_target(V2_HPARAM['gate_target'])
            d_in = D_MODEL if self.nodewise else self.front.out_dim
        else:
            d_in = self.node_raw_dim if self.nodewise else int(sum(dims))
        self.backbone = TCNBackbone(d_in, D_HIDDEN)
        head_in = (self.backbone.out_dim * self.n_nodes) if self.nodewise else self.backbone.out_dim
        self.head = DualPool(head_in, n_cls)

    def _as_nodes(self, x: torch.Tensor) -> torch.Tensor:
        pass
        B, T, C = x.shape
        return x.reshape(B, T, self.n_nodes, self.node_raw_dim)

    def _tcn_nodes(self, z: torch.Tensor) -> torch.Tensor:
        pass
        B, T, N, D = z.shape
        h = self.backbone(z.reshape(B * N, T, D)).reshape(B, T, N, -1)
        valid = (z.norm(dim=-1, keepdim=True) > 1e-6).float()
        h = h * valid
        return h.reshape(B, T, N * h.size(-1))

    def forward(self, x, return_aux=False):
        aux = None
        if self.nodewise:
            if self.front is None:
                seq = self._tcn_nodes(self._as_nodes(x))
            else:
                z, aux = self.front(x, flatten=False, return_features=True)
                seq = self._tcn_nodes(z)
        else:
            if self.front is None:
                seq = x
            else:
                seq, aux = self.front(x, return_features=True)
            seq = self.backbone(seq)
        logits = self.head(seq)
        return (logits, aux) if return_aux else logits


def metrics(yt, yp):
    return dict(
        acc=float(accuracy_score(yt, yp)),
        precision=float(precision_score(yt, yp, average='macro', zero_division=0)),
        recall=float(recall_score(yt, yp, average='macro', zero_division=0)),
        f1=float(f1_score(yt, yp, average='macro', zero_division=0)),
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    yt, yp = [], []
    for x, y in loader:
        logits = model(x.to(DEVICE))
        yp.extend(logits.argmax(1).cpu().tolist())
        yt.extend(y.tolist())
    return metrics(yt, yp)


def dann_alpha(epoch, max_ep=EPOCHS):
    p = epoch / max(max_ep, 1)
    return float(2 / (1 + math.exp(-10 * p)) - 1)


from paired_protocol import paired_fit

@paired_fit
def fit(use_front, train_ds, val_ds, test_ds, n_cls, cw, ckpt, tag, seed=SEED):
    set_seed(seed)
    counts = np.maximum(np.bincount(train_ds.labels, minlength=n_cls).astype(np.float64), 1)
    sw = 1.0 / np.sqrt(counts[train_ds.labels]); sw /= sw.mean()
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(train_ds), True, generator=gen)
    tr = DataLoader(train_ds, BATCH, sampler=sampler, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    va = DataLoader(val_ds, BATCH, shuffle=False, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    te = DataLoader(test_ds, BATCH, shuffle=False, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))

    model = PlugTCN(train_ds.device_dims, n_cls, use_front=use_front).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 3)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'\n===== {tag} front={use_front} gat={USE_GAT} '
          f'nodewise={NODEWISE} params={n_p:.3f}M =====', flush=True)

    hp = V2_HPARAM
    best_f1, best_ep, bad = -1.0, -1, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses, f1s, gates = [], [], []
        alpha = dann_alpha(ep) if use_front else 1.0
        for x, y in tqdm(tr, desc=f'{tag} ep{ep}', leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if use_front:
                logits, aux = model(x, return_aux=True)
                loss = F.cross_entropy(logits, y, weight=cw)
                hp_ = aux['h_prime']
                loss = loss + hp['lam_contra'] * model.front.cross_protocol_contrastive_loss(
                    hp_, y, temperature=hp['contra_temp'])
                loss = loss + hp['lam_domain'] * model.front.domain_adversarial_loss(hp_, alpha=alpha)
                loss = loss + hp['lam_gate'] * model.front.gate_reg_loss(aux['gate'])
                gates.append(float(aux['gate'].detach().mean()))
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y, weight=cw)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            f1s.append(f1_score(y.cpu().numpy(), logits.argmax(1).cpu().numpy(),
                                average='macro', zero_division=0))
        vm = evaluate(model, va)
        sch.step(vm['f1'])
        gtxt = f' gate={np.mean(gates):.3f}' if gates else ''
        print(f'Ep{ep:02d} loss={np.mean(losses):.4f} trF1={np.mean(f1s):.4f} '
              f'valF1={vm["f1"]:.4f} valAcc={vm["acc"]:.4f}{gtxt}', flush=True)
        if vm['f1'] > best_f1 + 1e-4:
            best_f1, best_ep, bad = vm['f1'], ep, 0
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_ep,
                        'val_f1': best_f1, 'use_front': use_front,
                        'dims': train_ds.device_dims, 'n_cls': n_cls}, ckpt)
            print(f'  >> saved {ckpt}', flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print('early stop', flush=True)
                break

    raw = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(raw['model_state'])
    tm = evaluate(model, te)
    print(f'TEST Acc={tm["acc"]:.4f} F1={tm["f1"]:.4f} best_ep={best_ep}', flush=True)
    return model, tm, best_ep


@torch.no_grad()
def bootstrap_eval(model, ds, seeds=EVAL_SEEDS):
    rows = []
    y = ds.labels
    for sd in seeds:
        rng = np.random.RandomState(sd)
        parts = [rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True)
                 for c in np.unique(y)]
        idx = np.concatenate(parts)
        sub = torch.utils.data.Subset(ds, idx.tolist())
        loader = DataLoader(sub, BATCH, shuffle=False, num_workers=0)
        m = evaluate(model, loader); m['seed'] = sd
        rows.append(m)
    return rows


def stratified_split(df, seed=42):
    from paper_protocol import paper_split
    return paper_split(df, seed)


def load_pack():
    df = pd.read_csv(DATA_CSV, low_memory=False)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'].astype(str))
    feats = public_feat_cols(df)
    d = len(feats)
    dims = [d] * len(DEVICE_NAMES)
    print(f'PUBLIC feats={d} (dropped HH_jit) devices={len(DEVICE_NAMES)} '
          f'classes={list(le.classes_)}', flush=True)
    tr_df, va_df, te_df = stratified_split(df, SEED)
    Xtr, ytr = build_aligned(tr_df, feats, seed=SEED)
    Xva, yva = build_aligned(va_df, feats, seed=SEED + 1)
    Xte, yte = build_aligned(te_df, feats, seed=SEED + 2)
    sc = StandardScaler()
    Xtr = np.nan_to_num(np.clip(sc.fit_transform(Xtr), -8, 8), nan=0.0).astype(np.float32)
    Xva = np.nan_to_num(np.clip(sc.transform(Xva), -8, 8), nan=0.0).astype(np.float32)
    Xte = np.nan_to_num(np.clip(sc.transform(Xte), -8, 8), nan=0.0).astype(np.float32)
    print(f'aligned {len(ytr)}/{len(yva)}/{len(yte)} dim={Xtr.shape[1]}=9×{d}', flush=True)
    train = ArrayWinDS(Xtr, ytr, dims, WINDOW, TRAIN_STEP, MAX_WINDOWS, SEED)
    val = ArrayWinDS(Xva, yva, dims, WINDOW, EVAL_STEP, MAX_WINDOWS // 2, SEED + 1)
    test = ArrayWinDS(Xte, yte, dims, WINDOW, EVAL_STEP, MAX_WINDOWS // 2, SEED + 2)
    train.X_table, train.y_table = Xtr, ytr
    val.X_table, val.y_table = Xva, yva
    test.X_table, test.y_table = Xte, yte
    n_cls = len(le.classes_)
    counts = np.maximum(np.bincount(train.labels, minlength=n_cls).astype(np.float64), 1)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0),
                      dtype=torch.float32, device=DEVICE)
    return dict(train=train, val=val, test=test, dims=dims, n_cls=n_cls,
                le=le, cw=cw, d_feat=d)


def write_summary(full_rows, lodo_rows):
    lines = [
        'TCN  Front-v2 vs Raw | N-BaIoT align (public feats, masked soft-fusion)',
        f'V2_HPARAM={V2_HPARAM} USE_GAT={USE_GAT} D_MODEL={D_MODEL} NODEWISE={NODEWISE}',
        '',
        '===== FULL per-run =====',
    ]
    df = pd.DataFrame(full_rows)
    lines.append(df.to_string(index=False))
    lines += ['', '===== FULL mean +/- std =====']
    for name, g in df.groupby('model'):
        lines.append(
            f'  {name} | acc {g.acc.mean():.4f}+/-{g.acc.std(ddof=0):.4f} | '
            f'f1 {g.f1.mean():.4f}+/-{g.f1.std(ddof=0):.4f}')
    if set(df.model) >= {'Front-v2+TCN', 'Raw+TCN'}:
        a = df[df.model == 'Front-v2+TCN'].set_index('seed')
        b = df[df.model == 'Raw+TCN'].set_index('seed')
        gap = a[['acc', 'f1']] - b[['acc', 'f1']]
        lines.append(f'FULL gap acc {gap.acc.mean():+.4f} f1 {gap.f1.mean():+.4f}')

    if lodo_rows:
        ld = pd.DataFrame(lodo_rows)
        lines += ['', '===== LODO per-seed (mean over devices) =====']
        seed_mean = ld.groupby(['model', 'seed'])[['acc', 'f1']].mean().reset_index()
        lines.append(seed_mean.to_string(index=False))
        lines += ['', '===== LODO mean +/- std =====']
        for name, g in seed_mean.groupby('model'):
            lines.append(
                f'  {name} | acc {g.acc.mean():.4f}+/-{g.acc.std(ddof=0):.4f} | '
                f'f1 {g.f1.mean():.4f}+/-{g.f1.std(ddof=0):.4f}')
        if set(seed_mean.model) >= {'Front-v2+TCN', 'Raw+TCN'}:
            a = seed_mean[seed_mean.model == 'Front-v2+TCN'].set_index('seed')
            b = seed_mean[seed_mean.model == 'Raw+TCN'].set_index('seed')
            gap = a[['acc', 'f1']] - b[['acc', 'f1']]
            lines.append(f'LODO gap acc {gap.acc.mean():+.4f} f1 {gap.f1.mean():+.4f}')
        per = ld.groupby(['model', 'drop_device'])[['acc', 'f1']].mean()
        lines += ['', '===== LODO per-device =====', str(per)]

    text = '\n'.join(lines) + '\n'
    path = os.path.join(OUT_DIR, 'tcn_align_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\n' + text, flush=True)
    return path


def main():
    only = 'all'
    if '--full' in sys.argv:
        only = 'full'
    if '--lodo' in sys.argv:
        only = 'lodo'
    for d in (OUT_DIR, LODO_DIR):
                                              
        if only == 'lodo':
            if d == LODO_DIR and os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
            continue
        if '--keep' not in sys.argv and os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    pack = load_pack()
    full_rows, lodo_rows = [], []

    if only != 'lodo':
        for use_front, stem, name in ((True, 'front', 'Front-v2+TCN'),
                                       (False, 'raw', 'Raw+TCN')):
            ckpt = os.path.join(OUT_DIR, f'best_tcn_{stem}.pth')
            model, tm, best_ep = fit(
                use_front, pack['train'], pack['val'], pack['test'],
                pack['n_cls'], pack['cw'], ckpt, f'{name}-FULL', SEED)
            for r in bootstrap_eval(model, pack['test']):
                full_rows.append(dict(setting='full', model=name, best_epoch=best_ep,
                                      seed=r['seed'], **{k: r[k] for k in ('acc', 'precision', 'recall', 'f1')}))
        if only == 'full':
            write_summary(full_rows, [])
            print('ALIGN_TCN_FULL_DONE', flush=True)
            return
    else:
                                        
        for use_front, stem, name in ((True, 'front', 'Front-v2+TCN'),
                                       (False, 'raw', 'Raw+TCN')):
            ckpt = os.path.join(OUT_DIR, f'best_tcn_{stem}.pth')
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f'missing FULL ckpt: {ckpt}')
            raw = torch.load(ckpt, map_location=DEVICE, weights_only=False)
            model = PlugTCN(pack['dims'], pack['n_cls'], use_front=use_front).to(DEVICE)
            model.load_state_dict(raw['model_state'])
            best_ep = int(raw.get('best_epoch', -1))
            for r in bootstrap_eval(model, pack['test']):
                full_rows.append(dict(setting='full', model=name, best_epoch=best_ep,
                                      seed=r['seed'], **{k: r[k] for k in ('acc', 'precision', 'recall', 'f1')}))
            print(f'reused FULL {name} best_ep={best_ep}', flush=True)

    dims = pack['dims']
    for pi, pname in enumerate(DEVICE_NAMES):
        print(f'\n########## LODO drop={pname} ##########', flush=True)
        tr = ArrayWinDS(zero_devices(pack['train'].X_table, dims, [pi]),
                        pack['train'].y_table, dims, WINDOW, TRAIN_STEP, MAX_WINDOWS, SEED)
        va = ArrayWinDS(zero_devices(pack['val'].X_table, dims, [pi]),
                        pack['val'].y_table, dims, WINDOW, EVAL_STEP, MAX_WINDOWS // 2, SEED + 1)
        te = ArrayWinDS(zero_devices(pack['test'].X_table, dims, [pi]),
                        pack['test'].y_table, dims, WINDOW, EVAL_STEP, MAX_WINDOWS // 2, SEED + 2)
        for use_front, stem, name in ((True, 'front', 'Front-v2+TCN'),
                                       (False, 'raw', 'Raw+TCN')):
            ckpt = os.path.join(LODO_DIR, f'best_lodo_{pname}_{stem}.pth')
            model, tm, best_ep = fit(
                use_front, tr, va, te, pack['n_cls'], pack['cw'], ckpt,
                f'LODO-{pname}-{name}', SEED)
            for r in bootstrap_eval(model, te):
                lodo_rows.append(dict(
                    setting='lodo', model=name, drop_device=pname,
                    best_epoch=best_ep, seed=r['seed'],
                    **{k: r[k] for k in ('acc', 'precision', 'recall', 'f1')}))

    write_summary(full_rows, lodo_rows)
    print('ALIGN_TCN_DONE', flush=True)


if __name__ == '__main__':
    main()
