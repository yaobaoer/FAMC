                       
pass
from __future__ import annotations

import os
import sys
import math
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(_ROOT, 'iot_fusion_weather_all.csv')):
    _ROOT = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TMP = os.path.join(_ROOT, '_tmp')
os.makedirs(_TMP, exist_ok=True)
for _k in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[_k] = _TMP
os.environ['TORCH_HOME'] = os.path.join(_TMP, 'torch')
os.environ['MPLCONFIGDIR'] = os.path.join(_TMP, 'matplotlib')
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
tempfile.tempdir = _TMP

from activity_gated_gat import ActivityGatedGAT, n_params
from activity_gated_gat_v2 import ActivityGatedGAT as ActivityGatedGATv2

DATA_CSV = os.path.join(_ROOT, 'iot_fusion_weather_all.csv')
DEVICE_PREFIXES = [
    'weather', 'modbus', 'thermostat', 'fridge',
    'garage_door', 'gps', 'motion_light',
]

WINDOW_SIZE, TRAIN_STEP, EVAL_STEP = 128, 1, 2
BATCH_SIZE, EPOCHS, LR, PATIENCE = 64, 50, 1e-3, 5
LAM_F1, DROPOUT = 0.15, 0.1
                                    
V2_HPARAM = dict(
    lam_contra=0.5,
    lam_domain=0.2,
    lam_gate=1.0,
    contra_temp=0.1,
    gate_target=0.3,
)
LAM_CONTRA = V2_HPARAM['lam_contra']
LAM_DOMAIN = V2_HPARAM['lam_domain']
LAM_GATE = V2_HPARAM['lam_gate']
CONTRA_TEMP = V2_HPARAM['contra_temp']
GATE_TARGET = V2_HPARAM['gate_target']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_DATA = None


class IoTFusionDataset(Dataset):
    pass

    def __init__(self, df, window_size=128, step=1, scaler=None, fit_scaler=False):
        self.window_size = window_size
        self.step = step
        self.devices = list(DEVICE_PREFIXES)
        self.device_feat_cols = {}
        self.device_dims = []
        self.all_feat_cols = []
        for dev in DEVICE_PREFIXES:
            cols = [c for c in df.columns if c.startswith(dev + '_')
                    and not c.endswith('_label') and not c.endswith('_type')]
            self.device_feat_cols[dev] = cols
            self.device_dims.append(len(cols))
            self.all_feat_cols.extend(cols)

        X = df[self.all_feat_cols].values.astype(np.float32)
        y_enc = df['label_encoded'].values.astype(np.int64)
        if scaler is None and fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaler is not None:
            self.scaler = scaler
            X = self.scaler.transform(X)
        else:
            self.scaler = None
        X = np.nan_to_num(np.clip(X, -8, 8), nan=0.0).astype(np.float32)

        samples, labels = [], []
        for i in range(0, len(X) - window_size + 1, step):
            samples.append(X[i:i + window_size])
            labels.append(int(y_enc[i + window_size - 1]))
        self.samples = np.stack(samples, axis=0)
        self.labels = np.asarray(labels, dtype=np.int64)
        print(f'windows={len(self)} dims={self.device_dims} step={step} '
              f'classes={sorted(np.unique(self.labels).tolist())}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class DualPoolHead(nn.Module):
    pass

    def __init__(self, d_in, num_classes):
        super().__init__()
        self.attn = nn.Linear(d_in, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in * 2),
            nn.Linear(d_in * 2, num_classes),
        )

    def forward(self, seq):
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=1)
        feat = torch.cat([(seq * w.unsqueeze(-1)).sum(1), seq.mean(1)], dim=-1)
        return self.classifier(feat)


class CNNBackbone(nn.Module):
    def __init__(self, d_in, d_hidden=32, kernel=3):
        super().__init__()
        pad = kernel - 1
        self.conv = nn.Conv1d(d_in, d_hidden, kernel, padding=pad)
        self.gate = nn.Conv1d(d_in, d_hidden, kernel, padding=pad)
        self.out_dim = d_hidden

    def forward(self, x):
        x_t, L = x.transpose(1, 2), x.size(1)
        h = self.conv(x_t)[:, :, :L] * torch.sigmoid(self.gate(x_t)[:, :, :L])
        return h.transpose(1, 2)


class PlugCNN(nn.Module):
    pass

    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
        super().__init__()
        self.front = None
        self.front_ver = str(front_ver).lower() if use_front else 'none'
        if use_front:
            cls = ActivityGatedGATv2 if self.front_ver == 'v2' else ActivityGatedGAT
            self.front = cls(device_dims, d_model=d_model, gat_heads=1, dropout=dropout)
            d_in = self.front.out_dim
        else:
            d_in = int(sum(device_dims))
        self.backbone = CNNBackbone(d_in, d_hidden=d_hidden)
        self.head = DualPoolHead(self.backbone.out_dim, num_classes)

    def forward(self, x, return_aux=False):
        aux = None
        if self.front is None:
            seq = x
        elif self.front_ver == 'v2':
            seq, aux = self.front(x, return_features=True)
        else:
            seq = self.front(x)
        logits = self.head(self.backbone(seq))
        if return_aux:
            return logits, aux
        return logits


def stratified_timegroup_split(df):
    df = df.copy()
    df['time_group'] = df.index // 2000
    train_indices, val_indices, test_indices = [], [], []
    for _, group in df.groupby('time_group'):
        if len(group) < 3:
            train_indices.extend(group.index.tolist())
            continue
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            tr_val_idx, test_idx = next(sss.split(group, group['label_encoded']))
        except ValueError:
            tr_val_idx = np.arange(len(group))
            test_idx = np.array([], dtype=int)
        if len(tr_val_idx) > 1:
            try:
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                sub = group.iloc[tr_val_idx]
                train_idx, val_idx = next(sss2.split(sub, sub['label_encoded']))
                train_indices.extend(group.index[tr_val_idx][train_idx].tolist())
                val_indices.extend(group.index[tr_val_idx][val_idx].tolist())
            except ValueError:
                train_indices.extend(group.index[tr_val_idx].tolist())
        else:
            train_indices.extend(group.index[tr_val_idx].tolist())
        if len(test_idx):
            test_indices.extend(group.index[test_idx].tolist())
    return (
        df.loc[train_indices].sort_index().reset_index(drop=True),
        df.loc[val_indices].sort_index().reset_index(drop=True),
        df.loc[test_indices].sort_index().reset_index(drop=True),
    )


def focal_loss(logits, labels, alpha=0.25, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits.float(), labels, reduction='none',
                         weight=weight, label_smoothing=0.015)
    pt = torch.exp(-ce.detach())
    return (alpha * (1 - pt).clamp(0, 1) ** gamma * ce).mean()


def soft_f1_loss(logits, y, n_cls, eps=1e-6):
    p = F.softmax(logits.float(), 1)
    t = F.one_hot(y, n_cls).float()
    tp = (p * t).sum(0)
    fp = (p * (1 - t)).sum(0)
    fn = ((1 - p) * t).sum(0)
    f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return 1 - f1.mean()


@torch.no_grad()
def evaluate(model, loader, device, label_encoder, desc='Eval'):
    model.eval()
    preds, trues = [], []
    all_labels = list(range(len(label_encoder.classes_)))
    names = list(label_encoder.classes_)
    for x, y in tqdm(loader, desc=desc, leave=False):
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        preds.extend(pred)
        trues.extend(y.numpy())
    report = classification_report(
        trues, preds, labels=all_labels, target_names=names, digits=4, zero_division=0)
    kw = dict(average='macro', labels=all_labels, zero_division=0)
    return {
        'macro_f1': float(f1_score(trues, preds, **kw)),
        'macro_precision': float(precision_score(trues, preds, **kw)),
        'macro_recall': float(recall_score(trues, preds, **kw)),
        'weighted_f1': float(f1_score(trues, preds, average='weighted', labels=all_labels, zero_division=0)),
        'accuracy': float(accuracy_score(trues, preds)),
        'report': report,
        'y_true': np.asarray(trues),
        'y_pred': np.asarray(preds),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(1, keepdims=True), 1) * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Count)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                vmin=0, vmax=100)
    axes[1].set_title('Confusion Matrix (Normalized %)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'confusion matrix saved: {save_path}')


def dann_alpha(epoch, max_epoch=10):
    t = 1.0 if max_epoch <= 0 else min(1.0, float(epoch) / float(max_epoch))
    return float(2.0 / (1.0 + math.exp(-10.0 * t)) - 1.0)


def _client_view(x, dims, keep):
    out = torch.zeros_like(x)
    start = 0
    for i, d in enumerate(dims):
        if i == keep:
            out[:, :, start:start + d] = x[:, :, start:start + d]
        start += d
    return out


def _cls_loss(logits, y, class_weight, n_cls):
    return focal_loss(logits, y, weight=class_weight) + LAM_F1 * soft_f1_loss(logits, y, n_cls)


def _head_grad_feature(logits, y, feat, class_weight, n_cls):
    pass
    feat = feat.detach()
    logits = logits.detach().requires_grad_(True)
    loss = _cls_loss(logits, y, class_weight, n_cls)
    (dlogits,) = torch.autograd.grad(loss, logits)
                                               
    return torch.einsum('bc,bd->cd', dlogits, feat).reshape(-1)


def perflid_meta_loss(model, x, y, class_weight, n_cls, alpha=1e-2):
    pass
    from torch.func import functional_call

    dims = [int(d) for d in model.device_dims]
    active = []
    start = 0
    for i, d in enumerate(dims):
        if float(x[:, :, start:start + d].detach().abs().sum()) > 0:
            active.append(i)
        start += d
    if not active:
        active = list(range(len(dims)))
    mid = max(int(x.size(0)) // 2, 1)
    if mid >= x.size(0):
        return _cls_loss(model(x), y, class_weight, n_cls)

    support = torch.cat([_client_view(x, dims, i)[:mid] for i in active], dim=0)
    with torch.backends.cudnn.flags(enabled=False):
        logits_s = model(support)
    chunks = logits_s.split(mid, dim=0)
    feats = []
    with torch.no_grad():
        for i in active:
            xi = _client_view(x, dims, i)[:mid]
            if model.front is None:
                feats.append(model.backbone(xi).mean(1))
            else:
                seq = model.front(xi, return_features=True)[0] if model.front_ver == 'v2' else model.front(xi)
                feats.append(model.backbone(seq).mean(1))
    flats = [_head_grad_feature(chunks[j], y[:mid], feats[j], class_weight, n_cls) for j in range(len(active))]
    gram = torch.stack(flats, 0)
    gram = gram / gram.norm(dim=1, keepdim=True).clamp_min(1e-8)
    centroid = gram.mean(0)
    centroid = centroid / centroid.norm().clamp_min(1e-8)
    cos = gram @ centroid
    if len(active) >= 3 and torch.isfinite(cos).all():
        chosen = [j for j in range(len(active)) if cos[j] >= cos.median()]
        chosen = chosen or list(range(len(active)))
    else:
        chosen = list(range(len(active)))

    weights = [float(dims[active[j]]) for j in chosen]
    wsum = sum(weights) or 1.0
    names, params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
            params.append(p)
    loss_s = x.new_zeros(())
    for j, wi in zip(chosen, weights):
        loss_s = loss_s + (wi / wsum) * _cls_loss(chunks[j], y[:mid], class_weight, n_cls)
    with torch.backends.cudnn.flags(enabled=False):
        gs = torch.autograd.grad(loss_s, params, create_graph=True, allow_unused=True)
        fast = {}
        for name, p, g in zip(names, params, gs):
            fast[name] = p if g is None else p - alpha * g
        query = torch.cat([_client_view(x, dims, active[j])[mid:] for j in chosen], dim=0)
        yq = y[mid:].repeat(len(chosen))
        logits_q = functional_call(model, fast, (query,), {'return_aux': False})
        qchunks = logits_q.split(x.size(0) - mid, dim=0)
        loss_q = x.new_zeros(())
        for q, wi in zip(qchunks, weights):
            loss_q = loss_q + (wi / wsum) * _cls_loss(q, y[mid:], class_weight, n_cls)
    return loss_q


def train_epoch(model, loader, optimizer, device, n_cls, class_weight,
                lam_contra=0.0, lam_domain=0.0, lam_gate=0.0, domain_alpha=1.0,
                contra_temp=0.1):
    model.train()
    total, all_preds, all_labels = 0.0, [], []
    gate_sum, gate_n = 0.0, 0
    pbar = tqdm(loader, desc='Train', leave=False)
    use_aux = getattr(model, 'front_ver', '') == 'v2' or getattr(model, 'needs_aux', False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if use_aux:
            logits, aux = model(x, return_aux=True)
        else:
            logits, aux = model(x), None
        if getattr(model, 'perflid', False):
            loss = perflid_meta_loss(model, x, y, class_weight, n_cls)
        else:
            loss = _cls_loss(logits, y, class_weight, n_cls)
        if getattr(model, 'aoc', False):
            loss = loss + model.backbone.aoc_loss(y)
        if getattr(model, 'feco', False):
            loss = loss + model.backbone.feco_loss(y)
        if aux is not None:
            if 'h_prime' in aux and getattr(model, 'front', None) is not None:
                hp = aux['h_prime']
                if lam_contra:
                    loss = loss + lam_contra * model.front.cross_protocol_contrastive_loss(
                        hp, y, temperature=contra_temp)
                if lam_domain:
                    loss = loss + lam_domain * model.front.domain_adversarial_loss(hp, alpha=domain_alpha)
                if lam_gate:
                    loss = loss + lam_gate * model.front.gate_reg_loss(aux['gate'])
                gate_sum += float(aux['gate'].mean().detach())
                gate_n += 1
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = x.size(0)
        total += loss.item() * bs
        all_preds.extend(logits.argmax(1).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix(loss=float(loss.item()))
    n = max(len(loader.dataset), 1)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    gate_mean = gate_sum / max(gate_n, 1)
    return total / n, f1, gate_mean


def artifact_paths(use_front):
    side = 'front' if use_front else 'raw'
    return (
        os.path.join(_ROOT, f'best_cnn_{side}.pth'),
        os.path.join(_ROOT, f'cnn_{side}_results.txt'),
        os.path.join(_ROOT, f'confusion_matrix_cnn_{side}.png'),
    )


def load_datasets(force=False):
    global _DATA
    if _DATA is not None and not force:
        return _DATA
    assert os.path.isfile(DATA_CSV), f'找不到 {DATA_CSV}'
    df = pd.read_csv(DATA_CSV, low_memory=False)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    valid_types = df['final_type'].value_counts()[df['final_type'].value_counts() > 10].index
    df = df[df['final_type'].isin(valid_types)].copy()
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['final_type'])
    n_cls = len(label_encoder.classes_)
    print('classes:', list(label_encoder.classes_))

    df_train, df_val, df_test = stratified_timegroup_split(df)
    print(f'train/val/test rows: {len(df_train)}/{len(df_val)}/{len(df_test)}')

    train_ds = IoTFusionDataset(df_train, window_size=WINDOW_SIZE, step=TRAIN_STEP, fit_scaler=True)
    val_ds = IoTFusionDataset(df_val, window_size=WINDOW_SIZE, step=EVAL_STEP, scaler=train_ds.scaler)
    test_ds = IoTFusionDataset(df_test, window_size=WINDOW_SIZE, step=EVAL_STEP, scaler=train_ds.scaler)

    counts = np.maximum(np.bincount(train_ds.labels, minlength=n_cls).astype(np.float64), 1.0)
    sw = 1.0 / np.sqrt(counts[train_ds.labels])
    sw = sw / sw.mean()
    cw = np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0)
    class_weight = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    _DATA = {
        'train_ds': train_ds, 'val_ds': val_ds, 'test_ds': test_ds,
        'n_cls': n_cls, 'label_encoder': label_encoder,
        'sample_weight': sw, 'class_weight': class_weight,
    }
    return _DATA


class ArrayWinDS(Dataset):
    def __init__(self, samples, labels, device_dims):
        self.samples = np.asarray(samples, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.device_dims = list(device_dims)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def device_offsets(device_dims):
    offs, s = [], 0
    for d in device_dims:
        offs.append(s)
        s += d
    return offs


def zero_devices(X, device_dims, device_ids):
    out = np.array(X, copy=True)
    offs = device_offsets(device_dims)
    for p in device_ids:
        s = offs[int(p)]
        out[:, :, s:s + device_dims[int(p)]] = 0.0
    return out


def set_seed(seed):
    import random
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_loaders(train_ds, val_ds, test_ds, sample_weight, seed=None):
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(sample_weight), len(train_ds), True, generator=gen)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    return train_loader, val_loader, test_loader


def eval_loader(ds):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))


from paired_protocol import paired_fit

@paired_fit
def fit_model(use_front, train_ds, val_ds, test_ds, n_cls, label_encoder,
              sample_weight, class_weight, ckpt_path, result_path=None,
              cm_path=None, tag='CNN', extra_evals=None, front_ver='v2',
              lam_contra=None, lam_domain=None, lam_gate=None,
              contra_temp=None, gate_target=None, model_cls=None, seed=42):
    if seed is not None:
        set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, sample_weight, seed=seed)
    cls = model_cls or PlugCNN
    model = cls(
        train_ds.device_dims, n_cls, use_front=use_front,
        d_model=16, d_hidden=32, dropout=DROPOUT, front_ver=front_ver,
    ).to(DEVICE)
    n_all = n_params(model)
    use_v2 = bool(use_front) and str(front_ver).lower() == 'v2'
    lc = LAM_CONTRA if lam_contra is None else float(lam_contra)
    ld = LAM_DOMAIN if lam_domain is None else float(lam_domain)
    lg = LAM_GATE if lam_gate is None else float(lam_gate)
    tau = CONTRA_TEMP if contra_temp is None else float(contra_temp)
    g0 = GATE_TARGET if gate_target is None else float(gate_target)
    if use_v2 and model.front is not None:
        model.front.set_gate_target(g0)
    if model.front is None:
        extra = '  feat=concat'
    elif use_v2:
        extra = (f'  front=v2 dim_gate+contra+dann'
                 f'  λc={lc} λd={ld} λg={lg} τ={tau} g0={g0}')
    else:
        extra = f'  gamma={float(model.front.gamma):.3f}'
    print(f'model={tag}  params={n_all/1e6:.3f}M  dims={train_ds.device_dims}{extra}')
    print(f'device={DEVICE}  window={WINDOW_SIZE}  step={TRAIN_STEP}/{EVAL_STEP}  '
          f'patience={PATIENCE}')
    print(f'save={ckpt_path}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)
    best_val_f1, best_epoch, bad = -1.0, 0, 0
    last_gate = None
    for epoch in range(1, EPOCHS + 1):
        print(f'\n--- {tag} Epoch {epoch}/{EPOCHS} ---')
        alpha = dann_alpha(epoch) if use_v2 else 1.0
        tr_loss, tr_f1, gate_mean = train_epoch(
            model, train_loader, optimizer, DEVICE, n_cls, class_weight,
            lam_contra=lc if use_v2 else 0.0,
            lam_domain=ld if use_v2 else 0.0,
            lam_gate=lg if use_v2 else 0.0,
            domain_alpha=alpha,
            contra_temp=tau,
        )
        last_gate = gate_mean
        val = evaluate(model, val_loader, DEVICE, label_encoder, desc='Valid')
        if model.front is None:
            gtxt = ''
        elif use_v2:
            gtxt = f' | gate={gate_mean:.3f} dann_a={alpha:.2f}'
        else:
            gtxt = f' | gamma={float(model.front.gamma):.3f}'
        print(f'Train Loss {tr_loss:.4f} | Train F1 {tr_f1:.4f} | '
              f'Val Macro-F1 {val["macro_f1"]:.4f}{gtxt}')
        if val['macro_f1'] > best_val_f1 + 1e-4:
            best_val_f1, best_epoch, bad = val['macro_f1'], epoch, 0
            torch.save({
                'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'epoch': best_epoch,
                'val_macro_f1': best_val_f1,
                'classes': list(label_encoder.classes_),
                'device_dims': list(train_ds.device_dims),
                'model': tag,
                'use_front': bool(use_front),
                'front_ver': model.front_ver,
                'seed': None if seed is None else int(seed),
            }, ckpt_path)
            print(f'  >> saved {ckpt_path}  Val Macro-F1={best_val_f1:.4f}')
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f'early stop (patience={PATIENCE})')
                break
        scheduler.step(val['macro_f1'])
        print(f'  lr={optimizer.param_groups[0]["lr"]:.2e}  bad={bad}/{PATIENCE}')

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    val = evaluate(model, val_loader, DEVICE, label_encoder, desc='Valid')
    test = evaluate(model, test_loader, DEVICE, label_encoder, desc='Test')
    extras = {}
    if extra_evals:
        for name, loader in extra_evals.items():
            extras[name] = evaluate(model, loader, DEVICE, label_encoder, desc=name)
            print(f'  {name}: Acc {extras[name]["accuracy"]:.4f}  '
                  f'P {extras[name]["macro_precision"]:.4f}  '
                  f'R {extras[name]["macro_recall"]:.4f}  '
                  f'F1 {extras[name]["macro_f1"]:.4f}')
    if model.front is None:
        g_now = None
        gtxt_end = ''
    elif use_v2:
        g_now = last_gate
        gtxt_end = f' | gate={last_gate:.3f}' if last_gate is not None else ''
    else:
        g_now = float(model.front.gamma)
        gtxt_end = f' | gamma={g_now:.3f}'
    print('\n' + '=' * 60)
    print(f'{tag}  best_ep={best_epoch}  params={n_all/1e6:.3f}M')
    print(f'Val Macro-F1 {val["macro_f1"]:.4f} | Test Acc {test["accuracy"]:.4f}  '
          f'P {test["macro_precision"]:.4f}  R {test["macro_recall"]:.4f}  '
          f'F1 {test["macro_f1"]:.4f}{gtxt_end}')
    print('=' * 60)
    print(test['report'])
    if cm_path:
        plot_confusion_matrix(test['y_true'], test['y_pred'], list(label_encoder.classes_), cm_path)
    if result_path:
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'model={tag}\n')
            f.write(f'params_M={n_all/1e6:.6f}\n')
            f.write(f'data=iot_fusion_weather_all.csv window={WINDOW_SIZE} '
                    f'step={TRAIN_STEP}/{EVAL_STEP}\n')
            f.write(f'use_front={int(bool(use_front))}\n')
            f.write(f'front_ver={model.front_ver}\n')
            f.write(f'best_epoch={best_epoch}\n')
            if seed is not None:
                f.write(f'seed={int(seed)}\n')
            f.write(f'val_macro_f1={val["macro_f1"]:.6f}\n')
            f.write(f'val_weighted_f1={val["weighted_f1"]:.6f}\n')
            f.write(f'test_accuracy={test["accuracy"]:.6f}\n')
            f.write(f'test_macro_precision={test["macro_precision"]:.6f}\n')
            f.write(f'test_macro_recall={test["macro_recall"]:.6f}\n')
            f.write(f'test_macro_f1={test["macro_f1"]:.6f}\n')
            f.write(f'test_weighted_f1={test["weighted_f1"]:.6f}\n')
            if use_v2:
                f.write(f'lam_contra={lc:.6f}\n')
                f.write(f'lam_domain={ld:.6f}\n')
                f.write(f'lam_gate={lg:.6f}\n')
                f.write(f'contra_temp={tau:.6f}\n')
                f.write(f'gate_target={g0:.6f}\n')
            if use_v2 and last_gate is not None:
                f.write(f'gate_mean={last_gate:.6f}\n')
            elif g_now is not None and not use_v2:
                f.write(f'gamma={g_now:.6f}\n')
            for name, ev in extras.items():
                f.write(f'{name}_accuracy={ev["accuracy"]:.6f}\n')
                f.write(f'{name}_macro_precision={ev["macro_precision"]:.6f}\n')
                f.write(f'{name}_macro_recall={ev["macro_recall"]:.6f}\n')
                f.write(f'{name}_macro_f1={ev["macro_f1"]:.6f}\n')
            f.write('\n===== VAL =====\n' + val['report'] + '\n')
            f.write('===== TEST =====\n' + test['report'] + '\n')
        print(f'saved: {ckpt_path} | {result_path}')
    else:
        print(f'saved: {ckpt_path}')
    return {
        'tag': tag, 'best_epoch': best_epoch,
        'val': val, 'test': test, 'gamma': g_now,
        'ckpt': ckpt_path, 'result': result_path,
        'extras': extras, 'params': n_all,
        'lam_contra': lc if use_v2 else None,
        'lam_domain': ld if use_v2 else None,
        'lam_gate': lg if use_v2 else None,
        'contra_temp': tau if use_v2 else None,
        'gate_target': g0 if use_v2 else None,
    }


from paired_protocol import paired_fit

@paired_fit
def fit(use_front=True):
    pack = load_datasets()
    ckpt_path, result_path, cm_path = artifact_paths(use_front)
    tag = 'Front-v2+CNN' if use_front else 'Raw+CNN'
    kw = dict(V2_HPARAM) if use_front else {}
    return fit_model(
        use_front=use_front, front_ver='v2' if use_front else 'v1',
        train_ds=pack['train_ds'], val_ds=pack['val_ds'], test_ds=pack['test_ds'],
        n_cls=pack['n_cls'], label_encoder=pack['label_encoder'],
        sample_weight=pack['sample_weight'], class_weight=pack['class_weight'],
        ckpt_path=ckpt_path, result_path=result_path, cm_path=cm_path, tag=tag,
        **kw,
    )


def _read_metrics(path):
    keys = ('model', 'params_M', 'best_epoch', 'val_macro_f1',
            'test_macro_f1', 'test_accuracy', 'gamma', 'use_front',
            'front_ver', 'gate_mean', 'lam_contra', 'lam_domain',
            'contra_temp', 'gate_target')
    kv = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('=====') or not line.strip() or '=' not in line:
                continue
            k, v = line.strip().split('=', 1)
            if k in keys:
                kv[k] = v
    return kv


def compare_results():
    pass
    rows = []
    for use_front, name in ((True, 'Front-v2+CNN'), (False, 'Raw+CNN')):
        _, path, _ = artifact_paths(use_front)
        if not os.path.isfile(path):
            print(f'missing {path}')
            continue
        kv = _read_metrics(path)
        rows.append((name, kv, path))
        print(f'{name}: val={kv.get("val_macro_f1")}  test={kv.get("test_macro_f1")}  '
              f'acc={kv.get("test_accuracy")}  ep={kv.get("best_epoch")}  '
              f'gate={kv.get("gate_mean", "-")}  ver={kv.get("front_ver", "-")}')
    if len(rows) == 2:
        def _f(d, k):
            try:
                return float(d[k])
            except (KeyError, TypeError, ValueError):
                return float('nan')
        d_front, d_raw = rows[0][1], rows[1][1]
        delta = _f(d_front, 'test_macro_f1') - _f(d_raw, 'test_macro_f1')
        print(f'\nTest Macro-F1 (Front-v2 - Raw) = {delta:+.4f}')
    return rows


def main(use_front=True):
    print(f'CNN contrast  use_front={use_front}  front=v2  device={DEVICE}  csv={DATA_CSV}')
    if use_front:
        print('V2_HPARAM', V2_HPARAM)
    return fit(use_front=use_front)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--raw', action='store_true', help='不加 ActivityGatedGAT')
    p.add_argument('--compare', action='store_true')
    args = p.parse_args()
    if args.compare:
        compare_results()
    else:
        main(use_front=not args.raw)
