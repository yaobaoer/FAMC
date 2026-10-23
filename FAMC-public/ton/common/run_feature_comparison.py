                       

from __future__ import annotations
import argparse
import gc
import json
import os
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, normalized_mutual_info_score)
from sklearn.cluster import KMeans
import run_cnn_align as cnn
from feature_backbones import build_paper_backbone
import plot_feature_clusters as viz

ROOT = Path(__file__).resolve().parent
JOBS = [('Ours', 'front'), ('CoLD', 'cold'), ('tFusion', 'tfusion'),
        ('TIPSO-Attn', 'tipsogan'), ('FeCo', 'feco'), ('AOC-IDS', 'aocids')]

class LazyWindows(Dataset):
    def __init__(self, x, y, step, limit, seed):
        self.x = x
        self.starts = np.arange(0, len(y) - cnn.WINDOW + 1, step)
        if len(self.starts) > limit:
            self.starts = np.sort(np.random.RandomState(seed).choice(self.starts, limit, False))
        self.labels = y[self.starts + cnn.WINDOW - 1]
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, i):
        start = self.starts[i]
        return torch.from_numpy(self.x[start:start + cnn.WINDOW]), torch.tensor(self.labels[i])

def load_data(args, out):
    cache = out / 'data.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}
    columns = pd.read_csv(cnn.DATA_CSV, nrows=0).columns
    feats = [c for c in columns if c not in {'device', 'device_full', 'final_type',
             'final_label', 'label_encoded'} and not c.startswith('HH_jit_')]
    rng = np.random.RandomState(42)
    reservoirs = {}
    started = time.perf_counter()
    for chunk_id, chunk in enumerate(pd.read_csv(cnn.DATA_CSV,
            usecols=feats + ['device', 'final_type'], chunksize=25000,
            dtype={c: 'float32' for c in feats})):
        chunk['_priority'] = rng.random_sample(len(chunk))
        for key, group in chunk.groupby(['device', 'final_type'], sort=False):
            previous = reservoirs.get(key)
            merged = group if previous is None else pd.concat([previous, group], ignore_index=True)
            reservoirs[key] = merged.nsmallest(args.rows_per_group, '_priority')
        if chunk_id % 20 == 0:
            print(f'DATA chunks={chunk_id + 1} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    df = pd.concat(list(reservoirs.values()), ignore_index=True).drop(columns='_priority')
    del reservoirs
    names = sorted(df.final_type.unique())
    df['label_encoded'] = df.final_type.map({name: i for i, name in enumerate(names)}).astype('int64')
    key = df.device.astype(str) + '/' + df.final_type.astype(str)
    trva, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=key, random_state=42)
    train, val = train_test_split(trva, test_size=.2, stratify=key.iloc[trva], random_state=43)
    scaler = StandardScaler().fit(df.iloc[train][feats])
                                                                  
    df.loc[:, feats] = np.nan_to_num(np.clip(scaler.transform(df[feats]), -8, 8),
                                     nan=0., posinf=8., neginf=-8.).astype('float32')
    result = {'names': np.asarray(names), 'dims': np.full(9, len(feats), dtype='int64')}
    for label, idx, seed in [('train', train, 42), ('val', val, 43), ('test', test, 44)]:
        x, y = cnn.build_aligned(df.iloc[idx], feats, seed=seed)
        if label != 'test':
            x[:, :len(feats)] = 0.                                                 
        result[f'{label}_x'], result[f'{label}_y'] = x, y
    np.savez_compressed(cache, **result)
    print(f'DATA_READY elapsed={time.perf_counter()-started:.1f}s', flush=True)
    del df
    gc.collect()
    return result

class Model(nn.Module):
    def __init__(self, dims, n_cls, kind, d_model):
        super().__init__()
        self.kind = kind
        if kind == 'front':
            self.extractor = cnn.FrontV2(dims, d_model=d_model, dropout=.1)
            self.extractor.use_gat = False
            self.extractor.set_gate_target(.12)
        else:
            self.extractor = build_paper_backbone(kind, int(sum(dims)), d_hidden=64)
        self.backbone = cnn.CNNBackbone(self.extractor.out_dim, 64)
        self.classifier = nn.Linear(64, n_cls)
    def forward(self, x):
        if self.kind == 'front':
            seq, aux = self.extractor(x, return_features=True)
        else:
            seq, aux = self.extractor(x), None
                                                                         
        return self.classifier(self.backbone(seq)[:, -1]), seq[:, -1], aux

def supcon(z, y, temperature=.1):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    diagonal = torch.eye(len(y), dtype=torch.bool, device=y.device)
    positives = y[:, None].eq(y[None, :]) & ~diagonal
    log_prob = sim - torch.logsumexp(sim.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    counts = positives.sum(1)
    valid = counts > 0
    if not valid.any():
        return z.sum() * 0.
    return -((log_prob.masked_fill(~positives, 0.).sum(1) / counts.clamp_min(1))[valid]).mean()

@torch.no_grad()
def evaluate(model, loader, feature_idx=None):
    model.eval()
    truth, pred, features = [], [], []
    for x, y in loader:
        logits, z, _ = model(x.to(cnn.DEVICE))
        truth.append(y.numpy()); pred.append(logits.argmax(1).cpu().numpy())
        features.append(z.cpu().numpy())
    y = np.concatenate(truth)
    z = np.concatenate(features)
    f1 = f1_score(y, np.concatenate(pred), average='macro', zero_division=0)
    if feature_idx is not None:
        z, y = z[feature_idx], y[feature_idx]
    return f1, z, y

def metrics(z, y, seed):
    z = normalize(z)
    cluster = KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=seed).fit_predict(z)
    return dict(silhouette=float(silhouette_score(z, y)),
        davies=float(davies_bouldin_score(z, y)), calinski=float(calinski_harabasz_score(z, y)),
        ari=float(adjusted_rand_score(y, cluster)), nmi=float(normalized_mutual_info_score(y, cluster)))

def train_one(kind, seed, data, loaders, args, out):
    cnn.set_seed(seed)
    n_cls = len(data['names'])
    model = Model(data['dims'].tolist(), n_cls, kind, args.d_model).to(cnn.DEVICE)
    checkpoint = out / f'{kind}_s{seed}.pth'
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=.5)
    counts = np.bincount(loaders['train'].dataset.labels, minlength=n_cls).clip(1)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum() / (n_cls * counts)), .5, 3),
                      dtype=torch.float32, device=cnn.DEVICE)
    val_idx = viz.stratified_idx(loaders['val'].dataset.labels, 70, 0)
    best, bad, best_epoch = -float('inf'), 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.
        for x, y in loaders['train']:
            x, y = x.to(cnn.DEVICE), y.to(cnn.DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, z, aux = model(x)
            loss = F.cross_entropy(logits, y, weight=cw)
            if kind == 'front':
                last_nodes = aux['h_prime'][:, -1:]
                loss = loss + args.supcon * supcon(z, y)
                loss = loss + .1 * model.extractor.cross_protocol_contrastive_loss(last_nodes, y)
                loss = loss + .3 * model.extractor.gate_reg_loss(aux['gate'])
                                                                     
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            total += loss.item()
        val_f1, z_val, y_val = evaluate(model, loaders['val'], val_idx)
        val_sil = float(silhouette_score(normalize(z_val), y_val))
        scheduler.step(val_sil)
        seconds = time.perf_counter() - started
        row = dict(epoch=epoch, loss=total/len(loaders['train']), val_f1=val_f1,
                   val_silhouette=val_sil, seconds=seconds)
        history.append(row)
        pd.DataFrame(history).to_csv(out / f'{kind}_s{seed}_history.csv', index=False)
        print(f'EPOCH {kind} seed={seed} ep={epoch} loss={row["loss"]:.4f} '
              f'valF1={val_f1:.4f} valSil={val_sil:.4f} seconds={seconds:.1f}', flush=True)
        if val_sil > best + .0001:
            best, bad, best_epoch = val_sil, 0, epoch
            torch.save(dict(model_state=model.state_dict(), best_epoch=epoch,
                            val_silhouette=best, seed=seed, kind=kind, args=vars(args)), checkpoint)
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=cnn.DEVICE, weights_only=False)['model_state'])
    return model, best_epoch

def smoke(args):
    cnn.set_seed(42)
    for name, kind in JOBS:
        model = Model([100]*9, 11, kind, args.d_model).to(cnn.DEVICE)
        x = torch.randn(8, 64, 900, device=cnn.DEVICE); x[:, :, :100] = 0.
        y = torch.arange(8, device=cnn.DEVICE) % 4
        logits, z, aux = model(x)
        loss = F.cross_entropy(logits, y)
        if kind == 'front':
            assert torch.count_nonzero(aux['h_prime'][:, :, 0]) == 0
            loss = loss + supcon(z, y) + model.extractor.cross_protocol_contrastive_loss(aux['h_prime'][:, -1:], y)
            loss = loss + model.extractor.domain_adversarial_loss(aux['h_prime'])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print('SMOKE_OK', name, tuple(z.shape), flush=True)
    print('SMOKE_DONE', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='cluster_retrain/pre_model_v1')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--rows-per-group', type=int, default=3000)
    parser.add_argument('--max-windows', type=int, default=12000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=32)
    parser.add_argument('--supcon', type=float, default=.2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.smoke:
        smoke(args); return
    out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    manifest = dict(args=vars(args), device=str(cnn.DEVICE), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        feature_location='extractor output, last position, before CNN',
        checkpoint_selection='validation original-space silhouette',
        protocol='synthetic same-label device alignment; train/val drop danmini; test nine devices',
        baselines='supervised core adaptations, not full paper reproductions',
        preprocessing='balanced random reservoir; stratified source split; training-only scaler; preserve missing zeros')
    manifest['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ['run_feature_comparison.py', 'feature_backbones.py', 'activity_gated_gat_v2.py']}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous != manifest:
            raise ValueError('Output directory belongs to a different configuration; choose a new --output.')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('START', json.dumps(manifest, ensure_ascii=False), flush=True)
    data = load_data(args, out)
    ds = {name: LazyWindows(data[f'{name}_x'], data[f'{name}_y'],
          4 if name == 'train' else 8, args.max_windows if name == 'train' else args.max_windows//2,
          42+i) for i, name in enumerate(['train', 'val', 'test'])}
    print('WINDOWS', {key: len(value) for key, value in ds.items()}, flush=True)
    path = out / 'metrics.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    idx = viz.stratified_idx(ds['test'].labels, 70, 0)
    for seed in [int(s) for s in args.seeds.split(',')]:
        counts = np.bincount(ds['train'].labels).clip(1)
        sampler = WeightedRandomSampler(torch.as_tensor(1/np.sqrt(counts[ds['train'].labels]), dtype=torch.double),
                   len(ds['train']), generator=torch.Generator().manual_seed(seed))
        loaders = {name: DataLoader(dataset, batch_size=args.batch,
                   sampler=sampler if name == 'train' else None, num_workers=0,
                   pin_memory=torch.cuda.is_available()) for name, dataset in ds.items()}
        for title, kind in JOBS:
            if any(row['kind'] == kind and int(row['seed']) == seed for row in rows):
                continue
                                                                   
            sampler.generator.manual_seed(seed)
            model, epoch = train_one(kind, seed, data, loaders, args, out)
            f1, z, y = evaluate(model, loaders['test'], idx)
            scores = metrics(z, y, seed)
            row = dict(method=title, kind=kind, seed=seed, epoch=epoch, f1=f1, dim=z.shape[1], **scores)
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            np.savez_compressed(out / f'{kind}_s{seed}_features.npz', features=z, labels=y, indices=idx)
            print('RESULT', json.dumps(row), flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        panels = []
        for title, kind in JOBS:
            with np.load(out / f'{kind}_s{seed}_features.npz') as saved:
                z, y = saved['features'], saved['labels']
            panels.append((title, viz.project(z), float(silhouette_score(normalize(z), y))))
        viz.draw(panels, y, [str(n).replace('_', ' ') for n in data['names']],
                 str(out / f'clusters_s{seed}.png'), str(out / f'clusters_s{seed}.pdf'),
                 'Pre-CNN extractor outputs at the labeled last position. Number: original-space silhouette.')
        pd.DataFrame(rows).groupby('method')[['f1', 'silhouette', 'davies', 'ari', 'nmi']].agg(['mean', 'std', 'count']).to_csv(out / 'summary.csv')
    print('FEATURE_COMPARISON_DONE', flush=True)

if __name__ == '__main__':
    main()
