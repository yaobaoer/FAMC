                       

from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '3')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import run_feature_comparison as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cluster_retrain' / 'capacity_enhanced_v1'
TITLES = {'enhanced': 'Ours enhanced', 'feco_small': 'FeCo small', 'aoc_small': 'AOC-IDS small'}
CONFIGS = [
    dict(name='metric_light', supcon=.2, prototype=.5, compact=.1),
    dict(name='metric_medium', supcon=.5, prototype=1., compact=.2),
    dict(name='metric_strong', supcon=1., prototype=1., compact=.5),
]

class SmallFeCo(nn.Module):
    out_dim = 16
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.net = nn.Sequential(nn.Linear(dims[0], 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(.1), nn.Linear(32, self.out_dim))
    def forward(self, x):
        h = x.reshape(x.shape[0], x.shape[1], len(self.dims), self.dims[0])
        valid = h.detach().norm(dim=-1) > 1e-6
        z = self.net(h)
        return (z * valid.unsqueeze(-1)).sum(2) / valid.sum(2, keepdim=True).clamp_min(1)

class SmallAOC(nn.Module):
    out_dim = 16
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.enc = nn.Sequential(nn.Linear(dims[0], 32), nn.ReLU(), nn.Linear(32, 16))
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, dims[0]))
        self.out = nn.Sequential(nn.Dropout(.1), nn.Linear(16 + dims[0], 16))
    def forward(self, x):
        h = x.reshape(x.shape[0], x.shape[1], len(self.dims), self.dims[0])
        z = self.enc(h)
        return self.out(torch.cat([z, self.dec(z)], dim=-1)).mean(2)

class Model(nn.Module):
    def __init__(self, dims, n_cls, kind):
        super().__init__()
        self.kind = kind
        if kind == 'enhanced':
            self.extractor = base.cnn.FrontV2(dims, d_model=32, dropout=.1)
            self.extractor.use_gat = False
            self.extractor.set_gate_target(.12)
                                                                               
            self.metric_projection = nn.Sequential(nn.Linear(self.extractor.out_dim, 128),
                nn.GELU(), nn.LayerNorm(128), nn.Linear(128, 64))
            self.prototypes = nn.Parameter(torch.randn(n_cls, 64))
            width = 64
        else:
            self.extractor = SmallFeCo(dims) if kind == 'feco_small' else SmallAOC(dims)
            width = self.extractor.out_dim
        self.backbone = base.cnn.CNNBackbone(width, 64)
        self.classifier = nn.Linear(64, n_cls)
    def forward(self, x):
        if self.kind == 'enhanced':
            seq, aux = self.extractor(x, return_features=True)
            seq = F.normalize(self.metric_projection(seq), dim=-1)
        else:
            seq, aux = self.extractor(x), None
        return self.classifier(self.backbone(seq)[:, -1]), seq[:, -1], aux

def fit(kind, seed, config, data, datasets, args, tag):
    base.cnn.set_seed(seed)
    counts = np.bincount(datasets['train'].labels, minlength=len(data['names'])).clip(1)
    sampler = WeightedRandomSampler(torch.tensor(1/np.sqrt(counts[datasets['train'].labels]),
                  dtype=torch.double), len(datasets['train']),
                  generator=torch.Generator().manual_seed(seed))
    train = DataLoader(datasets['train'], batch_size=64, sampler=sampler, pin_memory=True)
    val = DataLoader(datasets['val'], batch_size=64, pin_memory=True)
    idx = base.viz.stratified_idx(datasets['val'].labels, 70, 0)
    model = Model(data['dims'].tolist(), len(data['names']), kind).to(base.cnn.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=.5)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum()/(len(counts)*counts)), .5, 3),
                      dtype=torch.float32, device=base.cnn.DEVICE)
    path = OUT / f'{tag}.pth'
    best, bad, history = -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train(); losses = []
        for x, y in train:
            x, y = x.to(base.cnn.DEVICE), y.to(base.cnn.DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, z, aux = model(x)
            loss = F.cross_entropy(logits, y, weight=cw)
            if kind == 'enhanced':
                centers = F.normalize(model.prototypes, dim=-1)
                cosine = z @ centers.T
                margin = F.one_hot(y, len(counts)).to(cosine.dtype) * .15
                loss = loss + config['prototype'] * F.cross_entropy(16*(cosine-margin), y, weight=cw)
                loss = loss + config['supcon'] * base.supcon(z, y)
                loss = loss + config['compact'] * (1-(z*centers[y]).sum(-1)).mean()
                offdiag = ~torch.eye(len(counts), dtype=torch.bool, device=y.device)
                loss = loss + .05 * F.relu((centers@centers.T)[offdiag]-.1).square().mean()
                loss = loss + .1 * model.extractor.cross_protocol_contrastive_loss(aux['h_prime'][:, -1:], y)
                loss = loss + .3 * model.extractor.gate_reg_loss(aux['gate'])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            losses.append(loss.item())
        f1, z, y = base.evaluate(model, val, idx)
        sil = float(silhouette_score(normalize(z), y))
        scheduler.step(sil)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), val_f1=f1,
                   val_silhouette=sil, seconds=time.perf_counter()-started)
        history.append(row)
        pd.DataFrame(history).to_csv(OUT/f'{tag}_history.csv', index=False)
        print('EPOCH', tag, json.dumps(row), flush=True)
        if sil > best + .0001:
            best, bad = sil, 0
            torch.save(dict(model_state=model.state_dict(), seed=seed, kind=kind, config=config,
                epoch=epoch, val_silhouette=sil, val_f1=f1,
                parameters=sum(p.numel() for p in model.parameters())), path)
        else:
            bad += 1
            if bad >= 5: break
    del model
    torch.cuda.empty_cache()
    return torch.load(path, map_location='cpu', weights_only=False)

def smoke():
    for kind in TITLES:
        model = Model([100]*9, 11, kind).to(base.cnn.DEVICE)
        x = torch.randn(8, 64, 900, device=base.cnn.DEVICE); x[:, :, :100] = 0.
        y = torch.arange(8, device=base.cnn.DEVICE) % 4
        logits, z, aux = model(x)
        loss = F.cross_entropy(logits, y)
        if kind == 'enhanced':
            assert torch.count_nonzero(aux['h_prime'][:, :, 0]) == 0
            centers = F.normalize(model.prototypes, dim=-1)
            loss = loss + base.supcon(z, y) + F.cross_entropy(16*z@centers.T, y)
            loss = loss + (1-(z*centers[y]).sum(-1)).mean()
            loss = loss + model.extractor.cross_protocol_contrastive_loss(aux['h_prime'][:, -1:], y)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print('SMOKE_OK', kind, 'dim', z.shape[-1], 'params', sum(p.numel() for p in model.parameters()), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.smoke: smoke(); return
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'cluster_retrain/pre_model_v1'
    manifest = dict(args=vars(args), configs=CONFIGS, seeds=[42],
        source_data=str(source/'data.npz'), source_data_sha256=hashlib.sha256((source/'data.npz').read_bytes()).hexdigest(),
        source_hashes={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['run_feature_capacity.py', 'run_feature_comparison.py', 'feature_backbones.py', 'activity_gated_gat_v2.py']},
        protocol='Same frozen data/windows/test subset as pre_model_v1; small-capacity ablation, not a fair full-baseline replacement.',
        selection='Compare three enhanced configs on validation seed 42 only; lock winner before testing.',
        small_feco='hidden 128/256/128 -> 32/64/32; output 64 -> 16',
        small_aoc='encoder/decoder width 128 -> 32; latent/output 64 -> 16',
        enhanced='Front + learned 288-to-64 metric projection, normalized output, prototype margin CE, compactness and supervised contrastive objectives')
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text('utf-8')) != manifest:
        raise ValueError('Configuration changed: use a new output directory.')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    with np.load(source/'data.npz', allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    ds = {name: base.LazyWindows(data[f'{name}_x'], data[f'{name}_y'], 4 if name=='train' else 8,
                12000 if name=='train' else 6000, 42+i) for i,name in enumerate(['train','val','test'])}
    print('START_CAPACITY', json.dumps(manifest, ensure_ascii=False), flush=True)
    selection_path = OUT/'selection.json'
    if selection_path.exists():
        selected = json.loads(selection_path.read_text('utf-8'))['config']
    else:
        trials = []
        for config in CONFIGS:
            tag = 'trial_'+config['name']+'_s42'
            ckpt = OUT/f'{tag}.pth'
                                                                                            
            completed = OUT/f'{tag}_done.json'
            if completed.exists():
                info = json.loads(completed.read_text('utf-8'))
            else:
                blob = fit('enhanced', 42, config, data, ds, args, tag)
                info = dict(config=config, val_silhouette=blob['val_silhouette'], val_f1=blob['val_f1'],
                            epoch=blob['epoch'], checkpoint=str(ckpt))
                completed.write_text(json.dumps(info, indent=2), encoding='utf-8')
            trials.append(info)
        winner = max(trials, key=lambda row: row['val_silhouette'])
        selected = winner['config']
        selection_path.write_text(json.dumps(dict(config=selected, trials=trials,
                                   rule='maximum validation silhouette; no test access'), indent=2), encoding='utf-8')
        print('VALIDATION_SELECTION', json.dumps(winner), flush=True)
    path = OUT/'metrics.csv'
    rows = pd.read_csv(path).to_dict('records') if path.exists() else []
    test_loader = DataLoader(ds['test'], batch_size=64, pin_memory=True)
    idx = base.viz.stratified_idx(ds['test'].labels, 70, 0)
    originals = pd.read_csv(source/'metrics.csv')
    for seed in [42]:
        for kind, title in TITLES.items():
            if any(row['kind']==kind and int(row['seed'])==seed for row in rows): continue
            if kind=='enhanced' and seed==42:
                checkpoint = OUT/f'trial_{selected["name"]}_s42.pth'
                blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
            else:
                blob = fit(kind, seed, selected if kind=='enhanced' else {}, data, ds, args, f'{kind}_s{seed}')
            model = Model(data['dims'].tolist(), len(data['names']), kind).to(base.cnn.DEVICE)
            model.load_state_dict(blob['model_state'])
            f1,z,y = base.evaluate(model, test_loader, idx)
            row = dict(method=title, kind=kind, seed=seed, epoch=blob['epoch'], f1=f1,
                       dim=z.shape[-1], parameters=blob['parameters'], **base.metrics(z,y,seed))
            rows.append(row); pd.DataFrame(rows).to_csv(path,index=False)
            np.savez_compressed(OUT/f'{kind}_s{seed}_features.npz', features=z,labels=y,indices=idx)
            print('RESULT_CAPACITY', json.dumps(row), flush=True)
            del model; torch.cuda.empty_cache()
        panels=[]
        entries=[('Ours enhanced',OUT/f'enhanced_s{seed}_features.npz')]
        entries += [(title,source/f'{kind}_s{seed}_features.npz') for title,kind in base.JOBS[1:4]]
        entries += [('FeCo small',OUT/f'feco_small_s{seed}_features.npz'),
                    ('AOC-IDS small',OUT/f'aoc_small_s{seed}_features.npz')]
        for title, feature_path in entries:
            with np.load(feature_path) as saved:
                z,y=saved['features'],saved['labels']
                assert np.array_equal(saved['indices'],idx)
            panels.append((title,base.viz.project(z),float(silhouette_score(normalize(z),y))))
        base.viz.draw(panels,y,[str(name).replace('_',' ') for name in data['names']],
            str(OUT/f'capacity_clusters_s{seed}.png'),str(OUT/f'capacity_clusters_s{seed}.pdf'),
            'Capacity ablation: enhanced Ours; small FeCo/AOC-IDS; other methods unchanged. Pre-CNN features.')
                                                                                
        full_panels=[]
        full_entries=[('Ours enhanced',OUT/f'enhanced_s{seed}_features.npz')]
        full_entries += [(title,source/f'{kind}_s{seed}_features.npz') for title,kind in base.JOBS[1:]]
        for title,feature_path in full_entries:
            with np.load(feature_path) as saved:
                z,y=saved['features'],saved['labels']
                assert np.array_equal(saved['indices'],idx)
            full_panels.append((title,base.viz.project(z),float(silhouette_score(normalize(z),y))))
        base.viz.draw(full_panels,y,[str(name).replace('_',' ') for name in data['names']],
            str(OUT/f'full_baseline_clusters_s{seed}.png'),str(OUT/f'full_baseline_clusters_s{seed}.pdf'),
            'Enhanced Ours versus original full-capacity baselines. Same frozen test subset; pre-CNN features.')
        combined=pd.concat([originals,pd.DataFrame(rows)],ignore_index=True)
        combined.to_csv(OUT/'full_and_small_metrics.csv',index=False)
        combined.groupby('method')[['f1','silhouette','davies','ari','nmi']].agg(['mean','std','count']).to_csv(OUT/'summary.csv')
    print('CAPACITY_ENHANCED_DONE',flush=True)

if __name__=='__main__': main()
