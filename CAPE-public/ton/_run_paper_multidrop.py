                       
pass
from __future__ import annotations

import fcntl
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _run_cnn as base
import _run_paper_adapt as paper

base._ROOT = _HERE
base.DATA_CSV = os.path.join(_HERE, 'iot_fusion_weather_all.csv')
paper._ROOT = _HERE
os.chdir(_HERE)

OUT_DIR = os.path.join(_HERE, 'paper_multidrop_ckpts')
ROWS_PATH = os.path.join(OUT_DIR, 'rows.jsonl')

                               
ORDER = ('modbus', 'weather', 'gps', 'fridge', 'thermostat')
KS = (2, 3, 4, 5)
KINDS = ('cold', 'tfusion', 'tipsogan', 'rapier', 'hypervision')
TRAIN_SEED = 42
EVAL_SEEDS = (42, 123, 2024)


class IndexedWinDS(Dataset):
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = np.asarray(indices)
        self.labels = [ds.labels[int(i)] for i in self.indices]
        self.device_dims = ds.device_dims

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.ds[int(self.indices[i])]


def stratified_bootstrap_idx(labels, seed):
    rng = np.random.RandomState(int(seed))
    labels = np.asarray(labels)
    parts = []
    for c in np.unique(labels):
        pos = np.where(labels == c)[0]
        parts.append(rng.choice(pos, size=len(pos), replace=True))
    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return idx


def drop_ids(k):
    names = list(base.DEVICE_PREFIXES)
    return [names.index(n) for n in ORDER[:k]]


def drop_tag(k):
    return '-'.join(ORDER[:k])


def load_rows(path=None):
    path = ROWS_PATH if path is None else path
    rows = []
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_merged_rows():
    canon = os.path.join(OUT_DIR, 'rows.jsonl')
    paths = [canon]
    paths.extend(sorted(glob.glob(os.path.join(OUT_DIR, 'rows_*.jsonl'))))
    rows, seen = [], set()
    for path in paths:
        for r in load_rows(path):
            key = (r['kind'], int(r['k']), r['model'], r['eval'], int(r['seed']))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    return rows


def append_rows(rows):
    with open(ROWS_PATH, 'a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def done_keys(rows):
    return {(r['kind'], int(r['k']), r['model'], r['eval']) for r in rows}


def load_best(ckpt, pack, use_front):
    model = paper.MODEL_CLS(
        pack['train_ds'].device_dims, pack['n_cls'],
        use_front=use_front, d_model=16, d_hidden=32,
        dropout=base.DROPOUT, front_ver='v2' if use_front else 'v1',
    ).to(base.DEVICE)
    blob = torch.load(ckpt, map_location=base.DEVICE, weights_only=False)
    model.load_state_dict(blob['model_state'])
    model.eval()
    return model, blob


def eval_seeds(model, ds, pack, seeds, desc):
    rows = []
    labels = np.asarray(ds.labels)
    for seed in seeds:
        sub = IndexedWinDS(ds, stratified_bootstrap_idx(labels, seed))
        ev = base.evaluate(
            model, base.eval_loader(sub), base.DEVICE,
            pack['label_encoder'], desc=f'{desc}-s{seed}')
        print(f'  {desc} seed={seed}: Acc {ev["accuracy"]:.4f}  '
              f'P {ev["macro_precision"]:.4f}  R {ev["macro_recall"]:.4f}  '
              f'F1 {ev["macro_f1"]:.4f}', flush=True)
        rows.append(dict(
            seed=int(seed),
            acc=ev['accuracy'],
            precision=ev['macro_precision'],
            recall=ev['macro_recall'],
            f1=ev['macro_f1'],
        ))
    return rows


def write_summary(rows=None):
    import pandas as pd
    lock_path = os.path.join(OUT_DIR, '.summary.lock')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(lock_path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = load_merged_rows()
        if not rows:
            return
        _write_summary_body(rows)


def _write_summary_body(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    lines = [
        'paper multidrop | Front-v2 vs Raw | test=FULL is the headline',
        'order by mean drop-1 FULL macro-F1 gap of these 5 methods, high to low',
        f'order={list(ORDER)}',
        'left out: garage_door (+0.007), motion_light (-0.002)',
        '',
    ]
    cols = ('acc', 'precision', 'recall', 'f1')
    for eval_name, title in (('full', 'TEST=FULL'), ('drop', 'TEST=DROP')):
        sub = df[df['eval'] == eval_name]
        if sub.empty:
            continue
        lines.append(f'===== {title} =====')
        for k in KS:
            sk = sub[sub.k == k]
            if sk.empty:
                continue
            lines.append(f'-- k={k} drop={drop_tag(k)} --')
            for kind in KINDS:
                g = sk[sk.kind == kind]
                if g.empty:
                    continue
                for model, mg in g.groupby('model'):
                    lines.append(
                        f'  {model:28s}  acc {mg.acc.mean():.4f}  '
                        f'precision {mg.precision.mean():.4f}  '
                        f'recall {mg.recall.mean():.4f}  f1 {mg.f1.mean():.4f}'
                    )
                names = set(g.model)
                front = [n for n in names if str(n).startswith('Front')]
                raw = [n for n in names if str(n).startswith('Raw')]
                if front and raw:
                    a = g[g.model == front[0]].set_index('seed')
                    b = g[g.model == raw[0]].set_index('seed')
                    if len(a) and len(b):
                        parts = [f'{c} {(a[c] - b[c]).mean():+.4f}' for c in cols]
                        lines.append(f'  gap {front[0]} - {raw[0]}  ' + '  '.join(parts))
            lines.append('')
    text = '\n'.join(lines).rstrip() + '\n'
    path = os.path.join(OUT_DIR, 'paper_multidrop_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('\n' + text, flush=True)


def run_pair(pack, kind, k, done):
    ids = drop_ids(k)
    tag = drop_tag(k)
    train_ds, val_ds, test_ds = pack['train_ds'], pack['val_ds'], pack['test_ds']
    dims = list(train_ds.device_dims)
    tr = base.ArrayWinDS(base.zero_devices(train_ds.samples, dims, ids), train_ds.labels, dims)
    va = base.ArrayWinDS(base.zero_devices(val_ds.samples, dims, ids), val_ds.labels, dims)
    te = base.ArrayWinDS(base.zero_devices(test_ds.samples, dims, ids), test_ds.labels, dims)
    sub = dict(pack)
    sub.update(train_ds=tr, val_ds=va, test_ds=te)
    full_loader = base.eval_loader(test_ds)
    new = []
    for use_front, stem in ((True, 'front'), (False, 'raw')):
        name = paper.FRONT_NAME if use_front else paper.RAW_NAME
        need = any((kind, k, name, ev) not in done for ev in ('full', 'drop'))
        if not need:
            print(f'skip {kind} {name} k={k}', flush=True)
            continue
        ckpt = os.path.join(OUT_DIR, f'best_{kind}_{stem}_k{k}.pth')
        res = os.path.join(OUT_DIR, f'{kind}_{stem}_k{k}_train.txt')
        if os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1000:
            model, blob = load_best(ckpt, pack, use_front)
            best_ep = int(blob.get('epoch', -1))
            print(f'reuse {ckpt} ep={best_ep}', flush=True)
        else:
            print(f'\n# {name} k={k} drop={tag}', flush=True)
            out = paper._fit(
                use_front, sub, ckpt, res, f'{name}-k{k}',
                extra_evals={'test_full': full_loader}, seed=TRAIN_SEED)
            best_ep = int(out['best_epoch'])
            model, _ = load_best(ckpt, pack, use_front)
        for eval_name, ds in (('full', test_ds), ('drop', te)):
            if (kind, k, name, eval_name) in done:
                continue
            print(f'# 3-seed {name} k={k} eval={eval_name}', flush=True)
            for ev in eval_seeds(model, ds, pack, EVAL_SEEDS, f'{name}-k{k}-{eval_name}'):
                new.append(dict(
                    kind=kind, k=k, subset=tag, model=name, eval=eval_name,
                    best_epoch=best_ep, **ev,
                ))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if new:
        append_rows(new)
    return new


def main():
    global ROWS_PATH
    os.makedirs(OUT_DIR, exist_ok=True)
    only = [a for a in sys.argv[1:] if not a.startswith('-')]
    kinds = only or list(KINDS)
    if len(kinds) == 1:
        ROWS_PATH = os.path.join(OUT_DIR, f'rows_{kinds[0]}.jsonl')
    print('drop order', list(ORDER), 'ks', KS, 'kinds', kinds, flush=True)
    print('rows', ROWS_PATH, flush=True)
    print('DEVICE_PREFIXES', list(base.DEVICE_PREFIXES), flush=True)
    pack = base.load_datasets()
    rows = load_rows()
    done = done_keys(rows)
    torch.backends.cudnn.benchmark = True
    for kind in kinds:
        paper.configure(kind)
        print(f'\n========== {kind} {paper.TAG} V2={base.V2_HPARAM} ==========', flush=True)
        for k in KS:
            got = run_pair(pack, kind, k, done)
            rows.extend(got)
            for r in got:
                done.add((r['kind'], int(r['k']), r['model'], r['eval']))
            write_summary(rows)
    print('PAPER_MULTIDROP_DONE', flush=True)


if __name__ == '__main__':
    main()
