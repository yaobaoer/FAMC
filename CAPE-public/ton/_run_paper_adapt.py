                       
pass
from __future__ import annotations

import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

_ROOT = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
if not os.path.isfile(os.path.join(_ROOT, 'iot_fusion_weather_all.csv')):
    _ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import pandas as pd
import torch
import torch.nn as nn

import _run_cnn as base
from activity_gated_gat import ActivityGatedGAT
from activity_gated_gat_v2 import ActivityGatedGAT as ActivityGatedGATv2
from activity_gated_gat_v2 import n_params
from paper_backbones import PAPER_BACKBONES, build_paper_backbone

DEFAULT_KINDS = ('tfusion', 'tipsogan', 'cold', 'rapier', 'hypervision')

base._ROOT = _ROOT
base.DATA_CSV = os.path.join(_ROOT, 'iot_fusion_weather_all.csv')

TRAIN_SEED = 42
KIND = 'tfusion'
TAG = 'tFusion'
FRONT_NAME = 'Front-v2+tFusion'
RAW_NAME = 'Raw+tFusion'
OUT_DIR = ''
LODO_DIR = ''
MODEL_CLS = None
DO_FULL = False


class PlugPaper(nn.Module):
    pass

    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2',
                 backbone='tfusion'):
        super().__init__()
        self.front = None
        self.front_ver = str(front_ver).lower() if use_front else 'none'
        if use_front:
            cls = ActivityGatedGATv2 if self.front_ver == 'v2' else ActivityGatedGAT
            self.front = cls(device_dims, d_model=d_model, gat_heads=1, dropout=dropout)
            d_in = self.front.out_dim
        else:
            d_in = int(sum(device_dims))
        kind = str(backbone).lower()
        self.device_dims = [int(d) for d in device_dims]
        self.perflid = kind == 'perflid'
        self.aoc = kind == 'aoc'
        self.feco = kind == 'feco'
        self.backbone = build_paper_backbone(
            kind, d_in, d_hidden=d_hidden, dropout=dropout,
            device_dims=self.device_dims,
            front_dim=self.front.out_dim if self.front is not None else 0)
        self.head = base.DualPoolHead(self.backbone.out_dim, num_classes)
        self.needs_aux = self.front_ver == 'v2'
        self._kind = kind

    def forward(self, x, return_aux=False):
        aux = None
        if self.front is None:
            seq = self.backbone(x)
        elif self.front_ver == 'v2':
            seq, aux = self.front(x, return_features=True)
            seq = self.backbone(seq)
        else:
            seq = self.backbone(self.front(x))
        logits = self.head(seq)
        if return_aux:
            return logits, aux if aux is not None else {}
        return logits


def configure(kind: str):
    global KIND, TAG, FRONT_NAME, RAW_NAME, OUT_DIR, LODO_DIR, MODEL_CLS
    kind = str(kind).lower().strip()
    if kind not in PAPER_BACKBONES:
        raise ValueError(f'unknown kind={kind}, expect {list(PAPER_BACKBONES)}')
    KIND = kind
    TAG, _ = PAPER_BACKBONES[kind]
    FRONT_NAME = f'Front-v2+{TAG}'
    RAW_NAME = f'Raw+{TAG}'
    OUT_DIR = os.path.join(_ROOT, f'paper_{kind}_ckpts')
    LODO_DIR = os.path.join(_ROOT, f'paper_{kind}_lodo_ckpts')

    class _Plug(PlugPaper):
        def __init__(self, device_dims, num_classes, use_front=True,
                     d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
            super().__init__(
                device_dims, num_classes, use_front=use_front,
                d_model=d_model, d_hidden=d_hidden, dropout=dropout,
                front_ver=front_ver, backbone=KIND)

    MODEL_CLS = _Plug
    return MODEL_CLS


def _fit(use_front, pack, ckpt, res, tag, extra_evals=None, seed=None, cm=None):
    kw = dict(base.V2_HPARAM) if use_front else {}
    return base.fit_model(
        use_front=use_front, front_ver='v2' if use_front else 'v1',
        train_ds=pack['train_ds'], val_ds=pack['val_ds'], test_ds=pack['test_ds'],
        n_cls=pack['n_cls'], label_encoder=pack['label_encoder'],
        sample_weight=pack['sample_weight'], class_weight=pack['class_weight'],
        ckpt_path=ckpt, result_path=res, cm_path=cm, tag=tag,
        extra_evals=extra_evals, model_cls=MODEL_CLS, seed=seed, **kw,
    )


def _kv_float(text: str, key: str):
    for line in text.splitlines():
        if line.startswith(key + '='):
            try:
                return float(line.split('=', 1)[1].strip())
            except ValueError:
                return None
    return None


def _read_metrics_from_txt(res_path: str):
    pass
    if not os.path.isfile(res_path):
        return None
    text = open(res_path, encoding='utf-8', errors='ignore').read()
    best = _kv_float(text, 'best_epoch')
    val_f1 = _kv_float(text, 'val_macro_f1')
    drop_f1 = _kv_float(text, 'test_macro_f1')
    full_f1 = _kv_float(text, 'test_full_macro_f1')
    full_acc = _kv_float(text, 'test_full_accuracy')
    if None in (best, val_f1, drop_f1, full_f1, full_acc):
        return None
    return {
        'best_epoch': int(best),
        'val_drop_f1': val_f1,
        'test_drop_f1': drop_f1,
        'test_full_f1': full_f1,
        'test_full_acc': full_acc,
    }


def run_lodo(pack, seed=TRAIN_SEED, resume=True):
    os.makedirs(LODO_DIR, exist_ok=True)
    train_ds, val_ds, test_ds = pack['train_ds'], pack['val_ds'], pack['test_ds']
    dims = list(train_ds.device_dims)
    names = list(base.DEVICE_PREFIXES)
    full_loader = base.eval_loader(test_ds)
    rows = []
    for p, pname in enumerate(names[:len(dims)]):
        tr = base.ArrayWinDS(base.zero_devices(train_ds.samples, dims, [p]), train_ds.labels, dims)
        va = base.ArrayWinDS(base.zero_devices(val_ds.samples, dims, [p]), val_ds.labels, dims)
        te = base.ArrayWinDS(base.zero_devices(test_ds.samples, dims, [p]), test_ds.labels, dims)
        sub = dict(pack)
        sub.update(train_ds=tr, val_ds=va, test_ds=te)
        for use_front, stem, name in (
                (True, 'front', FRONT_NAME),
                (False, 'raw', RAW_NAME)):
            ckpt = os.path.join(LODO_DIR, f'best_{KIND}_lodo_{pname}_{stem}.pth')
            res = os.path.join(LODO_DIR, f'{KIND}_lodo_{pname}_{stem}.txt')
            cached = _read_metrics_from_txt(res) if resume else None
            if resume and os.path.isfile(ckpt) and cached and os.path.getsize(ckpt) > 1000:
                print(f'# SKIP LODO drop={pname}  {name}  (resume '
                      f'full_f1={cached["test_full_f1"]:.4f})')
                rows.append({
                    'drop_device': pname, 'model': name, 'ckpt': ckpt, **cached,
                })
                continue
            print('\n' + '#' * 64)
            print(f'# LODO drop={pname}  {name}')
            print('#' * 64)
            out = _fit(
                use_front, sub, ckpt, res, f'LODO-{pname}-{name}',
                extra_evals={'test_full': full_loader}, seed=seed)
            tf = out['extras']['test_full']
            rows.append({
                'drop_device': pname,
                'model': name,
                'best_epoch': out['best_epoch'],
                'val_drop_f1': out['val']['macro_f1'],
                'test_drop_f1': out['test']['macro_f1'],
                'test_full_f1': tf['macro_f1'],
                'test_full_acc': tf['accuracy'],
                'ckpt': ckpt,
            })
    return rows


def run_full(pack, seed=TRAIN_SEED):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for use_front, stem, name in (
            (True, 'front', FRONT_NAME),
            (False, 'raw', RAW_NAME)):
        print('\n' + '#' * 64)
        print(f'# FULL  {name}')
        print('#' * 64)
        ckpt = os.path.join(OUT_DIR, f'best_{KIND}_{stem}.pth')
        res = os.path.join(OUT_DIR, f'{KIND}_{stem}.txt')
        out = _fit(use_front, pack, ckpt, res, name, seed=seed)
        rows.append({
            'model': name,
            'best_epoch': out['best_epoch'],
            'test_f1': out['test']['macro_f1'],
            'test_acc': out['test']['accuracy'],
            'ckpt': ckpt,
        })
    return rows


def write_lodo_summary(rows):
    os.makedirs(LODO_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    summary_csv = os.path.join(LODO_DIR, f'{KIND}_lodo_summary.csv')
    summary_txt = os.path.join(LODO_DIR, f'{KIND}_lodo_results.txt')
    df.to_csv(summary_csv, index=False, float_format='%.6f')

    a_name, r_name = FRONT_NAME, RAW_NAME
    wide = df.pivot(index='drop_device', columns='model', values='test_full_f1')
    means = df.groupby('model')[['test_full_f1', 'test_full_acc']].mean()
    stds = df.groupby('model')[['test_full_f1', 'test_full_acc']].std()
    gap = None
    if a_name in means.index and r_name in means.index:
        gap = float(means.loc[a_name, 'test_full_f1'] - means.loc[r_name, 'test_full_f1'])

    lines = [
        f'setting=LODO train/val drop 1 device, test FULL, Front-v2 vs Raw, mean over 7',
        f'kind={KIND} TAG={TAG}  adapted from PDF core (not full paper system)',
        f'V2_HPARAM={base.V2_HPARAM}',
        f'ckpt_dir={LODO_DIR}',
        '',
        df.drop(columns=['ckpt']).to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '',
        '===== 每路 Test-FULL Macro-F1 =====',
        wide.to_string(float_format=lambda x: f'{x:.4f}'),
        '',
        '===== 7 路平均 =====',
    ]
    for m in (a_name, r_name):
        if m in means.index:
            lines.append(
                f'{m:28s}  Test F1 {means.loc[m, "test_full_f1"]:.4f} +/- {stds.loc[m, "test_full_f1"]:.4f}'
                f'  Acc {means.loc[m, "test_full_acc"]:.4f}')
    if gap is not None:
        lines.append(f'gap (Front-v2 - Raw)  {gap:+.4f}')
    text = '\n'.join(lines)
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nsaved {summary_csv}\n{summary_txt}')
    return gap


def main_one(kind: str):
    configure(kind)
    os.makedirs(LODO_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    print(f'\n======== PAPER ADAPT LODO  kind={KIND} TAG={TAG} ========')
    print(f'device={base.DEVICE}  V2={base.V2_HPARAM}')
    print(f'lodo={LODO_DIR}')

    x = torch.randn(2, 8, 17)
    m = MODEL_CLS([3, 2, 2, 2, 2, 3, 3], 8, use_front=True)
    y, aux = m(x, return_aux=True)
    print('smoke Front', tuple(y.shape), 'gate', float(aux['gate'].mean()), 'params', n_params(m))
    m2 = MODEL_CLS([3, 2, 2, 2, 2, 3, 3], 8, use_front=False)
    print('smoke Raw  ', tuple(m2(x).shape), 'params', n_params(m2))

    pack = base.load_datasets()
    if DO_FULL:
        os.makedirs(OUT_DIR, exist_ok=True)
        full_rows = run_full(pack, TRAIN_SEED)
        pd.DataFrame(full_rows).to_csv(
            os.path.join(OUT_DIR, f'{KIND}_full_summary.csv'), index=False, float_format='%.6f')

    rows = run_lodo(pack, TRAIN_SEED)
    gap = write_lodo_summary(rows)
    return gap


def main(kinds=None):
    global DO_FULL
    args = list(sys.argv[1:])
    if '--full' in args:
        DO_FULL = True
        args.remove('--full')
    if not kinds:
        kinds = args if args else list(DEFAULT_KINDS)
    for k in kinds:
        if k not in PAPER_BACKBONES:
            raise ValueError(f'unknown kind={k}')
    from paper_backbones import smoke_all
    print('=== backbone smoke ===')
    smoke_all()
    print(f'queue={kinds}')

    gaps = {}
    for k in PAPER_BACKBONES:
        p = os.path.join(_ROOT, f'paper_{k}_lodo_ckpts', f'{k}_lodo_results.txt')
        if os.path.isfile(p) and k not in kinds:
            for line in open(p, encoding='utf-8', errors='ignore'):
                if line.startswith('gap (Front-v2 - Raw)'):
                    try:
                        gaps[k] = float(line.strip().split()[-1])
                    except ValueError:
                        gaps[k] = None
                    break

    for k in kinds:
        gaps[k] = main_one(k)

    summary_path = os.path.join(_ROOT, 'paper_adapt_lodo_all.txt')
    lines = [
        'PDF-adapted backbones LODO gap (Front-v2 - Raw) on second ToN-IoT',
        'RAPIER=BiGRU core; TIPSO=decay-MHSA (no PSO-GAN).',
        '',
    ]
    for k in DEFAULT_KINDS:
        g = gaps.get(k)
        tag = PAPER_BACKBONES[k][0]
        if g is None:
            lines.append(f'{k:12s} ({tag:16s})  gap=NA/pending')
        else:
            lines.append(f'{k:12s} ({tag:16s})  gap={g:+.4f}')
    text = '\n'.join(lines)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'saved {summary_path}')


if __name__ == '__main__':
    main()
