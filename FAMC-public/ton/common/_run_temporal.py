                       
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE if os.path.isfile(os.path.join(_HERE, 'iot_fusion_weather_all.csv')) else os.path.dirname(os.path.abspath(__file__))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import _run_cnn as base
from activity_gated_gat import ActivityGatedGAT
from activity_gated_gat_v2 import ActivityGatedGAT as ActivityGatedGATv2
from activity_gated_gat_v2 import n_params

base._ROOT = _ROOT
base.DATA_CSV = os.path.join(os.path.dirname(_ROOT), 'iot_fusion_weather_all.csv')
os.chdir(_ROOT)

TRAIN_SEED = 42
KIND = 'tcn'
TAG = 'TCN'
FRONT_NAME = 'Front-v2+TCN'
RAW_NAME = 'Raw+TCN'
OUT_DIR = os.path.join(_ROOT, 'tcn_ckpts')
LODO_DIR = os.path.join(_ROOT, 'tcn_lodo_ckpts')
MODEL_CLS = None


class TCNBlock(nn.Module):
    pass

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
    def __init__(self, d_in, d_hidden=32, dropout=0.1):
        super().__init__()
        self.b1 = TCNBlock(d_in, d_hidden, kernel=3, dilation=1, dropout=dropout)
        self.b2 = TCNBlock(d_hidden, d_hidden, kernel=3, dilation=2, dropout=dropout)
        self.out_dim = d_hidden

    def forward(self, x):
        return self.b2(self.b1(x))


class LSTMBackbone(nn.Module):
    def __init__(self, d_in, d_hidden=32, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.n = nn.LayerNorm(d_hidden)
        self.out_dim = d_hidden

    def forward(self, x):
        y, _ = self.lstm(x)
        return self.n(self.drop(y))


class TransformerBackbone(nn.Module):
    pass

    def __init__(self, d_in, d_hidden=32, dropout=0.1, nhead=4):
        super().__init__()
        self.proj = nn.Identity() if d_in == d_hidden else nn.Linear(d_in, d_hidden)
        self.pos = nn.Parameter(torch.zeros(1, 256, d_hidden))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_hidden, nhead=nhead, dim_feedforward=d_hidden * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.out_dim = d_hidden

    def forward(self, x):
        h = self.proj(x)
        t = h.size(1)
        h = h + self.pos[:, :t]
        return self.enc(h)


class MLPBackbone(nn.Module):
    pass

    def __init__(self, d_in, d_hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
        )
        self.n = nn.LayerNorm(d_hidden)
        self.out_dim = d_hidden

    def forward(self, x):
        return self.n(self.net(x))


class PlugTemporal(nn.Module):
    pass

    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2',
                 backbone='tcn'):
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
        factories = {
            'lstm': LSTMBackbone,
            'transformer': TransformerBackbone,
            'mlp': MLPBackbone,
            'tcn': TCNBackbone,
        }
        if kind not in factories:
            raise ValueError(f'unknown backbone {kind}')
        self.backbone = factories[kind](d_in, d_hidden=d_hidden, dropout=dropout)
        self.head = base.DualPoolHead(self.backbone.out_dim, num_classes)
        self.needs_aux = self.front_ver == 'v2'

    def forward(self, x, return_aux=False):
        aux = None
        if self.front is None:
            seq = x
        elif self.front_ver == 'v2':
            seq, aux = self.front(x, return_features=True)
        else:
            seq = self.front(x)
        feat = self.backbone(seq)
        logits = self.head(feat)
        if return_aux:
            return logits, aux or {}
        return logits


class PlugTCN(PlugTemporal):
    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
        super().__init__(device_dims, num_classes, use_front=use_front,
                         d_model=d_model, d_hidden=d_hidden, dropout=dropout,
                         front_ver=front_ver, backbone='tcn')


class PlugLSTM(PlugTemporal):
    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
        super().__init__(device_dims, num_classes, use_front=use_front,
                         d_model=d_model, d_hidden=d_hidden, dropout=dropout,
                         front_ver=front_ver, backbone='lstm')


class PlugTransformer(PlugTemporal):
    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
        super().__init__(device_dims, num_classes, use_front=use_front,
                         d_model=d_model, d_hidden=d_hidden, dropout=dropout,
                         front_ver=front_ver, backbone='transformer')


class PlugMLP(PlugTemporal):
    def __init__(self, device_dims, num_classes, use_front=True,
                 d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
        super().__init__(device_dims, num_classes, use_front=use_front,
                         d_model=d_model, d_hidden=d_hidden, dropout=dropout,
                         front_ver=front_ver, backbone='mlp')


_KIND_TAG = {
    'tcn': 'TCN', 'lstm': 'LSTM', 'cnn': 'CNN',
    'transformer': 'Transformer', 'mlp': 'MLP',
}


def configure(kind='tcn'):
    global KIND, TAG, FRONT_NAME, RAW_NAME, OUT_DIR, LODO_DIR, MODEL_CLS
    KIND = str(kind).lower()
    if KIND not in _KIND_TAG:
        raise ValueError(f'unknown backbone {kind}')
    TAG = _KIND_TAG[KIND]
    FRONT_NAME = f'Front-v2+{TAG}'
    RAW_NAME = f'Raw+{TAG}'
    OUT_DIR = os.path.join(_ROOT, f'{KIND}_ckpts')
    LODO_DIR = os.path.join(_ROOT, f'{KIND}_lodo_ckpts')
    MODEL_CLS = {
        'lstm': PlugLSTM,
        'cnn': base.PlugCNN,
        'transformer': PlugTransformer,
        'mlp': PlugMLP,
        'tcn': PlugTCN,
    }[KIND]
    return MODEL_CLS


configure('tcn')


def _metrics(ev):
    return dict(
        acc=ev['accuracy'],
        precision=ev['macro_precision'],
        recall=ev['macro_recall'],
        f1=ev['macro_f1'],
    )


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














def run_full(pack, seed=TRAIN_SEED):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for use_front, stem, name in (
            (True, 'front', FRONT_NAME),
            (False, 'raw', RAW_NAME)):
        print('\n' + '#' * 64)
        print(f'# TRAIN-ONCE FULL  {name}  seed={seed}')
        print('#' * 64)
        ckpt = os.path.join(OUT_DIR, f'best_{KIND}_{stem}.pth')
        res = os.path.join(OUT_DIR, f'{KIND}_{stem}_train.txt')
        cm = os.path.join(OUT_DIR, f'cm_{KIND}_{stem}.png')
        out = _fit(use_front, pack, ckpt, res, f'{name}-train{seed}', seed=seed, cm=cm)
        m = _metrics(out['test'])
        rows.append(dict(setting='full-train', model=name, seed=seed,
                         best_epoch=out['best_epoch'], **m, ckpt=ckpt))
    return rows





def run_lodo(pack, seed=TRAIN_SEED):
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
            print('\n' + '#' * 64)
            print(f'# TRAIN-ONCE LODO drop={pname}  {name}')
            print('#' * 64)
            ckpt = os.path.join(LODO_DIR, f'best_{KIND}_lodo_{pname}_{stem}.pth')
            res = os.path.join(LODO_DIR, f'{KIND}_lodo_{pname}_{stem}_train.txt')
            out = _fit(
                use_front, sub, ckpt, res, f'LODO-{pname}-{name}-train',
                extra_evals={'test_full': full_loader}, seed=seed)
            m = _metrics(out['extras']['test_full'])
            md = _metrics(out['test'])
            rows.append(dict(
                setting='lodo-train', model=name, seed=seed, drop_device=pname,
                best_epoch=out['best_epoch'], test_drop_f1=md['f1'],
                acc=m['acc'], precision=m['precision'],
                recall=m['recall'], f1=m['f1'], ckpt=ckpt,
            ))
    return rows











def write_summary(full_rows, lodo_rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LODO_DIR, exist_ok=True)
    cols = ['acc', 'precision', 'recall', 'f1']
    lines = [f'{TAG} | Acc / Macro-P / Macro-R / Macro-F1']
    for label, rows, directory in [('full', full_rows, OUT_DIR), ('lodo', lodo_rows, LODO_DIR)]:
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        frame.to_csv(os.path.join(directory, f'{KIND}_{label}_fulltest_runs.csv'), index=False)
        lines += [f'===== {label.upper()} test =====', frame.to_string(index=False)]
        if label == 'lodo':
            lines += ['===== LODO mean over devices =====', frame.groupby('model')[cols].mean().to_string()]
    text = '\n'.join(lines) + '\n'
    with open(os.path.join(OUT_DIR, f'{KIND}_fulltest_results.txt'), 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(text)
    return text


def main(kind='tcn'):
    configure(kind)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LODO_DIR, exist_ok=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f'cwd={os.getcwd()}')
    print(f'data={base.DATA_CSV}')
    print(f'out={OUT_DIR}')
    print(f'lodo={LODO_DIR}')
    print(f'device={base.DEVICE}  Plug{TAG}  train_seed={TRAIN_SEED}')
    print(f'V2_HPARAM={base.V2_HPARAM}')
    x = torch.randn(2, 8, 17)
    m = MODEL_CLS([3, 4, 2, 2, 2, 2, 2], 8, use_front=True)
    y, aux = m(x, return_aux=True)
    print('smoke Front', tuple(y.shape), 'gate', float(aux['gate'].mean()), 'params', n_params(m))
    m2 = MODEL_CLS([3, 4, 2, 2, 2, 2, 2], 8, use_front=False)
    print('smoke Raw ', tuple(m2(x).shape), 'params', n_params(m2))

    pack = base.load_datasets()
    print(f'\n########## PHASE 1  train {FRONT_NAME} and {RAW_NAME} once ##########')
    full_official = run_full(pack, TRAIN_SEED)
    write_summary(full_official, [])
    lodo_official = run_lodo(pack, TRAIN_SEED)
    write_summary(full_official, lodo_official)
    print(f'{TAG}_DONE')


if __name__ == '__main__':
    kind = 'tcn'
    if len(sys.argv) > 1:
        kind = sys.argv[1]
    main(kind)
