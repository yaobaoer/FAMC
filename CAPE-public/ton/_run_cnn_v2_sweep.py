                       
pass
from __future__ import annotations

import os
import sys

import pandas as pd

_ROOT = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
if not os.path.isfile(os.path.join(_ROOT, 'iot_fusion_weather_all.csv')):
    _ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import _run_cnn as base

OUT_DIR = os.path.join(base._ROOT, 'cnn_v2_sweep')

DEFAULT = dict(lam_contra=0.1, lam_domain=0.05, contra_temp=0.1, gate_target=0.12, lam_gate=1.0)

                                                
AXES = (
    ('lam_contra', (0.0, 0.1, 0.3, 0.5)),
    ('lam_domain', (0.0, 0.05, 0.2)),
    ('contra_temp', (0.05, 0.1, 0.2)),
    ('gate_target', (0.12, 0.3, 0.5)),
)


def _tag(cfg):
    def _p(x):
        s = f'{float(x):g}'.replace('.', 'p')
        return s.replace('-', 'm')
    return (f"c{_p(cfg['lam_contra'])}_d{_p(cfg['lam_domain'])}"
            f"_t{_p(cfg['contra_temp'])}_g{_p(cfg['gate_target'])}")


def build_grid():
    seen, rows = set(), []
    for axis, values in AXES:
        for v in values:
            cfg = dict(DEFAULT)
            cfg[axis] = float(v)
            key = tuple(round(cfg[k], 6) for k in
                        ('lam_contra', 'lam_domain', 'contra_temp', 'gate_target'))
            if key in seen:
                continue
            seen.add(key)
            rows.append((axis if v != DEFAULT[axis] else 'default', cfg))
    return rows


def run_one(pack, axis, cfg):
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = _tag(cfg)
    ckpt = os.path.join(OUT_DIR, f'best_{tag}.pth')
    res = os.path.join(OUT_DIR, f'{tag}.txt')
    print('\n' + '#' * 64)
    print(f'# SWEEP axis={axis}  {_tag(cfg)}')
    print(f'# λc={cfg["lam_contra"]} λd={cfg["lam_domain"]} '
          f'τ={cfg["contra_temp"]} g0={cfg["gate_target"]}')
    print('#' * 64)
    out = base.fit_model(
        use_front=True, front_ver='v2',
        train_ds=pack['train_ds'], val_ds=pack['val_ds'], test_ds=pack['test_ds'],
        n_cls=pack['n_cls'], label_encoder=pack['label_encoder'],
        sample_weight=pack['sample_weight'], class_weight=pack['class_weight'],
        ckpt_path=ckpt, result_path=res, cm_path=None,
        tag=f'v2-sweep-{tag}',
        lam_contra=cfg['lam_contra'],
        lam_domain=cfg['lam_domain'],
        lam_gate=cfg['lam_gate'],
        contra_temp=cfg['contra_temp'],
        gate_target=cfg['gate_target'],
    )
    return {
        'axis': axis,
        'run': tag,
        'lam_contra': cfg['lam_contra'],
        'lam_domain': cfg['lam_domain'],
        'contra_temp': cfg['contra_temp'],
        'gate_target': cfg['gate_target'],
        'best_epoch': out['best_epoch'],
        'val_f1': out['val']['macro_f1'],
        'test_f1': out['test']['macro_f1'],
        'test_acc': out['test']['accuracy'],
        'gate_mean': out['gamma'],
        'ckpt': ckpt,
    }


def write_summary(rows):
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'sweep_summary.csv')
    txt_path = os.path.join(OUT_DIR, 'sweep_results.txt')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    lines = [
        'setting=v2 FULL-channel one-at-a-time sweep',
        'default: λc=0.1 λd=0.05 τ=0.1 g0=0.12 λg=1.0',
        f'ckpt_dir={OUT_DIR}',
        '',
        df.drop(columns=['ckpt']).to_string(index=False, float_format=lambda x: f'{x:.4f}'),
        '',
        '===== per-axis (test Macro-F1 / gate) =====',
    ]
    show = df.drop(columns=['ckpt'])
    for axis, _ in AXES:
        sub = show[(show['axis'] == axis) | (show['axis'] == 'default')].copy()
        col = axis
        sub = sub.sort_values(col)
        lines.append(f'\n[{axis}]')
        for _, r in sub.iterrows():
            lines.append(
                f"  {col}={r[col]:.4f}  test={r['test_f1']:.4f}  val={r['val_f1']:.4f}  "
                f"acc={r['test_acc']:.4f}  gate={r['gate_mean']:.3f}  ep={int(r['best_epoch'])}"
            )
    text = '\n'.join(lines)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nsaved {csv_path}\n{txt_path}')
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    grid = build_grid()
    print(f'device={base.DEVICE}  v2 FULL sweep  n={len(grid)}')
    print(f'ckpt_dir={OUT_DIR}')
    for axis, cfg in grid:
        print(f'  - {axis:12s} {_tag(cfg)}')
    pack = base.load_datasets()
    rows = []
    for i, (axis, cfg) in enumerate(grid, 1):
        print(f'\n>>>>> {i}/{len(grid)}')
        rows.append(run_one(pack, axis, cfg))
        write_summary(rows)
    print('\n===== DONE v2 sweep =====')
    return write_summary(rows)


if __name__ == '__main__':
    main()
