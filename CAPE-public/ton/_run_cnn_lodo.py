                       
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

OUT_DIR = os.path.join(base._ROOT, 'cnn_lodo_ckpts')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'device={base.DEVICE}  LODO drop-1 / test-full  Front-v2 vs Raw')
    print(f'V2_HPARAM={base.V2_HPARAM}')
    print(f'ckpt_dir={OUT_DIR}')
    pack = base.load_datasets()
    train_ds, val_ds, test_ds = pack['train_ds'], pack['val_ds'], pack['test_ds']
    dims = list(train_ds.device_dims)
    names = list(base.DEVICE_PREFIXES)
    full_test_loader = base.eval_loader(test_ds)
    rows = []
    a_name, r_name = 'Front-v2+CNN', 'Raw+CNN'

    for p, pname in enumerate(names[:len(dims)]):
        print('\n' + '#' * 64)
        print(f'# LODO {p + 1}/{len(dims)}  train/val zero [{pname}]   test = FULL')
        print('#' * 64)
        tr_drop = base.ArrayWinDS(
            base.zero_devices(train_ds.samples, dims, [p]), train_ds.labels, dims)
        va_drop = base.ArrayWinDS(
            base.zero_devices(val_ds.samples, dims, [p]), val_ds.labels, dims)
        te_drop = base.ArrayWinDS(
            base.zero_devices(test_ds.samples, dims, [p]), test_ds.labels, dims)

        for use_front, stem in ((True, 'front'), (False, 'raw')):
            tag = f'LODO-{pname}-{a_name if use_front else r_name}'
            ckpt = os.path.join(OUT_DIR, f'best_cnn_lodo_{pname}_{stem}.pth')
            res = os.path.join(OUT_DIR, f'cnn_lodo_{pname}_{stem}.txt')
            kw = dict(base.V2_HPARAM) if use_front else {}
            out = base.fit_model(
                use_front=use_front,
                front_ver='v2' if use_front else 'v1',
                train_ds=tr_drop, val_ds=va_drop, test_ds=te_drop,
                n_cls=pack['n_cls'], label_encoder=pack['label_encoder'],
                sample_weight=pack['sample_weight'], class_weight=pack['class_weight'],
                ckpt_path=ckpt, result_path=res, cm_path=None, tag=tag,
                extra_evals={'test_full': full_test_loader},
                **kw,
            )
            tf = out['extras']['test_full']
            rows.append({
                'drop_device': pname,
                'model': a_name if use_front else r_name,
                'best_epoch': out['best_epoch'],
                'val_drop_f1': out['val']['macro_f1'],
                'test_drop_f1': out['test']['macro_f1'],
                'test_full_f1': tf['macro_f1'],
                'test_full_acc': tf['accuracy'],
                'ckpt': ckpt,
            })

    df = pd.DataFrame(rows)
    summary_csv = os.path.join(OUT_DIR, 'cnn_lodo_summary.csv')
    summary_txt = os.path.join(OUT_DIR, 'cnn_lodo_results.txt')
    df.to_csv(summary_csv, index=False, float_format='%.6f')

    wide = df.pivot(index='drop_device', columns='model', values='test_full_f1')
    means = df.groupby('model')[['test_full_f1', 'test_full_acc']].mean()
    stds = df.groupby('model')[['test_full_f1', 'test_full_acc']].std()
    gap = None
    if a_name in means.index and r_name in means.index:
        gap = float(means.loc[a_name, 'test_full_f1'] - means.loc[r_name, 'test_full_f1'])

    lines = [
        'setting=LODO train/val drop 1 device, test FULL, Front-v2 vs Raw, mean over 7',
        f'V2_HPARAM={base.V2_HPARAM}',
        f'ckpt_dir={OUT_DIR}',
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
                f'{m:14s}  Test F1 {means.loc[m, "test_full_f1"]:.4f} +/- {stds.loc[m, "test_full_f1"]:.4f}'
                f'  Acc {means.loc[m, "test_full_acc"]:.4f}')
    if gap is not None:
        lines.append(f'gap (Front-v2 - Raw)  {gap:+.4f}')
    text = '\n'.join(lines)
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print('\n' + text)
    print(f'\nsaved {summary_csv}\n{summary_txt}')
    print(f'best models in {OUT_DIR}')
    return df


if __name__ == '__main__':
    main()
