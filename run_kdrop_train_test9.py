                       
from __future__ import annotations

import argparse
import gc
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

import run_transformer_align as common


ROOT = os.path.dirname(os.path.abspath(__file__))
RANK_SRC = os.path.join(ROOT, 'lodo_train8_test9_five_backbones_per_device.csv')
RANK_OUT = os.path.join(ROOT, 'kdrop_advantage_device_rank.csv')
CKPT_ROOT = os.path.join(ROOT, 'kdrop_adv_ckpts')
RUNS_CSV = os.path.join(ROOT, 'kdrop_train_test9_fulltest_runs.csv')
SUMMARY_CSV = os.path.join(ROOT, 'kdrop_train_test9_fulltest_summary.csv')
SUMMARY_MD = os.path.join(ROOT, 'kdrop_train_test9_fulltest_summary.md')

MODELS = (
    ('CNN', 'run_cnn_align', 'PlugCNN'),
    ('MLP', 'run_mlp_align', 'PlugMLP'),
    ('TCN', 'run_tcn_align', 'PlugTCN'),
    ('LSTM', 'run_lstm_align', 'PlugLSTM'),
    ('Transformer', 'run_transformer_align', 'PlugTransformer'),
)
SHARED_ATTRS = (
    'WINDOW', 'TRAIN_STEP', 'EVAL_STEP', 'MAX_WINDOWS', 'MAX_ROWS_PER_CLASS',
    'BATCH', 'EPOCHS', 'SEED', 'DEVICE_NAMES',
)


def advantage_order(path, device_names, backbones):
    pass
    df = pd.read_csv(path)
    need = {'backbone', 'drop_device', 'model', 'f1'}
    missing_cols = need.difference(df.columns)
    if missing_cols:
        raise ValueError(f'{path} 缺少列 {sorted(missing_cols)}')
    wide = df.pivot_table(index=['backbone', 'drop_device'], columns='model', values='f1')
    if not {'front', 'raw'}.issubset(wide.columns):
        raise ValueError(f'{path} 需要 front 与 raw 两列')
    gap = (wide['front'] - wide['raw']).rename('gap')
    observed = set(zip(gap.index.get_level_values(0), gap.index.get_level_values(1)))
    expect = {(b, d) for b in backbones for d in device_names}
    if observed != expect:
        raise ValueError(f'设备差距表不完整，缺 {sorted(expect - observed)[:8]}')
    mean_gap = gap.groupby('drop_device').mean()
    order = sorted(device_names, key=lambda name: (-float(mean_gap.loc[name]), device_names.index(name)))
    return order, mean_gap


def drop_plan(order, ks):
    plan = {}
    for k in ks:
        if not 1 <= int(k) < len(order):
            raise ValueError(f'k={k} 超出设备数 {len(order)}')
        plan[int(k)] = tuple(order[:int(k)])
    return plan


def entirely_zero_devices(samples, dims):
    ids, start = [], 0
    for i, width in enumerate(dims):
        if np.all(samples[:, :, start:start + width] == 0):
            ids.append(i)
        start += width
    if start != samples.shape[-1]:
        raise ValueError(f'特征宽度 {samples.shape[-1]} 与 dims 之和 {start} 不一致')
    return ids


def assert_train_val_dropped(train_ds, val_ds, test_ds, dims, ids):
    wanted = sorted(int(i) for i in ids)
    for name, ds in (('train', train_ds), ('val', val_ds)):
        got = entirely_zero_devices(ds.samples, dims)
        if got != wanted:
            raise RuntimeError(f'{name} 置零设备 {got}，应为 {wanted}')
    leaked = set(entirely_zero_devices(test_ds.samples, dims)) & set(wanted)
    if leaked:
        raise RuntimeError(f'测试集被清零的设备 {sorted(leaked)}')


def load_runs(path, overwrite):
    if overwrite or not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def finished(runs, backbone, k, stem, n_records):
    if runs.empty:
        return False
    hit = runs[(runs.backbone == backbone) & (runs.k == k) & (runs.model == stem)]
    return len(hit) >= n_records


def replace_rows(runs, rows):
    add = pd.DataFrame(rows)
    if runs.empty:
        return add
    key = ['backbone', 'k', 'model']
    keep = runs.merge(add[key].drop_duplicates(), on=key, how='left', indicator=True)
    keep = runs.loc[keep['_merge'].eq('left_only')].copy()
    return pd.concat([keep, add], ignore_index=True)


def write_runs(runs, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    runs.to_csv(path, index=False)


def write_summary(runs, rank, plan, md_path, csv_path):
    counts = runs.groupby(['backbone', 'k', 'model']).seed.nunique()
    if counts.min() < 1 or not bool((counts == 1).all()):
        raise RuntimeError('每个骨干、k、Front/Raw 都应有一条完整测试记录')
    seed_means = runs.groupby(['backbone', 'k', 'model', 'seed'])[['acc', 'f1']].mean()
    rows = []
    for backbone, _, _ in MODELS:
        if backbone not in set(runs.backbone):
            continue
        for k, devices in plan.items():
            if k not in set(runs.loc[runs.backbone == backbone, 'k'].astype(int)):
                continue
            front = seed_means.loc[(backbone, k, 'front')]
            raw = seed_means.loc[(backbone, k, 'raw')]
            gap = front - raw
            rec = dict(
                backbone=backbone, k=k, drop_devices='+'.join(devices),
                front_acc=float(front.acc.mean()), front_f1=float(front.f1.mean()),
                raw_acc=float(raw.acc.mean()), raw_f1=float(raw.f1.mean()),
                gap_f1=float(gap.f1.mean()),
            )
            rows.append(rec)
    out = pd.DataFrame(rows)
    for field in ('front_acc', 'front_f1', 'raw_acc', 'raw_f1', 'gap_f1'):
        out[field + '_text'] = [
            f'{100 * val:.2f}' for val in out[field]
        ]
    out.to_csv(csv_path, index=False)
    lines = [
        '# Drop 2/3/4 devices: train and validation missing, test full',
        '',
        '丢弃顺序按训 8 测 9 的五个骨干平均 Front−Raw Macro-F1 差值，从大到小。',
        '同一 k 的五个骨干丢掉同一组设备。训练和验证清零这些设备，测试使用原始九路。',
        'Acc / Macro-F1 为百分比，差值为百分点。',
        '',
        '优势从大到小：' + ' > '.join(
            f'{name} ({100 * float(rank.loc[name]):.2f} pp)' for name in rank.index),
        '',
        '| Backbone | k | Dropped | Front Acc | Front Macro-F1 | Raw Acc | Raw Macro-F1 | F1 gap (pp) |',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for rec in out.itertuples(index=False):
        lines.append(
            f'| {rec.backbone} | {rec.k} | {rec.drop_devices} | {rec.front_acc_text} '
            f'| {rec.front_f1_text} | {rec.raw_acc_text} | {rec.raw_f1_text} | {rec.gap_f1_text} |'
        )
    lines.append('')
    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    return out


def load_pack():
    pass
    df = pd.read_csv(common.DATA_CSV, low_memory=False)
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['final_type'].astype(str))
    feats = common.public_feat_cols(df)
    width = len(feats)
    dims = [width] * len(common.DEVICE_NAMES)
    print(f'PUBLIC feats={width} (dropped HH_jit) devices={len(common.DEVICE_NAMES)} '
          f'classes={list(le.classes_)}', flush=True)
    tr_df, va_df, te_df = common.stratified_split(df, common.SEED)
    del df
    gc.collect()
    Xtr, ytr = common.build_aligned(tr_df, feats, seed=common.SEED)
    Xva, yva = common.build_aligned(va_df, feats, seed=common.SEED + 1)
    Xte, yte = common.build_aligned(te_df, feats, seed=common.SEED + 2)
    del tr_df, va_df, te_df
    gc.collect()
    scaler = StandardScaler()
    Xtr = np.nan_to_num(np.clip(scaler.fit_transform(Xtr), -8, 8), nan=0.0).astype(np.float32)
    Xva = np.nan_to_num(np.clip(scaler.transform(Xva), -8, 8), nan=0.0).astype(np.float32)
    Xte = np.nan_to_num(np.clip(scaler.transform(Xte), -8, 8), nan=0.0).astype(np.float32)
    print(f'aligned {len(ytr)}/{len(yva)}/{len(yte)} dim={Xtr.shape[1]}=9×{width}', flush=True)
    test = common.ArrayWinDS(
        Xte, yte, dims, common.WINDOW, common.EVAL_STEP, common.MAX_WINDOWS // 2, common.SEED + 2)
    test.X_table, test.y_table = Xte, yte
    train_for_weights = common.ArrayWinDS(
        Xtr, ytr, dims, common.WINDOW, common.TRAIN_STEP, common.MAX_WINDOWS, common.SEED)
    n_cls = len(le.classes_)
    counts = np.maximum(np.bincount(train_for_weights.labels, minlength=n_cls).astype(np.float64), 1)
    cw = torch.tensor(np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0),
                      dtype=torch.float32, device=common.DEVICE)
    del train_for_weights
    gc.collect()
    train = SimpleNamespace(X_table=Xtr, y_table=ytr)
    val = SimpleNamespace(X_table=Xva, y_table=yva)
    return dict(train=train, val=val, test=test, dims=dims, n_cls=n_cls, le=le, cw=cw, d_feat=width)


def parse_args(argv):
    parser = argparse.ArgumentParser(description='按 Front 最大优势丢 2/3/4 路，测试保持全量')
    parser.add_argument('--ks', default='2,3,4')
    parser.add_argument('--models', default=','.join(name for name, _, _ in MODELS))
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def selected_models(names):
    wanted = [item.strip().lower() for item in names.split(',') if item.strip()]
    chosen = [item for item in MODELS if item[0].lower() in wanted]
    known = {item[0].lower() for item in MODELS}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise ValueError(f'未知骨干 {unknown}')
    if not chosen:
        raise ValueError('至少指定一个骨干')
    return chosen


def checkpoint_ready(path):
    pass
    if not os.path.isfile(path):
        return False
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return isinstance(blob, dict) and 'model_state' in blob and 'drop_devices' in blob


def eval_checkpoint(exp, class_name, pack, ckpt, use_front):
    blob = torch.load(ckpt, map_location=exp.DEVICE, weights_only=False)
    model = getattr(exp, class_name)(pack['dims'], pack['n_cls'], use_front=use_front).to(exp.DEVICE)
    model.load_state_dict(blob['model_state'])
    return model, int(blob['best_epoch'])


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    ks = tuple(int(part) for part in args.ks.split(',') if part.strip())
    models = selected_models(args.models)
    order, mean_gap = advantage_order(
        RANK_SRC, tuple(common.DEVICE_NAMES), tuple(name for name, _, _ in MODELS))
    plan = drop_plan(order, ks)
    rank = mean_gap.loc[order]
    rank_out = pd.DataFrame({
        'rank': np.arange(1, len(order) + 1),
        'device': order,
        'mean_gap_f1': [float(rank.loc[name]) for name in order],
    })
    for k, devices in plan.items():
        rank_out[f'drop_when_k{k}'] = rank_out.device.isin(devices)
    os.makedirs(os.path.dirname(RANK_OUT) or '.', exist_ok=True)
    rank_out.to_csv(RANK_OUT, index=False)
    print('advantage order:', ' > '.join(order), flush=True)
    for k, devices in plan.items():
        print(f'k={k} drop {"+".join(devices)}; train/val={9 - k}; test=9', flush=True)

    pack = load_pack()
    runs = load_runs(RUNS_CSV, args.overwrite)
    for backbone, module_name, class_name in models:
        exp = importlib.import_module(module_name)
        for attr in SHARED_ATTRS:
            if getattr(exp, attr) != getattr(common, attr):
                raise RuntimeError(f'{backbone} 的 {attr} 与 Transformer 不一致')
        os.makedirs(os.path.join(CKPT_ROOT, backbone), exist_ok=True)
        for k, devices in plan.items():
            ids = [exp.DEVICE_NAMES.index(name) for name in devices]
            print(f'\n########## {backbone} k={k} drop={"+".join(devices)} TEST=FULL ##########',
                  flush=True)
            train_ds = exp.ArrayWinDS(
                exp.zero_devices(pack['train'].X_table, pack['dims'], ids),
                pack['train'].y_table, pack['dims'], exp.WINDOW, exp.TRAIN_STEP,
                exp.MAX_WINDOWS, exp.SEED)
            val_ds = exp.ArrayWinDS(
                exp.zero_devices(pack['val'].X_table, pack['dims'], ids),
                pack['val'].y_table, pack['dims'], exp.WINDOW, exp.EVAL_STEP,
                exp.MAX_WINDOWS // 2, exp.SEED + 1)
            test_ds = pack['test']
            assert_train_val_dropped(train_ds, val_ds, test_ds, pack['dims'], ids)
            for use_front, stem, title in (
                    (True, 'front', 'Front-v2'), (False, 'raw', 'Raw')):
                if finished(runs, backbone, k, stem, 1) and not args.overwrite:
                    print(f'skip {backbone} k={k} {stem}', flush=True)
                    continue
                ckpt = os.path.join(CKPT_ROOT, backbone, f'best_k{k}_{stem}.pth')
                tag = f'k{k}-{title}+{backbone}'
                if checkpoint_ready(ckpt) and not args.overwrite:
                    model, best_ep = eval_checkpoint(exp, class_name, pack, ckpt, use_front)
                else:
                    model, _, best_ep = exp.fit(
                        use_front, train_ds, val_ds, test_ds, pack['n_cls'], pack['cw'],
                        ckpt, tag, exp.SEED)
                    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
                    blob.update(
                        k=k, drop_devices=list(devices), backbone=backbone,
                        model=stem, class_name=class_name, seed=int(exp.SEED),
                        protocol=f'train{9 - k}_val{9 - k}_test9')
                    torch.save(blob, ckpt)
                    print(f'  >> weights {ckpt}', flush=True)
                rows = []
                for metric in exp.evaluate_full(model, test_ds):
                    rows.append(dict(
                        backbone=backbone, model=stem, k=k,
                        drop_devices='+'.join(devices), seed=metric['seed'],
                        protocol=f'train{9 - k}_val{9 - k}_test9',
                        best_epoch=best_ep, checkpoint=ckpt,
                        acc=metric['acc'], precision=metric['precision'],
                        recall=metric['recall'], f1=metric['f1']))
                runs = replace_rows(runs, rows)
                write_runs(runs, RUNS_CSV)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del train_ds, val_ds
            gc.collect()
    if not runs.empty:
        write_summary(runs, rank, plan, SUMMARY_MD, SUMMARY_CSV)
    print('KDROP_TRAIN_TEST9_DONE', flush=True)


if __name__ == '__main__':
    main()
