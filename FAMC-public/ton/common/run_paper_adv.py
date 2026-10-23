                       

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import run_cnn_align as cnn
import run_kdrop_train_test9 as kdrop

_HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in (('..', 'second'), ('..', '..', 'second')):
    _SECOND = os.path.abspath(os.path.join(_HERE, *_rel))
    if os.path.isfile(os.path.join(_SECOND, 'paper_backbones.py')):
        if _SECOND not in sys.path:
            sys.path.insert(0, _SECOND)
        break
from paper_backbones import build_paper_backbone


ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.path.join(ROOT, 'paper_adv_ckpts')
RUNS_CSV = os.path.join(ROOT, 'paper_adv_fulltest_runs.csv')
SUMMARY_CSV = os.path.join(ROOT, 'paper_adv_fulltest_summary.csv')
SUMMARY_MD = os.path.join(ROOT, 'paper_adv_fulltest_summary.md')
KINDS = (
    ('cold', 'CoLD'),
    ('tfusion', 'tFusion'),
    ('tipsogan', 'TIPSO-Attn'),
    ('feco', 'FeCo'),
    ('aocids', 'AOC-IDS'),
)
D_HIDDEN = cnn.D_HIDDEN
RUN_D_MODEL = cnn.D_MODEL
RUN_LAM_DOMAIN = None
RUN_AXIS = False
RUN_GATE_TARGET = None
BASE_RUNS_CSV = os.path.join(ROOT, 'paper_adv_fulltest_runs.csv')


class TemporalResidual(nn.Module):
    pass

    def __init__(self, d, k=3, dropout=0.1):
        super().__init__()
        self.temp = nn.Conv1d(d, d, k, padding=k - 1, groups=d)
        self.drop = nn.Dropout(dropout)
        self.n = nn.LayerNorm(d)

    def forward(self, h):
        t = self.temp(h.transpose(1, 2))[:, :, :h.size(1)].transpose(1, 2)
        return self.n(h + self.drop(t))


class PlugPaper(nn.Module):
    def __init__(self, dims, n_cls, use_front=True, kind='cold'):
        super().__init__()
        self.front = None
        self.kind = kind
        if use_front:
            self.front = cnn.FrontV2(
                dims, d_model=RUN_D_MODEL, gat_heads=1, dropout=0.1,
                axis_residual=RUN_AXIS)
            self.front.use_gat = cnn.USE_GAT
            gate_target = cnn.V2_HPARAM['gate_target'] if RUN_GATE_TARGET is None else RUN_GATE_TARGET
            self.front.set_gate_target(gate_target)
            d_in = self.front.out_dim
        else:
            d_in = int(sum(dims))
        self.backbone = build_paper_backbone(kind, d_in, d_hidden=D_HIDDEN, dropout=0.1)
        self.temporal = TemporalResidual(self.backbone.out_dim)
        self.head = cnn.DualPool(self.backbone.out_dim, n_cls)

    def forward(self, x, return_aux=False):
        aux = None
        if self.front is None:
            seq = x
        else:
            seq, aux = self.front(x, return_features=True)
        logits = self.head(self.temporal(self.backbone(seq)))
        return (logits, aux) if return_aux else logits


SPREAD_NAMES = (
    'samsung', 'danmini', 'ecobee',
    'xcs1002', 'philips', 'pt737e',
    'xcs1003', 'pt838', 'ennio',
)


def permute_pack(pack, new_names):
    old = list(cnn.DEVICE_NAMES)
    perm = [old.index(name) for name in new_names]
    width = int(pack['dims'][0])

    def move_table(x):
        return np.concatenate([x[:, i * width:(i + 1) * width] for i in perm], axis=1)

    pack['train'].X_table = move_table(pack['train'].X_table)
    pack['val'].X_table = move_table(pack['val'].X_table)
    samples = pack['test'].samples
    pack['test'].samples = np.concatenate(
        [samples[:, :, i * width:(i + 1) * width] for i in perm], axis=2)
    pack['names'] = list(new_names)
    return pack


def settings(order):
    plan = [(1, (name,)) for name in cnn.DEVICE_NAMES]
    plan.extend((k, devices) for k, devices in kdrop.drop_plan(order, (2, 3, 4)).items())
    return plan


def checkpoint_ready(path):
    if not os.path.isfile(path):
        return False
    blob = torch.load(path, map_location='cpu', weights_only=False)
    return isinstance(blob, dict) and 'model_state' in blob and 'drop_devices' in blob


def fit(kind, use_front, train_ds, val_ds, test_ds, n_cls, cw, ckpt, tag, seed):
    cnn.set_seed(seed)
    counts = np.maximum(np.bincount(train_ds.labels, minlength=n_cls).astype(np.float64), 1)
    sw = 1.0 / np.sqrt(counts[train_ds.labels])
    sw /= sw.mean()
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(train_ds), True, generator=gen)
    tr = DataLoader(train_ds, cnn.BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    va = DataLoader(val_ds, cnn.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    model = PlugPaper(train_ds.device_dims, n_cls, use_front=use_front, kind=kind).to(cnn.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cnn.LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 3)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'\n===== {tag} front={use_front} gat={cnn.USE_GAT} d_model={RUN_D_MODEL} params={n_p:.3f}M =====', flush=True)
    hp = dict(cnn.V2_HPARAM)
    if RUN_LAM_DOMAIN is not None:
        hp['lam_domain'] = float(RUN_LAM_DOMAIN)
    best_f1, best_ep, bad = -1.0, -1, 0
    for ep in range(1, cnn.EPOCHS + 1):
        model.train()
        losses, f1s, gates = [], [], []
        alpha = cnn.dann_alpha(ep) if use_front else 1.0
        for x, y in tqdm(tr, desc=f'{tag} ep{ep}', leave=False):
            x, y = x.to(cnn.DEVICE), y.to(cnn.DEVICE)
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
        vm = cnn.evaluate(model, va)
        sch.step(vm['f1'])
        gtxt = f' gate={np.mean(gates):.3f}' if gates else ''
        print(f'Ep{ep:02d} loss={np.mean(losses):.4f} trF1={np.mean(f1s):.4f} '
              f'valF1={vm["f1"]:.4f} valAcc={vm["acc"]:.4f}{gtxt}', flush=True)
        if vm['f1'] > best_f1 + 1e-4:
            best_f1, best_ep, bad = vm['f1'], ep, 0
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_ep,
                        'val_f1': best_f1, 'use_front': use_front, 'kind': kind,
                        'dims': train_ds.device_dims, 'n_cls': n_cls}, ckpt)
            print(f'  >> saved {ckpt}', flush=True)
        else:
            bad += 1
            if bad >= cnn.PATIENCE:
                print('early stop', flush=True)
                break
    blob = torch.load(ckpt, map_location=cnn.DEVICE, weights_only=False)
    model.load_state_dict(blob['model_state'])
    return model, best_ep


def upsert(runs, rows):
    add = pd.DataFrame(rows)
    if runs.empty:
        return add
    key = ['backbone', 'k', 'model', 'drop_devices']
    merged = runs.merge(add[key].drop_duplicates(), on=key, how='left', indicator=True)
    keep = runs.loc[merged['_merge'].eq('left_only')].copy()
    return pd.concat([keep, add], ignore_index=True)


def write_summary(runs):
    rows = []
    for _, tag in KINDS:
        sub = runs[runs.backbone == tag]
        if sub.empty:
            continue
        for k, group in sub.groupby('k'):
            covered = group.groupby(['model', 'seed']).drop_devices.nunique()
            need = 9 if int(k) == 1 else 1
            if covered.empty or int(covered.min()) < need:
                continue
            seed_means = group.groupby(['model', 'seed'])[['acc', 'f1']].mean()
            if 'front' not in seed_means.index.get_level_values(0):
                continue
            if 'raw' not in seed_means.index.get_level_values(0):
                continue
            front, raw = seed_means.loc['front'], seed_means.loc['raw']
            gap = front - raw
            rows.append(dict(
                backbone=tag, k=int(k),
                front_f1=float(front.f1.mean()),
                raw_f1=float(raw.f1.mean()),
                gap_f1=float(gap.f1.mean()),
            ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.to_csv(SUMMARY_CSV, index=False)
    lines = [
        '# CoLD / tFusion / TIPSO-Attn on N-BaIoT',
        '',
        'Macro-F1. Slash left is Front, right is Raw. k=1 averages nine devices.',
        f'Front d_model={RUN_D_MODEL}.',
        '',
        '| 模型 | k | Front | Raw | F1 gap |',
        '|---|---:|---:|---:|---:|',
    ]
    for rec in out.itertuples(index=False):
        lines.append(
            f'| {rec.backbone} | {rec.k} | {rec.front_f1:.3f} '
            f'| {rec.raw_f1:.3f} | {rec.gap_f1:.3f} |')
    lines.append('')
    with open(SUMMARY_MD, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    return out


def use_suffix_paths(suffix):
    global CKPT_ROOT, RUNS_CSV, SUMMARY_CSV, SUMMARY_MD
    name = 'paper_adv_' + str(suffix).strip('_')
    CKPT_ROOT = os.path.join(ROOT, name + '_ckpts')
    RUNS_CSV = os.path.join(ROOT, name + '_fulltest_runs.csv')
    SUMMARY_CSV = os.path.join(ROOT, name + '_fulltest_summary.csv')
    SUMMARY_MD = os.path.join(ROOT, name + '_fulltest_summary.md')


def use_raw_seed_paths(seed):
    global CKPT_ROOT, RUNS_CSV, SUMMARY_CSV, SUMMARY_MD
    name = f'paper_adv_raw_s{int(seed)}'
    CKPT_ROOT = os.path.join(ROOT, name + '_ckpts')
    RUNS_CSV = os.path.join(ROOT, name + '_fulltest_runs.csv')
    SUMMARY_CSV = os.path.join(ROOT, name + '_fulltest_summary.csv')
    SUMMARY_MD = os.path.join(ROOT, name + '_fulltest_summary.md')


def use_d100_paths(suffix=''):
    global CKPT_ROOT, RUNS_CSV, SUMMARY_CSV, SUMMARY_MD, RUN_D_MODEL
    name = 'paper_adv_d100' + suffix
    CKPT_ROOT = os.path.join(ROOT, name + '_ckpts')
    RUNS_CSV = os.path.join(ROOT, name + '_fulltest_runs.csv')
    SUMMARY_CSV = os.path.join(ROOT, name + '_fulltest_summary.csv')
    SUMMARY_MD = os.path.join(ROOT, name + '_fulltest_summary.md')
    RUN_D_MODEL = 100


def use_axis_paths(suffix=''):
    global CKPT_ROOT, RUNS_CSV, SUMMARY_CSV, SUMMARY_MD, RUN_D_MODEL, RUN_AXIS
    name = 'paper_adv_axis' + suffix
    CKPT_ROOT = os.path.join(ROOT, name + '_ckpts')
    RUNS_CSV = os.path.join(ROOT, name + '_fulltest_runs.csv')
    SUMMARY_CSV = os.path.join(ROOT, name + '_fulltest_summary.csv')
    SUMMARY_MD = os.path.join(ROOT, name + '_fulltest_summary.md')
    RUN_D_MODEL = 100
    RUN_AXIS = True


def keep_d100(kind, k, use_front):
    if not use_front:
        return False
    if kind == 'cold':
        return int(k) == 1
    if kind in ('tipsogan', 'feco', 'aocids'):
        return True
    return False


def summary_against_base_raw(front_runs):
    if front_runs.empty or not os.path.isfile(BASE_RUNS_CSV):
        return front_runs
    base = pd.read_csv(BASE_RUNS_CSV)
    raw = base[base.model == 'raw']
    keys = front_runs[['backbone', 'k', 'drop_devices']].drop_duplicates()
    raw = raw.merge(keys, on=['backbone', 'k', 'drop_devices'])
    return pd.concat([front_runs, raw], ignore_index=True)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kinds', default=','.join(kind for kind, _ in KINDS))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--d100-front', action='store_true')
    parser.add_argument('--lam-domain', type=float, default=None)
    parser.add_argument('--axis-front', action='store_true')
    parser.add_argument('--gate-target', type=float, default=None)
    parser.add_argument('--only-k', type=int, default=0)
    parser.add_argument('--ks', default='')
    parser.add_argument('--spread-order', action='store_true')
    parser.add_argument('--train-seed', type=int, default=None)
    parser.add_argument('--only-raw', action='store_true')
    parser.add_argument('--only-front', action='store_true')
    parser.add_argument('--drops', default='')
    parser.add_argument('--out-suffix', default='')
    return parser.parse_args(argv)


def smoke():
    dims = [8] * 9
    y = torch.tensor([0, 1, 0, 1])
    x = torch.randn(4, 16, sum(dims))
    x[:, :, 6 * 8:9 * 8] = 0
    for kind, tag in KINDS:
        for use_front in (True, False):
            model = PlugPaper(dims, 2, use_front=use_front, kind=kind)
            batch, target = x, y
            if use_front:
                logits, aux = model(batch, return_aux=True)
                loss = F.cross_entropy(logits, target)
                loss = loss + model.front.cross_protocol_contrastive_loss(aux['h_prime'], target)
            else:
                loss = F.cross_entropy(model(batch), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{tag} front={use_front} loss={loss}')
            loss.backward()
            print(f'smoke {tag} front={use_front} ok', flush=True)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.smoke:
        smoke()
        return
    global RUN_LAM_DOMAIN, RUN_GATE_TARGET
    if args.lam_domain is not None:
        RUN_LAM_DOMAIN = float(args.lam_domain)
    if args.gate_target is not None:
        RUN_GATE_TARGET = float(args.gate_target)
    if args.train_seed is not None:
        cnn.SEED = int(args.train_seed)
    if args.axis_front:
        suffix = ''
        if args.spread_order:
            suffix = '_spread'
        elif args.only_k and RUN_GATE_TARGET is not None:
            suffix = f'_k{args.only_k}_g{str(RUN_GATE_TARGET).replace(".", "p")}'
        if args.train_seed is not None:
            suffix += f'_s{int(args.train_seed)}'
        if args.out_suffix:
            suffix += '_' + str(args.out_suffix).strip('_')
        use_axis_paths(suffix)
        if RUN_LAM_DOMAIN is None:
            RUN_LAM_DOMAIN = 0.0
    elif args.d100_front:
        suffix = '_ld0' if RUN_LAM_DOMAIN == 0 else ''
        use_d100_paths(suffix)
    elif args.out_suffix:
        use_suffix_paths(args.out_suffix)
    elif args.train_seed is not None:
        use_raw_seed_paths(args.train_seed)
    wanted = {item.strip().lower() for item in args.kinds.split(',') if item.strip()}
    kinds = [(kind, tag) for kind, tag in KINDS if kind in wanted or tag.lower() in wanted]
    if args.d100_front or args.axis_front:
        kinds = [(kind, tag) for kind, tag in kinds if kind in ('cold', 'tipsogan', 'feco', 'aocids')]
    if not kinds:
        raise ValueError(f'未知方法 {wanted}')
    lam_show = cnn.V2_HPARAM['lam_domain'] if RUN_LAM_DOMAIN is None else RUN_LAM_DOMAIN
    gate_show = cnn.V2_HPARAM['gate_target'] if RUN_GATE_TARGET is None else RUN_GATE_TARGET
    print(f'd_model={RUN_D_MODEL} lam_domain={lam_show} axis_residual={RUN_AXIS} gate_target={gate_show} train_seed={cnn.SEED}', flush=True)
    order, _ = kdrop.advantage_order(
        kdrop.RANK_SRC, tuple(cnn.DEVICE_NAMES),
        ('CNN', 'MLP', 'TCN', 'LSTM', 'Transformer'))
    plan = settings(order)
    ks = {int(item) for item in args.ks.split(',') if item.strip()}
    if ks:
        plan = [(k, devices) for k, devices in plan if int(k) in ks]
    drops = {item.strip() for item in args.drops.split(',') if item.strip()}
    if drops:
        plan = [(k, devices) for k, devices in plan if set(devices) <= drops]
    print('drop plan:', [(k, '+'.join(names)) for k, names in plan], flush=True)
    pack = kdrop.load_pack()
    names = list(cnn.DEVICE_NAMES)
    if args.spread_order:
        pack = permute_pack(pack, SPREAD_NAMES)
        names = list(SPREAD_NAMES)
        print('spread order:', names, flush=True)
        print('chunks:', [names[i:i + 3] for i in range(0, 9, 3)], flush=True)
    runs = kdrop.load_runs(RUNS_CSV, args.overwrite)
    for kind, tag in kinds:
        os.makedirs(os.path.join(CKPT_ROOT, tag), exist_ok=True)
        for k, devices in plan:
            if args.only_k and int(k) != int(args.only_k):
                continue
            if (args.d100_front or args.axis_front) and kind == 'cold' and int(k) != 1 and not ks:
                continue
            ids = [names.index(name) for name in devices]
            print(f'\n########## {tag} k={k} drop={"+".join(devices)} TEST=FULL ##########', flush=True)
            train_ds = cnn.ArrayWinDS(
                cnn.zero_devices(pack['train'].X_table, pack['dims'], ids),
                pack['train'].y_table, pack['dims'], cnn.WINDOW, cnn.TRAIN_STEP,
                cnn.MAX_WINDOWS, cnn.SEED)
            val_ds = cnn.ArrayWinDS(
                cnn.zero_devices(pack['val'].X_table, pack['dims'], ids),
                pack['val'].y_table, pack['dims'], cnn.WINDOW, cnn.EVAL_STEP,
                cnn.MAX_WINDOWS // 2, cnn.SEED + 1)
            test_ds = pack['test']
            kdrop.assert_train_val_dropped(train_ds, val_ds, test_ds, pack['dims'], ids)
            for use_front, stem in ((True, 'front'), (False, 'raw')):
                if args.only_raw and use_front:
                    continue
                if args.only_front and not use_front:
                    continue
                if (args.d100_front or args.axis_front) and not keep_d100(kind, k, use_front) and int(k) not in ks:
                    continue
                hit = pd.DataFrame()
                if not runs.empty and not args.overwrite:
                    hit = runs[(runs.backbone == tag) & (runs.k == k)
                               & (runs.model == stem) & (runs.drop_devices == '+'.join(devices))]
                if len(hit) >= 1 and not args.overwrite:
                    print(f'skip {tag} k={k} {stem} {"+".join(devices)}', flush=True)
                    continue
                ckpt = os.path.join(
                    CKPT_ROOT, tag, f'best_k{k}_{"_".join(devices)}_{stem}.pth')
                if checkpoint_ready(ckpt) and not args.overwrite:
                    blob = torch.load(ckpt, map_location=cnn.DEVICE, weights_only=False)
                    model = PlugPaper(pack['dims'], pack['n_cls'], use_front=use_front, kind=kind).to(cnn.DEVICE)
                    model.load_state_dict(blob['model_state'])
                    best_ep = int(blob['best_epoch'])
                else:
                    model, best_ep = fit(
                        kind, use_front, train_ds, val_ds, test_ds, pack['n_cls'], pack['cw'],
                        ckpt, f'k{k}-{tag}-{stem}', cnn.SEED)
                    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
                    blob.update(k=k, drop_devices=list(devices), backbone=tag, model=stem,
                                seed=int(cnn.SEED), protocol=f'train{9 - k}_val{9 - k}_test9',
                                d_model=int(RUN_D_MODEL))
                    torch.save(blob, ckpt)
                    print(f'  >> weights {ckpt}', flush=True)
                rows = []
                for metric in cnn.evaluate_full(model, test_ds):
                    rows.append(dict(
                        backbone=tag, kind=kind, model=stem, k=k,
                        drop_devices='+'.join(devices), seed=metric['seed'],
                        protocol=f'train{9 - k}_val{9 - k}_test9',
                        best_epoch=best_ep, checkpoint=ckpt,
                        acc=metric['acc'], f1=metric['f1'],
                        precision=metric['precision'], recall=metric['recall']))
                runs = upsert(runs, rows)
                kdrop.write_runs(runs, RUNS_CSV)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del train_ds, val_ds
            gc.collect()
    if not runs.empty:
        paired = (args.d100_front or args.axis_front) and not args.spread_order
        write_summary(summary_against_base_raw(runs) if paired else runs)
    print('PAPER_ADV_DONE', flush=True)


if __name__ == '__main__':
    main()
