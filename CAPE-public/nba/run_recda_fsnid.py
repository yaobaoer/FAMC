                       
pass
from __future__ import annotations

import argparse
import copy
import gc
import math
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
import run_paper_adv as adv


ROOT = os.path.dirname(os.path.abspath(__file__))
SIGMA = 0.6
TAU = 0.2
LAM_ALIGN = 0.1
MINE_STEPS = 20
MINE_BATCH = 512
SELECT_ROWS = 4096
JOBS = {'recda': 'ReCDA', 'fsnid': 'FSNID'}


class Plug(nn.Module):
    def __init__(self, dims, n_cls, use_front, kind, stat_mask=None):
        super().__init__()
        self.kind = kind
        self.front = None
        self.lam_domain = float(cnn.V2_HPARAM['lam_domain'])
        if use_front:
                                                       
            axis = all(int(dim) == 100 for dim in dims)
            d_model = 100 if axis else cnn.D_MODEL
            self.front = cnn.FrontV2(
                dims, d_model=d_model, gat_heads=1, dropout=0.1,
                axis_residual=axis)
            self.front.use_gat = cnn.USE_GAT
            self.front.set_gate_target(cnn.V2_HPARAM['gate_target'])
            d_in = self.front.out_dim
        else:
            d_in = int(sum(dims))
        self.backbone = adv.build_paper_backbone(kind, d_in, d_hidden=cnn.D_HIDDEN, dropout=0.1)
        self.temporal = adv.TemporalResidual(self.backbone.out_dim)
        self.head = cnn.DualPool(self.backbone.out_dim, n_cls)
        if stat_mask is None:
            self.stat_mask = None
        else:
            mask = torch.as_tensor(stat_mask, dtype=torch.float32).view(1, 1, -1)
            self.register_buffer('stat_mask', mask)

    def prepare(self, x):
        if self.stat_mask is not None:
            x = x * self.stat_mask
        if self.front is None:
            return x, None
        return self.front(x, return_features=True)

    def forward(self, x, return_aux=False):
        seq, aux = self.prepare(x)
        logits = self.head(self.temporal(self.backbone(seq)))
        return (logits, aux) if return_aux else logits


def perturb(x):
    drift = torch.roll(x, 1, 0)
    mask = torch.bernoulli(torch.full_like(x, SIGMA))
    return mask * drift + (1.0 - mask) * x


def pool_proj(model, seq):
    z = model.backbone.proj(model.backbone.encode(seq)).mean(1)
    return F.normalize(z, dim=-1)


def align_loss(model, seq):
    z = pool_proj(model, seq)
    zt = pool_proj(model, perturb(seq))
    logits = z @ zt.transpose(0, 1) / TAU
    target = torch.arange(z.size(0), device=z.device)
    return F.cross_entropy(logits, target)


def constraint_loss(model, seq, ref_enc):
    now = model.backbone.encode(seq)
    with torch.no_grad():
        ref = ref_enc(seq)
    return F.mse_loss(now, ref)


@torch.no_grad()
def instructive_keep(model, raw):
    training = model.training
    model.eval()
    flat = raw.reshape(raw.size(0), -1)
    drift_raw = torch.roll(raw, 1, 0)
    dflat = drift_raw.reshape(raw.size(0), -1)
    z = model.backbone.encode(model.prepare(raw)[0]).mean(1)
    zd = model.backbone.encode(model.prepare(drift_raw)[0]).mean(1)
    c0, c1 = zd[:1].clone(), zd[-1:].clone()
    left = torch.ones(zd.size(0), dtype=torch.bool, device=zd.device)
    for _ in range(4):
        d0 = (zd - c0).pow(2).sum(-1)
        d1 = (zd - c1).pow(2).sum(-1)
        left = d0 <= d1
        if bool(left.any()):
            c0 = zd[left].mean(0, keepdim=True)
        if bool((~left).any()):
            c1 = zd[~left].mean(0, keepdim=True)
    o0 = dflat[left].mean(0) if bool(left.any()) else dflat.mean(0)
    o1 = dflat[~left].mean(0) if bool((~left).any()) else dflat.mean(0)

    def cosine(a, b):
        return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)

    s_raw = torch.stack([
        cosine(flat, o0.expand_as(flat)),
        cosine(flat, o1.expand_as(flat)),
    ], dim=-1)
    s_lat = torch.stack([
        cosine(z, c0.expand_as(z)),
        cosine(z, c1.expand_as(z)),
    ], dim=-1)
    score = (s_raw / (s_lat.abs() + 1e-6)).max(-1).values
    k = max(1, int(round(DELTA * raw.size(0))))
    keep = torch.zeros(raw.size(0), dtype=torch.bool, device=raw.device)
    keep[score.topk(k).indices] = True
    if training:
        model.train()
    return keep


def mine_trace(x, y, n_cls, steps):
    n, d = x.shape
    yoh = F.one_hot(y, n_cls).float()
    net = nn.Sequential(nn.Linear(d + n_cls, 50), nn.ELU(), nn.Linear(50, 1)).to(x.device)
    for mod in net:
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.02)
            nn.init.zeros_(mod.bias)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    ma = torch.tensor(1.0, device=x.device)
    scores = []
    bs = min(MINE_BATCH, n)
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), device=x.device)
        perm = torch.randint(0, n, (bs,), device=x.device)
        joint = torch.cat([x[idx], yoh[idx]], dim=-1)
        marg = torch.cat([x[idx], yoh[perm]], dim=-1)
        t = net(joint)
        et = torch.exp(net(marg).clamp(-20, 20))
        ma = 0.999 * ma + 0.001 * et.mean().detach()
        loss = -(t.mean() - et.mean() / ma.clamp_min(1e-6))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scores.append(float((t.mean() - torch.log(et.mean().clamp_min(1e-8))).detach()))
    del net, opt
    return np.asarray(scores, dtype=np.float64)


def _tail_stats(trace):
    tail = trace[-max(3, len(trace) // 3):]
    mu = float(np.mean(tail))
    sd = float(np.std(tail))
    return mu, sd


def select_stats(table, labels, dims, drop_ids, n_cls):
    width = int(dims[0])
    keep_devs = [i for i in range(len(dims)) if i not in set(drop_ids)]
    cols, stat_of = [], []
    for dev in keep_devs:
        base = dev * width
        for stat in range(width):
            cols.append(base + stat)
            stat_of.append(stat)
    rng = np.random.RandomState(cnn.SEED)
    labels = np.asarray(labels)
    take = min(SELECT_ROWS, len(labels))
    pick = []
    for cls in range(n_cls):
        pool = np.flatnonzero(labels == cls)
        if len(pool) == 0:
            continue
        n = max(1, int(round(take * (len(pool) / len(labels)))))
        pick.append(rng.choice(pool, min(n, len(pool)), replace=False))
    if not pick:
        pick = [rng.choice(len(labels), take, replace=False)]
    rows = np.concatenate(pick)
    if len(rows) > take:
        rows = rng.choice(rows, take, replace=False)
    x_np = np.asarray(table[rows][:, cols], dtype=np.float32)
    y_np = labels[rows].astype(np.int64)
    device = cnn.DEVICE
    x = torch.as_tensor(x_np, device=device)
    y = torch.as_tensor(y_np, device=device)
    stat_of = np.asarray(stat_of)

    def columns(stats):
        want = np.isin(stat_of, list(stats))
        return x[:, torch.as_tensor(np.flatnonzero(want), device=device)]

    print(f'FSNID select rows={len(rows)} kept_devices={len(keep_devs)} stats={width}', flush=True)
    null = mine_trace(torch.rand(x.size(0), 1, device=device), y, n_cls, MINE_STEPS)
    nm_mu, nm_sd = _tail_stats(null)
    nm_upper = nm_mu + 2.0 * (nm_sd / math.sqrt(3.0))
    single = []
    for stat in range(width):
        trace = mine_trace(columns([stat]), y, n_cls, MINE_STEPS)
        single.append(_tail_stats(trace)[0])
        if (stat + 1) % 20 == 0:
            print(f'  order {stat + 1}/{width}', flush=True)
    order = list(np.argsort(single))
    alive = list(range(width))
    for stat in order:
        if len(alive) <= 1:
            break
        trial = [item for item in alive if item != stat]
        full = mine_trace(columns(alive), y, n_cls, MINE_STEPS)
        dropped = mine_trace(columns(trial), y, n_cls, MINE_STEPS)
        phi = full - dropped
        mu, sd = _tail_stats(phi)
        lower = mu - 2.0 * (sd / math.sqrt(3.0))
        if lower < nm_upper:
            alive = trial
    if len(alive) < 5:
        alive = list(np.argsort(single)[::-1][:10])
        print(f'FSNID keep fallback top10', flush=True)
    mask = np.zeros(int(sum(dims)), dtype=np.float32)
    for dev in range(len(dims)):
        for stat in alive:
            mask[dev * width + stat] = 1.0
    print(f'FSNID kept {len(alive)}/{width} stats, nm_upper={nm_upper:.4f}', flush=True)
    del x, y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mask


from paired_protocol import paired_fit

@paired_fit
def fit(kind, use_front, train_ds, val_ds, n_cls, cw, ckpt, tag, seed, stat_mask):
    cnn.set_seed(seed)
    counts = np.maximum(np.bincount(train_ds.labels, minlength=n_cls).astype(np.float64), 1)
    sw = 1.0 / np.sqrt(counts[train_ds.labels])
    sw /= sw.mean()
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(train_ds), True, generator=gen)
    tr = DataLoader(train_ds, cnn.BATCH, sampler=sampler, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    va = DataLoader(val_ds, cnn.BATCH, shuffle=False, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(42))
    model = Plug(train_ds.device_dims, n_cls, use_front, kind, stat_mask).to(cnn.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cnn.LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 3)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'\n===== {tag} front={use_front} params={n_p:.3f}M =====', flush=True)
    hp = dict(cnn.V2_HPARAM)
    hp['lam_domain'] = float(model.lam_domain)
    best_f1, best_ep, bad = -1.0, -1, 0
    for ep in range(1, cnn.EPOCHS + 1):
        model.train()
        losses, f1s = [], []
        alpha = cnn.dann_alpha(ep) if use_front else 1.0
        for x, y in tqdm(tr, desc=f'{tag} ep{ep}', leave=False):
            x, y = x.to(cnn.DEVICE), y.to(cnn.DEVICE)
            opt.zero_grad(set_to_none=True)
            seq, aux = model.prepare(x)
            if kind == 'recda':
                feat = model.backbone.to_out(model.backbone.encode(seq))
            else:
                feat = model.backbone(seq)
            logits = model.head(model.temporal(feat))
            loss = F.cross_entropy(logits, y, weight=cw)
            if kind == 'recda':
                loss = loss + LAM_ALIGN * align_loss(model, seq)
            if use_front:
                loss = loss + hp['lam_contra'] * model.front.cross_protocol_contrastive_loss(
                    aux['h_prime'], y, temperature=hp['contra_temp'])
                loss = loss + hp['lam_domain'] * model.front.domain_adversarial_loss(
                    aux['h_prime'], alpha=alpha)
                loss = loss + hp['lam_gate'] * model.front.gate_reg_loss(aux['gate'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            f1s.append(f1_score(
                y.cpu().numpy(), logits.argmax(1).detach().cpu().numpy(),
                average='macro', zero_division=0))
        vm = cnn.evaluate(model, va)
        sch.step(vm['f1'])
        print(f'Ep{ep:02d} loss={np.mean(losses):.4f} trF1={np.mean(f1s):.4f} '
              f'valF1={vm["f1"]:.4f} valAcc={vm["acc"]:.4f}', flush=True)
        if vm['f1'] > best_f1 + 1e-4:
            best_f1, best_ep, bad = vm['f1'], ep, 0
            torch.save({
                'model_state': model.state_dict(), 'best_epoch': best_ep,
                'val_f1': best_f1, 'use_front': use_front, 'kind': kind,
                'dims': train_ds.device_dims, 'n_cls': n_cls,
                'stat_mask': None if stat_mask is None else np.asarray(stat_mask),
            }, ckpt)
            print(f'  >> saved {ckpt}', flush=True)
        else:
            bad += 1
            if bad >= cnn.PATIENCE:
                print('early stop', flush=True)
                break
    blob = torch.load(ckpt, map_location=cnn.DEVICE, weights_only=False)
    model.load_state_dict(blob['model_state'])
    return model, best_ep


def smoke():
    dims = [8] * 9
    y = torch.tensor([0, 1, 2, 0])
    x = torch.randn(4, 16, sum(dims))
    for kind in ('recda', 'fsnid'):
        model = Plug(dims, 3, True, kind, np.ones(sum(dims), np.float32)).to(cnn.DEVICE)
        x = x.to(cnn.DEVICE)
        seq, aux = model.prepare(x)
        logits = model.head(model.temporal(model.backbone(seq)))
        loss = F.cross_entropy(logits, y.to(cnn.DEVICE))
        if kind == 'recda':
            loss = loss + align_loss(model, seq)
        loss.backward()
        print(f'smoke {kind} {tuple(logits.shape)} ok', flush=True)
    table = np.random.randn(600, sum(dims)).astype(np.float32)
    labels = np.random.randint(0, 3, size=600)
    global MINE_STEPS, SELECT_ROWS
    MINE_STEPS, SELECT_ROWS = 2, 128
    mask = select_stats(table, labels, dims, [0], 3)
    print('smoke mask', int(mask.sum()), '/', mask.size, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', required=True, choices=sorted(JOBS))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--only-k', type=int, default=0)
    parser.add_argument('--ks', default='')
    parser.add_argument('--only-raw', action='store_true')
    parser.add_argument('--only-front', action='store_true')
    parser.add_argument('--drops', default='')
    parser.add_argument('--out-suffix', default='')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.smoke:
        smoke()
        return
    kind = args.kind
    tag = JOBS[kind]
    adv.use_suffix_paths(args.out_suffix or kind)
    adv.KINDS = ((kind, tag),)
    order, _ = kdrop.advantage_order(
        kdrop.RANK_SRC, tuple(cnn.DEVICE_NAMES),
        ('CNN', 'MLP', 'TCN', 'LSTM', 'Transformer'))
    plan = adv.settings(order)
    ks = {int(item) for item in args.ks.split(',') if item.strip()}
    if ks:
        plan = [(k, devices) for k, devices in plan if int(k) in ks]
    if args.only_k:
        plan = [(k, devices) for k, devices in plan if int(k) == int(args.only_k)]
    drops = {item.strip() for item in args.drops.split(',') if item.strip()}
    if drops:
        plan = [(k, devices) for k, devices in plan if set(devices) <= drops]
    print('drop plan:', [(k, '+'.join(names)) for k, names in plan], flush=True)
    pack = kdrop.load_pack()
    names = list(cnn.DEVICE_NAMES)
    runs = kdrop.load_runs(adv.RUNS_CSV, args.overwrite)
    os.makedirs(os.path.join(adv.CKPT_ROOT, tag), exist_ok=True)
    for k, devices in plan:
        ids = [names.index(name) for name in devices]
        print(f'\n########## {tag} k={k} drop={"+".join(devices)} TEST=FULL ##########', flush=True)
        train_x = cnn.zero_devices(pack['train'].X_table, pack['dims'], ids)
        val_x = cnn.zero_devices(pack['val'].X_table, pack['dims'], ids)
        stat_mask = None
        if kind == 'fsnid':
            stat_mask = select_stats(train_x, pack['train'].y_table, pack['dims'], ids, pack['n_cls'])
        train_ds = cnn.ArrayWinDS(
            train_x, pack['train'].y_table, pack['dims'], cnn.WINDOW, cnn.TRAIN_STEP,
            cnn.MAX_WINDOWS, cnn.SEED)
        val_ds = cnn.ArrayWinDS(
            val_x, pack['val'].y_table, pack['dims'], cnn.WINDOW, cnn.EVAL_STEP,
            cnn.MAX_WINDOWS // 2, cnn.SEED + 1)
        test_ds = pack['test']
        kdrop.assert_train_val_dropped(train_ds, val_ds, test_ds, pack['dims'], ids)
        for use_front, stem in ((True, 'front'), (False, 'raw')):
            if args.only_raw and use_front:
                continue
            if args.only_front and not use_front:
                continue
            hit = pd.DataFrame()
            if not runs.empty and not args.overwrite:
                hit = runs[(runs.backbone == tag) & (runs.k == k)
                           & (runs.model == stem) & (runs.drop_devices == '+'.join(devices))]
            if len(hit) >= len(cnn.EVAL_SEEDS) and not args.overwrite:
                print(f'skip {tag} k={k} {stem} {"+".join(devices)}', flush=True)
                continue
            ckpt = os.path.join(adv.CKPT_ROOT, tag, f'best_k{k}_{"_".join(devices)}_{stem}.pth')
            if adv.checkpoint_ready(ckpt) and not args.overwrite:
                blob = torch.load(ckpt, map_location=cnn.DEVICE, weights_only=False)
                model = Plug(
                    pack['dims'], pack['n_cls'], use_front, kind, blob.get('stat_mask')).to(cnn.DEVICE)
                model.load_state_dict(blob['model_state'])
                best_ep = int(blob['best_epoch'])
            else:
                model, best_ep = fit(
                    kind, use_front, train_ds, val_ds, pack['n_cls'], pack['cw'],
                    ckpt, f'k{k}-{tag}-{stem}', cnn.SEED, stat_mask)
                blob = torch.load(ckpt, map_location='cpu', weights_only=False)
                blob.update(
                    k=k, drop_devices=list(devices), backbone=tag, model=stem,
                    seed=int(cnn.SEED), protocol=f'train{9 - k}_val{9 - k}_test9',
                    d_model=int(cnn.D_MODEL))
                torch.save(blob, ckpt)
            rows = []
            for metric in cnn.bootstrap_eval(model, test_ds):
                rows.append(dict(
                    backbone=tag, kind=kind, model=stem, k=k,
                    drop_devices='+'.join(devices), seed=metric['seed'],
                    protocol=f'train{9 - k}_val{9 - k}_test9',
                    best_epoch=best_ep, checkpoint=ckpt,
                    acc=metric['acc'], f1=metric['f1'],
                    precision=metric['precision'], recall=metric['recall']))
            runs = adv.upsert(runs, rows)
            kdrop.write_runs(runs, adv.RUNS_CSV)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del train_ds, val_ds, train_x, val_x
        gc.collect()
    if not runs.empty:
        adv.write_summary(runs)
    print(f'{tag}_DONE', flush=True)


if __name__ == '__main__':
    main()
