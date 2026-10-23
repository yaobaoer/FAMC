                       
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import WeightedRandomSampler

import run_cnn_align as base
from _run_ablation_swap import HP, SWAPS, SwapCNN

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, 'ablation_swap_lodo')
CACHE = os.path.join(_HERE, '.nba_lodo_tables.pt')
FAST_BATCH = 64


class TensorDS:
    def __init__(self, samples, labels, dims):
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
        y = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.samples = x
        self.labels = y
        self.device_dims = list(dims)

    def __len__(self):
        return int(self.y.numel())


class TensorLoader:
    def __init__(self, ds, batch, sampler=None):
        self.x, self.y = ds.x, ds.y
        self.batch = int(batch)
        self.sampler = sampler
        self.dataset = ds

    def __len__(self):
        n = len(self.sampler) if self.sampler is not None else int(self.y.numel())
        return max((n + self.batch - 1) // self.batch, 1)

    def __iter__(self):
        if self.sampler is not None:
            idx = torch.as_tensor(list(self.sampler), dtype=torch.long)
        else:
            idx = torch.arange(int(self.y.numel()))
        for i in range(0, len(idx), self.batch):
            sl = idx[i:i + self.batch]
            yield self.x[sl], self.y[sl]


def _pack_from_blob(blob):
    le = LabelEncoder()
    le.classes_ = np.asarray(blob['classes'])
    dims = list(blob['dims'])

    class Tables:
        pass

    train = Tables()
    train.X_table, train.y_table = blob['Xtr'], blob['ytr']
    val = Tables()
    val.X_table, val.y_table = blob['Xva'], blob['yva']
    test = TensorDS(blob['te_samples'], blob['te_labels'], dims)
    test.X_table, test.y_table = blob['Xte'], blob['yte']
    n_cls = len(le.classes_)
    counts = np.maximum(np.bincount(blob['tr_labels'], minlength=n_cls).astype(np.float64), 1)
    cw = torch.tensor(
        np.clip(np.sqrt(counts.sum() / (n_cls * counts)), 0.5, 3.0),
        dtype=torch.float32, device=base.DEVICE)
    return dict(train=train, val=val, test=test, dims=dims, n_cls=n_cls,
                le=le, cw=cw, d_feat=blob['d_feat'])


def load_pack_cached():
    lock = CACHE + '.lock'
    if os.path.isfile(CACHE) and os.path.getsize(CACHE) > 1000:
        print(f'load pack cache {CACHE}', flush=True)
        return _pack_from_blob(torch.load(CACHE, map_location='cpu', weights_only=False))
    got_lock = False
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        got_lock = True
    except FileExistsError:
        print('wait pack cache', flush=True)
        for _ in range(900):
            time.sleep(2)
            if os.path.isfile(CACHE) and os.path.getsize(CACHE) > 1000:
                return _pack_from_blob(torch.load(CACHE, map_location='cpu', weights_only=False))
        raise RuntimeError('NBA pack cache wait timeout')
    try:
        pack = base.load_pack()
        blob = dict(
            Xtr=pack['train'].X_table, ytr=pack['train'].y_table,
            Xva=pack['val'].X_table, yva=pack['val'].y_table,
            Xte=pack['test'].X_table, yte=pack['test'].y_table,
            te_samples=pack['test'].samples, te_labels=pack['test'].labels,
            tr_labels=pack['train'].labels,
            dims=list(pack['dims']),
            classes=list(pack['le'].classes_),
            d_feat=pack['d_feat'],
        )
        tmp = CACHE + '.tmp'
        torch.save(blob, tmp)
        os.replace(tmp, CACHE)
        print(f'saved pack cache {CACHE}', flush=True)
        return pack
    finally:
        if got_lock:
            try:
                os.remove(lock)
            except OSError:
                pass


def _windows(table, labels, dims, step, n_cap, seed):
    ds = base.ArrayWinDS(table, labels, dims, base.WINDOW, step, n_cap, seed)
    return TensorDS(ds.samples, ds.labels, dims)


def _has_test(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 32:
        return False
    text = open(path, encoding='utf-8', errors='ignore').read()
    return 'test_macro_f1=' in text and 'drop_device=' in text


def run_swap(name):
    if name not in SWAPS:
        raise SystemExit(f'未知替换 {name}，可选 {tuple(SWAPS)}')
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    spec = dict(SWAPS[name])
    spec['use_node_mask'] = False
    gamma = float(HP['gate_target'])
    pack = load_pack_cached()
    train_ds, val_ds, test_ds = pack['train'], pack['val'], pack['test']
    if not isinstance(test_ds, TensorDS):
        test_ds = TensorDS(test_ds.samples, test_ds.labels, pack['dims'])
    dims = list(pack['dims'])
    names = list(base.DEVICE_NAMES)
    n_cls, cw = pack['n_cls'], pack['cw']
    te = TensorLoader(test_ds, FAST_BATCH)
    print(
        f'SWAP_LODO_START {name} devices={names} gat={int(spec["use_gat"])} '
        f'w={spec["weight_mode"]} λc={spec["lam_contra"]} λd={spec["lam_domain"]} '
        f'λg={spec["lam_gate"]} γ={gamma} test=FULL batch={FAST_BATCH} '
        f'node_mask={int(spec.get("use_node_mask", True))} '
        f'contra_pos={spec.get("contra_pos", "xdev")}',
        flush=True,
    )
    for p, pname in enumerate(names):
        stem = f'{name}_{pname}'
        ckpt = os.path.join(OUT_DIR, f'best_{stem}.pth')
        res = os.path.join(OUT_DIR, f'{stem}_results.txt')
        if _has_test(res) and os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1000:
            print(f'skip {stem}', flush=True)
            continue
        tr_ds = _windows(
            base.zero_devices(train_ds.X_table, dims, [p]), train_ds.y_table,
            dims, base.TRAIN_STEP, base.MAX_WINDOWS, base.SEED)
        va_ds = _windows(
            base.zero_devices(val_ds.X_table, dims, [p]), val_ds.y_table,
            dims, base.EVAL_STEP, base.MAX_WINDOWS // 2, base.SEED + 1)
        tag = f'{spec["tag"]}-lodo-{pname}'
        print(f'\n# {tag} train/val zero {pname}  test=FULL', flush=True)
        base.set_seed(base.SEED)
        counts = np.maximum(np.bincount(tr_ds.labels, minlength=n_cls).astype(np.float64), 1)
        sw = 1.0 / np.sqrt(counts[tr_ds.labels])
        sw /= sw.mean()
        gen = torch.Generator().manual_seed(base.SEED)
        sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(tr_ds), True, generator=gen)
        tr = TensorLoader(tr_ds, FAST_BATCH, sampler=sampler)
        va = TensorLoader(va_ds, FAST_BATCH)
        model = SwapCNN(dims, n_cls, spec).to(base.DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 3)
        n_p = sum(t.numel() for t in model.parameters()) / 1e6
        print(f'model={tag}  params={n_p:.3f}M  save={ckpt}  batch={FAST_BATCH}', flush=True)
        best_f1, best_ep, bad, last_gate = -1.0, -1, 0, 0.0
        for ep in range(1, base.EPOCHS + 1):
            model.train()
            losses, gates = [], []
            pred_chunks, lab_chunks = [], []
            alpha = base.dann_alpha(ep)
            for x, y in tr:
                x = x.to(base.DEVICE, non_blocking=True)
                y = y.to(base.DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits, aux = model(x, return_aux=True)
                loss = F.cross_entropy(logits, y, weight=cw)
                if spec['lam_contra']:
                    loss = loss + spec['lam_contra'] * model.front.cross_protocol_contrastive_loss(
                        aux['h_prime'], y, temperature=HP['contra_temp'])
                if spec['lam_domain']:
                    loss = loss + spec['lam_domain'] * model.front.domain_adversarial_loss(
                        aux['h_prime'], alpha=alpha)
                if spec['lam_gate']:
                    loss = loss + spec['lam_gate'] * model.front.gate_reg_loss(aux['gate'])
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach()))
                pred_chunks.append(logits.argmax(1).detach())
                lab_chunks.append(y.detach())
                gates.append(float(aux['gate'].detach().mean()))
            yp = torch.cat(pred_chunks).cpu().numpy()
            yt = torch.cat(lab_chunks).cpu().numpy()
            tr_f1 = float(base.f1_score(yt, yp, average='macro', zero_division=0))
            vm = _eval(model, va)
            sch.step(vm['f1'])
            last_gate = float(np.mean(gates)) if gates else 0.0
            print(f'Ep{ep:02d} loss={np.mean(losses):.4f} trF1={tr_f1:.4f} '
                  f'valF1={vm["f1"]:.4f} valAcc={vm["acc"]:.4f} gate={last_gate:.3f}',
                  flush=True)
            if vm['f1'] > best_f1 + 1e-4:
                best_f1, best_ep, bad = vm['f1'], ep, 0
                torch.save({
                    'model_state': model.state_dict(),
                    'best_epoch': best_ep, 'val_f1': best_f1,
                    'swap': name, 'drop_device': pname,
                    'dims': dims, 'n_cls': n_cls, 'seed': base.SEED,
                }, ckpt)
                print(f'  >> saved {ckpt}', flush=True)
            else:
                bad += 1
                if bad >= base.PATIENCE:
                    print('early stop', flush=True)
                    break
        raw = torch.load(ckpt, map_location=base.DEVICE, weights_only=False)
        model.load_state_dict(raw['model_state'])
        tm = _eval(model, te)
        labels, pred = [], []
        model.eval()
        with torch.no_grad():
            for x, y in te:
                pred.extend(model(x.to(base.DEVICE, non_blocking=True)).argmax(1).cpu().tolist())
                labels.extend(y.tolist())
        names_cls = list(pack['le'].classes_)
        report = classification_report(
            labels, pred, target_names=names_cls, labels=list(range(n_cls)),
            digits=4, zero_division=0)
        text = (
            f'model={tag}\n'
            f'params_M={n_p:.6f}\n'
            f'data=nba_iot_11class.csv window={base.WINDOW} '
            f'step={base.TRAIN_STEP}/{base.EVAL_STEP}\n'
            f'best_epoch={best_ep}\n'
            f'seed={base.SEED}\n'
            f'val_macro_f1={best_f1:.6f}\n'
            f'test_accuracy={tm["acc"]:.6f}\n'
            f'test_macro_precision={tm["precision"]:.6f}\n'
            f'test_macro_recall={tm["recall"]:.6f}\n'
            f'test_macro_f1={tm["f1"]:.6f}\n'
            f'lam_contra={spec["lam_contra"]:.6f}\n'
            f'lam_domain={spec["lam_domain"]:.6f}\n'
            f'lam_gate={spec["lam_gate"]:.6f}\n'
            f'gate_target={gamma:.6f}\n'
            f'gate_mean={last_gate:.6f}\n'
            f'swap={name}\n'
            f'drop_device={pname}\n'
            f'protocol=train/val zero 1 device, test=FULL\n'
            f'gat={int(spec["use_gat"])}\n'
            f'weight_mode={spec["weight_mode"]}\n'
            f'node_mask={int(spec.get("use_node_mask", True))}\n'
            f'contra_pos={spec.get("contra_pos", "xdev")}\n\n'
            f'===== TEST =====\n{report}\n'
        )
        with open(res, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'TEST Acc={tm["acc"]:.4f} F1={tm["f1"]:.4f} best_ep={best_ep}', flush=True)
        print(f'saved {res}', flush=True)
    print(f'SWAP_LODO_DONE {name}', flush=True)


@torch.no_grad()
def _eval(model, loader):
    model.eval()
    pred_chunks, lab_chunks = [], []
    for x, y in loader:
        logits = model(x.to(base.DEVICE, non_blocking=True))
        pred_chunks.append(logits.argmax(1).cpu())
        lab_chunks.append(y.cpu() if torch.is_tensor(y) else torch.as_tensor(y))
    yp = torch.cat(pred_chunks).numpy()
    yt = torch.cat(lab_chunks).numpy()
    return base.metrics(yt, yp)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--swap', choices=list(SWAPS), required=True)
    args = p.parse_args()
    run_swap(args.swap)
