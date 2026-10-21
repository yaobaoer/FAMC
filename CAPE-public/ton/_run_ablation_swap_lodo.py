                       
pass
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score,
)

import _run_cnn as base
from _run_ablation_swap import HP, SWAPS, build_cls

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, 'ablation_swap_lodo')
FAST_BATCH = 256


class TensorDS:
    def __init__(self, samples, labels, device_dims):
        x = np.ascontiguousarray(np.asarray(samples, dtype=np.float32))
        y = np.ascontiguousarray(np.asarray(labels, dtype=np.int64))
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.samples = x
        self.labels = y
        self.device_dims = list(device_dims)

    def __len__(self):
        return int(self.y.numel())


class TensorLoader:
    def __init__(self, ds, batch, sampler=None, shuffle=False):
        self.x, self.y = ds.x, ds.y
        self.batch = int(batch)
        self.sampler = sampler
        self.shuffle = bool(shuffle)
        self.dataset = ds

    def __len__(self):
        n = len(self.sampler) if self.sampler is not None else int(self.y.numel())
        return max((n + self.batch - 1) // self.batch, 1)

    def __iter__(self):
        if self.sampler is not None:
            idx = torch.as_tensor(list(self.sampler), dtype=torch.long)
        elif self.shuffle:
            idx = torch.randperm(int(self.y.numel()))
        else:
            idx = torch.arange(int(self.y.numel()))
        for i in range(0, len(idx), self.batch):
            sl = idx[i:i + self.batch]
            yield self.x[sl], self.y[sl]


def _fast_make_loaders(train_ds, val_ds, test_ds, sample_weight, seed=None):
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(int(seed))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.DoubleTensor(sample_weight), len(train_ds), True, generator=gen)
    return (
        TensorLoader(train_ds, FAST_BATCH, sampler=sampler),
        TensorLoader(val_ds, FAST_BATCH),
        TensorLoader(test_ds, FAST_BATCH),
    )


def _fast_train_epoch(model, loader, optimizer, device, n_cls, class_weight,
                      lam_contra=0.0, lam_domain=0.0, lam_gate=0.0,
                      domain_alpha=1.0, contra_temp=0.1):
    model.train()
    total, n_seen = 0.0, 0
    gate_sum, gate_n = 0.0, 0
    pred_chunks, lab_chunks = [], []
    use_aux = getattr(model, 'front_ver', '') == 'v2' or getattr(model, 'needs_aux', False)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_aux:
            logits, aux = model(x, return_aux=True)
        else:
            logits, aux = model(x), None
        loss = base._cls_loss(logits, y, class_weight, n_cls)
        if aux is not None and 'h_prime' in aux and getattr(model, 'front', None) is not None:
            hp = aux['h_prime']
            if lam_contra:
                loss = loss + lam_contra * model.front.cross_protocol_contrastive_loss(
                    hp, y, temperature=contra_temp)
            if lam_domain:
                loss = loss + lam_domain * model.front.domain_adversarial_loss(
                    hp, alpha=domain_alpha)
            if lam_gate:
                loss = loss + lam_gate * model.front.gate_reg_loss(aux['gate'])
            gate_sum += float(aux['gate'].mean().detach())
            gate_n += 1
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = int(x.size(0))
        total += float(loss.detach()) * bs
        n_seen += bs
        pred_chunks.append(logits.argmax(1).detach())
        lab_chunks.append(y.detach())
    if pred_chunks:
        yp = torch.cat(pred_chunks).cpu().numpy()
        yt = torch.cat(lab_chunks).cpu().numpy()
        f1 = f1_score(yt, yp, average='macro', zero_division=0)
    else:
        f1 = 0.0
    return total / max(n_seen, 1), f1, gate_sum / max(gate_n, 1)


@torch.no_grad()
def _fast_evaluate(model, loader, device, label_encoder, desc='Eval'):
    del desc
    model.eval()
    pred_chunks, lab_chunks = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        pred_chunks.append(logits.argmax(1).cpu())
        lab_chunks.append(y.cpu() if torch.is_tensor(y) else torch.as_tensor(y))
    preds = torch.cat(pred_chunks).numpy()
    trues = torch.cat(lab_chunks).numpy()
    all_labels = list(range(len(label_encoder.classes_)))
    names = list(label_encoder.classes_)
    report = classification_report(
        trues, preds, labels=all_labels, target_names=names, digits=4, zero_division=0)
    kw = dict(average='macro', labels=all_labels, zero_division=0)
    return {
        'macro_f1': float(f1_score(trues, preds, **kw)),
        'macro_precision': float(precision_score(trues, preds, **kw)),
        'macro_recall': float(recall_score(trues, preds, **kw)),
        'weighted_f1': float(f1_score(trues, preds, average='weighted', labels=all_labels, zero_division=0)),
        'accuracy': float(accuracy_score(trues, preds)),
        'report': report,
        'y_true': np.asarray(trues),
        'y_pred': np.asarray(preds),
    }


def _install_fast():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    base.ArrayWinDS = TensorDS
    base.BATCH_SIZE = FAST_BATCH
    base.make_loaders = _fast_make_loaders
    base.train_epoch = _fast_train_epoch
    base.evaluate = _fast_evaluate


def _has_test(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 32:
        return False
    text = open(path, encoding='utf-8', errors='ignore').read()
    return 'test_macro_f1=' in text and 'drop_device=' in text


def run_swap(name):
    if name not in SWAPS:
        raise SystemExit(f'未知替换 {name}，可选 {tuple(SWAPS)}')
    _install_fast()
    os.makedirs(OUT_DIR, exist_ok=True)
    spec = dict(SWAPS[name])
    pack = base.load_datasets()
    train_ds, val_ds, test_ds = pack['train_ds'], pack['val_ds'], pack['test_ds']
    test_ds = TensorDS(test_ds.samples, test_ds.labels, test_ds.device_dims)
    dims = list(train_ds.device_dims)
    names = list(base.DEVICE_PREFIXES)[:len(dims)]
    print(
        f'SWAP_LODO_START {name} devices={names} gat={int(spec["use_gat"])} '
        f'w={spec["weight_mode"]} λc={spec["lam_contra"]} λd={spec["lam_domain"]} '
        f'λg={spec["lam_gate"]} contra_pos={spec.get("contra_pos", "xdev")} '
        f'test=FULL batch={FAST_BATCH}',
        flush=True,
    )
    model_cls = build_cls(spec)
    for p, pname in enumerate(names):
        stem = f'{name}_{pname}'
        ckpt = os.path.join(OUT_DIR, f'best_{stem}.pth')
        res = os.path.join(OUT_DIR, f'{stem}_results.txt')
        if _has_test(res) and os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1000:
            print(f'skip {stem}', flush=True)
            continue
        tr = TensorDS(base.zero_devices(train_ds.samples, dims, [p]), train_ds.labels, dims)
        va = TensorDS(base.zero_devices(val_ds.samples, dims, [p]), val_ds.labels, dims)
        tag = f'{spec["tag"]}-lodo-{pname}'
        print(f'\n# {tag} train/val zero {pname}  test=FULL', flush=True)
        base.fit_model(
            use_front=True, front_ver='v2',
            train_ds=tr, val_ds=va, test_ds=test_ds,
            n_cls=pack['n_cls'], label_encoder=pack['label_encoder'],
            sample_weight=pack['sample_weight'], class_weight=pack['class_weight'],
            ckpt_path=ckpt, result_path=res, cm_path=None, tag=tag,
            model_cls=model_cls, seed=42,
            lam_contra=spec['lam_contra'], lam_domain=spec['lam_domain'],
            lam_gate=spec['lam_gate'],
            contra_temp=HP['contra_temp'], gate_target=HP['gate_target'],
        )
        with open(res, 'a', encoding='utf-8') as f:
            f.write(f'swap={name}\n')
            f.write(f'drop_device={pname}\n')
            f.write('protocol=train/val zero 1 device, test=FULL\n')
            f.write(f'gat={int(spec["use_gat"])}\n')
            f.write(f'weight_mode={spec["weight_mode"]}\n')
            f.write(f'contra_pos={spec.get("contra_pos", "xdev")}\n')
    print(f'SWAP_LODO_DONE {name}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--swap', choices=list(SWAPS), required=True)
    args = p.parse_args()
    run_swap(args.swap)
