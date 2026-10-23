                       
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import run_cnn_align as base
from activity_gated_gat_v2 import ActivityGatedGAT as FrontV2

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, 'ablation_swap_cnn')
HP = base.V2_HPARAM

SWAPS = {
    'swap_w': dict(
        tag='SwapUniformW+CNN',
        weight_mode='uniform', ablation_mode=None, use_gat=False,
        lam_contra=HP['lam_contra'], lam_domain=HP['lam_domain'],
        lam_gate=HP['lam_gate'],
    ),
    'swap_g': dict(
        tag='SwapFixedGamma+CNN',
        weight_mode='consistency', ablation_mode='cons', use_gat=False,
        lam_contra=HP['lam_contra'], lam_domain=HP['lam_domain'],
        lam_gate=0.0,
    ),
    'swap_sem': dict(
        tag='SwapClassContra+CNN',
        weight_mode='consistency', ablation_mode=None, use_gat=False,
        lam_contra=HP['lam_contra'], lam_domain=0.0, lam_gate=HP['lam_gate'],
        contra_pos='class',
    ),
}


def _has_test(path):
    if not os.path.isfile(path) or os.path.getsize(path) < 32:
        return False
    text = open(path, encoding='utf-8', errors='ignore').read()
    return 'test_macro_f1=' in text and 'swap=' in text


class SwapCNN(nn.Module):
    def __init__(self, dims, n_cls, spec):
        super().__init__()
        self.front_ver = 'v2'
        self.front = FrontV2(dims, d_model=base.D_MODEL, gat_heads=1, dropout=0.1)
        self.front.use_gat = bool(spec['use_gat'])
        self.front.weight_mode = spec['weight_mode']
        self.front.contra_pos = spec.get('contra_pos', 'xdev')
        if 'use_node_mask' in spec:
            self.front.use_node_mask = bool(spec['use_node_mask'])
        if spec['ablation_mode']:
            self.front.ablation_mode = spec['ablation_mode']
        self.front.set_gate_target(float(HP['gate_target']))
        if spec['ablation_mode'] == 'cons':
            self.front.gate_target = float(HP['gate_target'])
        self.backbone = base.CNNBackbone(self.front.out_dim, base.D_HIDDEN)
        self.head = base.DualPool(self.backbone.out_dim, n_cls)

    def forward(self, x, return_aux=False):
        seq, aux = self.front(x, return_features=True)
        logits = self.head(self.backbone(seq))
        return (logits, aux) if return_aux else logits


def run_swap(name):
    if name not in SWAPS:
        raise SystemExit(f'未知替换 {name}，可选 {tuple(SWAPS)}')
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_num_threads(4)
    spec = dict(SWAPS[name])
    ckpt = os.path.join(OUT_DIR, f'best_{name}.pth')
    res = os.path.join(OUT_DIR, f'{name}_results.txt')
    if _has_test(res) and os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1000:
        print(f'skip {name}', flush=True)
        return
    pack = base.load_pack()
    train_ds, val_ds, test_ds = pack['train'], pack['val'], pack['test']
    dims, n_cls, cw = list(pack['dims']), pack['n_cls'], pack['cw']
    gamma = float(HP['gate_target'])
    print(
        f'SWAP_START {name} gat={int(spec["use_gat"])} w={spec["weight_mode"]} '
        f'mode={spec["ablation_mode"]} λc={spec["lam_contra"]} '
        f'λd={spec["lam_domain"]} λg={spec["lam_gate"]} γ={gamma} protocol=FULL',
        flush=True,
    )
    base.set_seed(base.SEED)
    counts = np.maximum(np.bincount(train_ds.labels, minlength=n_cls).astype(np.float64), 1)
    sw = 1.0 / np.sqrt(counts[train_ds.labels])
    sw /= sw.mean()
    gen = torch.Generator().manual_seed(base.SEED)
    sampler = WeightedRandomSampler(torch.DoubleTensor(sw), len(train_ds), True, generator=gen)
    tr = DataLoader(train_ds, base.BATCH, sampler=sampler, num_workers=0, pin_memory=True)
    va = DataLoader(val_ds, base.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    te = DataLoader(test_ds, base.BATCH, shuffle=False, num_workers=0, pin_memory=True)
    model = SwapCNN(dims, n_cls, spec).to(base.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.5, 3)
    n_p = sum(t.numel() for t in model.parameters()) / 1e6
    print(f'model={spec["tag"]}  params={n_p:.3f}M  save={ckpt}', flush=True)
    best_f1, best_ep, bad, last_gate = -1.0, -1, 0, 0.0
    for ep in range(1, base.EPOCHS + 1):
        model.train()
        losses, f1s, gates = [], [], []
        alpha = base.dann_alpha(ep)
        for x, y in tqdm(tr, desc=f'{name} ep{ep}', leave=False):
            x, y = x.to(base.DEVICE), y.to(base.DEVICE)
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
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            f1s.append(base.f1_score(
                y.cpu().numpy(), logits.argmax(1).cpu().numpy(),
                average='macro', zero_division=0))
            gates.append(float(aux['gate'].detach().mean()))
        vm = base.evaluate(model, va)
        sch.step(vm['f1'])
        last_gate = float(np.mean(gates)) if gates else 0.0
        print(f'Ep{ep:02d} loss={np.mean(losses):.4f} trF1={np.mean(f1s):.4f} '
              f'valF1={vm["f1"]:.4f} valAcc={vm["acc"]:.4f} gate={last_gate:.3f}',
              flush=True)
        if vm['f1'] > best_f1 + 1e-4:
            best_f1, best_ep, bad = vm['f1'], ep, 0
            torch.save({
                'model_state': model.state_dict(),
                'best_epoch': best_ep, 'val_f1': best_f1,
                'swap': name, 'dims': dims, 'n_cls': n_cls, 'seed': base.SEED,
            }, ckpt)
            print(f'  >> saved {ckpt}', flush=True)
        else:
            bad += 1
            if bad >= base.PATIENCE:
                print('early stop', flush=True)
                break
    raw = torch.load(ckpt, map_location=base.DEVICE, weights_only=False)
    model.load_state_dict(raw['model_state'])
    tm = base.evaluate(model, te)
    labels, pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in te:
            pred.extend(model(x.to(base.DEVICE)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    names_cls = list(pack['le'].classes_)
    report = classification_report(
        labels, pred, target_names=names_cls, labels=list(range(n_cls)),
        digits=4, zero_division=0)
    text = (
        f'model={spec["tag"]}\n'
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
        f'contra_temp={HP["contra_temp"]:.6f}\n'
        f'gate_target={gamma:.6f}\n'
        f'gate_mean={last_gate:.6f}\n'
        f'swap={name}\n'
        f'protocol=full-path all devices\n'
        f'gat={int(spec["use_gat"])}\n'
        f'weight_mode={spec["weight_mode"]}\n'
        f'ablation_mode={spec["ablation_mode"]}\n\n'
        f'===== TEST =====\n{report}\n'
    )
    with open(res, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'TEST Acc={tm["acc"]:.4f} F1={tm["f1"]:.4f} best_ep={best_ep}', flush=True)
    print(f'SWAP_DONE {name}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--swap', choices=list(SWAPS), required=True)
    args = p.parse_args()
    run_swap(args.swap)
