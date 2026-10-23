                       
from __future__ import annotations

import argparse
import os

import torch.nn as nn

import _run_cnn as base
from activity_gated_gat_v2 import ActivityGatedGAT as FrontV2
from _run_cnn import CNNBackbone, DualPoolHead

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


def build_cls(spec):
    class SwapCNN(nn.Module):
        def __init__(self, device_dims, num_classes, use_front=True,
                     d_model=16, d_hidden=32, dropout=0.1, front_ver='v2'):
            super().__init__()
            del use_front
            self.front_ver = 'v2'
            self.front = FrontV2(device_dims, d_model=d_model, gat_heads=1, dropout=dropout)
            self.front.use_gat = bool(spec['use_gat'])
            self.front.weight_mode = spec['weight_mode']
            self.front.contra_pos = spec.get('contra_pos', 'xdev')
            if spec['ablation_mode']:
                self.front.ablation_mode = spec['ablation_mode']
            self.backbone = CNNBackbone(self.front.out_dim, d_hidden=d_hidden)
            self.head = DualPoolHead(self.backbone.out_dim, num_classes)

        def forward(self, x, return_aux=False):
            if return_aux:
                seq, aux = self.front(x, return_features=True)
            else:
                seq, aux = self.front(x), None
            logits = self.head(self.backbone(seq))
            return (logits, aux) if return_aux else logits

    SwapCNN.__name__ = 'SwapCNN_' + spec['tag']
    return SwapCNN


def run_swap(name):
    if name not in SWAPS:
        raise SystemExit(f'未知替换 {name}，可选 {tuple(SWAPS)}')
    os.makedirs(OUT_DIR, exist_ok=True)
    spec = dict(SWAPS[name])
    ckpt = os.path.join(OUT_DIR, f'best_{name}.pth')
    res = os.path.join(OUT_DIR, f'{name}_results.txt')
    if _has_test(res) and os.path.isfile(ckpt) and os.path.getsize(ckpt) > 1000:
        print(f'skip {name}', flush=True)
        return
    pack = base.load_datasets()
    print(
        f'SWAP_START {name} gat={int(spec["use_gat"])} w={spec["weight_mode"]} '
        f'mode={spec["ablation_mode"]} λc={spec["lam_contra"]} '
        f'λd={spec["lam_domain"]} λg={spec["lam_gate"]} protocol=FULL',
        flush=True,
    )
    base.fit_model(
        use_front=True, front_ver='v2',
        train_ds=pack['train_ds'], val_ds=pack['val_ds'], test_ds=pack['test_ds'],
        n_cls=pack['n_cls'], label_encoder=pack['label_encoder'],
        sample_weight=pack['sample_weight'], class_weight=pack['class_weight'],
        ckpt_path=ckpt, result_path=res, cm_path=None, tag=spec['tag'],
        model_cls=build_cls(spec), seed=42,
        lam_contra=spec['lam_contra'], lam_domain=spec['lam_domain'],
        lam_gate=spec['lam_gate'],
        contra_temp=HP['contra_temp'], gate_target=HP['gate_target'],
    )
    with open(res, 'a', encoding='utf-8') as f:
        f.write(f'swap={name}\n')
        f.write('protocol=full-path all devices\n')
        f.write(f'gat={int(spec["use_gat"])}\n')
        f.write(f'weight_mode={spec["weight_mode"]}\n')
        f.write(f'ablation_mode={spec["ablation_mode"]}\n')
    print(f'SWAP_DONE {name}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--swap', choices=list(SWAPS), required=True)
    args = p.parse_args()
    run_swap(args.swap)
