                       
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

import run_cnn_align as cnn
import run_kdrop_train_test9 as kdrop
import run_paper_adv as adv

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(ROOT, 'feature_cluster_front.png')
OUT_PDF = os.path.join(ROOT, 'feature_cluster_front.pdf')
PER_CLASS = 70
SEED = 0

METHODS = (
    ('CoLD', 'cold',
     'paper_adv_ckpts/CoLD/best_k1_danmini_raw.pth',
     'paper_adv_axis_ckpts/CoLD/best_k1_danmini_front.pth'),
    ('tFusion', 'tfusion',
     'paper_adv_ckpts/tFusion/best_k1_danmini_raw.pth',
     'paper_adv_ckpts/tFusion/best_k1_danmini_front.pth'),
    ('TIPSO-Attn', 'tipsogan',
     'paper_adv_ckpts/TIPSO-Attn/best_k1_danmini_raw.pth',
     'paper_adv_axis_ckpts/TIPSO-Attn/best_k1_danmini_front.pth'),
    ('FeCo', 'feco',
     'paper_adv_axis_feco_aoc_dev_ckpts/FeCo/best_k1_danmini_raw.pth',
     'paper_adv_axis_feco_aoc_dev_ckpts/FeCo/best_k1_danmini_front.pth'),
    ('AOC-IDS', 'aocids',
     'paper_adv_axis_feco_aoc_dev_ckpts/AOC-IDS/best_k1_danmini_raw.pth',
     'paper_adv_axis_feco_aoc_dev_ckpts/AOC-IDS/best_k1_danmini_front.pth'),
)

COLORS = [
    '#4D4D4D',
    '#E69F00', '#D55E00', '#F0E442', '#CC79A7', '#882255',
    '#0072B2', '#56B4E9', '#009E73', '#000000', '#999933',
]


def stratified_idx(labels, per_class, seed):
    rng = np.random.RandomState(seed)
    picks = []
    for lab in np.unique(labels):
        pool = np.where(labels == lab)[0]
        take = min(per_class, len(pool))
        picks.append(rng.choice(pool, take, replace=False))
    return np.sort(np.concatenate(picks))


def load_model(path, kind, dims, n_cls):
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = blob['model_state']
    if kind == 'cnn':
        model = cnn.PlugCNN(dims, n_cls, use_front=True).to(cnn.DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model
    use_front = any(str(k).startswith('front.') for k in state)
    square = state.get('front.device_projectors.0.0.weight')
    axis_arch = (
        use_front and square is not None and tuple(square.shape)[0] == tuple(square.shape)[1]
        and 'front.device_projectors.0.3.weight' not in state
    )
    adv.RUN_D_MODEL = int(blob['d_model']) if use_front and blob.get('d_model') else cnn.D_MODEL
    adv.RUN_AXIS = bool(blob.get('axis')) or axis_arch
    adv.RUN_LAM_DOMAIN = 0.0 if adv.RUN_AXIS else None
    model = adv.PlugPaper(dims, n_cls, use_front=use_front, kind=kind).to(cnn.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def embed(model, samples, layer):
    outs = []
    for start in range(0, len(samples), 256):
        x = torch.as_tensor(samples[start:start + 256], device=cnn.DEVICE)
        if layer == 'front':
            _, aux = model.front(x, return_features=True)
            h_prime = aux['h_prime']
                                                                                    
            vec = h_prime.mean(dim=tuple(range(1, h_prime.ndim - 1)))
        elif layer == 'seq':
            if model.front is None:
                seq = x
            else:
                seq, _ = model.front(x, return_features=True)
            vec = model.backbone(seq).mean(1)
        else:
            if model.front is None:
                seq = x
            else:
                seq, _ = model.front(x, return_features=True)
            h = model.backbone(seq)
            if getattr(model, 'temporal', None) is not None:
                h = model.temporal(h)
            weight = torch.softmax(model.head.attn(h).squeeze(-1), dim=-1)
            pooled = (h * weight.unsqueeze(-1)).sum(1)
            vec = torch.cat([pooled, h.mean(1)], dim=-1)
        outs.append(vec.cpu().numpy())
    return np.concatenate(outs, axis=0)


def tightness(features, labels):
    z = normalize(features)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return float(silhouette_score(z, labels, metric='euclidean'))


def spread_ratio(features, labels):
    pass
    z = normalize(features)
    centers, intra = [], []
    for lab in np.unique(labels):
        pts = z[labels == lab]
        center = pts.mean(axis=0)
        centers.append(center)
        intra.append(np.linalg.norm(pts - center, axis=1).mean())
    centers = np.stack(centers)
    inter = [
        np.linalg.norm(centers[i] - centers[j])
        for i in range(len(centers)) for j in range(i + 1, len(centers))
    ]
    return float(np.mean(intra) / max(np.mean(inter), 1e-8))


def project(features):
    from sklearn.decomposition import PCA
    z = normalize(features)
    rank = min(32, z.shape[0] - 1, z.shape[1])
    z = PCA(n_components=rank, random_state=SEED).fit_transform(z)
    return TSNE(
        n_components=2, perplexity=30, init='pca', learning_rate='auto',
        random_state=SEED,
    ).fit_transform(z)


def family_of(names, labels):
    out = np.empty(len(labels), np.int64)
    for lab, name in enumerate(names):
        key = name.split()[0]
        out[labels == lab] = {'benign': 0, 'gafgyt': 1, 'mirai': 2}[key]
    return out


def knn_acc(features, labels, k=5):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    z = normalize(features)
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), z, labels, cv=5)
    return float(score.mean())


def draw(panels, labels, names, out_png=OUT_PNG, out_pdf=OUT_PDF, suptitle=None):
    pass
    n = len(panels)
    if n <= 2:
        fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.6), dpi=170)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.2), dpi=170)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, xy, fine) in zip(axes, panels):
        for lab in range(len(names)):
            m = labels == lab
            pts = xy[m]
            ax.scatter(pts[:, 0], pts[:, 1], s=9, c=COLORS[lab], alpha=0.28, linewidths=0, rasterized=True)
            center = pts.mean(axis=0)
            ax.scatter(
                [center[0]], [center[1]], s=86, c=COLORS[lab],
                edgecolors='white', linewidths=0.8, zorder=3,
            )
        ax.set_title(f'{title}    {fine:.2f}', fontsize=13, pad=6, color='#9A3412' if title == 'Ours' else '#1A1A1A')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#E2E2E2')
    handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS[i],
                   markeredgecolor='white', markersize=8, label=names[i])
        for i in range(len(names))
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        suptitle or 'Same test windows. Each panel is that method\'s own features. The number is the 11-class silhouette.',
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print('WROTE', out_png, flush=True)


def main():
    cache = os.path.join(ROOT, '_cluster_cache.npz')
    if os.path.exists(cache):
        saved = np.load(cache, allow_pickle=True)
        labels = saved['labels']
        names = [str(n) for n in saved['names']]
        panels = []
        for title in ('Front', 'CoLD', 'tFusion', 'TIPSO-Attn', 'FeCo', 'AOC-IDS'):
            feat = saved[f'{title}_backbone']
            fine = tightness(feat, labels)
            shown = 'Ours' if title == 'Front' else title
            panels.append((shown, project(feat), fine))
            print(f'{shown:12} silhouette={fine:.3f}', flush=True)
        draw(panels, labels, names)
        return

    pack = kdrop.load_pack()
    test = pack['test']
    labels_all = np.asarray(test.labels)
    idx = stratified_idx(labels_all, PER_CLASS, SEED)
    samples = np.asarray(test.samples)[idx]
    labels = labels_all[idx]
    names = [str(n).replace('_', ' ') for n in pack['le'].classes_]
    family = family_of(names, labels)
    print(f'samples={len(labels)} classes={names}', flush=True)

                                                                                 
    jobs = (
        ('Front', 'cnn', 'cnn_align_lodo_ckpts/best_lodo_danmini_front.pth'),
        ('CoLD', 'cold', 'paper_adv_ckpts/CoLD/best_k1_danmini_raw.pth'),
        ('tFusion', 'tfusion', 'paper_adv_ckpts/tFusion/best_k1_danmini_raw.pth'),
        ('TIPSO-Attn', 'tipsogan', 'paper_adv_ckpts/TIPSO-Attn/best_k1_danmini_raw.pth'),
        ('FeCo', 'feco', 'paper_adv_axis_feco_aoc_dev_ckpts/FeCo/best_k1_danmini_raw.pth'),
        ('AOC-IDS', 'aocids', 'paper_adv_axis_feco_aoc_dev_ckpts/AOC-IDS/best_k1_danmini_raw.pth'),
    )
    saved = {'labels': labels, 'family': family, 'names': np.array(names)}
    panels = []
    for title, kind, rel in jobs:
        model = load_model(os.path.join(ROOT, rel), kind, pack['dims'], pack['n_cls'])
        layers = ('backbone', 'seq')
        if model.front is not None:
            layers = layers + ('front',)
        for layer in layers:
            feat = embed(model, samples, layer)
            fine = tightness(feat, labels)
            coarse = tightness(feat, family)
            ratio = spread_ratio(feat, labels)
            acc = knn_acc(feat, labels)
            print(
                f'{title:12} {layer:9} fine={fine:.3f} family={coarse:.3f} '
                f'spread={ratio:.3f} knn={acc:.3f} dim={feat.shape[1]}',
                flush=True,
            )
            saved[f'{title}_{layer}'] = feat
            if layer == 'backbone':
                panels.append(('Ours' if title == 'Front' else title, feat, fine))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    np.savez_compressed(cache, **saved)
    drawn = []
    for title, feat, fine in panels:
        drawn.append((title, project(feat), fine))
    draw(drawn, labels, names)


if __name__ == '__main__':
    main()
