# CAPE

Code accompanying “CAPE: Feature Alignment Middleware for Cross-Device Attack Detection in Heterogeneous Networks”.

## Installation

Use Python 3.10 or newer. Install PyTorch appropriate for your hardware, then run `pip install -r requirements.txt`.

## Layout

- `nba/`: N-BaIoT middleware, five basic detectors, advanced detector adapters, missing-input experiments, ablations and feature diagnostics.
- `ton/`: ToN-IoT middleware, detector adapters, missing-input experiments, ablations and sensitivity sweep.
- `SOURCE_MANIFEST.txt`: provenance of retained source files.

Only source code and documentation are included. Datasets, weights, predictions, experiment outputs, caches, logs, reports and historical result-selection/rerun scripts are excluded. Original experiment directories are preserved separately.

## Running

From `nba/`, use `python run_kdrop_train_test9.py --help` and `python run_paper_adv.py --help` to inspect the training options. The five basic detector entry points are `run_cnn_align.py`, `run_lstm_align.py`, `run_mlp_align.py`, `run_tcn_align.py`, and `run_transformer_align.py`.

From `ton/`, the CNN LODO entry point is `python _run_cnn_lodo.py`; advanced adapters are in `_run_paper_adapt.py`. Ablations are in `_run_ablation_swap.py` and `_run_ablation_swap_lodo.py` in both folders. Inspect the main block before launching: these research scripts may execute long experiment grids.

