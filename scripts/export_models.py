#!/usr/bin/env python3
"""Export the original trained Torch/sklearn weights for NumPy-only Streamlit.

Run AFTER scripts/train_models.py. This script is not run in Streamlit Cloud.
Author Kevin Sun | Mentor Dr. Qingyang Xiao.
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / 'models'


def export() -> None:
    rf = joblib.load(MODELS / 'random_forest.joblib')
    scaler = joblib.load(MODELS / 'feature_scaler.joblib')
    trees = rf.estimators_
    n_classes = len(rf.classes_)
    max_nodes = max(t.tree_.node_count for t in trees)
    n_trees = len(trees)
    left = np.full((n_trees, max_nodes), -1, dtype=np.int32)
    right = np.full_like(left, -1)
    features = np.full_like(left, -2)
    thresholds = np.zeros((n_trees, max_nodes), dtype=np.float64)
    values = np.zeros((n_trees, max_nodes, n_classes), dtype=np.float32)
    for i, estimator in enumerate(trees):
        tree = estimator.tree_
        n = tree.node_count
        left[i, :n] = tree.children_left
        right[i, :n] = tree.children_right
        features[i, :n] = tree.feature
        thresholds[i, :n] = tree.threshold
        v = tree.value[:, 0, :]
        values[i, :n] = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-30)
    np.savez_compressed(MODELS / 'random_forest_portable.npz', left=left, right=right,
                        features=features, thresholds=thresholds, value=values)
    np.savez_compressed(MODELS / 'feature_scaler_portable.npz',
                        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32))
    for model in ('feature_dnn', 'cnn_bigru'):
        state = torch.load(MODELS / f'{model}_state.pt', map_location='cpu', weights_only=True)
        arrays = {k.replace('.', '__'): value.detach().cpu().numpy()
                  for k, value in state.items() if not k.endswith('num_batches_tracked')}
        np.savez_compressed(MODELS / f'{model}_portable.npz', **arrays)
    print('Exported portable forest, DNN, CNN-BiGRU and scaler. Commit *.npz to GitHub.')


if __name__ == '__main__':
    export()
