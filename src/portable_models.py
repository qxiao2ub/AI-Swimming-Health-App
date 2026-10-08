"""Zero-PyTorch, zero-scikit-learn inference for the ORIGINAL trained models.

The original Colab-trained random forest, batch-normalized MLP, and CNN/BiGRU
were exported to NumPy arrays offline. This module implements equivalent
forward passes with only NumPy. Training still uses the original notebook.

Author: Kevin Sun | Mentor: Dr. Qingyang Xiao.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    return prob / prob.sum(axis=1, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipped to avoid over/underflow on noisy real-world sensor inputs.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -75, 75)))


def _linear(x: np.ndarray, p: dict[str, np.ndarray], stem: str) -> np.ndarray:
    return x @ p[stem + '__weight'].T + p[stem + '__bias']


def _batch_norm(x: np.ndarray, p: dict[str, np.ndarray], stem: str) -> np.ndarray:
    variance = p[stem + '__running_var']
    scale = p[stem + '__weight'] / np.sqrt(variance + np.float32(1e-5))
    offset = p[stem + '__bias'] - p[stem + '__running_mean'] * scale
    return x * scale + offset


def _conv1d(x: np.ndarray, p: dict[str, np.ndarray], stem: str) -> np.ndarray:
    """Input/output are (batch,time,channels). PyTorch Conv1d cross-correlation."""
    weight = p[stem + '__weight']
    bias = p[stem + '__bias']
    kernel = weight.shape[2]
    xp = np.pad(x, ((0, 0), (kernel // 2, kernel // 2), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(xp, window_shape=kernel, axis=1)
    # windows: batch,time,input_channels,kernel
    return np.einsum('btik,oik->bto', windows, weight, optimize=True) + bias


def _gru_direction(x: np.ndarray, p: dict[str, np.ndarray], suffix: str) -> np.ndarray:
    """PyTorch GRU gate convention: reset, update, new; returns every hidden state."""
    w_ih = p['gru__weight_ih_l0' + suffix]
    w_hh = p['gru__weight_hh_l0' + suffix]
    b_ih = p['gru__bias_ih_l0' + suffix]
    b_hh = p['gru__bias_hh_l0' + suffix]
    batch, length, _ = x.shape
    hidden_size = w_hh.shape[1]
    h = np.zeros((batch, hidden_size), dtype=np.float32)
    out = np.zeros((batch, length, hidden_size), dtype=np.float32)
    order = range(length - 1, -1, -1) if suffix else range(length)
    # Matrix multiplication for all timesteps of the input is done once.
    gates_x = x @ w_ih.T + b_ih
    for step in order:
        gi = gates_x[:, step, :]
        gh = h @ w_hh.T + b_hh
        i_r, i_z, i_n = np.split(gi, 3, axis=1)
        h_r, h_z, h_n = np.split(gh, 3, axis=1)
        reset = _sigmoid(i_r + h_r)
        update = _sigmoid(i_z + h_z)
        candidate = np.tanh(i_n + reset * h_n)
        h = (1 - update) * candidate + update * h
        out[:, step, :] = h
    return out


@dataclass
class PortableModelBundle:
    class_names: list[str]
    feature_columns: list[str]
    ensemble_weights: np.ndarray
    sequence_mean: np.ndarray
    sequence_std: np.ndarray
    model_metrics: dict
    training_source: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    forest: dict[str, np.ndarray]
    dnn: dict[str, np.ndarray]
    seq: dict[str, np.ndarray]
    q_table: np.ndarray

    def predict_probabilities(self, features: np.ndarray, sequences: np.ndarray) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        sequences = np.asarray(sequences, dtype=np.float32)
        if len(features) == 0:
            raise ValueError('At least one motion window is required.')
        if features.shape[1] != len(self.feature_columns) or sequences.shape[2] != 8:
            raise ValueError('Unexpected feature/sequence shape; expected the source notebook schema.')
        probs_rf = self._forest_predict(features)
        scaled = (features - self.scaler_mean) / self.scaler_scale
        probs_dnn = self._dnn_predict(scaled)
        normalized = (sequences - self.sequence_mean) / self.sequence_std
        # Bound memory use for longer uploaded workouts; the GRU works on batches.
        parts = []
        for start in range(0, len(normalized), 64):
            parts.append(self._sequence_predict(normalized[start:start + 64]))
        probs_seq = np.vstack(parts)
        combined = (self.ensemble_weights[0] * probs_rf
                    + self.ensemble_weights[1] * probs_dnn
                    + self.ensemble_weights[2] * probs_seq)
        return {
            'random_forest': probs_rf,
            'feature_dnn': probs_dnn,
            'cnn_bigru': probs_seq,
            'ensemble': combined,
        }

    def _forest_predict(self, x: np.ndarray) -> np.ndarray:
        forest = self.forest
        ntrees = forest['left'].shape[0]
        predictions = np.zeros((len(x), len(self.class_names)), dtype=np.float64)
        rows = np.arange(len(x))
        for i in range(ntrees):
            pos = np.zeros(len(x), dtype=np.int32)
            features_i = forest['features'][i]
            left_i, right_i = forest['left'][i], forest['right'][i]
            threshold_i = forest['thresholds'][i]
            while True:
                active = features_i[pos] >= 0
                if not active.any():
                    break
                ix = rows[active]
                nodes = pos[active]
                goes_left = x[ix, features_i[nodes]] <= threshold_i[nodes]
                pos[active] = np.where(goes_left, left_i[nodes], right_i[nodes])
            predictions += forest['value'][i, pos]
        return predictions / ntrees

    def _dnn_predict(self, x: np.ndarray) -> np.ndarray:
        p = self.dnn
        x = _relu(_batch_norm(_linear(x, p, 'network__0'), p, 'network__1'))
        x = _relu(_batch_norm(_linear(x, p, 'network__4'), p, 'network__5'))
        x = _relu(_linear(x, p, 'network__8'))
        return _softmax(_linear(x, p, 'network__10'))

    def _sequence_predict(self, x: np.ndarray) -> np.ndarray:
        p = self.seq
        x = _relu(_batch_norm(_conv1d(x, p, 'conv__0'), p, 'conv__1'))
        x = np.maximum(x[:, 0::2, :], x[:, 1::2, :])
        x = _relu(_batch_norm(_conv1d(x, p, 'conv__4'), p, 'conv__5'))
        forward = _gru_direction(x, p, '')
        backward = _gru_direction(x, p, '_reverse')
        features = np.concatenate([forward, backward], axis=2).mean(axis=1)
        features = _relu(_linear(features, p, 'head__0'))
        return _softmax(_linear(features, p, 'head__3'))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    # allow_pickle=False: weights are data, not serialized executable objects.
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def load_portable_models(model_directory: str | Path) -> PortableModelBundle:
    folder = Path(model_directory)
    meta = json.loads((folder / 'model_metadata.json').read_text(encoding='utf-8'))
    scaler = _load_npz(folder / 'feature_scaler_portable.npz')
    return PortableModelBundle(
        class_names=list(meta['class_names']),
        feature_columns=list(meta['feature_columns']),
        ensemble_weights=np.array(meta['ensemble_weights'], dtype=np.float64),
        sequence_mean=np.array(meta['sequence_mean'], dtype=np.float32).reshape(1, 1, -1),
        sequence_std=np.array(meta['sequence_std'], dtype=np.float32).reshape(1, 1, -1),
        model_metrics=meta.get('model_metrics', {}),
        training_source=meta.get('training_source', 'synthetic demonstration data'),
        scaler_mean=scaler['mean'],
        scaler_scale=scaler['scale'],
        forest=_load_npz(folder / 'random_forest_portable.npz'),
        dnn=_load_npz(folder / 'feature_dnn_portable.npz'),
        seq=_load_npz(folder / 'cnn_bigru_portable.npz'),
        q_table=np.load(folder / 'rl_q_table.npy', allow_pickle=False),
    )
