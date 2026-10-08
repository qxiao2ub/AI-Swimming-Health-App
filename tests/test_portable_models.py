"""Check original trained weights remain mathematically equivalent after export."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.portable_models import load_portable_models
from src.swim_ai import build_window_bundle, clean_watch_data

ROOT = Path(__file__).resolve().parents[1]


def test_original_torch_and_sklearn_parity():
    """Frozen predictions generated from original Torch/sklearn artifact code."""
    model = load_portable_models(ROOT / 'models')
    sample = pd.read_csv(ROOT / 'sample_data' / 'kevin_demo_watch_session.csv')
    windows = build_window_bundle(clean_watch_data(sample))
    assert windows['features'].shape == (47, 94)
    features = windows['features'].reindex(columns=model.feature_columns).to_numpy(dtype=np.float32)
    current = model.predict_probabilities(features, windows['sequences'])
    with np.load(ROOT / 'tests' / 'known_torch_reference.npz', allow_pickle=False) as oracle:
        for key, ref in [('rf', 'random_forest'), ('dnn', 'feature_dnn'), ('cnn', 'cnn_bigru')]:
            assert np.max(np.abs(oracle[key] - current[ref])) < 1e-5, key
            assert np.array_equal(oracle[key].argmax(axis=1), current[ref].argmax(axis=1))
    for probs in current.values():
        assert probs.shape == (47, 5)
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
