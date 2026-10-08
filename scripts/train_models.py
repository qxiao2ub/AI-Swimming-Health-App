#!/usr/bin/env python
"""Train and export the app's model artifacts from structured watch data.

This script mirrors the attached notebook's grouped evaluation, Random Forest,
feature DNN, CNN-BiGRU, validation-weighted ensemble, and Q-learning policy.
It is not executed by Streamlit because pretrained artifacts are committed.

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""

from __future__ import annotations

from pathlib import Path
import copy
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.swim_ai_torch import (  # noqa: E402
    DEFAULT_CONFIG,
    FeatureDNN,
    CNNBiGRU,
    KNOWN_STROKES,
    RL_ACTIONS,
    SEQUENCE_CHANNELS,
    build_window_bundle,
    clean_watch_data,
    generate_synthetic_watch_data,
    seed_everything,
)

MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


def make_loader(
    inputs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict_probabilities(
    model: nn.Module,
    inputs: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch.to(DEVICE))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(outputs)


def train_torch_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    epochs: int,
    learning_rate: float = 1e-3,
    patience: int = 4,
) -> tuple[nn.Module, pd.DataFrame]:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for batch, labels in train_loader:
            batch = batch.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch)
            train_correct += int((logits.argmax(1) == labels).sum().item())
            train_count += len(batch)

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        validation_count = 0
        with torch.inference_mode():
            for batch, labels in validation_loader:
                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)
                logits = model(batch)
                loss = criterion(logits, labels)
                validation_loss_sum += float(loss.item()) * len(batch)
                validation_correct += int((logits.argmax(1) == labels).sum().item())
                validation_count += len(batch)

        train_loss = train_loss_sum / max(train_count, 1)
        validation_loss = validation_loss_sum / max(validation_count, 1)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_correct / max(train_count, 1),
            "validation_accuracy": validation_correct / max(validation_count, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | train loss {train_loss:.4f} | "
            f"validation loss {validation_loss:.4f} | "
            f"validation accuracy {row['validation_accuracy']:.3f}"
        )

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def metric_summary(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, average="weighted", zero_division=0)
        ),
    }


class SwimCoachEnv:
    """Educational Markov decision process from the source notebook."""

    def __init__(self, seed: int = 42, horizon: int = 12):
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.state = (0, 1, 1)
        self.step_count = 0

    def reset(self) -> tuple[int, int, int]:
        self.state = (
            int(self.rng.integers(0, 2)),
            int(self.rng.integers(1, 4)),
            int(self.rng.integers(0, 3)),
        )
        self.step_count = 0
        return self.state

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool]:
        fatigue, heart_rate_zone, technique = self.state
        noise = int(self.rng.choice([-1, 0, 0, 0, 1]))

        if action == 0:
            fatigue -= 1
            heart_rate_zone -= 1
            technique += int(self.rng.random() < 0.25)
        elif action == 1:
            fatigue += int(self.rng.random() < 0.30)
            heart_rate_zone += noise
        elif action == 2:
            heart_rate_zone -= int(self.rng.random() < 0.55)
            technique += 1
            fatigue -= int(self.rng.random() < 0.25)
        elif action == 3:
            heart_rate_zone += 1
            fatigue += int(self.rng.random() < 0.70)
            technique -= int(fatigue >= 1 and self.rng.random() < 0.55)

        fatigue = int(np.clip(fatigue, 0, 2))
        heart_rate_zone = int(np.clip(heart_rate_zone, 0, 4))
        technique = int(np.clip(technique, 0, 2))
        next_state = (fatigue, heart_rate_zone, technique)

        training_stimulus = [0.0, 0.35, 0.85, 1.0, 0.25][heart_rate_zone]
        reward = 0.55 * technique - 0.75 * fatigue + training_stimulus
        if heart_rate_zone == 4 and fatigue == 2:
            reward -= 4.5
        if action == 0 and (fatigue == 2 or heart_rate_zone == 4):
            reward += 1.5
        if action == 2 and technique == 0:
            reward += 1.2
        if action == 3 and fatigue == 0 and heart_rate_zone <= 2 and technique == 2:
            reward += 0.7

        self.state = next_state
        self.step_count += 1
        return next_state, float(reward), self.step_count >= self.horizon


def train_q_learning(episodes: int = 3500, seed: int = 42) -> np.ndarray:
    environment = SwimCoachEnv(seed=seed)
    q_table = np.zeros((3, 5, 3, len(RL_ACTIONS)), dtype=np.float32)
    alpha = 0.14
    gamma = 0.94

    for episode in range(episodes):
        state = environment.reset()
        epsilon = max(0.04, 0.95 * (1 - episode / episodes))
        done = False
        while not done:
            if environment.rng.random() < epsilon:
                action = int(environment.rng.integers(len(RL_ACTIONS)))
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, done = environment.step(action)
            target = reward + (0 if done else gamma * float(np.max(q_table[next_state])))
            q_table[state + (action,)] += alpha * (target - q_table[state + (action,)])
            state = next_state
    return q_table


def create_grouped_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_class_ids = set(range(len(np.unique(labels))))
    for attempt in range(60):
        outer = GroupShuffleSplit(
            n_splits=1,
            test_size=0.22,
            random_state=seed + attempt,
        )
        train_validation, test = next(outer.split(features, labels, groups=groups))
        inner = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=seed + 100 + attempt,
        )
        train_relative, validation_relative = next(
            inner.split(
                features[train_validation],
                labels[train_validation],
                groups=groups[train_validation],
            )
        )
        train = train_validation[train_relative]
        validation = train_validation[validation_relative]
        if (
            set(np.unique(labels[train])) == all_class_ids
            and set(np.unique(labels[validation])) == all_class_ids
            and set(np.unique(labels[test])) == all_class_ids
        ):
            return train, validation, test
    raise RuntimeError("Unable to create grouped splits containing every class.")


def main() -> None:
    seed_everything(DEFAULT_CONFIG.random_seed)
    print("Generating structured watch training data...")
    watch_data = generate_synthetic_watch_data(DEFAULT_CONFIG, n_sessions=24)
    clean_data = clean_watch_data(watch_data, DEFAULT_CONFIG)
    training_bundle = build_window_bundle(clean_data, DEFAULT_CONFIG, require_labels=True)

    feature_frame = training_bundle["features"].copy()
    feature_columns = feature_frame.columns.tolist()
    features = feature_frame.to_numpy(dtype=np.float32)
    sequences = training_bundle["sequences"]
    text_labels = training_bundle["meta"]["true_label"].astype(str).to_numpy()
    groups = training_bundle["meta"]["session_id"].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(text_labels)
    class_names = label_encoder.classes_.tolist()
    assert set(class_names) == set(KNOWN_STROKES)

    train, validation, test = create_grouped_split(
        features,
        labels,
        groups,
        DEFAULT_CONFIG.random_seed,
    )
    print(
        f"Windows: train={len(train)}, validation={len(validation)}, test={len(test)}; "
        f"classes={class_names}"
    )

    random_forest = RandomForestClassifier(
        n_estimators=260,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=DEFAULT_CONFIG.random_seed,
    )
    random_forest.fit(features[train], labels[train])
    rf_validation_prob = random_forest.predict_proba(features[validation])
    rf_test_prob = random_forest.predict_proba(features[test])

    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(features[train]).astype(np.float32)
    validation_features = feature_scaler.transform(features[validation]).astype(np.float32)
    test_features = feature_scaler.transform(features[test]).astype(np.float32)

    feature_dnn = FeatureDNN(features.shape[1], len(class_names))
    feature_dnn, feature_history = train_torch_classifier(
        feature_dnn,
        make_loader(train_features, labels[train], 64, True),
        make_loader(validation_features, labels[validation], 64, False),
        epochs=12,
    )
    dnn_validation_prob = predict_probabilities(feature_dnn, validation_features)
    dnn_test_prob = predict_probabilities(feature_dnn, test_features)

    sequence_mean = sequences[train].mean(axis=(0, 1), keepdims=True)
    sequence_std = sequences[train].std(axis=(0, 1), keepdims=True) + 1e-6
    normalized_sequences = ((sequences - sequence_mean) / sequence_std).astype(np.float32)

    sequence_model = CNNBiGRU(len(SEQUENCE_CHANNELS), len(class_names))
    sequence_model, sequence_history = train_torch_classifier(
        sequence_model,
        make_loader(normalized_sequences[train], labels[train], 64, True),
        make_loader(normalized_sequences[validation], labels[validation], 64, False),
        epochs=7,
    )
    sequence_validation_prob = predict_probabilities(
        sequence_model,
        normalized_sequences[validation],
    )
    sequence_test_prob = predict_probabilities(sequence_model, normalized_sequences[test])

    validation_scores = np.asarray(
        [
            f1_score(
                labels[validation],
                rf_validation_prob.argmax(1),
                average="macro",
                zero_division=0,
            ),
            f1_score(
                labels[validation],
                dnn_validation_prob.argmax(1),
                average="macro",
                zero_division=0,
            ),
            f1_score(
                labels[validation],
                sequence_validation_prob.argmax(1),
                average="macro",
                zero_division=0,
            ),
        ],
        dtype=float,
    )
    ensemble_weights = validation_scores / validation_scores.sum()
    ensemble_test_prob = (
        ensemble_weights[0] * rf_test_prob
        + ensemble_weights[1] * dnn_test_prob
        + ensemble_weights[2] * sequence_test_prob
    )

    model_metrics = {
        "random_forest": metric_summary(labels[test], rf_test_prob.argmax(1)),
        "feature_dnn": metric_summary(labels[test], dnn_test_prob.argmax(1)),
        "cnn_bigru": metric_summary(labels[test], sequence_test_prob.argmax(1)),
        "ensemble": metric_summary(labels[test], ensemble_test_prob.argmax(1)),
    }
    print(json.dumps(model_metrics, indent=2))
    print("Ensemble weights:", ensemble_weights)

    q_table = train_q_learning(episodes=3500, seed=DEFAULT_CONFIG.random_seed)

    joblib.dump(random_forest, MODELS_DIR / "random_forest.joblib")
    joblib.dump(feature_scaler, MODELS_DIR / "feature_scaler.joblib")
    torch.save(feature_dnn.state_dict(), MODELS_DIR / "feature_dnn_state.pt")
    torch.save(sequence_model.state_dict(), MODELS_DIR / "cnn_bigru_state.pt")
    np.save(MODELS_DIR / "rl_q_table.npy", q_table)

    feature_history.to_csv(MODELS_DIR / "feature_dnn_training_history.csv", index=False)
    sequence_history.to_csv(MODELS_DIR / "cnn_bigru_training_history.csv", index=False)

    metadata = {
        "project": "Kevin Sun AI Swimming Health App",
        "author": "Kevin Sun",
        "mentor": "Dr. Qingyang Xiao",
        "class_names": class_names,
        "feature_columns": feature_columns,
        "sequence_channels": SEQUENCE_CHANNELS,
        "sequence_mean": sequence_mean.reshape(-1).tolist(),
        "sequence_std": sequence_std.reshape(-1).tolist(),
        "ensemble_weights": ensemble_weights.tolist(),
        "validation_macro_f1": validation_scores.tolist(),
        "model_metrics": model_metrics,
        "training_source": "structured synthetic smartwatch data generated by the attached notebook",
        "training_windows": int(len(features)),
        "training_sessions": int(pd.Series(groups).nunique()),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
