"""Core analytics for the Kevin Sun AI Swimming Health App.

This module adapts the attached Google Colab notebook into reusable functions
for Streamlit deployment. It keeps the notebook's smartwatch ingestion,
signal cleaning, feature engineering, three-model ensemble, health summary,
and safety-constrained reinforcement-learning recommendation.

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import copy
import json
import math
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

AUTHOR_NAME = "Kevin Sun"
MENTOR_NAME = "Dr. Qingyang Xiao"
PROJECT_NAME = "Kevin Sun AI Swimming Health App"


@dataclass(frozen=True)
class AppConfig:
    """Signal and safety settings shared by the notebook and web app."""

    random_seed: int = 42
    sample_rate_hz: int = 20
    window_seconds: float = 4.0
    step_seconds: float = 2.0
    minimum_label_purity: float = 0.80
    high_hr_fraction_of_estimated_max: float = 0.95
    spo2_sensor_review_threshold_pct: float = 94.0
    sustained_alert_seconds: int = 10


@dataclass(frozen=True)
class AthleteProfile:
    """Editable swimmer profile used by the educational analytics."""

    athlete_id: str = "swimmer"
    age_years: int = 17
    mass_kg: float = 65.0
    resting_hr_bpm: float = 60.0
    pool_length_m: float = 25.0


@dataclass
class ModelBundle:
    """Loaded model and preprocessing artifacts for fast Streamlit inference."""

    rf_model: RandomForestClassifier
    feature_scaler: StandardScaler
    feature_dnn: nn.Module
    sequence_model: nn.Module
    sequence_mean: np.ndarray
    sequence_std: np.ndarray
    class_names: list[str]
    ensemble_weights: np.ndarray
    feature_columns: list[str]
    q_table: np.ndarray
    model_metrics: dict[str, Any]
    training_source: str


@dataclass
class AnalysisResult:
    """Container returned by an end-to-end session analysis."""

    report: dict[str, Any]
    windows: pd.DataFrame
    clean_data: pd.DataFrame
    session_data: pd.DataFrame
    quality_table: pd.DataFrame


DEFAULT_CONFIG = AppConfig()

COLUMN_ALIASES = {
    "time": "timestamp",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "athlete": "athlete_id",
    "user_id": "athlete_id",
    "session": "session_id",
    "workout_id": "session_id",
    "accelerometer_x": "acc_x",
    "accelerometer_y": "acc_y",
    "accelerometer_z": "acc_z",
    "ax": "acc_x",
    "ay": "acc_y",
    "az": "acc_z",
    "gyroscope_x": "gyro_x",
    "gyroscope_y": "gyro_y",
    "gyroscope_z": "gyro_z",
    "gx": "gyro_x",
    "gy": "gyro_y",
    "gz": "gyro_z",
    "heart_rate": "heart_rate_bpm",
    "hr": "heart_rate_bpm",
    "spo2": "spo2_pct",
    "oxygen_saturation": "spo2_pct",
    "distance": "distance_m",
    "lap": "lap_id",
    "stroke": "stroke_label",
    "style": "stroke_label",
}

MOTION_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
SEQUENCE_CHANNELS = MOTION_COLUMNS + ["acc_mag", "gyro_mag"]
KNOWN_STROKES = ["freestyle", "backstroke", "breaststroke", "butterfly", "rest"]

STROKE_PROFILES = {
    "freestyle": {
        "frequency_hz": 0.78,
        "speed_mps": 1.25,
        "intensity": 0.72,
        "acc_amp": (0.42, 0.30, 0.48),
        "gyro_amp": (115, 80, 165),
        "phase": (0.0, 1.1, 2.0),
    },
    "backstroke": {
        "frequency_hz": 0.68,
        "speed_mps": 1.08,
        "intensity": 0.68,
        "acc_amp": (0.32, 0.46, 0.35),
        "gyro_amp": (90, 150, 120),
        "phase": (1.6, 0.3, 2.4),
    },
    "breaststroke": {
        "frequency_hz": 0.50,
        "speed_mps": 0.88,
        "intensity": 0.63,
        "acc_amp": (0.58, 0.24, 0.28),
        "gyro_amp": (75, 95, 65),
        "phase": (0.2, 2.0, 1.0),
    },
    "butterfly": {
        "frequency_hz": 0.88,
        "speed_mps": 1.30,
        "intensity": 0.90,
        "acc_amp": (0.70, 0.44, 0.72),
        "gyro_amp": (175, 135, 190),
        "phase": (0.0, 0.6, 1.2),
    },
    "rest": {
        "frequency_hz": 0.10,
        "speed_mps": 0.0,
        "intensity": 0.22,
        "acc_amp": (0.04, 0.04, 0.05),
        "gyro_amp": (8, 8, 8),
        "phase": (0.0, 0.0, 0.0),
    },
}

RL_ACTIONS = ["recover_easy", "hold_steady", "technique_drill", "brief_pace_increase"]
ACTION_TEXT = {
    "recover_easy": (
        "Use easy swimming or a rest interval; prioritize controlled breathing "
        "and stop for concerning symptoms."
    ),
    "hold_steady": "Maintain the current sustainable effort while monitoring technique and comfort.",
    "technique_drill": (
        "Reduce speed slightly and focus on alignment, timing, and a consistent stroke rhythm."
    ),
    "brief_pace_increase": (
        "Try a short, controlled pace increase only when comfortable, supervised, "
        "and free of warning signs."
    ),
}


def seed_everything(seed: int = 42) -> None:
    """Make model training and synthetic demonstrations reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 2))
    except Exception:
        pass


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map common smartwatch export headings to the app's canonical schema."""

    renamed = {
        column: COLUMN_ALIASES.get(str(column).strip().lower(), str(column).strip().lower())
        for column in df.columns
    }
    return df.rename(columns=renamed)


def read_watch_file(file_obj: Any, filename: str | None = None) -> pd.DataFrame:
    """Read a Streamlit upload, bytes object, or binary file as CSV/JSON/JSONL."""

    inferred_name = filename or getattr(file_obj, "name", "uploaded.csv")
    suffix = Path(inferred_name).suffix.lower()

    if hasattr(file_obj, "getvalue"):
        payload = file_obj.getvalue()
    elif isinstance(file_obj, (bytes, bytearray)):
        payload = bytes(file_obj)
    elif hasattr(file_obj, "read"):
        payload = file_obj.read()
    else:
        raise TypeError("Unsupported upload type. Provide a CSV, JSON, or JSONL file.")

    buffer = BytesIO(payload)
    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".json", ".jsonl"}:
        try:
            return pd.read_json(buffer, lines=(suffix == ".jsonl"))
        except ValueError:
            buffer.seek(0)
            return pd.read_json(buffer)
    raise ValueError("Upload a .csv, .json, or .jsonl smartwatch export.")


def generate_synthetic_watch_data(
    config: AppConfig = DEFAULT_CONFIG,
    n_sessions: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Create structured demo watch signals with five recognizable activities."""

    if n_sessions < 1:
        raise ValueError("n_sessions must be at least 1.")

    rng = np.random.default_rng(config.random_seed if seed is None else seed)
    fs = config.sample_rate_hz
    frames: list[pd.DataFrame] = []

    base_plan = [
        ("freestyle", 18),
        ("rest", 6),
        ("backstroke", 18),
        ("rest", 6),
        ("breaststroke", 18),
        ("rest", 6),
        ("butterfly", 16),
        ("rest", 8),
    ]

    alternate_plan = [
        ("backstroke", 17),
        ("rest", 6),
        ("freestyle", 19),
        ("rest", 6),
        ("butterfly", 15),
        ("rest", 7),
        ("breaststroke", 19),
        ("rest", 7),
    ]

    for session_idx in range(n_sessions):
        athlete_idx = session_idx % 6
        athlete_id = f"athlete_{athlete_idx + 1:02d}"
        session_id = f"session_{session_idx + 1:03d}"
        start_time = pd.Timestamp("2026-01-05 07:00:00") + pd.Timedelta(days=session_idx)
        athlete_factor = rng.normal(1.0, 0.055)
        technique_factor = np.clip(rng.normal(1.0, 0.07), 0.82, 1.17)
        resting_hr = np.clip(56 + athlete_idx * 2 + rng.normal(0, 2), 50, 72)
        max_hr_est = 208 - 0.7 * (16 + athlete_idx % 3)
        current_hr = resting_hr + rng.uniform(10, 18)
        elapsed_samples = 0
        cumulative_distance = 0.0
        cumulative_strokes = 0
        swim_segments = alternate_plan if session_idx % 2 else base_plan

        for stroke, duration_s in swim_segments:
            profile = STROKE_PROFILES[stroke]
            sample_count = int(duration_s * fs)
            local_t = np.arange(sample_count, dtype=float) / fs
            global_t = (elapsed_samples + np.arange(sample_count)) / fs
            frequency = max(
                0.05,
                profile["frequency_hz"] * rng.normal(1.0, 0.04) * technique_factor,
            )
            phase0 = rng.uniform(0, 2 * np.pi)
            theta = 2 * np.pi * frequency * local_t + phase0
            second = 2 * theta + 0.35
            fatigue_progress = np.linspace(0, 1, sample_count)
            fatigue_attenuation = 1 - 0.07 * fatigue_progress * (profile["intensity"] > 0.7)

            acc_amp = np.asarray(profile["acc_amp"]) * athlete_factor * fatigue_attenuation[:, None]
            gyro_amp = np.asarray(profile["gyro_amp"]) * athlete_factor * fatigue_attenuation[:, None]
            phases = np.asarray(profile["phase"])

            acc_x = acc_amp[:, 0] * np.sin(theta + phases[0]) + 0.12 * acc_amp[:, 0] * np.sin(second)
            acc_y = acc_amp[:, 1] * np.sin(theta + phases[1]) + 0.10 * acc_amp[:, 1] * np.cos(second)
            acc_z = 1.0 + acc_amp[:, 2] * np.sin(theta + phases[2]) + 0.08 * acc_amp[:, 2] * np.sin(second + 0.7)
            gyro_x = gyro_amp[:, 0] * np.sin(theta + phases[0]) + 0.18 * gyro_amp[:, 0] * np.sin(second)
            gyro_y = gyro_amp[:, 1] * np.sin(theta + phases[1]) + 0.15 * gyro_amp[:, 1] * np.cos(second)
            gyro_z = gyro_amp[:, 2] * np.sin(theta + phases[2]) + 0.12 * gyro_amp[:, 2] * np.sin(second + 0.4)

            acc_x += rng.normal(0, 0.045, sample_count) + 0.015 * athlete_idx
            acc_y += rng.normal(0, 0.045, sample_count) - 0.010 * athlete_idx
            acc_z += rng.normal(0, 0.055, sample_count)
            gyro_x += rng.normal(0, 5.5, sample_count)
            gyro_y += rng.normal(0, 5.5, sample_count)
            gyro_z += rng.normal(0, 6.5, sample_count)

            if stroke == "rest":
                acc_x *= 0.45
                acc_y *= 0.45
                acc_z = 1.0 + (acc_z - 1.0) * 0.35
                gyro_x *= 0.35
                gyro_y *= 0.35
                gyro_z *= 0.35

            target_hr = resting_hr + profile["intensity"] * (max_hr_est - resting_hr) * 0.86
            heart_rate = np.empty(sample_count, dtype=float)
            for index in range(sample_count):
                tau = 28.0 if target_hr > current_hr else 18.0
                current_hr += (target_hr - current_hr) / (tau * fs) + rng.normal(0, 0.10)
                heart_rate[index] = current_hr
            heart_rate += 1.2 * np.sin(2 * np.pi * 0.18 * local_t + rng.uniform(0, 2 * np.pi))
            heart_rate = np.clip(heart_rate, 45, max_hr_est + 3)

            spo2 = 98.4 - np.maximum(0, heart_rate - 0.86 * max_hr_est) * 0.018
            spo2 += rng.normal(0, 0.18, sample_count)
            spo2 = np.clip(spo2, 92.5, 100.0)

            speed = profile["speed_mps"] * athlete_factor * (1 + 0.06 * np.sin(theta))
            speed *= np.clip(1 - 0.05 * fatigue_progress * (profile["intensity"] > 0.65), 0.88, 1.05)
            if stroke == "rest":
                speed[:] = 0.0
            segment_distance = np.cumsum(np.maximum(speed, 0)) / fs
            distance = cumulative_distance + segment_distance
            cumulative_distance = float(distance[-1]) if len(distance) else cumulative_distance

            if stroke == "rest":
                stroke_count = np.full(sample_count, cumulative_strokes, dtype=int)
            else:
                stroke_count = cumulative_strokes + np.floor(local_t * frequency).astype(int)
                cumulative_strokes = int(stroke_count[-1])

            timestamps = start_time + pd.to_timedelta(global_t, unit="s")
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "athlete_id": athlete_id,
                        "session_id": session_id,
                        "acc_x": acc_x,
                        "acc_y": acc_y,
                        "acc_z": acc_z,
                        "gyro_x": gyro_x,
                        "gyro_y": gyro_y,
                        "gyro_z": gyro_z,
                        "heart_rate_bpm": heart_rate,
                        "spo2_pct": spo2,
                        "water_temp_c": 27.0 + rng.normal(0, 0.08, sample_count),
                        "distance_m": distance,
                        "lap_id": np.floor(distance / 25.0).astype(int) + 1,
                        "stroke_count": stroke_count,
                        "stroke_label": stroke,
                    }
                )
            )
            elapsed_samples += sample_count

    data = pd.concat(frames, ignore_index=True)
    for column in ["acc_x", "gyro_y", "heart_rate_bpm", "spo2_pct"]:
        indexes = rng.choice(
            data.index.to_numpy(),
            size=max(1, len(data) // 1000),
            replace=False,
        )
        data.loc[indexes, column] = np.nan
    return data


def validate_watch_dataframe(df: pd.DataFrame) -> None:
    required = {"timestamp", *MOTION_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required smartwatch columns: "
            f"{missing}. Required motion schema: timestamp, "
            + ", ".join(MOTION_COLUMNS)
            + "."
        )


def _sampling_statistics(group: pd.DataFrame) -> tuple[float, float]:
    intervals = group["timestamp"].sort_values().diff().dt.total_seconds().dropna()
    if intervals.empty or float(intervals.median()) <= 0:
        return float("nan"), float("nan")
    frequency_hz = 1.0 / float(intervals.median())
    jitter_ms = float(intervals.std(ddof=0) * 1000) if len(intervals) > 1 else 0.0
    return frequency_hz, jitter_ms


def resample_session(group: pd.DataFrame, target_hz: int) -> pd.DataFrame:
    group = group.sort_values("timestamp").drop_duplicates("timestamp").copy()
    observed_hz, jitter_ms = _sampling_statistics(group)
    if (
        np.isfinite(observed_hz)
        and abs(observed_hz - target_hz) / target_hz < 0.03
        and jitter_ms < 2.5
    ):
        return group

    period = pd.to_timedelta(1 / target_hz, unit="s")
    indexed = group.set_index("timestamp")
    numeric_columns = indexed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in indexed.columns if column not in numeric_columns]

    numeric = indexed[numeric_columns].resample(period).mean()
    numeric = numeric.interpolate(method="time").ffill().bfill()
    if categorical_columns:
        categorical = indexed[categorical_columns].resample(period).nearest().ffill().bfill()
        output = pd.concat([numeric, categorical], axis=1)
    else:
        output = numeric
    return output.reset_index()


def clean_watch_data(df: pd.DataFrame, config: AppConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Validate, interpolate, resample, and derive motion magnitudes."""

    clean = standardize_column_names(df.copy())
    validate_watch_dataframe(clean)

    if "athlete_id" not in clean:
        clean["athlete_id"] = "uploaded_swimmer"
    if "session_id" not in clean:
        clean["session_id"] = "uploaded_session_001"
    if "stroke_label" not in clean:
        clean["stroke_label"] = "unknown"

    optional_columns = [
        "heart_rate_bpm",
        "spo2_pct",
        "distance_m",
        "lap_id",
        "stroke_count",
        "water_temp_c",
    ]
    for optional in optional_columns:
        if optional not in clean:
            clean[optional] = np.nan

    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=False)
    clean = clean.dropna(subset=["timestamp"]).copy()
    if clean.empty:
        raise ValueError("No valid timestamps remained after parsing the uploaded data.")

    clean["athlete_id"] = clean["athlete_id"].astype(str)
    clean["session_id"] = clean["session_id"].astype(str)
    clean["stroke_label"] = clean["stroke_label"].astype(str).str.lower().str.strip()

    numeric_candidates = list(dict.fromkeys(MOTION_COLUMNS + optional_columns))
    for column in numeric_candidates:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.sort_values(["session_id", "timestamp"]).drop_duplicates(
        ["session_id", "timestamp"]
    )

    interpolated_parts: list[pd.DataFrame] = []
    for _, group in clean.groupby("session_id", sort=False):
        group = group.copy()
        group[numeric_candidates] = group[numeric_candidates].interpolate(
            limit_direction="both", limit=10
        )
        interpolated_parts.append(resample_session(group, config.sample_rate_hz))

    clean = pd.concat(interpolated_parts, ignore_index=True)
    for column in MOTION_COLUMNS:
        clean[column] = clean.groupby("session_id")[column].transform(
            lambda values: values.fillna(values.median())
        )
        clean[column] = clean[column].fillna(0.0)

    clean["acc_mag"] = np.sqrt(clean[["acc_x", "acc_y", "acc_z"]].pow(2).sum(axis=1))
    clean["gyro_mag"] = np.sqrt(clean[["gyro_x", "gyro_y", "gyro_z"]].pow(2).sum(axis=1))
    return clean.sort_values(["session_id", "timestamp"]).reset_index(drop=True)


def build_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session_id, group in df.groupby("session_id"):
        frequency_hz, jitter_ms = _sampling_statistics(group)
        duration_minutes = (
            group["timestamp"].max() - group["timestamp"].min()
        ).total_seconds() / 60
        rows.append(
            {
                "session_id": str(session_id),
                "samples": int(len(group)),
                "duration_min": float(duration_minutes),
                "observed_hz": float(frequency_hz),
                "timing_jitter_ms": float(jitter_ms),
                "missing_signal_pct": float(100 * group[MOTION_COLUMNS].isna().mean().mean()),
                "has_labels": bool(
                    (~group["stroke_label"].isin(["unknown", "nan", "none", ""])).any()
                ),
            }
        )
    return pd.DataFrame(rows)


def dominant_frequency(
    values: np.ndarray,
    sampling_rate_hz: int,
    low_hz: float = 0.15,
    high_hz: float = 2.5,
) -> float:
    signal_values = np.asarray(values, dtype=float)
    signal_values = np.nan_to_num(
        signal_values - np.nanmean(signal_values),
        nan=0.0,
    )
    if len(signal_values) < 4 or np.allclose(signal_values, 0):
        return 0.0
    frequencies = np.fft.rfftfreq(len(signal_values), d=1 / sampling_rate_hz)
    power = np.abs(np.fft.rfft(signal_values)) ** 2
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not mask.any() or np.all(power[mask] <= 0):
        return 0.0
    return float(frequencies[mask][np.argmax(power[mask])])


def spectral_entropy(values: np.ndarray) -> float:
    signal_values = np.asarray(values, dtype=float)
    signal_values = np.nan_to_num(
        signal_values - np.nanmean(signal_values),
        nan=0.0,
    )
    power = np.abs(np.fft.rfft(signal_values)) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 1e-12 or len(power) <= 1:
        return 0.0
    probabilities = power / total
    return float(
        -(probabilities * np.log(probabilities + 1e-12)).sum() / np.log(len(probabilities))
    )


def zero_crossing_rate(values: np.ndarray) -> float:
    signal_values = np.asarray(values, dtype=float)
    signal_values = np.nan_to_num(
        signal_values - np.nanmean(signal_values),
        nan=0.0,
    )
    if len(signal_values) < 2:
        return 0.0
    return float(np.mean(np.signbit(signal_values[1:]) != np.signbit(signal_values[:-1])))


def motion_feature_vector(segment: pd.DataFrame, sampling_rate_hz: int) -> dict[str, float]:
    features: dict[str, float] = {}
    for column in SEQUENCE_CHANNELS:
        values = segment[column].to_numpy(dtype=float)
        median = np.nanmedian(values)
        values = np.nan_to_num(values, nan=float(median) if np.isfinite(median) else 0.0)
        centered = values - values.mean()
        features.update(
            {
                f"{column}_mean": float(values.mean()),
                f"{column}_std": float(values.std(ddof=0)),
                f"{column}_min": float(values.min()),
                f"{column}_max": float(values.max()),
                f"{column}_range": float(np.ptp(values)),
                f"{column}_rms": float(np.sqrt(np.mean(values**2))),
                f"{column}_q25": float(np.quantile(values, 0.25)),
                f"{column}_q75": float(np.quantile(values, 0.75)),
                f"{column}_zcr": zero_crossing_rate(values),
                f"{column}_entropy": spectral_entropy(values),
                f"{column}_dominant_hz": dominant_frequency(centered, sampling_rate_hz),
            }
        )

    correlation = segment[MOTION_COLUMNS].corr().fillna(0.0)
    for first, second in [
        ("acc_x", "acc_y"),
        ("acc_x", "acc_z"),
        ("acc_y", "acc_z"),
        ("gyro_x", "gyro_y"),
        ("gyro_x", "gyro_z"),
        ("gyro_y", "gyro_z"),
    ]:
        features[f"corr_{first}_{second}"] = float(correlation.loc[first, second])
    return features


def safe_mode(values: pd.Series, default: str = "unknown") -> str:
    text_values = values.dropna().astype(str)
    if text_values.empty:
        return default
    modes = text_values.mode()
    return str(modes.iloc[0]) if not modes.empty else default


def build_window_bundle(
    df: pd.DataFrame,
    config: AppConfig = DEFAULT_CONFIG,
    require_labels: bool = False,
) -> dict[str, Any]:
    """Convert cleaned sessions into engineered features and raw IMU windows."""

    window_samples = int(round(config.window_seconds * config.sample_rate_hz))
    step_samples = int(round(config.step_seconds * config.sample_rate_hz))
    feature_rows: list[dict[str, float]] = []
    sequences: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for session_id, group in df.groupby("session_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < window_samples:
            continue

        for start in range(0, len(group) - window_samples + 1, step_samples):
            segment = group.iloc[start : start + window_samples]
            label = safe_mode(segment["stroke_label"])
            label_purity = float((segment["stroke_label"].astype(str) == label).mean())
            label_is_known = label in KNOWN_STROKES
            if require_labels and (
                not label_is_known or label_purity < config.minimum_label_purity
            ):
                continue

            sequence = segment[SEQUENCE_CHANNELS].to_numpy(dtype=np.float32)
            if not np.isfinite(sequence).all():
                median = np.nanmedian(sequence, axis=0)
                median = np.where(np.isfinite(median), median, 0.0)
                bad = ~np.isfinite(sequence)
                sequence[bad] = np.take(median, np.where(bad)[1])

            feature_rows.append(motion_feature_vector(segment, config.sample_rate_hz))
            sequences.append(sequence)

            duration_seconds = max(
                (segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0]).total_seconds(),
                1e-6,
            )
            distance = segment["distance_m"].dropna()
            distance_delta = (
                float(distance.iloc[-1] - distance.iloc[0])
                if len(distance) >= 2
                else float("nan")
            )
            speed_mps = (
                distance_delta / duration_seconds
                if np.isfinite(distance_delta) and distance_delta >= 0
                else float("nan")
            )
            pace = (
                100 / speed_mps
                if np.isfinite(speed_mps) and speed_mps > 0.05
                else float("nan")
            )
            gyro_frequency = dominant_frequency(
                segment["gyro_mag"].to_numpy(),
                config.sample_rate_hz,
            )

            metadata.append(
                {
                    "athlete_id": str(segment["athlete_id"].iloc[0]),
                    "session_id": str(session_id),
                    "window_start": segment["timestamp"].iloc[0],
                    "window_end": segment["timestamp"].iloc[-1],
                    "true_label": label,
                    "label_purity": label_purity,
                    "heart_rate_mean_bpm": float(segment["heart_rate_bpm"].mean()),
                    "heart_rate_max_bpm": float(segment["heart_rate_bpm"].max()),
                    "spo2_min_pct": float(segment["spo2_pct"].min()),
                    "distance_delta_m": distance_delta,
                    "speed_mps": speed_mps,
                    "pace_sec_per_100m": pace,
                    "stroke_rate_spm": gyro_frequency * 60.0,
                    "motion_rms": float(
                        np.sqrt(np.mean(segment["gyro_mag"].to_numpy(dtype=float) ** 2))
                    ),
                }
            )

    if not sequences:
        minimum_seconds = config.window_seconds
        raise ValueError(
            "No valid analysis windows were created. Each session must contain at least "
            f"{minimum_seconds:.1f} seconds of timestamped accelerometer and gyroscope data."
        )

    features = (
        pd.DataFrame(feature_rows)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return {
        "features": features,
        "sequences": np.stack(sequences).astype(np.float32),
        "meta": pd.DataFrame(metadata),
    }


class FeatureDNN(nn.Module):
    """Dense neural network architecture retained from the source notebook."""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 160),
            nn.BatchNorm1d(160),
            nn.ReLU(),
            nn.Dropout(0.22),
            nn.Linear(160, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.18),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CNNBiGRU(nn.Module):
    """1D convolution plus bidirectional GRU sequence classifier."""

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.12),
        )
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=36,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(72, 48),
            nn.ReLU(),
            nn.Dropout(0.18),
            nn.Linear(48, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = inputs.transpose(1, 2)
        transformed = self.conv(transformed)
        transformed = transformed.transpose(1, 2)
        transformed, _ = self.gru(transformed)
        transformed = transformed.mean(dim=1)
        return self.head(transformed)


def _load_torch_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_bundle(model_directory: str | Path) -> ModelBundle:
    """Load the included pretrained artifacts without retraining in Streamlit."""

    model_directory = Path(model_directory)
    metadata_path = model_directory / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts in {model_directory}. Run scripts/train_models.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = [str(value) for value in metadata["class_names"]]
    feature_columns = [str(value) for value in metadata["feature_columns"]]

    feature_dnn = FeatureDNN(len(feature_columns), len(class_names))
    feature_dnn.load_state_dict(_load_torch_state(model_directory / "feature_dnn_state.pt"))
    feature_dnn.eval()

    sequence_model = CNNBiGRU(len(SEQUENCE_CHANNELS), len(class_names))
    sequence_model.load_state_dict(_load_torch_state(model_directory / "cnn_bigru_state.pt"))
    sequence_model.eval()

    return ModelBundle(
        rf_model=joblib.load(model_directory / "random_forest.joblib"),
        feature_scaler=joblib.load(model_directory / "feature_scaler.joblib"),
        feature_dnn=feature_dnn,
        sequence_model=sequence_model,
        sequence_mean=np.asarray(metadata["sequence_mean"], dtype=np.float32).reshape(
            1, 1, len(SEQUENCE_CHANNELS)
        ),
        sequence_std=np.asarray(metadata["sequence_std"], dtype=np.float32).reshape(
            1, 1, len(SEQUENCE_CHANNELS)
        ),
        class_names=class_names,
        ensemble_weights=np.asarray(metadata["ensemble_weights"], dtype=float),
        feature_columns=feature_columns,
        q_table=np.load(model_directory / "rl_q_table.npy"),
        model_metrics=metadata.get("model_metrics", {}),
        training_source=str(metadata.get("training_source", "structured synthetic watch data")),
    )


def predict_probabilities(model: nn.Module, inputs: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(inputs, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch,) in loader:
            logits = model(batch)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probabilities)


def infer_bundle(bundle: dict[str, Any], models: ModelBundle) -> tuple[np.ndarray, np.ndarray]:
    feature_matrix = (
        bundle["features"]
        .reindex(columns=models.feature_columns, fill_value=0.0)
        .to_numpy(dtype=np.float32)
    )
    sequence_matrix = bundle["sequences"].astype(np.float32)

    rf_probabilities = models.rf_model.predict_proba(feature_matrix)
    scaled_features = models.feature_scaler.transform(feature_matrix).astype(np.float32)
    dnn_probabilities = predict_probabilities(models.feature_dnn, scaled_features)
    normalized_sequences = (
        (sequence_matrix - models.sequence_mean) / models.sequence_std
    ).astype(np.float32)
    sequence_probabilities = predict_probabilities(models.sequence_model, normalized_sequences)

    combined = (
        models.ensemble_weights[0] * rf_probabilities
        + models.ensemble_weights[1] * dnn_probabilities
        + models.ensemble_weights[2] * sequence_probabilities
    )
    predictions = combined.argmax(axis=1)
    return predictions, combined


def estimated_max_hr(age_years: int) -> float:
    return 208.0 - 0.7 * age_years


def heart_rate_zone_number(heart_rate: pd.Series, max_hr_estimate: float) -> pd.Series:
    fraction = heart_rate / max(max_hr_estimate, 1.0)
    return pd.cut(
        fraction,
        bins=[-np.inf, 0.60, 0.70, 0.80, 0.90, np.inf],
        labels=[1, 2, 3, 4, 5],
        right=False,
    ).astype(float)


def has_sustained_condition(condition: pd.Series, required_samples: int) -> bool:
    values = condition.fillna(False).astype(int)
    if values.empty:
        return False
    rolling = values.rolling(
        max(1, required_samples),
        min_periods=max(1, required_samples),
    ).sum()
    maximum = rolling.max()
    return bool(np.isfinite(maximum) and maximum >= required_samples)


def coefficient_of_variation(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    numeric = numeric[numeric.abs() > 1e-6]
    if len(numeric) < 2:
        return 0.0
    return float(numeric.std(ddof=0) / max(abs(numeric.mean()), 1e-6))


def technique_consistency_score(window_results: pd.DataFrame) -> float:
    swim = window_results.query("predicted_stroke != 'rest'").copy()
    if swim.empty:
        return 0.0
    per_stroke_scores: list[float] = []
    for _, group in swim.groupby("predicted_stroke"):
        if len(group) < 2:
            continue
        rate_cv = coefficient_of_variation(group["stroke_rate_spm"])
        motion_cv = coefficient_of_variation(group["motion_rms"])
        score = np.clip(100 - 150 * rate_cv - 80 * motion_cv, 0, 100)
        per_stroke_scores.extend([float(score)] * len(group))
    if not per_stroke_scores:
        return 55.0
    return float(np.mean(per_stroke_scores))


def fatigue_trend_score(session_df: pd.DataFrame, window_results: pd.DataFrame) -> float:
    if len(window_results) < 6:
        return 0.0
    ordered = window_results.sort_values("window_start")
    subset_size = max(2, len(ordered) // 4)
    first = ordered.head(subset_size)
    last = ordered.tail(subset_size)

    hr_first = first["heart_rate_mean_bpm"].mean()
    hr_last = last["heart_rate_mean_bpm"].mean()
    hr_drift = (
        max(0.0, (hr_last - hr_first) / max(abs(hr_first), 1.0))
        if np.isfinite(hr_first) and np.isfinite(hr_last)
        else 0.0
    )

    motion_first = first["motion_rms"].mean()
    motion_last = last["motion_rms"].mean()
    motion_drop = (
        max(0.0, (motion_first - motion_last) / max(abs(motion_first), 1.0))
        if np.isfinite(motion_first) and np.isfinite(motion_last)
        else 0.0
    )

    rate_first = first.query("predicted_stroke != 'rest'")["stroke_rate_spm"].median()
    rate_last = last.query("predicted_stroke != 'rest'")["stroke_rate_spm"].median()
    rate_drift = (
        abs(rate_last - rate_first) / max(abs(rate_first), 1.0)
        if np.isfinite(rate_first) and np.isfinite(rate_last)
        else 0.0
    )

    return float(np.clip(420 * hr_drift + 180 * motion_drop + 120 * rate_drift, 0, 100))


def data_quality_score(session_df: pd.DataFrame, target_hz: int) -> float:
    missing = float(session_df[MOTION_COLUMNS].isna().mean().mean())
    observed_hz, jitter_ms = _sampling_statistics(session_df)
    frequency_penalty = (
        0.0
        if not np.isfinite(observed_hz)
        else min(1.0, abs(observed_hz - target_hz) / target_hz)
    )
    jitter_penalty = min(
        1.0,
        (jitter_ms if np.isfinite(jitter_ms) else 1000.0) / 25.0,
    )
    return float(
        np.clip(100 - 100 * missing - 35 * frequency_penalty - 20 * jitter_penalty, 0, 100)
    )


def _optional_round(value: float, decimals: int) -> float | None:
    return round(float(value), decimals) if np.isfinite(value) else None


def build_health_report(
    session_df: pd.DataFrame,
    window_results: pd.DataFrame,
    profile: AthleteProfile,
    config: AppConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    session_df = session_df.sort_values("timestamp").copy()
    duration_seconds = max(
        (session_df["timestamp"].max() - session_df["timestamp"].min()).total_seconds(),
        0.0,
    )
    distance_values = session_df["distance_m"].dropna()
    distance_m = (
        float(distance_values.iloc[-1] - distance_values.iloc[0])
        if len(distance_values) >= 2
        else float("nan")
    )
    speed_mps = (
        distance_m / duration_seconds
        if np.isfinite(distance_m) and duration_seconds > 0
        else float("nan")
    )
    pace_sec_per_100m = (
        100 / speed_mps
        if np.isfinite(speed_mps) and speed_mps > 0.05
        else float("nan")
    )

    heart_rate = pd.to_numeric(session_df["heart_rate_bpm"], errors="coerce")
    spo2 = pd.to_numeric(session_df["spo2_pct"], errors="coerce")
    max_hr_estimate = estimated_max_hr(profile.age_years)
    zones = heart_rate_zone_number(heart_rate, max_hr_estimate)
    dt_minutes = 1 / config.sample_rate_hz / 60
    zone_minutes = {
        f"zone_{zone}_minutes": float((zones == zone).sum() * dt_minutes)
        for zone in range(1, 6)
    }
    training_load_index = float(
        sum((zone**2) * (zones == zone).sum() * dt_minutes for zone in range(1, 6))
    )

    technique = technique_consistency_score(window_results)
    fatigue = fatigue_trend_score(session_df, window_results)
    quality = data_quality_score(session_df, config.sample_rate_hz)
    stroke_distribution = (
        window_results["predicted_stroke"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .to_dict()
        if not window_results.empty
        else {}
    )

    required_samples = int(config.sustained_alert_seconds * config.sample_rate_hz)
    high_hr = has_sustained_condition(
        heart_rate >= config.high_hr_fraction_of_estimated_max * max_hr_estimate,
        required_samples,
    )
    low_spo2_sensor = has_sustained_condition(
        spo2 < config.spo2_sensor_review_threshold_pct,
        required_samples,
    )

    flags: list[dict[str, str]] = []
    if high_hr:
        flags.append(
            {
                "level": "review",
                "title": "Sustained high-intensity estimate",
                "detail": (
                    "The watch heart-rate estimate stayed near the configurable maximum-heart-rate "
                    "boundary. Slow down or stop if the swimmer feels unwell, and validate this rule "
                    "clinically before production use."
                ),
            }
        )
    if low_spo2_sensor:
        flags.append(
            {
                "level": "review",
                "title": "Oxygen sensor reading needs review",
                "detail": (
                    "A sustained low watch SpO2 estimate was detected. Check sensor fit and signal "
                    "quality; seek qualified medical assessment for concerning symptoms. This "
                    "prototype does not diagnose hypoxemia."
                ),
            }
        )
    if fatigue >= 70:
        flags.append(
            {
                "level": "caution",
                "title": "High fatigue trend",
                "detail": (
                    "Late-session heart-rate and motion changes suggest reducing intensity and "
                    "prioritizing technique or recovery."
                ),
            }
        )
    if quality < 75:
        flags.append(
            {
                "level": "data",
                "title": "Sensor quality limitation",
                "detail": (
                    "Missing values, timing irregularity, or unexpected sampling frequency reduced "
                    "confidence in the analysis."
                ),
            }
        )
    if not flags:
        flags.append(
            {
                "level": "info",
                "title": "No demonstration threshold crossed",
                "detail": (
                    "No configured review rule was triggered. This is not a medical clearance or "
                    "guarantee of safety."
                ),
            }
        )

    median_stroke_rate = window_results.query("predicted_stroke != 'rest'")[
        "stroke_rate_spm"
    ].median()

    return {
        "project": PROJECT_NAME,
        "author": AUTHOR_NAME,
        "mentor": MENTOR_NAME,
        "session_id": str(session_df["session_id"].iloc[0]),
        "athlete_profile": asdict(profile),
        "summary": {
            "duration_minutes": round(duration_seconds / 60, 2),
            "distance_m": _optional_round(distance_m, 2),
            "average_speed_mps": _optional_round(speed_mps, 3),
            "pace_sec_per_100m": _optional_round(pace_sec_per_100m, 2),
            "average_heart_rate_bpm": (
                round(float(heart_rate.mean()), 1) if heart_rate.notna().any() else None
            ),
            "maximum_heart_rate_bpm": (
                round(float(heart_rate.max()), 1) if heart_rate.notna().any() else None
            ),
            "estimated_max_heart_rate_bpm": round(max_hr_estimate, 1),
            "minimum_spo2_sensor_pct": (
                round(float(spo2.min()), 1) if spo2.notna().any() else None
            ),
            "median_estimated_stroke_rate_spm": _optional_round(median_stroke_rate, 1),
            "technique_consistency_score_0_to_100": round(technique, 1),
            "fatigue_trend_score_0_to_100": round(fatigue, 1),
            "data_quality_score_0_to_100": round(quality, 1),
            "demo_training_load_index": round(training_load_index, 2),
        },
        "heart_rate_zones": zone_minutes,
        "predicted_stroke_distribution_pct": stroke_distribution,
        "flags": flags,
        "limitations": [
            "Educational prototype; not a medical device or diagnosis.",
            "Estimated maximum heart rate and alert thresholds are configurable approximations.",
            "Underwater optical heart-rate and SpO2 readings may be inaccurate.",
            "Synthetic-model performance does not establish accuracy on Kevin's custom watch.",
        ],
    }


def report_to_rl_state(report: dict[str, Any]) -> tuple[int, int, int]:
    summary = report["summary"]
    fatigue_score = float(summary["fatigue_trend_score_0_to_100"])
    fatigue = 0 if fatigue_score < 35 else (1 if fatigue_score < 70 else 2)

    average_hr = summary["average_heart_rate_bpm"]
    max_hr_estimate = summary["estimated_max_heart_rate_bpm"]
    fraction = (
        average_hr / max_hr_estimate
        if average_hr is not None and max_hr_estimate
        else 0.65
    )
    hr_zone = int(np.digitize(fraction, [0.60, 0.70, 0.80, 0.90]))
    hr_zone = int(np.clip(hr_zone, 0, 4))

    technique_score = float(summary["technique_consistency_score_0_to_100"])
    technique = 2 if technique_score >= 75 else (1 if technique_score >= 50 else 0)
    return fatigue, hr_zone, technique


def constrained_rl_recommendation(
    q_table: np.ndarray,
    report: dict[str, Any],
) -> dict[str, Any]:
    state = report_to_rl_state(report)
    learned_index = int(np.argmax(q_table[state]))
    learned_action = RL_ACTIONS[learned_index]
    final_action = learned_action
    override_reason: str | None = None

    flag_titles = {flag["title"] for flag in report.get("flags", [])}
    fatigue, hr_zone, technique = state
    if hr_zone == 4 or fatigue == 2 or "Sustained high-intensity estimate" in flag_titles:
        final_action = "recover_easy"
        override_reason = (
            "A safety rule overrode the learned policy because estimated intensity or fatigue was high."
        )
    elif technique == 0:
        final_action = "technique_drill"
        override_reason = "A technique rule prioritized a drill over increasing effort."

    return {
        "discrete_state": {
            "fatigue_level_0_to_2": state[0],
            "hr_zone_0_to_4": state[1],
            "technique_level_0_to_2": state[2],
        },
        "learned_action": learned_action,
        "final_safety_constrained_action": final_action,
        "guidance": ACTION_TEXT[final_action],
        "override_reason": override_reason,
        "disclaimer": "Research demonstration only; not medical advice or autonomous supervision.",
    }


def analyze_watch_data(
    raw_watch_df: pd.DataFrame,
    models: ModelBundle,
    profile: AthleteProfile,
    session_id: str | None = None,
    config: AppConfig = DEFAULT_CONFIG,
) -> AnalysisResult:
    """Run the full app pipeline for one selected smartwatch session."""

    clean = clean_watch_data(raw_watch_df, config)
    available_sessions = clean["session_id"].astype(str).drop_duplicates().tolist()
    if not available_sessions:
        raise ValueError("The uploaded file does not contain any sessions.")

    selected_session = str(session_id or available_sessions[0])
    if selected_session not in available_sessions:
        raise ValueError(
            f"Session {selected_session!r} is not present. Available sessions: {available_sessions}."
        )

    session_data = clean[clean["session_id"].astype(str).eq(selected_session)].copy()
    bundle = build_window_bundle(session_data, config=config, require_labels=False)
    prediction_ids, probabilities = infer_bundle(bundle, models)

    results = bundle["meta"].copy().reset_index(drop=True)
    class_array = np.asarray(models.class_names, dtype=object)
    results["predicted_stroke"] = class_array[prediction_ids]
    results["prediction_confidence"] = probabilities.max(axis=1)
    for class_index, class_name in enumerate(models.class_names):
        results[f"prob_{class_name}"] = probabilities[:, class_index]

    report = build_health_report(session_data, results, profile, config)
    report["model_information"] = {
        "ensemble_weights": {
            "random_forest": float(models.ensemble_weights[0]),
            "feature_dnn": float(models.ensemble_weights[1]),
            "cnn_bigru": float(models.ensemble_weights[2]),
        },
        "mean_window_confidence": round(float(results["prediction_confidence"].mean()), 4),
        "training_source": models.training_source,
        "model_metrics": models.model_metrics,
    }
    report["coaching_recommendation"] = constrained_rl_recommendation(
        models.q_table,
        report,
    )

    return AnalysisResult(
        report=report,
        windows=results,
        clean_data=clean,
        session_data=session_data,
        quality_table=build_data_quality_table(clean),
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def report_to_json_bytes(report: dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2, default=str).encode("utf-8")


def seconds_to_pace_label(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "N/A"
    minutes = int(seconds // 60)
    remaining = int(round(seconds - 60 * minutes))
    if remaining == 60:
        minutes += 1
        remaining = 0
    return f"{minutes}:{remaining:02d} /100 m"


def safe_metric(value: Any, suffix: str = "", decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:,.{decimals}f}{suffix}"


seed_everything(DEFAULT_CONFIG.random_seed)
