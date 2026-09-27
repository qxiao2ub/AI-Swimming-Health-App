from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.swim_ai import (
    AUTHOR_NAME,
    MENTOR_NAME,
    AthleteProfile,
    analyze_watch_data,
    clean_watch_data,
    generate_synthetic_watch_data,
    load_model_bundle,
)
from src.visitor_counter import register_visit


def test_end_to_end_demo_analysis():
    models = load_model_bundle(ROOT / "models")
    raw = generate_synthetic_watch_data(n_sessions=1, seed=123)
    profile = AthleteProfile(
        athlete_id="test_swimmer",
        age_years=17,
        mass_kg=65.0,
        resting_hr_bpm=60.0,
        pool_length_m=25.0,
    )
    result = analyze_watch_data(raw, models, profile, session_id="session_001")

    assert result.report["author"] == AUTHOR_NAME
    assert result.report["mentor"] == MENTOR_NAME
    assert len(result.windows) > 10
    assert set(result.windows["predicted_stroke"]).issubset(set(models.class_names))
    assert result.windows["prediction_confidence"].between(0, 1).all()
    assert 0 <= result.report["summary"]["data_quality_score_0_to_100"] <= 100
    assert "coaching_recommendation" in result.report
    assert not result.session_data.empty


def test_column_aliases_and_cleaning():
    raw = generate_synthetic_watch_data(n_sessions=1, seed=44).head(240).copy()
    aliased = raw.rename(
        columns={
            "timestamp": "time",
            "acc_x": "ax",
            "acc_y": "ay",
            "acc_z": "az",
            "gyro_x": "gx",
            "gyro_y": "gy",
            "gyro_z": "gz",
            "heart_rate_bpm": "heart_rate",
        }
    )
    cleaned = clean_watch_data(aliased)
    assert {"timestamp", "acc_x", "gyro_z", "heart_rate_bpm", "acc_mag", "gyro_mag"}.issubset(
        cleaned.columns
    )
    assert np.isfinite(cleaned[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]]).all().all()


def test_visitor_counter_once_per_session(tmp_path):
    database = tmp_path / "visitors.sqlite3"
    first_session = {}
    first_count, first_backend = register_visit(
        first_session,
        secrets={},
        database_path=database,
    )
    same_count, same_backend = register_visit(
        first_session,
        secrets={},
        database_path=database,
    )
    second_count, _ = register_visit(
        {},
        secrets={},
        database_path=database,
    )

    assert first_backend == "local SQLite"
    assert same_backend == first_backend
    assert same_count == first_count
    assert second_count == first_count + 1
