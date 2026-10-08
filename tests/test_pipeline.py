"""Full real pipeline smoke tests, safety, schema, and artifacts."""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd
import pytest
from src.swim_ai import (
    analyze_watch_data, AthleteProfile, clean_watch_data, load_model_bundle,
    read_watch_file, build_window_bundle, estimated_max_hr,
)
from src.visuals import (
    heart_rate_zone_figure, score_figure, sensor_timeline_figure,
    motion_timeline_figure, stroke_distribution_figure, stroke_timeline_figure,
    animated_swimmer_html,
)
from src.gif_export import swimmer_gif

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def sample():
    return pd.read_csv(ROOT / 'sample_data' / 'kevin_demo_watch_session.csv')


@pytest.fixture(scope='module')
def analysis(sample):
    model = load_model_bundle(ROOT / 'models')
    return analyze_watch_data(sample, model, AthleteProfile(), session_id='session_001')


def test_report_contains_credits_and_five_activity_types(analysis):
    assert analysis.report['author'] == 'Kevin Sun'
    assert analysis.report['mentor'] == 'Dr. Qingyang Xiao'
    assert len(analysis.windows) == 47
    assert set(analysis.windows['predicted_stroke']) == {
        'freestyle','backstroke','breaststroke','butterfly','rest'
    }
    assert analysis.report['summary']['data_quality_score_0_to_100'] >= 95
    assert analysis.report['coaching_recommendation']['final_safety_constrained_action']
    json.dumps(analysis.report)


def test_visualizations_replay_and_gif(analysis):
    charts = [
        heart_rate_zone_figure(analysis.report),
        score_figure(analysis.report),
        sensor_timeline_figure(analysis.session_data),
        motion_timeline_figure(analysis.session_data),
        stroke_distribution_figure(analysis.report),
        stroke_timeline_figure(analysis.session_data, analysis.windows),
    ]
    assert all(len(chart.data) >= 1 for chart in charts)
    replay = animated_swimmer_html(analysis.session_data, analysis.windows)
    assert '<canvas' in replay and 'Kevin Sun' in replay
    gif = swimmer_gif(analysis.session_data, analysis.windows)
    assert gif[:6] in (b'GIF89a', b'GIF87a') and len(gif)>10_000


def test_column_aliases_and_optional_sensor_absence(sample):
    reduced = sample[['timestamp','acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].copy()
    reduced = reduced.rename(columns={'acc_x':'ax','gyro_z':'gz'})
    clean = clean_watch_data(reduced)
    assert clean['heart_rate_bpm'].isna().all()
    assert len(build_window_bundle(clean)['meta']) == 47


def test_errors_for_bad_input_and_long_session(sample):
    with pytest.raises(ValueError, match='Missing required'):
        clean_watch_data(sample.drop(columns=['acc_x']))
    bad = sample.copy()
    bad['timestamp'] = 'not-a-time'
    with pytest.raises(ValueError, match='valid timestamps'):
        clean_watch_data(bad)
    long = sample.copy()
    long.loc[long.index[-1], 'timestamp'] = '2027-02-15T00:00:00'
    with pytest.raises(ValueError, match='2 hours'):
        clean_watch_data(long)


def test_csv_and_json_file_upload_readers(sample):
    csv_payload = sample.head(4).to_csv(index=False).encode()
    assert len(read_watch_file(csv_payload, 'watch.csv')) == 4
    json_payload = sample.head(3).to_json(orient='records').encode()
    assert len(read_watch_file(json_payload, 'watch.json')) == 3
    with pytest.raises(ValueError, match='Upload'):
        read_watch_file(b'bad', 'random.xml')


def test_age_max_formula():
    assert abs(estimated_max_hr(17) - 196.1) < 0.001
