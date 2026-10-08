"""Streamlit Community Cloud entry point (no Torch or scikit-learn dependency).

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile

import streamlit as st

ROOT = Path(__file__).resolve().parent
DEMO_FILE = ROOT / 'sample_data' / 'kevin_demo_watch_session.csv'
MODELS_DIR = ROOT / 'models'
AUTHOR = 'Kevin Sun'
MENTOR = 'Dr. Qingyang Xiao'

st.set_page_config(
    page_title='AquaMind | AI Swimming Health — Kevin Sun',
    page_icon='🏊',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown('''
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 95% 10%,rgba(23,169,210,.12),transparent 31%),
                radial-gradient(circle at 0% 55%,rgba(45,102,181,.08),transparent 36%);
}
[data-testid="stMetric"] {
    border: 1px solid rgba(60,130,170,.19); border-radius: 15px;
    padding: .78rem 1rem; background: rgba(255,255,255,.035);
}
[data-testid="stSidebar"] { border-right: 1px solid rgba(55,136,170,.18); }
.hero-swim {
    background: linear-gradient(122deg,#071b34 0%,#07466c 51%,#0e7890 100%);
    border:1px solid rgba(156,238,255,.17);border-radius:22px;
    padding:2.15rem 2.15rem;color:white; margin-bottom:1.15rem;
    box-shadow:0 13px 36px rgba(3,34,59,.15);
}
.hero-swim h1 {font-size:clamp(1.8rem,3.65vw,3.1rem);font-weight:800; margin:0 0 .35rem;letter-spacing:-.025em;}
.hero-swim p {color:#d1eef6;font-size:1.02rem;max-width:940px;margin:.35rem 0;}
.hero-swim .credit {color:#b7efff;font-weight:650;margin-top:1rem;font-size:.93rem;}
.kicker {text-transform:uppercase;letter-spacing:.14em;font-size:.75rem;font-weight:750;color:#a8eeff;margin-bottom:.6rem;}
.eyebrow {font-size:.86rem;opacity:.74;}
.flowtile {border:1px solid rgba(60,130,170,.23);border-radius:16px;padding:1rem 1.05rem;
    background:rgba(80,170,205,.055);min-height:137px;}
.flowtile b {font-size:1rem;}
.flowtile p {font-size:.88rem;opacity:.75;margin:.6rem 0 0;}
.creditfooter {margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(110,150,180,.2);
    text-align:center;font-size:.87rem;opacity:.78;}
</style>''', unsafe_allow_html=True)

st.markdown(f'''
<div class="hero-swim">
  <div class="kicker">● Smartwatch motion intelligence · Research prototype</div>
  <h1>🏊 AquaMind — AI Swimming Health</h1>
  <p>Turn wrist-mounted accelerometer, gyroscope and optional health signals into
     stroke classifications, personalized swim analytics, interactive plots and
     a research coaching agent.</p>
  <div class="credit">Student author: {AUTHOR} &nbsp;|&nbsp; Mentor: {MENTOR}</div>
</div>''', unsafe_allow_html=True)


def _visitor_counter():
    from src.visitor_counter import register_visit
    # Accessing st.secrets when no secrets file exists raises; no network requests
    # are made unless a complete Supabase configuration is available.
    try:
        secrets = st.secrets.to_dict()
    except Exception:
        secrets = {}
    try:
        return register_visit(
            st.session_state, secrets=secrets,
            database_path=Path('/tmp/kevin_sun_swim_visitors.sqlite3'),
        )
    except Exception:
        return None, 'unavailable'


@st.cache_resource(show_spinner=False)
def _models():
    from src.portable_models import load_portable_models
    return load_portable_models(MODELS_DIR)


def _read_sessions(content: bytes, filename: str) -> list[str]:
    from src.swim_ai import read_watch_file, standardize_column_names
    dataframe = standardize_column_names(read_watch_file(content, filename))
    if 'session_id' not in dataframe:
        return ['uploaded_session_001']
    return dataframe['session_id'].fillna('uploaded_session_001').astype(str).drop_duplicates().tolist()[:30]


def _analyze(data: bytes, filename: str, profile_values: tuple, session_id: str):
    from src.swim_ai import analyze_watch_data, AthleteProfile, read_watch_file
    profile = AthleteProfile(*profile_values)
    raw = read_watch_file(data, filename)
    return analyze_watch_data(raw, _models(), profile, session_id=session_id)


def _package_result(result, animation_html: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('health_report.json', json.dumps(result.report, indent=2, default=str))
        archive.writestr('window_predictions.csv', result.windows.to_csv(index=False))
        archive.writestr('cleaned_watch_data.csv', result.session_data.to_csv(index=False))
        archive.writestr('data_quality.csv', result.quality_table.to_csv(index=False))
        archive.writestr('swimmer_animation.html', animation_html)
        archive.writestr('README.txt',
            'AquaMind — Kevin Sun / Dr. Qingyang Xiao\n'
            'Research demonstration only; not diagnostic or a medical device.\n'
            'Models originally trained on synthetic smartwatch signals.\n')
    return buffer.getvalue()


def _display_number(number, suffix='', digits=1):
    try:
        import numpy as np
        if number is None or not np.isfinite(float(number)):
            return 'N/A'
        return f'{float(number):,.{digits}f}{suffix}'
    except (ValueError, TypeError):
        return 'N/A'


st.sidebar.markdown('## 🏊 AquaMind control center')
count, backend = _visitor_counter()
if count is None:
    st.sidebar.metric('App visits', 'Unavailable')
else:
    st.sidebar.metric('App visits', f'{count:,}')
st.sidebar.caption('One visit per new browser session. Storage: ' + backend + '.')
if backend != 'Supabase':
    st.sidebar.caption('Local counter can reset after cloud restarts. Configure Supabase for persistent totals.')
st.sidebar.divider()

source_mode = st.sidebar.radio(
    'Watch data source', ['Included demonstration', 'Upload watch export'],
    help='Demo is synthetic. Real data must follow the six-axis motion schema.'
)
content: bytes | None
filename: str
if source_mode == 'Included demonstration':
    content = DEMO_FILE.read_bytes()
    filename = DEMO_FILE.name
    st.sidebar.success('Demo ready · 20 Hz motion signals')
else:
    uploaded = st.sidebar.file_uploader(
        'Upload a CSV / JSON / JSONL file', type=['csv', 'json', 'jsonl'],
        help='File limit 15 MB. Required columns: timestamp, acc_x/y/z, gyro_x/y/z.'
    )
    content = uploaded.getvalue() if uploaded is not None else None
    filename = uploaded.name if uploaded is not None else 'watch.csv'
    if content is not None and len(content) > 15 * 1024 * 1024:
        st.sidebar.error('This cloud demo accepts uploads smaller than 15 MB.')
        content = None

st.sidebar.markdown('### Swimmer profile')
athlete_id = st.sidebar.text_input('Display athlete ID', 'demo_swimmer')
age = st.sidebar.number_input('Age (years)', 8, 100, 17, 1)
mass = st.sidebar.number_input('Mass (kg)', 20.0, 250.0, 65.0, 1.0)
rest_hr = st.sidebar.number_input('Resting HR (bpm)', 35.0, 130.0, 60.0, 1.0)
pool_length = st.sidebar.selectbox('Pool length', [25.0, 50.0], index=0, format_func=lambda v: f'{v:.0f} m')
profile_values = (athlete_id.strip() or 'swimmer', int(age), float(mass), float(rest_hr), float(pool_length))

sessions = ['session_001'] if source_mode == 'Included demonstration' else []
if source_mode == 'Upload watch export' and content is not None:
    try:
        sessions = _read_sessions(content, filename)
    except Exception as exc:
        st.sidebar.error(f'Cannot read watch export: {exc}')
        content = None
if not sessions:
    sessions = ['uploaded_session_001']
selected_session = st.sidebar.selectbox('Workout session', sessions)

if content is not None:
    signature = hashlib.sha256(content).hexdigest() + repr((filename, profile_values, selected_session))
else:
    signature = None
if st.session_state.get('data_signature') != signature:
    st.session_state.pop('analysis_result', None)
    st.session_state.pop('swim_gif', None)
    st.session_state['data_signature'] = signature

run = st.sidebar.button('🚀 Analyze swimming session', type='primary', use_container_width=True, disabled=content is None)
if run and content is not None:
    with st.spinner('Analyzing IMU windows with Random Forest, DNN, CNN–BiGRU and Q-learning coaching...'):
        try:
            st.session_state['analysis_result'] = _analyze(content, filename, profile_values, selected_session)
            st.success('Session analysis complete.')
        except Exception as exc:
            st.session_state.pop('analysis_result', None)
            st.error(f'The session could not be analyzed: {exc}')
            st.info('Check timestamps, sensor units, required column names, and workout duration (at least four seconds).')

st.sidebar.divider()
st.sidebar.caption('Author · **Kevin Sun**')
st.sidebar.caption('Mentor · **Dr. Qingyang Xiao**')
st.sidebar.caption('Educational and athletic research; not clinical advice.')

result = st.session_state.get('analysis_result')
if result is None:
    st.subheader('Explore the full AI swimming pipeline')
    cols = st.columns(4)
    features = [
        ('01 · Capture', '⌚ Wearable sensors', 'Record 6-axis wrist motion, HR, SpO₂, distance and stroke labels when available.'),
        ('02 · Clean', '🧭 Signal processing', 'Validate timestamped records, interpolate gaps and create 4-second motion windows.'),
        ('03 · Understand', '🧠 Three AI models', 'Random Forest + deep neural network + CNN–BiGRU sequence model, deployed with NumPy.'),
        ('04 · Coach', '🌊 Health insights', 'Summaries, five stroke classes, reinforcement learning, plots, replay and exports.'),
    ]
    for col, (kicker, heading, description) in zip(cols, features):
        with col:
            st.markdown(f'<div class="flowtile"><div class="eyebrow">{kicker}</div><b>{heading}</b><p>{description}</p></div>', unsafe_allow_html=True)
    st.write('')
    c1, c2 = st.columns([3, 2])
    with c1:
        st.info('Select a data source and click **Analyze swimming session** in the sidebar. The landing page launches without loading large ML libraries or retraining models.')
        st.markdown('#### Supported smartwatch export fields')
        st.code('timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate_bpm,spo2_pct,distance_m,session_id,stroke_label', language='text')
        st.caption('Accelerometers: g; gyroscopes: degrees/s. Input format must be adapted if Kevin’s physical watch uses other sensor units.')
    with c2:
        st.markdown('#### Original pretrained ensemble')
        st.markdown('**Random Forest** · 260 trees  \n**Feature DNN** · 4 dense layers  \n**CNN–BiGRU** · time-series deep model  \n**Q-learning** · safety-constrained coaching')
        st.caption('Real trained weights are bundled as portable arrays, not placeholder outputs. Original Colab notebook supports retraining with PyTorch.')
    st.download_button('Download example smartwatch CSV', DEMO_FILE.read_bytes(),
                       'kevin_demo_watch_session.csv', 'text/csv')
    st.warning('Prototype: cannot diagnose illness, confirm blood oxygen, detect drowning, or replace a lifeguard. Validation on Kevin’s actual watch is still required.')
else:
    from src import visuals
    report = result.report
    summary = report['summary']
    coach = report['coaching_recommendation']
    st.subheader(f'Workout intelligence · {report["session_id"]}')
    metrics = st.columns(6)
    metric_data = [
        ('Duration', _display_number(summary['duration_minutes'], ' min', 2)),
        ('Distance', _display_number(summary['distance_m'], ' m', 1)),
        ('Avg heart rate', _display_number(summary['average_heart_rate_bpm'], ' bpm', 0)),
        ('Pace / 100 m', 'N/A' if summary['pace_sec_per_100m'] is None else f'{int(summary["pace_sec_per_100m"] // 60):02d}:{int(summary["pace_sec_per_100m"] % 60):02d}'),
        ('Technique proxy', _display_number(summary['technique_consistency_score_0_to_100'], ' /100', 0)),
        ('Data quality', _display_number(summary['data_quality_score_0_to_100'], ' /100', 0)),
    ]
    for col, (label, metric) in zip(metrics, metric_data):
        col.metric(label, metric)

    st.info('**AI coaching suggestion: ' + coach['final_safety_constrained_action'].replace('_', ' ').title()
            + '** — ' + coach['guidance'])
    if coach.get('override_reason'):
        st.warning('Safety rule override: ' + coach['override_reason'])

    tab_overview, tab_strokes, tab_health, tab_signals, tab_replay, tab_export = st.tabs([
        '📊 Dashboard', '🧠 AI Stroke Models', '💗 Health & Coaching',
        '📈 Motion Signals', '🎞️ Swimmer Animation', '📥 Export & Methods',
    ])
    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(visuals.stroke_distribution_figure(report), use_container_width=True)
        with c2:
            st.plotly_chart(visuals.score_figure(report), use_container_width=True)
        if result.session_data[['heart_rate_bpm', 'spo2_pct']].notna().any().any():
            st.plotly_chart(visuals.sensor_timeline_figure(result.session_data), use_container_width=True)
        else:
            st.info('No heart-rate or oxygen-sensor channels were supplied; health-sensor charts are unavailable.')
        st.caption('Technique, fatigue and load are nonclinical heuristic/proxy scores.')

    with tab_strokes:
        st.plotly_chart(visuals.stroke_timeline_figure(result.session_data, result.windows), use_container_width=True)
        st.markdown('#### Stroke classification and estimated confidence')
        st.dataframe(result.windows[['window_start', 'predicted_stroke', 'prediction_confidence',
                                     'stroke_rate_spm', 'heart_rate_mean_bpm', 'speed_mps']].round(3),
                     use_container_width=True, hide_index=True)
        with st.expander('About the underlying ML, DNN and CNN–BiGRU ensemble'):
            st.markdown('Three original trained models contribute to each 4-second motion window. The same weight values and architectures are evaluated entirely with NumPy, without installing PyTorch or scikit-learn on Streamlit Cloud.')
            st.json(report.get('model_information', {}))

    with tab_health:
        c1, c2 = st.columns(2)
        with c1:
            if result.session_data['heart_rate_bpm'].notna().any():
                st.plotly_chart(visuals.heart_rate_zone_figure(report), use_container_width=True)
            else:
                st.info('Heart-rate zone chart is unavailable without a watch heart-rate channel.')
        with c2:
            st.markdown('#### Research monitoring flags')
            for flag in report.get('flags', []):
                with st.container(border=True):
                    st.markdown('**' + flag['title'] + '**')
                    st.caption(flag['detail'])
        st.markdown('#### Reinforcement-learning coaching agent')
        st.write('Learned suggestion: **' + coach['learned_action'].replace('_', ' ').title() + '**')
        st.write('Final recommendation: **' + coach['final_safety_constrained_action'].replace('_', ' ').title() + '**')
        st.json({'state': coach['discrete_state'], 'recommendation': coach['guidance']})
        st.caption('This RL policy was trained in simulation; it is not real-time medical supervision.')

    with tab_signals:
        st.plotly_chart(visuals.motion_timeline_figure(result.session_data), use_container_width=True)
        st.markdown('#### Preprocessed smartwatch data')
        st.dataframe(result.session_data.head(800), use_container_width=True, hide_index=True)
        with st.expander('Sensor resampling / quality metrics'):
            st.dataframe(result.quality_table, use_container_width=True, hide_index=True)

    with tab_replay:
        st.markdown('### Interactive, sensor-driven swimming replay')
        animation_html = visuals.animated_swimmer_html(result.session_data, result.windows)
        st.components.v1.html(animation_html, height=420, scrolling=False)
        st.caption('A conceptual 2D animation of IMU-driven arm/kick motion, not a captured camera video or anatomical reconstruction.')
        st.download_button('Download replay as self-contained animated HTML', animation_html.encode('utf-8'),
                           'kevin_swimming_animation.html', 'text/html')
        if st.button('Generate downloadable GIF animation (on demand)'):
            with st.spinner('Drawing GIF frames...'):
                try:
                    from src.gif_export import swimmer_gif
                    st.session_state['swim_gif'] = swimmer_gif(result.session_data, result.windows)
                except Exception as exc:
                    st.error('GIF export unavailable: ' + str(exc))
        if 'swim_gif' in st.session_state:
            st.download_button('Download swimmer GIF', st.session_state['swim_gif'],
                               'kevin_swimming_animated.gif', 'image/gif')

    with tab_export:
        st.markdown('### Download the analysis')
        animation_html = visuals.animated_swimmer_html(result.session_data, result.windows)
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button('📋 Health report JSON', json.dumps(report, indent=2, default=str).encode(),
                           'swim_health_report.json', 'application/json', use_container_width=True)
        c2.download_button('🧠 AI predictions CSV', result.windows.to_csv(index=False).encode(),
                           'swim_window_predictions.csv', 'text/csv', use_container_width=True)
        c3.download_button('⌚ Clean watch CSV', result.session_data.to_csv(index=False).encode(),
                           'swim_cleaned_watch.csv', 'text/csv', use_container_width=True)
        c4.download_button('📦 Complete results ZIP', _package_result(result, animation_html),
                           'kevin_swimming_ai_results.zip', 'application/zip', use_container_width=True)
        st.markdown('#### Processing steps')
        st.markdown('Watch CSV/JSON → schema aliases and validation → optional resampling at 20 Hz → 4-second overlapping windows → 94 time/frequency/correlation features → Random Forest + dense DNN + CNN–BiGRU → weighted probabilities → health summary → simulation-trained Q-learning advice with safety overrides.')
        st.markdown('**Student author:** Kevin Sun  \n**Mentor:** Dr. Qingyang Xiao')
        st.warning('Models were originally trained using synthetic smartwatch recordings. Their reported demonstration metrics must not be interpreted as validated accuracy on real swimmers or medical performance.')

st.markdown(f'<div class="creditfooter">AquaMind · AI Swimming Health App &nbsp;|&nbsp; Author: {AUTHOR} &nbsp;|&nbsp; Mentor: {MENTOR} &nbsp;|&nbsp; Educational research prototype</div>', unsafe_allow_html=True)
