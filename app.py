"""Streamlit entry point for the Kevin Sun AI Swimming Health App.

FAST-START DESIGN
------------------
The heavy PyTorch/scikit-learn analytics module is intentionally NOT imported
when Streamlit boots. The public landing page renders first; the AI models are
loaded only after the user clicks ``Run Full AI Analysis``. This prevents the
Community Cloud app from appearing stuck on the startup/"oven" screen while
Python imports deep-learning libraries and loads model weights.

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from src.visitor_counter import register_visit

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
SAMPLE_PATH = ROOT / "sample_data" / "kevin_demo_watch_session.csv"
AUTHOR_NAME = "Kevin Sun"
MENTOR_NAME = "Dr. Qingyang Xiao"
PROJECT_NAME = "Kevin Sun AI Swimming Health App"

# Lightweight aliases used before the AI stack is imported.
ALIASES = {
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
REQUIRED_COLUMNS = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

st.set_page_config(
    page_title="AI Swimming Health App | Kevin Sun",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --navy:#06172a; --blue:#0b6f99; --cyan:#36c5e8; }
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 92% 0%, rgba(54,197,232,.13), transparent 28%),
              radial-gradient(circle at 0% 26%, rgba(11,111,153,.08), transparent 34%);
}
.hero {
  padding: 2.1rem 2.15rem;
  border-radius: 24px;
  background: linear-gradient(125deg, #06172a 0%, #0b4868 58%, #0a7a9f 100%);
  box-shadow: 0 20px 55px rgba(3,27,44,.22);
  color: white;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}
.hero:after {
  content:""; position:absolute; width:310px; height:310px; right:-95px; top:-130px;
  border-radius:50%; border:42px solid rgba(255,255,255,.08);
}
.hero h1 { margin:0 0 .55rem 0; font-size:clamp(2rem,4vw,3.7rem); line-height:1.02; }
.hero p { max-width:900px; margin:.35rem 0; color:rgba(255,255,255,.88); font-size:1.02rem; }
.hero .credit { margin-top:1.05rem; font-weight:700; color:#d7f7ff; }
.badge { display:inline-block; padding:.38rem .72rem; margin:0 .35rem .45rem 0;
  border:1px solid rgba(255,255,255,.25); border-radius:999px; background:rgba(255,255,255,.09); font-size:.82rem; color:white; }
.callout { border:1px solid rgba(11,111,153,.18); border-left:5px solid #0b6f99; border-radius:14px;
  padding:1rem 1.1rem; background:rgba(234,250,255,.62); margin:.5rem 0 1rem; }
.notice { border:1px solid rgba(11,111,153,.18); border-radius:16px; padding:1.05rem 1.15rem;
  background:rgba(255,255,255,.7); }
.footer { margin-top:2.4rem; padding:1.2rem; text-align:center; border-top:1px solid rgba(120,130,140,.18); color:rgba(85,95,105,.95); }
[data-testid="stMetric"] { border:1px solid rgba(120,130,140,.16); border-radius:16px; padding:.75rem .95rem; background:rgba(255,255,255,.65); }
</style>
""",
    unsafe_allow_html=True,
)


def standardize_light(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        c: ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
        for c in df.columns
    }
    out = df.rename(columns=renamed).copy()
    if "athlete_id" not in out:
        out["athlete_id"] = "swimmer"
    if "session_id" not in out:
        out["session_id"] = "uploaded_session_001"
    if "stroke_label" not in out:
        out["stroke_label"] = "unknown"
    return out


def read_upload_light(payload: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = BytesIO(payload)
    if suffix == ".csv":
        return standardize_light(pd.read_csv(buffer))
    if suffix == ".jsonl":
        return standardize_light(pd.read_json(buffer, lines=True))
    if suffix == ".json":
        try:
            return standardize_light(pd.read_json(buffer))
        except ValueError:
            buffer.seek(0)
            obj = json.loads(buffer.read().decode("utf-8"))
            return standardize_light(pd.DataFrame(obj))
    raise ValueError("Upload a CSV, JSON, or JSONL watch export.")


def lightweight_preview(df: pd.DataFrame) -> dict[str, float]:
    """Calculate quick, non-AI preview metrics without importing torch."""
    out = standardize_light(df)
    duration_min = 0.0
    if "timestamp" in out:
        ts = pd.to_datetime(out["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            duration_min = float((ts.max() - ts.min()).total_seconds() / 60.0)
    distance = pd.to_numeric(out.get("distance_m", pd.Series(dtype=float)), errors="coerce").dropna()
    hr = pd.to_numeric(out.get("heart_rate_bpm", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "samples": float(len(out)),
        "duration_min": duration_min,
        "distance_m": float(max(distance.iloc[-1] - distance.iloc[0], 0.0)) if len(distance) >= 2 else 0.0,
        "avg_hr": float(hr.mean()) if len(hr) else float("nan"),
        "sessions": float(out["session_id"].astype(str).nunique()),
    }


@st.cache_resource(show_spinner=False)
def load_ai_stack():
    """Import the heavy analytics stack only after the user requests analysis."""
    from src.swim_ai import load_model_bundle

    return load_model_bundle(MODELS_DIR)


@st.cache_data(show_spinner=False)
def run_full_analysis(raw_df: pd.DataFrame, profile_values: tuple, session_id: str):
    """Run the heavy pipeline only after the app has already rendered."""
    from src.swim_ai import AthleteProfile, analyze_watch_data

    profile = AthleteProfile(
        athlete_id=str(profile_values[0]),
        age_years=int(profile_values[1]),
        mass_kg=float(profile_values[2]),
        resting_hr_bpm=float(profile_values[3]),
        pool_length_m=float(profile_values[4]),
    )
    models = load_ai_stack()
    return analyze_watch_data(
        raw_watch_df=raw_df,
        models=models,
        profile=profile,
        session_id=str(session_id),
    )


def build_download_bundle(result) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("health_report.json", json.dumps(result.report, indent=2, default=str))
        archive.writestr("window_predictions.csv", result.windows.to_csv(index=False))
        archive.writestr("clean_session_data.csv", result.session_data.to_csv(index=False))
        archive.writestr("data_quality.csv", result.quality_table.to_csv(index=False))
        archive.writestr(
            "README.txt",
            "Kevin Sun AI Swimming Health App\nAuthor: Kevin Sun\nMentor: Dr. Qingyang Xiao\n\n"
            "Educational prototype only. Not a medical device or diagnosis.\n",
        )
    return buffer.getvalue()


def reset_analysis_if_input_changed(signature: str) -> None:
    if st.session_state.get("analysis_signature") != signature:
        st.session_state.pop("analysis_result", None)


# Visitor count is intentionally lightweight. Supabase uses a short fail-fast timeout;
# otherwise local SQLite keeps the app usable immediately.
try:
    app_secrets = st.secrets
except Exception:
    app_secrets = {}
visitor_count, counter_backend = register_visit(
    st.session_state,
    secrets=app_secrets,
    database_path=ROOT / ".app_data" / "visitor_counter.sqlite3",
)

st.markdown(
    f"""
<div class="hero">
  <span class="badge">Smartwatch analytics</span>
  <span class="badge">Machine learning</span>
  <span class="badge">Deep learning</span>
  <span class="badge">Reinforcement learning</span>
  <h1>AI Swimming Health App</h1>
  <p>Transform wearable swimming signals into stroke recognition, health-oriented summaries, interactive visual analytics, animated swimmer replay, and safety-constrained coaching guidance.</p>
  <p class="credit">Author: {AUTHOR_NAME} &nbsp;&nbsp;•&nbsp;&nbsp; Mentor: {MENTOR_NAME}</p>
</div>
""",
    unsafe_allow_html=True,
)

left_intro, right_intro = st.columns([5, 1.2], vertical_alignment="center")
with left_intro:
    st.markdown(
        """
<div class="callout">
<strong>Research and education prototype.</strong> The application does not diagnose medical conditions, certify swimming safety, or replace qualified supervision. Wearable and underwater sensor measurements can contain error.
</div>
""",
        unsafe_allow_html=True,
    )
with right_intro:
    st.metric("App visitors", f"{visitor_count:,}")
    st.caption(f"Count backend: {counter_backend}")

with st.sidebar:
    st.markdown("## 🏊 Analysis setup")
    data_source = st.radio(
        "Smartwatch data source",
        ["Built-in demonstration", "Upload watch export"],
        index=0,
    )

    raw_watch_df: pd.DataFrame | None = None
    data_signature = ""

    if data_source == "Built-in demonstration":
        if not SAMPLE_PATH.exists():
            st.error("The bundled demonstration CSV is missing.")
        else:
            raw_watch_df = pd.read_csv(SAMPLE_PATH)
            data_signature = "demo:bundled-kevin-session-v2"
            st.success("Demo data loaded immediately — no AI model loading occurs until you click Run Full AI Analysis.")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV, JSON, or JSONL",
            type=["csv", "json", "jsonl"],
        )
        if SAMPLE_PATH.exists():
            st.download_button(
                "Download sample watch CSV",
                data=SAMPLE_PATH.read_bytes(),
                file_name=SAMPLE_PATH.name,
                mime="text/csv",
                use_container_width=True,
            )
        if uploaded_file is not None:
            payload = uploaded_file.getvalue()
            data_signature = f"upload:{uploaded_file.name}:{hashlib.sha256(payload).hexdigest()[:16]}"
            try:
                raw_watch_df = read_upload_light(payload, uploaded_file.name)
                st.success(f"Loaded {len(raw_watch_df):,} raw sensor rows.")
            except Exception as exc:
                st.error(f"The uploaded file could not be read: {exc}")

    st.markdown("---")
    st.markdown("### Swimmer profile")
    athlete_id = st.text_input("Athlete ID", value="kevin_demo")
    age_years = int(st.number_input("Age (years)", 10, 90, 17, 1))
    mass_kg = float(st.number_input("Mass (kg)", 25.0, 250.0, 65.0, 0.5))
    resting_hr_bpm = float(st.number_input("Resting heart rate (bpm)", 35.0, 120.0, 60.0, 1.0))
    pool_length_m = float(st.number_input("Pool length (m)", 10.0, 100.0, 25.0, 1.0))

    session_options: list[str] = []
    if raw_watch_df is not None:
        preview = standardize_light(raw_watch_df)
        session_options = preview["session_id"].astype(str).dropna().drop_duplicates().tolist()
    if not session_options:
        session_options = ["uploaded_session_001"]
    selected_session = st.selectbox("Session", session_options)

    profile_signature = (athlete_id.strip() or "swimmer", age_years, mass_kg, resting_hr_bpm, pool_length_m, selected_session)
    current_signature = repr((data_signature, profile_signature))
    reset_analysis_if_input_changed(current_signature)

    run_clicked = st.button(
        "🚀 Run Full AI Analysis",
        type="primary",
        use_container_width=True,
        disabled=raw_watch_df is None,
    )
    if run_clicked:
        with st.spinner("Starting AI models only now — cleaning signals, running ensemble inference, and generating the report..."):
            try:
                result = run_full_analysis(
                    standardize_light(raw_watch_df),
                    profile_signature[:5],
                    selected_session,
                )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_signature"] = current_signature
                st.success("Full AI analysis complete.")
            except Exception as exc:
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("analysis_signature", None)
                st.error(f"Analysis could not be completed: {exc}")

if raw_watch_df is None:
    st.info("Choose the built-in demonstration or upload a smartwatch export to begin.")
    st.stop()

# Fast preview is always displayed before AI analysis.
preview_stats = lightweight_preview(raw_watch_df)
metric_cols = st.columns(5)
metric_cols[0].metric("Raw samples", f"{int(preview_stats['samples']):,}")
metric_cols[1].metric("Duration", f"{preview_stats['duration_min']:.1f} min")
metric_cols[2].metric("Distance", f"{preview_stats['distance_m']:.0f} m")
metric_cols[3].metric("Avg HR", "—" if np.isnan(preview_stats["avg_hr"]) else f"{preview_stats['avg_hr']:.0f} bpm")
metric_cols[4].metric("Sessions", f"{int(preview_stats['sessions']):,}")

result = st.session_state.get("analysis_result")
if result is None or st.session_state.get("analysis_signature") != current_signature:
    st.markdown(
        """
<div class="notice">
<h3>Fast-start mode is active</h3>
<p>The website is already loaded. Heavy PyTorch model weights are not loaded at startup. Click <strong>Run Full AI Analysis</strong> in the sidebar when you are ready to run stroke classification, deep learning, reinforcement-learning coaching, health analytics, and visualizations.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### Watch-data preview")
    st.dataframe(raw_watch_df.head(30), use_container_width=True, hide_index=True)
    st.markdown("### Required sensor columns")
    missing = [c for c in REQUIRED_COLUMNS if c not in standardize_light(raw_watch_df).columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.success("Required accelerometer and gyroscope channels are present.")
    st.markdown(
        f"**Student:** {AUTHOR_NAME}  \n**Mentor:** {MENTOR_NAME}  \n**Deployment design:** lightweight landing page → on-demand AI stack → cached inference."
    )
    st.stop()

def get_result_dependencies():
    """Import heavy analytics/visualization modules only after analysis exists."""
    from src.swim_ai import (
        AUTHOR_NAME as core_author_name,
        MENTOR_NAME as core_mentor_name,
        PROJECT_NAME as core_project_name,
        dataframe_to_csv_bytes,
        report_to_json_bytes,
        safe_metric,
        seconds_to_pace_label,
    )
    from src.visuals import (
        animated_swimmer_html,
        heart_rate_zone_figure,
        motion_timeline_figure,
        score_figure,
        sensor_timeline_figure,
        stroke_distribution_figure,
        stroke_timeline_figure,
    )
    return {
        "core_author_name": core_author_name,
        "core_mentor_name": core_mentor_name,
        "core_project_name": core_project_name,
        "dataframe_to_csv_bytes": dataframe_to_csv_bytes,
        "report_to_json_bytes": report_to_json_bytes,
        "safe_metric": safe_metric,
        "seconds_to_pace_label": seconds_to_pace_label,
        "animated_swimmer_html": animated_swimmer_html,
        "heart_rate_zone_figure": heart_rate_zone_figure,
        "motion_timeline_figure": motion_timeline_figure,
        "score_figure": score_figure,
        "sensor_timeline_figure": sensor_timeline_figure,
        "stroke_distribution_figure": stroke_distribution_figure,
        "stroke_timeline_figure": stroke_timeline_figure,
    }


deps = get_result_dependencies()
report = result.report
summary = report["summary"]
recommendation = report["coaching_recommendation"]

def format_action(action: str) -> str:
    return action.replace("_", " ").title()

st.markdown(f"## Session analysis · `{report['session_id']}`")
metric_columns = st.columns(6)
metric_columns[0].metric("Duration", deps["safe_metric"](summary["duration_minutes"], " min", 2))
metric_columns[1].metric("Distance", deps["safe_metric"](summary["distance_m"], " m", 1))
metric_columns[2].metric("Average HR", deps["safe_metric"](summary["average_heart_rate_bpm"], " bpm", 1))
metric_columns[3].metric("Pace", deps["seconds_to_pace_label"](summary["pace_sec_per_100m"]))
metric_columns[4].metric("Technique", deps["safe_metric"](summary["technique_consistency_score_0_to_100"], "/100", 1))
metric_columns[5].metric("Data quality", deps["safe_metric"](summary["data_quality_score_0_to_100"], "/100", 1))

st.info(
    f"AI coaching action: **{format_action(recommendation['final_safety_constrained_action'])}** — {recommendation['guidance']}"
)

if report["flags"]:
    flag_columns = st.columns(min(len(report["flags"]), 3))
    for idx, flag in enumerate(report["flags"]):
        with flag_columns[idx % len(flag_columns)]:
            with st.container(border=True):
                st.markdown(f"**{flag['title']}**")
                st.write(flag["detail"])

tab_dashboard, tab_ai, tab_health, tab_data, tab_about = st.tabs(
    ["Dashboard", "AI Stroke Analysis", "Health & Training", "Data Explorer & Downloads", "About the Project"]
)

with tab_dashboard:
    a, b = st.columns(2)
    with a:
        st.plotly_chart(deps["stroke_distribution_figure"](report), use_container_width=True)
    with b:
        st.plotly_chart(deps["score_figure"](report), use_container_width=True)
    st.plotly_chart(deps["sensor_timeline_figure"](result.session_data), use_container_width=True)
    st.caption("Prototype technique/fatigue indicators are signal-derived research features, not clinical measurements.")

with tab_ai:
    st.markdown("### Watch-driven swimmer animation")
    st.components.v1.html(deps["animated_swimmer_html"](result.session_data, result.windows), height=410, scrolling=False)
    st.plotly_chart(deps["stroke_timeline_figure"](result.session_data, result.windows), use_container_width=True)
    st.markdown("### Window-level predictions")
    display_columns = [
        "window_start", "window_end", "predicted_stroke", "prediction_confidence",
        "stroke_rate_spm", "heart_rate_mean_bpm", "spo2_min_pct",
    ]
    st.dataframe(result.windows[display_columns], use_container_width=True, hide_index=True)

with tab_health:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(deps["heart_rate_zone_figure"](result.session_data, summary), use_container_width=True)
    with c2:
        st.plotly_chart(deps["motion_timeline_figure"](result.session_data), use_container_width=True)
    st.json({"summary": summary, "flags": report["flags"], "recommendation": recommendation})

with tab_data:
    st.markdown("### Cleaned watch data")
    st.dataframe(result.session_data.head(500), use_container_width=True, hide_index=True)
    d = st.columns(4)
    d[0].download_button("Health report JSON", deps["report_to_json_bytes"](report), f"{report['session_id']}_health_report.json", "application/json", use_container_width=True)
    d[1].download_button("Predictions CSV", deps["dataframe_to_csv_bytes"](result.windows), f"{report['session_id']}_window_predictions.csv", "text/csv", use_container_width=True)
    d[2].download_button("Clean data CSV", deps["dataframe_to_csv_bytes"](result.session_data), f"{report['session_id']}_clean_watch_data.csv", "text/csv", use_container_width=True)
    d[3].download_button("Complete result ZIP", build_download_bundle(result), f"{report['session_id']}_swimming_ai_results.zip", "application/zip", use_container_width=True)
    with st.expander("Accepted smartwatch columns"):
        st.markdown("**Required:** `timestamp`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`  \n**Optional:** `athlete_id`, `session_id`, `heart_rate_bpm`, `spo2_pct`, `water_temp_c`, `distance_m`, `lap_id`, `stroke_count`, `stroke_label`")

with tab_about:
    st.markdown(
        f"""
## {deps["core_project_name"]}

**Author / Student:** {deps["core_author_name"]}  
**Mentor:** {deps["core_mentor_name"]}

### Fast startup architecture
The application deliberately separates the lightweight website shell from the heavy analytics stack. Streamlit renders the landing page, visitor count, demo preview, upload controls, and profile controls without importing PyTorch model classes. The AI stack is imported and cached only after **Run Full AI Analysis** is clicked.

### AI pipeline
1. Smartwatch data validation and cleaning.
2. Motion-window feature extraction.
3. Random Forest classical machine learning.
4. Feature-based deep neural network.
5. 1D CNN + bidirectional GRU sequence deep learning.
6. Confidence-weighted ensemble stroke classification.
7. Q-learning coaching policy with deterministic safety overrides.
8. Health/training summaries, plots, animated swimmer replay, and downloads.

The included pretrained artifacts are demonstration models trained on structured synthetic watch signals. They are not evidence of clinical diagnostic performance.
"""
    )

st.markdown(
    f"""
<div class="footer"><strong>{PROJECT_NAME}</strong><br/>Author: {AUTHOR_NAME} &nbsp;•&nbsp; Mentor: {MENTOR_NAME}<br/>Educational AI prototype — not a medical device, diagnosis, or substitute for qualified supervision.</div>
""",
    unsafe_allow_html=True,
)
