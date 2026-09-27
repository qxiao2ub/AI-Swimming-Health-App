"""Streamlit entry point for the Kevin Sun AI Swimming Health App.

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import hashlib
import json
import zipfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.swim_ai import (
    AUTHOR_NAME,
    MENTOR_NAME,
    PROJECT_NAME,
    AthleteProfile,
    analyze_watch_data,
    clean_watch_data,
    dataframe_to_csv_bytes,
    generate_synthetic_watch_data,
    load_model_bundle,
    read_watch_file,
    report_to_json_bytes,
    safe_metric,
    seconds_to_pace_label,
    standardize_column_names,
)
from src.visitor_counter import register_visit
from src.visuals import (
    animated_swimmer_html,
    heart_rate_zone_figure,
    motion_timeline_figure,
    score_figure,
    sensor_timeline_figure,
    stroke_distribution_figure,
    stroke_timeline_figure,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
SAMPLE_PATH = ROOT / "sample_data" / "kevin_demo_watch_session.csv"

st.set_page_config(
    page_title="AI Swimming Health App | Kevin Sun",
    page_icon="🏊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --navy: #071b2f;
  --blue: #0b6f99;
  --cyan: #36c5e8;
  --foam: #eafaff;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 90% 0%, rgba(54,197,232,.12), transparent 28%),
              radial-gradient(circle at 0% 25%, rgba(11,111,153,.08), transparent 32%);
}
.hero {
  padding: 2.0rem 2.1rem;
  border-radius: 24px;
  background: linear-gradient(125deg, #06172a 0%, #0b4868 58%, #0a7a9f 100%);
  box-shadow: 0 20px 55px rgba(3,27,44,.22);
  color: white;
  margin-bottom: 1.1rem;
  position: relative;
  overflow: hidden;
}
.hero:after {
  content: "";
  position: absolute;
  width: 310px;
  height: 310px;
  right: -95px;
  top: -130px;
  border-radius: 50%;
  border: 42px solid rgba(255,255,255,.08);
}
.hero h1 { margin: 0 0 .55rem 0; font-size: clamp(2rem, 4vw, 3.7rem); line-height: 1.02; }
.hero p { max-width: 900px; margin: .35rem 0; color: rgba(255,255,255,.88); font-size: 1.02rem; }
.hero .credit { margin-top: 1.05rem; font-weight: 650; color: #d7f7ff; }
.badge {
  display: inline-block;
  padding: .38rem .72rem;
  margin: 0 .35rem .45rem 0;
  border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px;
  background: rgba(255,255,255,.09);
  font-size: .82rem;
  color: white;
}
.callout {
  border: 1px solid rgba(11,111,153,.18);
  border-left: 5px solid #0b6f99;
  border-radius: 14px;
  padding: 1rem 1.1rem;
  background: rgba(234,250,255,.62);
  margin: .5rem 0 1rem;
}
.recommendation {
  border-radius: 18px;
  padding: 1.15rem 1.25rem;
  background: linear-gradient(135deg, rgba(11,111,153,.12), rgba(54,197,232,.10));
  border: 1px solid rgba(11,111,153,.20);
}
.recommendation h3 { margin: 0 0 .35rem 0; }
.small-muted { color: rgba(100,110,120,.90); font-size: .88rem; }
.footer {
  margin-top: 2.4rem;
  padding: 1.2rem;
  text-align: center;
  border-top: 1px solid rgba(120,130,140,.18);
  color: rgba(85,95,105,.95);
}
[data-testid="stMetric"] {
  border: 1px solid rgba(120,130,140,.16);
  border-radius: 16px;
  padding: .85rem 1rem;
  background: rgba(255,255,255,.62);
}
[data-testid="stSidebar"] { border-right: 1px solid rgba(120,130,140,.12); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading pretrained AI models...")
def get_model_bundle():
    return load_model_bundle(MODELS_DIR)


@st.cache_data(show_spinner=False)
def demo_data(seed: int) -> pd.DataFrame:
    return generate_synthetic_watch_data(n_sessions=1, seed=seed)


def build_download_bundle(result) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "health_report.json",
            json.dumps(result.report, indent=2, default=str),
        )
        archive.writestr("window_predictions.csv", result.windows.to_csv(index=False))
        archive.writestr("clean_session_data.csv", result.session_data.to_csv(index=False))
        archive.writestr("data_quality.csv", result.quality_table.to_csv(index=False))
        archive.writestr(
            "README.txt",
            "Kevin Sun AI Swimming Health App\n"
            "Author: Kevin Sun\n"
            "Mentor: Dr. Qingyang Xiao\n\n"
            "Educational prototype only. Not a medical device or diagnosis.\n",
        )
    return buffer.getvalue()


def format_action(action: str) -> str:
    return action.replace("_", " ").title()


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
  <div>
    <span class="badge">Smartwatch analytics</span>
    <span class="badge">Machine learning</span>
    <span class="badge">Deep neural networks</span>
    <span class="badge">Reinforcement learning</span>
  </div>
  <h1>AI Swimming Health App</h1>
  <p>Transform accelerometer, gyroscope, heart-rate, oxygen-sensor, distance, and lap data into stroke predictions, health-aware summaries, visual analytics, and safety-constrained coaching guidance.</p>
  <p class="credit">Author: {AUTHOR_NAME} &nbsp;&nbsp;•&nbsp;&nbsp; Mentor: {MENTOR_NAME}</p>
</div>
""",
    unsafe_allow_html=True,
)

intro_left, intro_right = st.columns([5, 1.25], vertical_alignment="center")
with intro_left:
    st.markdown(
        """
<div class="callout">
<strong>Research and education prototype.</strong> This application does not diagnose medical conditions, replace a clinician, certify swimming safety, or provide autonomous supervision. Underwater wrist-sensor readings can be inaccurate.
</div>
""",
        unsafe_allow_html=True,
    )
with intro_right:
    st.metric(
        "App visitors",
        f"{visitor_count:,}",
        help=(
            "One count is recorded per Streamlit browser session. Configure the included "
            "Supabase option for persistence across cloud restarts; otherwise the app uses SQLite."
        ),
    )
    st.caption(f"Counter backend: {counter_backend}")

with st.sidebar:
    st.markdown("## 🏊 Analysis setup")
    data_source = st.radio(
        "Smartwatch data source",
        ["Built-in demonstration", "Upload watch export"],
        help="Use the demo immediately or upload a CSV, JSON, or JSONL file from Kevin's watch.",
    )

    raw_watch_df: pd.DataFrame | None = None
    data_signature = ""
    uploaded_name = ""

    if data_source == "Built-in demonstration":
        demo_seed = int(st.number_input("Demonstration seed", 1, 100000, 42, 1))
        raw_watch_df = demo_data(demo_seed)
        data_signature = f"demo:{demo_seed}"
        st.caption("Structured synthetic signals reproduce the notebook's five-class demonstration.")
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
            uploaded_payload = uploaded_file.getvalue()
            uploaded_name = uploaded_file.name
            data_signature = (
                f"upload:{uploaded_name}:"
                f"{hashlib.sha256(uploaded_payload).hexdigest()[:16]}"
            )
            try:
                raw_watch_df = read_watch_file(uploaded_payload, filename=uploaded_name)
            except Exception as exc:
                st.error(f"The uploaded file could not be read: {exc}")

    st.markdown("---")
    st.markdown("### Swimmer profile")
    athlete_id = st.text_input("Athlete ID", value="kevin_demo")
    age_years = int(
        st.number_input(
            "Age (years)",
            min_value=10,
            max_value=90,
            value=17,
            help="Editable demonstration input; school level alone is not used to infer age.",
        )
    )
    mass_kg = float(st.number_input("Mass (kg)", 25.0, 250.0, 65.0, 0.5))
    resting_hr_bpm = float(st.number_input("Resting heart rate (bpm)", 35.0, 120.0, 60.0, 1.0))
    pool_length_m = float(st.number_input("Pool length (m)", 10.0, 100.0, 25.0, 1.0))

    session_options: list[str] = []
    if raw_watch_df is not None:
        preview = standardize_column_names(raw_watch_df.copy())
        if "session_id" in preview:
            session_options = preview["session_id"].astype(str).dropna().drop_duplicates().tolist()
        if not session_options:
            session_options = ["uploaded_session_001"]

    selected_session = st.selectbox(
        "Session",
        session_options if session_options else ["Upload data first"],
        disabled=not bool(session_options),
    )

    profile = AthleteProfile(
        athlete_id=athlete_id.strip() or "swimmer",
        age_years=age_years,
        mass_kg=mass_kg,
        resting_hr_bpm=resting_hr_bpm,
        pool_length_m=pool_length_m,
    )
    profile_signature = (
        profile.athlete_id,
        profile.age_years,
        profile.mass_kg,
        profile.resting_hr_bpm,
        profile.pool_length_m,
        selected_session,
    )
    current_signature = repr((data_signature, profile_signature))

    analyze_clicked = st.button(
        "Run AI analysis",
        type="primary",
        use_container_width=True,
        disabled=raw_watch_df is None,
    )

if raw_watch_df is None:
    st.info(
        "Upload a watch export in the sidebar, or switch to the built-in demonstration to explore the app."
    )
    st.markdown(
        """
### Minimum upload schema

The file must contain a timestamp and six motion channels: `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, and `gyro_z`. Optional columns include `heart_rate_bpm`, `spo2_pct`, `distance_m`, `lap_id`, `stroke_count`, `session_id`, and `stroke_label`.
"""
    )
    st.stop()

stored_signature = st.session_state.get("analysis_signature")
should_run = analyze_clicked or (
    data_source == "Built-in demonstration"
    and ("analysis_result" not in st.session_state or stored_signature != current_signature)
)

if should_run:
    try:
        with st.spinner("Running sensor cleaning, AI inference, health analytics, and coaching policy..."):
            models = get_model_bundle()
            analysis_result = analyze_watch_data(
                raw_watch_df=raw_watch_df,
                models=models,
                profile=profile,
                session_id=str(selected_session),
            )
        st.session_state["analysis_result"] = analysis_result
        st.session_state["analysis_signature"] = current_signature
    except Exception as exc:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_signature", None)
        st.error(f"Analysis could not be completed: {exc}")

result = st.session_state.get("analysis_result")
if result is None or st.session_state.get("analysis_signature") != current_signature:
    st.info("The inputs changed. Select **Run AI analysis** in the sidebar to refresh the results.")
    try:
        preview_clean = clean_watch_data(raw_watch_df)
        st.dataframe(preview_clean.head(25), use_container_width=True, hide_index=True)
    except Exception:
        pass
    st.stop()

report = result.report
summary = report["summary"]
recommendation = report["coaching_recommendation"]

st.markdown(f"## Session analysis · `{report['session_id']}`")
metric_columns = st.columns(6)
metric_columns[0].metric("Duration", safe_metric(summary["duration_minutes"], " min", 2))
metric_columns[1].metric("Distance", safe_metric(summary["distance_m"], " m", 1))
metric_columns[2].metric("Average HR", safe_metric(summary["average_heart_rate_bpm"], " bpm", 1))
metric_columns[3].metric("Pace", seconds_to_pace_label(summary["pace_sec_per_100m"]))
metric_columns[4].metric(
    "Technique",
    safe_metric(summary["technique_consistency_score_0_to_100"], "/100", 1),
)
metric_columns[5].metric(
    "Data quality",
    safe_metric(summary["data_quality_score_0_to_100"], "/100", 1),
)

st.markdown(
    f"""
<div class="recommendation">
  <h3>AI coaching action: {format_action(recommendation['final_safety_constrained_action'])}</h3>
  <div>{recommendation['guidance']}</div>
  <div class="small-muted" style="margin-top:.55rem;">Learned policy: {format_action(recommendation['learned_action'])}{' · ' + recommendation['override_reason'] if recommendation.get('override_reason') else ''}</div>
</div>
""",
    unsafe_allow_html=True,
)

flag_columns = st.columns(min(len(report["flags"]), 3))
for index, flag in enumerate(report["flags"]):
    with flag_columns[index % len(flag_columns)]:
        with st.container(border=True):
            st.markdown(f"**{flag['title']}**")
            st.write(flag["detail"])

tab_dashboard, tab_ai, tab_health, tab_data, tab_about = st.tabs(
    [
        "Dashboard",
        "AI Stroke Analysis",
        "Health & Training",
        "Data Explorer & Downloads",
        "About the Project",
    ]
)

with tab_dashboard:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(stroke_distribution_figure(report), use_container_width=True)
    with chart_right:
        st.plotly_chart(score_figure(report), use_container_width=True)
    st.plotly_chart(sensor_timeline_figure(result.session_data), use_container_width=True)
    st.caption(
        "The technique and fatigue scores are prototype signal-derived indicators. They are not clinical measurements."
    )

with tab_ai:
    st.markdown("### Watch-driven swimmer animation")
    components.html(
        animated_swimmer_html(result.session_data, result.windows),
        height=410,
        scrolling=False,
    )
    st.plotly_chart(
        stroke_timeline_figure(result.session_data, result.windows),
        use_container_width=True,
    )

    model_info = report["model_information"]
    details_left, details_right = st.columns([1.05, 1])
    with details_left:
        st.markdown("### Window-level predictions")
        display_columns = [
            "window_start",
            "window_end",
            "predicted_stroke",
            "prediction_confidence",
            "stroke_rate_spm",
            "heart_rate_mean_bpm",
            "spo2_min_pct",
        ]
        st.dataframe(
            result.windows[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config={
                "prediction_confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                )
            },
        )
    with details_right:
        st.markdown("### Ensemble information")
        weights = model_info["ensemble_weights"]
        st.json(
            {
                "mean_window_confidence": model_info["mean_window_confidence"],
                "ensemble_weights": weights,
                "training_source": model_info["training_source"],
            },
            expanded=True,
        )
        st.info(
            "The bundled models were trained on structured synthetic watch data from the source notebook. "
            "Perfect synthetic test scores do not establish accuracy on real swimmers or Kevin's custom hardware."
        )

with tab_health:
    health_left, health_right = st.columns(2)
    with health_left:
        st.plotly_chart(heart_rate_zone_figure(report), use_container_width=True)
    with health_right:
        st.plotly_chart(motion_timeline_figure(result.session_data), use_container_width=True)

    st.markdown("### Health and training summary")
    health_rows = pd.DataFrame(
        [
            ("Maximum watch HR", safe_metric(summary["maximum_heart_rate_bpm"], " bpm", 1)),
            ("Estimated maximum HR", safe_metric(summary["estimated_max_heart_rate_bpm"], " bpm", 1)),
            ("Minimum SpO2 sensor value", safe_metric(summary["minimum_spo2_sensor_pct"], "%", 1)),
            ("Median stroke rate", safe_metric(summary["median_estimated_stroke_rate_spm"], " spm", 1)),
            ("Fatigue trend", safe_metric(summary["fatigue_trend_score_0_to_100"], "/100", 1)),
            ("Training-load index", safe_metric(summary["demo_training_load_index"], "", 2)),
        ],
        columns=["Measure", "Result"],
    )
    st.dataframe(health_rows, use_container_width=True, hide_index=True)

    with st.expander("Safety limitations and responsible-use notes", expanded=True):
        for limitation in report["limitations"]:
            st.markdown(f"- {limitation}")

with tab_data:
    st.markdown("### Data quality")
    st.dataframe(
        result.quality_table.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Cleaned session data")
    st.dataframe(
        result.session_data.head(500),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing the first {min(500, len(result.session_data)):,} of {len(result.session_data):,} cleaned samples."
    )

    download_columns = st.columns(4)
    download_columns[0].download_button(
        "Health report JSON",
        data=report_to_json_bytes(report),
        file_name=f"{report['session_id']}_health_report.json",
        mime="application/json",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Predictions CSV",
        data=dataframe_to_csv_bytes(result.windows),
        file_name=f"{report['session_id']}_window_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[2].download_button(
        "Clean data CSV",
        data=dataframe_to_csv_bytes(result.session_data),
        file_name=f"{report['session_id']}_clean_watch_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_columns[3].download_button(
        "Complete result ZIP",
        data=build_download_bundle(result),
        file_name=f"{report['session_id']}_swimming_ai_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

    with st.expander("Accepted smartwatch columns"):
        st.markdown(
            """
**Required:** `timestamp`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`

**Optional:** `athlete_id`, `session_id`, `heart_rate_bpm`, `spo2_pct`, `water_temp_c`, `distance_m`, `lap_id`, `stroke_count`, `stroke_label`

Common aliases such as `time`, `ax`, `gyroscope_z`, `heart_rate`, `spo2`, `distance`, and `stroke` are standardized automatically.
"""
        )

with tab_about:
    st.markdown(
        f"""
## {PROJECT_NAME}

**Author / Student:** {AUTHOR_NAME}  
**Mentor:** {MENTOR_NAME}

This Streamlit repository translates the attached Colab notebook into a deployable application. The pipeline validates smartwatch signals, repairs short missing runs, resamples data, creates overlapping motion windows, extracts signal features, and combines three classifiers:

1. a Random Forest for classical machine learning;
2. a feature-based deep neural network;
3. a 1D CNN plus bidirectional GRU for raw motion sequences.

Validation macro-F1 values determine the ensemble weights. A tabular Q-learning policy proposes one of four coaching actions, while deterministic safety rules can override the learned action when estimated intensity, fatigue, or technique quality warrants caution.
"""
    )
    st.markdown("### Repository capabilities")
    capability_columns = st.columns(3)
    for column, title, text in [
        (
            capability_columns[0],
            "Sensor intelligence",
            "CSV/JSON ingestion, schema aliases, validation, interpolation, resampling, quality checks, and motion windows.",
        ),
        (
            capability_columns[1],
            "AI analytics",
            "Classical ML, DNN, sequence deep learning, confidence-weighted ensemble, and RL coaching policy.",
        ),
        (
            capability_columns[2],
            "Product experience",
            "Visitor count, interactive charts, animation, health summaries, data explorer, and downloadable outputs.",
        ),
    ]:
        with column:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.write(text)

st.markdown(
    f"""
<div class="footer">
  <strong>{PROJECT_NAME}</strong><br/>
  Author: {AUTHOR_NAME} &nbsp;•&nbsp; Mentor: {MENTOR_NAME}<br/>
  Educational AI prototype — not a medical device, diagnosis, or substitute for qualified supervision.
</div>
""",
    unsafe_allow_html=True,
)
