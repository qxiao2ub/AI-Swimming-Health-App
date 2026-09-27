# AI Swimming Health App

**Author / Student:** Kevin Sun  
**Mentor:** Dr. Qingyang Xiao

A Streamlit-ready AI application that converts smartwatch swimming signals into stroke classifications, interactive health and training summaries, visual analytics, an animated swimmer, and safety-constrained coaching guidance.

> **Responsible-use notice:** This repository is an educational and research prototype. It is not a medical device, diagnosis, medical clearance, or substitute for qualified clinical care or swimming supervision.

## Product highlights

- Accepts Kevin's watch exports in CSV, JSON, or JSONL format.
- Standardizes common sensor-column aliases and validates the required schema.
- Repairs short missing runs, resamples irregular timestamps, and calculates data-quality diagnostics.
- Uses overlapping four-second windows for motion analysis.
- Combines three pretrained classifiers:
  - Random Forest classical machine learning;
  - feature-based deep neural network;
  - 1D CNN plus bidirectional GRU deep sequence model.
- Produces predicted stroke, confidence, stroke-rate, heart-rate, oxygen-sensor, distance, pace, fatigue, technique, and data-quality summaries.
- Applies a tabular reinforcement-learning coaching policy with deterministic safety overrides.
- Displays interactive Plotly charts and a watch-driven swimmer animation.
- Exports a health report, window predictions, cleaned data, and a complete ZIP result bundle.
- Shows an app visitor count, counted once per Streamlit browser session.

## Repository structure

```text
.
├── app.py                                  # Streamlit application
├── src/
│   ├── swim_ai.py                          # Sensor pipeline, models, health analytics, RL inference
│   ├── visuals.py                          # Plotly charts and browser animation
│   └── visitor_counter.py                  # Supabase counter with SQLite fallback
├── models/                                 # Pretrained inference artifacts
├── sample_data/
│   └── kevin_demo_watch_session.csv        # Upload-ready sample watch data
├── notebooks/
│   └── Kevin_Sun_AI_Swimming_Health_App_Colab.ipynb
├── scripts/
│   └── train_models.py                     # Reproducible model-training script
├── docs/
│   └── supabase_visitor_counter.sql         # Optional persistent counter setup
├── tests/                                  # Automated smoke, pipeline, and Streamlit tests
├── .github/workflows/tests.yml             # GitHub Actions continuous integration
├── requirements.txt
└── runtime.txt
```

## Run locally

Python 3.13 is recommended because the committed model artifacts were produced with the versions pinned in `requirements.txt`.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens with a complete demonstration session. Use the sidebar to upload watch data and edit the swimmer profile.

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository and upload every file and folder from this project ZIP.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Select `app.py` as the main file.
4. Open **Advanced settings** and select Python 3.13 to match the model-build environment. The included `runtime.txt` also documents the intended runtime for other hosting platforms.
5. Deploy. The root `requirements.txt`, committed model artifacts, and `.streamlit/config.toml` provide the remaining environment.
6. For a visitor count that survives restarts and redeployments, configure the optional Supabase secrets described below.

No model training occurs when the website starts. Streamlit loads the committed pretrained artifacts through `st.cache_resource`, which keeps inference responsive.

## Smartwatch upload schema

### Required columns

| Column | Meaning |
|---|---|
| `timestamp` | Date/time or parseable sample time |
| `acc_x`, `acc_y`, `acc_z` | Three-axis accelerometer readings |
| `gyro_x`, `gyro_y`, `gyro_z` | Three-axis gyroscope readings |

### Optional columns

| Column | Meaning |
|---|---|
| `athlete_id` | Swimmer identifier |
| `session_id` | Workout/session identifier |
| `heart_rate_bpm` | Watch heart-rate estimate |
| `spo2_pct` | Watch oxygen-saturation estimate |
| `water_temp_c` | Water-temperature estimate |
| `distance_m` | Cumulative distance |
| `lap_id` | Lap identifier |
| `stroke_count` | Cumulative stroke count |
| `stroke_label` | Optional ground-truth label |

The app also recognizes aliases such as `time`, `datetime`, `ax`, `gyroscope_z`, `heart_rate`, `spo2`, `distance`, `lap`, and `stroke`.

## Visitor counter

A visit is registered once per Streamlit browser session.

### Default mode: SQLite

The app works immediately with a local SQLite database. This count persists across ordinary Streamlit reruns within the same running deployment, but a managed cloud restart or redeployment can reset it.

### Persistent mode: Supabase

1. Create a free Supabase project.
2. Run [`docs/supabase_visitor_counter.sql`](docs/supabase_visitor_counter.sql) in the Supabase SQL editor.
3. In Streamlit Cloud, open **App settings → Secrets** and add:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_ANON_KEY = "YOUR_SUPABASE_ANON_KEY"
COUNTER_SLUG = "kevin-sun-ai-swimming-health"
```

The app automatically selects Supabase when those secrets are available and falls back to SQLite when they are not.

## AI pipeline

1. **Ingestion:** read CSV/JSON/JSONL watch exports and standardize column names.
2. **Signal preparation:** parse timestamps, remove duplicates, interpolate short gaps, resample to 20 Hz, and calculate acceleration and gyroscope magnitude.
3. **Windowing:** form four-second windows with two-second overlap.
4. **Feature engineering:** calculate descriptive statistics, energy, percentiles, dominant frequency, spectral entropy, zero-crossing rate, and cross-axis correlation.
5. **Classical ML:** classify motion windows with a Random Forest.
6. **Feature DNN:** classify engineered features with a multilayer neural network.
7. **Sequence model:** classify raw eight-channel motion sequences with a 1D CNN and bidirectional GRU.
8. **Ensemble:** combine model probabilities using validation macro-F1 weights.
9. **Health analytics:** summarize duration, distance, pace, watch heart rate, watch SpO2, stroke rate, technique consistency, fatigue trend, training load, and data quality.
10. **RL coaching:** select a demonstration coaching action, with safety rules able to override the learned policy.

## Model limitations

The included classifiers were trained and evaluated on structured synthetic watch signals generated by the attached notebook. Their perfect synthetic test scores demonstrate pipeline execution, not real-world clinical or sports-performance accuracy. Before any real deployment, Kevin's watch requires data collection from diverse swimmers, expert labeling, subject-independent validation, calibration, bias analysis, sensor reliability testing, and appropriate medical-device review when applicable.

## Retrain the models

```bash
python scripts/train_models.py
```

This regenerates the artifacts in `models/`. Retraining is not required to launch the included Streamlit app.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The included GitHub Actions workflow runs compilation and the full test suite on every push and pull request.

## Credits

- **Author / Student:** Kevin Sun
- **Mentor:** Dr. Qingyang Xiao
