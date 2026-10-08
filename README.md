# 🏊 AquaMind — AI Swimming Health App

**Author:** Kevin Sun (junior student)  
**Mentor:** Dr. Qingyang Xiao

A complete Streamlit web app built from Kevin Sun's AI-based swimming-health Colab prototype. Upload smartwatch movement signals or use a synthetic demo, classify strokes using a real ensemble of pretrained models, view health/training analytics, watch an animated swimmer, receive a safety-constrained coaching suggestion, and export results.

## Why this edition starts faster

The **old** Community Cloud requirements contained `torch==2.10.0`, `scikit-learn==1.8.0` and an older pandas version. The deployment screenshot stopped at `Processing dependencies` with Python 3.14.8. Lazy imports cannot solve an installation stall.

This rebuild uses only **Streamlit + NumPy + pandas + Plotly + Pillow** (plus PyArrow, required by Streamlit itself) for the website. The pretrained Random Forest, dense deep neural network, and 1D CNN/BiGRU were converted from their **real trained weights** to portable `.npz` arrays and evaluated with pure NumPy. No retraining, PyTorch wheel download, scikit-learn, SciPy, joblib, or ffmpeg is required at website boot time **or on any website page**. The model probability predictions were compared against the original PyTorch and scikit-learn implementations using the included 47-window demonstration recording (maximum difference below 1e-6). The original PyTorch training and notebook workflows are preserved as **optional** sources.

**Important:** No local build can guarantee Streamlit Cloud installation when an external service has an outage. The Python version shown in screenshots is chosen by Streamlit Cloud's deployment settings; it does not necessarily follow a repository `runtime.txt`. Select Python **3.13** if offered. This rebuild's dependencies can also be resolved on a current Python 3.14 environment.

## Replace the broken GitHub deployment

1. Unzip this repository. **Upload the extracted files and folders**, with `app.py` and `requirements.txt` in the GitHub repository **root** — not nested under an extra folder.
2. Replace the old files in `qxiao2ub/ai-swimming-health-app` or create a new public GitHub repository. In particular, **replace the old `requirements.txt`**: verify that it does **not** contain `torch`, `scikit-learn`, or a pinned `pandas==2.2.3`. Remove any old `pyproject.toml`, `Pipfile`, `uv.lock`, and `packages.txt` from the GitHub root if they were added by an earlier attempt and are not part of this ZIP.
3. Open [share.streamlit.io](https://share.streamlit.io) → app `ai-swimming-health` → Settings / Manage app → **Reboot** or **Clear cache and reboot** (wording can vary). On a new deployment, select the GitHub repository and `main` branch.
4. Set the Streamlit entry point to **`app.py`** (file name is case-sensitive). Select **Python 3.13** in Advanced settings if the UI offers it; a new deployment may be required to change the interpreter.
5. Wait for dependency installation, then the splash page should appear without doing model training. Click **Analyze swimming session** in the sidebar to see the full demo.
6. For a counter that survives restarts and deploys, follow [docs/VISITOR_COUNTER.md](docs/VISITOR_COUNTER.md).

**Do not** run `pip install -r requirements-training.txt` on Streamlit Cloud. `requirements.txt` is the only mandatory production requirements file.

## Website functions

| Function | Status |
|---|---|
| Smartwatch CSV/JSON/JSONL upload | ✅ |
| Synthetic sensor demo recording | ✅ |
| Validation, timestamp parsing, resampling, missing data repair | ✅ |
| 4-second IMU windows with 2-second stride | ✅ |
| 94 time/frequency/correlation features | ✅ |
| Original 260-tree Random Forest | ✅ NumPy inference |
| Original batch-normalized deep neural network | ✅ NumPy inference |
| Original 1D CNN + bidirectional GRU | ✅ NumPy inference |
| Confidence-weighted five-class stroke ensemble | ✅ |
| Heart rate, SpO₂ sensor, pace, distance, zone time | ✅ when watch fields exist |
| Consistency, fatigue and training load indices | ✅ research proxies |
| Q-learning simulation policy and rule-based overrides | ✅ |
| Interactive Plotly health, sensor and stroke charts | ✅ |
| Animated swimmer replay and downloadable HTML | ✅ |
| Downloadable GIF animation | ✅ generated on demand |
| Download JSON, CSV and full analysis ZIP | ✅ |
| Visitor access counter | ✅ local SQLite fallback / optional persistent Supabase |
| Original Colab training notebook and original Torch weights | ✅ for offline development |
| Live Bluetooth reception from Kevin's watch | ⚠️ Requires hardware-specific device/API integration; file upload works today |
| Medical-grade health alerts or drowning detection | ❌ Not validated or claimed |

The five classes are freestyle, backstroke, breaststroke, butterfly, and rest. The confidence values are model scores on synthetic-trained classifiers, **not clinically calibrated probabilities**.

## Data schema

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate_bpm,spo2_pct,distance_m,stroke_label,session_id
2026-01-05T07:00:00.000,0.02,0.11,1.04,-5.1,24.0,79.5,88,98,0.1,freestyle,session_001
```

Required: `timestamp`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`. The others are optional. File aliases including `ax`, `ay`, `az`, `gx`, `gy`, `gz`, `heart_rate`, `spo2`, `lap` and `stroke` are accepted. Expected accelerometer units **g** and gyro **degrees/second**. Convert hardware units before uploading if needed. Relative distance in **meters**. Session timestamps must be valid and each analyzed workout ≥4 seconds. For hosting safety, max file size in the app is 15 MB and a single session must not span >2 hours.

**Privacy:** Uploaded files are processed in memory and are not automatically retained in a shared persistent database. Sharing the app does not require a sign-in. Do not upload identifiable health records to a public prototype without a privacy review.

## Run locally

```bash
python -m venv .venv
# Activate the environment for your OS
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit. Default demo can run without a smartwatch.

## Optional training (not part of production deployment)

```bash
python -m pip install -r requirements-training.txt
python scripts/train_models.py
python scripts/export_models.py
```

This optional route uses PyTorch and scikit-learn locally to train on generated demo sessions (adapt the script to use labeled real data), then exports `.npz` weights for fast deployment. The pre-exported portable weights are already included in `models/`, so **do not run this to launch the web app**. The Jupyter notebook in `notebooks/` retains the Colab workflow.

## Repository structure

```text
app.py                              # Streamlit entry point (no heavy imports)
requirements.txt                    # six cloud production dependencies (including Streamlit’s PyArrow wheel)
requirements-training.txt           # optional PyTorch training dependencies
.python-version                     # local interpreter suggestion (3.13)
.streamlit/config.toml              # theme and upload size
.streamlit/secrets.toml.example     # optional persistent visitor counter
src/swim_ai.py                      # preprocessing, 94 features, health, RL advice
src/portable_models.py              # authentic forest, dense DNN, CNN/BiGRU in NumPy
src/visuals.py                      # Plotly visuals + animated HTML canvas
src/gif_export.py                   # GIF replay via Streamlit's Pillow
src/visitor_counter.py              # once-per-session visits, Supabase/SQLite
src/swim_ai_torch.py                # original OPTIONAL training core
models/*_portable.npz               # original weights converted without retraining
models/model_metadata.json          # classes, feature names, training provenance
models/rl_q_table.npy               # offline Q-learning policy table
models/*.pt,models/*.joblib          # original training checkpoints (not loaded by app)
notebooks/                           # Colab notebook
sample_data/kevin_demo_watch_session.csv
scripts/train_models.py             # optional offline retraining
scripts/export_models.py            # offline conversion to portable weights
tests/                               # core/model/counter/deployment checks
.github/workflows/tests.yml         # automated tests + startup health check
```

## Boundaries

This is a student research prototype. Watch optical HR and underwater SpO₂ can be unreliable; very low readings require sensor review, not diagnosis. Simulated RL coaching is not real-world safety supervision. A real validation cohort, labeled stroke ground truth from Kevin’s physical watch, and hardware calibration are necessary before deployment beyond demonstrations.

**Author: Kevin Sun · Mentor: Dr. Qingyang Xiao**
