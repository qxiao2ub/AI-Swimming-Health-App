# Kevin Sun AI Swimming Health App — Streamlit Fast-Start Repository

**Author / Student:** Kevin Sun  
**Mentor:** Dr. Qingyang Xiao

This repository is a Streamlit deployment of the supplied Colab prototype. It retains the smartwatch data pipeline, classical machine learning, deep neural network, CNN-BiGRU sequence model, reinforcement-learning coaching policy, health/training analytics, visualizations, animation, and downloads.

## Why this version fixes the long startup

The previous deployment could appear stuck on Streamlit Community Cloud's startup screen because the application imported the full PyTorch analytics stack and loaded model artifacts as part of the first page execution.

This revision changes the startup sequence:

1. `app.py` loads only lightweight packages plus the visitor counter.
2. The demo CSV is read directly; no synthetic generator or PyTorch model is imported at boot.
3. The website landing page renders before any AI weights are loaded.
4. PyTorch/scikit-learn analytics modules are imported only when the user clicks **Run Full AI Analysis**.
5. The heavy model bundle is cached after the first analysis so reruns are much faster.
6. The visitor counter uses local SQLite immediately and only attempts Supabase when credentials exist, with a 1.5-second fail-fast timeout.

This means the app can become visibly interactive before the expensive AI pipeline starts.

## Streamlit deployment

Use `app.py` as the entry point. The repository includes `requirements.txt`, `.streamlit/config.toml`, pretrained model artifacts, the sample smartwatch CSV, and the source notebook.

For a cloud-persistent visitor count, configure the included Supabase SQL and secrets. Without Supabase, the counter remains useful for the current runtime but can reset after a managed service restart/redeployment.

## Visitor counting

The counter records **one visit per Streamlit browser session** rather than attempting to identify a unique person. No IP address or user name is collected by the counter.

## Important model note

The included models are demonstration models trained on structured synthetic swimming-watch signals. Their successful execution validates the engineering pipeline, but it does not establish real-world clinical accuracy, sports-performance accuracy, or medical-device performance.

## Repository credits

**Student / Author:** Kevin Sun  
**Mentor:** Dr. Qingyang Xiao
