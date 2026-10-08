# Fix for the Streamlit Cloud installation stall

**Student:** Kevin Sun  
**Mentor:** Dr. Qingyang Xiao

The screenshot shows the process paused after **"Processing dependencies... Using uv pip install ... Resolved 68 packages"** and before any `app.py` code ran. Importing PyTorch lazily in an earlier version could not help because the hosting platform installs `requirements.txt` before starting the app. The old requirements also pinned pandas 2.2.3 even though the Streamlit environment shown used Python 3.14.8.

This build fixes the installation design:

- `requirements.txt` contains NO `torch`, `scikit-learn`, `scipy`, `tensorflow`, `joblib`, or ffmpeg.
- pandas **2.3.3 or newer in its 2.x series** and PyArrow **22+** have Python 3.14 wheels. Streamlit (recent 1.x) is required.
- Original **260-tree Random Forest, deep feature neural network, and CNN–BiGRU** weights are included as `.npz` arrays. Forward passes use plain NumPy and reproduce original model outputs; the original checkpoints remain for offline training only.
- Only one `app.py` at the root, one production `requirements.txt`, and no `uv.lock`, `pyproject.toml` or `packages.txt`.

## To deploy

1. Extract this ZIP to a folder on your PC.
2. On GitHub, open your `qxiao2ub/ai-swimming-health-app` repository and **replace old files** with all the contents of this extracted folder (especially `app.py`, `requirements.txt`, `src/`, and `models/`). Keep `app.py` directly at the repository root. Commit the changes to `main`.
3. On GitHub, click **requirements.txt**. Confirm its contents are the six entries: `streamlit`, `numpy`, `pandas`, `plotly`, `pillow`, `pyarrow`. **No torch.** This confirms that Streamlit Cloud is reading the new commit.
4. In Streamlit Community Cloud, keep **Main file path: `app.py`**. Prefer **Python 3.13** in deployment settings; if your app is already created with Python 3.14, this build also selects binary-compatible minimum versions.
5. **Reboot the app**, or clear deployment cache and reboot if offered. In some cases deleting and creating a new Streamlit deployment is quicker if an old environment remains unhealthy. The site should load its landing page before any analysis.
6. Click **Analyze swimming session** to run the included synthetic demo and check charts, animations and downloads.

## If it still stays in the oven

- Open the **full deployment logs**, not only the first 15 lines. Check for `Installing collected packages`, `Building wheel for ...`, `Killed`, `ERROR`, or `No matching distribution`.
- Confirm Streamlit is using the **new requirements.txt from the active branch**. If the log shows Torch being installed, the old file is still deployed or another dependency definition remains at the root.
- If Python or platform dependency resolution is incompatible, create a fresh Streamlit deployment using Python 3.13 and the same `app.py` entry point.
- If the service cannot provision a machine or install **Streamlit itself**, the issue may be with the hosting platform or an external network; no app code can guarantee a fix.
- If the page loads but analytics fail, check the app exception logs and upload a sample of watch CSV without personally identifying data.

The number in the sidebar is **browser sessions counted**, not verified distinct people. Without Supabase, the local counter can reset during cloud redeploys; follow `docs/VISITOR_COUNTER.md` for persistence.
