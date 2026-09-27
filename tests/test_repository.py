from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def test_required_streamlit_files_exist():
    required = [
        "app.py",
        "requirements.txt",
        "runtime.txt",
        "README.md",
        ".streamlit/config.toml",
        "models/model_metadata.json",
        "sample_data/kevin_demo_watch_session.csv",
        "notebooks/Kevin_Sun_AI_Swimming_Health_App_Colab.ipynb",
    ]
    for relative_path in required:
        assert (ROOT / relative_path).exists(), relative_path


def test_names_are_present_in_app_and_readme():
    combined = (ROOT / "app.py").read_text(encoding="utf-8") + (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Kevin Sun" in combined
    assert "Dr. Qingyang Xiao" in combined


def test_model_metadata_is_complete():
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["author"] == "Kevin Sun"
    assert metadata["mentor"] == "Dr. Qingyang Xiao"
    assert len(metadata["class_names"]) == 5
    assert len(metadata["feature_columns"]) > 50
    assert len(metadata["ensemble_weights"]) == 3
