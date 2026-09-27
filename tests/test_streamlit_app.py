from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_starts_and_renders_demo():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.run()

    assert not app.exception
    assert any("AI Swimming Health App" in item.value for item in app.markdown)
    assert any(metric.label == "App visitors" for metric in app.metric)
    assert any("Kevin Sun" in item.value for item in app.markdown)
    assert any("Dr. Qingyang Xiao" in item.value for item in app.markdown)
