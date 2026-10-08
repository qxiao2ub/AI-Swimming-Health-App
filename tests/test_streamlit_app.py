"""Browserless Streamlit UI check; executes in GitHub CI with Streamlit installed."""
import pytest


def test_streamlit_landing_and_run_full_ensemble():
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=120)
    app.run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert len(app.button) > 0
    app.button[0].click().run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert 'analysis_result' in app.session_state
    assert len(app.session_state['analysis_result'].windows) == 47
