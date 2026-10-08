from pathlib import Path
import ast
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_cloud_dependencies_no_heavy_training_libs():
    req = ROOT.joinpath('requirements.txt').read_text().lower()
    lines = '\n'.join(l for l in req.splitlines() if l.strip() and not l.lstrip().startswith('#'))
    for term in ['torch', 'tensorflow', 'scikit-learn', 'scipy', 'joblib', 'opencv']:
        assert term not in lines
    for term in ['streamlit', 'numpy', 'pandas', 'plotly']:
        assert term in lines
    assert not (ROOT / 'uv.lock').exists()
    assert not (ROOT / 'pyproject.toml').exists()


def test_website_and_core_have_no_heavy_ml_imports():
    for name in ['app.py', 'src/swim_ai.py', 'src/portable_models.py']:
        source = (ROOT / name).read_text()
        ast.parse(source)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] not in {'torch','sklearn','scipy','joblib'} for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split('.')[0] not in {'torch','sklearn','scipy','joblib'}
    assert 'Kevin Sun' in (ROOT / 'app.py').read_text()
    assert 'Dr. Qingyang Xiao' in (ROOT / 'app.py').read_text()


def test_packaging_has_all_required_artifacts():
    for p in ['app.py', 'requirements.txt', '.streamlit/config.toml',
              'models/cnn_bigru_portable.npz', 'models/feature_dnn_portable.npz',
              'models/random_forest_portable.npz', 'models/feature_scaler_portable.npz',
              'models/model_metadata.json', 'models/rl_q_table.npy',
              'notebooks/Kevin_Sun_AI_Swimming_Health_App_Colab.ipynb']:
        assert (ROOT / p).is_file(), p
