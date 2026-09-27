from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_TEXT = (ROOT / "app.py").read_text(encoding="utf-8")


def test_no_heavy_analytics_imports_at_module_top_level():
    tree = ast.parse(APP_TEXT)
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    assert not any(name.startswith("src.swim_ai") for name in top_level_imports)
    assert not any(name == "torch" or name.startswith("torch.") for name in top_level_imports)


def test_full_ai_imports_are_present_inside_deferred_functions_or_post_analysis():
    assert "def load_ai_stack" in APP_TEXT
    assert "from src.swim_ai import load_model_bundle" in APP_TEXT
    assert "Run Full AI Analysis" in APP_TEXT
