from pathlib import Path
import sqlite3
from src.visitor_counter import register_visit


def test_visitor_counter_one_per_session_and_persistent_file(tmp_path: Path):
    path = tmp_path / 'visitors.sqlite3'
    first = {}
    one, first_mode = register_visit(first, {}, database_path=path)
    again, same_mode = register_visit(first, {}, database_path=path)
    second = {}
    two, other_mode = register_visit(second, {}, database_path=path)
    assert one == 1
    assert again == 1
    assert two == 2
    assert first_mode == same_mode == other_mode == 'local SQLite'
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT visit_count FROM app_counters').fetchone()[0] == 2
