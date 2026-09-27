"""Visitor counting for the Streamlit app.

A visit is counted once per Streamlit browser session. When Supabase secrets are
configured, the count persists across app restarts and redeployments. Without
secrets, the app automatically falls back to a local SQLite counter.

Author: Kevin Sun
Mentor: Dr. Qingyang Xiao
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping
import json
import sqlite3
import threading
import urllib.error
import urllib.request

_LOCAL_LOCK = threading.Lock()
_SESSION_COUNT_KEY = "_kevin_swim_visitor_count"
_SESSION_BACKEND_KEY = "_kevin_swim_visitor_backend"


def _secret_value(secrets: Mapping[str, Any] | None, key: str) -> str | None:
    if secrets is None:
        return None
    try:
        value = secrets.get(key)
    except Exception:
        try:
            value = secrets[key]
        except Exception:
            return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _increment_supabase(
    secrets: Mapping[str, Any],
    app_slug: str,
    timeout_seconds: float = 5.0,
) -> int:
    base_url = _secret_value(secrets, "SUPABASE_URL")
    api_key = _secret_value(secrets, "SUPABASE_ANON_KEY")
    configured_slug = _secret_value(secrets, "COUNTER_SLUG") or app_slug
    if not base_url or not api_key:
        raise RuntimeError("Supabase visitor-counter secrets are not configured.")

    endpoint = f"{base_url.rstrip('/')}/rest/v1/rpc/increment_app_counter"
    payload = json.dumps({"counter_slug": configured_slug}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        parsed = json.loads(response.read().decode("utf-8"))

    if isinstance(parsed, (int, float)):
        return int(parsed)
    if isinstance(parsed, list) and parsed:
        item = parsed[0]
        if isinstance(item, (int, float)):
            return int(item)
        if isinstance(item, dict):
            for value in item.values():
                if isinstance(value, (int, float)):
                    return int(value)
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, (int, float)):
                return int(value)
    raise RuntimeError(f"Unexpected Supabase counter response: {parsed!r}")


def _increment_sqlite(database_path: Path, app_slug: str) -> int:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCAL_LOCK:
        connection = sqlite3.connect(database_path, timeout=10)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS app_counters "
                "(slug TEXT PRIMARY KEY, visit_count INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO app_counters(slug, visit_count) VALUES(?, 1) "
                "ON CONFLICT(slug) DO UPDATE SET visit_count = visit_count + 1",
                (app_slug,),
            )
            row = connection.execute(
                "SELECT visit_count FROM app_counters WHERE slug = ?",
                (app_slug,),
            ).fetchone()
            connection.commit()
            if row is None:
                raise RuntimeError("Local visitor counter did not return a value.")
            return int(row[0])
        finally:
            connection.close()


def register_visit(
    session_state: MutableMapping[str, Any],
    secrets: Mapping[str, Any] | None = None,
    app_slug: str = "kevin-sun-ai-swimming-health",
    database_path: str | Path = ".app_data/visitor_counter.sqlite3",
) -> tuple[int, str]:
    """Register one visit per browser session and return ``(count, backend)``.

    ``backend`` is ``"Supabase"`` when persistent cloud storage is configured,
    otherwise ``"local SQLite"``. The fallback count survives Streamlit reruns
    but can reset after a cloud restart or redeployment.
    """

    if _SESSION_COUNT_KEY in session_state:
        return int(session_state[_SESSION_COUNT_KEY]), str(
            session_state.get(_SESSION_BACKEND_KEY, "local SQLite")
        )

    try:
        count = _increment_supabase(secrets or {}, app_slug)
        backend = "Supabase"
    except (RuntimeError, OSError, urllib.error.URLError, ValueError, KeyError):
        try:
            count = _increment_sqlite(Path(database_path), app_slug)
            backend = "local SQLite"
        except (OSError, sqlite3.Error, RuntimeError):
            # Some managed platforms expose a read-only repository directory.
            count = _increment_sqlite(
                Path("/tmp/kevin_sun_swimming_visitor_counter.sqlite3"),
                app_slug,
            )
            backend = "temporary SQLite"

    session_state[_SESSION_COUNT_KEY] = int(count)
    session_state[_SESSION_BACKEND_KEY] = backend
    return int(count), backend
