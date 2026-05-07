"""
PostgreSQL persistence for extraction workspace history.
Stores per-workspace snapshots so UI can restore previous work sessions.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None


def init_workspace_db() -> None:
    """Best-effort init; keep server alive if DB unavailable."""
    try:
        init_pool()
    except Exception:
        # Persistence is optional at runtime.
        pass


def init_pool() -> None:
    """Initialize pool and ensure schema exists."""
    global _pool
    if _pool is not None:
        return

    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL is required for workspace history storage.")

    _pool = ConnectionPool(conninfo=db_url, min_size=1, max_size=5, kwargs={"row_factory": dict_row})
    with _pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_sessions (
                  id TEXT PRIMARY KEY,
                  title TEXT NOT NULL,
                  input_text TEXT NOT NULL DEFAULT '',
                  graph_data JSONB,
                  metrics_data JSONB,
                  insight_markdown TEXT,
                  chat_session_id TEXT,
                  chat_engine TEXT,
                  chat_history JSONB,
                  active_tab TEXT NOT NULL DEFAULT 'graph',
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            # Lightweight migration for existing databases.
            cur.execute("ALTER TABLE workspace_sessions ADD COLUMN IF NOT EXISTS insight_markdown TEXT;")
            cur.execute("ALTER TABLE workspace_sessions ADD COLUMN IF NOT EXISTS chat_session_id TEXT;")
            cur.execute("ALTER TABLE workspace_sessions ADD COLUMN IF NOT EXISTS chat_engine TEXT;")
            cur.execute("ALTER TABLE workspace_sessions ADD COLUMN IF NOT EXISTS chat_history JSONB;")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_workspace_sessions_updated_at
                ON workspace_sessions (updated_at DESC);
                """
            )
        conn.commit()


def _ensure_pool() -> ConnectionPool:
    if _pool is None:
        init_pool()
    if _pool is None:
        raise RuntimeError("Workspace DB pool is not initialized.")
    return _pool


def _make_title(input_text: str, fallback: str = "Phiên làm việc mới") -> str:
    first_line = (input_text or "").strip().splitlines()
    if not first_line:
        return fallback
    title = first_line[0].strip()
    return title[:80] if title else fallback


def save_workspace(
    session_id: str | None,
    input_text: str,
    graph_data: dict[str, Any] | None,
    metrics_data: dict[str, Any] | None,
    insight_markdown: str | None = None,
    chat_session_id: str | None = None,
    chat_engine: str | None = None,
    chat_history: list[dict[str, Any]] | None = None,
    active_tab: str = "graph",
    title: str | None = None,
) -> str:
    pool = _ensure_pool()
    sid = session_id or uuid.uuid4().hex
    resolved_title = (title or "").strip() or _make_title(input_text)

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_sessions (
                  id, title, input_text, graph_data, metrics_data, insight_markdown,
                  chat_session_id, chat_engine, chat_history, active_tab
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (id) DO UPDATE SET
                  title = EXCLUDED.title,
                  input_text = EXCLUDED.input_text,
                  graph_data = EXCLUDED.graph_data,
                  metrics_data = EXCLUDED.metrics_data,
                  insight_markdown = EXCLUDED.insight_markdown,
                  chat_session_id = EXCLUDED.chat_session_id,
                  chat_engine = EXCLUDED.chat_engine,
                  chat_history = EXCLUDED.chat_history,
                  active_tab = EXCLUDED.active_tab,
                  updated_at = NOW()
                """,
                (
                    sid,
                    resolved_title,
                    input_text or "",
                    json.dumps(graph_data) if graph_data is not None else None,
                    json.dumps(metrics_data) if metrics_data is not None else None,
                    insight_markdown,
                    chat_session_id,
                    chat_engine,
                    json.dumps(chat_history) if chat_history is not None else None,
                    active_tab or "graph",
                ),
            )
        conn.commit()
    return sid


def list_workspaces(limit: int = 50) -> list[dict[str, Any]]:
    pool = _ensure_pool()
    safe_limit = max(1, min(limit, 200))
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  id,
                  title,
                  LEFT(input_text, 160) AS preview_text,
                  COALESCE(jsonb_array_length(graph_data->'entities'), 0) AS entities_count,
                  COALESCE(jsonb_array_length(graph_data->'relations'), 0) AS relations_count,
                  created_at,
                  updated_at
                FROM workspace_sessions
                ORDER BY updated_at DESC, created_at DESC, id DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            rows = cur.fetchall()
    return rows


def get_workspace(session_id: str) -> dict[str, Any] | None:
    pool = _ensure_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, input_text, graph_data, metrics_data, active_tab, created_at, updated_at
                     , insight_markdown, chat_session_id, chat_engine, chat_history
                FROM workspace_sessions
                WHERE id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
    return row


def delete_workspace(session_id: str) -> bool:
    pool = _ensure_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM workspace_sessions WHERE id = %s", (session_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
