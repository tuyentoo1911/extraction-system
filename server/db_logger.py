"""
db_logger.py — Ghi log vào bảng documents và request_performance_log.

Sử dụng connection pool từ workspace_memory để tránh tạo pool thứ 2.
Tất cả hàm đều best-effort: lỗi chỉ log warning, không raise exception.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pool():
    """Lấy pool từ workspace_memory (đã init khi startup)."""
    try:
        import workspace_memory as wm
        return wm._pool
    except Exception:
        return None


@contextmanager
def _conn():
    pool = _get_pool()
    if pool is None:
        raise RuntimeError("DB pool not available")
    with pool.connection() as conn:
        yield conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_document(
    workspace_id: Optional[str],
    filename: str,
    file_type: str,           # 'pdf' | 'text'
    page_count: Optional[int] = None,
    char_count: Optional[int] = None,
    truncated: bool = False,
) -> Optional[str]:
    """
    Ghi thông tin tài liệu vào bảng documents.
    Trả về document id nếu thành công, None nếu lỗi.
    """
    try:
        with _conn() as conn:
            row = conn.execute(
                """
                INSERT INTO documents
                    (workspace_id, filename, file_type, page_count, char_count, truncated)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (workspace_id, filename, file_type, page_count, char_count, truncated),
            ).fetchone()
            conn.commit()
        doc_id = row[0] if row else None
        logger.info("document logged: id=%s filename=%s", doc_id, filename)
        return doc_id
    except Exception as exc:
        logger.warning("log_document failed: %s", exc)
        return None


def log_performance(
    endpoint: str,
    duration_ms: int,
    status: str = "success",           # 'success' | 'timeout' | 'error'
    workspace_id: Optional[str] = None,
    model_name: Optional[str] = None,
    input_length: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Ghi thống kê hiệu năng vào bảng request_performance_log.
    """
    try:
        with _conn() as conn:
            conn.execute(
                """
                INSERT INTO request_performance_log
                    (workspace_id, endpoint, model_name, input_length, duration_ms, status, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (workspace_id, endpoint, model_name, input_length, duration_ms, status, error_message),
            )
            conn.commit()
        logger.debug("perf logged: %s %dms [%s]", endpoint, duration_ms, status)
    except Exception as exc:
        logger.warning("log_performance failed: %s", exc)


# ---------------------------------------------------------------------------
# Context manager helper để đo thời gian và tự log
# ---------------------------------------------------------------------------

class PerfTimer:
    """
    Usage:
        with PerfTimer("/extract", workspace_id=sid, model_name="phobert-ner", input_length=len(text)):
            result = do_work()
    """

    def __init__(
        self,
        endpoint: str,
        workspace_id: Optional[str] = None,
        model_name: Optional[str] = None,
        input_length: Optional[int] = None,
    ):
        self.endpoint = endpoint
        self.workspace_id = workspace_id
        self.model_name = model_name
        self.input_length = input_length
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((time.perf_counter() - self._start) * 1000)

        if exc_type is None:
            status = "success"
            error_msg = None
        elif "timeout" in str(exc_val).lower() or "timed out" in str(exc_val).lower():
            status = "timeout"
            error_msg = str(exc_val)[:500]
        else:
            status = "error"
            error_msg = str(exc_val)[:500]

        log_performance(
            endpoint=self.endpoint,
            duration_ms=duration_ms,
            status=status,
            workspace_id=self.workspace_id,
            model_name=self.model_name,
            input_length=self.input_length,
            error_message=error_msg,
        )
        # Không suppress exception
        return False
