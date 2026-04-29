"""
LLM provider adapter (local only).

Supported provider:
  - local   — Qwen2.5 + LoRA trong thư mục model (xem LLM_LOCAL_*)

Khi chưa cấu hình, `is_configured()` trả về False để dùng rule-based.
"""

from __future__ import annotations

import asyncio
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER: Optional[str] = None

def _load_config() -> None:
    global _PROVIDER
    _PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None

_load_config()

def is_configured() -> bool:
    _load_config()
    if _PROVIDER != "local":
        return False
    try:
        from local_llm import is_adapter_path_ready

        return is_adapter_path_ready()
    except Exception:
        return False

async def generate(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    """Call the configured LLM. Raises on failure so caller can fallback."""
    _load_config()
    if not is_configured():
        raise RuntimeError("LLM not configured")

    if _PROVIDER == "local":
        from local_llm import generate_local_sync

        return await asyncio.to_thread(generate_local_sync, system_prompt, messages)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_PROVIDER}")
