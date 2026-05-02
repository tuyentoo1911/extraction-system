"""
LLM provider adapter.

Supported providers:
  - local   — Qwen2.5 + LoRA trong thư mục model (xem LLM_LOCAL_*)
  - ollama  — Ollama local server  (xem OLLAMA_BASE_URL / OLLAMA_MODEL)

Env vars for Ollama:
  LLM_PROVIDER=ollama
  OLLAMA_BASE_URL=http://localhost:11434   (default)
  OLLAMA_MODEL=qwen2.5:3b                 (default)
  OLLAMA_TIMEOUT=60                       (seconds, default)
  LLM_MAX_NEW_TOKENS=512                  (default)
  LLM_TEMPERATURE=0.2                     (default)

Khi chưa cấu hình hợp lệ, `is_configured()` trả về False → rule-based.
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
    if _PROVIDER == "local":
        try:
            from local_llm import is_adapter_path_ready
            return is_adapter_path_ready()
        except Exception:
            return False
    if _PROVIDER == "ollama":
        # We optimistically report True; actual availability is checked at generate time
        return True
    return False


async def generate(
    system_prompt: str,
    messages: list[dict[str, str]],
    timeout: float | None = None,
) -> str:
    """
    Call the configured LLM provider.
    Raises RuntimeError on any failure so the caller can fall back to rule-based.
    """
    _load_config()
    if not is_configured():
        raise RuntimeError("LLM not configured")

    if _PROVIDER == "local":
        from local_llm import generate_local_sync
        return await asyncio.to_thread(generate_local_sync, system_prompt, messages)

    if _PROVIDER == "ollama":
        return await _generate_ollama(system_prompt, messages, timeout=timeout)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {_PROVIDER!r}")


# ── Ollama backend ─────────────────────────────────────────────────────────────

async def _generate_ollama(
    system_prompt: str,
    messages: list[dict[str, str]],
    timeout: float | None = None,
) -> str:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx is required for the Ollama provider. "
            "Install it with: pip install httpx"
        ) from exc

    base_url     = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    model        = (os.getenv("OLLAMA_MODEL") or "qwen2.5:3b").strip()
    max_tokens   = int(os.getenv("LLM_MAX_NEW_TOKENS") or "512")
    temperature  = float(os.getenv("LLM_TEMPERATURE") or "0.2")
    req_timeout  = timeout or float(os.getenv("OLLAMA_TIMEOUT") or "60")

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    logger.info("Ollama → %s [%s] max_tokens=%d", base_url, model, max_tokens)

    async with httpx.AsyncClient(timeout=req_timeout) as client:
        try:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            ) from exc
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:400]
            raise RuntimeError(
                f"Ollama HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama request timed out after {req_timeout}s"
            ) from exc

    data    = response.json()
    content = data.get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("Ollama returned an empty response")

    logger.info("Ollama ← %d chars", len(content))
    return content
