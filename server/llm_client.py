"""
LLM provider adapter with auto-fallback.

Supported providers (set LLM_PROVIDER env var):
  - openai   (requires LLM_API_KEY, optional LLM_MODEL default gpt-4o-mini)
  - gemini   (requires LLM_API_KEY, optional LLM_MODEL default gemini-2.0-flash)

When no provider is configured the module exposes `is_configured() == False`
so the chat service can fall back to rule-based answers.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0

_PROVIDER: Optional[str] = None
_API_KEY: Optional[str] = None
_MODEL: Optional[str] = None


def _load_config() -> None:
    global _PROVIDER, _API_KEY, _MODEL
    _PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None
    _API_KEY = (os.getenv("LLM_API_KEY") or "").strip() or None
    _MODEL = (os.getenv("LLM_MODEL") or "").strip() or None


_load_config()


def is_configured() -> bool:
    return bool(_PROVIDER and _API_KEY)


async def generate(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    """Call the configured LLM. Raises on failure so caller can fallback."""
    _load_config()
    if not is_configured():
        raise RuntimeError("LLM not configured")

    if _PROVIDER == "openai":
        return await _call_openai(system_prompt, messages)
    if _PROVIDER == "gemini":
        return await _call_gemini(system_prompt, messages)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_PROVIDER}")


# ── OpenAI-compatible ────────────────────────────────────────────

async def _call_openai(system: str, messages: list[dict[str, str]]) -> str:
    model = _MODEL or "gpt-4o-mini"
    oai_messages = [{"role": "system", "content": system}]
    for m in messages:
        role = "assistant" if m["role"] == "model" else m["role"]
        oai_messages.append({"role": role, "content": m["content"]})

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={"model": model, "messages": oai_messages, "temperature": 0.4},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Google Gemini ────────────────────────────────────────────────

async def _call_gemini(system: str, messages: list[dict[str, str]]) -> str:
    model = _MODEL or "gemini-2.0-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={_API_KEY}"
    )

    contents = []
    for m in messages:
        role = "model" if m["role"] == "model" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4},
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
