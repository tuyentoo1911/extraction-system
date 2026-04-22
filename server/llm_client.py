"""
LLM provider adapter with auto-fallback.

Supported providers (set LLM_PROVIDER env var):
  - openai    (requires LLM_API_KEY, optional LLM_MODEL default gpt-4o-mini)
  - gemini    (requires LLM_API_KEY, optional LLM_MODEL default gemini-2.0-flash)
  - local_hf  (requires LLM_MODEL as local path or Hugging Face model id)

When no provider is configured the module exposes `is_configured() == False`
so the chat service can fall back to rule-based answers.
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0

_PROVIDER: Optional[str] = None
_API_KEY: Optional[str] = None
_MODEL: Optional[str] = None
_LOCAL_MODEL: Any = None
_LOCAL_TOKENIZER: Any = None
_LOCAL_MODEL_ID: Optional[str] = None

def _load_config() -> None:
    global _PROVIDER, _API_KEY, _MODEL
    _PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None
    _API_KEY = (os.getenv("LLM_API_KEY") or "").strip() or None
    _MODEL = (os.getenv("LLM_MODEL") or "").strip() or None

_load_config()

def is_configured() -> bool:
    _load_config()
    if _PROVIDER == "local_hf":
        return bool(_MODEL)
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
    if _PROVIDER == "local_hf":
        return await _call_local_hf(system_prompt, messages)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_PROVIDER}")

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

async def _call_local_hf(system: str, messages: list[dict[str, str]]) -> str:
    tokenizer, model = _get_local_model()
    prompt = _build_local_prompt(tokenizer, system, messages)
    return await asyncio.to_thread(_generate_local_text, tokenizer, model, prompt)

def _get_local_model():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_MODEL_ID

    model_id = _MODEL
    if not model_id:
        raise RuntimeError("LLM_MODEL must be set for LLM_PROVIDER=local_hf")

    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None and _LOCAL_MODEL_ID == model_id:
        return _LOCAL_TOKENIZER, _LOCAL_MODEL

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Local HF provider requires transformers and torch to be installed."
        ) from exc

    device_map = (os.getenv("LLM_DEVICE_MAP") or "auto").strip()
    torch_dtype = (os.getenv("LLM_TORCH_DTYPE") or "auto").strip().lower()

    dtype = "auto"
    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float32":
        dtype = torch.float32

    logger.info("Loading local chat model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    _LOCAL_TOKENIZER = tokenizer
    _LOCAL_MODEL = model
    _LOCAL_MODEL_ID = model_id
    return tokenizer, model

def _build_local_prompt(tokenizer, system: str, messages: list[dict[str, str]]) -> str:
    chat_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for m in messages:
        role = "assistant" if m["role"] == "model" else m["role"]
        chat_messages.append({"role": role, "content": m["content"]})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            logger.warning("Tokenizer chat template failed; falling back to plain prompt.", exc_info=True)

    chunks: list[str] = [f"System:\n{system}"]
    for msg in chat_messages[1:]:
        label = "Assistant" if msg["role"] == "assistant" else "User"
        chunks.append(f"{label}:\n{msg['content']}")
    chunks.append("Assistant:\n")
    return "\n\n".join(chunks)

def _generate_local_text(tokenizer, model, prompt: str) -> str:
    import torch

    max_input_tokens = int(os.getenv("LLM_MAX_INPUT_TOKENS", "3072"))
    max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_p = float(os.getenv("LLM_TOP_P", "0.9"))
    do_sample = temperature > 0

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    model_device = getattr(model, "device", None)
    if model_device is not None and model_device.type != "meta":
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Local model returned an empty response.")
    return text
