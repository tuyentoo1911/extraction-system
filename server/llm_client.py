"""
LLM provider adapter with auto-fallback.

Supported providers (set LLM_PROVIDER env var):
  - local_hf    (requires LLM_MODEL as local path or Hugging Face model id)
  - local_lora  (requires LLM_MODEL as path to LoRA adapter directory;
                 reads adapter_config.json for base model, or override via LLM_BASE_MODEL)

When no provider is configured the module exposes `is_configured() == False`
so the chat service can fall back to rule-based answers.

Environment variables for local providers:
  LLM_MODEL           - Path to model dir / HF model id (local_hf) or LoRA adapter dir (local_lora)
  LLM_BASE_MODEL      - (local_lora only) Override base model instead of reading adapter_config.json
  LLM_DEVICE_MAP      - "auto" (default), "cpu", "cuda", etc.
  LLM_TORCH_DTYPE     - "auto" (default), "float16", "bfloat16", "float32"
  LLM_LOAD_IN_4BIT    - "true" to enable bitsandbytes 4-bit quantization (requires bitsandbytes)
  LLM_MAX_INPUT_TOKENS - Max input tokens before truncation (default 3072)
  LLM_MAX_NEW_TOKENS  - Max tokens to generate (default 512)
  LLM_TEMPERATURE     - Sampling temperature (default 0.2)
  LLM_TOP_P           - Top-p nucleus sampling (default 0.9)
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROVIDER: Optional[str] = None
_MODEL: Optional[str] = None
_LOCAL_MODEL: Any = None
_LOCAL_TOKENIZER: Any = None
_LOCAL_MODEL_ID: Optional[str] = None

# LoRA model cache
_LORA_MODEL: Any = None
_LORA_TOKENIZER: Any = None
_LORA_MODEL_PATH: Optional[str] = None

def _load_config() -> None:
    global _PROVIDER, _MODEL
    _PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip().lower() or None
    _MODEL = (os.getenv("LLM_MODEL") or "").strip() or None

_load_config()

def is_configured() -> bool:
    _load_config()
    return bool(_PROVIDER in ("local_hf", "local_lora") and _MODEL)

async def generate(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    """Call the configured LLM. Raises on failure so caller can fallback."""
    _load_config()
    if not is_configured():
        raise RuntimeError("LLM not configured")

    if _PROVIDER == "local_hf":
        return await _call_local_hf(system_prompt, messages)
    if _PROVIDER == "local_lora":
        return await _call_local_lora(system_prompt, messages)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_PROVIDER}")

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
    max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))
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


# ── local_lora provider ──────────────────────────────────────────────────────

async def _call_local_lora(system: str, messages: list[dict[str, str]]) -> str:
    tokenizer, model = _get_lora_model()
    prompt = _build_local_prompt(tokenizer, system, messages)
    return await asyncio.to_thread(_generate_local_text, tokenizer, model, prompt)


def _get_lora_model():
    """Load (and cache) base model + LoRA adapter for local_lora provider."""
    global _LORA_MODEL, _LORA_TOKENIZER, _LORA_MODEL_PATH

    adapter_path = _MODEL
    if not adapter_path:
        raise RuntimeError(
            "LLM_MODEL must be set to the LoRA adapter directory for LLM_PROVIDER=local_lora"
        )

    if (_LORA_MODEL is not None and _LORA_TOKENIZER is not None
            and _LORA_MODEL_PATH == adapter_path):
        return _LORA_TOKENIZER, _LORA_MODEL

    import json
    from pathlib import Path

    adapter_dir = Path(adapter_path)
    if not adapter_dir.is_absolute():
        # Resolve relative paths against project root (parent of server/)
        adapter_dir = Path(__file__).resolve().parent.parent / adapter_path

    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists():
        raise RuntimeError(
            f"adapter_config.json not found in '{adapter_dir}'. "
            "Make sure LLM_MODEL points to the LoRA adapter directory."
        )

    with open(config_file, encoding="utf-8") as f:
        adapter_cfg = json.load(f)

    # LLM_BASE_MODEL env var overrides the base model stored in adapter_config.json
    base_model_id = (os.getenv("LLM_BASE_MODEL") or "").strip()
    if not base_model_id:
        base_model_id = adapter_cfg.get("base_model_name_or_path", "").strip()
    if not base_model_id:
        raise RuntimeError(
            "Cannot determine base model. "
            "Set LLM_BASE_MODEL env var or ensure adapter_config.json has base_model_name_or_path."
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "local_lora provider requires: pip install transformers peft torch"
        ) from exc

    device_map = (os.getenv("LLM_DEVICE_MAP") or "auto").strip()
    torch_dtype_str = (os.getenv("LLM_TORCH_DTYPE") or "auto").strip().lower()
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype_str, "auto")

    load_kwargs: dict = dict(torch_dtype=dtype, device_map=device_map, trust_remote_code=True)

    use_4bit = os.getenv("LLM_LOAD_IN_4BIT", "").strip().lower() in ("1", "true", "yes")
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs.pop("torch_dtype", None)
            logger.info("Loading base model with 4-bit quantization: %s", base_model_id)
        except ImportError:
            logger.warning(
                "bitsandbytes not installed; loading base model without 4-bit quantization."
            )
            logger.info("Loading base model: %s", base_model_id)
    else:
        logger.info("Loading base model: %s", base_model_id)

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)

    logger.info("Applying LoRA adapter from: %s", adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Tokenizer is stored alongside the adapter (fine-tuned tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if (getattr(model, "generation_config", None) is not None
            and model.generation_config.pad_token_id is None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    _LORA_TOKENIZER = tokenizer
    _LORA_MODEL = model
    _LORA_MODEL_PATH = adapter_path
    logger.info("LoRA model ready.")
    return tokenizer, model
