"""
Local Qwen2.5 + PEFT LoRA inference (used when LLM_PROVIDER=local).

Expects a folder with adapter_config.json, tokenizer, and LoRA weights
(e.g. adapter_model.safetensors). Weights are gitignored by default.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model: Any = None
_tokenizer: Any = None
_load_error: Optional[str] = None

_DEFAULT_BASE_FP = "Qwen/Qwen2.5-3B-Instruct"
_IM_END = re.compile(r"<\|im_end\|>|<\|redacted_im_end\|>")

# Common PEFT output names
_LORA_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_lora_weights.safetensors",
    "adapters.safetensors",
)


def default_adapter_path() -> Path:
    p = (os.getenv("LLM_LOCAL_ADAPTER_PATH") or "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "model" / "kge_chatbot_lora").resolve()


def _has_adapter_weights(p: Path) -> bool:
    for name in _LORA_FILES:
        if (p / name).is_file():
            return True
    return any(p.glob("adapter_model*.safetensors")) or any(p.glob("adapter_model*.bin"))


def is_adapter_path_ready() -> bool:
    """True if the adapter directory looks loadable (config + weight files on disk)."""
    d = default_adapter_path()
    if not d.is_dir():
        return False
    if not (d / "adapter_config.json").is_file():
        return False
    return _has_adapter_weights(d)


def _read_base_from_adapter(adapter_path: Path) -> str:
    cfg = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    return (cfg.get("base_model_name_or_path") or _DEFAULT_BASE_FP).strip()


def _build_chat_messages(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role", "user")
        if role == "model":
            role = "assistant"
        if role not in ("user", "assistant"):
            continue
        out.append({"role": role, "content": m.get("content", "")})
    return out


def _load_model_once() -> None:
    global _model, _tokenizer, _load_error
    with _lock:
        if _model is not None:
            return
        if _load_error is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        adapter_path = default_adapter_path()
        if not is_adapter_path_ready():
            _load_error = f"Local adapter not ready: {adapter_path} (need adapter_config.json + LoRA weights)"
            logger.error(_load_error)
            return

        override_base = (os.getenv("LLM_LOCAL_BASE_MODEL") or "").strip()
        cfg_base = _read_base_from_adapter(adapter_path)
        use_4bit = (os.getenv("LLM_LOCAL_4BIT") or "").strip().lower() in (
            "1", "true", "yes", "y", "on",
        )
        if not (os.getenv("LLM_LOCAL_4BIT") or "").strip():
            use_4bit = bool(torch.cuda.is_available())

        has_bnb = True
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception:
                has_bnb = False
                use_4bit = False
                logger.warning("bitsandbytes not available; loading non-4bit base model (more RAM).")

        base_id = override_base or (cfg_base if (use_4bit and has_bnb) else _DEFAULT_BASE_FP)

        logger.info(
            "Loading local LLM: base=%s, adapter=%s, 4bit=%s, cuda=%s",
            base_id,
            adapter_path,
            use_4bit,
            torch.cuda.is_available(),
        )

        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_path),
                trust_remote_code=True,
            )
        except Exception as e:
            _load_error = f"Tokenizer load failed: {e}"
            logger.exception("Tokenizer load from adapter path failed")
            return

        if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
            _tokenizer.pad_token = _tokenizer.eos_token

        try:
            if use_4bit and has_bnb:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
                _model = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            _model = PeftModel.from_pretrained(_model, str(adapter_path), is_trainable=False)
            _model.eval()
        except Exception as e:
            _load_error = f"Model load failed: {e}"
            _model = None
            _tokenizer = None
            logger.exception("Local model load failed")


def generate_local_sync(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> str:
    _load_model_once()
    if _model is None or _tokenizer is None:
        raise RuntimeError(_load_error or "Local model not loaded")

    import torch

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_new = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

    chat = _build_chat_messages(system_prompt, messages)
    try:
        prompt = _tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        raise RuntimeError(f"apply_chat_template failed: {e}") from e

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    if torch.cuda.is_available():
        inputs = {k: v.to(_model.get_input_embeddings().weight.device) for k, v in inputs.items()}
    else:
        try:
            device = next(_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

    in_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=temperature > 0.001,
            temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    gen_ids = out[0][in_len:]
    text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = _IM_END.sub("", text).strip()
    if not text:
        raise RuntimeError("Local model returned an empty response.")
    return text
