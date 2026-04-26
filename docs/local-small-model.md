# Chatbot với model local nhỏ

Tài liệu này hướng dẫn chạy chatbot bằng model local, không dùng nhà cung cấp API bên ngoài.
Backend hỗ trợ hai provider local trong `server/llm_client.py`:

| Provider | Mô tả |
|---|---|
| `local_lora` | Load base model + áp dụng LoRA adapter (PEFT) — **khuyến nghị** cho model đã fine-tune |
| `local_hf` | Load thẳng model HuggingFace (không có adapter) |

---

## 1. Provider `local_lora` — Model đã fine-tune (kge_chatbot_lora)

Model `kge_chatbot_lora` là Qwen2.5-3B-Instruct đã được fine-tune với LoRA trên bộ dữ liệu KGE.

**Cấu hình `.env`:**

```env
LLM_PROVIDER=local_lora
LLM_MODEL=./model/kge_chatbot_lora

# Tuỳ chọn: override base model (mặc định đọc từ adapter_config.json)
# CPU: đổi sang Qwen/Qwen2.5-3B-Instruct để tránh cần bitsandbytes
LLM_BASE_MODEL=

# GPU: bật 4-bit quantization để tiết kiệm VRAM (~2 GB)
LLM_LOAD_IN_4BIT=false

LLM_DEVICE_MAP=auto
LLM_TORCH_DTYPE=auto
LLM_MAX_INPUT_TOKENS=3072
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.9
```

**Cài đặt phụ thuộc:**

```bash
pip install -r requirements.txt

# Nếu dùng LLM_LOAD_IN_4BIT=true (CUDA GPU):
pip install bitsandbytes
```

**Cơ chế hoạt động:**
1. Backend đọc `adapter_config.json` trong `./model/kge_chatbot_lora/` để xác định base model.
2. Download base model từ HuggingFace (nếu chưa có trong cache).
3. Áp dụng LoRA adapter weights từ `adapter_model.safetensors`.
4. Load tokenizer từ thư mục adapter.
5. Cache model trong RAM/VRAM — lần đầu chậm (~30–120s), các lần sau tức thì.

**Yêu cầu phần cứng:**

| Chế độ | VRAM / RAM | Ghi chú |
|---|---|---|
| GPU + 4-bit (`LLM_LOAD_IN_4BIT=true`) | ~2–3 GB VRAM | Cần bitsandbytes + CUDA |
| GPU full precision | ~6–7 GB VRAM | `LLM_TORCH_DTYPE=float16` |
| CPU | ~7–8 GB RAM | Chậm hơn, đặt `LLM_DEVICE_MAP=cpu` |

---

## 2. Provider `local_hf` — Model HuggingFace không có LoRA

Dùng khi muốn chạy thẳng một model từ HuggingFace (không có fine-tune LoRA).

```env
LLM_PROVIDER=local_hf
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
LLM_DEVICE_MAP=auto
LLM_TORCH_DTYPE=auto
LLM_MAX_INPUT_TOKENS=3072
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.9
```

Model nhỏ gợi ý:
- `Qwen/Qwen2.5-3B-Instruct` — cân bằng tốt chất lượng / tốc độ
- `Qwen/Qwen2.5-1.5B-Instruct` — nhẹ hơn, vẫn hỗ trợ tiếng Việt
- `Qwen/Qwen2.5-0.5B-Instruct` — rất nhẹ, chất lượng thấp hơn

---

## 3. Biến môi trường tham chiếu đầy đủ

| Biến | Mặc định | Mô tả |
|---|---|---|
| `LLM_PROVIDER` | _(trống)_ | `local_lora`, `local_hf` |
| `LLM_MODEL` | _(trống)_ | Đường dẫn adapter (local_lora) hoặc model id |
| `LLM_BASE_MODEL` | _(trống)_ | Override base model cho `local_lora` |
| `LLM_LOAD_IN_4BIT` | `false` | Bật 4-bit BNB quantization |
| `LLM_DEVICE_MAP` | `auto` | `auto`, `cpu`, `cuda:0`, ... |
| `LLM_TORCH_DTYPE` | `auto` | `auto`, `float16`, `bfloat16`, `float32` |
| `LLM_MAX_INPUT_TOKENS` | `3072` | Cắt prompt nếu vượt quá |
| `LLM_MAX_NEW_TOKENS` | `512` | Số token tối đa sinh ra |
| `LLM_TEMPERATURE` | `0.2` | Nhiệt độ sampling (0 = greedy) |
| `LLM_TOP_P` | `0.9` | Top-p nucleus sampling |

---

## 4. Xử lý sự cố

**Lỗi `ModuleNotFoundError: peft`:**
```bash
pip install peft>=0.10.0
```

**Lỗi `ModuleNotFoundError: bitsandbytes`:**
```bash
pip install bitsandbytes
# Hoặc tắt: LLM_LOAD_IN_4BIT=false
```

**Base model không load được (CUDA OOM):**
- Giảm `LLM_TORCH_DTYPE=float16`
- Hoặc bật `LLM_LOAD_IN_4BIT=true` (cần bitsandbytes)
- Hoặc đặt `LLM_DEVICE_MAP=cpu`

**Override base model sang bản không quantize (CPU-friendly):**
```env
LLM_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
LLM_DEVICE_MAP=cpu
LLM_TORCH_DTYPE=float32
```

**Chatbot vẫn hoạt động nếu model không load được:**
Backend tự động fallback sang rule-based engine nếu LLM raise exception.

---

## 5. Luồng hoạt động

```
.env: LLM_PROVIDER=local_lora, LLM_MODEL=./model/kge_chatbot_lora
         │
         ▼
llm_client._get_lora_model()
   ├─ Đọc adapter_config.json → base_model = "unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit"
   ├─ AutoModelForCausalLM.from_pretrained(base_model, ...)
   ├─ PeftModel.from_pretrained(base_model, adapter_dir)
   └─ AutoTokenizer.from_pretrained(adapter_dir)
         │
         ▼ (cache trong bộ nhớ)
chat_service.handle_chat()
   ├─ RAG retrieval (BM25)
   ├─ _build_graph_context()
   └─ llm_client.generate(system_prompt, history)
              │
              ▼
        _call_local_lora()
              │
              ▼
        _generate_local_text()  → trả về câu trả lời
```
