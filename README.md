 # Knowledge Graph Extractor

Ứng dụng trích xuất thực thể (NER), xây dựng quan hệ và trực quan hóa đồ thị tri thức từ văn bản/PDF tiếng Việt.

Stack chính:
- **Frontend**: React + Vite + TypeScript
- **Backend**: FastAPI (Python)
- **Mô hình NER**: PhoBERT (VinAI)
- **Chat memory**: PostgreSQL
- **LLM (tuỳ chọn)**: OpenAI / Gemini (fallback rule-based khi không cấu hình)

---

## 1) Tính năng chính

- Trích xuất thực thể và quan hệ từ văn bản (`/extract`)
- Upload PDF, tự bóc tách text (`/upload-pdf`)
- Dự đoán liên kết mới giữa các node (`/predict-links`)
- Tính graph metrics (degree, pagerank, betweenness, ...) (`/metrics`)
- Tra cứu Knowledge Base cục bộ (`/kb/*`)
- Dashboard trực quan: Graph, Entities, Relations, Insight, Chatbot, Metrics, Highlight
- **Chatbot hỏi đáp** với memory hội thoại bền vững (PostgreSQL), hỗ trợ multi-turn, hybrid rule-based + LLM

---

## 2) Kiến trúc hệ thống

```text
React Frontend (Vite, :3000)
        ↕ HTTP
FastAPI Backend (Python, :8000)
        ↕
PhoBERT NER + Knowledge Base + Metrics + ChatService
        ↕
PostgreSQL (chat memory)   +   LLM Provider (tuỳ chọn)
```

---

## 3) Cấu trúc thư mục

```text
.
├─ src/                         # Frontend React
│  ├─ components/views/         # Các màn hình Graph/Entities/Relations/...
│  ├─ lib/ai.ts                 # API client gọi backend
│  ├─ types.ts                  # Kiểu dữ liệu dùng chung
│  └─ Dashboard.tsx
├─ server/                      # Backend FastAPI
│  ├─ main.py                   # Entry point API
│  ├─ ner.py                    # Chạy NER
│  ├─ graph.py                  # Build graph + predict links
│  ├─ metrics.py                # Tính metrics đồ thị
│  ├─ knowledge_base.py         # KB load/search
│  ├─ schemas.py                # Pydantic schemas
│  ├─ chat_service.py           # Hybrid chat orchestrator
│  ├─ chat_memory.py            # PostgreSQL chat memory CRUD
│  ├─ llm_client.py             # LLM provider adapter (OpenAI/Gemini)
│  └─ sql/chat_memory.sql       # Schema migration cho chat tables
├─ model/                       # Trọng số model + dữ liệu hỗ trợ
├─ feature_engineering_output/  # Artifacts huấn luyện/đánh giá
└─ README.md
```

---

## 4) Yêu cầu môi trường

- Node.js 18+ (khuyến nghị 20+)
- Python 3.11+
- Pip
- PostgreSQL 14+ (dùng cho chat memory)

> Lưu ý: lần chạy đầu backend có thể mất thời gian do tải tokenizer/model.

---

## 5) Cài đặt

### 5.1 Cài frontend dependencies

```bash
npm install
```

### 5.2 Cài backend dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5.3 Chuẩn bị PostgreSQL

Tạo database và chạy migration:

```bash
createdb kge
psql kge -f server/sql/chat_memory.sql
```

Đặt `DATABASE_URL` trong file `.env` (copy từ `.env.example`):

```env
DATABASE_URL="postgresql://user:password@localhost:5432/kge"
```

### 5.4 (Tuỳ chọn) Cấu hình LLM

Để chatbot dùng LLM thay vì rule-based, thêm vào `.env`:

```env
LLM_PROVIDER="gemini"        # hoặc "openai"
LLM_API_KEY="your-api-key"
LLM_MODEL=""                 # để trống = dùng default (gemini-2.0-flash / gpt-4o-mini)
```

---

## 6) Chạy dự án local

Mở 2 terminal:

### Terminal 1 - Backend

```bash
npm run server
```

hoặc:

```bash
python server/main.py
```

Backend mặc định chạy ở `http://localhost:8000`.

### Terminal 2 - Frontend

```bash
npm run dev
```

Frontend chạy ở `http://localhost:3000`.

---

## 7) API backend

| Endpoint | Method | Mô tả |
|---|---|---|
| `/health` | GET | Trạng thái model và knowledge base |
| `/extract` | POST | Trích xuất entities + relations từ text |
| `/upload-pdf` | POST | Upload file PDF, trả text đã bóc tách |
| `/predict-links` | POST | Dự đoán quan hệ mới |
| `/metrics` | POST | Tính chỉ số đồ thị |
| `/chat` | POST | Chatbot hỏi đáp (hybrid rule-based + LLM, có memory) |
| `/kb/stats` | GET | Thống kê knowledge base |
| `/kb/search` | GET | Tìm kiếm entity trong KB |
| `/kb/entity` | GET | Lấy triples theo entity |

### Ví dụ request `/extract`

```json
{
  "text": "Vingroup đầu tư vào VinFast tại Hải Phòng.",
  "use_deep_analysis": false
}
```

### Ví dụ request `/chat`

```json
{
  "session_id": null,
  "message": "Cho tôi biết về Vingroup",
  "entities": [{"id": "e1", "name": "Vingroup", "type": "Organization"}],
  "relations": [],
  "input_text": "Vingroup đầu tư vào VinFast."
}
```

Response trả về `session_id` để dùng cho các lượt hỏi tiếp theo.

---

## 8) Scripts NPM

| Script | Mô tả |
|---|---|
| `npm run dev` | Chạy frontend Vite ở port 3000 |
| `npm run server` | Chạy FastAPI backend (`python server/main.py`) |
| `npm run build` | Build frontend production |
| `npm run preview` | Preview bản build |
| `npm run lint` | Type check (`tsc --noEmit`) |

---

## 9) Cấu hình môi trường

File mẫu: `.env.example`

```env
DATABASE_URL="postgresql://user:password@localhost:5432/kge"
LLM_PROVIDER=""     # "openai" hoặc "gemini" (để trống = rule-based)
LLM_API_KEY=""
LLM_MODEL=""
```

- `DATABASE_URL` **bắt buộc** cho chat memory (PostgreSQL).
- `LLM_PROVIDER` + `LLM_API_KEY` tuỳ chọn — khi không cấu hình, chatbot tự động dùng chế độ rule-based.

---

## 10) Chatbot & tích hợp AI

Chatbot hoạt động theo kiến trúc **hybrid**:

1. **Rule-based** (mặc định): trả lời dựa trên entity/relation matching, không cần API key.
2. **LLM mode**: khi cấu hình `LLM_PROVIDER` + `LLM_API_KEY`, chatbot gọi LLM với graph context và memory hội thoại.
3. **Fallback**: nếu LLM timeout hoặc lỗi, tự động chuyển sang rule-based.

Memory hội thoại lưu bền vững trong PostgreSQL theo `session_id`. Frontend persist session qua `localStorage`, có nút "Reset" để bắt đầu hội thoại mới.

Điểm mở rộng chính:

- `server/llm_client.py`: thêm provider mới (Anthropic, Ollama, ...) bằng cách thêm hàm `_call_<provider>`.
- `server/chat_service.py`: tuỳ chỉnh system prompt, context builder, hoặc rule-based logic.

---

## 11) Lỗi thường gặp

- **Không kết nối được server**
  - Đảm bảo backend đang chạy ở `:8000` hoặc `:8001`
  - Kiểm tra endpoint `GET /health`
- **Model chưa sẵn sàng**
  - Chờ thêm vài phút ở lần chạy đầu
  - Xem log backend để kiểm tra lỗi tải model
- **Upload PDF lỗi**
  - Chỉ hỗ trợ `.pdf`
  - PDF scan ảnh thuần có thể không trích xuất text được
- **Chat không lưu lịch sử**
  - Kiểm tra `DATABASE_URL` đúng và PostgreSQL đang chạy
  - Chạy migration: `psql $DATABASE_URL -f server/sql/chat_memory.sql`
- **LLM không phản hồi**
  - Kiểm tra `LLM_PROVIDER`, `LLM_API_KEY` trong `.env`
  - Chat sẽ tự fallback sang rule-based nếu LLM lỗi

---

## 12) Gợi ý cải thiện tiếp

- Thêm Docker Compose cho frontend + backend + PostgreSQL
- Bổ sung test API (`pytest`) và test frontend
- Chuẩn hóa script đa nền tảng (Windows/Linux/macOS)
- Thêm provider LLM mới (Anthropic, Ollama local, ...)
