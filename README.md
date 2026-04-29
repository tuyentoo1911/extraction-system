# Knowledge Graph Extractor

Ứng dụng **trích xuất thực thể (NER)**, **xây dựng đồ thị quan hệ** và **trực quan hóa đồ thị tri thức** từ văn bản/PDF tiếng Việt. Tích hợp chatbot hỏi đáp thông minh với bộ nhớ hội thoại bền vững.

---

## Mục lục

1. [Tính năng chính](#1-tính-năng-chính)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Hiệu năng mô hình NER](#4-hiệu-năng-mô-hình-ner)
5. [Yêu cầu môi trường](#5-yêu-cầu-môi-trường)
6. [Cài đặt](#6-cài-đặt)
7. [Cấu hình môi trường](#7-cấu-hình-môi-trường)
8. [Chạy dự án local](#8-chạy-dự-án-local)
9. [API Backend](#9-api-backend)
10. [Giao diện Dashboard](#10-giao-diện-dashboard)
11. [Chatbot hỏi đáp](#11-chatbot-hỏi-đáp)
12. [Scripts NPM](#12-scripts-npm)
13. [Lỗi thường gặp](#13-lỗi-thường-gặp)
14. [Gợi ý mở rộng](#14-gợi-ý-mở-rộng)

---

## 1. Tính năng chính

| Tính năng | Mô tả |
|---|---|
| **NER tiếng Việt** | Trích xuất 9 loại thực thể từ văn bản bằng PhoBERT fine-tuned |
| **Xây dựng đồ thị tri thức** | Tự động nhận diện và kết nối quan hệ giữa các thực thể |
| **Upload PDF** | Bóc tách text từ PDF (tối đa 20 MB), hỗ trợ đa trang |
| **Dự đoán liên kết** | Gợi ý quan hệ mới có khả năng tồn tại giữa các node |
| **Graph Metrics** | Tính degree, PageRank, betweenness centrality, clustering coefficient |
| **Insight Report** | Tổng hợp phân tích cấu trúc đồ thị dạng Markdown |
| **Knowledge Base** | Tra cứu triple từ KB cục bộ (GEXF/JSON) |
| **Chatbot hybrid** | Hỏi đáp bằng tiếng Việt/Anh với 17+ loại intent, hỗ trợ LLM và rule-based |
| **Bộ nhớ hội thoại** | Lưu trữ lịch sử chat bền vững qua PostgreSQL theo session |
| **RAG pipeline** | Truy xuất ngữ cảnh BM25 từ văn bản, đồ thị, và KB để trả lời chính xác hơn |
| **Rate limiting** | Bảo vệ API với SlowAPI, giới hạn theo IP |

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────┐
│           React Frontend (Vite, :3000)          │
│  Graph │ Highlight │ Metrics │ Insight │ Chatbot │
└────────────────────┬────────────────────────────┘
                     │ HTTP REST
┌────────────────────▼────────────────────────────┐
│         FastAPI Backend (Python, :8000)          │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  NER     │  │  Graph   │  │  ChatService  │  │
│  │ PhoBERT  │  │ NetworkX │  │  RAG + LLM    │  │
│  └──────────┘  └──────────┘  └───────┬───────┘  │
│  ┌──────────┐  ┌──────────┐          │           │
│  │ Metrics  │  │    KB    │          │           │
│  │ PageRank │  │  Loader  │          │           │
│  └──────────┘  └──────────┘          │           │
└────────────────────┬─────────────────┼───────────┘
                     │                 │
        ┌────────────┘      ┌──────────▼──────────┐
        │ Model files       │  PostgreSQL          │
        │ phobert-ner-final │  (chat_memory)       │
        │ influence_pred.   └─────────────────────┘
        └──────────────────────────────────────────
                                    +
                       ┌────────────────────────┐
                       │ LLM Provider (tuỳ chọn)│
                       │   OpenAI / Gemini API   │
                       │    (tuỳ chọn)           │
                       └────────────────────────┘
```

**Luồng xử lý chính:**

1. Người dùng nhập văn bản hoặc upload PDF trên frontend.
2. Backend chạy **NER** bằng PhoBERT → trích xuất thực thể.
3. **Graph builder** tạo các cạnh quan hệ từ thực thể → trả về đồ thị.
4. Frontend hiển thị đồ thị tương tác, cho phép phân tích metrics, insight.
5. **Chatbot** nhận câu hỏi → RAG retrieval → LLM hoặc rule-based → trả lời.
6. Mọi lượt hội thoại được lưu vào **PostgreSQL** theo `session_id`.

---

## 3. Cấu trúc thư mục

```
knowledge-graph-extractor/
│
├─ src/                              # Frontend React + TypeScript
│  ├─ components/
│  │  └─ views/
│  │     ├─ GraphView.tsx            # Trực quan hoá đồ thị (react-force-graph-2d)
│  │     ├─ HighlightView.tsx        # Highlight thực thể trong văn bản gốc
│  │     ├─ ChatbotView.tsx          # Giao diện chatbot với suggestion chips
│  │     └─ ...                      # EntitiesView, RelationsView, MetricsView, InsightView
│  ├─ lib/
│  │  └─ ai.ts                       # API client gọi backend (fetch wrapper)
│  ├─ types.ts                       # Kiểu dữ liệu dùng chung (Entity, Relation, ...)
│  ├─ App.tsx                        # Router / layout chính
│  └─ Dashboard.tsx                  # State management trung tâm
│
├─ server/                           # Backend FastAPI (Python)
│  ├─ main.py                        # Entry point, khai báo routes, CORS, rate limiter
│  ├─ ner.py                         # Chạy PhoBERT NER, post-process token → entity
│  ├─ graph.py                       # Build đồ thị tri thức, predict new links
│  ├─ metrics.py                     # Tính graph metrics (NetworkX)
│  ├─ insights.py                    # Tạo báo cáo insight từ cấu trúc đồ thị
│  ├─ knowledge_base.py              # Load KB từ GEXF/JSON, search, get triples
│  ├─ rag.py                         # RAG pipeline: BM25 index + retrieval
│  ├─ chat_service.py                # Điều phối chat: session, RAG, LLM, rule-based
│  ├─ chat_memory.py                 # PostgreSQL CRUD cho lịch sử hội thoại
│  ├─ llm_client.py                  # LLM adapter cloud (OpenAI / Gemini)
│  ├─ schemas.py                     # Pydantic schemas (request/response models)
│  ├─ model.py                       # Load PhoBERT model state
│  ├─ constants.py                   # Hằng số dùng chung
│  └─ sql/
│     └─ chat_memory.sql             # DDL: tạo bảng sessions, messages
│
├─ model/                            # Model weights + KB output
│  ├─ phobert-ner-final/             # PhoBERT fine-tuned cho NER tiếng Việt
│  │  ├─ model.safetensors           # Trọng số model
│  │  ├─ config.json                 # Cấu hình model
│  │  ├─ id2label.json               # Mapping ID → nhãn entity
│  │  ├─ label2id.json               # Mapping nhãn → ID
│  │  ├─ test_metrics.json           # Kết quả đánh giá trên tập test
│  │  └─ vocab.txt / bpe.codes       # Tokenizer vocabulary
│  ├─ influence_predictor/
│  │  └─ influence_predictor.joblib  # Model dự đoán liên kết (scikit-learn)
│  └─ knowledge_graph_output/
│     ├─ knowledge_graph.gexf        # Đồ thị tri thức xuất từ quá trình huấn luyện
│     ├─ knowledge_graph.graphml.xml # Định dạng GraphML
│     ├─ triples.csv / triples.json  # Danh sách triple
│     └─ *.png                       # Ảnh thống kê đồ thị
│
├─ feature_engineering_output/       # Artifacts feature engineering (train/val/test split)
│  ├─ node_features.csv              # Feature matrix của các node
│  ├─ feature_importance.csv         # Tầm quan trọng của từng feature
│  ├─ scaler_params.json             # Tham số chuẩn hoá
│  └─ X_train/val/test.csv           # Tập dữ liệu đã split
│
├─ docs/
│  └─ chatbot.md                     # Tài liệu chi tiết về kiến trúc chatbot
│
├─ train_influence_predictor.ipynb   # Notebook huấn luyện model dự đoán liên kết
├─ run_influence_predictor.ipynb     # Notebook chạy inference dự đoán liên kết
├─ requirements.txt                  # Python dependencies
├─ package.json                      # Node.js dependencies và scripts
├─ vite.config.ts                    # Cấu hình Vite
├─ tsconfig.json                     # Cấu hình TypeScript
├─ .env.example                      # Mẫu biến môi trường
└─ README.md
```

---

## 4. Hiệu năng mô hình NER

Mô hình **PhoBERT fine-tuned** được đánh giá trên tập test với **19 nhãn** (BIO tagging), đạt kết quả:

| Chỉ số | Giá trị |
|--------|---------|
| Precision | **94.17%** |
| Recall | **97.44%** |
| F1 Score | **95.78%** |
| Accuracy | **99.51%** |

**F1 theo từng loại thực thể:**

| Loại thực thể | F1 |
|---|---|
| PERCENT | 99.71% |
| MONEY | 99.66% |
| DATE | 99.36% |
| PRODUCT | 99.15% |
| PERSON | 96.66% |
| INDUSTRY | 95.72% |
| LOCATION | 95.61% |
| ORGANIZATION | 92.31% |
| EVENT | 79.32% |

---

## 5. Yêu cầu môi trường

| Phần mềm | Phiên bản | Ghi chú |
|---|---|---|
| **Node.js** | 18+ (khuyến nghị 20+) | Dùng nvm hoặc tải từ [nodejs.org](https://nodejs.org) |
| **Python** | 3.11+ | Cần hỗ trợ `asyncio`, `psycopg` v3 |
| **pip** | Mới nhất | `python -m pip install --upgrade pip` |
| **PostgreSQL** | 14+ | Dùng cho chat memory; bắt buộc nếu cần lưu lịch sử |

> **Lưu ý:** Lần chạy đầu tiên, backend cần tải tokenizer và model PhoBERT (~500 MB), có thể mất vài phút.

---

## 6. Cài đặt

### 6.1 Clone repository

```bash
git clone https://github.com/<your-org>/knowledge-graph-extractor.git
cd knowledge-graph-extractor
```

### 6.2 Cài frontend dependencies

```bash
npm install
```

### 6.3 Cài backend dependencies

Cài PyTorch (CPU):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Cài các package còn lại:

```bash
pip install -r requirements.txt
```

**Danh sách dependencies backend:**

| Package | Mục đích |
|---|---|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `slowapi` | Rate limiting theo IP |
| `transformers` | Tải và chạy PhoBERT |
| `torch` | Deep learning runtime |
| `pypdf` | Đọc file PDF |
| `networkx` | Xây dựng và phân tích đồ thị |
| `psycopg[binary]` | Kết nối PostgreSQL (v3) |
| `psycopg_pool` | Connection pooling |
| `httpx` | HTTP client async để gọi LLM API |
| `python-dotenv` | Đọc biến môi trường từ `.env` |
| `rank-bm25` | BM25 retrieval cho RAG pipeline |

### 6.4 Chuẩn bị PostgreSQL

Tạo database:

```bash
createdb kge
```

Chạy migration tạo bảng:

```bash
psql kge -f server/sql/chat_memory.sql
```

Schema tạo ra hai bảng:
- `sessions` — quản lý phiên hội thoại
- `messages` — lưu từng lượt hỏi/đáp theo `session_id`

### 6.5 Tạo file `.env`

Copy từ file mẫu:

```bash
cp .env.example .env
```

Chỉnh sửa các giá trị cần thiết (xem [Mục 7](#7-cấu-hình-môi-trường)).

---

## 7. Cấu hình môi trường

File `.env` đặt ở thư mục gốc của project:

```env
# ─── PostgreSQL (bắt buộc cho chat memory) ──────────────────────────────
DATABASE_URL="postgresql://user:password@localhost:5432/kge"

# ─── LLM Provider (tuỳ chọn) ────────────────────────────────────────────
LLM_PROVIDER=""          # "openai" hoặc "gemini" (để trống = rule-based)
LLM_API_KEY=""           # API key cho provider
LLM_MODEL=""             # (tuỳ chọn) model name, để trống = default theo provider
LLM_BASE_URL=""          # (tuỳ chọn) OpenAI-compatible endpoint
```

**Chi tiết từng biến:**

| Biến | Bắt buộc | Mô tả |
|---|---|---|
| `DATABASE_URL` | Có (cho chat) | Chuỗi kết nối PostgreSQL chuẩn |
| `LLM_PROVIDER` | Không | `openai` hoặc `gemini`; nếu trống → rule-based |
| `LLM_API_KEY` | Không | API key cho provider |
| `LLM_MODEL` | Không | Tên model (nếu để trống dùng default) |
| `LLM_BASE_URL` | Không | Endpoint OpenAI-compatible (tuỳ chọn) |

**Ví dụ cấu hình từng provider:**

```env
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-key
LLM_MODEL=gpt-4o-mini

# Hoặc Gemini
# LLM_PROVIDER=gemini
# LLM_API_KEY=your-gemini-key
# LLM_MODEL=gemini-1.5-flash
```

---

## 8. Chạy dự án local

Mở **2 terminal song song**:

### Terminal 1 — Backend (FastAPI)

```bash
npm run server
# hoặc
python server/main.py
```

Backend khởi động ở `http://localhost:8000`.  
Khi khởi động, backend tự động:
- Tải PhoBERT model và tokenizer
- Load Knowledge Base từ `model/knowledge_graph_output/`
- Khởi tạo connection pool PostgreSQL cho chat memory

Kiểm tra trạng thái:

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_ready": true,
  "model_error": null,
  "kb_ready": true,
  "kb_triples": 1234
}
```

### Terminal 2 — Frontend (React + Vite)

```bash
npm run dev
```

Frontend chạy ở `http://localhost:3000`.

---

## 9. API Backend

Base URL: `http://localhost:8000`

### Tổng quan endpoints

| Endpoint | Method | Rate limit | Mô tả |
|---|---|---|---|
| `/health` | GET | Không giới hạn | Trạng thái model và KB |
| `/extract` | POST | 10 req/phút/IP | Trích xuất entities + relations từ text |
| `/upload-pdf` | POST | 5 req/phút/IP | Upload PDF, trả về text đã bóc tách |
| `/predict-links` | POST | 30 req/phút/IP | Dự đoán quan hệ mới giữa các node |
| `/metrics` | POST | 20 req/phút/IP | Tính chỉ số đồ thị |
| `/insight` | POST | 20 req/phút/IP | Tạo báo cáo insight dạng Markdown |
| `/chat` | POST | 30 req/phút/IP | Chatbot hỏi đáp (hybrid, có memory) |
| `/kb/stats` | GET | Không giới hạn | Thống kê Knowledge Base |
| `/kb/search` | GET | 30 req/phút/IP | Tìm kiếm entity trong KB |
| `/kb/entity` | GET | Không giới hạn | Lấy triples theo entity |

---

### `POST /extract`

Trích xuất đồ thị tri thức từ văn bản. Input tối đa **50.000 ký tự**.

**Request:**
```json
{
  "text": "Vingroup đầu tư vào VinFast tại Hải Phòng năm 2017.",
  "use_deep_analysis": false
}
```

**Response:**
```json
{
  "entities": [
    { "id": "e1", "name": "Vingroup", "type": "Organization", "properties": [] },
    { "id": "e2", "name": "VinFast", "type": "Organization", "properties": [] },
    { "id": "e3", "name": "Hải Phòng", "type": "Location", "properties": [] }
  ],
  "relations": [
    {
      "id": "r1",
      "source": "e1",
      "target": "e2",
      "label": "INVEST_IN",
      "isPredicted": false
    }
  ]
}
```

---

### `POST /upload-pdf`

Upload file PDF (multipart/form-data), trả về text đã bóc tách.

- Giới hạn kích thước: **20 MB**
- Chỉ hỗ trợ file `.pdf`
- PDF scan ảnh (không có text layer) sẽ trả về lỗi 422

**Response:**
```json
{
  "text": "Nội dung văn bản đã trích xuất...",
  "page_count": 10,
  "extracted_pages": 10,
  "filename": "document.pdf",
  "truncated": false
}
```

---

### `POST /predict-links`

Dự đoán quan hệ mới giữa các node bằng `influence_predictor.joblib`.

**Request:**
```json
{
  "entities": [...],
  "relations": [...]
}
```

**Response:**
```json
{
  "predicted_relations": [
    {
      "id": "pred_r1",
      "source": "e1",
      "target": "e3",
      "label": "LOCATED_IN",
      "isPredicted": true
    }
  ]
}
```

---

### `POST /metrics`

Tính các chỉ số phân tích đồ thị bằng NetworkX.

**Các chỉ số trả về:** degree centrality, in-degree, out-degree, PageRank, betweenness centrality, closeness centrality, clustering coefficient, số connected components.

---

### `POST /insight`

Tạo báo cáo phân tích đồ thị dạng Markdown, bao gồm: phân bố entity types, hub nodes, cấu trúc cộng đồng, và các nhận xét ngữ nghĩa.

---

### `POST /chat`

Chatbot hỏi đáp với bộ nhớ hội thoại.

**Request:**
```json
{
  "session_id": null,
  "message": "Mối quan hệ giữa Vingroup và VinFast?",
  "entities": [...],
  "relations": [...],
  "input_text": "Vingroup đầu tư vào VinFast tại Hải Phòng."
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "reply": "## Quan hệ trực tiếp giữa Vingroup và VinFast\n\n- **Vingroup** → *INVEST_IN* → **VinFast**",
  "engine": "rule-based",
  "history": [
    { "role": "user", "content": "Mối quan hệ giữa Vingroup và VinFast?" },
    { "role": "model", "content": "## Quan hệ trực tiếp..." }
  ]
}
```

- `session_id = null` → server tạo session mới, trả về ID để dùng cho các lượt tiếp theo.
- `engine`: `"llm"` hoặc `"rule-based"` cho biết chatbot dùng engine nào để trả lời.

---

### `GET /kb/search?q=<query>&limit=20`

Tìm kiếm entity trong Knowledge Base. Trả về danh sách entity có tên gần khớp.

### `GET /kb/entity?name=<name>&limit=50`

Lấy tất cả triple trong Knowledge Base liên quan đến một entity.

---

## 10. Giao diện Dashboard

Frontend có **7 màn hình chính** được chuyển đổi qua tab navigation:

| Tab | Mô tả |
|---|---|
| **Graph** | Đồ thị tương tác (force-directed), click node để xem chi tiết, zoom/pan |
| **Entities** | Bảng danh sách thực thể: lọc theo loại, sắp xếp theo số kết nối |
| **Relations** | Bảng danh sách quan hệ: phân biệt extracted vs. predicted |
| **Metrics** | Biểu đồ và bảng các chỉ số đồ thị (PageRank, degree, ...) |
| **Insight** | Báo cáo phân tích đồ thị dạng Markdown, tự động tổng hợp |
| **Highlight** | Văn bản gốc với các thực thể được highlight theo màu loại |
| **Chatbot** | Giao diện chat với gợi ý câu hỏi tự động từ đồ thị |

**Quy trình sử dụng cơ bản:**

1. Nhập văn bản tiếng Việt vào ô text hoặc upload file PDF.
2. Nhấn **Trích xuất** → đồ thị được dựng trong tab Graph.
3. (Tuỳ chọn) Nhấn **Dự đoán liên kết** để thêm quan hệ dự đoán vào đồ thị.
4. Chuyển sang tab **Metrics** hoặc **Insight** để phân tích sâu hơn.
5. Mở tab **Chatbot** để hỏi đáp tự nhiên về đồ thị vừa tạo.

---

## 11. Chatbot hỏi đáp

Chatbot là thành phần hỏi đáp chính, hoạt động theo kiến trúc **hybrid**:

```
Câu hỏi người dùng
       │
       ▼
RAG Pipeline (BM25)
  ├── Chunk văn bản gốc
  ├── Entity cards
  ├── Relation documents
  └── KB triple documents
       │
       ▼
  LLM configured?
  ├── Có → gọi local LLM với graph context + RAG context + lịch sử 20 lượt
  └── Không (hoặc LLM lỗi) → Rule-based engine
       │
       ▼
  Lưu vào PostgreSQL → Trả về response
```

### 11.1 Rule-Based Engine (17+ intents)

Hoạt động hoàn toàn offline, không cần API key. Hỗ trợ tiếng Việt và tiếng Anh.

| # | Intent | Ví dụ câu hỏi |
|---|--------|---------------|
| 1 | Chào hỏi | "Xin chào!", "Hello" |
| 2 | Trợ giúp / Capabilities | "Tôi có thể hỏi gì?", "Help" |
| 3 | Tóm tắt đồ thị | "Tóm tắt", "Overview", "Tổng quan đồ thị" |
| 4 | Quan hệ giữa A và B | "Mối quan hệ giữa Vingroup và VinFast?" |
| 5 | So sánh A và B | "So sánh FPT và Viettel", "FPT vs Viettel" |
| 6 | Kết nối của X | "Vingroup kết nối với gì?", "Connections of X" |
| 7 | Top node quan trọng | "Top 5 thực thể quan trọng nhất", "Hub nodes" |
| 8 | Quan hệ dự đoán | "Có quan hệ dự đoán nào?", "Predicted links" |
| 9 | Lọc quan hệ theo loại | "Những ai đầu tư?", "Quan hệ hợp tác" |
| 10 | Trích đoạn văn bản gốc | "Văn bản gốc nói gì về FPT?" |
| 11 | Đếm / thống kê | "Có bao nhiêu tổ chức?", "Count entities" |
| 12 | Danh sách theo loại | "Liệt kê tất cả sự kiện", "Danh sách công ty" |
| 13 | Danh sách quan hệ | "Liệt kê tất cả quan hệ" |
| 14 | Tra cứu KB | "KB biết gì về Hà Nội?" |
| 15 | Tra cứu thực thể (fuzzy) | "Vingroup" (tìm kiếm mờ) |
| 16 | Đa thực thể | "FPT và Viettel" |
| 17 | RAG fallback | Tự động tìm trong BM25 index |
| 18 | Smart fallback | Gợi ý câu hỏi thay thế |

**Fuzzy Entity Matching** — tìm thực thể theo 4 tầng ưu tiên:
1. Exact match (sau lowercase)
2. Substring match
3. Diacritics-stripped match (bỏ dấu tiếng Việt)
4. SequenceMatcher fuzzy (ngưỡng ≥ 0.50)

### 11.2 LLM Mode

Khi cấu hình `LLM_PROVIDER` + `LLM_API_KEY`, chatbot gọi cloud LLM với:
- **System prompt** chi tiết: vai trò, nguyên tắc chống hallucination, format trả lời
- **Graph context**: danh sách entities (có degree score), relations, hub nodes, reasoning hints
- **RAG context**: top-12 đoạn văn bản liên quan từ BM25 index
- **Chat history**: tối đa 20 lượt hội thoại gần nhất

Nếu LLM timeout hoặc lỗi, chatbot tự động fallback sang rule-based — người dùng không bị gián đoạn.

### 11.3 Bộ nhớ hội thoại

- `session_id` lưu trong `localStorage` của trình duyệt (key: `kge_chat_session_id`).
- Lịch sử hội thoại lưu persistent trong PostgreSQL — không mất khi reload trang.
- Nhấn nút **Reset** để xoá session và bắt đầu hội thoại mới.
- Nếu PostgreSQL không khả dụng, chatbot tự động chuyển sang ephemeral mode (không lưu lịch sử) và vẫn hoạt động bình thường.

---

## 12. Scripts NPM

| Script | Mô tả |
|---|---|
| `npm run dev` | Chạy frontend Vite dev server ở port 3000 (hot reload) |
| `npm run server` | Chạy FastAPI backend (`python server/main.py`) |
| `npm run build` | Build frontend cho production (output: `dist/`) |
| `npm run preview` | Preview bản build production local |
| `npm run clean` | Xoá thư mục `dist/` |
| `npm run lint` | Type check TypeScript (`tsc --noEmit`) |

---

## 13. Lỗi thường gặp

### Backend không khởi động

**Triệu chứng:** `python server/main.py` báo lỗi import.

```bash
# Kiểm tra phiên bản Python
python --version  # cần >= 3.11

# Cài lại dependencies
pip install -r requirements.txt
```

### Model chưa sẵn sàng (`model_ready: false`)

**Nguyên nhân:** Lần đầu chạy cần tải model (~500 MB) hoặc thiếu file model.

- Chờ thêm 1–3 phút và kiểm tra lại `GET /health`.
- Xem log terminal backend để biết chi tiết lỗi tải model.
- Đảm bảo thư mục `model/phobert-ner-final/` có đủ file (đặc biệt `model.safetensors`).

### Upload PDF lỗi

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `400 Only PDF files are supported` | Sai định dạng file | Chỉ upload file `.pdf` |
| `413 File exceeds maximum size` | File > 20 MB | Nén hoặc chia nhỏ PDF |
| `422 Could not extract text` | PDF scan ảnh thuần | Cần OCR trước khi upload |

### Chat không lưu lịch sử

1. Kiểm tra PostgreSQL đang chạy: `pg_isready`
2. Kiểm tra `DATABASE_URL` trong `.env` đúng cú pháp.
3. Chạy lại migration: `psql $DATABASE_URL -f server/sql/chat_memory.sql`
4. Xem log backend: tìm dòng `"Chat memory DB not available"`.

### LLM không phản hồi

1. Kiểm tra `LLM_PROVIDER`, `LLM_API_KEY` (và `LLM_MODEL` nếu có) trong `.env`.
2. Kiểm tra API key còn hiệu lực và model name hợp lệ.
3. Chat sẽ tự fallback sang rule-based nếu LLM lỗi — xem field `engine` trong response.

### Frontend không kết nối được backend

1. Đảm bảo backend đang chạy ở `:8000`: `curl http://localhost:8000/health`
2. Nếu backend chạy ở port khác, cập nhật `VITE_API_BASE_URL` hoặc cấu hình trong `src/lib/ai.ts`.
3. Kiểm tra CORS — backend mặc định cho phép tất cả origin (`allow_origins=["*"]`).

---

## 14. Gợi ý mở rộng

- **Docker Compose** — đóng gói frontend + backend + PostgreSQL để deploy dễ dàng hơn.
- **OCR support** — tích hợp Tesseract hoặc PaddleOCR để xử lý PDF scan ảnh.
- **Thêm provider mới** — mở rộng `server/llm_client.py` cho Claude/Ollama/OpenAI-compatible endpoint khác.
- **Export đồ thị** — thêm tính năng export sang GEXF, GraphML, hoặc PNG từ giao diện.
- **Authentication** — thêm JWT auth để phân quyền truy cập multi-user.
- **Tests** — bổ sung `pytest` cho backend API và Vitest cho frontend components.
- **Chuẩn hóa script đa nền tảng** — thay `rm -rf` trong `npm run clean` bằng `rimraf` cho Windows/Linux/macOS.
- **Streaming chat** — chuyển `/chat` sang Server-Sent Events để stream token từ LLM theo thời gian thực.
- **Fine-tune thêm** — huấn luyện lại PhoBERT trên domain cụ thể (y tế, pháp lý, ...) để tăng độ chính xác NER.

---

## Stack công nghệ

| Layer | Công nghệ |
|---|---|
| **Frontend** | React 19, Vite 6, TypeScript, Tailwind CSS 4 |
| **Đồ thị UI** | react-force-graph-2d, D3.js |
| **Markdown render** | react-markdown, remark-gfm |
| **Animation** | motion (Framer Motion) |
| **Icons** | lucide-react |
| **Backend** | FastAPI, Uvicorn, Python 3.11+ |
| **NER Model** | PhoBERT (VinAI) fine-tuned, HuggingFace Transformers |
| **Graph analysis** | NetworkX |
| **Link prediction** | scikit-learn (influence_predictor.joblib) |
| **RAG** | rank-bm25 |
| **Database** | PostgreSQL 14+, psycopg v3, psycopg_pool |
| **LLM** | OpenAI / Gemini (tuỳ chọn) |
| **Rate limiting** | SlowAPI |
