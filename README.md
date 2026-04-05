 # Knowledge Graph Extractor

Ứng dụng trích xuất thực thể (NER), xây dựng quan hệ và trực quan hóa đồ thị tri thức từ văn bản/PDF tiếng Việt.

Stack chính:
- **Frontend**: React + Vite + TypeScript
- **Backend**: FastAPI (Python)
- **Mô hình NER**: PhoBERT (VinAI)

---

## 1) Tính năng chính

- Trích xuất thực thể và quan hệ từ văn bản (`/extract`)
- Upload PDF, tự bóc tách text (`/upload-pdf`)
- Dự đoán liên kết mới giữa các node (`/predict-links`)
- Tính graph metrics (degree, pagerank, betweenness, ...) (`/metrics`)
- Tra cứu Knowledge Base cục bộ (`/kb/*`)
- Dashboard trực quan: Graph, Entities, Relations, Insight, Chatbot, Metrics, Highlight

---

## 2) Kiến trúc hệ thống

```text
React Frontend (Vite, :3000)
        ↕ HTTP
FastAPI Backend (Python, :8000)
        ↕
PhoBERT NER + Knowledge Base + Metrics
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
│  └─ schemas.py                # Pydantic schemas
├─ model/                       # Trọng số model + dữ liệu hỗ trợ
├─ feature_engineering_output/  # Artifacts huấn luyện/đánh giá
└─ README.md
```

---

## 4) Yêu cầu môi trường

- Node.js 18+ (khuyến nghị 20+)
- Python 3.11+
- Pip

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
pip install fastapi uvicorn transformers scikit-learn joblib pypdf networkx
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
API_KEY="YOUR_API_KEY_HERE"
APP_URL="MY_APP_URL"
```

Hiện tại các chức năng cốt lõi hoạt động không bắt buộc API key. Biến môi trường dùng khi bạn mở rộng tích hợp LLM/provider ngoài.

---

## 10) Mở rộng tích hợp AI

Điểm mở rộng chính nằm ở `src/lib/ai.ts`:

- `callExtract`: gọi backend NER
- `callPredictLinks`: gọi backend dự đoán quan hệ
- `callMetrics`: gọi backend metrics
- `callInsight`: hiện là rule-based local
- `callChat`: hiện là rule-based local

Bạn có thể thay `callInsight` và `callChat` bằng OpenAI/Gemini/Ollama tùy nhu cầu.

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

---

## 12) Gợi ý cải thiện tiếp

- Thêm Docker Compose cho frontend + backend
- Bổ sung test API (`pytest`) và test frontend
- Chuẩn hóa script đa nền tảng (Windows/Linux/macOS)
