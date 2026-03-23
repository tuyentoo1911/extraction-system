# Knowledge Graph Extractor

Hệ thống trích xuất thông tin từ tài liệu, nhận diện thực thể, xác định mối quan hệ và xây dựng đồ thị tri thức tự động — sử dụng mô hình NER **PhoBERT-large** (VinAI, tiếng Việt).

## Kiến trúc hệ thống

```
React Frontend (Vite)  ←→  FastAPI Backend (Python)  ←→  PhoBERT NER Model
     port 3000                   port 8000                  model/model.bin
```

## Cấu trúc dự án

```
model/
├── model.bin            # PhoBERT-large NER model (PyTorch, ~1.4 GB)
└── meta.bin             # sklearn LabelEncoder (85 nhãn NER, BIO format)
server/
└── main.py              # FastAPI server — load model và serve API
src/
├── types.ts             # Interfaces dùng chung
├── lib/
│   └── ai.ts            # Gọi backend NER API (callExtract, callChat, ...)
├── constants/
│   └── graph.tsx        # Màu sắc, icon, hằng số đồ thị
├── components/
│   ├── TabButton.tsx
│   ├── InputPanel.tsx
│   └── views/
│       ├── GraphView.tsx
│       ├── EntitiesView.tsx
│       ├── RelationsView.tsx
│       ├── InsightView.tsx
│       └── ChatbotView.tsx
├── Dashboard.tsx        # Màn hình chính
└── App.tsx              # Landing page
```

## Chạy locally

**Yêu cầu:** Node.js, Python 3.11+

### 1. Cài frontend dependencies

```bash
npm install
```

### 2. Cài Python backend dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn transformers scikit-learn joblib
```

### 3. Chạy backend server (Terminal 1)

```bash
npm run server
# hoặc: python server/main.py
# Server tại http://localhost:8000
# Lần đầu sẽ tự tải PhoBERT tokenizer (~vài phút)
```

### 4. Chạy frontend (Terminal 2)

```bash
npm run dev
# App tại http://localhost:3000
```

## API Backend

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/health` | GET | Kiểm tra model đã sẵn sàng chưa |
| `/extract` | POST | Trích xuất entities và relations từ văn bản |
| `/predict-links` | POST | Dự đoán liên kết mới giữa các entities |

## Model NER

- **Kiến trúc**: PhoBERT-large (vocab 64001, hidden 1024, 24 layers)
- **Tokenizer**: `vinai/phobert-large`
- **Nhãn**: 85 nhãn BIO — PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, DATETIME, QUANTITY, ADDRESS, EMAIL, URL, SKILL...
- **Ngôn ngữ**: Tiếng Việt

## Tích hợp AI model khác

Để thêm LLM cho chat/insight, mở `src/lib/ai.ts` và implement:

| Hàm | Hiện tại | Có thể thêm |
|-----|----------|-------------|
| `callExtract` | PhoBERT NER (backend) | — |
| `callPredictLinks` | Rule-based (backend) | — |
| `callInsight` | Rule-based (local) | OpenAI, Gemini, Ollama... |
| `callChat` | Rule-based (local) | OpenAI, Gemini, Ollama... |
