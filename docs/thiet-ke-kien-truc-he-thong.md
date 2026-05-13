# Thiet Ke Kien Truc He Thong - Knowledge Graph Extractor

## 1) Tong quan kien truc

He thong su dung mo hinh 3 lop:

- Frontend: React + TypeScript (Vite)
- Backend API: FastAPI (Python)
- Data layer: PostgreSQL + file storage cho model/graph artifacts

## 2) Muc tieu kien truc

- Dap ung luong xu ly NER + graph trong thoi gian gan realtime
- Tach biet trach nhiem giua cac module de de bao tri
- Mo rong de tich hop AI provider va nguon du lieu moi
- Dam bao kha nang theo doi, truy vet va xu ly loi

## 3) Thanh phan chinh

### 3.1 Frontend

- Dashboard dieu huong cac tab: Graph, Entities, Relations, Metrics, Insight, Chatbot
- Quan ly state phien lam viec, input van ban/PDF, ket qua extraction
- Goi REST API qua lop client tap trung (`src/lib/ai.ts`)

### 3.2 Backend FastAPI

- `main.py`: route registry, CORS, rate limiting
- `ner.py`: trich xuat entity bang PhoBERT fine-tuned
- `graph.py`: tao relation va do thi tri thuc
- `metrics.py`: degree, pagerank, betweenness, clustering
- `insights.py`: tong hop nhan dinh tu cau truc do thi
- `rag.py` + `chat_service.py`: pipeline retrieval + answer generation
- `chat_memory.py`: luu hoi thoai vao PostgreSQL

### 3.3 Data va model

- Model NER tai thu muc `model/phobert-ner-final`
- Artifacts graph va triples tai `model/knowledge_graph_output`
- Du lieu hoi thoai tai PostgreSQL (sessions/messages)

## 4) Luong xu ly nghiep vu

1. Nguoi dung nhap text hoac upload PDF.
2. Frontend goi API extraction.
3. Backend chay NER -> tra danh sach entity.
4. Graph builder suy luan relation -> tao graph.
5. Metrics/insight duoc tinh toan tu graph.
6. Chatbot nhan cau hoi, lay context bang RAG, tra loi bang rule-based hoac LLM.
7. Lich su chat duoc luu theo `session_id`.

## 5) Kien truc logic (text diagram)

```text
[React Frontend]
   |
   | HTTP/JSON
   v
[FastAPI Gateway]
   |-- NER Service (PhoBERT)
   |-- Graph Service (NetworkX)
   |-- Metrics/Insight Service
   |-- Chat Service (RAG + LLM adapter)
   |
   |-- PostgreSQL (chat memory)
   |-- Model/Artifact Storage (local files)
```

## 6) Non-functional requirements

- Performance:
  - Thoi gian phan tich 1 van ban ngan < 3s (muc tieu staging)
  - Thoi gian phan hoi chat thong thuong < 5s (khong tinh timeout provider)
- Reliability:
  - Co co che retry cho loi tam thoi tu LLM provider
  - Logging co `request_id` de truy vet
- Security:
  - Rate limiting theo IP
  - Validate input size/file type
  - Quan ly secret qua bien moi truong

## 7) Huong mo rong de xuat

- Tach `NER` va `Chat` thanh microservice khi tai tang
- Them message queue cho batch ingestion
- Them cache cho retrieval va metrics tinh toan lap lai
- Tich hop observability stack (OpenTelemetry + Prometheus + Grafana)
