# 🗺️ Sơ đồ Kiến trúc Hệ thống — Knowledge Graph Extractor

## 1. Kiến trúc Tổng thể (High-Level)

```mermaid
graph TB
    subgraph USER[" Người dùng"]
        B[Trình duyệt Web]
    end

    subgraph FE["🖥️ Frontend — React 19 + Vite + TypeScript"]
        direction TB
        APP[App.tsx — Routing & State]
        DASH[Dashboard.tsx — Tab Orchestrator]

        subgraph VIEWS["Views / Components"]
            V1[📊 GraphView\nForce-Graph 2D]
            V2[🔤 EntitiesView\nDanh sách thực thể]
            V3[🔗 RelationsView\nDanh sách quan hệ]
            V4[📈 MetricsView\nChỉ số đồ thị]
            V5[💡 InsightView\nPhân tích tự động]
            V6[🤖 ChatbotView\nHội thoại AI]
            V7[🖊️ HighlightView\nVăn bản gốc]
        end

        INPUT[InputPanel.tsx\nNhập text / Upload PDF]
        APICLIENT[src/lib/ai.ts\nHTTP Client Layer]
        APP --> DASH
        DASH --> VIEWS
        DASH --> INPUT
        DASH --> APICLIENT
    end

    subgraph BE["⚙️ Backend — FastAPI + Python + Uvicorn"]
        direction TB
        MAIN[main.py\nRoute Registry + CORS + Rate Limiter]

        subgraph SERVICES["Core Services"]
            NER[ner.py\nNER Engine]
            GRAPH[graph.py\nGraph Builder + Relation Inference]
            METRICS[metrics.py\nGraph Metrics\n— PageRank, Betweenness, Clustering]
            INSIGHT[insights.py\nInsight Report Generator]
            LINKPRED[link_predictor.py\nLink Prediction Pipeline]
        end

        subgraph CHAT["Chat & RAG Pipeline"]
            CHATSVC[chat_service.py\nHybrid Chat Orchestrator]
            RAG[rag.py\nBM25 Retrieval]
            QU[query_understanding.py\nIntent Classification]
            CTXF[context_filter.py\nContext Ranker]
            CHATMEM[chat_memory.py\nSession Memory]
            WSMEM[workspace_memory.py\nWorkspace Persistence]
        end

        subgraph LLMADAPTER["LLM Adapter Layer"]
            LLMC[llm_client.py\nRemote LLM Dispatcher]
            LOCAL[local_llm.py\nOllama Local Model]
        end

        DBLOG[db_logger.py\nPerformance Logger]
        KB[knowledge_base.py\nLocal Knowledge Base]
        ANSVAL[answer_validator.py\nAnswer Quality Check]

        MAIN --> SERVICES
        MAIN --> CHAT
        CHATSVC --> RAG & QU & CTXF & CHATMEM & LLMADAPTER & ANSVAL
        LLMADAPTER --> LLMC & LOCAL
    end

    subgraph ML["🧠 ML Models Layer"]
        PHOBERT[(PhoBERT NER\nmodel/phobert-ner-final)]
        SKLMODEL[(HistGradientBoosting\ninfluence_predictor.joblib)]
    end

    subgraph EXT["🌐 External LLM APIs"]
        OPENAI[OpenAI GPT]
        GEMINI[Google Gemini]
        OLLAMA[Ollama — Local\nqwen2.5 / llama]
    end

    subgraph STORAGE["🗄️ Data & Storage Layer"]
        PG[(PostgreSQL\n— chat_sessions\n— chat_messages\n— workspaces\n— documents\n— request_logs)]
        FS[(Local Filesystem\n— GEXF / GraphML\n— Triples JSON\n— PDF uploads)]
        KBFILES[(Knowledge Base Files\n— knowledge_graph.gexf\n— graph_triples.json)]
    end

    %% ── Connections ──
    B <-->|HTTP| FE
    FE <-->|"REST / JSON\nHTTP :8000"| BE

    NER -->|Token Classification| PHOBERT
    LINKPRED -->|Inference| SKLMODEL

    GRAPH --> FS
    KB --> KBFILES
    RAG --> KB

    CHATSVC --> PG
    CHATMEM --> PG
    WSMEM --> PG
    DBLOG --> PG

    LLMC -->|API Key| OPENAI & GEMINI
    LOCAL -->|HTTP :11434| OLLAMA
```

---

## 2. Luồng Xử lý Trích xuất Đồ thị

```mermaid
sequenceDiagram
    actor User as 👤 Người dùng
    participant FE as Frontend
    participant API as FastAPI /extract
    participant NER as ner.py (PhoBERT)
    participant Graph as graph.py (NetworkX)
    participant DB as db_logger → PostgreSQL

    User->>FE: Nhập text / Upload PDF
    alt Upload PDF
        FE->>API: POST /upload-pdf
        API-->>FE: { text, page_count }
    end
    FE->>API: POST /extract { text }
    API->>DB: Ghi log request (PerfTimer start)
    API->>NER: run_ner(text)
    NER->>NER: PhoBERT tokenize + inference
    NER-->>API: List[Entity] (type, label, span)
    API->>Graph: build_graph(entities, text)
    Graph->>Graph: Suy luận quan hệ (rule + co-occurrence)
    Graph-->>API: GraphData { entities, relations }
    API->>DB: Ghi log response time
    API-->>FE: GraphData JSON
    FE->>FE: Render Force-Graph 2D
```

---

## 3. Luồng Chatbot Hybrid (RAG + LLM)

```mermaid
sequenceDiagram
    actor User as  Người dùng
    participant FE as Frontend (ChatbotView)
    participant API as FastAPI /chat
    participant QU as query_understanding.py
    participant RAG as rag.py (BM25)
    participant KB as knowledge_base.py
    participant MEM as chat_memory.py (PostgreSQL)
    participant LLM as LLM Client\n(OpenAI / Gemini / Ollama)

    User->>FE: Gửi câu hỏi
    FE->>API: POST /chat { message, session_id, graph_context }
    API->>MEM: Lấy lịch sử hội thoại (session_id)
    MEM-->>API: List[ChatTurn]
    API->>QU: Phân loại Intent + trích xuất keyword
    QU-->>API: Intent + entities
    API->>RAG: Truy xuất ngữ cảnh liên quan
    RAG->>KB: BM25 search(query, top-k)
    KB-->>RAG: Relevant triples / nodes
    RAG-->>API: Context chunks

    alt Có API Key LLM
        API->>LLM: Prompt = System + Context + History + Question
        LLM-->>API: Generated answer
    else Rule-based fallback
        API->>API: Pattern matching → Template answer
    end

    API->>MEM: Lưu turn (user + assistant)
    API-->>FE: { answer, confidence, sources }
    FE->>FE: Render markdown response
```

---

## 4. Luồng Link Prediction

```mermaid
sequenceDiagram
    actor User as  Người dùng
    participant FE as Frontend (GraphView)
    participant API as FastAPI /predict-links
    participant LP as link_predictor.py
    participant ML as HistGradientBoosting Model
    participant KB as knowledge_base.py

    User->>FE: Click "Dự đoán quan hệ mới"
    FE->>API: POST /predict-links { entities, relations }
    API->>LP: predict_new_links(entities, relations)
    LP->>LP: 1. Type-pair heuristics
    LP->>LP: 2. Neighbor / transitivity scoring
    LP->>KB: 3. Tra cứu KB triples
    LP->>ML: 4. Influence Predictor inference
    ML-->>LP: Confidence scores
    LP->>LP: 5. Merge + sort by confidence
    LP-->>API: List[PredictedRelation] (score, confidence)
    API-->>FE: PredictLinksResponse
    FE->>FE: Hiển thị predicted edges (nét đứt)
```

---

## 5. Sơ đồ Cơ sở dữ liệu

```mermaid
erDiagram
    CHAT_SESSIONS {
        uuid id PK
        text session_name
        timestamp created_at
        timestamp updated_at
    }

    CHAT_MESSAGES {
        uuid id PK
        uuid session_id FK
        text role
        text content
        jsonb metadata
        timestamp created_at
    }

    WORKSPACES {
        uuid id PK
        text title
        text input_text
        jsonb graph_data
        jsonb metrics_data
        text insight_markdown
        text chat_session_id
        text chat_engine
        jsonb chat_history
        text active_tab
        timestamp created_at
        timestamp updated_at
    }

    DOCUMENTS {
        uuid id PK
        uuid workspace_id FK
        text filename
        text file_type
        int page_count
        int char_count
        bool truncated
        timestamp created_at
    }

    REQUEST_LOGS {
        uuid id PK
        text endpoint
        text model_name
        int input_length
        float duration_ms
        int status_code
        timestamp created_at
    }

    CHAT_SESSIONS ||--o{ CHAT_MESSAGES : "có"
    WORKSPACES ||--o{ DOCUMENTS : "chứa"
```

---

## 6. Stack Công nghệ Tóm tắt

| Tầng                | Công nghệ                                               |
| ------------------- | ------------------------------------------------------- |
| **Frontend**        | React 19, TypeScript, Vite, react-force-graph-2d, D3.js |
| **Backend**         | FastAPI, Uvicorn, Python 3.11+                          |
| **NER Model**       | PhoBERT fine-tuned (HuggingFace Transformers + PyTorch) |
| **Link Prediction** | Scikit-learn HistGradientBoostingClassifier             |
| **Graph Engine**    | NetworkX                                                |
| **RAG Search**      | BM25 (rank-bm25)                                        |
| **Chat Memory**     | PostgreSQL (psycopg2)                                   |
| **LLM Adapters**    | OpenAI API, Google Gemini API, Ollama (local)           |
| **File Storage**    | Local Filesystem — GEXF, GraphML, JSON triples          |
| **Rate Limiting**   | SlowAPI (per-IP)                                        |
| **PDF Parsing**     | pypdf                                                   |
