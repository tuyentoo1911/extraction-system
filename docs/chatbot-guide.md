# Chatbot Guide - Knowledge Graph Extractor

Tai lieu nay mo ta chi tiet chatbot trong project: kien truc, luong xu ly, cau hinh, API, va cach mo rong.

---

## 1. Tong quan

Chatbot la thanh phan hoi dap tu dong dua tren du lieu do thi tri thuc duoc trich xuat tu van ban/PDF.

He thong hoat dong theo mo hinh **hybrid**:
- **LLM mode**: neu da cau hinh `LLM_PROVIDER` (`ollama` hoac `local`).
- **Rule-based mode**: fallback tu dong khi LLM khong cau hinh hoac goi LLM loi, hoac khi query duoc phan loai la "deterministic" (count, summary, help...).

Chatbot ho tro:
- Hoi dap tren entities/relations hien co.
- Truy van bo sung tu Knowledge Base.
- RAG retrieval + Graph retrieval de tang do lien quan cua context.
- Phan loai intent va chon mode xu ly (deterministic / graph_first / hybrid).
- Kiem tra va loc hallucination qua Answer Validator.
- Luu lich su hoi thoai bang PostgreSQL theo `session_id`.
- Nhan them `insight_markdown` va `metrics_summary` tu frontend de enrich context.

---

## 2. Kien truc chatbot

```text
User question
   |
   v
Frontend ChatbotView (React)
   |
   | POST /chat
   v
chat_service.handle_chat()
   |
   |── 1. Session management (ensure_session, load history)
   |
   |── 2. Query Understanding
   |       parse_query() → ParsedQuery
   |         intent / mode / entities_mentioned / is_followup
   |
   |── 3. Routing
   |
   |   [deterministic intent hoac LLM khong co] ──→ _rule_based_reply()
   |                                                  engine = "rule-based"
   |
   |   [LLM path] ──→ hybrid_retrieve()
   |                   filter_context()
   |                   format_context_for_llm()
   |                   _call_llm_with_context()
   |                     └─ one-shot continuation neu trucat
   |                   validate_answer()
   |                     └─ neu low grounding → NO_DATA_REPLY
   |                   engine = "ollama" | "local"
   |
   |── 4. Fallback neu LLM loi → _rule_based_reply()
   |
   |── 5. _normalize_reply_text()
   |── 6. Save response to memory
   |── 7. Return ChatResponse
   v
ChatResponse {session_id, reply, engine, history, confidence, evidence_count, intent}
```

---

## 3. Cac module lien quan

### `server/chat_service.py`
Trung tam dieu phoi chatbot:
- Quan ly session va history.
- Goi `parse_query` de hieu intent.
- Dieu huong sang rule-based hoac LLM path.
- Build compact KG string va KB text cho LLM prompt.
- Goi `hybrid_retrieve → filter_context → LLM → validate_answer`.
- One-shot continuation khi LLM tra loi bi cut off.
- Normalize reply text truoc khi luu va tra ve.
- Tra ve `ChatResponse`.

### `server/query_understanding.py`
Phan tich query truoc retrieval:
- Phan loai **intent**: `relationship`, `count`, `summary`, `entity_lookup`, `compare`, `kb_lookup`, `help`, `greeting`, `relation_list`, `neighbors`, `top_nodes`, `predicted`, `source_text`, `type_list`, `unknown`.
- Phan loai **mode**: `deterministic` (route rule-based thang), `graph_first`, `hybrid`.
- Detect entity mention (exact → substring → bo dau → fuzzy).
- Detect follow-up query (pronoun + short query + history).
- Resolve entity tu history khi query follow-up mo ho.

### `server/graph_retriever.py`
Hybrid retrieval layer:
- Graph docs: `graph_entity`, `graph_relation`, `graph_path`, `kb_triple`.
- BM25 docs tu `rag.py`: `input_text`, `entity`, `relation`, `kb_triple`, `insight`, `metrics`.
- Fusion strategy:
  - `graph_first`: graph docs truoc, BM25 bo sung.
  - `hybrid`: interleave graph va BM25 theo rank.

### `server/context_filter.py`
Loc va tinh gia context truoc khi gui LLM:
- Tinh overlap theo entity trong query.
- Boost score theo source type (graph_relation/graph_path uu tien cao).
- Loai bo chunk gan-trung (SequenceMatcher dedup).
- Cat theo token budget theo mode.

### `server/rag.py`
BM25 RAG pipeline:
- Index cac nguon: `input_text` chunks, `entity` docs, `relation` docs, `kb_triple` docs, `insight` chunks, `metrics` chunks.
- Cat `input_text` thanh chunks (`CHUNK_SIZE=300`, overlap `80`).
- Build cache index theo content hash.
- Retrieve top-K context (`TOP_K=12`, max context chars `3000`).
- Ho tro `retrieve_for_rule_based()` cho rule-based fallback.

### `server/answer_validator.py`
Guardrail sau LLM:
- Kiem tra no-data phrase.
- Trich xuat claims tu markdown pattern.
- Doi chieu claims voi retrieved context.
- Tinh `confidence` (0-1), `evidence_count`, `unsupported_claims`.
- Neu grounding qua thap va khong co evidence → thay no-data response chuan.

### `server/llm_client.py`
Adapter goi LLM provider:
- `ollama` (khuyen nghi): goi `POST /api/chat` den Ollama local server.
- `local`: Qwen2.5 + LoRA trong thu muc model.
- Neu khong cau hinh hop le → `is_configured() == False`.

### `server/chat_memory` (module)
Luu bo nho hoi thoai tren PostgreSQL:
- `chat_sessions`: thong tin session.
- `chat_messages`: tung luot chat.
- Dung `psycopg_pool` de quan ly ket noi.
- Tu dong apply schema khi init.

---

## 4. Luong xu ly 1 request chat

1. Frontend gui `POST /chat` kem:
   - `session_id`
   - `message`
   - `entities`, `relations`
   - `input_text`
   - `insight_markdown` (optional — markdown tu Insight tab)
   - `metrics_summary` (optional — text tu Metrics tab)
2. Backend dam bao session ton tai (`ensure_session`).
3. Load history truoc khi add user message (de detect follow-up).
4. Luu user message vao DB.
5. `parse_query()` → `ParsedQuery` (intent, mode, entities_mentioned, is_followup).
6. Neu `deterministic` intent hoac LLM khong cau hinh → rule-based truc tiep.
7. Neu LLM path:
   - `hybrid_retrieve()` → scored docs (graph + BM25).
   - `filter_context()` → filtered docs.
   - `format_context_for_llm()` → context string.
   - `_call_llm_with_context()` → LLM reply (+ one-shot continuation neu trucat).
   - `validate_answer()` → confidence, evidence_count.
   - Neu low grounding → thay `NO_DATA_REPLY`.
8. Neu LLM loi → fallback rule-based.
9. `_normalize_reply_text()` de clean up escape chars.
10. Luu model response vao DB.
11. Lay history gan nhat (toi da 20 turns) de tra ve frontend.
12. Tra `ChatResponse`.

---

## 5. Rule-based intents

Rule-based engine ho tro nhieu intent cho TV/EN (15+):
- Chao hoi
- Help/capabilities (data-aware, goi y cau hoi cu the)
- Tom tat do thi
- Tim quan he giua A va B (truc tiep + 2-hop + KB fallback)
- So sanh A va B (bang markdown)
- Lay neighbors cua 1 entity
- Top hub node (most connected)
- Liet ke quan he du doan
- Loc theo loai quan he
- Trich doan tu van ban goc
- Dem/thong ke entities-relations
- Liet ke theo entity type
- Liet ke tat ca relations
- Tra cuu Knowledge Base
- RAG fallback (insight/metrics/entity/relation docs)
- Smart fallback (KB search + goi y)

Rule-based cung nhan `insight_markdown` va `metrics_summary` de phong phu them ket qua fallback.

### Fuzzy matching
Thu tu uu tien:
1. Exact match
2. Substring match
3. Bo dau tieng Viet de so khop
4. SequenceMatcher (threshold = 0.50)

---

## 6. API `POST /chat`

### Request

```json
{
  "session_id": null,
  "message": "Moi quan he giua Vingroup va VinFast la gi?",
  "entities": [],
  "relations": [],
  "input_text": "Vingroup dau tu vao VinFast tai Hai Phong.",
  "insight_markdown": "",
  "metrics_summary": ""
}
```

### Response

```json
{
  "session_id": "a1b2c3d4e5f6g7h8",
  "reply": "## Quan he truc tiep ...",
  "engine": "ollama",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "model", "content": "..." }
  ],
  "confidence": 0.82,
  "evidence_count": 3,
  "intent": "relationship"
}
```

### Cac truong response
| Truong | Mo ta |
|---|---|
| `session_id` | ID phien hoi thoai |
| `reply` | Cau tra loi (markdown) |
| `engine` | `"ollama"` / `"local"` / `"rule-based"` |
| `history` | Lich su hoi thoai gan nhat (toi da 20 turns) |
| `confidence` | Diem tin cay 0-1 (heuristic grounding) |
| `evidence_count` | So chunks context ho tro cau tra loi |
| `intent` | Intent duoc phan loai: `relationship`, `count`, `summary`... |

### Gioi han theo schema
- `message`: toi da 4000 ky tu.
- `input_text`: toi da 50000 ky tu.
- `insight_markdown`: toi da 200000 ky tu.
- `metrics_summary`: toi da 80000 ky tu.
- `entities`: toi da 500.
- `relations`: toi da 5000.

---

## 7. Cau hinh bien moi truong

Dat trong file `.env` tai root project.

### Bat buoc cho memory

```env
DATABASE_URL="postgresql://user:password@localhost:5432/kge"
```

### Cau hinh Ollama (khuyen nghi)

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT=60
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.2
```

### Cau hinh Local model (Qwen2.5 + LoRA)

```env
LLM_PROVIDER=local
LLM_LOCAL_MODEL_DIR=./model
```

### Bien tuy chon

```env
CHAT_PREFER_RULE_FOR_KEYWORDS=true   # Route keyword-heavy query sang rule-based
```

---

## 8. Session memory va persistence

- Neu frontend gui `session_id` hop le: tiep tuc hoi thoai cu.
- Neu `session_id` null/khong ton tai: tao session moi.
- Tat ca messages duoc luu vao `chat_messages`.
- History duoc load TRUOC khi add user message moi (can thiet cho follow-up detection).
- Frontend co the luu `session_id` trong localStorage de multi-turn.
- Khi DB khong san sang, chatbot van tra loi (degraded/ephemeral mode), nhung khong luu duoc lich su.

---

## 9. RAG context va Insight/Metrics

Context gui cho LLM duoc tong hop tu nhieu nguon:

**BM25 RAG sources:**
- Source Text Excerpts (input_text chunks)
- Entity Information
- Graph Relations
- Knowledge Base triples
- Insight chunks (tu `insight_markdown`)
- Metrics chunks (tu `metrics_summary`)

**Graph sources (graph_retriever):**
- Entity cards + related relations
- Direct triples giua entities
- 2-hop paths cho relationship query
- KB triples bo sung

**Context sau filter:**
- Overlap score theo entity mention
- Boost theo source type (graph relation uu tien)
- Dedup by SequenceMatcher
- Cat theo token budget

---

## 10. Van de thuong gap

### Khong luu duoc chat history
- Kiem tra `DATABASE_URL`.
- Kiem tra PostgreSQL dang chay.
- Kiem tra logs backend: "Chat memory DB not available".

### Chatbot luon o rule-based mode
- Kiem tra `LLM_PROVIDER` la `ollama` hoac `local`.
- Neu dung Ollama: dam bao `ollama serve` dang chay va model da duoc pull.
- Kiem tra logs backend de xem loi ket noi.

### Phan hoi chua sat ngu canh
- Tang chat luong entities/relations tu buoc extract.
- Dam bao `input_text` day du.
- Dieu chinh `TOP_K`, `MAX_CONTEXT_CHARS` trong `rag.py`.
- Gui them `insight_markdown` va `metrics_summary` trong request.

### Cau tra loi bi cat giua chung
- Tang `LLM_MAX_NEW_TOKENS` (mac dinh 512).
- He thong da co one-shot continuation tu dong, nhung neu van trucat → tang token budget.

### Confidence thap / evidence_count = 0
- Answer Validator khong tim thay du lieu ho tro trong context → thay bang `NO_DATA_REPLY`.
- Xem logs backend: "Answer rejected (low_grounding, evidence=0)".

---

## 11. Goi y mo rong

- Them re-ranker sau BM25 de cai thien retrieval quality.
- Them stream response (SSE/WebSocket) cho tra loi real-time.
- Them citation (`evidence_ids`) trong ChatResponse tra ve frontend.
- Them dashboard debug trace: `query → retrieved → filtered → answer`.
- Nang cao Answer Validator: NLI hoac claim verifier thay vi lexical matching.
- Them bo loc an toan output (safety + PII masking) truoc khi tra frontend.

---

## 12. File tham khao nhanh

- `server/chat_service.py` — orchestration chinh
- `server/query_understanding.py` — intent/mode parsing
- `server/graph_retriever.py` — hybrid retrieval
- `server/context_filter.py` — context scoring va filtering
- `server/rag.py` — BM25 RAG pipeline
- `server/answer_validator.py` — grounding check
- `server/llm_client.py` — LLM adapter (ollama/local)
- `server/schemas.py` — ChatRequest / ChatResponse schema
- `server/main.py` — `/chat` route
- `src/components/views/ChatbotView.tsx` — frontend UI
- `src/lib/ai.ts` — frontend API call
