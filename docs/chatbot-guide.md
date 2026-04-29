# Chatbot Guide - Knowledge Graph Extractor

Tai lieu nay mo ta chi tiet chatbot trong project: kien truc, luong xu ly, cau hinh, API, va cach mo rong.

---

## 1. Tong quan

Chatbot la thanh phan hoi dap tu dong dua tren du lieu do thi tri thuc duoc trich xuat tu van ban/PDF.

He thong hoat dong theo mo hinh **hybrid**:
- **LLM mode**: neu da cau hinh `LLM_PROVIDER` + `LLM_MODEL`.
- **Rule-based mode**: fallback tu dong khi LLM khong cau hinh hoac goi LLM loi.

Chatbot ho tro:
- Hoi dap tren entities/relations hien co.
- Truy van bo sung tu Knowledge Base.
- RAG retrieval de tang do lien quan cua context.
- Luu lich su hoi thoai bang PostgreSQL theo `session_id`.

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
   |--- chat_memory.ensure_session()
   |--- chat_memory.add_message(user)
   |--- _build_graph_context(...)
   |--- rag.retrieve_context(...)
   |
   |--- if llm_client.is_configured():
   |        llm_client.generate(...)
   |        engine = "llm"
   |    else:
   |        _rule_based_reply(...)
   |        engine = "rule-based"
   |
   |--- chat_memory.add_message(model)
   |--- chat_memory.get_recent_messages()
   v
ChatResponse {session_id, reply, engine, history}
```

---

## 3. Cac module lien quan

### `server/chat_service.py`
Trung tam dieu phoi chatbot:
- Khoi tao session.
- Ghi/Doc lich su hoi thoai.
- Tao graph context cho LLM.
- Goi RAG pipeline.
- Chon engine (LLM hoac rule-based).
- Tra ve `ChatResponse`.

### `server/rag.py`
RAG pipeline su dung BM25 (`rank-bm25`):
- Cat `input_text` thanh chunks (`CHUNK_SIZE=300`, overlap `80`).
- Tao document tu entities.
- Tao document tu relations.
- Nap bo sung KB triples.
- Build cache index theo content hash.
- Retrieve top-K context (`TOP_K=12`, max context chars `3000`).

### `server/chat_memory.py`
Luu bo nho hoi thoai tren PostgreSQL:
- `chat_sessions`: thong tin session.
- `chat_messages`: tung luot chat.
- Dung `psycopg_pool` de quan ly ket noi.
- Tu dong apply schema `server/sql/chat_memory.sql` khi init.

### `server/llm_client.py`
Adapter goi LLM provider:
- `openai`
- `gemini`
- Neu khong cau hinh hop le -> `is_configured() == False`.

---

## 4. Luong xu ly 1 request chat

1. Frontend gui `POST /chat` kem:
   - `session_id`
   - `message`
   - `entities`, `relations`
   - `input_text`
2. Backend dam bao session ton tai (`ensure_session`).
3. Luu user message vao DB.
4. Tao graph context + retrieve context tu RAG.
5. Thu goi LLM neu da cau hinh.
6. Neu LLM loi/khong co cau hinh -> fallback rule-based.
7. Luu model response vao DB.
8. Lay history gan nhat (toi da 20 turns de tra ve frontend).
9. Tra `ChatResponse`.

---

## 5. Rule-based intents

Rule-based engine ho tro nhieu intent cho TV/EN (17+):
- Chao hoi
- Help/capabilities
- Tom tat do thi
- Tim quan he giua A va B
- So sanh A va B
- Lay neighbors cua 1 entity
- Top hub node (most connected)
- Liet ke quan he du doan
- Loc theo loai quan he
- Trich doan tu van ban goc
- Dem/thong ke entities-relations
- Liet ke theo entity type
- Liet ke tat ca relations
- Tra cuu Knowledge Base
- Fuzzy entity lookup
- Multi-entity matching
- RAG fallback + smart fallback

### Fuzzy matching
Thu tu uu tien:
1. Exact match
2. Substring match
3. Bo dau tieng Viet de so khop
4. SequenceMatcher (threshold ~= 0.50)

---

## 6. API `POST /chat`

### Request

```json
{
  "session_id": null,
  "message": "Moi quan he giua Vingroup va VinFast la gi?",
  "entities": [],
  "relations": [],
  "input_text": "Vingroup dau tu vao VinFast tai Hai Phong."
}
```

### Response

```json
{
  "session_id": "a1b2c3d4e5f6g7h8",
  "reply": "## Quan he truc tiep ...",
  "engine": "rule-based",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "model", "content": "..." }
  ]
}
```

### Gioi han theo schema
- `message`: toi da 4000 ky tu.
- `input_text`: toi da 50000 ky tu.
- `entities`: toi da 500.
- `relations`: toi da 5000.

---

## 7. Cau hinh bien moi truong

Dat trong file `.env` tai root project.

### Bat buoc cho memory

```env
DATABASE_URL="postgresql://user:password@localhost:5432/kge"
```

### Tuy chon cho LLM cloud (nhu truoc)

```env
LLM_PROVIDER="openai"        # hoac "gemini"
LLM_API_KEY="your-api-key"
LLM_MODEL=""                 # de trong = dung model mac dinh
```

Bien bo sung:
- `LLM_BASE_URL` (tuy chon, cho OpenAI-compatible endpoint)

---

## 8. Session memory va persistence

- Neu frontend gui `session_id` hop le: tiep tuc hoi thoai cu.
- Neu `session_id` null/khong ton tai: tao session moi.
- Tat ca messages duoc luu vao `chat_messages`.
- Frontend co the luu `session_id` trong localStorage de multi-turn.
- Khi DB khong san sang, chatbot van tra loi (degraded mode), nhung co the khong luu duoc lich su.

---

## 9. RAG context format

Context gui cho LLM duoc tong hop theo cac section:
- Source Text Excerpts
- Entity Information
- Graph Relations
- Knowledge Base

Ngoai ra, `chat_service` con bo sung graph-level insights:
- Top hub nodes
- Degree cua entities
- Danh dau entity duoc nhac truc tiep trong query
- Canh bao relation la predicted (lower confidence)

---

## 10. Van de thuong gap

### Khong luu duoc chat history
- Kiem tra `DATABASE_URL`.
- Kiem tra PostgreSQL dang chay.
- Dam bao schema da duoc tao (`chat_memory.sql`).

### Chatbot luon o rule-based mode
- Kiem tra `LLM_PROVIDER` la `openai` hoac `gemini`.
- Kiem tra `LLM_API_KEY` da duoc set dung.
- Kiem tra logs backend de xem loi goi provider.

### Phan hoi chua sat ngu canh
- Tang chat luong entities/relations tu buoc extract.
- Dam bao `input_text` day du.
- Dieu chinh `TOP_K`, `MAX_CONTEXT_CHARS` trong `rag.py`.

---

## 11. Goi y mo rong

- Them provider moi (Claude, Ollama...) vao `llm_client.py`.
- Them stream response (SSE/WebSocket) cho tra loi real-time token-by-token.
- Them reranking sau BM25 de cai thien retrieval.
- Them danh gia offline (intent accuracy, answer faithfulness, latency).
- Them bo loc an toan output (safety + PII masking) truoc khi tra frontend.

---

## 12. File tham khao nhanh

- `server/chat_service.py`
- `server/rag.py`
- `server/chat_memory.py`
- `server/llm_client.py`
- `server/schemas.py`
- `server/main.py` (`/chat` route)
- `src/components/views/ChatbotView.tsx`
- `src/lib/ai.ts`

