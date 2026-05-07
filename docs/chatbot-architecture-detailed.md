# Chatbot Architecture (Detailed)

Tai lieu nay mo ta chi tiet chatbot hien tai trong project, tap trung vao luong xu ly thuc te dang chay trong code.

## 1) Muc tieu chatbot

- Tra loi cau hoi dua tren du lieu co san trong he thong (Knowledge Graph + Input text + Knowledge Base + Insight/Metrics).
- Giam hallucination bang retrieval + context filter + answer validation.
- Ho tro hoi dap da luot (follow-up) voi session memory.
- Hoat dong on dinh tren local voi Ollama, va co fallback an toan sang rule-based.

## 2) Kien truc tong the

```mermaid
flowchart TD
  userQ[User Query] --> parseQ[Query Understanding]
  parseQ --> route[Mode Routing]

  route -->|deterministic| ruleB[Rule-based Engine]
  route -->|graph_first / hybrid| graphR[Graph Retriever]
  route -->|graph_first / hybrid| bm25R[BM25 Retriever]
  graphR --> fusion[Hybrid Fusion]
  bm25R --> fusion

  fusion --> cFilter[Context Filter]
  cFilter --> llm[Ollama / Local LLM]
  llm --> cont[Truncation Continuation?]
  cont --> validator[Answer Validator]

  validator -->|pass| okResp[Chat Response]
  validator -->|low grounding| noData[No-data Response]
  ruleB --> okResp

  userQ --> memRead[Load Session History]
  okResp --> memWrite[Save Session History]
  noData --> memWrite
```

## 3) Cac thanh phan chinh

### 3.1 `server/query_understanding.py`

Lop parse query dau vao truoc retrieval:

- **Intent** (15 loai):
  `relationship`, `count`, `summary`, `entity_lookup`, `compare`,
  `kb_lookup`, `help`, `greeting`, `relation_list`, `neighbors`,
  `top_nodes`, `predicted`, `source_text`, `type_list`, `unknown`
- **Mode**:
  - `deterministic`: route rule-based truc tiep, khong goi LLM.
  - `graph_first`: uu tien graph retrieval.
  - `hybrid`: can bang graph + BM25.
- Detect entity mention: exact → substring → bo dau tieng Viet → fuzzy (SequenceMatcher).
- Detect follow-up: pronoun + short-query + history check.
- Resolve entity tu history khi query follow-up mo ho.

### 3.2 `server/graph_retriever.py`

Hybrid retrieval layer:

- **Graph-first docs** (nguon chinh):
  - `graph_entity` — entity card + quan he lien quan.
  - `graph_relation` — triple truc tiep giua entities.
  - `graph_path` — 2-hop path cho relationship query.
  - `kb_triple` — triple bo sung tu KB.
- **BM25 docs** (nguon `rag.py`):
  - `input_text`, `entity`, `relation`, `kb_triple`, `insight`, `metrics`.
- **Fusion strategy**:
  - `graph_first`: graph docs truoc, BM25 bo sung.
  - `hybrid`: interleave graph va BM25 theo rank.
- Nhan them `insight_markdown` va `metrics_summary` de index them nguon Insight/Metrics.

### 3.3 `server/context_filter.py`

Loc context truoc khi gui LLM:

1. Tinh overlap theo entity mention trong query.
2. Boost score theo source type (`graph_relation` / `graph_path` uu tien cao nhat).
3. Loai chunk gan-trung (SequenceMatcher dedup).
4. Cat theo token budget (` graph_first` < `hybrid`).

Muc dich: giam nhieu context, tang grounding, tranh prompt qua dai.

### 3.4 `server/llm_client.py`

LLM adapter:

- Provider ho tro:
  - `ollama` (khuyen nghi): goi `POST /api/chat` den Ollama server.
  - `local`: Qwen2.5 + LoRA adapter trong thu muc model.
- Cau hinh Ollama qua env: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`.
- Mac dinh: `OLLAMA_BASE_URL=http://localhost:11434`, `OLLAMA_MODEL=qwen2.5:3b`, `LLM_MAX_NEW_TOKENS=512`, `LLM_TEMPERATURE=0.2`.
- Loi ket noi / timeout / HTTP error se throw `RuntimeError` de `chat_service` fallback sang rule-based.
- `is_configured()`:
  - `ollama`: luon True (kiem tra ket noi thuc su o `generate` time).
  - `local`: kiem tra adapter path hop le.
  - khac: False → rule-based.

### 3.5 `server/answer_validator.py`

Guardrail sau LLM:

- Kiem tra no-data phrase (`khong du du lieu`, `no information`, ...).
- Trich xuat claims tu markdown pattern (`**entity**`, `[entity]`, `--(relation)-->`).
- Doi chieu claims voi retrieved context.
- Tinh:
  - `confidence` (0.0–1.0)
  - `evidence_count`
  - `unsupported_claims`
  - `speculation_flags` (detect cac cum tu suy doan: "toi nghi", "probably", ...)
- Neu `low_grounding` va `evidence_count == 0` → thay bang `NO_DATA_REPLY` chuan.

### 3.6 `server/chat_service.py`

Orchestration trung tam:

1. Ensure session + load history truoc khi add message moi.
2. `parse_query()` → `ParsedQuery` (intent, mode, entities, followup).
3. Neu `deterministic` intent hoac `!llm_client.is_configured()` → rule-based truc tiep.
4. Neu LLM path:
   - `hybrid_retrieve()` (truyen them `insight_markdown`, `metrics_summary`).
   - `filter_context()`.
   - `format_context_for_llm()`.
   - `_call_llm_with_context()`:
     - Build user message: Question + KG + Context + Input Text + Insight + Metrics.
     - Gui `history[-20:-1]` lam conversation context cho follow-up.
     - Neu `_looks_truncated_reply()` → goi them 1 luot continuation.
   - `validate_answer()`.
   - Neu `should_replace_with_no_data()` → thay `NO_DATA_REPLY`.
5. Neu LLM loi (exception) → fallback rule-based.
6. `_normalize_reply_text()`: clean up `\n`, `\t` escaped chars tu LLM output.
7. Save response vao memory.
8. Return `ChatResponse`.

**Structured logging** theo `request_id` cho:
- intent / mode / entities_mentioned / is_followup
- retrieve timing va so docs scored/filtered/context_chars
- LLM timing, validation result, confidence, evidence_count
- Tong latency

**`_PREFER_RULE_FOR_KEYWORDS`** (env `CHAT_PREFER_RULE_FOR_KEYWORDS=true`):
- Routing keyword-heavy query (count, summary, compare, top, help...) sang rule-based ngay ca khi LLM co san, via `_should_prefer_rule_based()`. Hien tai logic nay duoc `query_understanding.py` xu ly thong qua `MODE_DETERMINISTIC`.

### 3.7 One-shot Continuation (`_looks_truncated_reply`)

Phat hien cau tra loi bi cat giua chung:
- Ket thuc bang `:`, `-`, `•`, `,`, `**`, `` ` ``, `(`, `[`, `{`.
- Ket thuc bang cum tu nhu "thuc the", "ket luan", "tong ket".
- Response dai > 250 ky tu nhung khong co dau cau ket thuc.

Neu phat hien truncation → goi them 1 LLM request voi prompt `"Tiep tuc phan tra loi con dang do..."` va noi ket qua.

### 3.8 `_normalize_reply_text`

Chuan hoa output LLM truoc khi luu va tra ve:
- Chuan hoa CRLF → LF.
- Convert escaped `\n` → newline thuc, `\t` → tab (mot so model emit escaped chars).
- Trim trailing spaces tung dong, giu markdown line breaks.

## 4) Chat API contract hien tai

`POST /chat`

### Request

```json
{
  "session_id": null,
  "message": "string (max 4000)",
  "entities": [],
  "relations": [],
  "input_text": "string (max 50000)",
  "insight_markdown": "string (max 200000, optional)",
  "metrics_summary": "string (max 80000, optional)"
}
```

### Response (`schemas.ChatResponse`)

```json
{
  "session_id": "string",
  "reply": "markdown string",
  "engine": "ollama | local | rule-based",
  "history": [{"role": "user|model", "content": "..."}],
  "confidence": 0.82,
  "evidence_count": 3,
  "intent": "relationship"
}
```

| Truong | Mo ta |
|---|---|
| `engine` | Provider LLM da dung hoac `rule-based` |
| `confidence` | Diem grounding heuristic 0.0–1.0 |
| `evidence_count` | So context chunks ho tro cau tra loi |
| `intent` | Intent duoc parse (visible cho frontend/debug) |

## 5) Memory va follow-up

- Memory luu qua `chat_memory` module (PostgreSQL, `psycopg_pool`).
- History duoc load TRUOC khi add user message moi (can cho follow-up detection trong `parse_query`).
- Follow-up query co the thua huong entity tu cac turn user gan nhat.
- LLM nhan toi da 10 turns lich su (`_MAX_HISTORY_FOR_LLM=10`) lam conversation context.
- Neu DB khong kha dung: chat van chay o che do `ephemeral` (khong luu lich su).

## 6) Cac che do tra loi

- **Deterministic mode** (uu tien toc do + on dinh):
  - count, top, help, greeting, relation_list, source_text, predicted, type_list, kb_lookup.
  - Route truc tiep rule-based, khong goi LLM.
- **Graph-first mode**:
  - relationship, entity_lookup, compare, neighbors.
  - Graph retrieval la nguon chinh, LLM tong hop.
- **Hybrid mode**:
  - summary, unknown.
  - Ket hop graph + BM25 de mo rong context cho LLM.

## 7) Fallback va anti-hallucination

- **LLM loi** (timeout / connect / HTTP): fallback sang rule-based, engine = `"rule-based"`.
- **Low grounding**: answer khong co evidence trong context → thay no-data response:
  - `"Khong du du lieu trong he thong de tra loi cau hoi nay. Hay thu hoi cu the hon..."`
- **Speculation flags**: log cac cum tu suy doan, hien thi trong debug.
- Tat ca result luon co `engine`, `confidence`, `evidence_count`, `intent` de frontend/noi bo giam sat chat luong.

## 8) Cau hinh env khuyen nghi (local Ollama)

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT=60
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.2
CHAT_PREFER_RULE_FOR_KEYWORDS=true
DATABASE_URL=postgresql://user:password@localhost:5432/kge
```

## 9) Van hanh nhanh

1. Start Ollama:
   - `ollama serve`
2. Pull model:
   - `ollama pull qwen2.5:3b`
3. Start backend:
   - `npm run server` (hoac `py server/main.py`)
4. Health check:
   - `GET /health`
5. Test chat:
   - `POST /chat` voi graph context.

## 10) Gioi han hien tai va huong mo rong

### Gioi han
- Confidence la heuristic grounding, chua phai factuality metric chuan.
- Validator grounding hien o muc lexical/pattern, chua co NLI hoac claim verifier nang cao.
- Continuation chi 1 luot (one-shot), chua co multi-step continuation.
- Dedup context dung SequenceMatcher, co the bo sot duplicate semantic phuc tap.

### Mo rong tiep theo (neu can)
- Re-ranker cho retrieval sau BM25 (cross-encoder, ColBERT...).
- Validator claim-level nang cao (NLI / symbolic check).
- Response citation (`evidence_ids`) tra ve frontend.
- Dashboard debug trace cho tung request: `query → retrieved → filtered → answer`.
- Stream response (SSE / WebSocket) cho tra loi token-by-token.
- Bo loc an toan output (safety + PII masking) truoc khi tra frontend.
