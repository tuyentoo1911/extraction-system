"""
Hybrid chat service: memory + graph context + KB enrichment + LLM / rule-based fallback.

Rule-based engine supports 15+ intent types (Vietnamese + English):
  - Entity lookup (fuzzy)       - Count / statistics
  - Type listing                - Relation listing & filtering
  - Relationship path (A→B)     - Neighbors / connections
  - Comparison (A vs B)         - Top / most-connected nodes
  - Graph summary / overview    - KB deep lookup
  - Relation type search        - Predicted links info
  - Help / capabilities         - Greeting
  - Source text excerpt
"""

from __future__ import annotations

import os
import re
import time
import uuid
import logging
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Optional

import chat_memory as mem
import llm_client
import knowledge_base as kb
import rag as rag_mod
from query_understanding import parse_query, ParsedQuery, MODE_DETERMINISTIC
from graph_retriever import hybrid_retrieve
from context_filter import filter_context, format_context_for_llm
from answer_validator import validate_answer, should_replace_with_no_data, NO_DATA_REPLY
from schemas import (
    ChatRequest,
    ChatResponse,
    ChatTurn,
    Entity,
    Relation,
)

logger = logging.getLogger(__name__)

_MAX_CONTEXT_ENTITIES = 30
_MAX_CONTEXT_RELATIONS = 60
_MAX_HISTORY_FOR_LLM = 10
_PREFER_RULE_FOR_KEYWORDS = (
    (os.getenv("CHAT_PREFER_RULE_FOR_KEYWORDS", "true")).strip().lower()
    in {"1", "true", "yes", "y", "on"}
)

# Engine tag shown in response — derived from LLM_PROVIDER env at runtime
def _get_engine_tag() -> str:
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    return provider if provider in ("ollama", "local") else "llm"

_PROVIDER_ENGINE_TAG = _get_engine_tag()

# System prompt khớp với định dạng model đã được fine-tune
_SYSTEM_PROMPT = """\
Bạn là AI Chatbot của hệ thống Knowledge Graph Extractor.

Bạn có quyền truy cập vào:
1. Knowledge Graph (thực thể và quan hệ)
2. Knowledge Base (các triple)
3. Văn bản gốc (input text)
4. Ngữ cảnh truy xuất (RAG context — có thể gồm Insight / Metrics)
5. Báo cáo Insight và tóm tắt Metrics (nếu được đính kèm)

NHIỆM VỤ:
- Trả lời câu hỏi của người dùng dựa hoàn toàn trên dữ liệu được cung cấp trong context.
- Phân tích mối quan hệ giữa các thực thể nếu có.
- Đưa ra câu trả lời rõ ràng, chính xác, có cấu trúc.

QUY TẮC QUAN TRỌNG:
- KHÔNG được bịa thông tin.
- KHÔNG suy đoán ngoài dữ liệu.
- Nếu không có thông tin -> trả lời: "Không tìm thấy thông tin trong dữ liệu."
- Ưu tiên: (1) Quan hệ trực tiếp → (2) Quan hệ gián tiếp → (3) Knowledge Base → (4) Insight/Metrics → (5) Văn bản gốc

CÁCH TRẢ LỜI:
- Nếu là quan hệ: [Thực thể A] --(quan hệ)--> [Thực thể B]
- Nếu là liệt kê: dùng bullet points
- Nếu là so sánh: trình bày rõ từng tiêu chí
- Nếu là phân tích: giải thích ngắn gọn + kết luận
- Luôn trả lời bằng ngôn ngữ của người dùng (Việt/Anh)
- Không tự ý chèn từ/câu tiếng Trung, Nhật, Hàn nếu người dùng không yêu cầu.
- Chỉ trả lời đúng trọng tâm câu hỏi người dùng; không tự thêm câu hỏi ngược/gợi ý nếu không được yêu cầu.
- Trả lời ngắn gọn, tối đa 5-8 dòng nếu không cần thiết dài hơn.

MỤC TIÊU:
Trả lời chính xác câu hỏi người dùng bằng cách phân tích dữ liệu từ Knowledge Graph.

NĂNG LỰC BẮT BUỘC:
- Hiểu câu hỏi tiếng Việt tự nhiên, kể cả câu ngắn, thiếu chủ ngữ, hoặc có lỗi chính tả nhẹ.
- Giữ ngữ cảnh hội thoại giữa nhiều lượt hỏi đáp để trả lời nhất quán.
- Hỗ trợ cả câu hỏi trực tiếp, câu hỏi suy luận, câu hỏi nhiều bước và diễn đạt tự do.
- Kết hợp dữ liệu từ nhiều nguồn (KG + KB + insight + metrics + input text) để trả lời đầy đủ.
- Trả lời tự nhiên, thân thiện, linh hoạt như giao tiếp thực tế, nhưng vẫn chính xác dữ liệu.

=====================
NGUYÊN TẮC QUAN TRỌNG
=====================

1. CHỈ dùng dữ liệu trong context
- Không suy đoán
- Không bịa thêm
- Nếu thiếu dữ liệu → nói rõ: "Không tìm thấy dữ liệu phù hợp trong hệ thống"

2. Ưu tiên thông tin theo thứ tự:
(1) Quan hệ trực tiếp trong Knowledge Graph
(2) Quan hệ gián tiếp (2-hop qua node trung gian)
(3) Knowledge Base triple (triple)
(4) Văn bản gốc (RAG excerpt)

3. Khi phân tích graph:
- Xác định thực thể chính (được đánh dấu ← [MENTIONED IN QUERY])
- Ưu tiên hub nodes (node có degree cao, nhiều kết nối)
- Nếu có nhiều quan hệ → chọn quan hệ trực tiếp liên quan nhất
- Xem phần "Graph Insights" trong context để hiểu cấu trúc đồ thị

4. Khi không có dữ liệu:
- Nói rõ: "Không tìm thấy dữ liệu phù hợp trong hệ thống"
- Gợi ý 1-2 câu hỏi khác có thể trả lời được

=====================
CHỐNG HALLUCINATION (BẮT BUỘC)
=====================

TUYỆT ĐỐI KHÔNG:
- Tạo ra quan hệ không có trong Knowledge Graph context
- Đặt tên thực thể không xuất hiện trong danh sách entities
- Suy diễn từ kiến thức chung (world knowledge) khi context không hỗ trợ
- Khẳng định chắc chắn điều gì đó nếu chỉ đến từ quan hệ [PREDICTED]

Khi dùng quan hệ [PREDICTED — lower confidence]:
→ PHẢI ghi rõ: "đây là dự đoán, chưa được xác nhận từ văn bản gốc"

Khi dùng triple từ Knowledge Base (KB):
→ Nên ghi rõ: "theo Knowledge Base"

=====================
CÁCH TRẢ LỜI
=====================

- Ngắn gọn, rõ ràng, có cấu trúc
- Ưu tiên insight, không chỉ liệt kê
- KHÔNG viết mở đầu thừa như "Dựa trên context được cung cấp..." hay "Theo thông tin tôi có..."

Nếu là phân tích:
→ Giải thích + ý nghĩa + kết luận (tối đa 200 từ)

Nếu là liệt kê:
→ Bullet points ngắn gọn

Nếu là quan hệ:
→ Dùng format: [Thực thể A] --(quan hệ)--> [Thực thể B]

Nếu là so sánh:
→ Bảng hoặc đối chiếu rõ ràng

=====================
KIỂM SOÁT ĐỘ DÀI
=====================

- Câu hỏi đơn giản (lookup, count, tìm tên): 1-3 câu
- Câu hỏi vừa (quan hệ, neighbors): 3-8 dòng
- Câu hỏi phức tạp (summary, compare, phân tích): tối đa 200 từ
- KHÔNG lặp lại thông tin đã nêu
- KHÔNG thêm phần kết luận thừa nếu đã rõ ràng

=====================
XỬ LÝ CÂU HỎI
=====================

Bước 1: Xác định intent
  summary / relationship / compare / ranking / count / lookup / kb-search

Bước 2: Xác định entity (dùng danh sách entities trong context)

Bước 3: Tìm dữ liệu trong Knowledge Graph context

Bước 4: Suy luận từ graph nếu cần (2-hop, hub node)

Bước 5: Trả lời theo format phù hợp với intent

=====================
NGÔN NGỮ
=====================

- Trả lời theo ngôn ngữ của người dùng
- Nếu user dùng tiếng Việt → trả lời tiếng Việt
- Viết tự nhiên, dễ hiểu, không dịch thuật cứng nhắc
"""

def init_chat_db() -> None:
    """Called once at server startup."""
    try:
        mem.init_pool()
    except Exception:
        logger.warning("Chat memory DB not available — chat will still work without persistence.", exc_info=True)

async def handle_chat(req: ChatRequest) -> ChatResponse:
    request_id = uuid.uuid4().hex[:8]
    t_start    = time.monotonic()

    # ── 1. Session management ──────────────────────────────────────────────────
    session_id = _safe_ensure_session(req.session_id)

    # Load history BEFORE adding current message (needed for follow-up detection)
    history = _safe_get_history(session_id, limit=20)
    _safe_add_message(session_id, "user", req.message)

    # ── 2. Query Understanding ─────────────────────────────────────────────────
    pq = parse_query(req.message, req.entities, req.relations, history)
    logger.info(
        "[%s] intent=%s mode=%s entities=%s followup=%s",
        request_id, pq.intent, pq.mode, pq.entities_mentioned, pq.is_followup,
    )

    engine        = "rule-based"
    reply: Optional[str] = None
    confidence    = 0.0
    evidence_count = 0

    # ── 3. Deterministic intents → rule-based directly (no LLM needed) ────────
    if pq.is_deterministic() or not llm_client.is_configured():
        reply  = _rule_based_reply(
            req.message, req.entities, req.relations, req.input_text,
            insight_markdown=req.insight_markdown or "",
            metrics_summary=req.metrics_summary or "",
        )
        engine = "rule-based"
        # Rule-based answers are grounded by definition
        confidence     = 0.9 if reply and len(reply) > 20 else 0.5
        evidence_count = len(pq.entities_mentioned)

    # ── 4. LLM path: hybrid retrieve → filter → LLM → validate ───────────────
    else:
        t_retrieve = time.monotonic()
        scored_docs  = hybrid_retrieve(
            pq, req.input_text, req.entities, req.relations,
            insight_markdown=req.insight_markdown or "",
            metrics_summary=req.metrics_summary or "",
        )
        filtered_docs = filter_context(scored_docs, pq)
        context_str  = format_context_for_llm(filtered_docs)
        t_retrieve_ms = int((time.monotonic() - t_retrieve) * 1000)

        logger.info(
            "[%s] retrieve=%dms scored=%d filtered=%d context_chars=%d",
            request_id, t_retrieve_ms, len(scored_docs), len(filtered_docs), len(context_str),
        )

        try:
            t_llm = time.monotonic()
            reply = await _call_llm_with_context(
                history=history,
                question=req.message,
                entities=req.entities,
                relations=req.relations,
                input_text=req.input_text,
                context_str=context_str,
                insight_markdown=req.insight_markdown or "",
                metrics_summary=req.metrics_summary or "",
            )
            t_llm_ms = int((time.monotonic() - t_llm) * 1000)
            engine   = _PROVIDER_ENGINE_TAG

            # ── 5. Answer validation ──────────────────────────────────────────
            v_result = validate_answer(reply, filtered_docs, pq.entities_mentioned)
            confidence     = v_result.confidence
            evidence_count = v_result.evidence_count

            logger.info(
                "[%s] llm=%dms engine=%s valid=%s confidence=%.2f evidence=%d reason=%s",
                request_id, t_llm_ms, engine,
                v_result.passed, confidence, evidence_count, v_result.reason,
            )

            if v_result.speculation_flags:
                logger.debug("[%s] speculation flags: %s", request_id, v_result.speculation_flags)

            if should_replace_with_no_data(v_result):
                logger.warning(
                    "[%s] Answer rejected (low_grounding, evidence=0) → no-data reply",
                    request_id,
                )
                reply          = NO_DATA_REPLY
                confidence     = 0.1
                evidence_count = 0

        except Exception as exc:
            logger.warning("[%s] LLM failed, falling back to rule-based: %s", request_id, exc)

    # ── 6. Fallback to rule-based if LLM path produced nothing ────────────────
    if reply is None:
        reply          = _rule_based_reply(
            req.message, req.entities, req.relations, req.input_text,
            insight_markdown=req.insight_markdown or "",
            metrics_summary=req.metrics_summary or "",
        )
        engine         = "rule-based"
        confidence     = 0.9 if reply and len(reply) > 20 else 0.5
        evidence_count = len(pq.entities_mentioned)

    # ── 7. Persist & return ───────────────────────────────────────────────────
    target_lang = _detect_user_language(req.message)
    reply = _normalize_reply_text_for_language(
        reply,
        target_lang=target_lang,
        user_message=req.message,
    )
    _safe_add_message(session_id, "model", reply)

    recent = _safe_get_history(session_id, limit=20)
    turns  = [ChatTurn(role=r["role"], content=r["content"]) for r in recent]

    t_total_ms = int((time.monotonic() - t_start) * 1000)
    logger.info(
        "[%s] DONE total=%dms engine=%s confidence=%.2f",
        request_id, t_total_ms, engine, confidence,
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        engine=engine,
        history=turns,
        confidence=round(confidence, 3),
        evidence_count=evidence_count,
        intent=pq.intent,
    )

def _should_prefer_rule_based(user_message: str) -> bool:
    """Route keyword-heavy intents to deterministic rule-based engine."""
    if not _PREFER_RULE_FOR_KEYWORDS:
        return False

    q = user_message.lower().strip()
    if len(q) <= 2:
        return False
    q_ascii = _strip_diacritics(q)

    keyword_hints = {
        "bao nhiêu", "count", "thống kê", "statistics", "stats",
        "tóm tắt", "tổng quan", "summary", "overview",
        "so sánh", "compare", "vs",
        "mối quan hệ", "relationship", "liên quan", "kết nối", "neighbors",
        "top", "quan trọng nhất", "most connected",
        "liệt kê", "list", "danh sách",
        "kb", "knowledge base", "tra cứu",
        "văn bản gốc", "source text", "trích đoạn",
        "giúp", "help",
    }
    keyword_hints_ascii = {
        "bao nhieu", "thong ke", "tom tat", "tong quan", "so sanh",
        "moi quan he", "lien quan", "ket noi", "liet ke", "danh sach",
        "van ban goc", "trich doan", "giup", "huong dan", "tro giup",
        "hoi gi", "hoi duoc gi", "biet gi", "lam gi", "chuc nang",
        "ban co the lam duoc gi", "ban lam duoc gi",
    }
    return (
        any(k in q for k in keyword_hints)
        or any(k in q_ascii for k in keyword_hints_ascii)
    )


def _normalize_reply_text(text: str) -> str:
    """
    Normalize escaped formatting tokens from LLM outputs.
    Example: "\\n" -> newline so markdown renders correctly.
    """
    return _normalize_reply_text_for_language(text, target_lang="vi")


def _detect_user_language(user_message: str) -> str:
    """
    Lightweight language detector for reply formatting.
    Returns: 'vi' | 'en'
    """
    q = (user_message or "").lower()
    vi_markers = {
        "không", "bao nhiêu", "tóm tắt", "quan hệ", "liệt kê", "giúp", "với", "được", "thế nào",
    }
    if any(m in q for m in vi_markers):
        return "vi"
    # Presence of Vietnamese diacritics is a strong signal.
    if re.search(r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", q):
        return "vi"
    return "en"


def _strip_unexpected_cjk(text: str) -> str:
    """
    Remove accidental CJK fragments from model output when the user did not ask for them.
    """
    if not text:
        return text
    cjk_re = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        # Drop line only when it contains CJK and has little/no Latin content.
        if cjk_re.search(line):
            latin_count = len(re.findall(r"[A-Za-zÀ-ỹ]", line))
            cjk_count = len(cjk_re.findall(line))
            if cjk_count >= max(2, latin_count):
                continue
            line = cjk_re.sub("", line)
            line = re.sub(r"\s{2,}", " ", line).strip()
            if not line:
                continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _wants_suggestions(user_message: str) -> bool:
    q = (user_message or "").lower()
    suggestion_triggers = {
        "gợi ý", "goi y", "đề xuất", "de xuat", "help", "giúp", "huong dan", "hướng dẫn",
        "what can", "suggest", "ví dụ", "vi du",
    }
    return any(t in q for t in suggestion_triggers)


def _trim_unsolicited_followups(reply: str, user_message: str) -> str:
    """
    Remove trailing meta-prompts like "Bạn có câu hỏi nào khác...?"
    unless user explicitly asks for suggestions/help.
    """
    if not reply:
        return reply
    if _wants_suggestions(user_message):
        return reply

    lines = [ln.rstrip() for ln in reply.split("\n")]
    trimmed: list[str] = []
    stop_markers = [
        r"^bạn có câu hỏi.*\?$",
        r"^nếu bạn muốn biết thêm.*$",
        r"^hãy hỏi.*cụ thể.*$",
        r"^bạn có thể hỏi.*$",
        r"^you can ask.*$",
        r"^would you like.*\?$",
        r"^if you want.*$",
    ]
    stop_re = [re.compile(p, re.IGNORECASE) for p in stop_markers]

    for ln in lines:
        normalized = ln.strip()
        if any(rx.match(normalized) for rx in stop_re):
            break
        trimmed.append(ln)

    # Remove trailing empty lines.
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return "\n".join(trimmed).strip() or reply.strip()


def _collapse_repeated_content(text: str) -> str:
    """
    Collapse duplicated lines/paragraphs that sometimes appear in unstable LLM outputs.
    """
    if not text:
        return text

    # 1) Remove consecutive duplicate lines.
    raw_lines = [ln.rstrip() for ln in text.split("\n")]
    dedup_lines: list[str] = []
    prev_norm = ""
    for ln in raw_lines:
        norm = re.sub(r"\s+", " ", ln.strip().lower())
        if norm and norm == prev_norm:
            continue
        dedup_lines.append(ln)
        prev_norm = norm
    cleaned = "\n".join(dedup_lines).strip()

    # 2) Remove repeated paragraphs (same normalized paragraph appears many times).
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        return cleaned

    seen: set[str] = set()
    unique_paragraphs: list[str] = []
    for p in paragraphs:
        p_norm = re.sub(r"\s+", " ", p.lower())
        if p_norm in seen:
            continue
        seen.add(p_norm)
        unique_paragraphs.append(p)
    return "\n\n".join(unique_paragraphs).strip()


def _is_low_quality_continuation(base: str, cont: str) -> bool:
    """
    Detect continuation chunks that mostly repeat/noise and should be discarded.
    """
    if not cont.strip():
        return True
    base_norm = re.sub(r"\s+", " ", base.lower())
    cont_norm = re.sub(r"\s+", " ", cont.lower())
    # Mostly duplicated from first part.
    if cont_norm in base_norm:
        return True
    # Too many CJK chars compared to Latin/Vietnamese letters for this project.
    cjk_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", cont))
    latin_count = len(re.findall(r"[A-Za-zÀ-ỹ]", cont))
    if cjk_count > 0 and cjk_count >= max(3, latin_count):
        return True
    return False


def _normalize_reply_text_for_language(
    text: str,
    target_lang: str = "vi",
    user_message: str = "",
) -> str:
    if not text:
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Convert common escaped control chars emitted by some models.
    normalized = normalized.replace("\\n", "\n").replace("\\t", "\t")

    if target_lang in {"vi", "en"}:
        normalized = _strip_unexpected_cjk(normalized)
    normalized = _collapse_repeated_content(normalized)
    normalized = _trim_unsolicited_followups(normalized, user_message)

    # Trim trailing spaces but keep meaningful line breaks for markdown.
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()

_memory_available = True

def _safe_ensure_session(session_id: Optional[str]) -> str:
    global _memory_available
    if not _memory_available:
        return session_id or "ephemeral"
    try:
        return mem.ensure_session(session_id)
    except Exception:
        logger.warning("Memory unavailable; using ephemeral session.", exc_info=True)
        _memory_available = False
        return session_id or "ephemeral"

def _safe_add_message(session_id: str, role: str, content: str) -> None:
    if not _memory_available:
        return
    try:
        mem.add_message(session_id, role, content)
    except Exception:
        logger.warning("Failed to persist chat message.", exc_info=True)

def _safe_get_history(session_id: str, limit: int = 50) -> list[dict]:
    if not _memory_available:
        return []
    try:
        return mem.get_recent_messages(session_id, limit=limit)
    except Exception:
        logger.warning("Failed to load chat history.", exc_info=True)
        return []


def _compact_kg(entities: list[Entity], relations: list[Relation]) -> str:
    """Build compact KG string in training format: [A] --(rel)--> [B]."""
    entity_map = {e.id: e.name for e in entities}
    lines: list[str] = []
    for r in relations[:_MAX_CONTEXT_RELATIONS]:
        src = entity_map.get(r.source, r.source)
        tgt = entity_map.get(r.target, r.target)
        pred_tag = " [dự đoán]" if r.isPredicted else ""
        lines.append(f"[{src}] --({r.label})--> [{tgt}]{pred_tag}")
    if not lines:
        entity_names = [e.name for e in entities[:20]]
        if entity_names:
            return "Entities: " + ", ".join(entity_names) + "\n(chưa có quan hệ)"
        return "(no data)"
    return "\n".join(lines)


def _compact_kb(entities: list[Entity]) -> str:
    """Fetch relevant KB triples for the given entities."""
    if not kb.kb_ready or not entities:
        return "(none)"
    lines: list[str] = []
    seen: set[str] = set()
    for e in entities[:10]:
        for t in kb.get_entity_triples(e.name, limit=5):
            row = f"{t['subject']} | {t['relation']} | {t['object']}"
            if row not in seen:
                seen.add(row)
                lines.append(row)
    return "\n".join(lines) if lines else "(none)"


def _build_structured_user_msg(
    question: str,
    entities: list[Entity],
    relations: list[Relation],
    rag_context: str,
    input_text: str,
) -> str:
    """Build the user message in the format the fine-tuned model was trained on."""
    kg_text = _compact_kg(entities, relations)
    kb_text = _compact_kb(entities)
    text_snippet = (input_text[:400] + "...") if len(input_text) > 400 else input_text
    rag_text = rag_context if rag_context and rag_context.strip() not in {
        "(No relevant context found.)", ""
    } else "(none)"
    return (
        f"Question:\n{question}\n\n"
        f"Knowledge Graph:\n{kg_text}\n\n"
        f"Knowledge Base:\n{kb_text}\n\n"
        f"Input Text:\n{text_snippet or '(none)'}\n\n"
        f"RAG Context:\n{rag_text}"
    )


async def _call_llm(
    history: list[dict],
    question: str,
    entities: list[Entity],
    relations: list[Relation],
    input_text: str = "",
    rag_context: str = "",
) -> str:
    """Legacy LLM caller — kept for backward compatibility."""
    structured_user_msg = _build_structured_user_msg(
        question, entities, relations, rag_context, input_text
    )
    msgs: list[dict[str, str]] = []
    past_history = history[-(_MAX_HISTORY_FOR_LLM * 2 + 1):-1]
    for row in past_history:
        role = "assistant" if row["role"] == "model" else row["role"]
        msgs.append({"role": role, "content": row["content"]})
    msgs.append({"role": "user", "content": structured_user_msg})
    return await llm_client.generate(_SYSTEM_PROMPT, msgs)


def _prompt_snippet(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0]
    return cut + "\n..."


def _build_history_context(history: list[dict], max_turns: int = 6, max_chars: int = 1200) -> str:
    """
    Build compact conversation context for continuity in follow-up questions.
    """
    if not history:
        return "(none)"
    # Use newest turns, but preserve chronological order.
    selected = history[-max_turns:]
    lines: list[str] = []
    total_chars = 0
    for row in selected:
        role = "Assistant" if row.get("role") == "model" else "User"
        content = (row.get("content") or "").strip()
        if not content:
            continue
        line = f"{role}: {content}"
        if total_chars + len(line) > max_chars:
            break
        lines.append(line)
        total_chars += len(line)
    return "\n".join(lines) if lines else "(none)"


async def _call_llm_with_context(
    history: list[dict],
    question: str,
    entities: list[Entity],
    relations: list[Relation],
    input_text: str = "",
    context_str: str = "",
    insight_markdown: str = "",
    metrics_summary: str = "",
) -> str:
    """
    New LLM caller using the pre-filtered hybrid context string.
    Passes recent history for follow-up continuity (up to _MAX_HISTORY_FOR_LLM turns).
    """
    kg_text      = _compact_kg(entities, relations)
    text_snippet = (input_text[:300] + "...") if len(input_text) > 300 else input_text

    insight_excerpt = _prompt_snippet(insight_markdown, 2800)
    metrics_excerpt = _prompt_snippet(metrics_summary, 2000)
    history_context = _build_history_context(history, max_turns=8, max_chars=1500)

    user_msg = (
        f"Question:\n{question}\n\n"
        f"Recent conversation context:\n{history_context}\n\n"
        f"Knowledge Graph:\n{kg_text}\n\n"
        f"Context:\n{context_str or '(none)'}\n\n"
        f"Input Text:\n{text_snippet or '(none)'}\n\n"
        f"Insight report (excerpt, optional):\n{insight_excerpt or '(none)'}\n\n"
        f"Metrics summary (optional):\n{metrics_excerpt or '(none)'}\n\n"
        "Response requirements:\n"
        "- Understand natural Vietnamese phrasing, including short and implicit follow-up questions.\n"
        "- If the question requires multi-step reasoning, reason over available graph/context evidence before concluding.\n"
        "- Keep answer natural and human-like, concise, and grounded in provided data.\n"
        "- If data is insufficient, state clearly that the system has no matching information."
    )

    msgs: list[dict[str, str]] = []
    # Include recent history for follow-up awareness (skip the current user turn at end)
    past_history = history[-(_MAX_HISTORY_FOR_LLM * 2 + 1):-1]
    for row in past_history:
        role = "assistant" if row["role"] == "model" else row["role"]
        msgs.append({"role": role, "content": row["content"]})
    msgs.append({"role": "user", "content": user_msg})

    ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT") or "60")
    first = await llm_client.generate(_SYSTEM_PROMPT, msgs, timeout=ollama_timeout)

    # One-shot continuation for truncated answers.
    if _looks_truncated_reply(first):
        continue_msgs = [
            *msgs,
            {"role": "assistant", "content": first},
            {
                "role": "user",
                "content": (
                    "Tiếp tục phần trả lời còn dang dở ngay từ chỗ trước đó. "
                    "Không lặp lại nội dung đã viết."
                ),
            },
        ]
        try:
            cont = await llm_client.generate(
                _SYSTEM_PROMPT,
                continue_msgs,
                timeout=ollama_timeout,
            )
            cont = (cont or "").strip()
            if cont and not _is_low_quality_continuation(first, cont):
                return f"{first.rstrip()}\n{cont}"
        except Exception:
            # Ignore continuation failure and keep first answer.
            pass

    return first


def _looks_truncated_reply(text: str) -> bool:
    """
    Heuristic detector for incomplete model replies.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < 80:
        return False
    if t.endswith((":", "-", "•", ",")):
        return True
    if t.endswith(("**", "`", "(", "[", "{")):
        return True
    if t.lower().endswith(("thực thể", "kết luận", "tổng kết")):
        return True
    # No sentence-ending punctuation in a long response often means cutoff.
    if len(t) > 250 and t[-1] not in (".", "!", "?", "\"", "”", "`"):
        return True
    return False

_FUZZY_THRESHOLD = 0.50

_VIET_DIACRITICS = str.maketrans(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
)

def _strip_diacritics(s: str) -> str:
    return s.translate(_VIET_DIACRITICS)

def _fuzzy_find_entity(
    query: str, entities: list[Entity],
) -> Optional[Entity]:
    """Find best matching entity using exact → substring → diacritics-stripped → fuzzy."""
    q = query.lower().strip()

    for e in entities:
        if e.name.lower() == q:
            return e

    for e in entities:
        if e.name.lower() in q or q in e.name.lower():
            return e

    q_ascii = _strip_diacritics(q)
    for e in entities:
        if _strip_diacritics(e.name.lower()) == q_ascii:
            return e
    for e in entities:
        if q_ascii in _strip_diacritics(e.name.lower()) or _strip_diacritics(e.name.lower()) in q_ascii:
            return e

    best, best_ratio = None, 0.0
    for e in entities:
        r1 = SequenceMatcher(None, q, e.name.lower()).ratio()
        r2 = SequenceMatcher(None, q_ascii, _strip_diacritics(e.name.lower())).ratio()
        ratio = max(r1, r2)
        if ratio > best_ratio:
            best, best_ratio = e, ratio
    return best if best_ratio >= _FUZZY_THRESHOLD else None

def _fuzzy_find_entities_multi(
    query: str, entities: list[Entity],
) -> list[Entity]:
    """Find all entities mentioned in query."""
    q = query.lower()
    found: list[Entity] = []
    for e in entities:
        if e.name.lower() in q:
            found.append(e)
    return found

def _name(eid: str, entity_map: dict[str, Entity]) -> str:
    ent = entity_map.get(eid)
    return ent.name if ent else eid

def _degree_map(entities: list[Entity], relations: list[Relation]) -> dict[str, int]:
    deg: dict[str, int] = defaultdict(int)
    for e in entities:
        deg[e.id] = 0
    for r in relations:
        deg[r.source] += 1
        deg[r.target] += 1
    return deg

def _entity_rels(eid: str, relations: list[Relation]) -> list[Relation]:
    return [r for r in relations if r.source == eid or r.target == eid]

def _format_entity_card(
    e: Entity, relations: list[Relation], entity_map: dict[str, Entity],
) -> str:
    rels = _entity_rels(e.id, relations)
    props = (
        "\n".join(f"- **{p.key}**: {p.value}" for p in e.properties)
        if e.properties else "_Không có thuộc tính_"
    )

    outgoing = [r for r in rels if r.source == e.id]
    incoming = [r for r in rels if r.target == e.id]

    rel_lines: list[str] = []
    for r in outgoing:
        rel_lines.append(f"- → *{r.label}* → **{_name(r.target, entity_map)}**")
    for r in incoming:
        rel_lines.append(f"- ← *{r.label}* ← **{_name(r.source, entity_map)}**")
    rel_text = "\n".join(rel_lines) if rel_lines else "_Chưa có quan hệ_"

    return (
        f"## {e.name} ({e.type})\n\n"
        f"**Thuộc tính:**\n{props}\n\n"
        f"**Quan hệ ({len(rels)}):**\n{rel_text}"
    )

def _rule_based_reply(
    user_message: str,
    entities: list[Entity],
    relations: list[Relation],
    input_text: str = "",
    insight_markdown: str = "",
    metrics_summary: str = "",
) -> str:
    q = user_message.lower().strip()
    q_ascii = _strip_diacritics(q)
    entity_map = {e.id: e for e in entities}

    greetings = {"xin chào", "chào", "hello", "hi", "hey", "xin chao"}
    if q in greetings or any(q.startswith(g) for g in greetings):
        top3 = sorted(entities, key=lambda e: len(_entity_rels(e.id, relations)), reverse=True)[:3]
        names = ", ".join(f"**{e.name}**" for e in top3)
        return (
            f"Xin chào! Đồ thị hiện có **{len(entities)}** thực thể và **{len(relations)}** quan hệ.\n\n"
            f"Các thực thể nổi bật: {names}\n\n"
            "Bạn có thể hỏi:\n"
            '- "Cho tôi biết về [tên]"\n'
            '- "Mối quan hệ giữa A và B"\n'
            '- "Thực thể quan trọng nhất"\n'
            '- "Tóm tắt đồ thị"'
        )

    kb_patterns = [
        r"(?:kb|knowledge\s*base)\s+(?:biết gì|nói gì|cho biết|search|lookup|tra cứu)\s+(?:về\s+)?(.+?)(?:\?|$)",
        r"(?:tra cứu|tìm|search)\s+(?:trong\s+)?(?:kb|knowledge\s*base|cơ sở tri thức)\s+(.+?)(?:\?|$)",
        r"(?:cơ sở tri thức|knowledge base)\s+(?:biết gì|nói gì)\s+(?:về\s+)?(.+?)(?:\?|$)",
    ]
    for pat in kb_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            return _intent_kb_lookup(m.group(1).strip())

    help_kw = {"help", "giúp", "hướng dẫn", "trợ giúp", "hỏi gì", "hỏi được gì",
               "biết gì", "làm gì", "chức năng", "capability", "what can"}
    help_kw_ascii = {
        "giup", "huong dan", "tro giup", "hoi gi", "hoi duoc gi",
        "biet gi", "lam gi", "chuc nang", "ban co the lam duoc gi",
        "ban lam duoc gi",
    }
    if any(kw in q for kw in help_kw) or any(kw in q_ascii for kw in help_kw_ascii):
        return _intent_help(entities, relations, entity_map)

    if not entities and not relations:
        return (
            "Đồ thị hiện tại chưa có dữ liệu. "
            "Vui lòng nhập văn bản và nhấn **Trích xuất** trước khi hỏi đáp."
        )

    summary_kw = {"tóm tắt", "tổng quan", "overview", "summary", "summar",
                  "mô tả đồ thị", "describe graph", "describe the graph"}
    if any(kw in q for kw in summary_kw):
        return _intent_summary(entities, relations)

    _question_words = {"gì", "ai", "đâu", "nào", "sao", "những gì", "cái gì", "thế nào", "như thế nào"}
    path_patterns = [
        r"(?:mối\s+)?quan\s+hệ\s+(?:giữa|của)\s+(.+?)\s+và\s+(.+?)(?:\?|$)",
        r"(?:mối\s+)?liên\s+(?:hệ|quan)\s+(?:giữa|của)\s+(.+?)\s+và\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:liên quan|kết nối|quan hệ)\s+(?:gì|thế nào|như thế nào)?\s*(?:với|đến|tới)\s+(.+?)(?:\?|$)",
        r"relationship\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"how\s+(?:is|are)\s+(.+?)\s+(?:related|connected)\s+to\s+(.+?)(?:\?|$)",
    ]
    for pat in path_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            if a.lower() in _question_words or b.lower() in _question_words:
                break
            return _intent_path(a, b, entities, relations, entity_map)

    compare_patterns = [
        r"so\s+sánh\s+(.+?)\s+và\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:and|with|vs\.?)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
    ]
    for pat in compare_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            return _intent_compare(m.group(1).strip(), m.group(2).strip(), entities, relations, entity_map)

    neighbor_triggers = [
        "kết nối với", "liên quan với", "liên quan đến", "liên quan tới",
        "kết nối gì", "liên quan gì", "quan hệ gì", "quan hệ với",
        "kết nối với gì", "kết nối với ai", "kết nối với đâu",
        "connected to", "linked to", "neighbors of",
        "các kết nối của", "kết nối của",
    ]
    if any(kw in q for kw in neighbor_triggers):
        for e in sorted(entities, key=lambda x: len(x.name), reverse=True):
            if e.name.lower() in q:
                return _intent_neighbors(e.name, entities, relations, entity_map)
        neighbor_patterns = [
            r"(?:các\s+)?(?:kết nối|liên quan|quan hệ)\s+(?:của\s+)?(.+?)(?:\?|$)",
            r"(.+?)\s+(?:kết nối|liên quan|quan hệ)\s+(?:với\s+)?(?:gì|những gì|ai|cái gì|đâu)",
            r"(?:connections?|neighbors?|linked)\s+(?:of|to|with)\s+(.+?)(?:\?|$)",
            r"what\s+is\s+connected\s+to\s+(.+?)(?:\?|$)",
        ]
        for pat in neighbor_patterns:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip().rstrip("?.,!")
                entity = _fuzzy_find_entity(candidate, entities)
                if entity:
                    return _intent_neighbors(entity.name, entities, relations, entity_map)

    top_kw = {"quan trọng nhất", "nổi bật nhất", "top", "most important",
              "most connected", "nhiều kết nối nhất", "node chính",
              "thực thể chính", "hub", "trung tâm"}
    if any(kw in q for kw in top_kw):
        n = 5
        m = re.search(r"top\s*(\d+)", q)
        if m:
            n = min(int(m.group(1)), 20)
        return _intent_top_nodes(n, entities, relations, entity_map)

    predict_kw = {"dự đoán", "predicted", "predict", "dự báo", "gợi ý quan hệ"}
    if any(kw in q for kw in predict_kw):
        return _intent_predicted(relations, entity_map)

    rel_filter_patterns = [
        r"(?:quan hệ|relation|liên kết)\s+(?:loại\s+)?(.+?)(?:\?|$)",
        r"(?:những\s+)?(?:ai|gì)\s+(đầu tư|hợp tác|làm việc|sáng lập|cung cấp|cạnh tranh|thành lập)",
        r"(?:liệt kê|list)\s+(?:các\s+)?(?:quan hệ|relation)\s+(.+?)(?:\?|$)",
    ]
    for pat in rel_filter_patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            label_q = m.group(1).strip()
            result = _intent_relation_filter(label_q, relations, entity_map)
            if result:
                return result

    src_kw = {"văn bản gốc", "source text", "nguyên văn", "trích đoạn",
              "original text", "input text", "text gốc", "đoạn văn"}
    if any(kw in q for kw in src_kw):
        return _intent_source_text(q, input_text, entities)

    count_kw = {"bao nhiêu", "how many", "count", "total", "tổng",
                "đếm", "số lượng", "thống kê", "statistics", "stats"}
    count_kw_ascii = {
        "bao nhieu", "co bao nhieu", "tong so", "so luong", "thong ke",
        "number of", "how many",
    }
    if any(kw in q for kw in count_kw) or any(kw in q_ascii for kw in count_kw_ascii):
        return _intent_count(q, entities, relations)

    type_keywords: dict[str, list[str]] = {
        "Person":       ["person", "người", "nhân vật", "ai", "who"],
        "Organization": ["organization", "company", "tổ chức", "công ty", "doanh nghiệp"],
        "Location":     ["location", "place", "địa điểm", "thành phố", "quốc gia", "đâu", "where", "nơi"],
        "Product":      ["product", "sản phẩm"],
        "Event":        ["event", "sự kiện"],
        "Money":        ["money", "tiền", "doanh thu", "giá trị", "vốn"],
        "Date":         ["date", "time", "ngày", "năm", "thời gian", "khi nào", "when"],
        "Industry":     ["industry", "ngành", "lĩnh vực", "sector"],
        "Percent":      ["percent", "phần trăm", "tỷ lệ"],
    }
    for etype, keywords in type_keywords.items():
        if any(kw in q for kw in keywords):
            match = _intent_type_listing(etype, q, entities, relations, entity_map)
            if match:
                return match

    rel_kw_vi = {"quan hệ", "liên kết", "kết nối", "mối quan hệ", "cạnh"}
    rel_kw_en_tokens = {"relation", "link", "edge", "connection"}
    en_tokens = set(re.findall(r"[a-z]+", q_ascii))
    if any(kw in q for kw in rel_kw_vi) or bool(rel_kw_en_tokens & en_tokens):
        return _intent_relations_list(relations, entity_map)

    matched = _fuzzy_find_entity(q, entities)
    if matched:
        card = _format_entity_card(matched, relations, entity_map)
        kb_extra = _kb_enrich_entity(matched.name)
        return card + kb_extra

    multi = _fuzzy_find_entities_multi(q, entities)
    if len(multi) >= 2:
        parts = []
        for e in multi[:5]:
            parts.append(_format_entity_card(e, relations, entity_map))
        return "\n\n---\n\n".join(parts)
    if len(multi) == 1:
        card = _format_entity_card(multi[0], relations, entity_map)
        kb_extra = _kb_enrich_entity(multi[0].name)
        return card + kb_extra

    rag_docs = rag_mod.retrieve_for_rule_based(
        user_message, input_text, entities, relations,
        insight_markdown=insight_markdown,
        metrics_summary=metrics_summary,
        top_k=5,
    )
    if rag_docs:
        return _intent_rag_fallback(user_message, rag_docs, entities, relations, entity_map)

    return _intent_smart_fallback(q, entities, relations, entity_map)

def _intent_summary(entities: list[Entity], relations: list[Relation]) -> str:
    type_counts = Counter(e.type for e in entities)
    rel_label_counts = Counter(r.label for r in relations)
    predicted = sum(1 for r in relations if r.isPredicted)

    type_table = "\n".join(
        f"| {t} | {c} |" for t, c in type_counts.most_common()
    )
    rel_table = "\n".join(
        f"| {l} | {c} |" for l, c in rel_label_counts.most_common(10)
    )

    deg = _degree_map(entities, relations)
    top3 = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:3]
    entity_map = {e.id: e for e in entities}
    top3_text = ", ".join(f"**{_name(eid, entity_map)}** ({d} kết nối)" for eid, d in top3)

    return (
        f"## Tổng quan đồ thị\n\n"
        f"- **{len(entities)}** thực thể, **{len(relations)}** quan hệ"
        f"{f' (trong đó {predicted} dự đoán)' if predicted else ''}\n"
        f"- **{len(type_counts)}** loại thực thể, **{len(rel_label_counts)}** loại quan hệ\n\n"
        f"### Phân bố loại thực thể\n"
        f"| Loại | Số lượng |\n|---|---|\n{type_table}\n\n"
        f"### Top quan hệ phổ biến\n"
        f"| Nhãn | Số lượng |\n|---|---|\n{rel_table}\n\n"
        f"### Thực thể trung tâm\n{top3_text}"
    )

def _intent_help(
    entities: list[Entity],
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    """Suggest concrete, data-aware follow-up questions."""
    suggestions: list[str] = [
        "Tóm tắt đồ thị hiện tại",
        "Có bao nhiêu thực thể và quan hệ?",
        "Top 5 thực thể quan trọng nhất",
        "Liệt kê tất cả loại thực thể",
        "Liệt kê các quan hệ phổ biến",
    ]

    if entities:
        top_entities = sorted(
            entities, key=lambda e: len(_entity_rels(e.id, relations)), reverse=True,
        )[:3]
        for e in top_entities:
            suggestions.append(f"Cho tôi biết về {e.name}")
            suggestions.append(f"{e.name} kết nối với gì?")

    if len(entities) >= 2:
        a, b = entities[0], entities[1]
        suggestions.append(f"Mối quan hệ giữa {a.name} và {b.name} là gì?")
        suggestions.append(f"So sánh {a.name} và {b.name}")

    if any(r.isPredicted for r in relations):
        suggestions.append("Có quan hệ dự đoán nào?")

    # Remove duplicates but preserve order.
    seen: set[str] = set()
    unique_suggestions: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    lines = [
        "## Tôi có thể giúp bạn hỏi đáp về đồ thị",
        "",
        f"- Hiện có **{len(entities)}** thực thể và **{len(relations)}** quan hệ trong phiên này.",
        "- Bạn có thể thử các câu hỏi sau:",
    ]
    lines.extend(f'  - "{q}"' for q in unique_suggestions[:12])
    lines.extend([
        "",
        "Nếu muốn, bạn cứ hỏi tự nhiên bằng tiếng Việt hoặc tiếng Anh, tôi sẽ tự nhận diện intent.",
    ])
    return "\n".join(lines)

def _intent_path(
    name_a: str, name_b: str,
    entities: list[Entity], relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    ea = _fuzzy_find_entity(name_a, entities)
    eb = _fuzzy_find_entity(name_b, entities)
    if not ea:
        return f"Không tìm thấy thực thể nào khớp với **{name_a}** trong đồ thị."
    if not eb:
        return f"Không tìm thấy thực thể nào khớp với **{name_b}** trong đồ thị."

    direct = []
    for r in relations:
        if (r.source == ea.id and r.target == eb.id):
            direct.append(f"- **{ea.name}** → *{r.label}* → **{eb.name}**")
        elif (r.source == eb.id and r.target == ea.id):
            direct.append(f"- **{eb.name}** → *{r.label}* → **{ea.name}**")

    if direct:
        return (
            f"## Quan hệ trực tiếp giữa {ea.name} và {eb.name}\n\n"
            + "\n".join(direct)
        )

    a_neighbors = {r.target if r.source == ea.id else r.source: r for r in relations if r.source == ea.id or r.target == ea.id}
    b_neighbors = {r.target if r.source == eb.id else r.source: r for r in relations if r.source == eb.id or r.target == eb.id}
    common = set(a_neighbors.keys()) & set(b_neighbors.keys())

    if common:
        paths = []
        for mid_id in list(common)[:5]:
            mid_name = _name(mid_id, entity_map)
            ra = a_neighbors[mid_id]
            rb = b_neighbors[mid_id]
            paths.append(f"- **{ea.name}** →*{ra.label}*→ **{mid_name}** →*{rb.label}*→ **{eb.name}**")
        return (
            f"## Đường đi gián tiếp giữa {ea.name} và {eb.name}\n\n"
            f"Không có quan hệ trực tiếp, nhưng liên kết qua:\n\n"
            + "\n".join(paths)
        )

    kb_rel = kb.find_relation(ea.name, eb.name) if kb.kb_ready else None
    if kb_rel:
        return (
            f"## {ea.name} & {eb.name}\n\n"
            f"Không có quan hệ trong đồ thị hiện tại, nhưng **Knowledge Base** ghi nhận:\n\n"
            f"- **{ea.name}** → *{kb_rel}* → **{eb.name}**"
        )

    return (
        f"Không tìm thấy quan hệ trực tiếp hoặc gián tiếp giữa "
        f"**{ea.name}** và **{eb.name}** trong đồ thị hiện tại."
    )

def _intent_compare(
    name_a: str, name_b: str,
    entities: list[Entity], relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    ea = _fuzzy_find_entity(name_a, entities)
    eb = _fuzzy_find_entity(name_b, entities)
    if not ea:
        return f"Không tìm thấy thực thể **{name_a}**."
    if not eb:
        return f"Không tìm thấy thực thể **{name_b}**."

    rels_a = _entity_rels(ea.id, relations)
    rels_b = _entity_rels(eb.id, relations)

    neighbors_a = {r.target if r.source == ea.id else r.source for r in rels_a}
    neighbors_b = {r.target if r.source == eb.id else r.source for r in rels_b}
    common = neighbors_a & neighbors_b

    props_a = "\n".join(f"  - {p.key}: {p.value}" for p in (ea.properties or [])) or "  _Không có_"
    props_b = "\n".join(f"  - {p.key}: {p.value}" for p in (eb.properties or [])) or "  _Không có_"

    common_names = ", ".join(f"**{_name(c, entity_map)}**" for c in list(common)[:8])
    common_text = common_names if common else "_Không có_"

    return (
        f"## So sánh {ea.name} vs {eb.name}\n\n"
        f"| | **{ea.name}** | **{eb.name}** |\n"
        f"|---|---|---|\n"
        f"| Loại | {ea.type} | {eb.type} |\n"
        f"| Số quan hệ | {len(rels_a)} | {len(rels_b)} |\n"
        f"| Số kết nối | {len(neighbors_a)} | {len(neighbors_b)} |\n\n"
        f"**Thuộc tính {ea.name}:**\n{props_a}\n\n"
        f"**Thuộc tính {eb.name}:**\n{props_b}\n\n"
        f"**Thực thể chung ({len(common)}):** {common_text}"
    )

def _intent_neighbors(
    name: str,
    entities: list[Entity], relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    e = _fuzzy_find_entity(name, entities)
    if not e:
        return f"Không tìm thấy thực thể nào khớp với **{name}**."

    rels = _entity_rels(e.id, relations)
    if not rels:
        return f"**{e.name}** hiện không có kết nối nào trong đồ thị."

    outgoing = [r for r in rels if r.source == e.id]
    incoming = [r for r in rels if r.target == e.id]

    lines = [f"## Các kết nối của {e.name} ({len(rels)} quan hệ)\n"]
    if outgoing:
        lines.append(f"**Đi ra ({len(outgoing)}):**")
        for r in outgoing:
            lines.append(f"- → *{r.label}* → **{_name(r.target, entity_map)}**")
    if incoming:
        lines.append(f"\n**Đi vào ({len(incoming)}):**")
        for r in incoming:
            lines.append(f"- ← *{r.label}* ← **{_name(r.source, entity_map)}**")

    return "\n".join(lines)

def _intent_top_nodes(
    n: int,
    entities: list[Entity], relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    deg = _degree_map(entities, relations)
    ranked = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:n]
    rows = "\n".join(
        f"| {i+1} | **{_name(eid, entity_map)}** | {entity_map[eid].type if eid in entity_map else '?'} | {d} |"
        for i, (eid, d) in enumerate(ranked)
    )
    return (
        f"## Top {n} thực thể quan trọng nhất\n\n"
        f"| # | Tên | Loại | Số kết nối |\n|---|---|---|---|\n{rows}"
    )

def _intent_predicted(relations: list[Relation], entity_map: dict[str, Entity]) -> str:
    predicted = [r for r in relations if r.isPredicted]
    if not predicted:
        return "Hiện tại không có quan hệ dự đoán nào trong đồ thị. Thử nhấn **Dự đoán liên kết** trên dashboard."

    rows = "\n".join(
        f"- **{_name(r.source, entity_map)}** → *{r.label}* → **{_name(r.target, entity_map)}**"
        for r in predicted[:15]
    )
    return (
        f"## Quan hệ dự đoán ({len(predicted)})\n\n{rows}"
        + (f"\n\n_... và {len(predicted) - 15} quan hệ khác._" if len(predicted) > 15 else "")
    )

def _intent_relation_filter(
    label_query: str,
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> Optional[str]:
    lq = label_query.lower().strip()
    matched = [
        r for r in relations
        if lq in r.label.lower() or SequenceMatcher(None, lq, r.label.lower().replace("_", " ")).ratio() > 0.5
    ]
    if not matched:
        return None

    label_groups: dict[str, list[Relation]] = defaultdict(list)
    for r in matched:
        label_groups[r.label].append(r)

    lines = [f"## Quan hệ khớp \"{label_query}\" ({len(matched)} kết quả)\n"]
    for label, rels in label_groups.items():
        lines.append(f"### {label} ({len(rels)})")
        for r in rels[:10]:
            lines.append(f"- **{_name(r.source, entity_map)}** → **{_name(r.target, entity_map)}**")
        if len(rels) > 10:
            lines.append(f"_... và {len(rels) - 10} quan hệ khác_")

    return "\n".join(lines)

def _intent_kb_lookup(query: str) -> str:
    if not kb.kb_ready:
        return "Knowledge Base chưa sẵn sàng."

    triples = kb.get_entity_triples(query, limit=15)
    if not triples:
        results = kb.search_entities(query, limit=5)
        if results:
            listing = "\n".join(
                f"- **{r['name']}** ({r['type']}) — {r['triple_count']} triples"
                for r in results
            )
            return (
                f"Không tìm chính xác **{query}**, nhưng KB có các entity gần đúng:\n\n{listing}\n\n"
                f"Thử hỏi lại với tên chính xác hơn."
            )
        return f"Knowledge Base không có thông tin về **{query}**."

    lines = [f"## Knowledge Base: {query} ({len(triples)} triples)\n"]
    for t in triples:
        conf = t.get("confidence", 0)
        lines.append(
            f"- **{t['subject']}** → *{t['relation']}* → **{t['object']}** "
            f"({conf:.0%})"
        )
    return "\n".join(lines)

def _intent_source_text(
    q: str, input_text: str, entities: list[Entity],
) -> str:
    if not input_text:
        return "Không có văn bản gốc trong phiên hiện tại."

    mentioned = _fuzzy_find_entities_multi(q, entities)
    if mentioned:
        excerpts = []
        sentences = re.split(r'[.!?\n]+', input_text)
        for e in mentioned[:3]:
            for sent in sentences:
                if e.name.lower() in sent.lower() and len(sent.strip()) > 10:
                    excerpts.append(f"- ...{sent.strip()}...")
                    break
        if excerpts:
            names = ", ".join(f"**{e.name}**" for e in mentioned[:3])
            return f"## Trích đoạn về {names}\n\n" + "\n".join(excerpts)

    preview = input_text[:500]
    if len(input_text) > 500:
        preview += "..."
    return f"## Văn bản gốc (trích)\n\n> {preview}"

def _intent_count(
    q: str, entities: list[Entity], relations: list[Relation],
) -> str:
    type_counts = Counter(e.type for e in entities)
    rel_counts = Counter(r.label for r in relations)
    predicted = sum(1 for r in relations if r.isPredicted)

    type_table = "\n".join(f"| {t} | {c} |" for t, c in type_counts.most_common())
    rel_table = "\n".join(f"| {l} | {c} |" for l, c in rel_counts.most_common(8))

    return (
        f"## Thống kê đồ thị\n\n"
        f"- **{len(entities)}** thực thể\n"
        f"- **{len(relations)}** quan hệ"
        f"{f' (trong đó {predicted} dự đoán)' if predicted else ''}\n\n"
        f"### Theo loại thực thể\n"
        f"| Loại | Số lượng |\n|---|---|\n{type_table}\n\n"
        f"### Theo loại quan hệ\n"
        f"| Nhãn | Số lượng |\n|---|---|\n{rel_table}"
    )

def _intent_type_listing(
    etype: str, q: str,
    entities: list[Entity], relations: list[Relation],
    entity_map: dict[str, Entity],
) -> Optional[str]:
    filtered = [e for e in entities if e.type == etype]
    if not filtered:
        return None

    deg = _degree_map(entities, relations)
    filtered.sort(key=lambda e: deg.get(e.id, 0), reverse=True)

    rows: list[str] = []
    for e in filtered[:20]:
        d = deg.get(e.id, 0)
        rows.append(f"| **{e.name}** | {d} kết nối |")

    return (
        f"## Danh sách {etype} ({len(filtered)})\n\n"
        f"| Tên | Kết nối |\n|---|---|\n"
        + "\n".join(rows)
        + (f"\n\n_... và {len(filtered) - 20} thực thể khác._" if len(filtered) > 20 else "")
    )

def _intent_relations_list(
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    label_counts = Counter(r.label for r in relations)
    lines = [f"## Tất cả quan hệ ({len(relations)})\n"]
    lines.append(f"### Phân bố theo loại ({len(label_counts)} loại)\n")
    for label, cnt in label_counts.most_common():
        lines.append(f"- **{label}**: {cnt}")

    lines.append(f"\n### Danh sách chi tiết (tối đa 20)\n")
    for r in relations[:20]:
        pred = " _(dự đoán)_" if r.isPredicted else ""
        lines.append(
            f"- **{_name(r.source, entity_map)}** → *{r.label}* → "
            f"**{_name(r.target, entity_map)}**{pred}"
        )
    if len(relations) > 20:
        lines.append(f"\n_... và {len(relations) - 20} quan hệ khác._")

    return "\n".join(lines)

def _kb_enrich_entity(name: str) -> str:
    """Append KB triples if available."""
    if not kb.kb_ready:
        return ""
    triples = kb.get_entity_triples(name, limit=8)
    if not triples:
        return ""
    lines = ["\n\n---\n### Thông tin thêm từ Knowledge Base\n"]
    for t in triples:
        lines.append(f"- **{t['subject']}** → *{t['relation']}* → **{t['object']}**")
    return "\n".join(lines)

def _intent_rag_fallback(
    question: str,
    docs: list[rag_mod.Document],
    entities: list[Entity],
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    """Build an answer from RAG-retrieved documents when no specific intent matched."""
    lines: list[str] = []
    lines.append(f"Dựa trên dữ liệu liên quan tìm được cho câu hỏi **\"{question}\"**:\n")

    text_chunks = [d for d in docs if d.source == "input_text"]
    entity_docs = [d for d in docs if d.source == "entity"]
    rel_docs = [d for d in docs if d.source == "relation"]
    kb_docs = [d for d in docs if d.source == "kb_triple"]
    insight_docs = [d for d in docs if d.source == "insight"]
    metrics_docs = [d for d in docs if d.source == "metrics"]

    if text_chunks:
        lines.append(" **Đoạn văn bản liên quan:**")
        for d in text_chunks[:3]:
            lines.append(f"> {d.text[:300]}")
        lines.append("")

    if insight_docs:
        lines.append(" **Insight (trích đoạn):**")
        for d in insight_docs[:4]:
            lines.append(f"> {d.text[:350]}")
        lines.append("")

    if metrics_docs:
        lines.append(" **Metrics:**")
        for d in metrics_docs[:4]:
            lines.append(f"- {d.text[:400]}")
        lines.append("")

    if entity_docs:
        lines.append(" **Thực thể liên quan:**")
        for d in entity_docs[:3]:
            name = d.metadata.get("entity_name", "?")
            etype = d.metadata.get("entity_type", "")
            lines.append(f"- **{name}** ({etype}): {d.text[:200]}")
        lines.append("")

    if rel_docs:
        lines.append(" **Quan hệ liên quan:**")
        for d in rel_docs[:4]:
            lines.append(f"- {d.text}")
        lines.append("")

    if kb_docs:
        lines.append(" **Từ Knowledge Base:**")
        for d in kb_docs[:3]:
            lines.append(f"- {d.text}")
        lines.append("")

    if not any([text_chunks, insight_docs, metrics_docs, entity_docs, rel_docs, kb_docs]):
        return _intent_smart_fallback(question.lower(), entities, relations, entity_map)

    lines.append(" _Hãy hỏi cụ thể hơn để tôi trả lời chính xác hơn._")
    return "\n".join(lines)

def _intent_smart_fallback(
    q: str,
    entities: list[Entity], relations: list[Relation],
    entity_map: dict[str, Entity],
) -> str:
    """When no intent matches, try KB search then give helpful suggestions."""
    if kb.kb_ready:
        words = [w for w in q.split() if len(w) > 2]
        for word in words:
            results = kb.search_entities(word, limit=3)
            if results:
                listing = "\n".join(
                    f"- **{r['name']}** ({r['type']})"
                    for r in results
                )
                return (
                    f"Tôi không chắc bạn đang hỏi về gì, nhưng tìm thấy trong Knowledge Base:\n\n"
                    f"{listing}\n\n"
                    f"Hãy thử: _\"Cho tôi biết về {results[0]['name']}\"_ hoặc "
                    f"_\"KB biết gì về {results[0]['name']}\"_"
                )

    top3 = sorted(entities, key=lambda e: len(_entity_rels(e.id, relations)), reverse=True)[:3]
    suggestions = "\n".join(f'- "Cho tôi biết về {e.name}"' for e in top3)

    return (
        f"Tôi chưa hiểu câu hỏi này. Đồ thị hiện có **{len(entities)}** thực thể "
        f"và **{len(relations)}** quan hệ.\n\n"
        f"Thử hỏi:\n{suggestions}\n"
        '- "Tóm tắt đồ thị"\n'
        '- "Thực thể quan trọng nhất"\n'
        '- "Giúp" (xem tất cả dạng câu hỏi)\n'
    )
