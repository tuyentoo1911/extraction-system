"""
Query Understanding layer — parses user query before retrieval.

Outputs a ParsedQuery with:
  - intent: relationship | count | summary | entity_lookup | compare |
            kb_lookup | help | greeting | relation_list | neighbors |
            top_nodes | predicted | source_text | type_list | unknown
  - mode:   deterministic (rule-only) | graph_first | hybrid
  - entities_mentioned: list of entity names found in query
  - constraints: top_n, entity_type, kb_query, entity_a/b, etc.
  - is_followup: True when query refers to a previous turn
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from schemas import Entity, Relation

# ── Intent constants ──────────────────────────────────────────────────────────
INTENT_RELATIONSHIP  = "relationship"
INTENT_COUNT         = "count"
INTENT_SUMMARY       = "summary"
INTENT_ENTITY_LOOKUP = "entity_lookup"
INTENT_COMPARE       = "compare"
INTENT_KB_LOOKUP     = "kb_lookup"
INTENT_HELP          = "help"
INTENT_GREETING      = "greeting"
INTENT_RELATION_LIST = "relation_list"
INTENT_NEIGHBORS     = "neighbors"
INTENT_TOP_NODES     = "top_nodes"
INTENT_PREDICTED     = "predicted"
INTENT_SOURCE_TEXT   = "source_text"
INTENT_TYPE_LIST     = "type_list"
INTENT_UNKNOWN       = "unknown"

# ── Query mode constants ──────────────────────────────────────────────────────
MODE_DETERMINISTIC = "deterministic"   # rule-based only, no LLM needed
MODE_GRAPH_FIRST   = "graph_first"     # graph retrieval priority + LLM
MODE_HYBRID        = "hybrid"          # graph + BM25 balanced + LLM

# Follow-up pronouns (Vietnamese + English)
_FOLLOWUP_PRONOUNS = {
    "nó", "họ", "chúng", "đó", "này", "kia",
    "công ty đó", "tổ chức đó", "thực thể này", "thực thể đó",
    "người đó", "nơi đó", "sự kiện đó",
    "it", "they", "that", "this", "those", "them",
}

_QUESTION_WORDS = {"gì", "ai", "đâu", "nào", "sao", "những gì", "cái gì"}

_VIET_DIACRITICS = str.maketrans(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ",
    "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd"
    "AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD",
)


def _strip_diacritics(s: str) -> str:
    return s.translate(_VIET_DIACRITICS)


def _fuzzy_score(a: str, b: str) -> float:
    return max(
        SequenceMatcher(None, a, b).ratio(),
        SequenceMatcher(None, _strip_diacritics(a), _strip_diacritics(b)).ratio(),
    )


# ── Public dataclass ──────────────────────────────────────────────────────────

@dataclass
class ParsedQuery:
    """Structured representation of a parsed user query."""
    intent: str
    mode: str
    entities_mentioned: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    is_followup: bool = False
    raw: str = ""

    def is_deterministic(self) -> bool:
        return self.mode == MODE_DETERMINISTIC

    def needs_llm(self) -> bool:
        return self.mode in (MODE_GRAPH_FIRST, MODE_HYBRID)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_entities(q: str, entities: list[Entity]) -> list[str]:
    """Find entity names mentioned in query (exact → substring → stripped → fuzzy)."""
    q_lower = q.lower()
    q_ascii = _strip_diacritics(q_lower)
    found: list[str] = []
    seen_ids: set[str] = set()

    for e in sorted(entities, key=lambda x: len(x.name), reverse=True):
        if e.id in seen_ids:
            continue
        name_lower = e.name.lower()
        name_ascii = _strip_diacritics(name_lower)

        if name_lower in q_lower or q_lower in name_lower:
            found.append(e.name)
            seen_ids.add(e.id)
            continue
        if name_ascii in q_ascii or q_ascii in name_ascii:
            found.append(e.name)
            seen_ids.add(e.id)
            continue

    # Fuzzy fallback for short single-entity queries
    if not found and len(q.split()) <= 4:
        best_score, best_name = 0.0, None
        for e in entities:
            s = _fuzzy_score(q_lower, e.name.lower())
            if s > best_score:
                best_score, best_name = s, e.name
        if best_score >= 0.55 and best_name:
            found.append(best_name)

    return found


def _detect_followup(q: str, history: list[dict]) -> bool:
    """Return True when the query seems to continue a previous conversation turn."""
    if not history:
        return False
    q_lower = q.lower()
    if any(p in q_lower for p in _FOLLOWUP_PRONOUNS):
        return True
    # Very short query after an established session is likely a follow-up
    if len(q.split()) <= 3 and len(history) >= 2:
        return True
    return False


def _resolve_followup_entities(
    pq: ParsedQuery,
    history: list[dict],
    entities: list[Entity],
) -> list[str]:
    """
    If the query is a follow-up with no detected entities, try to inherit
    entities from the previous turn.
    """
    if not pq.is_followup or pq.entities_mentioned:
        return pq.entities_mentioned

    # Gather entity names from last 2 user turns
    entity_names = {e.name.lower(): e.name for e in entities}
    inherited: list[str] = []
    user_turns = [h for h in history if h.get("role") == "user"][-2:]
    for turn in user_turns:
        content = turn.get("content", "").lower()
        for name_lower, name in entity_names.items():
            if name_lower in content and name not in inherited:
                inherited.append(name)
    return inherited


# ── Public API ────────────────────────────────────────────────────────────────

def parse_query(
    raw: str,
    entities: list[Entity],
    relations: list[Relation],
    history: Optional[list[dict]] = None,
) -> ParsedQuery:
    """
    Parse a user query into a structured ParsedQuery.
    Uses rule-based pattern matching; no external model required.
    """
    q = raw.lower().strip()
    q_ascii = _strip_diacritics(q)
    history = history or []

    is_followup = _detect_followup(raw, history)
    mentioned = _detect_entities(raw, entities)
    constraints: dict = {}

    # ── Greeting ──────────────────────────────────────────────────────────────
    greetings = {"xin chào", "chào", "hello", "hi", "hey"}
    if q in greetings or any(q.startswith(g) for g in greetings):
        return ParsedQuery(
            intent=INTENT_GREETING, mode=MODE_DETERMINISTIC, raw=raw,
        )

    # ── Help ──────────────────────────────────────────────────────────────────
    help_kw    = {"help", "giúp", "hướng dẫn", "trợ giúp", "hỏi gì", "chức năng"}
    help_ascii = {"giup", "huong dan", "tro giup", "hoi gi", "chuc nang"}
    if any(k in q for k in help_kw) or any(k in q_ascii for k in help_ascii):
        return ParsedQuery(intent=INTENT_HELP, mode=MODE_DETERMINISTIC, raw=raw)

    # ── KB lookup ─────────────────────────────────────────────────────────────
    kb_patterns = [
        r"(?:kb|knowledge\s*base|cơ sở tri thức)\s+(?:biết gì|nói gì|cho biết|"
        r"search|lookup|tra cứu)?\s*(?:về\s+)?(.+?)(?:\?|$)",
        r"(?:tra cứu|tìm trong)\s+(?:kb|knowledge base)\s+(.+?)(?:\?|$)",
    ]
    for pat in kb_patterns:
        m = re.search(pat, q)
        if m:
            constraints["kb_query"] = m.group(1).strip()
            return ParsedQuery(
                intent=INTENT_KB_LOOKUP, mode=MODE_DETERMINISTIC,
                entities_mentioned=mentioned, constraints=constraints, raw=raw,
            )

    # ── Summary / overview ────────────────────────────────────────────────────
    summary_kw    = {"tóm tắt", "tổng quan", "overview", "summary", "mô tả đồ thị"}
    summary_ascii = {"tom tat", "tong quan"}
    if any(k in q for k in summary_kw) or any(k in q_ascii for k in summary_ascii):
        return ParsedQuery(
            intent=INTENT_SUMMARY, mode=MODE_HYBRID,
            entities_mentioned=mentioned, raw=raw,
        )

    # ── Relationship path ─────────────────────────────────────────────────────
    path_patterns = [
        r"(?:mối\s+)?quan\s+hệ\s+(?:giữa|của)\s+(.+?)\s+và\s+(.+?)(?:\?|$)",
        r"(?:mối\s+)?liên\s+(?:hệ|quan)\s+(?:giữa|của)\s+(.+?)\s+và\s+(.+?)(?:\?|$)",
        r"relationship\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:liên quan|kết nối|quan hệ)\s+(?:gì|thế nào)?"
        r"\s*(?:với|đến|tới)\s+(.+?)(?:\?|$)",
    ]
    for pat in path_patterns:
        m = re.search(pat, q)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            if a not in _QUESTION_WORDS and b not in _QUESTION_WORDS:
                constraints["entity_a"] = a
                constraints["entity_b"] = b
                return ParsedQuery(
                    intent=INTENT_RELATIONSHIP, mode=MODE_GRAPH_FIRST,
                    entities_mentioned=mentioned or [a, b],
                    constraints=constraints, is_followup=is_followup, raw=raw,
                )

    # ── Compare ───────────────────────────────────────────────────────────────
    compare_patterns = [
        r"so\s+sánh\s+(.+?)\s+và\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:and|with|vs\.?)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
    ]
    for pat in compare_patterns:
        m = re.search(pat, q)
        if m:
            constraints["entity_a"] = m.group(1).strip()
            constraints["entity_b"] = m.group(2).strip()
            return ParsedQuery(
                intent=INTENT_COMPARE, mode=MODE_GRAPH_FIRST,
                entities_mentioned=mentioned, constraints=constraints,
                is_followup=is_followup, raw=raw,
            )

    # ── Neighbors ─────────────────────────────────────────────────────────────
    neighbor_kw = {
        "kết nối với", "liên quan với", "liên quan đến", "liên quan tới",
        "connected to", "linked to", "neighbors of",
        "các kết nối của", "kết nối của",
    }
    if any(k in q for k in neighbor_kw):
        return ParsedQuery(
            intent=INTENT_NEIGHBORS, mode=MODE_GRAPH_FIRST,
            entities_mentioned=mentioned, is_followup=is_followup, raw=raw,
        )

    # ── Top nodes ─────────────────────────────────────────────────────────────
    top_kw = {"quan trọng nhất", "nổi bật nhất", "most important", "most connected", "hub node"}
    if any(k in q for k in top_kw) or re.search(r"\btop\s*\d*\b", q):
        m = re.search(r"top\s*(\d+)", q)
        constraints["top_n"] = min(int(m.group(1)), 20) if m else 5
        return ParsedQuery(
            intent=INTENT_TOP_NODES, mode=MODE_DETERMINISTIC,
            constraints=constraints, raw=raw,
        )

    # ── Predicted links ───────────────────────────────────────────────────────
    predict_kw = {"dự đoán", "predicted", "predict", "dự báo", "gợi ý quan hệ"}
    if any(k in q for k in predict_kw):
        return ParsedQuery(intent=INTENT_PREDICTED, mode=MODE_DETERMINISTIC, raw=raw)

    # ── Source text ───────────────────────────────────────────────────────────
    source_kw = {"văn bản gốc", "source text", "nguyên văn", "trích đoạn", "text gốc"}
    if any(k in q for k in source_kw):
        return ParsedQuery(
            intent=INTENT_SOURCE_TEXT, mode=MODE_DETERMINISTIC,
            entities_mentioned=mentioned, raw=raw,
        )

    # ── Count / statistics ────────────────────────────────────────────────────
    count_kw    = {"bao nhiêu", "how many", "count", "total", "thống kê", "statistics", "stats"}
    count_ascii = {"bao nhieu", "thong ke", "so luong"}
    if any(k in q for k in count_kw) or any(k in q_ascii for k in count_ascii):
        return ParsedQuery(
            intent=INTENT_COUNT, mode=MODE_DETERMINISTIC,
            entities_mentioned=mentioned, raw=raw,
        )

    # ── Type listing ──────────────────────────────────────────────────────────
    type_trigger_kw    = {"liệt kê", "danh sách", "list all", "tất cả", "toàn bộ"}
    type_trigger_ascii = {"liet ke", "danh sach", "tat ca"}
    if any(k in q for k in type_trigger_kw) or any(k in q_ascii for k in type_trigger_ascii):
        _entity_type_map = {
            "person": "Person",       "người": "Person",        "nhân vật": "Person",
            "organization": "Organization", "công ty": "Organization", "tổ chức": "Organization",
            "location": "Location",   "địa điểm": "Location",
            "product": "Product",     "sản phẩm": "Product",
            "event": "Event",         "sự kiện": "Event",
            "money": "Money",         "tiền": "Money",
            "date": "Date",           "ngày": "Date",
            "industry": "Industry",   "ngành": "Industry",
            "percent": "Percent",     "phần trăm": "Percent",
        }
        for kw, etype in _entity_type_map.items():
            if kw in q:
                constraints["entity_type"] = etype
                break
        return ParsedQuery(
            intent=INTENT_TYPE_LIST, mode=MODE_DETERMINISTIC,
            constraints=constraints, raw=raw,
        )

    # ── Relation list ─────────────────────────────────────────────────────────
    rel_kw = {"quan hệ", "liên kết", "mối quan hệ", "relation", "edge"}
    if any(k in q for k in rel_kw) and not mentioned:
        return ParsedQuery(intent=INTENT_RELATION_LIST, mode=MODE_DETERMINISTIC, raw=raw)

    # ── Entity lookup (with entities detected) ────────────────────────────────
    if mentioned:
        pq = ParsedQuery(
            intent=INTENT_ENTITY_LOOKUP, mode=MODE_GRAPH_FIRST,
            entities_mentioned=mentioned, is_followup=is_followup, raw=raw,
        )
        if is_followup:
            pq.entities_mentioned = _resolve_followup_entities(pq, history, entities)
        return pq

    # ── Follow-up with no entities → inherit from history ────────────────────
    if is_followup:
        inherited = _resolve_followup_entities(
            ParsedQuery(intent=INTENT_UNKNOWN, mode=MODE_HYBRID,
                        is_followup=True, raw=raw),
            history, entities,
        )
        if inherited:
            return ParsedQuery(
                intent=INTENT_ENTITY_LOOKUP, mode=MODE_GRAPH_FIRST,
                entities_mentioned=inherited, is_followup=True, raw=raw,
            )

    # ── Unknown → hybrid LLM ─────────────────────────────────────────────────
    return ParsedQuery(
        intent=INTENT_UNKNOWN, mode=MODE_HYBRID,
        entities_mentioned=mentioned, is_followup=is_followup, raw=raw,
    )
