"""
Context filter — reduces retrieved documents to the most relevant set
before sending to the LLM, respecting a token budget.

Pipeline:
  1. Boost score by entity overlap with query
  2. Apply source-type boost (graph > bm25 for graph_first mode)
  3. Deduplicate semantically similar chunks
  4. Truncate to token budget
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Union

from rag import Document
from graph_retriever import ScoredDocument
from query_understanding import ParsedQuery, MODE_GRAPH_FIRST

# ~4 chars per token is a conservative estimate for mixed Vietnamese/English text
_APPROX_CHARS_PER_TOKEN = 4

# Default token budgets
TOKEN_BUDGET_GRAPH_FIRST = 1600
TOKEN_BUDGET_HYBRID      = 2000
TOKEN_BUDGET_DEFAULT     = 1800

# Similarity threshold to consider two chunks as duplicates
_DEDUP_THRESHOLD = 0.82


def _entity_overlap_score(text: str, entity_names: list[str]) -> float:
    """Return fraction of entity names that appear in text."""
    if not entity_names:
        return 0.3
    text_lower = text.lower()
    hits = sum(1 for name in entity_names if name.lower() in text_lower)
    return hits / len(entity_names)


def _is_near_duplicate(text: str, seen_texts: list[str]) -> bool:
    """Return True if text is too similar to any already-accepted chunk."""
    t_norm = re.sub(r"\s+", " ", text.lower().strip())[:300]
    for seen in seen_texts:
        s_norm = re.sub(r"\s+", " ", seen.lower().strip())[:300]
        if SequenceMatcher(None, t_norm, s_norm).ratio() > _DEDUP_THRESHOLD:
            return True
    return False


def filter_context(
    docs: list[Union[Document, ScoredDocument]],
    pq: ParsedQuery,
    token_budget: int | None = None,
) -> list[Document]:
    """
    Filter and rank docs, returning an ordered list ready for LLM formatting.

    Args:
        docs:         Mixed list of Document or ScoredDocument from retrieval.
        pq:           ParsedQuery with intent, mode, and entity names.
        token_budget: Max tokens to allow (chars / _APPROX_CHARS_PER_TOKEN).
                      Defaults to mode-appropriate budget.
    """
    if token_budget is None:
        if pq.mode == MODE_GRAPH_FIRST:
            token_budget = TOKEN_BUDGET_GRAPH_FIRST
        else:
            token_budget = TOKEN_BUDGET_HYBRID

    budget_chars = token_budget * _APPROX_CHARS_PER_TOKEN

    # ── Normalize to (base_score, source_type, doc) ───────────────────────────
    items: list[tuple[float, str, Document]] = []
    for item in docs:
        if isinstance(item, ScoredDocument):
            items.append((item.score, item.source_type, item.doc))
        else:
            items.append((0.4, "bm25", item))

    # ── Compute final score ───────────────────────────────────────────────────
    scored: list[tuple[float, Document]] = []
    for base_score, source_type, doc in items:
        overlap    = _entity_overlap_score(doc.text, pq.entities_mentioned)
        src_boost  = 1.3 if source_type == "graph" else 1.0
        # Direct relations carry highest signal
        if doc.source in ("graph_relation", "graph_path"):
            src_boost = 1.5
        final = base_score * (1.0 + overlap) * src_boost
        scored.append((final, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Dedup + budget truncation ─────────────────────────────────────────────
    filtered: list[Document] = []
    seen_texts: list[str] = []
    used_chars = 0

    for score, doc in scored:
        if used_chars >= budget_chars:
            break
        if _is_near_duplicate(doc.text, seen_texts):
            continue
        filtered.append(doc)
        seen_texts.append(doc.text)
        used_chars += len(doc.text)

    return filtered


def format_context_for_llm(
    docs: list[Document],
    evidence_count: int = 0,
) -> str:
    """
    Format filtered documents into a compact context string for the LLM prompt.
    Groups by source type with clear section headers.
    """
    sections: dict[str, list[str]] = {
        "graph_relation": [],
        "graph_path":     [],
        "graph_entity":   [],
        "kb_triple":      [],
        "relation":       [],
        "entity":         [],
        "insight":        [],
        "metrics":        [],
        "input_text":     [],
    }

    for doc in docs:
        sections.setdefault(doc.source, []).append(doc.text)

    lines: list[str] = [
        f"### Context ({len(docs)} chunks, hybrid retrieval)", ""
    ]

    if sections["graph_relation"] or sections["graph_path"]:
        lines.append("**Quan hệ trực tiếp từ Knowledge Graph:**")
        for t in sections["graph_relation"][:8]:
            lines.append(f"- {t}")
        for t in sections["graph_path"][:4]:
            lines.append(f"- {t}")
        lines.append("")

    if sections["graph_entity"]:
        lines.append("**Thông tin thực thể:**")
        for t in sections["graph_entity"][:4]:
            lines.append(f"- {t[:300]}")
        lines.append("")

    if sections["kb_triple"] or sections["relation"]:
        lines.append("**Knowledge Base / Quan hệ bổ sung:**")
        for t in (sections["kb_triple"] + sections["relation"])[:6]:
            lines.append(f"- {t}")
        lines.append("")

    if sections["insight"]:
        lines.append("**Báo cáo Insight (trích đoạn):**")
        for t in sections["insight"][:5]:
            lines.append(f"> {t[:350]}")
        lines.append("")

    if sections["metrics"]:
        lines.append("**Metrics đồ thị:**")
        for t in sections["metrics"][:6]:
            lines.append(f"- {t[:400]}")
        lines.append("")

    if sections["entity"]:
        lines.append("**Entity context (BM25):**")
        for t in sections["entity"][:3]:
            lines.append(f"- {t[:250]}")
        lines.append("")

    if sections["input_text"]:
        lines.append("**Trích đoạn văn bản nguồn:**")
        for t in sections["input_text"][:3]:
            lines.append(f"> {t[:300]}")
        lines.append("")

    if not any(v for v in sections.values()):
        return "(Không có context liên quan)"

    return "\n".join(lines)
