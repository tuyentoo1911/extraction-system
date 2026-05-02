"""
Answer validator — checks that the LLM-generated answer is grounded in
the retrieved context, assigns a confidence score, and flags unsupported claims.

Used to:
  - Reject hallucinated answers (replace with no-data response)
  - Compute a heuristic confidence score (0-1) for the response schema
  - Surface evidence_count to the caller
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag import Document

# ── Phrases the model uses when it correctly reports missing data ─────────────
_NO_DATA_PHRASES = {
    "không đủ dữ liệu",
    "không tìm thấy thông tin",
    "không có thông tin",
    "không có dữ liệu phù hợp",
    "không tìm thấy dữ liệu",
    "no data",
    "not found",
    "insufficient data",
    "no relevant",
    "no information",
}

# ── Phrases that indicate speculative / hallucinated content ─────────────────
_SPECULATION_PATTERNS = [
    r"\btôi nghĩ\b",
    r"\btheo tôi\b",
    r"\bcó thể là\b",
    r"\bchắc là\b",
    r"\bchắc chắn\b(?!\s+(?:không|rằng))",   # "chắc chắn" without negation
    r"\bI think\b",
    r"\bI believe\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
]

# ── Standard no-data reply ────────────────────────────────────────────────────
NO_DATA_REPLY = (
    "Không đủ dữ liệu trong hệ thống để trả lời câu hỏi này.\n\n"
    "_Hãy thử hỏi cụ thể hơn hoặc đảm bảo đồ thị đã được trích xuất từ văn bản liên quan._"
)


@dataclass
class ValidationResult:
    """Result of validating a single LLM answer."""
    passed: bool
    confidence: float          # 0.0 – 1.0
    evidence_count: int
    unsupported_claims: list[str] = field(default_factory=list)
    speculation_flags: list[str] = field(default_factory=list)
    reason: str = "ok"         # ok | empty_answer | model_no_data | low_grounding | speculation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_claimed_names(text: str) -> list[str]:
    """
    Pull entity names from bold markdown (**name**) and bracket notation ([name]).
    These are the "claims" we check against the retrieved context.
    """
    bold      = re.findall(r"\*\*(.+?)\*\*", text)
    bracketed = re.findall(r"\[([^\[\]]+?)\]", text)
    # Also pick out relation labels in --(...)--> notation
    relation_labels = re.findall(r"--\((.+?)\)-->", text)
    all_names = bold + bracketed + relation_labels
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for name in all_names:
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _name_in_context(name: str, context_corpus: str) -> bool:
    """Return True if the name (or a close variant) appears in the context."""
    name_lower = name.lower().strip()
    corpus_lower = context_corpus.lower()

    if name_lower in corpus_lower:
        return True

    # Partial match: if at least 60% of words in name appear in corpus
    words = [w for w in name_lower.split() if len(w) > 2]
    if not words:
        return False
    found = sum(1 for w in words if w in corpus_lower)
    return found / len(words) >= 0.6


# ── Public API ────────────────────────────────────────────────────────────────

def validate_answer(
    answer: str,
    context_docs: list[Document],
    query_entities: list[str],
) -> ValidationResult:
    """
    Validate an LLM answer against retrieved context.

    Args:
        answer:         Raw answer string from the LLM.
        context_docs:   Filtered context documents used to generate the answer.
        query_entities: Entity names from the original ParsedQuery.

    Returns:
        ValidationResult with passed flag, confidence, and diagnostic info.
    """
    # ── Empty answer ──────────────────────────────────────────────────────────
    if not answer or not answer.strip():
        return ValidationResult(
            passed=False, confidence=0.0, evidence_count=0,
            reason="empty_answer",
        )

    answer_lower = answer.lower()

    # ── Model self-reports no data ────────────────────────────────────────────
    if any(phrase in answer_lower for phrase in _NO_DATA_PHRASES):
        return ValidationResult(
            passed=True, confidence=0.85, evidence_count=0,
            reason="model_no_data",
        )

    # ── Check speculation / hallucination-risk phrases ────────────────────────
    speculation_hits: list[str] = []
    for pattern in _SPECULATION_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            speculation_hits.append(pattern)

    # ── Build context corpus ──────────────────────────────────────────────────
    context_corpus = " ".join(d.text for d in context_docs)
    evidence_count = 0
    unsupported: list[str] = []

    # Check claimed entity / relation names
    claimed = _extract_claimed_names(answer)
    for name in claimed:
        if _name_in_context(name, context_corpus):
            evidence_count += 1
        else:
            unsupported.append(name)

    # Check that query entities themselves appear in context
    for name in query_entities:
        if _name_in_context(name, context_corpus):
            evidence_count += 1

    # ── Compute confidence ────────────────────────────────────────────────────
    total_claims = max(1, len(claimed) + len(query_entities))
    grounding_ratio = evidence_count / total_claims

    confidence = grounding_ratio
    if speculation_hits:
        confidence *= 0.75
    if not context_docs:
        confidence *= 0.5
    confidence = round(min(1.0, confidence), 3)

    # ── Pass / fail decision ──────────────────────────────────────────────────
    # Pass if: grounding >= 30% OR answer has no extracted names (e.g. simple listing)
    passed = grounding_ratio >= 0.30 or not claimed

    reason = "ok"
    if not passed:
        reason = "low_grounding"
    elif speculation_hits:
        reason = "speculation"

    return ValidationResult(
        passed=passed,
        confidence=confidence,
        evidence_count=evidence_count,
        unsupported_claims=unsupported[:5],
        speculation_flags=speculation_hits[:3],
        reason=reason,
    )


def should_replace_with_no_data(result: ValidationResult) -> bool:
    """
    Return True when the answer should be replaced with the standard no-data reply.
    We only replace for clear grounding failures, not speculation warnings.
    """
    return (
        not result.passed
        and result.reason == "low_grounding"
        and result.evidence_count == 0
    )
