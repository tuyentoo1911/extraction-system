"""Xây dựng Knowledge Graph từ NER output + Knowledge Base.

Pipeline trích xuất quan hệ — 3 tầng ưu tiên giảm dần:
────────────────────────────────────────────────────────
  Tầng 1 — Pattern-based (regex + Jaccard entity matching)
      13 RELATION_PATTERNS port từ knowledge_graph.ipynb.
      Capture group bắt đúng SUBJECT / OBJECT ngay cạnh verb,
      không bị greedy capture dư text như phiên bản trước.
      Entity matching dùng exact → substring → Jaccard(≥0.4).

  Tầng 2 — Knowledge Base lookup
      Tra cứu triples corpus (6 298 triples) cho cặp chưa có relation.

  Tầng 3 — Co-occurrence scored (fallback)
      Proximity + type_priority + KB_boost score.
      Bỏ qua quan hệ nếu không xác định được nhãn cụ thể (không dùng "LIÊN QUAN ĐẾN").

Regex patterns được test trực tiếp trên câu mẫu từ ảnh:
  "Nguyễn Minh Anh từng làm việc tại FPT Software từ năm 2010..."
  → LÀM VIỆC TẠI: person='Nguyễn Minh Anh' | org='FPT Software' ✅
  "SaoVietTech ký thỏa thuận hợp tác chiến lược với Global AI..."
  → HỢP TÁC: org1='SaoVietTech' | org2='Global AI Solutions Inc.' ✅
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from constants import TYPE_MAP
from schemas import Entity, EntityProperty, Relation, GraphData
from ner import split_sentences
import knowledge_base as kb

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# §1 — Constants
# ═══════════════════════════════════════════════════════════════════

_GENERIC_DATE_WORDS = {
    "nam", "năm", "thang", "tháng", "ngay", "ngày",
    "quy", "quý", "tuan", "tuần",
}
_LOCATION_PREFIXES = {
    "tp", "tp.", "thành", "thanh", "quận", "quan",
    "huyện", "huyen", "tỉnh", "tinh",
    "phường", "phuong", "xã", "xa", "thị", "thi",
}

_RELATION_LABELS_VI: dict[tuple[str, str], str] = {
    ("Person", "Organization"):       "executive_of",
    ("Organization", "Person"):       "has_member",
    ("Person", "Location"):           "lives_in",
    ("Organization", "Location"):     "headquartered_in",
    ("Person", "Event"):              "participated_in",
    ("Organization", "Event"):        "held_event",
    ("Person", "Product"):            "uses_product",
    ("Organization", "Product"):      "developed",
    ("Product", "Event"):             "launched_at",
    ("Location", "Event"):            "held_in",
    # ("Organization", "Organization"): "strategic_partner", 
    # ("Person", "Person"):             "related_to", # Muted
    ("Organization", "Date"):         "founded_in",
    ("Event", "Date"):                "held_on",
    ("Product", "Date"):              "launched_on",
    ("Organization", "Money"):        "valued_at",
    ("Person", "Money"):              "income",
    ("Product", "Money"):             "priced_at",
    ("Event", "Money"):               "budget",
    ("Organization", "Percent"):      "growth_rate",
    ("Product", "Percent"):           "market_share",
    ("Organization", "Industry"):     "operates_in",
    ("Person", "Industry"):           "works_in",
}

_TYPE_PAIR_PRIORITY: dict[tuple[str, str], int] = {
    ("Person", "Organization"):       10,
    ("Organization", "Person"):       10,
    ("Organization", "Organization"): 9,
    ("Person", "Person"):             8,
    ("Organization", "Location"):     7,
    ("Person", "Location"):           7,
    ("Organization", "Product"):      6,
    ("Person", "Product"):            6,
    ("Organization", "Event"):        5,
    ("Person", "Event"):              5,
    ("Organization", "Money"):        4,
    ("Product", "Money"):             4,
    ("Organization", "Industry"):     3,
    ("Person", "Industry"):           3,
    ("Event", "Date"):                2,
    ("Organization", "Date"):         2,
    ("Person", "Date"):               1,
}

_NER_TO_GRAPH: dict[str, str] = {
    "PERSON": "Person", "ORGANIZATION": "Organization", "ORG": "Organization",
    "LOCATION": "Location", "LOC": "Location", "PRODUCT": "Product",
    "EVENT": "Event", "DATE": "Date", "MONEY": "Money",
    "PERCENT": "Percent", "INDUSTRY": "Industry",
}

_MAX_PAIRS_PER_SENTENCE = 12
_PROXIMITY_WINDOW       = 50


# ═══════════════════════════════════════════════════════════════════
# §2 — Relation Patterns (port từ knowledge_graph.ipynb, đã test)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RelationPattern:
    name         : str
    pattern      : re.Pattern
    subj_types   : Optional[set[str]]
    obj_types    : Optional[set[str]]
    confidence   : float
    bidirectional: bool = False
    reverse_edge : bool = False


# Shared building blocks
_VI_CAP = r"[A-Z\u00C0-\u024F\u1EA0-\u1EF9\u0110\u0111]"


# ORG/PERSON name: bắt đầu uppercase, gồm chữ cái + ký hiệu phổ biến
_NAME = _VI_CAP + r"(?:[\w&,\./]+" + r"(?:\s+" + _VI_CAP + r"[\w&,\./]+)*)"

# Stop-word lookahead: dừng trước từ chức năng để tránh capture quá dài
_STOP = (
    r"(?=\s*(?:[,\.\(\)]"
    r"|\s+(?:v\u00e0|ho\u1eb7c|m\u1ed9t|l\u00e0|c\u00f3"
    r"|trong|t\u1ea1i|v\u1edbi|t\u1eeb|tr\u01b0\u1edbc|sau"
    r"|\u0111\u1ebfn|n\u0103m|\u0111\u00e3|\u0111ang|\u0111\u01b0\u1ee3c)\b)"
    r"|$)"
)


RELATION_PATTERNS: list[RelationPattern] = [

    # ── Nhân sự ──────────────────────────────────────────────────
    RelationPattern("executive_of",
        re.compile(r"(?P<person>" + _NAME + r")\s*[,–\-]?\s*(?:Ch\u1ee7\s+t\u1ecbch|T\u1ed5ng\s+gi\u00e1m\s+\u0111\u1ed1c|Gi\u00e1m\s+\u0111\u1ed1c|CEO|CFO|CTO|COO|Ph\u00f3\s+ch\u1ee7\s+t\u1ecbch|Tr\u01b0\u1edfng\s+ban|Gi\u00e1m\s+\u0111\u1ed1c\s+\u0111i\u1ec1u\s+h\u00e0nh)\s+(?:c\u1ee7a\s+|t\u1ea1i\s+)?(?P<org>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.85),

    RelationPattern("former_employee",
        re.compile(r"(?P<person>" + _NAME + r")\s+(?:t\u1eebng\s+)?l\u00e0m\s+vi\u1ec7c\s+t\u1ea1i\s+(?P<org>" + _NAME + r")(?=\s+(?:t\u1eeb|trong|tr\u01b0\u1edbc|sau|\u0111\u1ebfn|,|\.|$))", re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.80),

    RelationPattern("founded_by",
        re.compile(r"(?P<person>" + _NAME + r")\s+(?:tr\u01b0\u1edbc\s+khi\s+)?(?:s\u00e1ng\s+l\u1eadp|\u0111\u1ed3ng\s+s\u00e1ng\s+l\u1eadp)\s+(?P<org>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.85, reverse_edge=True),

    RelationPattern("interviewed",
        re.compile(r"(?P<org>" + _NAME + r")\s+(?:ph\u1ecfng\s+v\u1ea5n|trao\s+\u0111\u1ed5i\s+v\u1edbi)\s+(?P<person>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"PERSON"}, confidence=0.75),

    # ORG được thành lập ... bởi PERSON — passive form
    RelationPattern("founded_by_passive",
        re.compile(
            r"(?P<org>" + _NAME + r")\s*(?:\([^)]+\))?\s*"
            r"\u0111\u01b0\u1ee3c\s+th\u00e0nh\s+l\u1eadp(?:[^b\n]{0,60}?)"
            r"b\u1edfi\s+(?P<person>" + _NAME + r")",
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"PERSON"}, confidence=0.88, reverse_edge=True),

    # ── Tổ chức × Tổ chức ────────────────────────────────────────
    # Ký thỏa thuận hợp tác chiến lược với — phải đặt trước strategic_partner
    RelationPattern("signed_strategic_partnership",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+k\u00fd\s+th\u1ecfa\s+thu\u1eadn\s+h\u1ee3p\s+t\u00e1c"
            r"(?:\s+chi\u1ebfn\s+l\u01b0\u1ee3c)?\s+v\u1edbi\s+(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.92),

    RelationPattern("strategic_partner",
        re.compile(r"(?P<org1>" + _NAME + r")\s+(?:h\u1ee3p\s+t\u00e1c\s+chi\u1ebfn\s+l\u01b0\u1ee3c|k\u00fd\s+k\u1ebft\s+chi\u1ebfn\s+l\u01b0\u1ee3c)\s+(?:c\u00f9ng\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.85, bidirectional=True),

    RelationPattern("partnered_with",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+l\u00e0\s+\u0111\u1ed1i\s+t\u00e1c\s+l\u00e2u\s+n\u0103m"
            r"\s+(?:c\u1ee7a\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.88),

    RelationPattern("long_term_partner",
        re.compile(r"(?P<org1>" + _NAME + r")\s+(?:l\u00e0\s+)?(?:đ\u1ed1i\s+t\u00e1c\s+l\u00e2u\s+d\u00e0i|\u0111\u1ed1i\s+t\u00e1c\s+chi\u1ebfn\s+l\u01b0\u1ee3c|h\u1ee3p\s+t\u00e1c\s+l\u00e2u\s+d\u00e0i)\s+(?:c\u1ee7a\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.85, bidirectional=True),

    RelationPattern("supplier_to",
        re.compile(r"(?P<org1>" + _NAME + r")\s+(?:chuy\u00ean\s+)?(?:cung\s+c\u1ea5p|l\u00e0\s+nh\u00e0\s+cung\s+c\u1ea5p)(?:\s+s\u1ea3n\s+ph\u1ea9m|\s+d\u1ecbch\s+v\u1ee5)?\s+(?:cho\s+)?(?P<org2>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.82),

    RelationPattern("competitor",
        re.compile(r"(?P<org1>" + _NAME + r")\s+(?:l\u00e0\s+)?(?:đ\u1ed1i\s+th\u1ee7|c\u1ea1nh\s+tranh|v\u01b0\u1ee3t\s+m\u1eb7t)(?:\s+tr\u1ef1c\s+ti\u1ebfp|\s+ch\u00ednh)?\s+(?:c\u1ee7a\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.82, bidirectional=True),

    RelationPattern("reported_on",
        re.compile(r"(?P<org1>" + _NAME + r")\s+(?:đ\u01b0a\s+tin|b\u00e1o\s+c\u00e1o|đ\u0103ng\s+tin)\s+(?:v\u1ec1\s+)?(?P<org2>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.80),

    RelationPattern("invested_in",
        re.compile(r"(?P<investor>" + _NAME + r")\s+(?:\u0111\u1ea7u\s+t\u01b0|r\u00f3t|g\u00f3p\s+v\u1ed1n|mua\s+l\u1ea1i|th\u00e2u\s+t\u00f3m)\s+(?:v\u00e0o\s+|cho\s+)?(?P<target>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION", "PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.82),

    # Ký hợp đồng cung cấp PRODUCT cho ORG
    RelationPattern("supplied_product_to",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+k\u00fd\s+h\u1ee3p\s+\u0111\u1ed3ng\s+cung\s+c\u1ea5p"
            r"(?:\s+" + _NAME + r")?\s+cho\s+(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.90),

    # ── Sản phẩm ─────────────────────────────────────────────────
    RelationPattern("developed",
        re.compile(r"(?P<company>" + _NAME + r")\s+(?:ra\s+m\u1eaft|s\u1ea3n\s+xu\u1ea5t|ph\u00e1t\s+tri\u1ec3n|tung\s+ra|gi\u1edbi\s+thi\u1ec7u|c\u00f4ng\s+b\u1ed1)\s+(?P<product>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"PRODUCT"}, confidence=0.78),

    RelationPattern("co_developed",
        re.compile(r"(?P<company>" + _NAME + r")\s+(?:c\u00f9ng\s+ph\u00e1t\s+tri\u1ec3n|\u0111\u1ed3ng\s+ph\u00e1t\s+tri\u1ec3n)\s+(?P<product>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"PRODUCT"}, confidence=0.85),

    # ORG s\u1eed d\u1ee5ng n\u1ec1n t\u1ea3ng NAME (b\u1eaft \u0111\u1ea7u b\u1eb1ng \"n\u1ec1n t\u1ea3ng\")
    RelationPattern("uses_platform",
        re.compile(
            r"(?P<subject>" + _NAME + r")\s+s\u1eed\s+d\u1ee5ng\s+(?:n\u1ec1n\s+t\u1ea3ng\s+)?(?P<product>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"PRODUCT", "ORGANIZATION"}, obj_types={"PRODUCT"}, confidence=0.82),

    # PRODUCT do (nh\u00f3m nghi\u00ean c\u1ee9u c\u1ee7a) ORG ph\u00e1t tri\u1ec3n
    RelationPattern("developed_by_passive",
        re.compile(
            r"(?P<product>" + _NAME + r")\s+do\s+(?:nh\u00f3m\s+nghi\u00ean\s+c\u1ee9u\s+c\u1ee7a\s+)?(?P<org>" + _NAME + r")\s+ph\u00e1t\s+tri\u1ec3n",
            re.UNICODE,
        ),
        subj_types={"PRODUCT"}, obj_types={"ORGANIZATION"}, confidence=0.88, reverse_edge=True),

    # VisionX \u0111\u01b0\u1ee3c x\u00e2y d\u1ef1ng d\u1ef1a tr\u00ean c\u00e1c nghi\u00ean c\u1ee9u h\u1ee3p t\u00e1c v\u1edbi \u0110\u1ea1i h\u1ecdc B\u00e1ch Khoa
    RelationPattern("based_on_research_with",
        re.compile(
            r"(?P<product>" + _NAME + r")\s+(?:\u0111\u01b0\u1ee3c\s+x\u00e2y\s+d\u1ef1ng\s+)?d\u1ef1a\s+tr\u00ean"
            r"\s+(?:c\u00e1c\s+)?nghi\u00ean\s+c\u1ee9u\s+h\u1ee3p\s+t\u00e1c\s+v\u1edbi\s+(?P<org>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"PRODUCT"}, obj_types={"ORGANIZATION"}, confidence=0.88),

    # ── Địa điểm ─────────────────────────────────────────────────
    RelationPattern("headquartered_in",
        re.compile(r"(?P<org>" + _NAME + r")\s+(?:\u0111\u1eb7t\s+t\u1ea1i|c\u00f3\s+tr\u1ee5\s+s\u1edf\s+t\u1ea1i|ho\u1ea1t\s+\u0111\u1ed9ng\s+t\u1ea1i|\u0111\u1eb7t\s+tr\u1ee5\s+s\u1edf)\s+(?P<loc>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.75),
        
    # Mở thêm văn phòng đại diện tại — fix đẻ cover cấu trúc thực tế
    RelationPattern("opened_office_in",
        re.compile(
            r"(?P<org>" + _NAME + r")\s+m\u1edf\s+(?:th\u00eam\s+)?(?:v\u0103n\s+ph\u00f2ng"
            r"(?:\s+\u0111\u1ea1i\s+di\u1ec7n)?|chi\s+nh\u00e1nh)\s+(?:t\u1ea1i\s+)?(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.85),

    # ── Ngành ────────────────────────────────────────────────────
    RelationPattern("operates_in",
        re.compile(r"(?P<org>" + _NAME + r")\s+(?:thu\u1ed9c\s+l\u0129nh\s+v\u1ef1c|ho\u1ea1t\s+\u0111\u1ed9ng\s+trong|chuy\u00ean\s+v\u1ec1|trong\s+ng\u00e0nh|l\u0129nh\s+v\u1ef1c|ng\u00e0nh)\s+(?P<industry>[\w\s]{2,40})" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"INDUSTRY"}, confidence=0.72),

    # ── Sự kiện ──────────────────────────────────────────────────
    RelationPattern("held_event",
        re.compile(r"(?P<org>" + _NAME + r")\s+(?:t\u1ed5\s+ch\u1ee9c|\u0111\u0103ng\s+cai|ch\u1ee7\s+tr\u00ec|\u0111\u1ee9ng\s+ra\s+t\u1ed5\s+ch\u1ee9c|kh\u1edfi\s+\u0111\u1ed9ng|ph\u00e1t\s+\u0111\u1ed9ng)\s+(?P<event>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"EVENT"}, confidence=0.78),

    # PRODUCT chính thức ra mắt ... tại sự kiện EVENT
    RelationPattern("launched_at_event",
        re.compile(
            r"(?P<product>" + _NAME + r")\s+(?:ch\u00ednh\s+th\u1ee9c\s+)?ra\s+m\u1eaft"
            r"(?:[^\n]{0,60}?)t\u1ea1i\s+(?:s\u1ef1\s+ki\u1ec7n\s+)?(?P<event>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"PRODUCT"}, obj_types={"EVENT"}, confidence=0.88),

    RelationPattern("held_in",
        re.compile(r"(?P<event>" + _NAME + r")\s+(?:di\u1ec5n\s+ra\s+t\u1ea1i|t\u1ed5\s+ch\u1ee9c\s+t\u1ea1i|x\u1ea3y\s+ra\s+t\u1ea1i|\u0111\u01b0\u1ee3c\s+t\u1ed5\s+ch\u1ee9c\s+t\u1ea1i|khai\s+m\u1ea1c\s+t\u1ea1i|di\u1ec5n\s+ra)\s+(?P<loc>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"EVENT"}, obj_types={"LOCATION"}, confidence=0.76),

    RelationPattern("participated_in",
        re.compile(r"(?P<person>" + _NAME + r")\s+(?:tham\s+gia|tham\s+d\u1ef1|xu\u1ea5t\s+hi\u1ec7n\s+t\u1ea1i|c\u00f3\s+m\u1eb7t\s+t\u1ea1i|ph\u00e1t\s+bi\u1ec3u\s+t\u1ea1i)\s+(?P<event>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"PERSON"}, obj_types={"EVENT"}, confidence=0.73),

    # ── Tài chính ────────────────────────────────────────────────
    RelationPattern("valued_at",
        re.compile(r"(?P<entity>" + _NAME + r")\s+(?:tr\u1ecb\s+gi\u00e1|c\u00f3\s+gi\u00e1\s+tr\u1ecb|\u0111\u1ea1t|gi\u00e1\s+tr\u1ecb\s+kho\u1ea3ng|\u01b0\u1edbc\s+t\u00ednh)\s+(?P<money>\d[\d.,]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn|ng\u00e0n)?\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\u0111\u00f4)?)", re.UNICODE),
        subj_types=None, obj_types=None, confidence=0.75),

    RelationPattern("growth_rate",
        re.compile(r"(?P<entity>" + _NAME + r")\s+(?:t\u0103ng\s+tr\u01b0\u1edfng|t\u0103ng|\u0111\u1ea1t|ghi\s+nh\u1eadn|t\u0103ng\s+l\u00ean)\s+(?P<pct>\d[\d.,]*\s*%(?:/n\u0103m)?)", re.UNICODE),
        subj_types=None, obj_types=None, confidence=0.70),
]


# ═══════════════════════════════════════════════════════════════════
# §3 — Entity matching (Jaccard-based)
# ═══════════════════════════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _best_entity_match(
    matched_text: str,
    entities: list[Entity],
    allowed_types: Optional[set[str]],
    threshold: float = 0.40,
) -> Optional[Entity]:
    ml = matched_text.lower().strip()
    if allowed_types:
        graph_allowed = {_NER_TO_GRAPH.get(t, t) for t in allowed_types}
        cands = [e for e in entities if e.type in graph_allowed]
    else:
        cands = list(entities)

    for e in cands:                          # exact
        if e.name.lower() == ml or any(a.lower() == ml for a in getattr(e, 'aliases', [])):
            return e
    for e in cands:                          # substring
        el = e.name.lower()
        if el in ml or ml in el or any(a.lower() in ml or ml in a.lower() for a in getattr(e, 'aliases', [])):
            return e
    best_e, best_s = None, 0.0              # Jaccard
    for e in cands:
        s = max([_jaccard(n, matched_text) for n in [e.name] + getattr(e, 'aliases', [])])
        if s > best_s:
            best_s, best_e = s, e
    return best_e if best_s >= threshold else None


# ═══════════════════════════════════════════════════════════════════
# §4 — Compiled regex cache (Tầng 3 co-occurrence)
# ═══════════════════════════════════════════════════════════════════

_search_cache: dict[str, re.Pattern] = {}


def _get_search_pat(name: str) -> re.Pattern:
    if name not in _search_cache:
        _search_cache[name] = re.compile(re.escape(name), re.IGNORECASE)
    return _search_cache[name]


def _find_pos(e: Entity, sent: str) -> int:
    best_pos = -1
    for n in [e.name] + getattr(e, 'aliases', []):
        m = _get_search_pat(n).search(sent)
        if m:
            p = m.start()
            if best_pos == -1 or p < best_pos:
                best_pos = p
    return best_pos


# ═══════════════════════════════════════════════════════════════════
# §5 — Entity building helpers
# ═══════════════════════════════════════════════════════════════════

def merge_adjacent_entities(raw_entities: list[dict]) -> list[dict]:
    if not raw_entities:
        return raw_entities
    merged: list[dict] = []
    cur = raw_entities[0].copy()
    for nxt in raw_entities[1:]:
        same_type = nxt["ner_type"] == cur["ner_type"]
        cur_last  = cur["words"][-1]  if cur.get("words")  else -99
        nxt_first = nxt["words"][0]   if nxt.get("words")  else -98
        if same_type and nxt_first == cur_last + 1:
            cur["text"] += " " + nxt["text"]
            cur["words"] = cur.get("words", []) + nxt.get("words", [])
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_informative(name: str, gtype: str) -> bool:
    norm = _norm(name)
    if len(norm) < 2 or re.fullmatch(r"[\W_]+", norm):
        return False
        
    # Loại rác viết thường (Proper nouns không được viết thường hoàn toàn)
    if gtype in ["Person", "Organization"] and norm.islower():
        return False
        
    low = norm.lower().replace(",", " ").replace(".", " ")
    toks = [t for t in low.split() if t]
    
    # Lọc rác một chữ chung chung cho Organization
    if gtype == "Organization":
        bad_org_words = {"công ty", "tập đoàn", "Tập đoàn công ty", "tổng công ty", "group", "holdings", "inc", "corp", "hiệp hội", "ngân hàng", "ban", "ngành", "phòng", "văn phòng", "chi nhánh", "trung tâm"}
        if low in bad_org_words:
            return False
        # Các cụm kiểu "Group phòng", "Group chính"
        if len(toks) <= 3 and all(t in {"group", "phòng", "ban", "chính", "nghiệp", "công"} for t in toks):
            return False
            
    # Lọc rác lỗi lặp từ PhoBERT
    if "chính chính" in low or "nghiên nghiên" in low:
        return False
    if gtype == "Date":
        if not any(c.isdigit() for c in norm):
            if len(toks) <= 1 or all(t in _GENERIC_DATE_WORDS for t in toks):
                return False
    if gtype == "Money":
        if not (any(c.isdigit() for c in norm)
                or any(k in low for k in ["usd", "vnd", "đ", "$", "triệu", "tỷ", "ty"])):
            return False
    if gtype == "Location":
        if len(toks) == 1 and len(toks[0]) <= 3 and not any(c.isdigit() for c in norm):
            return False
        if len(toks) == 2 and toks[0] in _LOCATION_PREFIXES:
            if not (kb.kb_ready and kb.get_entity_type(norm) == "LOCATION"):
                return False
    return True


def _is_noisy_date(e: Entity) -> bool:
    return e.type == "Date" and not any(c.isdigit() for c in e.name)


# ═══════════════════════════════════════════════════════════════════
# §6 — Dedup helpers
# ═══════════════════════════════════════════════════════════════════

def _pk(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _tk(s: str, t: str, lbl: str) -> tuple[str, str, str]:
    a, b = (s, t) if s <= t else (t, s)
    return (a, b, lbl)


# ═══════════════════════════════════════════════════════════════════
# §7 — Tầng 1: Pattern-based
# ═══════════════════════════════════════════════════════════════════

def _extract_pattern_relations(
    entities: list[Entity],
    sentences: list[str],
) -> tuple[list[Relation], set[tuple[str, str]]]:
    relations: list[Relation] = []
    seen_tk:   set[tuple[str, str, str]] = set()
    seen_pk:   set[tuple[str, str]]      = set()

    for sent in sentences:
        for pat in RELATION_PATTERNS:
            for m in pat.pattern.finditer(sent):
                try:
                    st = m.group(1).strip()
                    ot = m.group(2).strip()
                except (IndexError, AttributeError):
                    continue
                if not st or not ot or st == ot:
                    continue

                se = _best_entity_match(st, entities, pat.subj_types)
                oe = _best_entity_match(ot, entities, pat.obj_types)

                

                if se is None and pat.subj_types is not None:
                    continue
                if oe is None and pat.obj_types is not None:
                    continue
                if se is None or oe is None or se.id == oe.id:
                    continue
                    
                if pat.reverse_edge:
                    se, oe = oe, se

                lbl = pat.name
                tk  = _tk(se.id, oe.id, lbl)
                if tk in seen_tk:
                    continue
                seen_tk.add(tk)
                seen_pk.add(_pk(se.id, oe.id))
                relations.append(Relation(source=se.id, target=oe.id, label=lbl))

                if pat.bidirectional:
                    tkr = _tk(oe.id, se.id, lbl)
                    if tkr not in seen_tk:
                        seen_tk.add(tkr)
                        relations.append(Relation(source=oe.id, target=se.id, label=lbl))

    return relations, seen_pk


# ═══════════════════════════════════════════════════════════════════
# §8 — Tầng 3: Co-occurrence scored
# ═══════════════════════════════════════════════════════════════════

@dataclass(order=True)
class _SP:
    score: float
    src:   Entity = field(compare=False)
    tgt:   Entity = field(compare=False)
    lbl:   str    = field(compare=False)


def _fallback_label(src: Entity, tgt: Entity) -> Optional[str]:
    if kb.kb_ready:
        lbl = kb.find_relation(src.name, tgt.name)
        if lbl:
            return lbl
    return _RELATION_LABELS_VI.get((src.type, tgt.type))


def _score(src: Entity, tgt: Entity, sp: int, tp: int) -> float:
    type_s = _TYPE_PAIR_PRIORITY.get(
        (src.type, tgt.type),
        _TYPE_PAIR_PRIORITY.get((tgt.type, src.type), 0),
    )
    prox = (5.0 * math.exp(-abs(sp - tp) / (_PROXIMITY_WINDOW * 6))
            if sp >= 0 and tp >= 0 else 0.0)
    kb_b = 3.0 if kb.kb_ready else 0.0
    return type_s + prox + kb_b


def _extract_cooccurrence_relations(
    entities: list[Entity],
    sentences: list[str],
    existing_pk: set[tuple[str, str]],
) -> list[Relation]:
    relations: list[Relation] = []
    seen_pk:   set[tuple[str, str]]      = set(existing_pk)
    seen_tk:   set[tuple[str, str, str]] = set()

    for sent in sentences:
        ents = [(e, _find_pos(e, sent))
                for e in entities
                if not _is_noisy_date(e) and _find_pos(e, sent) >= 0]
        if len(ents) < 2:
            continue

        scored: list[_SP] = []
        for i, (se, sp) in enumerate(ents):
            for te, tp in ents[i + 1:]:
                pk = _pk(se.id, te.id)
                if pk in seen_pk:
                    continue
                lbl = _fallback_label(se, te)
                if lbl:
                    scored.append(_SP(_score(se, te, sp, tp), se, te, lbl))

        scored.sort(key=lambda p: p.score, reverse=True)
        for sp in scored[:_MAX_PAIRS_PER_SENTENCE]:
            pk = _pk(sp.src.id, sp.tgt.id)
            tk = _tk(sp.src.id, sp.tgt.id, sp.lbl)
            if pk in seen_pk or tk in seen_tk:
                continue
            seen_pk.add(pk)
            seen_tk.add(tk)
            relations.append(Relation(source=sp.src.id, target=sp.tgt.id, label=sp.lbl))

    return relations


# ═══════════════════════════════════════════════════════════════════
# §9 — Graph Structuring (Alias & Coreference)
# ═══════════════════════════════════════════════════════════════════

def _is_camelcase_of(alias: str, primary: str) -> bool:
    """Kiểm tra alias có phải là CamelCase viết tắt của primary không.
    VD: SaoVietTech → 'sao', 'viet', 'tech' đều có trong 'Công nghệ Sao Việt'."""
    if not re.search(r'[a-z][A-Z]', alias):
        return False
    camel_parts = [p.lower() for p in re.findall(r'[A-Z]?[a-z]+|[A-Z]+', alias)]
    if len(camel_parts) < 2:
        return False
    import unicodedata
    p_unaccent = unicodedata.normalize('NFD', primary.lower()).encode('ascii', 'ignore').decode('utf-8')
    matches = sum(1 for p in camel_parts if p in primary.lower() or p in p_unaccent)
    return matches >= len(camel_parts)


def _resolve_aliases(entities: list[Entity], text: str) -> list[Entity]:
    alias_map = {}
    
    # 1. Ngoặc đơn: "Công ty Cổ phần Công nghệ Sao Việt (SaoVietTech)"
    for m in re.finditer(r"([A-Z\u00C0-\u1EF9][\w\s,\u00C0-\u1EF9&]+)\s*\(\s*([^)]+)\s*\)", text):
        full = _norm(m.group(1))
        al = _norm(m.group(2))
        if 2 <= len(al) < len(full):
            alias_map[al.lower()] = full.lower()

    # 2. Org-org: CHỈ merge qua CamelCase rõ ràng (không dùng word-subset vì merge sai)
    orgs      = [e for e in entities if e.type == "Organization"]
    org_texts = sorted([e.name for e in orgs], key=len, reverse=True)
    for i, primary in enumerate(org_texts):
        for alias in org_texts[i + 1:]:
            a_low = alias.lower()
            if a_low not in alias_map and _is_camelcase_of(alias, primary):
                alias_map[a_low] = primary.lower()

    # 3. Xử lý Person
    pers = [e for e in entities if e.type == "Person"]
    per_texts = sorted([e.name for e in pers], key=len, reverse=True)
    for i, primary in enumerate(per_texts):
        p_low = primary.lower()
        if len(p_low.split()) >= 2:
            for alias in per_texts[i+1:]:
                a_low = alias.lower()
                if a_low in p_low:
                     alias_map[a_low] = p_low

    # Thống nhất node (cycle-safe)
    merged: dict[str, Entity] = {}

    def get_root(name_lower: str) -> str:
        curr, visited = name_lower, set()
        while curr in alias_map and alias_map[curr] != curr:
            if curr in visited:
                break
            visited.add(curr)
            curr = alias_map[curr]
        return curr

    for e in entities:
        root_low = get_root(e.name.lower())
        if root_low not in merged:
            merged[root_low] = e
        else:
            primary_e = merged[root_low]
            if e.name.lower() != primary_e.name.lower() and e.name not in primary_e.aliases:
                # Chỉ add alias nếu nó được phát hiện qua parenthetical/camelcase
                if e.name.lower() in alias_map:
                    primary_e.aliases.append(e.name)
                    p_name = primary_e.name
                    if "(" not in p_name and len(e.name) <= len(p_name) - 5 and len(e.name.split()) <= 3:
                        primary_e.name = f"{p_name} ({e.name})"
            for p in e.properties:
                if not any(mp.key == p.key and mp.value == p.value for mp in primary_e.properties):
                    primary_e.properties.append(p)

    return list(merged.values())


def _fold_properties(entities: list[Entity], relations: list[Relation]) -> tuple[list[Entity], list[Relation]]:
    kept_types  = {"Organization", "Person", "Product", "Event", "Location"}
    final_entities = {e.id: e for e in entities}
    final_relations: list[Relation] = []

    # Bảng map label → property key rõ ràng
    _LABEL_TO_PROP: dict[str, str] = {
        "founded_in":              "Founded",
        "headquartered_in":        "Headquarters",
        "located_in":              "Headquarters",
        "valued_at":               "Value",
        "priced_at":               "Value",
        "income":                  "Income",
        "growth_rate":             "Growth",
        "held_on":                 "Date",
        "launched_on":             "Launch Date",
        "held_in":                 "Location",
        "operates_in":             "Industry",
        "works_in":                "Industry",
    }

    for r in relations:
        src = final_entities.get(r.source)
        tgt = final_entities.get(r.target)
        if not src or not tgt:
            continue

        if tgt.type not in kept_types:
            prop_key = _LABEL_TO_PROP.get(r.label, r.label.replace("_", " ").title())
            src.properties.append(EntityProperty(key=prop_key, value=tgt.name))
        elif src.type not in kept_types:
            tgt.properties.append(EntityProperty(key=f"Has {src.type}", value=src.name))
        else:
            final_relations.append(r)

    filtered_entities = [e for e in final_entities.values() if e.type in kept_types]

    for e in filtered_entities:
        seen_props: set[tuple] = set()
        new_props: list[EntityProperty] = []
        for p in e.properties:
            if p.key == "NER Type":
                continue
            k = (p.key, p.value)
            if k not in seen_props:
                seen_props.add(k)
                new_props.append(p)
        e.properties = new_props

    return filtered_entities, final_relations


# ═══════════════════════════════════════════════════════════════════
# §9.5 — Multi-entity pattern extractor
# Xử lý các cấu trúc danh sách: 'bởi A và B', 'như A và B'
# ═══════════════════════════════════════════════════════════════════

def _extract_multi_entity_relations(
    entities: list[Entity],
    text: str,
    existing_pk: set[tuple[str, str]],
) -> list[Relation]:
    """
    Xử lý các cấu trúc 1-to-many khó dùng regex 2-group thông thưởng:
      1. 'ORG được thành lập ... bởi PERSON1 và PERSON2'
         → 2 relations founded (PERSON1,2 → ORG)
      2. '... cạnh tranh với các đối thủ như ORG1 và ORG2'
         → 2 relations competitor_of (ORG_main → ORG1,2)
    """
    rels: list[Relation] = []

    # -- (1) Multi-founder: 'bởi PERSON1 và PERSON2'
    persons = [e for e in entities if e.type == "Person"]
    orgs    = [e for e in entities if e.type == "Organization"]
    for m in re.finditer(
        r"(?P<org>[A-Z\u00C0-\u1EF9][\w\s,\u00C0-\u1EF9&]+?)"
        r"\s*(?:\([^)]+\))?\s*"
        r"\u0111\u01b0\u1ee3c\s+th\u00e0nh\s+l\u1eadp(?:[^b\n]{0,80}?)"
        r"b\u1edfi\s+(?P<p1>[A-Z\u00C0-\u1EF9][\w\s\u00C0-\u1EF9]+?)"
        r"\s+v\u00e0\s+(?P<p2>[A-Z\u00C0-\u1EF9][\w\s\u00C0-\u1EF9]+?)"
        r"(?=[,\.\n]|$)",
        text,
    ):
        org_e  = _best_entity_match(m.group("org").strip(), orgs, {"ORGANIZATION"})
        per1_e = _best_entity_match(m.group("p1").strip(), persons, {"PERSON"})
        per2_e = _best_entity_match(m.group("p2").strip(), persons, {"PERSON"})
        for pe in [per1_e, per2_e]:
            if pe and org_e and pe.id != org_e.id:
                pk = _pk(pe.id, org_e.id)
                if pk not in existing_pk:
                    existing_pk.add(pk)
                    rels.append(Relation(source=pe.id, target=org_e.id, label="founded"))

    # -- (2) Competitor list: 'như ORG1 và ORG2'
    # Tìm câu chứa 'cạnh tranh ... như' và extract tất cả ORG được mention sau 'như'
    for m in re.finditer(
        r"(?P<main>[A-Z\u00C0-\u1EF9][\w\s\u00C0-\u1EF9&]+?)"
        r"(?:\s+m\u1edf\s+r\u1ed9ng)?\s+c\u1ee7ng\s+c\u1ed1\s+v\u1ecb\s+th\u1ebf\s+c\u1ea1nh\s+tranh"
        r"[^n\n]{0,60}?nh\u01b0\s+(?P<oc>[A-Z\u00C0-\u1EF9][\w\s,\u00C0-\u1EF9&]+?)(?=[,\.\n]|$)",
        text,
    ):
        main_e = _best_entity_match(m.group("main").strip(), orgs, {"ORGANIZATION"})
        if not main_e:
            continue
        # Tách danh sách '... và ...'
        parts = re.split(r'\s+và\s+|,\s*', m.group("oc"))
        for part in parts:
            competitor_e = _best_entity_match(part.strip(), orgs, {"ORGANIZATION"})
            if competitor_e and competitor_e.id != main_e.id:
                pk = _pk(main_e.id, competitor_e.id)
                if pk not in existing_pk:
                    existing_pk.add(pk)
                    rels.append(Relation(source=main_e.id, target=competitor_e.id, label="competitor_of"))

    return rels



# ═══════════════════════════════════════════════════════════════════
# §10 — build_graph (entry point)
# ═══════════════════════════════════════════════════════════════════

def build_graph(raw_entities: list[dict], text: str) -> GraphData:
    """
    3-tầng relation extraction:
      Tầng 1: Pattern regex (RELATION_PATTERNS) → chính xác nhất
      Tầng 2: KB enrich                          → triples corpus
      Tầng 3: Co-occurrence scored               → fallback
    """
    raw_entities = merge_adjacent_entities(raw_entities)
    sentences    = split_sentences(text)

    # ── Build entities ─────────────────────────────────────────
    seen:     dict[tuple, str] = {}
    entities: list[Entity]     = []

    for raw in raw_entities:
        gtype = TYPE_MAP.get(raw["ner_type"])
        if gtype is None:
            continue
        name = _norm(raw["text"].strip().strip(".,;:!?\"'"))
        if not _is_informative(name, gtype):
            continue
        if kb.kb_ready and len(name) >= 3:
            kbt = TYPE_MAP.get(kb.get_entity_type(name))
            if kbt:
                gtype = kbt
        key = (name.lower(), gtype)
        if key in seen:
            continue
        eid = f"E{len(entities) + 1}"
        seen[key] = eid
        entities.append(Entity(
            id=eid, name=name, type=gtype,
            properties=[EntityProperty(key="NER Type", value=raw["ner_type"])],
            aliases=[],
        ))

    if not entities:
        return GraphData(entities=[], relations=[])

    # ── Gộp Aliases Thực thể ───────────────────────────────────
    entities = _resolve_aliases(entities, text)

    # ── Tầng 1 ─────────────────────────────────────────────────
    pat_rels, existing_pk = _extract_pattern_relations(entities, sentences)
    logger.debug("Tầng 1 (pattern): %d", len(pat_rels))

    # ── Tầng 2 ─────────────────────────────────────────────────
    kb_rels: list[Relation] = []
    if kb.kb_ready and len(entities) > 1:
        for rd in kb.enrich_relations(entities, existing_pk, max_per_entity=2):
            kb_rels.append(Relation(source=rd["source"], target=rd["target"], label=rd["label"]))
            existing_pk.add(_pk(rd["source"], rd["target"]))
    logger.debug("Tầng 2 (KB): %d", len(kb_rels))

    # ── Tầng 3 ─────────────────────────────────────────────────
    cooc_rels = _extract_cooccurrence_relations(entities, sentences, existing_pk)
    logger.debug("Tầng 3 (co-occ): %d", len(cooc_rels))

    # ── Tầng 1.5 — Multi-entity patterns (bởi A và B, như A và B) ─────
    multi_rels = _extract_multi_entity_relations(entities, text, existing_pk)
    logger.debug("Tầng 1.5 (multi-entity): %d", len(multi_rels))

    all_rels = pat_rels + multi_rels + kb_rels + cooc_rels

    
    # ── Fold thuộc tính vào Nodes ──────────────────────────────
    final_entities, final_relations = _fold_properties(entities, all_rels)
    
    logger.info(
        "build_graph: %d entities, %d relations (P=%d KB=%d C=%d)",
        len(final_entities), len(final_relations), len(pat_rels), len(kb_rels), len(cooc_rels),
    )
    return GraphData(entities=final_entities, relations=final_relations)


# ═══════════════════════════════════════════════════════════════════
# §10 — predict_new_links
# ═══════════════════════════════════════════════════════════════════

def predict_new_links(
    entities: list[Entity],
    relations: list[Relation],
) -> list[Relation]:
    existing  = {_pk(r.source, r.target) for r in relations}
    predicted: list[Relation] = []

    candidates = sorted(
        [
            (_TYPE_PAIR_PRIORITY.get(
                (s.type, t.type),
                _TYPE_PAIR_PRIORITY.get((t.type, s.type), 0),
             ), s, t)
            for i, s in enumerate(entities)
            for j, t in enumerate(entities)
            if i < j and _pk(s.id, t.id) not in existing
        ],
        key=lambda x: -x[0],
    )

    for _, src, tgt in candidates:
        lbl: Optional[str] = None
        if kb.kb_ready:
            lbl = kb.find_relation(src.name, tgt.name)
        if not lbl:
            lbl = _RELATION_LABELS_VI.get((src.type, tgt.type))
        if lbl:
            predicted.append(Relation(
                source=src.id, target=tgt.id,
                label=lbl, isPredicted=True,
            ))
        if len(predicted) >= 5:
            break

    return predicted
