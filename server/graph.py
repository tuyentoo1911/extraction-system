"""Xây dựng Knowledge Graph từ NER output + Knowledge Base.

Pipeline trích xuất quan hệ — strict whitelist:
────────────────────────────────────────────────────────
  ALLOWED_RELATIONS (9):
      founded, headquartered_in, has_office, partnered_with,
      competitor_of, former_employee, launched_at, developed_by,
      operates_in

  Tầng 1 — Pattern-based (regex + Jaccard entity matching)
      Chỉ giữ pattern map vào whitelist.

  Tầng 1.5 — Multi-entity patterns (bởi A và B, như A và B)

  Tầng 2 — KB lookup (same-sentence evidence only)
      Chỉ thêm edge khi 2 entity cùng câu VÀ KB label map vào whitelist.

  Output guard: drop mọi relation ngoài whitelist + domain/range check.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from constants import TYPE_MAP
from schemas import Entity, EntityProperty, Relation, GraphData
from ner import split_sentences
import knowledge_base as kb

logger = logging.getLogger(__name__)

ALLOWED_RELATIONS: frozenset[str] = frozenset({
    "founded", "headquartered_in", "has_office", "partnered_with",
    "competitor_of", "former_employee", "launched_at", "developed_by",
    "operates_in",

    "occurred_in",      # Event → Location  (xảy ra tại X)
    "has_revenue",      # Organization        → Money (báo cáo doanh thu X)
    "has_value",        # Organization/Product→ Money (giá trị X, ký hợp đồng X)
    "held_in",          # Event               → Location (tổ chức tại X)
})

_RELATION_NORM_MAP: dict[str, str | None] = {
    "founded_by":                  "founded",
    "founded_by_passive":          "founded",
    "founded":                     "founded",
    "headquartered_in":            "headquartered_in",
    "opened_office_in":            "has_office",
    "has_office":                  "has_office",
    "signed_strategic_partnership": "partnered_with",
    "strategic_partner":           "partnered_with",
    "partnered_with":              "partnered_with",
    "long_term_partner":           "partnered_with",
    "competitor":                  "competitor_of",
    "competitor_of":               "competitor_of",
    "former_employee":             "former_employee",
    "launched_at_event":           "launched_at",
    "launched_at":                 "launched_at",
    "developed":                   "developed_by",
    "co_developed":                "developed_by",
    "developed_by_passive":        "developed_by",
    "developed_by":                "developed_by",
    "operates_in":                 "operates_in",

    "occurred_in":                 "occurred_in",
    "has_revenue":                 "has_revenue",
    "has_value":                   "has_value",
    "held_in":                     "held_in",
    "HỢP TÁC":                    "partnered_with",
    "ĐẶT TẠI":                    "headquartered_in",
    "SẢN XUẤT":                    "developed_by",
    "THUỘC NGÀNH":                 "operates_in",
    "XẢY RA TẠI":                  None,
    "LÃNH ĐẠO":                   None,
    "LIÊN QUAN":                   None,
    "CÓ GIÁ TRỊ":                None,
    "TĂNG TRƯỞNG":                None,
    "TỔ CHỨC SỰ KIỆN":            None,
    "THAM GIA":                    None,
    "ĐẦU TƯ":                     None,
}

_RELATION_DOMAIN_RANGE: dict[str, list[tuple[set[str], set[str]]]] = {
    "founded":          [({"Person", "Organization"}, {"Organization"})],
    "headquartered_in": [({"Organization"},           {"Location"})],
    "has_office":       [({"Organization"},           {"Location"})],
    "partnered_with":   [({"Organization"},           {"Organization"})],
    "competitor_of":    [({"Organization"},           {"Organization"})],
    "former_employee":  [({"Person"},                 {"Organization"})],
    "launched_at":      [({"Product"},                {"Event"})],
    "developed_by":     [({"Product"},                {"Organization", "Person"})],
    "operates_in":      [({"Organization"},           {"Industry"})],

    "occurred_in":      [({"Event"}, {"Location"})],
    "has_revenue":      [({"Organization"},           {"Money"})],
    "has_value":        [({"Organization", "Product"}, {"Money"})],
    "held_in":          [({"Event"},                  {"Location"})],
}

_NEWS_SOURCES: frozenset[str] = frozenset({
    "bloomberg", "reuters", "cnn", "cnbc", "bbc", "ap", "afp",
    "vnexpress", "tuổi trẻ", "thanh niên", "dân trí", "vietnamnet",
    "cafef", "the wall street journal", "financial times", "nikkei",
    "vietnam business review", "techcrunch", "the verge", "wired",
    "forbes", "business insider", "the guardian", "new york times",
})

_GENERIC_DATE_WORDS = {
    "nam", "năm", "thang", "tháng", "ngay", "ngày",
    "quy", "quý", "tuan", "tuần",
}

_GENERIC_INDUSTRY_WORDS: frozenset[str] = frozenset({
    "xây dựng", "tài chính", "công nghệ", "quản lý", "phát triển",
    "kinh doanh", "sản xuất", "nghiên cứu", "giáo dục", "đào tạo",
    "thiết kế", "vận hành", "bảo trì", "quảng cáo", "truyền thông",
    "marketing", "bán hàng", "dịch vụ", "thương mại", "kỹ thuật",
})
_LOCATION_PREFIXES = {
    "tp", "tp.", "thành", "thanh", "quận", "quan",
    "huyện", "huyen", "tỉnh", "tinh",
    "phường", "phuong", "xã", "xa", "thị", "thi",
}

_NER_TO_GRAPH: dict[str, str] = {
    "PERSON": "Person", "ORGANIZATION": "Organization", "ORG": "Organization",
    "LOCATION": "Location", "LOC": "Location", "PRODUCT": "Product",
    "EVENT": "Event", "DATE": "Date", "MONEY": "Money",
    "PERCENT": "Percent", "INDUSTRY": "Industry",
}

def _normalize_label(raw_label: str) -> str | None:
    """Map internal/KB label to whitelist. Returns None if not in whitelist."""
    norm = _RELATION_NORM_MAP.get(raw_label)
    if norm is not None:
        return norm
    return raw_label if raw_label in ALLOWED_RELATIONS else None

def _validate_domain_range(label: str, src_type: str, tgt_type: str) -> bool:
    """Check entity types match the relation's domain/range constraint."""
    constraints = _RELATION_DOMAIN_RANGE.get(label)
    if not constraints:
        return False
    return any(src_type in dom and tgt_type in rng for dom, rng in constraints)

def _is_news_source(name: str) -> bool:
    return name.lower().strip() in _NEWS_SOURCES

@dataclass
class RelationPattern:
    name         : str
    pattern      : re.Pattern
    subj_types   : Optional[set[str]]
    obj_types    : Optional[set[str]]
    confidence   : float
    bidirectional: bool = False
    reverse_edge : bool = False

_VI_CAP = r"[A-Z\u00C0-\u024F\u1EA0-\u1EF9\u0110\u0111]"

_VI_LOWER_CONT = (
    r"(?:ty|cổ|ph\u1ea7n|h\u1eefu|tr\u00e1ch|nhi\u1ec7m|tnhh|jsc|ltd|corp|co"
    r"|vi\u1ec7t|nam|nh\u1eadt|h\u00e0n|anh|ph\u00e1p|\u0111\u1ee9c|m\u1ef9"
    r"|qu\u1ed1c|t\u1ebf|b\u1eafc|trung|\u0111\u00f4ng|t\u00e2y|b\u1ea3n|m\u1edbi)"
)

_NAME = (
    _VI_CAP
    + r"(?:[\w&,\./]+"
    + r"(?:\s+(?:" + _VI_CAP + r"|\d|" + _VI_LOWER_CONT + r")[\w&,\./]+)*)"
)

_STOP = r"(?=\s*(?:[,\.\(\);:\-]|\s+[a-z\u00e0-\u01b0\u1ea1-\u1ef9])|$)"

RELATION_PATTERNS: list[RelationPattern] = [

    RelationPattern("founded_by",
        re.compile(
            r"(?P<person>" + _NAME + r")\s+"
            r"(?:tr\u01b0\u1edbc\s+khi\s+)?"
            r"(?:s\u00e1ng\s+l\u1eadp|\u0111\u1ed3ng\s+s\u00e1ng\s+l\u1eadp|th\u00e0nh\s+l\u1eadp|s\u00e1ng\s+ki\u1ebfn|kh\u1edfi\s+x\u01b0\u1edbng)\s+"
            r"(?P<org>" + _NAME + r")" + _STOP, re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.85),

    RelationPattern("founded_by_passive",
        re.compile(
            r"(?P<org>" + _NAME + r")\s*(?:\([^)]+\))?\s*"
            r"\u0111\u01b0\u1ee3c\s+(?:th\u00e0nh\s+l\u1eadp|s\u00e1ng\s+l\u1eadp|x\u00e2y\s+d\u1ef1ng)"
            r"(?:[^b\n]{0,60}?)b\u1edfi\s+(?P<person>" + _NAME + r")",
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"PERSON"}, confidence=0.88, reverse_edge=True),

    RelationPattern("former_employee",
        re.compile(
            r"(?P<person>" + _NAME + r")\s+"
            r"(?:t\u1eebng\s+)?(?:l\u00e0m\s+vi\u1ec7c|c\u00f4ng\s+t\u00e1c|l\u00e0\s+nh\u00e2n\s+vi\u00ean|gi\u1eef\s+ch\u1ee9c|l\u00e0\s+gi\u00e1m\s+\u0111\u1ed1c)"
            r"\s+(?:t\u1ea1i\s+)?(?P<org>" + _NAME + r")"
            r"(?=\s+(?:t\u1eeb|trong|tr\u01b0\u1edbc|sau|\u0111\u1ebfn|,|\.|$))",
            re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.80),

    RelationPattern("former_employee",
        re.compile(
            r"(?P<person>" + _NAME + r")\s+"
            r"l\u00e0\s+c\u1ef1u\s+(?:gi\u00e1m\s+\u0111\u1ed1c|c\u1ed1\u0111\u00f4ng|nh\u00e2n\s+vi\u00ean|l\u00e3nh\s+\u0111\u1ea1o)?"
            r"\s*(?:c\u1ee7a\s+)?(?P<org>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.82),

    RelationPattern("signed_strategic_partnership",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+"
            r"k\u00fd\s+(?:th\u1ecfa\s+thu\u1eadn|h\u1ee3p\s+\u0111\u1ed3ng)\s+"
            r"(?:h\u1ee3p\s+t\u00e1c\s+)?(?:chi\u1ebfn\s+l\u01b0\u1ee3c\s+)?"
            r"v\u1edbi\s+(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.92, bidirectional=True),

    RelationPattern("strategic_partner",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+"
            r"(?:h\u1ee3p\s+t\u00e1c\s+chi\u1ebfn\s+l\u01b0\u1ee3c|k\u00fd\s+k\u1ebft\s+chi\u1ebfn\s+l\u01b0\u1ee3c|li\u00ean\s+k\u1ebft\s+chi\u1ebfn\s+l\u01b0\u1ee3c)"
            r"\s+(?:c\u00f9ng\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.85, bidirectional=True),

    RelationPattern("partnered_with",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+l\u00e0\s+\u0111\u1ed1i\s+t\u00e1c"
            r"\s+(?:l\u00e2u\s+(?:n\u0103m|d\u00e0i)|chi\u1ebfn\s+l\u01b0\u1ee3c|tin\s+c\u1eady)"
            r"\s+(?:c\u1ee7a\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.88, bidirectional=True),

    RelationPattern("partnered_with",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+"
            r"(?:h\u1ee3p\s+t\u00e1c|li\u00ean\s+doanh|li\u00ean\s+k\u1ebft|li\u00ean\s+minh)"
            r"\s+v\u1edbi\s+(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.72, bidirectional=True),

    RelationPattern("competitor",
        re.compile(
            r"(?P<org1>" + _NAME + r")\s+"
            r"(?:l\u00e0\s+)?"
            r"(?:\u0111\u1ed1i\s+th\u1ee7(?:\s+c\u1ea1nh\s+tranh)?|c\u1ea1nh\s+tranh|v\u01b0\u1ee3t\s+m\u1eb7t)"
            r"(?:\s+tr\u1ef1c\s+ti\u1ebfp|\s+ch\u00ednh)?\s+"
            r"(?:c\u1ee7a\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.82, bidirectional=True),

    RelationPattern("developed",
        re.compile(
            r"(?P<company>" + _NAME + r")\s+"
            r"(?:ra\s+m\u1eaft|s\u1ea3n\s+xu\u1ea5t|ph\u00e1t\s+tri\u1ec3n|tung\s+ra"
            r"|gi\u1edbi\s+thi\u1ec7u|c\u00f4ng\s+b\u1ed1|ch\u1ebf\s+t\u1ea1o|x\u00e2y\s+d\u1ef1ng)\s+"
            r"(?P<product>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"PRODUCT"}, confidence=0.78, reverse_edge=True),

    RelationPattern("co_developed",
        re.compile(
            r"(?P<company>" + _NAME + r")\s+"
            r"(?:c\u00f9ng\s+ph\u00e1t\s+tri\u1ec3n|\u0111\u1ed3ng\s+ph\u00e1t\s+tri\u1ec3n|c\u00f9ng\s+x\u00e2y\s+d\u1ef1ng)"
            r"\s+(?P<product>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"PRODUCT"}, confidence=0.85, reverse_edge=True),

    RelationPattern("developed_by_passive",
        re.compile(
            r"(?P<product>" + _NAME + r")\s+"
            r"(?:do|b\u1edfi)\s+(?:nh\u00f3m\s+nghi\u00ean\s+c\u1ee9u\s+c\u1ee7a\s+)?"
            r"(?P<org>" + _NAME + r")\s+"
            r"(?:ph\u00e1t\s+tri\u1ec3n|x\u00e2y\s+d\u1ef1ng|s\u1ea3n\s+xu\u1ea5t|ch\u1ebf\s+t\u1ea1o)",
            re.UNICODE,
        ),
        subj_types={"PRODUCT"}, obj_types={"ORGANIZATION"}, confidence=0.88),

    RelationPattern("developed_by_passive",
        re.compile(
            r"(?P<product>" + _NAME + r")\s+"
            r"\u0111\u01b0\u1ee3c\s+(?P<org>" + _NAME + r")\s+"
            r"(?:ph\u00e1t\s+tri\u1ec3n|x\u00e2y\s+d\u1ef1ng|s\u1ea3n\s+xu\u1ea5t|thi\u1ebft\s+k\u1ebf)",
            re.UNICODE,
        ),
        subj_types={"PRODUCT"}, obj_types={"ORGANIZATION"}, confidence=0.85),

    RelationPattern("headquartered_in",
        re.compile(
            r"(?P<org>" + _NAME + r")\s+"
            r"(?:\u0111\u1eb7t\s+t\u1ea1i|c\u00f3\s+tr\u1ee5\s+s\u1edf\s+t\u1ea1i"
            r"|t\u1ecda\s+l\u1ea1c\s+t\u1ea1i|c\u00f3\s+\u0111\u1ecba\s+ch\u1ec9\s+t\u1ea1i"
            r"|c\u00f3\s+v\u0103n\s+ph\u00f2ng\s+ch\u00ednh\s+t\u1ea1i"
            r"|\u0111\u1eb7t\s+tr\u1ee5\s+s\u1edf(?:\s+t\u1ea1i)?)\s+"
            r"(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.75),

    RelationPattern("headquartered_in",
        re.compile(
            r"[Tt]r\u1ee5\s+s\u1edf\s+(?:ch\u00ednh\s+)?c\u1ee7a\s+(?:c\u00f4ng\s+ty\s+)?(?P<org>" + _NAME + r")"
            r"\s+(?:\u0111\u1eb7t\s+)?t\u1ea1i\s+(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.88),

    RelationPattern("opened_office_in",
        re.compile(
            r"(?P<org>" + _NAME + r")\s+"
            r"m\u1edf\s+(?:th\u00eam\s+)?"
            r"(?:v\u0103n\s+ph\u00f2ng(?:\s+\u0111\u1ea1i\s+di\u1ec7n)?|chi\s+nh\u00e1nh|v\u0103n\s+ph\u00f2ng\s+\u0111\u1ea1i\s+di\u1ec7n)"
            r"\s+(?:t\u1ea1i\s+)?(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.85),

    RelationPattern("has_office",
        re.compile(
            r"(?P<org>" + _NAME + r")\s+"
            r"(?:c\u00f3|thi\u1ebft\s+l\u1eadp)\s+"
            r"(?:v\u0103n\s+ph\u00f2ng(?:\s+\u0111\u1ea1i\s+di\u1ec7n)?|chi\s+nh\u00e1nh)\s+"
            r"(?:t\u1ea1i\s+)?(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.82),

    RelationPattern("operates_in",
        re.compile(
            r"(?P<org>" + _NAME + r")\s+"
            r"(?:thu\u1ed9c\s+l\u0129nh\s+v\u1ef1c|ho\u1ea1t\s+\u0111\u1ed9ng\s+trong\s+l\u0129nh\s+v\u1ef1c"
            r"|chuy\u00ean\s+v\u1ec1|chuy\u00ean\s+trong|trong\s+ng\u00e0nh|l\u0129nh\s+v\u1ef1c\s+ch\u00ednh)"
            r"\s+(?P<industry>[\w\s]{2,40})" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"INDUSTRY"}, confidence=0.80),

    RelationPattern("launched_at_event",
        re.compile(
            r"(?P<product>" + _NAME + r")\s+"
            r"(?:ch\u00ednh\s+th\u1ee9c\s+)?(?:ra\s+m\u1eaft|gi\u1edbi\s+thi\u1ec7u|c\u00f4ng\s+b\u1ed1)"
            r"(?:[^\n]{0,60}?)"
            r"t\u1ea1i\s+(?:s\u1ef1\s+ki\u1ec7n\s+)?(?P<event>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"PRODUCT"}, obj_types={"EVENT"}, confidence=0.88),

    RelationPattern("held_in",
        re.compile(
            r"(?P<event>" + _NAME + r")\s+"
            r"(?:t\u1ed5\s+ch\u1ee9c|di\u1ec5n\s+ra|khai\s+m\u1ea1c|ch\u00ednh\s+th\u1ee9c\s+di\u1ec5n\s+ra)"
            r"\s+(?:\u1edf\s+|t\u1ea1i\s+)(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"EVENT"}, obj_types={"LOCATION"}, confidence=0.85),

    RelationPattern("has_revenue",
        re.compile(
            r"(?P<org>" + _NAME + r")\s+"
            r"(?:b\u00e1o\s+c\u00e1o|ghi\s+nh\u1eadn|thu\s+v\u1ec1|\u0111\u1ea1t)"
            r"(?:\s+[\w\s]{0,20}?)?doanh\s+thu"
            r"(?:\s+\u0111\u1ea1t)?\s+"
            r"(?P<money>[\d][[\d,\.]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn)?\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\$)?)",
            re.UNICODE | re.IGNORECASE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"MONEY"}, confidence=0.82),

    RelationPattern("has_revenue",
        re.compile(
            r"doanh\s+thu(?:\s+(?:c\u1ee7a|c\u00f4ng\s+ty))?\s+(?P<org>" + _NAME + r")"
            r"\s+(?:l\u00e0|\u0111\u1ea1t|l\u00ean\s+t\u1edbi)\s+"
            r"(?P<money>[\d][[\d,\.]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn)?\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\$)?)",
            re.UNICODE | re.IGNORECASE,
        ),
        subj_types={"ORGANIZATION"}, obj_types={"MONEY"}, confidence=0.80),


    RelationPattern("occurred_in",
        re.compile(
            r"(?P<event>" + _NAME + r")\s+"
            r"(?:x\u1ea3y\s+ra|di\u1ec5n\s+ra|\u0111\u01b0\u1ee3c\s+t\u1ed5\s+ch\u1ee9c)\s+"
            r"(?:t\u1ea1i|\u1edf)\s+(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE,
        ),
        subj_types={"EVENT"}, obj_types={"LOCATION"}, confidence=0.75),
]

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

def merge_adjacent_entities(raw_entities: list[dict]) -> list[dict]:
    if not raw_entities:
        return raw_entities
    merged: list[dict] = []
    cur = raw_entities[0].copy()
    for nxt in raw_entities[1:]:
        same_type = nxt["ner_type"] == cur["ner_type"]
        cur_words = cur.get("words") or []
        nxt_words = nxt.get("words") or []
        cur_last  = cur_words[-1] if cur_words else None
        nxt_first = nxt_words[0]  if nxt_words else None
        if same_type and cur_last is not None and nxt_first is not None and nxt_first == cur_last + 1:
            cur["text"] += " " + nxt["text"]
            cur["words"] = cur_words + nxt_words
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

    _MAX_WORDS = {"Organization": 10, "Person": 6, "Location": 8,
                  "Product": 8, "Event": 10, "Date": 6,
                  "Money": 6, "Industry": 6, "Percent": 4}
    max_w = _MAX_WORDS.get(gtype, 10)
    if len(norm.split()) > max_w:
        return False

    if gtype in ["Person", "Organization"] and norm.islower():
        return False
        
    low = norm.lower().replace(",", " ").replace(".", " ")
    toks = [t for t in low.split() if t]
    
    if gtype == "Organization":
        bad_org_words = {"công ty", "tập đoàn", "Tập đoàn công ty", "tổng công ty", "group", "holdings", "inc", "corp", "hiệp hội", "ngân hàng", "ban", "ngành", "phòng", "văn phòng", "chi nhánh", "trung tâm"}
        if low in bad_org_words:
            return False
        if len(toks) <= 3 and all(t in {"group", "phòng", "ban", "chính", "nghiệp", "công"} for t in toks):
            return False
            
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
    if gtype == "Industry":
        if low in _GENERIC_INDUSTRY_WORDS:
            return False
        if len(norm) < 4 or (len(toks) == 1 and len(toks[0]) <= 4):
            return False
        _PERSON_TITLES = {"giáo", "ông", "bà", "anh", "chị", "tiến", "sĩ", "phó", "giám", "tổng"}
        orig_toks = norm.split()
        last_orig = orig_toks[-1] if orig_toks else ""
        if last_orig[0:1].isupper() or last_orig.lower() in _PERSON_TITLES:
            return False
    if gtype == "Organization" and _is_news_source(norm):
        return False
    return True

def _is_noisy_date(e: Entity) -> bool:
    return e.type == "Date" and not any(c.isdigit() for c in e.name)

def _pk(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))

def _tk(s: str, t: str, lbl: str) -> tuple[str, str, str]:
    a, b = (s, t) if s <= t else (t, s)
    return (a, b, lbl)

def _extract_pattern_relations(
    entities: list[Entity],
    sentences: list[str],
) -> tuple[list[Relation], set[tuple[str, str]]]:
    relations: list[Relation] = []
    seen_tk:   set[tuple[str, str, str]] = set()
    seen_pk:   set[tuple[str, str]]      = set()

    entity_map = {e.id: e for e in entities}

    for sent in sentences:
        for pat in RELATION_PATTERNS:
            lbl = _normalize_label(pat.name)
            if lbl is None:
                continue

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

                if _is_news_source(se.name) or _is_news_source(oe.name):
                    continue
                if not _validate_domain_range(lbl, se.type, oe.type):
                    continue

                tk = _tk(se.id, oe.id, lbl)
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

    _resolve_pronoun_relations(entities, sentences, relations, seen_tk, seen_pk)

    return relations, seen_pk

def _resolve_pronoun_relations(
    entities: list[Entity],
    sentences: list[str],
    relations: list[Relation],
    seen_tk: set[tuple[str, str, str]],
    seen_pk: set[tuple[str, str]],
) -> None:
    """Handle pronouns and generic subjects referencing the last-mentioned Organization."""
    orgs = [e for e in entities if e.type == "Organization"]
    locs = [e for e in entities if e.type == "Location"]
    if not orgs or not locs:
        return

    last_org: Optional[Entity] = None
    _LOC_PREFIX = r"(?:(?:qu\u1eadn|huy\u1ec7n|t\u1ec9nh|ph\u01b0\u1eddng|x\u00e3|th\u00e0nh\s+ph\u1ed1|tp\.?)\s+)?"
    pronoun_hq = re.compile(
        r"(?:(?:[Tt]r\u1ee5\s+s\u1edf\s+(?:ch\u00ednh\s+)?c\u1ee7a\s+c\u00f4ng\s+ty)"
        r"|(?:h\u1ecd|c\u00f4ng\s+ty))\s+"
        r"(?:\u0111\u1eb7t\s+(?:tr\u1ee5\s+s\u1edf\s+)?t\u1ea1i"
        r"|c\u00f3\s+tr\u1ee5\s+s\u1edf\s+t\u1ea1i"
        r"|m\u1edf\s+(?:th\u00eam\s+)?(?:v\u0103n\s+ph\u00f2ng"
        r"(?:\s+\u0111\u1ea1i\s+di\u1ec7n)?|chi\s+nh\u00e1nh)\s+t\u1ea1i)"
        r"\s+" + _LOC_PREFIX + r"(?P<loc>" + _NAME + r")",
        re.UNICODE,
    )

    for sent in sentences:
        for org in orgs:
            if _find_pos(org, sent) >= 0:
                last_org = org

        for m in pronoun_hq.finditer(sent):
            if last_org is None:
                continue
            loc_text = m.group("loc").strip()
            loc_e = _best_entity_match(loc_text, locs, {"LOCATION"})
            if not loc_e or loc_e.id == last_org.id:
                continue
            full_match = m.group(0).lower()
            lbl = "has_office" if "văn phòng" in full_match or "chi nhánh" in full_match else "headquartered_in"
            if not _validate_domain_range(lbl, last_org.type, loc_e.type):
                continue
            tk = _tk(last_org.id, loc_e.id, lbl)
            if tk in seen_tk:
                continue
            seen_tk.add(tk)
            seen_pk.add(_pk(last_org.id, loc_e.id))
            relations.append(Relation(source=last_org.id, target=loc_e.id, label=lbl))

def _extract_kb_relations(
    entities: list[Entity],
    sentences: list[str],
    existing_pk: set[tuple[str, str]],
) -> list[Relation]:
    """KB lookup restricted to entity pairs co-occurring in the same sentence."""
    if not kb.kb_ready:
        return []

    relations: list[Relation] = []
    seen_pk: set[tuple[str, str]] = set(existing_pk)

    for sent in sentences:
        ents_in_sent = [
            e for e in entities
            if not _is_noisy_date(e) and _find_pos(e, sent) >= 0
        ]
        if len(ents_in_sent) < 2:
            continue

        for i, se in enumerate(ents_in_sent):
            for te in ents_in_sent[i + 1:]:
                pk = _pk(se.id, te.id)
                if pk in seen_pk:
                    continue
                if _is_news_source(se.name) or _is_news_source(te.name):
                    continue

                raw_lbl = kb.find_relation(se.name, te.name)
                if not raw_lbl:
                    continue
                lbl = _normalize_label(raw_lbl)
                if lbl is None:
                    continue
                if not _validate_domain_range(lbl, se.type, te.type):
                    if _validate_domain_range(lbl, te.type, se.type):
                        se, te = te, se
                    else:
                        continue

                seen_pk.add(pk)
                relations.append(Relation(source=se.id, target=te.id, label=lbl))

    return relations

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
    return matches >= max(2, len(camel_parts) - 1)

def _resolve_aliases(entities: list[Entity], text: str) -> list[Entity]:
    alias_map = {}
    
    for m in re.finditer(r"([A-Z\u00C0-\u1EF9][\w\s,\u00C0-\u1EF9&]+)\s*\(\s*([^)]+)\s*\)", text):
        full = _norm(m.group(1))
        al = _norm(m.group(2))
        if 2 <= len(al) < len(full):
            best_ename = None
            for e in entities:
                if e.name.lower() in full.lower():
                    if best_ename is None or len(e.name) > len(best_ename):
                        best_ename = e.name
            target = best_ename.lower() if best_ename else full.lower()
            alias_map[al.lower()] = target

    orgs      = [e for e in entities if e.type == "Organization"]
    org_texts = sorted([e.name for e in orgs], key=len, reverse=True)
    for i, primary in enumerate(org_texts):
        for alias in org_texts[i + 1:]:
            a_low = alias.lower()
            if a_low not in alias_map and _is_camelcase_of(alias, primary):
                alias_map[a_low] = primary.lower()

    pers = [e for e in entities if e.type == "Person"]
    per_texts = sorted([e.name for e in pers], key=len, reverse=True)
    for i, primary in enumerate(per_texts):
        p_low = primary.lower()
        if len(p_low.split()) >= 2:
            for alias in per_texts[i+1:]:
                a_low = alias.lower()
                if a_low in p_low:
                     alias_map[a_low] = p_low

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
                if e.name.lower() in alias_map:
                    primary_e.aliases.append(e.name)
                    p_name = primary_e.name
                    if "(" not in p_name and len(e.name) <= len(p_name) - 5 and len(e.name.split()) <= 3:
                        primary_e.name = f"{p_name} ({e.name})"
            for p in e.properties:
                if not any(mp.key == p.key and mp.value == p.value for mp in primary_e.properties):
                    primary_e.properties.append(p)

    return list(merged.values())

def _fold_properties(
    entities: list[Entity],
    relations: list[Relation],
    sentences: list[str],
) -> tuple[list[Entity], list[Relation]]:
    """Giữ tất cả 9 entity types làm nodes. Không fold orphans."""
    kept_types = {
        "Organization", "Person", "Product", "Event", "Location",
        "Date", "Money", "Percent", "Industry",
    }
    entity_map = {e.id: e for e in entities}
    final_relations: list[Relation] = []
    for r in relations:
        if entity_map.get(r.source) and entity_map.get(r.target):
            final_relations.append(r)

    filtered_entities = [e for e in entities if e.type in kept_types]

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

_PRODUCT_CONTEXT_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"(?:sản\s+phẩm|gi\u1ea3i\s+ph\u00e1p|ph\u1ea7n\s+m\u1ec1m|\u1ee9ng\s+d\u1ee5ng|h\u1ec7\s+th\u1ed1ng|d\u1ecbch\s+v\u1ee5)"
        r"(?:\s+[\w\s\u00C0-\u1EF9]{0,40}?)?"
        r"(?:mang\s+t\u00ean|c\u00f3\s+t\u00ean|g\u1ecdi\s+l\u00e0|t\u00ean\s+l\u00e0|l\u00e0)\s+"
        r"(?P<name>[A-Z][\w]*(?:\s+[A-Z][\w]*)*)",
        re.UNICODE,
    ),
    re.compile(
        r"(?:n\u1ec1n\s+t\u1ea3ng|platform)\s+"
        r"(?P<name>[A-Z][\w]*(?:\s+[A-Z][\w]*)*)",
        re.UNICODE,
    ),
    re.compile(
        r"(?:s\u1ea3n\s+ph\u1ea9m|gi\u1ea3i\s+ph\u00e1p)\s+"
        r"(?P<name>[A-Z][\w]*(?:\s+[A-Z][\w]*)*)\s+"
        r"(?:do|b\u1edfi|\u0111\u01b0\u1ee3c)",
        re.UNICODE,
    ),
    re.compile(
        r"(?:ra\s+m\u1eaft|gi\u1edbi\s+thi\u1ec7u|c\u00f4ng\s+b\u1ed1)\s+"
        r"(?:s\u1ea3n\s+ph\u1ea9m\s+)?"
        r"(?P<name>[A-Z][\w]*(?:\s+[A-Z][\w]*)*)"
        r"(?=\s+(?:v\u00e0o|t\u1ea1i|t\u1ea1i\s+s\u1ef1\s+ki\u1ec7n|$|\.))",
        re.UNICODE,
    ),
]

def _detect_products_from_context(
    text: str,
    entities: list[Entity],
    seen: dict[tuple, str],
) -> list[Entity]:
    """Scan text for Product names using context clues that NER missed."""
    existing_lower = {e.name.lower() for e in entities}
    new_products: list[Entity] = []

    for pat in _PRODUCT_CONTEXT_PATTERNS:
        for m in pat.finditer(text):
            name = _norm(m.group("name"))
            if len(name) < 2:
                continue
            key = (name.lower(), "Product")
            if key in seen or name.lower() in existing_lower:
                for e in entities:
                    if e.name.lower() == name.lower() and e.type != "Product":
                        e.type = "Product"
                continue
            eid = f"E{len(entities) + len(new_products) + 1}"
            seen[key] = eid
            new_products.append(Entity(
                id=eid, name=name, type="Product",
                properties=[], aliases=[],
            ))
            existing_lower.add(name.lower())

    return new_products

def _extract_contextual_relations(
    entities: list[Entity],
    sentences: list[str],
    text: str,
    existing_pk: set[tuple[str, str]],
) -> list[Relation]:
    """Extract relations from complex sentence structures:
    1. Appositive: 'ORG, một doanh nghiệp có trụ sở tại LOC' → headquartered_in
    2. Compound: 'ORG ... và là đối tác lâu năm/dài/chiến lược của ORG2' → partnered_with
    3. Research: 'PRODUCT được xây dựng dựa trên nghiên cứu (hợp tác) với ORG' → developed_by
    4. operates_in: ORG develops product in INDUSTRY → ORG operates_in INDUSTRY
    """
    rels: list[Relation] = []
    orgs = [e for e in entities if e.type == "Organization"]
    locs = [e for e in entities if e.type == "Location"]
    products = [e for e in entities if e.type == "Product"]
    industries = [e for e in entities if e.type == "Industry"]

    def _add(src: Entity, tgt: Entity, lbl: str) -> bool:
        if src.id == tgt.id:
            return False
        if _is_news_source(src.name) or _is_news_source(tgt.name):
            return False
        if not _validate_domain_range(lbl, src.type, tgt.type):
            return False
        pk = _pk(src.id, tgt.id)
        if pk in existing_pk:
            return False
        existing_pk.add(pk)
        rels.append(Relation(source=src.id, target=tgt.id, label=lbl))
        return True

    _LOC_PREFIX = r"(?:(?:qu\u1eadn|huy\u1ec7n|t\u1ec9nh|tp\.?|th\u00e0nh\s+ph\u1ed1)\s+)?"

    appositive_hq = re.compile(
        r"(?P<org>" + _NAME + r")"
        r"(?:\s+Inc\.?|\s+Corp\.?|\s+Co\.?|\s+JSC\.?)?"
        r",\s*(?:m\u1ed9t\s+)?(?:doanh\s+nghi\u1ec7p|c\u00f4ng\s+ty|t\u1ed5\s+ch\u1ee9c)"
        r"(?:\s+[\w\s\u00C0-\u1EF9]{0,30}?)?"
        r"\s+(?:c\u00f3\s+)?tr\u1ee5\s+s\u1edf\s+(?:ch\u00ednh\s+)?t\u1ea1i\s+"
        + _LOC_PREFIX + r"(?P<loc>" + _NAME + r")",
        re.UNICODE,
    )
    compound_hq = re.compile(
        r"(?P<org>" + _NAME + r")"
        r"(?:[^.]{0,60}?)"
        r"(?:v\u00e0\s+)?(?:c\u00f3\s+)?tr\u1ee5\s+s\u1edf\s+(?:ch\u00ednh\s+)?t\u1ea1i\s+"
        + _LOC_PREFIX + r"(?P<loc>" + _NAME + r")",
        re.UNICODE,
    )
    for sent in sentences:
        for pat in [appositive_hq, compound_hq]:
            for m in pat.finditer(sent):
                org_e = _best_entity_match(m.group("org").strip(), orgs, {"ORGANIZATION"})
                loc_e = _best_entity_match(m.group("loc").strip(), locs, {"LOCATION"})
                if org_e and loc_e:
                    _add(org_e, loc_e, "headquartered_in")

    compound_partner = re.compile(
        r"v\u00e0\s+l\u00e0\s+\u0111\u1ed1i\s+t\u00e1c"
        r"\s+(?:l\u00e2u\s+(?:n\u0103m|d\u00e0i)|chi\u1ebfn\s+l\u01b0\u1ee3c)"
        r"\s+(?:c\u1ee7a\s+|v\u1edbi\s+)?(?P<org2>" + _NAME + r")",
        re.UNICODE,
    )
    for sent in sentences:
        for m in compound_partner.finditer(sent):
            match_start = m.start()
            subj_org = None
            best_pos = -1
            for org in orgs:
                pos = _find_pos(org, sent)
                if 0 <= pos < match_start and pos > best_pos:
                    best_pos = pos
                    subj_org = org
            if subj_org is None:
                continue
            org2_e = _best_entity_match(m.group("org2").strip(), orgs, {"ORGANIZATION"})
            if org2_e:
                _add(subj_org, org2_e, "partnered_with")
                _add(org2_e, subj_org, "partnered_with")

    research_dev = re.compile(
        r"(?P<product>" + _NAME + r")"
        r"\s+\u0111\u01b0\u1ee3c\s+(?:x\u00e2y\s+d\u1ef1ng|ph\u00e1t\s+tri\u1ec3n)"
        r"(?:\s+d\u1ef1a\s+tr\u00ean)?"
        r"(?:[^.]{0,80}?)"
        r"(?:nghi\u00ean\s+c\u1ee9u\s+)?(?:h\u1ee3p\s+t\u00e1c\s+)?v\u1edbi\s+"
        r"(?P<org>" + _NAME + r")",
        re.UNICODE,
    )
    for sent in sentences:
        for m in research_dev.finditer(sent):
            prod_e = _best_entity_match(m.group("product").strip(), products, {"PRODUCT"})
            org_e = _best_entity_match(m.group("org").strip(), orgs, {"ORGANIZATION"})
            if prod_e and org_e:
                _add(prod_e, org_e, "developed_by")

    _operate_explicit = re.compile(
        r"(?:thu\u1ed9c\s+l\u0129nh\s+v\u1ef1c|chuy\u00ean\s+v\u1ec1|chuy\u00ean\s+trong"
        r"|ho\u1ea1t\s+\u0111\u1ed9ng\s+trong\s+l\u0129nh\s+v\u1ef1c|trong\s+ng\u00e0nh"
        r"|l\u0129nh\s+v\u1ef1c\s+ch\u00ednh|ng\u00e0nh\s+ch\u00ednh|thu\u1ed9c\s+ng\u00e0nh)",
        re.UNICODE | re.IGNORECASE,
    )
    recent_orgs: list[Entity] = []
    for sent in sentences:
        sent_orgs = [e for e in orgs if _find_pos(e, sent) >= 0]
        if sent_orgs:
            recent_orgs = sent_orgs
        if not _operate_explicit.search(sent):
            continue
        inds_here = [e for e in industries if _find_pos(e, sent) >= 0]
        if not inds_here:
            continue
        orgs_to_link = sent_orgs if sent_orgs else []
        for org_e in orgs_to_link:
            for ind_e in inds_here:
                _add(org_e, ind_e, "operates_in")

    dates = [e for e in entities if e.type == "Date"]
    events = [e for e in entities if e.type == "Event"]
    products_all = [e for e in entities if e.type == "Product"]







    _held_pat = re.compile(
        r"(?P<event>" + _NAME + r")\s+t\u1ed5\s+ch\u1ee9c"
        r"\s+(?:\u1edf\s+|t\u1ea1i\s+)(?:TP\.?\s+)?(?P<loc>" + _NAME + r")",
        re.UNICODE,
    )
    for sent in sentences:
        for m in _held_pat.finditer(sent):
            ev_e = _best_entity_match(m.group("event").strip(), events, {"EVENT"})
            loc_e = _best_entity_match(m.group("loc").strip(), locs, {"LOCATION"})
            if ev_e and loc_e:
                _add(ev_e, loc_e, "held_in")
        m2 = re.search(
            r"t\u1ed5\s+ch\u1ee9c\s+(?:\u1edf\s+|t\u1ea1i\s+)(?:TP\.?\s+)?(?P<loc>" + _NAME + r")",
            sent, re.UNICODE,
        )
        if m2:
            for ev_e in [e for e in events if _find_pos(e, sent) >= 0]:
                loc_e = _best_entity_match(m2.group("loc").strip(), locs, {"LOCATION"})
                if loc_e:
                    _add(ev_e, loc_e, "held_in")

    persons = [e for e in entities if e.type == "Person"]





    moneys = [e for e in entities if e.type == "Money"]

    def _match_money(text_fragment: str) -> Optional[Entity]:
        frag = text_fragment.strip()
        digits = re.findall(r"[\d,\.]+", frag)
        return next(
            (mo for mo in moneys
             if frag in mo.name or mo.name in frag
             or any(t in mo.name for t in frag.split() if len(t) > 1)
             or any(d in mo.name for d in digits if len(d) > 1)),
            None,
        )

    _revenue_pat = re.compile(
        r"(?P<org>" + _NAME + r")\s+"
        r"(?:b\u00e1o\s+c\u00e1o|ghi\s+nh\u1eadn|thu\s+v\u1ec1)"
        r"(?:\s+[\w\s]{0,20}?)?doanh\s+thu"
        r"(?:\s+\u0111\u1ea1t)?\s+"
        r"(?P<money>[\d][[\d,\.]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn)?\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\$)?)",
        re.UNICODE | re.IGNORECASE,
    )
    for sent in sentences:
        for m in _revenue_pat.finditer(sent):
            org_e = _best_entity_match(m.group("org").strip(), orgs, {"ORGANIZATION"})
            mn_e = _match_money(m.group("money"))
            if org_e and mn_e:
                _add(org_e, mn_e, "has_revenue")

    _value_buyer = re.compile(
        r"cho\s+(?P<subj>" + _NAME + r")\s*(?:\([^)]+\))?\s*"
        r"v\u1edbi\s+(?:gi\u00e1\s+tr\u1ecb|tr\u1ecb\s+gi\u00e1)\s+"
        r"(?P<money>[\d][[\d,\.]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn)?\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\$)?)",
        re.UNICODE | re.IGNORECASE,
    )
    _value_bare = re.compile(
        r"v\u1edbi\s+(?:gi\u00e1\s+tr\u1ecb|tr\u1ecb\s+gi\u00e1)\s+"
        r"(?P<money>[\d][[\d,\.]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn)?\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\$)?)",
        re.UNICODE | re.IGNORECASE,
    )
    for sent in sentences:
        matched_money: set[str] = set()
        for m in _value_buyer.finditer(sent):
            mn_e = _match_money(m.group("money"))
            if not mn_e:
                continue
            subj = _best_entity_match(m.group("subj").strip(), orgs + products_all, None)
            if subj and _add(subj, mn_e, "has_value"):
                matched_money.add(mn_e.id)

        for m in _value_bare.finditer(sent):
            mn_e = _match_money(m.group("money"))
            if not mn_e or mn_e.id in matched_money:
                continue
            match_pos = m.start()
            cands = [e for e in orgs + products_all if _find_pos(e, sent) >= 0]
            if not cands:
                continue
            closest = min(cands, key=lambda e: abs(_find_pos(e, sent) - match_pos))
            if _add(closest, mn_e, "has_value"):
                matched_money.add(mn_e.id)

    _ORPHAN_FALLBACK: dict[str, tuple[list[Entity], str]] = {
        "Money":   (orgs + products_all,          "has_value"),
        "Percent": (orgs + products_all,          "has_value"),
        "Location":(events,                       "held_in"),
    }
    for fb_type, (fb_cands, fb_lbl) in _ORPHAN_FALLBACK.items():
        for fb_e in [e for e in entities if e.type == fb_type]:
            if any(r.source == fb_e.id or r.target == fb_e.id for r in rels):
                continue
            best_target2: Optional[Entity] = None
            best_dist2 = 999999
            for sent in sentences:
                fb_pos = _find_pos(fb_e, sent)
                if fb_pos < 0:
                    continue
                for cand in fb_cands:
                    if not _validate_domain_range(fb_lbl, cand.type, fb_e.type):
                        continue
                    c_pos = _find_pos(cand, sent)
                    if c_pos < 0:
                        continue
                    dist = abs(fb_pos - c_pos)
                    if dist < best_dist2:
                        best_dist2 = dist
                        best_target2 = cand
            if best_target2 and best_dist2 < 999999:
                _add(best_target2, fb_e, fb_lbl)

    return rels

_ORPHAN_RULES: dict[str, list[tuple[str, set[str], bool]]] = {
    "Date": [],
    "Money": [
        ("has_revenue", {"Organization"}, False),
        ("has_value",   {"Organization", "Product"}, False),
    ],
    "Percent": [
        ("has_value", {"Organization", "Product"}, False),
    ],
    "Location": [
        ("held_in",         {"Event"},         False),
        ("occurred_in",     {"Event"},         False),
        ("headquartered_in",{"Organization"},  False),
        ("has_office",      {"Organization"},  False),
    ],
    "Industry": [
        ("operates_in", {"Organization"}, False),
    ],
    "Product": [
        ("developed_by", {"Organization"}, True),
        ("launched_at",  {"Event"},        True),
    ],
    "Event": [
        ("held_in",    {"Location"},    True),
        ("occurred_in",{"Location"},    True),
    ],
    "Person": [
        ("former_employee", {"Organization"}, True),
        ("founded",         {"Organization"}, True),
    ],
    "Organization": [
        ("partnered_with",   {"Organization"}, True),
        ("headquartered_in", {"Location"},     True),
        ("operates_in",      {"Industry"},     True),
    ],
}

def _connect_all_orphans(
    entities: list[Entity],
    relations: list[Relation],
    sentences: list[str],
    existing_pk: set[tuple[str, str]],
) -> list[Relation]:
    """Tầng cuối — kết nối mọi node còn mồ côi.
    Chiến lược tìm partner:
      1. Cùng câu (ưu tiên entity đã có edge)
      2. Câu lân cận (±2 câu)
      3. Toàn document (khoảng cách character)
    Luôn validate domain/range trước khi tạo edge.
    """
    new_rels: list[Relation] = []
    connected: set[str] = {r.source for r in relations} | {r.target for r in relations}

    def _add_orphan(src: Entity, tgt: Entity, lbl: str) -> bool:
        if src.id == tgt.id:
            return False
        if _is_news_source(src.name) or _is_news_source(tgt.name):
            return False
        if not _validate_domain_range(lbl, src.type, tgt.type):
            return False
        pk = _pk(src.id, tgt.id)
        if pk in existing_pk:
            return False
        existing_pk.add(pk)
        new_rels.append(Relation(source=src.id, target=tgt.id, label=lbl))
        connected.add(src.id)
        connected.add(tgt.id)
        return True

    def sent_idx(e: Entity) -> int:
        for i, s in enumerate(sentences):
            if _find_pos(e, s) >= 0:
                return i
        return -1

    sent_map: dict[str, int] = {e.id: sent_idx(e) for e in entities}

    def proximity_key(orphan: Entity, partner: Entity) -> tuple:
        oi = sent_map.get(orphan.id, -1)
        pi = sent_map.get(partner.id, -1)
        if oi >= 0 and pi >= 0:
            dist = abs(oi - pi)
        elif oi < 0 and pi < 0:
            dist = 0
        else:
            dist = 999          # one not found → penalise
        already_connected = 0 if partner.id in connected else 1
        return (dist, already_connected)

    orphans = [e for e in entities if e.id not in connected]

    for orphan in orphans:
        rules = _ORPHAN_RULES.get(orphan.type, [])
        linked = False

        for rel_lbl, partner_types, orphan_is_subject in rules:
            if linked:
                break
            partners = [
                e for e in entities
                if e.type in partner_types and e.id != orphan.id
            ]
            if not partners:
                continue

            partners.sort(key=lambda p: proximity_key(orphan, p))

            for partner in partners:
                if orphan_is_subject:
                    ok = _add_orphan(orphan, partner, rel_lbl)
                else:
                    ok = _add_orphan(partner, orphan, rel_lbl)
                if ok:
                    linked = True
                    break

    return new_rels

_KW_BRIDGE: list[tuple[str, re.Pattern, set[str], set[str], bool]] = [
    ("partnered_with",
     re.compile(
         r"h\u1ee3p\s+t\u00e1c|li\u00ean\s+k\u1ebft|li\u00ean\s+doanh|li\u00ean\s+minh"
         r"|k\u00fd\s+th\u1ecfa\s+thu\u1eadn|k\u00fd\s+h\u1ee3p\s+\u0111\u1ed3ng\s+h\u1ee3p\s+t\u00e1c",
         re.IGNORECASE | re.UNICODE),
     {"Organization"}, {"Organization"}, True),

    ("competitor_of",
     re.compile(
         r"\u0111\u1ed1i\s+th\u1ee7|c\u1ea1nh\s+tranh|v\u01b0\u1ee3t\s+m\u1eb7t",
         re.IGNORECASE | re.UNICODE),
     {"Organization"}, {"Organization"}, True),

    ("headquartered_in",
     re.compile(
         r"tr\u1ee5\s+s\u1edf\s+t\u1ea1i|\u0111\u1eb7t\s+t\u1ea1i"
         r"|t\u1ecda\s+l\u1ea1c\s+t\u1ea1i|c\u00f3\s+\u0111\u1ecba\s+ch\u1ec9\s+t\u1ea1i"
         r"|th\u00e0nh\s+l\u1eadp\s+t\u1ea1i|s\u00e1ng\s+l\u1eadp\s+t\u1ea1i",
         re.IGNORECASE | re.UNICODE),
     {"Organization"}, {"Location"}, False),

    ("has_office",
     re.compile(
         r"v\u0103n\s+ph\u00f2ng(?:\s+\u0111\u1ea1i\s+di\u1ec7n)?|chi\s+nh\u00e1nh",
         re.IGNORECASE | re.UNICODE),
     {"Organization"}, {"Location"}, False),

    ("founded",
     re.compile(
         r"s\u00e1ng\s+l\u1eadp|\u0111\u1ed3ng\s+s\u00e1ng\s+l\u1eadp|th\u00e0nh\s+l\u1eadp|kh\u1edfi\s+x\u01b0\u1edbng",
         re.IGNORECASE | re.UNICODE),
     {"Person"}, {"Organization"}, True),

    ("developed_by",
     re.compile(
         r"ph\u00e1t\s+tri\u1ec3n|x\u00e2y\s+d\u1ef1ng|thi\u1ebft\s+k\u1ebf",
         re.IGNORECASE | re.UNICODE),
     {"Product"}, {"Organization"}, True),

    ("launched_at",
     re.compile(
         r"ra\s+m\u1eaft|gi\u1edbi\s+thi\u1ec7u|c\u00f4ng\s+b\u1ed1",
         re.IGNORECASE | re.UNICODE),
     {"Product"}, {"Event"}, False),

    ("held_in",
     re.compile(
         r"t\u1ed5\s+ch\u1ee9c|di\u1ec5n\s+ra|khai\s+m\u1ea1c",
         re.IGNORECASE | re.UNICODE),
     {"Event"}, {"Location"}, False),

    ("former_employee",
     re.compile(
         r"t\u1eebng\s+l\u00e0m\s+vi\u1ec7c|t\u1eebng\s+c\u00f4ng\s+t\u00e1c"
         r"|t\u1eebng\s+gi\u1eef\s+ch\u1ee9c|c\u1ef1u",
         re.IGNORECASE | re.UNICODE),
     {"Person"}, {"Organization"}, False),

    ("operates_in",
     re.compile(
         r"thu\u1ed9c\s+l\u0129nh\s+v\u1ef1c|chuy\u00ean\s+v\u1ec1|chuy\u00ean\s+trong"
         r"|ho\u1ea1t\s+\u0111\u1ed9ng\s+trong\s+l\u0129nh\s+v\u1ef1c|thu\u1ed9c\s+ng\u00e0nh",
         re.IGNORECASE | re.UNICODE),
     {"Organization"}, {"Industry"}, False),

    ("has_revenue",
     re.compile(
         r"doanh\s+thu|thu\s+nh\u1eadp|thu\s+v\u1ec1",
         re.IGNORECASE | re.UNICODE),
     {"Organization"}, {"Money"}, False),

    ("has_value",
     re.compile(
         r"gi\u00e1\s+tr\u1ecb|tr\u1ecb\s+gi\u00e1|gi\u00e1\s+h\u1ee3p\s+\u0111\u1ed3ng",
         re.IGNORECASE | re.UNICODE),
     {"Organization", "Product"}, {"Money"}, False),

    ("occurred_in",
     re.compile(
         r"x\u1ea3y\s+ra\s+t\u1ea1i|di\u1ec5n\s+ra\s+t\u1ea1i|t\u1ed5\s+ch\u1ee9c\s+t\u1ea1i",
         re.IGNORECASE | re.UNICODE),
     {"Event"}, {"Location"}, True),
]

def _extract_keyword_bridge(
    entities: list[Entity],
    sentences: list[str],
    existing_pk: set[tuple[str, str]],
) -> list[Relation]:
    """Tìm relation bằng keyword-bridge:
    1. Tìm keyword trong câu
    2. Tìm entity trước keyword (subject) và sau keyword (object)
    3. Áp dụng domain/range filter để xác định chiều quan hệ
    Không dùng _NAME regex → hoạt động với mọi cấu trúc tên tiếng Việt.
    """
    rels: list[Relation] = []

    def _add(src: Entity, tgt: Entity, lbl: str) -> bool:
        if src.id == tgt.id:
            return False
        if _is_news_source(src.name) or _is_news_source(tgt.name):
            return False
        if not _validate_domain_range(lbl, src.type, tgt.type):
            return False
        pk = _pk(src.id, tgt.id)
        if pk in existing_pk:
            return False
        existing_pk.add(pk)
        rels.append(Relation(source=src.id, target=tgt.id, label=lbl))
        return True

    _PRONOUN_SUBJ = re.compile(
        r"^(?:[Hh]\u1ecd|[Cc]\u00f4ng\s+ty|[Tt]\u1eadp\s+\u0111o\u00e0n|[Cc]\u00f4ng\s+ty\s+n\u00e0y"
        r"|[Cc]\u00f4ng\s+ty\s+tr\u00ean|[Nn]h\u00e0\s+m\u00e1y)\b",
        re.UNICODE,
    )
    orgs_in_context = [e for e in entities if e.type == "Organization"]
    last_seen_org: Optional[Entity] = None

    for sent in sentences:
        for e in orgs_in_context:
            if _find_pos(e, sent) >= 0:
                last_seen_org = e

        ent_positions: list[tuple[int, Entity]] = []
        for e in entities:
            p = _find_pos(e, sent)
            if p >= 0:
                ent_positions.append((p, e))

        if _PRONOUN_SUBJ.match(sent) and last_seen_org and \
                not any(e.type == "Organization" for _, e in ent_positions):
            ent_positions.insert(0, (0, last_seen_org))

        if len(ent_positions) < 2:
            continue
        ent_positions.sort(key=lambda x: x[0])

        for label, kw_pat, subj_types, obj_types, allow_reverse in _KW_BRIDGE:
            for km in kw_pat.finditer(sent):
                kw_start = km.start()
                kw_end = km.end()

                before = [(p, e) for p, e in ent_positions if p < kw_start and e.type in subj_types]
                after  = [(p, e) for p, e in ent_positions if p >= kw_end and e.type in obj_types]

                if before and after:
                    subj_e = max(before, key=lambda x: x[0])[1]
                    obj_e  = min(after,  key=lambda x: x[0])[1]
                    _add(subj_e, obj_e, label)
                elif allow_reverse:
                    before_rev = [(p, e) for p, e in ent_positions if p < kw_start and e.type in obj_types]
                    after_rev  = [(p, e) for p, e in ent_positions if p >= kw_end and e.type in subj_types]
                    if before_rev and after_rev:
                        subj_e = min(after_rev, key=lambda x: x[0])[1]
                        obj_e  = max(before_rev, key=lambda x: x[0])[1]
                        _add(subj_e, obj_e, label)

    return rels

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

    for m in re.finditer(
        r"(?P<main>[A-Z\u00C0-\u1EF9][\w\s\u00C0-\u1EF9&]+?)"
        r"(?:\s+m\u1edf\s+r\u1ed9ng)?\s+c\u1ee7ng\s+c\u1ed1\s+v\u1ecb\s+th\u1ebf\s+c\u1ea1nh\s+tranh"
        r"[^n\n]{0,60}?nh\u01b0\s+(?P<oc>[A-Z\u00C0-\u1EF9][\w\s,\u00C0-\u1EF9&]+?)(?=[,\.\n]|$)",
        text,
    ):
        main_e = _best_entity_match(m.group("main").strip(), orgs, {"ORGANIZATION"})
        if not main_e:
            continue
        parts = re.split(r'\s+và\s+|,\s*', m.group("oc"))
        for part in parts:
            competitor_e = _best_entity_match(part.strip(), orgs, {"ORGANIZATION"})
            if competitor_e and competitor_e.id != main_e.id:
                pk = _pk(main_e.id, competitor_e.id)
                if pk not in existing_pk:
                    existing_pk.add(pk)
                    rels.append(Relation(source=main_e.id, target=competitor_e.id, label="competitor_of"))

    return rels

def _output_guard(
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> list[Relation]:
    """Final filter: only keep relations in ALLOWED_RELATIONS with valid domain/range."""
    clean: list[Relation] = []
    for r in relations:
        if r.label not in ALLOWED_RELATIONS:
            continue
        src = entity_map.get(r.source)
        tgt = entity_map.get(r.target)
        if not src or not tgt:
            continue
        if _is_news_source(src.name) or _is_news_source(tgt.name):
            continue
        if not _validate_domain_range(r.label, src.type, tgt.type):
            continue
        clean.append(r)
    return clean

def build_graph(raw_entities: list[dict], text: str) -> GraphData:
    """
    Strict relation extraction — whitelist only:
      Tầng 1:   Pattern regex (RELATION_PATTERNS)
      Tầng 1.5: Multi-entity patterns (bởi A và B, như A và B)
      Tầng 2:   KB lookup (same-sentence evidence only)
      Guard:    Drop anything outside ALLOWED_RELATIONS
    """
    raw_entities = merge_adjacent_entities(raw_entities)
    sentences    = split_sentences(text)

    seen:     dict[tuple, str] = {}
    entities: list[Entity]     = []

    for raw in raw_entities:
        gtype = TYPE_MAP.get(raw["ner_type"])
        if gtype is None:
            continue
        name = _norm(raw["text"].strip().strip(".,;:!?\"'()"))
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

    new_products = _detect_products_from_context(text, entities, seen)
    entities.extend(new_products)

    entities = _resolve_aliases(entities, text)

    pat_rels, existing_pk = _extract_pattern_relations(entities, sentences)
    logger.debug("Tầng 1 (pattern): %d", len(pat_rels))

    multi_rels = _extract_multi_entity_relations(entities, text, existing_pk)
    logger.debug("Tầng 1.5 (multi-entity): %d", len(multi_rels))

    ctx_rels = _extract_contextual_relations(entities, sentences, text, existing_pk)
    logger.debug("Tầng 1.6 (contextual): %d", len(ctx_rels))

    kw_rels = _extract_keyword_bridge(entities, sentences, existing_pk)
    logger.debug("Tầng 1.7 (keyword-bridge): %d", len(kw_rels))

    kb_rels = _extract_kb_relations(entities, sentences, existing_pk)
    logger.debug("Tầng 2 (KB same-sentence): %d", len(kb_rels))

    all_rels = pat_rels + multi_rels + ctx_rels + kw_rels + kb_rels

    final_entities, final_relations = _fold_properties(entities, all_rels, [])

    orphan_rels = _connect_all_orphans(final_entities, final_relations, sentences, existing_pk)
    final_relations += orphan_rels

    entity_map = {e.id: e for e in final_entities}
    final_relations = _output_guard(final_relations, entity_map)

    logger.info(
        "build_graph: %d entities, %d relations (P=%d M=%d C=%d KW=%d KB=%d OR=%d)",
        len(final_entities), len(final_relations),
        len(pat_rels), len(multi_rels), len(ctx_rels), len(kw_rels), len(kb_rels),
        len(orphan_rels),
    )
    return GraphData(entities=final_entities, relations=final_relations)

def predict_new_links(
    entities: list[Entity],
    relations: list[Relation],
) -> list[Relation]:
    """Predict new links using KB only — all results are marked isPredicted=True
    and filtered through the whitelist."""
    if not kb.kb_ready:
        return []

    existing = {_pk(r.source, r.target) for r in relations}
    predicted: list[Relation] = []

    for i, src in enumerate(entities):
        for tgt in entities[i + 1:]:
            if _pk(src.id, tgt.id) in existing:
                continue
            if _is_news_source(src.name) or _is_news_source(tgt.name):
                continue

            raw_lbl = kb.find_relation(src.name, tgt.name)
            if not raw_lbl:
                continue
            lbl = _normalize_label(raw_lbl)
            if lbl is None:
                continue

            if _validate_domain_range(lbl, src.type, tgt.type):
                predicted.append(Relation(
                    source=src.id, target=tgt.id, label=lbl, isPredicted=True,
                ))
            elif _validate_domain_range(lbl, tgt.type, src.type):
                predicted.append(Relation(
                    source=tgt.id, target=src.id, label=lbl, isPredicted=True,
                ))

            if len(predicted) >= 5:
                return predicted

    return predicted
