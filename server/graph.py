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
    ("Person", "Organization"):       "LÃNH ĐẠO",
    ("Organization", "Person"):       "CÓ THÀNH VIÊN",
    ("Person", "Location"):           "Ở TẠI",
    ("Organization", "Location"):     "ĐẶT TẠI",
    ("Person", "Event"):              "THAM GIA",
    ("Organization", "Event"):        "TỔ CHỨC SỰ KIỆN",
    ("Person", "Product"):            "SỬ DỤNG",
    ("Organization", "Product"):      "SẢN XUẤT",
    ("Product", "Event"):             "RA MẮT TẠI",
    ("Location", "Event"):            "XẢY RA TẠI",
    ("Organization", "Organization"): "HỢP TÁC",
    ("Person", "Person"):             "LIÊN QUAN",
    ("Organization", "Date"):         "THÀNH LẬP",
    ("Event", "Date"):                "DIỄN RA VÀO",
    ("Product", "Date"):              "RA MẮT",
    ("Organization", "Money"):        "CÓ GIÁ TRỊ",
    ("Person", "Money"):              "THU NHẬP",
    ("Product", "Money"):             "GIÁ",
    ("Event", "Money"):               "NGÂN SÁCH",
    ("Organization", "Percent"):      "TĂNG TRƯỞNG",
    ("Product", "Percent"):           "PHẦN TRĂM",
    ("Organization", "Industry"):     "THUỘC NGÀNH",
    ("Person", "Industry"):           "HOẠT ĐỘNG TRONG",
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
    # "Nguyễn Minh Anh - CEO của SaoVietTech"
    RelationPattern("LÃNH ĐẠO",
        re.compile(
            r"(?P<person>" + _NAME + r")"
            r"\s*[,–\-]?\s*"
            r"(?:Ch\u1ee7\s+t\u1ecbch|T\u1ed5ng\s+gi\u00e1m\s+\u0111\u1ed1c"
            r"|Gi\u00e1m\s+\u0111\u1ed1c|CEO|CFO|CTO|COO"
            r"|Ph\u00f3\s+ch\u1ee7\s+t\u1ecbch|Tr\u01b0\u1edfng\s+ban"
            r"|Gi\u00e1m\s+\u0111\u1ed1c\s+\u0111i\u1ec1u\s+h\u00e0nh)"
            r"\s+(?:c\u1ee7a\s+|t\u1ea1i\s+)?"
            r"(?P<org>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.85),

    # "Nguyễn Minh Anh từng làm việc tại FPT Software từ..."
    # Lookahead tại cuối org để dừng trước "từ", "đến", "trước"
    RelationPattern("LÀM VIỆC TẠI",
        re.compile(
            r"(?P<person>" + _NAME + r")"
            r"\s+(?:t\u1eebng\s+)?l\u00e0m\s+vi\u1ec7c\s+t\u1ea1i"
            r"\s+(?P<org>" + _NAME + r")"
            r"(?=\s+(?:t\u1eeb|trong|tr\u01b0\u1edbc|sau|\u0111\u1ebfn|,|\.|$))",
            re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.80),

    # "Nguyễn Minh Anh sáng lập SaoVietTech"
    # "... trước khi sáng lập SaoVietTech" — trước khi là optional prefix
    RelationPattern("SÁNG LẬP",
        re.compile(
            r"(?P<person>" + _NAME + r")"
            r"\s+(?:tr\u01b0\u1edbc\s+khi\s+)?"
            r"(?:s\u00e1ng\s+l\u1eadp|\u0111\u1ed3ng\s+s\u00e1ng\s+l\u1eadp)"
            r"\s+(?P<org>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.85),

    # ── Tổ chức × Tổ chức ────────────────────────────────────────
    # "SaoVietTech hợp tác chiến lược với Global AI Solutions"
    RelationPattern("HỢP TÁC",
        re.compile(
            r"(?P<org1>" + _NAME + r")"
            r"\s+(?:h\u1ee3p\s+t\u00e1c\s+chi\u1ebfn\s+l\u01b0\u1ee3c"
            r"|h\u1ee3p\s+t\u00e1c|k\u00fd\s+k\u1ebft|b\u1eaft\s+tay"
            r"|li\u00ean\s+k\u1ebft|ph\u1ed1i\s+h\u1ee3p"
            r"|k\u00fd\s+bi\u00ean\s+b\u1ea3n\s+ghi\s+nh\u1edb)"
            r"\s+(?:c\u00f9ng\s+|v\u1edbi\s+)?"
            r"(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"},
        confidence=0.80, bidirectional=True),

    # "SaoVietTech ký thỏa thuận/hợp đồng ... cho/với Hikari"
    RelationPattern("HỢP TÁC",
        re.compile(
            r"(?P<org1>" + _NAME + r")"
            r"\s+k\u00fd\s+(?:th\u1ecfa\s+thu\u1eadn|h\u1ee3p\s+\u0111\u1ed3ng)"
            r"(?:\s+\w+){0,5}?"
            r"\s+(?:cho|v\u1edbi)\s+"
            r"(?P<org2>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"ORGANIZATION"}, confidence=0.82),

    # "SaoVietTech đầu tư vào Hikari"
    RelationPattern("ĐẦU TƯ",
        re.compile(
            r"(?P<investor>" + _NAME + r")"
            r"\s+(?:\u0111\u1ea7u\s+t\u01b0|r\u00f3t\s+v\u1ed1n|r\u00f3t"
            r"|g\u00f3p\s+v\u1ed1n|mua\s+c\u1ed5\s+ph\u1ea7n"
            r"|mua\s+l\u1ea1i|th\u00e2u\s+t\u00f3m)"
            r"\s+(?:v\u00e0o\s+|cho\s+)?"
            r"(?P<target>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION", "PERSON"}, obj_types={"ORGANIZATION"}, confidence=0.82),

    # ── Sản phẩm ─────────────────────────────────────────────────
    # "SaoVietTech ra mắt VisionX"
    RelationPattern("SẢN XUẤT",
        re.compile(
            r"(?P<company>" + _NAME + r")"
            r"\s+(?:ra\s+m\u1eaft|s\u1ea3n\s+xu\u1ea5t|ph\u00e1t\s+tri\u1ec3n"
            r"|tung\s+ra|gi\u1edbi\s+thi\u1ec7u|c\u00f4ng\s+b\u1ed1"
            r"|ph\u00e1t\s+h\u00e0nh|tr\u00ecnh\s+l\u00e0ng"
            r"|ch\u00ednh\s+th\u1ee9c\s+ra\s+m\u1eaft|cho\s+ra\s+m\u1eaft)"
            r"\s+(?P<product>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"PRODUCT"}, confidence=0.78),

    # ── Địa điểm ─────────────────────────────────────────────────
    RelationPattern("ĐẶT TẠI",
        re.compile(
            r"(?P<org>" + _NAME + r")"
            r"\s+(?:\u0111\u1eb7t\s+t\u1ea1i|c\u00f3\s+tr\u1ee5\s+s\u1edf\s+t\u1ea1i"
            r"|ho\u1ea1t\s+\u0111\u1ed9ng\s+t\u1ea1i|\u0111\u1eb7t\s+tr\u1ee5\s+s\u1edf"
            r"|th\u00e0nh\s+l\u1eadp\s+t\u1ea1i|khai\s+tr\u01b0\u01a1ng\s+t\u1ea1i"
            r"|m\u1edf\s+r\u1ed9ng\s+t\u1ea1i|c\u00f3\s+v\u0103n\s+ph\u00f2ng\s+t\u1ea1i)"
            r"\s+(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"LOCATION"}, confidence=0.75),

    # ── Ngành ────────────────────────────────────────────────────
    RelationPattern("THUỘC NGÀNH",
        re.compile(
            r"(?P<org>" + _NAME + r")"
            r"\s+(?:thu\u1ed9c\s+l\u0129nh\s+v\u1ef1c|ho\u1ea1t\s+\u0111\u1ed9ng\s+trong"
            r"|chuy\u00ean\s+v\u1ec1|trong\s+ng\u00e0nh|l\u0129nh\s+v\u1ef1c|ng\u00e0nh)"
            r"\s+(?P<industry>[\w\s]{2,40})" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"INDUSTRY"}, confidence=0.72),

    # ── Sự kiện ──────────────────────────────────────────────────
    RelationPattern("TỔ CHỨC SỰ KIỆN",
        re.compile(
            r"(?P<org>" + _NAME + r")"
            r"\s+(?:t\u1ed5\s+ch\u1ee9c|\u0111\u0103ng\s+cai|ch\u1ee7\s+tr\u00ec"
            r"|\u0111\u1ee9ng\s+ra\s+t\u1ed5\s+ch\u1ee9c|kh\u1edfi\s+\u0111\u1ed9ng"
            r"|ph\u00e1t\s+\u0111\u1ed9ng)"
            r"\s+(?P<event>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"ORGANIZATION"}, obj_types={"EVENT"}, confidence=0.78),

    RelationPattern("XẢY RA TẠI",
        re.compile(
            r"(?P<event>" + _NAME + r")"
            r"\s+(?:di\u1ec5n\s+ra\s+t\u1ea1i|t\u1ed5\s+ch\u1ee9c\s+t\u1ea1i"
            r"|x\u1ea3y\s+ra\s+t\u1ea1i|\u0111\u01b0\u1ee3c\s+t\u1ed5\s+ch\u1ee9c\s+t\u1ea1i"
            r"|khai\s+m\u1ea1c\s+t\u1ea1i|di\u1ec5n\s+ra)"
            r"\s+(?P<loc>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"EVENT"}, obj_types={"LOCATION"}, confidence=0.76),

    RelationPattern("THAM GIA",
        re.compile(
            r"(?P<person>" + _NAME + r")"
            r"\s+(?:tham\s+gia|tham\s+d\u1ef1|xu\u1ea5t\s+hi\u1ec7n\s+t\u1ea1i"
            r"|c\u00f3\s+m\u1eb7t\s+t\u1ea1i|ph\u00e1t\s+bi\u1ec3u\s+t\u1ea1i)"
            r"\s+(?P<event>" + _NAME + r")" + _STOP,
            re.UNICODE),
        subj_types={"PERSON"}, obj_types={"EVENT"}, confidence=0.73),

    # ── Tài chính ────────────────────────────────────────────────
    RelationPattern("CÓ GIÁ TRỊ",
        re.compile(
            r"(?P<entity>" + _NAME + r")"
            r"\s+(?:tr\u1ecb\s+gi\u00e1|c\u00f3\s+gi\u00e1\s+tr\u1ecb|\u0111\u1ea1t"
            r"|gi\u00e1\s+tr\u1ecb\s+kho\u1ea3ng|\u01b0\u1edbc\s+t\u00ednh)"
            r"\s+(?P<money>\d[\d.,]*\s*(?:tri\u1ec7u|t\u1ef7|ngh\u00ecn|ng\u00e0n)?"
            r"\s*(?:USD|VND|VN\u0110|\u0111\u1ed3ng|\u0111\u00f4)?)",
            re.UNICODE),
        subj_types=None, obj_types=None, confidence=0.75),

    RelationPattern("TĂNG TRƯỞNG",
        re.compile(
            r"(?P<entity>" + _NAME + r")"
            r"\s+(?:t\u0103ng\s+tr\u01b0\u1edfng|t\u0103ng|\u0111\u1ea1t"
            r"|ghi\s+nh\u1eadn|t\u0103ng\s+l\u00ean)"
            r"\s+(?P<pct>\d[\d.,]*\s*%(?:/n\u0103m)?)",
            re.UNICODE),
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
        if e.name.lower() == ml:
            return e
    for e in cands:                          # substring
        el = e.name.lower()
        if el in ml or ml in el:
            return e
    best_e, best_s = None, 0.0              # Jaccard
    for e in cands:
        s = _jaccard(e.name, matched_text)
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


def _find_pos(name: str, sent: str) -> int:
    m = _get_search_pat(name).search(sent)
    return m.start() if m else -1


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
    low = norm.lower().replace(",", " ").replace(".", " ")
    toks = [t for t in low.split() if t]
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
        ents = [(e, _find_pos(e.name, sent))
                for e in entities
                if not _is_noisy_date(e) and _find_pos(e.name, sent) >= 0]
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
# §9 — build_graph (entry point)
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
        ))

    if not entities:
        return GraphData(entities=[], relations=[])

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

    all_rels = pat_rels + kb_rels + cooc_rels
    logger.info(
        "build_graph: %d entities, %d relations (P=%d KB=%d C=%d)",
        len(entities), len(all_rels), len(pat_rels), len(kb_rels), len(cooc_rels),
    )
    return GraphData(entities=entities, relations=all_rels)


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