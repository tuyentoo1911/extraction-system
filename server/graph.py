"""Xây dựng Knowledge Graph từ NER output + Knowledge Base."""

import re
import logging

from constants import TYPE_MAP
from schemas import Entity, EntityProperty, Relation, GraphData
from ner import split_sentences
import knowledge_base as kb

logger = logging.getLogger(__name__)

_GENERIC_DATE_WORDS = {
    "nam", "năm", "thang", "tháng", "ngay", "ngày", "quy", "quý", "tuan", "tuần",
}
_LOCATION_PREFIXES = {
    "tp", "tp.", "thành", "thanh", "quận", "quan", "huyện", "huyen", "tỉnh", "tinh",
    "phường", "phuong", "xã", "xa", "thị", "thi",
}

# Fallback: nhãn quan hệ theo cặp loại entity (khi không tìm thấy trong KB)
_RELATION_LABELS: dict[tuple[str, str], str] = {
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
    ("Person", "Date"):               "XẢY RA",
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


def infer_relation_label(src: Entity, tgt: Entity) -> str:
    """Tra KB trước, nếu không có thì dùng rule-based theo loại entity."""
    # 1. Tra Knowledge Base (quan hệ thực từ corpus)
    if kb.kb_ready:
        label = kb.find_relation(src.name, tgt.name)
        if label:
            return label

    # 2. Fallback rule-based
    return _RELATION_LABELS.get((src.type, tgt.type), "LIÊN QUAN ĐẾN")


def merge_adjacent_entities(raw_entities: list[dict]) -> list[dict]:
    """Gộp các entity liền kề có cùng NER type thành một."""
    if not raw_entities:
        return raw_entities

    merged: list[dict] = []
    current = raw_entities[0].copy()

    for nxt in raw_entities[1:]:
        same_type = nxt["ner_type"] == current["ner_type"]
        cur_last  = current["words"][-1] if current.get("words") else -99
        nxt_first = nxt["words"][0]      if nxt.get("words")     else -98

        if same_type and nxt_first == cur_last + 1:
            current["text"] += " " + nxt["text"]
            current["words"] = current.get("words", []) + nxt.get("words", [])
        else:
            merged.append(current)
            current = nxt.copy()

    merged.append(current)
    return merged


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_informative_entity(name: str, graph_type: str) -> bool:
    """
    Lọc entity quá chung chung để tránh graph dạng "star".
    Ví dụ: Date chỉ có "Năm", "Tháng" sẽ bị loại.
    """
    norm = _normalize_text(name)
    if len(norm) < 2:
        return False
    if re.fullmatch(r"[\W_]+", norm):
        return False

    lowered = norm.lower()
    lowered = lowered.replace(",", " ").replace(".", " ")
    tokens = [t for t in lowered.split() if t]

    if graph_type == "Date":
        # Date cần có chữ số hoặc >= 2 token có nghĩa.
        if not any(ch.isdigit() for ch in norm):
            if len(tokens) <= 1:
                return False
            if all(t in _GENERIC_DATE_WORDS for t in tokens):
                return False

    if graph_type == "Money":
        # Money tối thiểu cần số hoặc đơn vị tiền phổ biến.
        has_number = any(ch.isdigit() for ch in norm)
        has_currency = any(k in lowered for k in ["usd", "vnd", "đ", "$", "triệu", "tỷ", "ty"])
        if not (has_number or has_currency):
            return False

    if graph_type == "Location":
        # Loại các location quá ngắn/cụt như "Hà", "San", "quận Cầu".
        if len(tokens) == 1 and len(tokens[0]) <= 3 and not any(ch.isdigit() for ch in norm):
            return False
        if len(tokens) == 2 and tokens[0] in _LOCATION_PREFIXES:
            # Chỉ giữ nếu KB xác nhận exact-name là LOCATION.
            if not (kb.kb_ready and kb.get_entity_type(norm) == "LOCATION"):
                return False

    return True


def build_graph(raw_entities: list[dict], text: str) -> GraphData:
    """
    Chuyển NER output thành GraphData.
    Quan hệ được lấy theo thứ tự ưu tiên:
      1. KB lookup (quan hệ thực từ corpus)
      2. Co-occurrence trong câu + KB lookup
      3. Rule-based fallback
    """
    raw_entities = merge_adjacent_entities(raw_entities)

    # ── Build entities ────────────────────────────────────────────────────────
    seen: dict[tuple, str] = {}
    entities: list[Entity] = []

    for raw in raw_entities:
        graph_type = TYPE_MAP.get(raw["ner_type"])
        if graph_type is None:
            continue

        name = _normalize_text(raw["text"].strip().strip(".,;:!?\"'"))
        if not _is_informative_entity(name, graph_type):
            continue

        # Nếu entity có trong KB thì ưu tiên loại từ KB để giảm lỗi NER cục bộ.
        if kb.kb_ready and len(name) >= 3:
            kb_ner_type = kb.get_entity_type(name)
            kb_graph_type = TYPE_MAP.get(kb_ner_type) if kb_ner_type else None
            if kb_graph_type:
                graph_type = kb_graph_type

        key = (name.lower(), graph_type)
        if key in seen:
            continue

        eid = f"E{len(entities) + 1}"
        seen[key] = eid
        entities.append(Entity(
            id=eid,
            name=name,
            type=graph_type,
            properties=[EntityProperty(key="NER Type", value=raw["ner_type"])],
        ))

    # ── Build relations từ co-occurrence trong câu ────────────────────────────
    relations: list[Relation] = []
    existing_pairs: set[tuple[str, str]] = set()

    for sent in split_sentences(text):
        ents_in_sent = [
            e for e in entities
            if re.search(r'\b' + re.escape(e.name) + r'\b', sent, re.IGNORECASE)
        ]
        pairs = [
            (ents_in_sent[i], ents_in_sent[j])
            for i in range(len(ents_in_sent))
            for j in range(i + 1, len(ents_in_sent))
        ]
        for src, tgt in pairs[:5]:
            # Tránh nối Date chung chung với quá nhiều node.
            if src.type == "Date" and not any(ch.isdigit() for ch in src.name):
                continue
            if tgt.type == "Date" and not any(ch.isdigit() for ch in tgt.name):
                continue
            if (src.id, tgt.id) not in existing_pairs:
                label = infer_relation_label(src, tgt)
                if label == "LIÊN QUAN ĐẾN":
                    continue
                relations.append(Relation(
                    source=src.id,
                    target=tgt.id,
                    label=label,
                ))
                existing_pairs.add((src.id, tgt.id))

    # ── Enrich từ KB: tìm quan hệ giữa các entity chưa có relation ───────────
    if kb.kb_ready and len(entities) > 1:
        kb_relations = kb.enrich_relations(entities, existing_pairs, max_per_entity=2)
        for rel_dict in kb_relations:
            relations.append(Relation(
                source=rel_dict["source"],
                target=rel_dict["target"],
                label=rel_dict["label"],
            ))

    return GraphData(entities=entities, relations=relations)


def predict_new_links(entities: list[Entity], relations: list[Relation]) -> list[Relation]:
    """Dự đoán liên kết mới — KB trước, fallback rule-based."""
    existing = {(r.source, r.target) for r in relations}
    predicted: list[Relation] = []

    for i, src in enumerate(entities):
        for j, tgt in enumerate(entities):
            if i >= j or (src.id, tgt.id) in existing:
                continue

            # Thử KB trước
            label = None
            if kb.kb_ready:
                label = kb.find_relation(src.name, tgt.name)

            # Fallback rule-based
            if not label:
                label = _RELATION_LABELS.get((src.type, tgt.type))

            if label and label != "LIÊN QUAN ĐẾN":
                predicted.append(Relation(
                    source=src.id, target=tgt.id,
                    label=label, isPredicted=True,
                ))
            if len(predicted) >= 5:
                return predicted

    return predicted
