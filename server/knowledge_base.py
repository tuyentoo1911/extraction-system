"""
Knowledge Base — Quản lý 6298 triples pre-built từ phobert-ner-final.
Load một lần lúc startup, cung cấp lookup nhanh theo tên entity.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent / "model" / "knowledge_graph_output" / "triples.json"

# ── State ─────────────────────────────────────────────────────────────────────
triples: list[dict] = []          # toàn bộ triple dicts
metadata: dict = {}

# Index: tên entity (lowercase) → set các chỉ số triple
_subject_index: dict[str, set[int]] = defaultdict(set)
_object_index:  dict[str, set[int]] = defaultdict(set)
# Index từng từ (cho fuzzy): word → set các chỉ số triple
_word_index:    dict[str, set[int]] = defaultdict(set)
_entity_type_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

kb_ready = False


def load_kb() -> None:
    """Load triples.json và build index."""
    global triples, metadata, kb_ready
    try:
        logger.info(f"Loading knowledge base from {KB_PATH}...")
        with open(KB_PATH, encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        triples  = data.get("triples", [])

        for i, t in enumerate(triples):
            subj = t["subject"].lower().strip()
            obj  = t["object"].lower().strip()
            conf = float(t.get("confidence", 0.5))

            _subject_index[subj].add(i)
            _object_index[obj].add(i)
            _entity_type_scores[subj][t["subject_type"]] += conf
            _entity_type_scores[obj][t["object_type"]] += conf

            # Word-level index cho tìm kiếm partial match
            for word in subj.split():
                if len(word) > 1:
                    _word_index[word].add(i)
            for word in obj.split():
                if len(word) > 1:
                    _word_index[word].add(i)

        kb_ready = True
        logger.info(
            f"✅ Knowledge base loaded: {len(triples)} triples, "
            f"{len(_subject_index)} subjects, {len(_object_index)} objects"
        )
    except Exception as e:
        logger.error(f"❌ Knowledge base load failed: {e}", exc_info=True)


# ── Lookup functions ──────────────────────────────────────────────────────────

def find_relation(subject: str, obj: str) -> str | None:
    """
    Tra cứu quan hệ giữa 2 entity trong KB.
    Trả về nhãn quan hệ (e.g. "LÃNH_ĐẠO") hoặc None nếu không tìm thấy.
    Lấy triple có confidence cao nhất.
    """
    subj_key = subject.lower().strip()
    obj_key  = obj.lower().strip()

    candidate_indices = _subject_index.get(subj_key, set()) & _object_index.get(obj_key, set())

    if not candidate_indices:
        # Thử chiều ngược lại
        candidate_indices = _subject_index.get(obj_key, set()) & _object_index.get(subj_key, set())

    if not candidate_indices:
        return None

    best = max(candidate_indices, key=lambda i: triples[i].get("confidence", 0))
    return triples[best]["relation"].replace("_", " ")


def search_entities(query: str, limit: int = 20) -> list[dict]:
    """
    Tìm kiếm entity trong KB theo tên (partial match).
    Trả về list entity dicts: {name, type, triple_count}.
    """
    q = query.lower().strip()
    if not q:
        return []

    # Tìm theo exact + word index
    candidate_indices: set[int] = set()
    for word in q.split():
        candidate_indices |= _word_index.get(word, set())

    # Tính điểm: entity nào xuất hiện trong query nhiều nhất
    entity_scores: dict[tuple, float] = defaultdict(float)
    for i in candidate_indices:
        t = triples[i]
        for entity, etype in [(t["subject"], t["subject_type"]), (t["object"], t["object_type"])]:
            if q in entity.lower():
                key = (entity, etype)
                entity_scores[key] += t.get("confidence", 0.5)

    sorted_entities = sorted(entity_scores.items(), key=lambda x: -x[1])

    # Đếm số triple cho mỗi entity
    result = []
    seen: set[str] = set()
    for (name, etype), score in sorted_entities:
        if name in seen:
            continue
        seen.add(name)
        n_key = name.lower().strip()
        count = len(_subject_index.get(n_key, set())) + len(_object_index.get(n_key, set()))
        result.append({"name": name, "type": etype, "triple_count": count, "score": round(score, 3)})
        if len(result) >= limit:
            break

    return result


def get_entity_triples(entity_name: str, limit: int = 50) -> list[dict]:
    """
    Lấy tất cả triples liên quan đến một entity (subject hoặc object).
    """
    key = entity_name.lower().strip()
    indices = _subject_index.get(key, set()) | _object_index.get(key, set())
    sorted_indices = sorted(indices, key=lambda i: -triples[i].get("confidence", 0))
    return [triples[i] for i in sorted_indices[:limit]]


def get_entity_type(entity_name: str) -> str | None:
    """
    Trả về loại entity phổ biến nhất trong KB cho exact-name match.
    Ví dụ: "Hà Nội" -> "LOCATION"
    """
    key = entity_name.lower().strip()
    scores = _entity_type_scores.get(key)
    if not scores:
        return None
    return max(scores.items(), key=lambda x: x[1])[0]


def enrich_relations(
    entities: list,   # list[Entity]
    existing_pairs: set[tuple[str, str]],
    max_per_entity: int = 3,
) -> list[dict]:
    """
    Với mỗi cặp entity trong graph, tra KB để lấy quan hệ thực.
    Trả về list relation dicts sẵn sàng tạo Relation object.
    """
    enriched = []
    seen_pairs: set[tuple[str, str]] = set(existing_pairs)

    for i, src in enumerate(entities):
        for tgt in entities[i + 1:]:
            if (src.id, tgt.id) in seen_pairs or (tgt.id, src.id) in seen_pairs:
                continue

            label = find_relation(src.name, tgt.name)
            if label:
                enriched.append({
                    "source": src.id,
                    "target": tgt.id,
                    "label": label,
                    "isPredicted": False,
                    "from_kb": True,
                })
                seen_pairs.add((src.id, tgt.id))
                if len(enriched) >= max_per_entity * len(entities):
                    return enriched

    return enriched


def get_stats() -> dict:
    return {
        "ready": kb_ready,
        "total_triples": len(triples),
        "total_subjects": len(_subject_index),
        "total_objects": len(_object_index),
        "relation_types": metadata.get("relation_types", []),
        "entity_types": metadata.get("entity_types", []),
    }
