"""
Knowledge Base — Quản lý 6298 triples pre-built từ phobert-ner-final.
Load một lần lúc startup, cung cấp lookup nhanh theo tên entity.

Cải tiến #2: find_relation hỗ trợ fuzzy matching
  - Fast path: exact lowercase match (O(1) dict lookup, không thay đổi)
  - Fuzzy path 1: token-overlap — "Công ty SaoVietTech" khớp "SaoVietTech"
  - Fuzzy path 2: difflib.SequenceMatcher — ratio >= FUZZY_THRESHOLD
  Cả hai fuzzy path đều sử dụng word_index đã build sẵn → tránh full scan.
"""

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent / "model" / "knowledge_graph_output" / "triples.json"

FUZZY_THRESHOLD = 0.75
TOKEN_OVERLAP_THRESHOLD = 0.6

triples: list[dict] = []          # toàn bộ triple dicts
metadata: dict = {}

_subject_index: dict[str, set[int]] = defaultdict(set)
_object_index:  dict[str, set[int]] = defaultdict(set)
_word_index:    dict[str, set[int]] = defaultdict(set)
_entity_type_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

_all_subjects: list[str] = []
_all_objects:  list[str] = []

kb_ready = False

def load_kb() -> None:
    """Load triples.json và build index."""
    global triples, metadata, kb_ready, _all_subjects, _all_objects
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

            for word in subj.split():
                if len(word) > 1:
                    _word_index[word].add(i)
            for word in obj.split():
                if len(word) > 1:
                    _word_index[word].add(i)

        _all_subjects = list(_subject_index.keys())
        _all_objects  = list(_object_index.keys())

        kb_ready = True
        logger.info(
            f" Knowledge base loaded: {len(triples)} triples, "
            f"{len(_subject_index)} subjects, {len(_object_index)} objects"
        )
    except Exception as e:
        logger.error(f" Knowledge base load failed: {e}", exc_info=True)

def _best_triple(candidate_indices: set[int]) -> str | None:
    """Trả về nhãn quan hệ của triple có confidence cao nhất."""
    if not candidate_indices:
        return None
    best = max(candidate_indices, key=lambda i: triples[i].get("confidence", 0))
    return triples[best]["relation"].replace("_", " ")

def _fuzzy_match_entity(query: str, index: dict[str, set[int]]) -> tuple[str | None, float]:
    """
    Tìm tên entity trong index khớp nhất với query.

    Chiến lược 2 bước:
      1. Token overlap — lọc nhanh các ứng viên có ít nhất 1 từ chung.
         Ưu điểm: O(k) với k = số triple chứa bất kỳ từ nào trong query.
      2. SequenceMatcher — đánh giá chính xác ratio trên ứng viên đã lọc.

    Trả về (tên entity tốt nhất, ratio) hoặc (None, 0.0) nếu không đạt ngưỡng.
    """
    q_lower = query.lower().strip()
    q_tokens = set(q_lower.split())

    candidate_keys: set[str] = set()
    for word in q_tokens:
        if len(word) > 1:
            for idx in _word_index.get(word, set()):
                t = triples[idx]
                subj_key = t["subject"].lower().strip()
                obj_key  = t["object"].lower().strip()
                if subj_key in index:
                    candidate_keys.add(subj_key)
                if obj_key in index:
                    candidate_keys.add(obj_key)

    if not candidate_keys:
        candidate_keys = set(index.keys())

    best_key: str | None = None
    best_ratio: float = 0.0

    for candidate in candidate_keys:
        c_tokens = set(candidate.split())
        if q_tokens and c_tokens:
            overlap = len(q_tokens & c_tokens) / len(q_tokens)
            if overlap < TOKEN_OVERLAP_THRESHOLD and best_ratio >= FUZZY_THRESHOLD:
                continue  # Đã có match tốt rồi, bỏ qua ứng viên kém

        ratio = SequenceMatcher(None, q_lower, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = candidate

    if best_ratio >= FUZZY_THRESHOLD and best_key is not None:
        return best_key, best_ratio
    return None, 0.0

def find_relation(subject: str, obj: str) -> str | None:
    """
    Tra cứu quan hệ giữa 2 entity trong KB.
    Trả về nhãn quan hệ hoặc None nếu không tìm thấy.

    Thứ tự lookup:
      1. Exact match (cả hai hướng)           — O(1)
      2. Fuzzy match subject + exact object   — gọi _fuzzy_match_entity()
      3. Exact subject + fuzzy match object
      4. Fuzzy cả hai
    Lấy triple có confidence cao nhất trong mỗi bước.
    """
    subj_key = subject.lower().strip()
    obj_key  = obj.lower().strip()

    candidate_indices = _subject_index.get(subj_key, set()) & _object_index.get(obj_key, set())
    if not candidate_indices:
        candidate_indices = _subject_index.get(obj_key, set()) & _object_index.get(subj_key, set())

    if candidate_indices:
        return _best_triple(candidate_indices)

    fuzzy_subj, _ = _fuzzy_match_entity(subj_key, _subject_index)
    if fuzzy_subj:
        candidate_indices = _subject_index.get(fuzzy_subj, set()) & _object_index.get(obj_key, set())
        if candidate_indices:
            logger.debug("fuzzy match subject: '%s' → '%s'", subject, fuzzy_subj)
            return _best_triple(candidate_indices)

    fuzzy_obj, _ = _fuzzy_match_entity(obj_key, _object_index)
    if fuzzy_obj:
        candidate_indices = _subject_index.get(subj_key, set()) & _object_index.get(fuzzy_obj, set())
        if candidate_indices:
            logger.debug("fuzzy match object: '%s' → '%s'", obj, fuzzy_obj)
            return _best_triple(candidate_indices)

    if fuzzy_subj and fuzzy_obj:
        candidate_indices = _subject_index.get(fuzzy_subj, set()) & _object_index.get(fuzzy_obj, set())
        if candidate_indices:
            logger.debug(
                "fuzzy match both: '%s'→'%s', '%s'→'%s'",
                subject, fuzzy_subj, obj, fuzzy_obj,
            )
            return _best_triple(candidate_indices)

    return None

def search_entities(query: str, limit: int = 20) -> list[dict]:
    """
    Tìm kiếm entity trong KB theo tên (partial match).
    Trả về list entity dicts: {name, type, triple_count}.
    """
    q = query.lower().strip()
    if not q:
        return []

    candidate_indices: set[int] = set()
    for word in q.split():
        candidate_indices |= _word_index.get(word, set())

    entity_scores: dict[tuple, float] = defaultdict(float)
    for i in candidate_indices:
        t = triples[i]
        for entity, etype in [(t["subject"], t["subject_type"]), (t["object"], t["object_type"])]:
            if q in entity.lower():
                key = (entity, etype)
                entity_scores[key] += t.get("confidence", 0.5)

    sorted_entities = sorted(entity_scores.items(), key=lambda x: -x[1])

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
