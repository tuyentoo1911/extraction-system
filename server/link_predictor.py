"""
Link Prediction Pipeline — 4-tầng tích hợp ML model.

Tầng 1: ML Influence Scoring — dùng HistGradientBoostingClassifier dự đoán
         mức influence (LOW/MEDIUM/HIGH) của từng node → ưu tiên predict links
         giữa các high-influence nodes.
Tầng 2: KB Lookup — tra cứu triples.json với fuzzy matching + confidence score.
Tầng 3: Type-Pair Heuristic — predict dựa trên xác suất quan hệ theo loại entity.
Tầng 4: Graph Transitivity — suy luận từ cấu trúc đồ thị hiện có.

Output: list[Relation] sorted by confidence, deduped, max=dynamic.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import knowledge_base as kb

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent
_MODEL_PATH   = _BASE_DIR / "model" / "influence_predictor" / "influence_predictor.joblib"
_FE_DIR       = _BASE_DIR / "feature_engineering_output"
_FEATURE_FILE = _FE_DIR / "feature_names.json"
_SCALER_FILE  = _FE_DIR / "scaler_params.json"
_LABEL_FILE   = _FE_DIR / "label_encoder.json"

# ── Module-level singletons ────────────────────────────────────────────────────
_ml_model   = None
_scaler_center: dict[str, float] = {}
_scaler_scale:  dict[str, float] = {}
_selected_features: list[str]    = []
_label_names: list[str]          = []
_ml_ready   = False

# ── Type-Pair Heuristics ───────────────────────────────────────────────────────
# (src_type, tgt_type) → [(label, base_confidence), ...]
_TYPE_PAIR_HEURISTICS: dict[tuple[str, str], list[tuple[str, float]]] = {
    ("Organization", "Organization"): [
        ("partnered_with", 0.55),
        ("competitor_of",  0.50),
        ("supply_to",      0.45),
    ],
    ("Person", "Organization"): [
        ("founded",         0.60),
        ("former_employee", 0.55),
    ],
    ("Organization", "Person"): [
        ("founded",         0.55),
        ("former_employee", 0.50),
    ],
    ("Product", "Organization"): [
        ("developed_by", 0.65),
    ],
    ("Organization", "Location"): [
        ("headquartered_in", 0.58),
        ("has_office",       0.50),
    ],
    ("Event", "Location"): [
        ("held_in",     0.65),
        ("occurred_in", 0.55),
    ],
    ("Organization", "Industry"): [
        ("operates_in", 0.60),
    ],
    ("Product", "Event"): [
        ("launched_at", 0.58),
    ],
    ("Organization", "Date"): [
        ("founded_at", 0.52),
    ],
    ("Organization", "Money"): [
        ("has_revenue", 0.55),
        ("has_value",   0.48),
    ],
    ("Product", "Money"): [
        ("has_value", 0.52),
    ],
}

# ── Transitivity rules: (label_A, label_B) → implied label_C ──────────────────
# If A --label_A--> B and B --label_B--> C → maybe A --label_C--> C
_TRANSITIVITY_RULES: list[tuple[str, str, str, float]] = [
    ("founded",        "partnered_with",  "associated_with", 0.42),
    ("partnered_with", "operates_in",     "operates_in",     0.40),
    ("developed_by",   "headquartered_in","associated_with", 0.38),
]


def load_predictor() -> None:
    """Load ML model và scaler một lần lúc startup."""
    global _ml_model, _scaler_center, _scaler_scale, _selected_features
    global _label_names, _ml_ready

    try:
        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ml_model = joblib.load(_MODEL_PATH)

        with open(_FEATURE_FILE, encoding="utf-8") as f:
            feat_data = json.load(f)
        _selected_features = feat_data.get("selected_features", [])

        with open(_SCALER_FILE, encoding="utf-8") as f:
            scaler = json.load(f)
        _scaler_center = scaler.get("center", {})
        _scaler_scale  = scaler.get("scale",  {})

        with open(_LABEL_FILE, encoding="utf-8") as f:
            label_data = json.load(f)
        _label_names = label_data.get("label_names", ["LOW", "MEDIUM", "HIGH"])

        _ml_ready = True
        logger.info("Link predictor ML model loaded (%d features).", len(_selected_features))
    except Exception as exc:
        logger.warning("Link predictor ML model NOT loaded: %s — will use heuristics only.", exc)
        _ml_ready = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _robust_scale(value: float, feature: str) -> float:
    center = _scaler_center.get(feature, 0.0)
    scale  = _scaler_scale.get(feature, 1.0) or 1.0
    return (value - center) / scale


def _build_graph_stats(entities, relations) -> dict[str, dict]:
    """Tính các graph metrics nhẹ cho từng entity (không cần networkx)."""
    in_deg:  dict[str, int]   = defaultdict(int)
    out_deg: dict[str, int]   = defaultdict(int)
    neighbors: dict[str, set] = defaultdict(set)
    rel_types: dict[str, set] = defaultdict(set)

    for r in relations:
        out_deg[r.source] += 1
        in_deg[r.target]  += 1
        neighbors[r.source].add(r.target)
        neighbors[r.target].add(r.source)
        rel_types[r.source].add(r.label)
        rel_types[r.target].add(r.label)

    total_nodes = max(len(entities), 1)

    # Simple PageRank approximation (degree-based)
    total_edges = len(relations) or 1
    stats: dict[str, dict] = {}
    for e in entities:
        eid = e.id
        ind  = in_deg.get(eid, 0)
        outd = out_deg.get(eid, 0)
        totd = ind + outd
        nb   = neighbors.get(eid, set())

        # Degree-based PageRank proxy
        pr_approx = (ind + 1) / (total_edges + total_nodes)

        ego_types = {
            next((x.type for x in entities if x.id == nid), "Unknown")
            for nid in nb
        }

        stats[eid] = {
            "in_degree":               ind,
            "out_degree":              outd,
            "total_degree":            totd,
            "log_in_degree":           math.log1p(ind),
            "log_out_degree":          math.log1p(outd),
            "log_pagerank":            math.log1p(pr_approx * 1e6),
            "pr_per_degree":           pr_approx / max(totd, 1) * 1e6,
            "in_out_ratio":            ind / max(outd, 1e-9),
            "log_betweenness":         0.0,
            "bw_per_degree":           0.0,
            "is_high_pagerank":        1 if pr_approx > 0.01 else 0,
            "is_bridge":               1 if totd == 1 else 0,
            "ego_max_degree":          max((in_deg.get(n, 0) + out_deg.get(n, 0)) for n in nb) if nb else 0,
            "ego_mean_degree":         sum(in_deg.get(n, 0) + out_deg.get(n, 0) for n in nb) / max(len(nb), 1),
            "ego_density":             len(nb) / max(total_nodes - 1, 1),
            "ego_triangles":           0,
            "ego_type_diversity":      len(ego_types),
            "subject_relation_types":  len(rel_types.get(eid, set())),
            "object_relation_types":   len(rel_types.get(eid, set())),
            "subject_mean_confidence": 0.7,
            "object_mean_confidence":  0.7,
            "subject_mean_frequency":  1.0,
            "object_mean_frequency":   1.0,
            "object_total_frequency":  float(ind),
            "triple_total_appearances":float(totd),
            "triple_overall_conf_mean":0.375,
            "triple_rel_type_coverage":1.0,
            "entity_type_freq":        0.33,
            # type one-hot
            f"type_{e.type.upper()}":  1,
            "has_rel_HỢP_TÁC":         1 if "partnered_with" in rel_types.get(eid, set()) else 0,
            "has_rel_LÃNH_ĐẠO":       0,
            "has_rel_ĐẦU_TƯ":         0,
            "has_rel_SẢN_XUẤT":        1 if "developed_by" in rel_types.get(eid, set()) else 0,
        }
    return stats


def _score_entities_ml(entities, graph_stats: dict) -> dict[str, float]:
    """
    Dùng ML model để predict influence class → convert sang numeric score.
    Trả về {entity_id: score 0..1}.
    """
    if not _ml_ready or _ml_model is None:
        return {}

    label_score = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 1.0}
    scores: dict[str, float] = {}

    if not _numpy_available:
        logger.warning("numpy not available — ML scoring disabled.")
        return {}

    rows = []
    eids = []
    for e in entities:
        st = graph_stats.get(e.id, {})
        row = []
        for feat in _selected_features:
            raw = st.get(feat, 0.0)
            row.append(_robust_scale(float(raw), feat))
        rows.append(row)
        eids.append(e.id)

    if not rows:
        return {}

    try:
        X = np.array(rows, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = _ml_model.predict(X)
            probas = _ml_model.predict_proba(X)

        for i, eid in enumerate(eids):
            # Use probability of HIGH class as score
            high_idx = list(_ml_model.classes_).index("HIGH") if "HIGH" in list(_ml_model.classes_) else -1
            if high_idx >= 0:
                scores[eid] = float(probas[i][high_idx])
            else:
                scores[eid] = label_score.get(str(preds[i]), 0.5)
    except Exception as exc:
        logger.warning("ML scoring failed: %s", exc)
        return {}

    return scores


def _pk(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _validate_dr(label: str, src_type: str, tgt_type: str) -> bool:
    """Import lazily to avoid circular import."""
    from graph import _validate_domain_range
    return _validate_domain_range(label, src_type, tgt_type)


def _normalize(label: str) -> Optional[str]:
    from graph import _normalize_label
    return _normalize_label(label)


# ── Main API ───────────────────────────────────────────────────────────────────

def predict_new_links(entities, relations) -> list:
    """
    Pipeline predict 4 tầng. Trả về list[Relation] với isPredicted=True.
    """
    from schemas import Relation

    if not entities:
        return []

    existing_pk: set[tuple[str, str]] = {_pk(r.source, r.target) for r in relations}
    entity_map  = {e.id: e for e in entities}

    # ── Compute graph stats ────────────────────────────────────────────────────
    graph_stats = _build_graph_stats(entities, relations)

    # ── Tầng 1: ML influence scoring ──────────────────────────────────────────
    ml_scores = _score_entities_ml(entities, graph_stats)

    # Default score = degree-based proxy
    def node_score(eid: str) -> float:
        if eid in ml_scores:
            return ml_scores[eid]
        st = graph_stats.get(eid, {})
        d  = st.get("total_degree", 0)
        return min(0.5 + d * 0.05, 0.9)

    # ── Candidate pairs (prioritize high-score nodes) ─────────────────────────
    sorted_entities = sorted(entities, key=lambda e: node_score(e.id), reverse=True)

    # Dynamic max predictions
    max_preds = min(20, max(8, len(entities) // 2))

    predicted: list[dict] = []  # {source, target, label, confidence}
    seen_pk: set[tuple[str, str]] = set(existing_pk)

    def _add(src_id: str, tgt_id: str, label: str, conf: float) -> bool:
        pk = _pk(src_id, tgt_id)
        if pk in seen_pk:
            return False
        src = entity_map.get(src_id)
        tgt = entity_map.get(tgt_id)
        if not src or not tgt:
            return False
        nl = _normalize(label)
        if nl is None:
            return False
        if not _validate_dr(nl, src.type, tgt.type):
            return False
        seen_pk.add(pk)
        predicted.append({
            "source": src_id, "target": tgt_id,
            "label": nl, "confidence": round(conf, 3),
        })
        return True

    # ── Tầng 2: KB Lookup ─────────────────────────────────────────────────────
    if kb.kb_ready:
        for i, src in enumerate(sorted_entities):
            for tgt in sorted_entities[i + 1:]:
                if len(predicted) >= max_preds:
                    break
                raw_lbl = kb.find_relation(src.name, tgt.name)
                if not raw_lbl:
                    continue
                # Confidence = average of ML scores, boosted for KB match
                conf = 0.65 + 0.2 * (node_score(src.id) + node_score(tgt.id)) / 2
                conf = min(conf, 0.95)
                if not _add(src.id, tgt.id, raw_lbl, conf):
                    # Try reverse
                    _add(tgt.id, src.id, raw_lbl, conf * 0.9)

    # ── Tầng 3: Type-Pair Heuristic ───────────────────────────────────────────
    for i, src in enumerate(sorted_entities):
        if len(predicted) >= max_preds:
            break
        for tgt in sorted_entities[i + 1:]:
            if len(predicted) >= max_preds:
                break

            type_key = (src.type, tgt.type)
            rules = _TYPE_PAIR_HEURISTICS.get(type_key, [])
            if not rules:
                type_key_rev = (tgt.type, src.type)
                rules_rev = _TYPE_PAIR_HEURISTICS.get(type_key_rev, [])
                if rules_rev:
                    for lbl, base_conf in rules_rev:
                        boost = (node_score(src.id) + node_score(tgt.id)) / 2 * 0.15
                        _add(tgt.id, src.id, lbl, base_conf + boost)
                continue

            for lbl, base_conf in rules:
                boost = (node_score(src.id) + node_score(tgt.id)) / 2 * 0.15
                _add(src.id, tgt.id, lbl, base_conf + boost)

    # ── Tầng 4: Graph Transitivity ────────────────────────────────────────────
    # Build adjacency: src → {tgt: label}
    adj: dict[str, dict[str, str]] = defaultdict(dict)
    for r in relations:
        adj[r.source][r.target] = r.label

    for a_id, a_neighbors in adj.items():
        if len(predicted) >= max_preds:
            break
        for b_id, lbl_ab in a_neighbors.items():
            for c_id, lbl_bc in adj.get(b_id, {}).items():
                if c_id == a_id:
                    continue
                if len(predicted) >= max_preds:
                    break
                for rule_ab, rule_bc, implied, conf in _TRANSITIVITY_RULES:
                    if lbl_ab == rule_ab and lbl_bc == rule_bc:
                        _add(a_id, c_id, implied, conf)

    # ── Sort by confidence desc ────────────────────────────────────────────────
    predicted.sort(key=lambda x: x["confidence"], reverse=True)

    result = []
    for p in predicted[:max_preds]:
        result.append(Relation(
            source=p["source"],
            target=p["target"],
            label=p["label"],
            isPredicted=True,
            confidence=p["confidence"],
        ))

    logger.info(
        "predict_new_links: %d candidates → %d predictions (ML=%s, KB=%s)",
        len(entities) * (len(entities) - 1) // 2,
        len(result),
        "ready" if _ml_ready else "off",
        "ready" if kb.kb_ready else "off",
    )
    return result
