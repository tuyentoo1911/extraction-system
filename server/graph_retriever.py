"""
Graph-first retriever for KGE chatbot.

Produces ScoredDocument items directly from the Knowledge Graph:
  - Direct entity cards with their relations
  - Direct triples between two mentioned entities
  - 2-hop indirect paths (for INTENT_RELATIONSHIP)
  - KB triples for mentioned entities

These are merged with BM25 results in hybrid_retrieve().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import knowledge_base as kb_mod
from rag import Document, build_index, retrieve
from query_understanding import (
    ParsedQuery,
    INTENT_RELATIONSHIP,
    INTENT_SUMMARY,
    MODE_GRAPH_FIRST,
    MODE_HYBRID,
)
from schemas import Entity, Relation

logger = logging.getLogger(__name__)

# Score weights
_SCORE_DIRECT_RELATION = 1.2
_SCORE_ENTITY_CARD     = 1.0
_SCORE_2HOP_PATH       = 0.9
_SCORE_KB_TRIPLE       = 0.7
_SCORE_BM25_RELATION   = 0.6
_SCORE_BM25_ENTITY     = 0.55
_SCORE_BM25_INSIGHT    = 0.53
_SCORE_BM25_METRICS    = 0.52
_SCORE_BM25_TEXT       = 0.5
_SCORE_BM25_KB         = 0.45

# How many graph docs to pull before blending
_GRAPH_TOP_K     = 20
_BM25_TOP_K      = 12
_BM25_TEXT_TOP_K = 6   # fewer text chunks for graph_first mode


@dataclass
class ScoredDocument:
    """A retrievable document with an attached relevance score."""
    doc: Document
    score: float
    source_type: str = "bm25"   # "graph" | "bm25"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _entity_by_name(name: str, entities: list[Entity]) -> Optional[Entity]:
    name_lower = name.lower()
    for e in sorted(entities, key=lambda x: len(x.name), reverse=True):
        if e.name.lower() == name_lower:
            return e
    for e in sorted(entities, key=lambda x: len(x.name), reverse=True):
        if name_lower in e.name.lower() or e.name.lower() in name_lower:
            return e
    return None


def _resolve_mentioned(pq: ParsedQuery, entities: list[Entity]) -> list[Entity]:
    resolved: list[Entity] = []
    seen_ids: set[str] = set()
    for name in pq.entities_mentioned:
        e = _entity_by_name(name, entities)
        if e and e.id not in seen_ids:
            resolved.append(e)
            seen_ids.add(e.id)
    return resolved


# ── Graph retrieval ───────────────────────────────────────────────────────────

def graph_retrieve(
    pq: ParsedQuery,
    entities: list[Entity],
    relations: list[Relation],
) -> list[ScoredDocument]:
    """
    Retrieve documents directly from the Knowledge Graph structure.
    Returns a ranked list of ScoredDocument (source_type='graph').
    """
    entity_map = {e.id: e for e in entities}
    results: list[ScoredDocument] = []
    seen: set[str] = set()

    def _add(doc: Document, score: float) -> None:
        key = doc.text[:100]
        if key not in seen:
            seen.add(key)
            results.append(ScoredDocument(doc=doc, score=score, source_type="graph"))

    mentioned = _resolve_mentioned(pq, entities)

    # ── Entity cards with all their direct relations ──────────────────────────
    for e in mentioned:
        rels = [r for r in relations if r.source == e.id or r.target == e.id]
        parts = [f"{e.name} là {e.type}."]
        if e.properties:
            for p in e.properties[:4]:
                parts.append(f"{e.name} có {p.key}: {p.value}.")
        for r in rels[:12]:
            src = entity_map.get(r.source)
            tgt = entity_map.get(r.target)
            sname = src.name if src else r.source
            tname = tgt.name if tgt else r.target
            pred  = " [dự đoán]" if r.isPredicted else ""
            parts.append(f"[{sname}] --({r.label})--> [{tname}]{pred}")
        _add(
            Document(
                text=" ".join(parts),
                source="graph_entity",
                metadata={"entity_id": e.id, "entity_name": e.name, "entity_type": e.type},
            ),
            score=_SCORE_ENTITY_CARD,
        )

    # ── Direct triples between any two mentioned entities ─────────────────────
    for i, ea in enumerate(mentioned):
        for eb in mentioned[i + 1:]:
            for r in relations:
                if (r.source == ea.id and r.target == eb.id) or \
                   (r.source == eb.id and r.target == ea.id):
                    src = entity_map.get(r.source)
                    tgt = entity_map.get(r.target)
                    sname = src.name if src else r.source
                    tname = tgt.name if tgt else r.target
                    pred  = " [dự đoán]" if r.isPredicted else ""
                    _add(
                        Document(
                            text=f"[{sname}] --({r.label})--> [{tname}]{pred}",
                            source="graph_relation",
                            metadata={"label": r.label, "source": sname, "target": tname,
                                      "is_predicted": r.isPredicted},
                        ),
                        score=_SCORE_DIRECT_RELATION,
                    )

    # ── 2-hop indirect paths (for relationship queries) ───────────────────────
    if pq.intent == INTENT_RELATIONSHIP and len(mentioned) >= 2:
        ea, eb = mentioned[0], mentioned[1]
        a_neighbors: dict[str, Relation] = {}
        b_neighbors: dict[str, Relation] = {}
        for r in relations:
            if r.source == ea.id:
                a_neighbors.setdefault(r.target, r)
            elif r.target == ea.id:
                a_neighbors.setdefault(r.source, r)
            if r.source == eb.id:
                b_neighbors.setdefault(r.target, r)
            elif r.target == eb.id:
                b_neighbors.setdefault(r.source, r)

        for mid_id in set(a_neighbors) & set(b_neighbors):
            mid = entity_map.get(mid_id)
            if not mid:
                continue
            ra, rb = a_neighbors[mid_id], b_neighbors[mid_id]
            text = (
                f"[{ea.name}] --({ra.label})--> [{mid.name}] "
                f"--({rb.label})--> [{eb.name}]"
            )
            _add(
                Document(
                    text=text,
                    source="graph_path",
                    metadata={"path_via": mid.name, "entity_a": ea.name, "entity_b": eb.name},
                ),
                score=_SCORE_2HOP_PATH,
            )

    # ── KB triples for mentioned entities ─────────────────────────────────────
    if kb_mod.kb_ready:
        seen_kb: set[str] = set()
        for name in pq.entities_mentioned[:5]:
            for t in kb_mod.get_entity_triples(name, limit=6):
                key = f"{t['subject']}|{t['relation']}|{t['object']}"
                if key in seen_kb:
                    continue
                seen_kb.add(key)
                rel  = t["relation"].replace("_", " ")
                conf = t.get("confidence", 0)
                _add(
                    Document(
                        text=f"{t['subject']} {rel} {t['object']} (KB, {conf:.0%}).",
                        source="kb_triple",
                        metadata=t,
                    ),
                    score=_SCORE_KB_TRIPLE,
                )

    return results


# ── Hybrid retrieval (graph + BM25) ──────────────────────────────────────────

def hybrid_retrieve(
    pq: ParsedQuery,
    input_text: str,
    entities: list[Entity],
    relations: list[Relation],
    insight_markdown: str = "",
    metrics_summary: str = "",
) -> list[ScoredDocument]:
    """
    Merge graph-first retrieval with BM25 retrieval.
    Score weighting depends on query mode and intent.
    """
    graph_docs = graph_retrieve(pq, entities, relations)

    # BM25 retrieval
    bm25_top_k = _BM25_TEXT_TOP_K if pq.mode == MODE_GRAPH_FIRST else _BM25_TOP_K
    try:
        index = build_index(
            input_text, entities, relations,
            insight_markdown=insight_markdown,
            metrics_summary=metrics_summary,
        )
        bm25_raw = retrieve(pq.raw, index, top_k=bm25_top_k)
    except Exception as exc:
        logger.warning("BM25 retrieval failed: %s", exc)
        bm25_raw = []

    # Assign scores to BM25 docs based on source type
    _bm25_source_score = {
        "relation":   _SCORE_BM25_RELATION,
        "entity":     _SCORE_BM25_ENTITY,
        "insight":    _SCORE_BM25_INSIGHT,
        "metrics":    _SCORE_BM25_METRICS,
        "input_text": _SCORE_BM25_TEXT,
        "kb_triple":  _SCORE_BM25_KB,
    }
    bm25_docs: list[ScoredDocument] = []
    for d in bm25_raw:
        base_score = _bm25_source_score.get(d.source, 0.4)
        bm25_docs.append(ScoredDocument(doc=d, score=base_score, source_type="bm25"))

    # For HYBRID mode: balance graph and BM25 (interleave by rank)
    # For GRAPH_FIRST: prioritize graph docs, append BM25 remainder
    if pq.mode == MODE_HYBRID or pq.intent == INTENT_SUMMARY:
        merged = _interleave(graph_docs, bm25_docs)
    else:
        # graph_first: graph docs first, then any BM25 docs not already covered
        graph_texts = {sd.doc.text[:100] for sd in graph_docs}
        extra_bm25 = [sd for sd in bm25_docs if sd.doc.text[:100] not in graph_texts]
        merged = graph_docs + extra_bm25

    logger.debug(
        "hybrid_retrieve: graph=%d bm25=%d merged=%d intent=%s mode=%s",
        len(graph_docs), len(bm25_docs), len(merged), pq.intent, pq.mode,
    )
    return merged


def _interleave(
    a: list[ScoredDocument],
    b: list[ScoredDocument],
) -> list[ScoredDocument]:
    """Merge two ranked lists by alternating items (dedup by text prefix)."""
    seen: set[str] = set()
    result: list[ScoredDocument] = []
    ia, ib = 0, 0
    while ia < len(a) or ib < len(b):
        if ia < len(a):
            sd = a[ia]; ia += 1
            key = sd.doc.text[:100]
            if key not in seen:
                seen.add(key); result.append(sd)
        if ib < len(b):
            sd = b[ib]; ib += 1
            key = sd.doc.text[:100]
            if key not in seen:
                seen.add(key); result.append(sd)
    return result
