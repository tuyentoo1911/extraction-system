"""
RAG (Retrieval-Augmented Generation) pipeline for the KGE chatbot.

Document sources indexed:
  1. Input text chunks  — passage-level windows from the user's source text
  2. Entity documents   — entity cards built from graph entities + relations
  3. KB triple docs     — formatted triples from the Knowledge Base corpus
  4. Insight chunks     — markdown from the Insight tab (optional)
  5. Metrics summary    — compact text from Metrics tab (optional)

Retrieval uses BM25 (keyword) for fast, dependency-light search.
The retrieved context is formatted for LLM consumption or rule-based enrichment.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

import knowledge_base as kb_mod
from schemas import Entity, Relation

logger = logging.getLogger(__name__)

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80
TOP_K = 12
MAX_CONTEXT_CHARS = 3000

@dataclass
class Document:
    """A retrievable chunk with source metadata."""
    text: str
    source: str          # "input_text" | "entity" | "kb_triple" | "relation" | "insight" | "metrics"
    metadata: dict = field(default_factory=dict)

@dataclass
class RAGIndex:
    """BM25 index over a list of Documents."""
    documents: list[Document]
    bm25: BM25Okapi
    content_hash: str

_cached_index: Optional[RAGIndex] = None

_VIET_WORD_RE = re.compile(r"[\w\u00C0-\u024F\u1EA0-\u1EF9\u0110\u0111]+", re.UNICODE)

def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _VIET_WORD_RE.findall(text) if len(w) > 1]

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows, breaking on sentence boundaries."""
    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?。\n])\s+', text.strip())
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    return chunks

def _build_insight_docs(insight_md: str) -> list[Document]:
    chunks = _chunk_text(insight_md)
    return [
        Document(text=c, source="insight", metadata={"chunk_idx": i})
        for i, c in enumerate(chunks)
    ]


def _build_metrics_docs(metrics_summary: str) -> list[Document]:
    if not metrics_summary or not metrics_summary.strip():
        return []
    chunks = _chunk_text(metrics_summary)
    return [
        Document(text=c, source="metrics", metadata={"chunk_idx": i})
        for i, c in enumerate(chunks)
    ]


def _build_text_docs(input_text: str) -> list[Document]:
    chunks = _chunk_text(input_text)
    return [
        Document(text=c, source="input_text", metadata={"chunk_idx": i})
        for i, c in enumerate(chunks)
    ]

def _build_entity_docs(
    entities: list[Entity],
    relations: list[Relation],
) -> list[Document]:
    entity_map = {e.id: e for e in entities}
    docs: list[Document] = []

    for e in entities:
        parts = [f"{e.name} là {e.type}."]

        if e.properties:
            for p in e.properties[:8]:
                parts.append(f"{e.name} có {p.key}: {p.value}.")

        rels_out = [r for r in relations if r.source == e.id]
        rels_in = [r for r in relations if r.target == e.id]

        for r in rels_out[:6]:
            target = entity_map.get(r.target)
            tname = target.name if target else r.target
            parts.append(f"{e.name} {r.label.replace('_', ' ')} {tname}.")

        for r in rels_in[:6]:
            source = entity_map.get(r.source)
            sname = source.name if source else r.source
            parts.append(f"{sname} {r.label.replace('_', ' ')} {e.name}.")

        docs.append(Document(
            text=" ".join(parts),
            source="entity",
            metadata={"entity_id": e.id, "entity_name": e.name, "entity_type": e.type},
        ))

    return docs

def _build_relation_docs(
    relations: list[Relation],
    entity_map: dict[str, Entity],
) -> list[Document]:
    docs: list[Document] = []
    for r in relations:
        src = entity_map.get(r.source)
        tgt = entity_map.get(r.target)
        sname = src.name if src else r.source
        tname = tgt.name if tgt else r.target
        label = r.label.replace("_", " ")
        pred = " (dự đoán)" if r.isPredicted else ""
        text = f"{sname} {label} {tname}{pred}."
        docs.append(Document(
            text=text,
            source="relation",
            metadata={"label": r.label, "source": sname, "target": tname},
        ))
    return docs

def _build_kb_docs(entity_names: list[str], max_per_entity: int = 5) -> list[Document]:
    """Fetch KB triples for mentioned entities."""
    if not kb_mod.kb_ready:
        return []

    docs: list[Document] = []
    seen: set[str] = set()

    for name in entity_names:
        triples = kb_mod.get_entity_triples(name, limit=max_per_entity)
        for t in triples:
            key = f"{t['subject']}|{t['relation']}|{t['object']}"
            if key in seen:
                continue
            seen.add(key)
            rel = t["relation"].replace("_", " ")
            conf = t.get("confidence", 0)
            text = f"{t['subject']} {rel} {t['object']} (KB, confidence {conf:.0%})."
            docs.append(Document(
                text=text,
                source="kb_triple",
                metadata={"subject": t["subject"], "relation": t["relation"], "object": t["object"]},
            ))

    return docs

def _content_hash(
    input_text: str,
    entities: list[Entity],
    relations: list[Relation],
    insight_markdown: str = "",
    metrics_summary: str = "",
) -> str:
    h = hashlib.md5()
    h.update(input_text.encode("utf-8", errors="ignore"))
    h.update(insight_markdown.encode("utf-8", errors="ignore"))
    h.update(metrics_summary.encode("utf-8", errors="ignore"))
    h.update(str(len(entities)).encode())
    h.update(str(len(relations)).encode())
    if entities:
        h.update(entities[0].id.encode())
    return h.hexdigest()[:12]

def build_index(
    input_text: str,
    entities: list[Entity],
    relations: list[Relation],
    insight_markdown: str = "",
    metrics_summary: str = "",
) -> RAGIndex:
    """Build or return cached BM25 index over all document sources."""
    global _cached_index
    ch = _content_hash(input_text, entities, relations, insight_markdown, metrics_summary)

    if _cached_index and _cached_index.content_hash == ch:
        return _cached_index

    entity_map = {e.id: e for e in entities}
    entity_names = [e.name for e in entities]

    docs: list[Document] = []
    docs.extend(_build_text_docs(input_text))
    docs.extend(_build_insight_docs(insight_markdown))
    docs.extend(_build_metrics_docs(metrics_summary))
    docs.extend(_build_entity_docs(entities, relations))
    docs.extend(_build_relation_docs(relations, entity_map))
    docs.extend(_build_kb_docs(entity_names, max_per_entity=5))

    if not docs:
        docs.append(Document(text="No data available.", source="empty"))

    tokenized = [_tokenize(d.text) for d in docs]
    bm25 = BM25Okapi(tokenized)

    idx = RAGIndex(documents=docs, bm25=bm25, content_hash=ch)
    _cached_index = idx

    logger.info(
        "RAG index built: %d docs (text=%d, insight=%d, metrics=%d, entity=%d, relation=%d, kb=%d)",
        len(docs),
        sum(1 for d in docs if d.source == "input_text"),
        sum(1 for d in docs if d.source == "insight"),
        sum(1 for d in docs if d.source == "metrics"),
        sum(1 for d in docs if d.source == "entity"),
        sum(1 for d in docs if d.source == "relation"),
        sum(1 for d in docs if d.source == "kb_triple"),
    )
    return idx

def retrieve(
    query: str,
    index: RAGIndex,
    top_k: int = TOP_K,
    source_filter: Optional[set[str]] = None,
) -> list[Document]:
    """Retrieve top-K documents by BM25 relevance."""
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = index.bm25.get_scores(tokens)

    scored_docs = list(zip(scores, index.documents))
    if source_filter:
        scored_docs = [(s, d) for s, d in scored_docs if d.source in source_filter]

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored_docs[:top_k] if _ > 0]

def retrieve_context(
    query: str,
    input_text: str,
    entities: list[Entity],
    relations: list[Relation],
    insight_markdown: str = "",
    metrics_summary: str = "",
    top_k: int = TOP_K,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """One-shot: build index + retrieve + format as context string."""
    index = build_index(input_text, entities, relations, insight_markdown, metrics_summary)
    docs = retrieve(query, index, top_k=top_k)

    if not docs:
        return "(No relevant context found.)"

    sections: dict[str, list[str]] = {
        "input_text": [],
        "insight": [],
        "metrics": [],
        "entity": [],
        "relation": [],
        "kb_triple": [],
    }

    total_chars = 0
    for d in docs:
        if total_chars + len(d.text) > max_chars:
            break
        sections.setdefault(d.source, []).append(d.text)
        total_chars += len(d.text)

    lines = [f"### Retrieved Context ({len(docs)} chunks, BM25)", ""]

    if sections["input_text"]:
        lines.append("**Source Text Excerpts:**")
        for t in sections["input_text"]:
            lines.append(f"> {t}")
        lines.append("")

    if sections["insight"]:
        lines.append("**Insight Report:**")
        for t in sections["insight"]:
            lines.append(f"> {t}")
        lines.append("")

    if sections["metrics"]:
        lines.append("**Metrics Summary:**")
        for t in sections["metrics"]:
            lines.append(f"- {t}")
        lines.append("")

    if sections["entity"]:
        lines.append("**Entity Information:**")
        for t in sections["entity"]:
            lines.append(f"- {t}")
        lines.append("")

    if sections["relation"]:
        lines.append("**Graph Relations:**")
        for t in sections["relation"]:
            lines.append(f"- {t}")
        lines.append("")

    if sections["kb_triple"]:
        lines.append("**Knowledge Base:**")
        for t in sections["kb_triple"]:
            lines.append(f"- {t}")
        lines.append("")

    return "\n".join(lines)

def retrieve_for_rule_based(
    query: str,
    input_text: str,
    entities: list[Entity],
    relations: list[Relation],
    insight_markdown: str = "",
    metrics_summary: str = "",
    top_k: int = 5,
) -> list[Document]:
    """Retrieve documents for rule-based enrichment (returns raw docs)."""
    index = build_index(input_text, entities, relations, insight_markdown, metrics_summary)
    return retrieve(query, index, top_k=top_k)
