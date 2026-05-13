"""
re_model.py — PhoBERT Relation Extraction inference module.

Usage (lazy-loaded, thread-safe):
    from re_model import predict_relations
    rels = predict_relations(entities, sentences)

The model is loaded once on first call (lazy init).
If the model directory is missing, all calls return [] silently.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ── Model path ────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "phobert_re_best_model"

# ── Entity type → marker tag  (must match training notebook) ─────
_ENTITY_TYPE_TAG: dict[str, str] = {
    "Person":       "PER",
    "Organization": "ORG",
    "Location":     "LOC",
    "Product":      "PRD",
    "Event":        "EVT",
    "Date":         "DAT",
    "Money":        "MON",
    "Percent":      "PCT",
    "Industry":     "IND",
}

# ── Lazy-load state ───────────────────────────────────────────────
_lock      = threading.Lock()
_tokenizer = None
_model     = None
_id2label: dict[int, str] = {}
_max_len: int = 256
_ready: bool = False


def _load() -> bool:
    """Load tokenizer + model once. Returns True if successful."""
    global _tokenizer, _model, _id2label, _max_len, _ready

    if _ready:
        return True

    if not _MODEL_DIR.exists():
        logger.warning(
            "re_model: model dir not found at %s — PhoBERT RE disabled.", _MODEL_DIR
        )
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        meta_path = _MODEL_DIR / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            _max_len  = meta.get("max_len", 256)
            _id2label = {int(k): v for k, v in meta.get("id2label", {}).items()}
        
        logger.info("re_model: loading tokenizer from %s …", _MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))

        logger.info("re_model: loading model …")
        _model = AutoModelForSequenceClassification.from_pretrained(str(_MODEL_DIR))
        _model.eval()

        if torch.cuda.is_available():
            _model = _model.cuda()
            logger.info("re_model: model on GPU")
        else:
            logger.info("re_model: model on CPU")

        _ready = True
        logger.info("re_model: ready. Labels: %s", list(_id2label.values()))
        return True

    except Exception as exc:
        logger.error("re_model: failed to load — %s", exc, exc_info=True)
        return False


def _tag(entity_type: str) -> str:
    return _ENTITY_TYPE_TAG.get(entity_type, "UNK")


def _insert_markers(text: str, subj: str, s_type: str, obj: str, o_type: str) -> Optional[str]:
    """Wrap subject/object with special tokens. Returns None if either not found."""
    st  = f"[SUBJ-{_tag(s_type)}]"
    st_ = f"[/SUBJ-{_tag(s_type)}]"
    ot  = f"[OBJ-{_tag(o_type)}]"
    ot_ = f"[/OBJ-{_tag(o_type)}]"

    tl = text.lower()
    si = tl.find(subj.strip().lower())
    oi = tl.find(obj.strip().lower())
    if si == -1 or oi == -1:
        return None

    # Process from back to front to preserve indices
    parts = sorted(
        [(si, si + len(subj), st, st_),
         (oi, oi + len(obj),  ot, ot_)],
        key=lambda x: -x[0],
    )
    result = text
    for s, e, open_tag, close_tag in parts:
        result = result[:s] + open_tag + result[s:e] + close_tag + result[e:]
    return result


def predict_relations(
    entities,           # list[Entity] from schemas.py
    sentences: list[str],
    paired_ids: set | None = None,
    min_confidence: float = 0.70,
) -> list:
    """
    Run PhoBERT RE on all entity pairs within the same sentence.

    Args:
        entities:       list of Entity objects (must have .id, .name, .type)
        sentences:      list of sentence strings (from split_sentences)
        paired_ids:     set of frozenset({src_id, tgt_id}) for pairs that
                        already have a relation from upstream tiers.
                        Pairs in this set are SKIPPED (no duplicates).
        min_confidence: minimum softmax probability to accept a prediction

    Returns:
        list[Relation] with isPredicted=False (treated as extracted, not predicted)
    """
    with _lock:
        ok = _load()
    if not ok:
        return []

    from schemas import Relation

    # Normalise: always work with a mutable set of frozensets
    seen_pairs: set[frozenset] = set(paired_ids) if paired_ids else set()
    results: list[Relation] = []

    # For each sentence, find entities that appear in it
    for sent in sentences:
        sent_lower = sent.lower()
        sent_ents = [e for e in entities if e.name.lower() in sent_lower]
        if len(sent_ents) < 2:
            continue

        # Enumerate all ordered pairs (subj → obj)
        for subj_e in sent_ents:
            for obj_e in sent_ents:
                if obj_e.id == subj_e.id:
                    continue

                # Skip if this pair already has ANY relation from prior tiers
                pair_key = frozenset({subj_e.id, obj_e.id})
                if pair_key in seen_pairs:
                    continue

                # Build marked text
                marked = _insert_markers(
                    sent,
                    subj_e.name, subj_e.type,
                    obj_e.name,  obj_e.type,
                )
                if marked is None:
                    continue

                # Tokenize
                inputs = _tokenizer(
                    marked,
                    return_tensors="pt",
                    truncation=True,
                    max_length=_max_len,
                    padding=True,
                )
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # Inference
                with torch.no_grad():
                    logits = _model(**inputs).logits
                    probs  = torch.softmax(logits, dim=-1)[0]

                pred_id   = int(probs.argmax())
                pred_conf = float(probs[pred_id])
                pred_lbl  = _id2label.get(pred_id, "No_Relation")

                # Skip non-relations and low-confidence predictions
                if pred_lbl == "No_Relation" or pred_conf < min_confidence:
                    continue

                logger.debug(
                    "re_model: [%s] --(%s %.2f)--> [%s]",
                    subj_e.name, pred_lbl, pred_conf, obj_e.name,
                )

                results.append(Relation(
                    source=subj_e.id,
                    target=obj_e.id,
                    label=pred_lbl,
                    isPredicted=False,
                ))
                # Mark pair as taken so we don't add the reverse direction
                seen_pairs.add(pair_key)

    return results

