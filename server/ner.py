"""Chạy NER inference trên văn bản."""

import re
import logging
from typing import Optional

import model as model_state

logger = logging.getLogger(__name__)

MAX_LEN = 256  # PhoBERT/RoBERTa max_position_embeddings = 258


def split_sentences(text: str) -> list[str]:
    """Chia văn bản thành câu theo dấu câu và xuống dòng."""
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def run_ner(text: str) -> list[dict]:
    """Chạy NER trên toàn bộ văn bản, trả về list raw entity dicts."""
    import torch
    all_entities: list[dict] = []
    for sent in split_sentences(text):
        words = sent.split()
        if words:
            all_entities.extend(_predict_sentence(words, torch))
    return all_entities


def _predict_sentence(words: list[str], torch) -> list[dict]:
    """Chạy NER trên một câu."""
    tokenizer = model_state.ner_tokenizer
    model     = model_state.ner_model
    dev       = model_state.device

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    # Build word_ids thủ công (slow tokenizer không hỗ trợ .word_ids())
    word_ids: list[Optional[int]] = [None]  # [CLS] / <s>
    for word_idx, word in enumerate(words):
        sub_tokens = tokenizer.tokenize(word) or [tokenizer.unk_token]
        word_ids.extend([word_idx] * len(sub_tokens))
        if len(word_ids) >= MAX_LEN - 1:
            break
    word_ids.append(None)                           # [SEP] / </s>
    word_ids += [None] * (MAX_LEN - len(word_ids)) # padding

    with torch.no_grad():
        logits = model(
            input_ids=encoding["input_ids"].to(dev),
            attention_mask=encoding["attention_mask"].to(dev),
        ).logits

    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    return _decode_bio(words, word_ids, predictions)


def _decode_bio(
    words: list[str],
    word_ids: list[Optional[int]],
    predictions,
) -> list[dict]:
    """Chuyển BIO predictions thành list entity dicts."""
    id2label = model_state.id2label
    entities: list[dict] = []
    current:  Optional[dict] = None
    prev_word_id: Optional[int] = None

    for token_idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            prev_word_id = word_id
            continue

        label = id2label.get(int(predictions[token_idx]), "O")
        word  = words[word_id]
        prev_word_id = word_id

        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"text": word, "ner_type": label[2:], "words": [word_id]}

        elif label.startswith("I-") and current:
            ner_type = label[2:]
            if ner_type == current["ner_type"]:
                current["text"] += " " + word
                current["words"].append(word_id)
            else:
                entities.append(current)
                current = {"text": word, "ner_type": ner_type, "words": [word_id]}

        else:  # O
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    return entities
