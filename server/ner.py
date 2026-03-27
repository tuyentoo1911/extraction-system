"""Chạy NER inference trên văn bản.

Cải tiến #3: Xử lý văn bản dài
  - MAX_LEN giữ nguyên 256 (PhoBERT limit), nhưng text dài được chunk
    theo sliding window với overlap để không mất entity ở ranh giới.
  - split_sentences() được tăng cường: ngoài dấu câu còn tách theo
    newline kép và độ dài ký tự tối đa (SENTENCE_CHAR_LIMIT).
  - _chunk_words() chia list từ thành các chunk có overlap, đảm bảo
    entity nhiều từ nằm ở biên không bị cắt đứt.
  - Deduplication: entity trùng (text + type) từ các chunk overlap
    được loại bỏ theo vị trí offset toàn cục.
"""

import re
import logging
from typing import Optional

import model as model_state

logger = logging.getLogger(__name__)

# PhoBERT/RoBERTa max_position_embeddings = 258 → thực dùng 256 (trừ [CLS]/[SEP])
MAX_LEN = 256
# Window size (số từ) mỗi chunk gửi vào model
CHUNK_SIZE = 200
# Overlap (số từ) giữa 2 chunk liên tiếp — đủ để entity ở biên vẫn đầy đủ
CHUNK_OVERLAP = 30
# Độ dài ký tự tối đa mỗi "câu" trước khi cắt tiếp theo từ
SENTENCE_CHAR_LIMIT = 500


def split_sentences(text: str) -> list[str]:
    """
    Chia văn bản thành câu theo:
      1. Dấu câu kết thúc (.  !  ?)
      2. Xuống dòng (\\n)
      3. Chuỗi dài không có dấu câu sẽ bị cắt tại SENTENCE_CHAR_LIMIT

    Cải tiến: regex gốc chỉ split theo lookbehind [.!?\\n] — không tách
    được đoạn PDF extract dài không dấu câu. Phiên bản mới thêm bước
    cắt theo ký tự để đảm bảo mỗi câu không vượt SENTENCE_CHAR_LIMIT.
    """
    # Bước 1: split theo dấu câu và xuống dòng
    raw = re.split(r'(?<=[.!?\n])\s+|\n{2,}', text.strip())
    sentences: list[str] = []

    for seg in raw:
        seg = seg.strip()
        if not seg:
            continue
        # Bước 2: cắt tiếp nếu đoạn còn quá dài (không có dấu câu)
        if len(seg) <= SENTENCE_CHAR_LIMIT:
            sentences.append(seg)
        else:
            # Cắt tại ranh giới từ gần nhất với SENTENCE_CHAR_LIMIT
            while seg:
                if len(seg) <= SENTENCE_CHAR_LIMIT:
                    sentences.append(seg)
                    break
                # Tìm khoảng trắng gần nhất để cắt
                cut = seg.rfind(" ", 0, SENTENCE_CHAR_LIMIT)
                if cut == -1:
                    cut = SENTENCE_CHAR_LIMIT
                sentences.append(seg[:cut].strip())
                seg = seg[cut:].strip()

    return sentences


def _chunk_words(words: list[str]) -> list[tuple[list[str], int]]:
    """
    Chia list từ thành các chunk [(words_chunk, start_offset), ...].

    Mỗi chunk có tối đa CHUNK_SIZE từ.
    Chunk liên tiếp overlap nhau CHUNK_OVERLAP từ để entity ở biên không bị mất.
    Trả về list (chunk_words, start_word_idx_in_original).
    """
    if len(words) <= CHUNK_SIZE:
        return [(words, 0)]

    chunks: list[tuple[list[str], int]] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append((words[start:end], start))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def run_ner(text: str) -> list[dict]:
    """
    Chạy NER trên toàn bộ văn bản, trả về list raw entity dicts.

    Quy trình:
      1. split_sentences() — tách câu (hỗ trợ text dài không dấu câu)
      2. Với mỗi câu, _chunk_words() — chia thành chunk nếu câu dài
      3. _predict_sentence() — inference từng chunk
      4. Dedup entity trùng (text + type) ở vùng overlap giữa 2 chunk
    """
    import torch
    all_entities: list[dict] = []

    for sent in split_sentences(text):
        words = sent.split()
        if not words:
            continue

        chunks = _chunk_words(words)
        if len(chunks) == 1:
            # Câu ngắn — đường nhanh, không cần dedup
            all_entities.extend(_predict_sentence(words, torch))
        else:
            # Câu dài — chạy từng chunk rồi dedup theo offset
            seen_spans: set[tuple[int, int, str]] = set()
            for chunk_words, start_offset in chunks:
                chunk_entities = _predict_sentence(chunk_words, torch)
                for ent in chunk_entities:
                    # Chuyển word positions về offset toàn cục
                    global_words = [w + start_offset for w in ent.get("words", [])]
                    if not global_words:
                        continue
                    span_key = (global_words[0], global_words[-1], ent["ner_type"])
                    if span_key not in seen_spans:
                        seen_spans.add(span_key)
                        all_entities.append({**ent, "words": global_words})

    return all_entities


def _predict_sentence(words: list[str], torch) -> list[dict]:
    """Chạy NER inference trên một list từ (đã đảm bảo <= CHUNK_SIZE từ)."""
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
