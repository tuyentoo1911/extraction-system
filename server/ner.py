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
import knowledge_base as kb

logger = logging.getLogger(__name__)

MAX_LEN = 256
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30
SENTENCE_CHAR_LIMIT = 500
MIN_ENTITY_CONFIDENCE = 0.55
MIN_SINGLE_TOKEN_NAME_CONFIDENCE = 0.78
LONG_TEXT_CHAR_THRESHOLD = 1800
LONG_TEXT_THRESHOLD_RELAX = 0.08
_MIN_CONF_BY_TYPE: dict[str, float] = {
    "PERSON": 0.58,
    "LOCATION": 0.62,
    "INDUSTRY": 0.70,
}
_LOCATION_PREFIX_HINTS = {
    "tp", "tp.", "thành", "thanh", "quận", "quan", "huyện", "huyen",
    "tỉnh", "tinh", "châu", "chau", "đảo", "dao",
}
_NOISE_BY_TYPE: dict[str, set[str]] = {
    "PERSON": {
        "song", "dong", "đồng", "thoi", "thời", "hang", "hàng", "trung",
    },
    "LOCATION": {
        "song", "dong", "đồng", "thoi", "thời", "hang", "hàng", "trung", "đầu",
    },
}
_GAZETTEER_BLOCKLIST = {
    "hàng", "hang", "trung", "đồng", "dong", "song", "đầu",
    "thời", "thoi", "đồng thời", "song song",
}
_INDUSTRY_GENERIC_TERMS = {
    "chuyển đổi", "kinh tế", "công nghiệp", "công nghệ", "viễn thông",
}
_PRIORITY_LOCATIONS = {
    "Hải Phòng",
    "Hà Nội",
    "Việt Nam",
    "Mỹ",
    "châu Âu",
    "TP. Hồ Chí Minh",
    "Thành phố Hồ Chí Minh",
}

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
    _ABBR = {"TP.", "Tp.", "tp.", "Inc.", "Corp.", "Co.", "Ltd.", "Dr.", "Mr.", "Ms.", "vs."}
    protected = text.strip()
    abbr_map: dict[str, str] = {}
    for abbr in _ABBR:
        placeholder = abbr.replace(".", "\x00")
        abbr_map[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    raw = re.split(r'(?<=[.!?\n])\s+|\n{2,}', protected)
    sentences: list[str] = []

    for seg in raw:
        seg = seg.strip()
        for placeholder, original in abbr_map.items():
            seg = seg.replace(placeholder, original)
        if not seg:
            continue
        if len(seg) <= SENTENCE_CHAR_LIMIT:
            sentences.append(seg)
        else:
            while seg:
                if len(seg) <= SENTENCE_CHAR_LIMIT:
                    sentences.append(seg)
                    break
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
            all_entities.extend(_predict_sentence(words, torch))
        else:
            seen_spans: set[tuple[int, int, str]] = set()
            for chunk_words, start_offset in chunks:
                chunk_entities = _predict_sentence(chunk_words, torch)
                for ent in chunk_entities:
                    global_words = [w + start_offset for w in ent.get("words", [])]
                    if not global_words:
                        continue
                    span_key = (global_words[0], global_words[-1], ent["ner_type"])
                    if span_key not in seen_spans:
                        seen_spans.add(span_key)
                        all_entities.append({**ent, "words": global_words})

    all_entities = post_process_entities(text, all_entities)
    all_entities = gazetteer_scan(text, all_entities)
    all_entities = _recover_priority_locations(text, all_entities)
    # Gazetteer có thể thêm candidate quá generic; lọc một lần ở cuối pipeline.
    return _drop_obvious_noise_entities(all_entities, text_len=len(text))

def post_process_entities(text: str, entities: list[dict]) -> list[dict]:
    """
    Sửa các lỗi thường gặp của PhoBERT NER:
    - Ranh giới sai (thừa tiền tố, thiếu hậu tố chữ/số)
    - Gán nhãn sai loại (VD: Hồ Chí Minh -> LOCATION)
    """
    processed = []
    
    for e in entities:
        e_text = e["text"].strip(".,;:()[]{}'\" \t\n")
        e_type = e["ner_type"]
        
        if e_type == "EVENT":
            e_text = re.sub(r"^(tại|vào|trong|ở)?\s*(sự kiện|chương trình|lễ hội|buổi|cuộc)?\s*", "", e_text, flags=re.IGNORECASE).strip()
            
        start_idx = text.find(e_text)
        if start_idx != -1:
            end_idx = start_idx + len(e_text)
            after_text = text[end_idx:]
            
            if e_text.endswith("Khoa Hà") and after_text.startswith(" Nội"):
                e_text += " Nội"
            elif e_type == "MONEY":
                m = re.match(r"^\s*(USD|VNĐ|VND|đồng|đô|euro|đ)\b", after_text, flags=re.IGNORECASE)
                if m:
                    e_text += m.group(0)
            elif e_type == "DATE" and e_text.lower().endswith("năm"):
                m = re.match(r"^\s*\d{4}\b", after_text)
                if m:
                    e_text += m.group(0)
                    
        if e_text in ["Hồ Chí Minh", "TP. Hồ Chí Minh", "Thành phố Hồ Chí Minh", "Hà Nội"]:
            e_type = "LOCATION"
            
        if not e_text:
            continue
            
        processed.append({
            "text": e_text.strip(),
            "ner_type": e_type,
            "words": e.get("words", []),
            "confidence": float(e.get("confidence", 0.0)),
        })
        
    # Tránh lọc sớm để không làm rơi recall ở đoạn văn dài;
    # lọc chính được thực hiện một lần ở cuối run_ner().
    return _merge_adjacent_entities(text, processed)

def _normalize_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _effective_thresholds(text_len: int) -> tuple[dict[str, float], float, float]:
    """
    Tính ngưỡng động theo độ dài văn bản.
    Văn bản dài được nới nhẹ threshold để tăng recall.
    """
    type_thresholds = dict(_MIN_CONF_BY_TYPE)
    base_min = MIN_ENTITY_CONFIDENCE
    single_token_min = MIN_SINGLE_TOKEN_NAME_CONFIDENCE

    if text_len >= LONG_TEXT_CHAR_THRESHOLD:
        for k, v in type_thresholds.items():
            type_thresholds[k] = max(0.45, v - LONG_TEXT_THRESHOLD_RELAX)
        base_min = max(0.45, base_min - LONG_TEXT_THRESHOLD_RELAX)
        single_token_min = max(0.68, single_token_min - 0.05)

    return type_thresholds, base_min, single_token_min

def _drop_obvious_noise_entities(entities: list[dict], text_len: int = 0) -> list[dict]:
    """Loại bỏ các token nhiễu thường bị model gán nhãn PERSON/LOCATION."""
    type_thresholds, base_min, single_token_min = _effective_thresholds(text_len)
    cleaned: list[dict] = []
    for ent in entities:
        ent_text = ent.get("text", "").strip()
        ent_type = ent.get("ner_type", "")
        ent_conf = float(ent.get("confidence", 0.0))
        if not ent_text:
            continue

        min_conf = type_thresholds.get(ent_type, base_min)
        if ent_conf < min_conf:
            continue

        # Chỉ lọc mạnh với entity quá ngắn (1 token) để tránh xóa thực thể hợp lệ.
        if len(ent_text.split()) == 1:
            noise_words = _NOISE_BY_TYPE.get(ent_type, set())
            if _normalize_token(ent_text) in noise_words:
                continue
            if ent_type in {"PERSON", "LOCATION"}:
                # Single-token PERSON/LOCATION cần confidence cao hơn.
                if ent_conf < single_token_min:
                    continue
                if not _looks_like_named_token(ent_text, ent_type):
                    continue
            if ent_type == "INDUSTRY":
                norm = _normalize_token(ent_text)
                if norm in _INDUSTRY_GENERIC_TERMS:
                    continue
        if ent_type == "INDUSTRY":
            norm = _normalize_token(ent_text)
            # Chặn cụm quá generic hay gây over-detect trong văn bản kinh tế tổng quát.
            if norm in _INDUSTRY_GENERIC_TERMS:
                continue
        cleaned.append(ent)
    return cleaned

def _looks_like_named_token(text: str, ner_type: str) -> bool:
    token = text.strip()
    if not token:
        return False
    # Ưu tiên token có chữ hoa đầu từ hoặc all-caps (ví dụ "Mỹ", "EU").
    if token[0].isupper() or token.isupper():
        return True
    if ner_type == "LOCATION" and _normalize_token(token) in _LOCATION_PREFIX_HINTS:
        return True
    return False

def _locate_entity_spans(text: str, entities: list[dict]) -> list[dict]:
    """Tìm span tuần tự để xử lý merge entity liền kề ổn định hơn text.find()."""
    spans: list[dict] = []
    cursor = 0
    for ent in entities:
        ent_text = ent.get("text", "").strip()
        if not ent_text:
            continue
        start = text.find(ent_text, cursor)
        if start == -1:
            start = text.find(ent_text)
        if start == -1:
            continue
        end = start + len(ent_text)
        cursor = end
        spans.append({
            "start": start,
            "end": end,
            "text": ent_text,
            "ner_type": ent.get("ner_type", ""),
            "words": ent.get("words", []),
            "confidence": float(ent.get("confidence", 0.0)),
        })
    return spans

def _merge_adjacent_entities(text: str, entities: list[dict]) -> list[dict]:
    """
    Gộp entity liền kề cùng type nếu giữa chúng chỉ là khoảng trắng/newline.
    Giảm lỗi tách tên riêng kiểu "Phạm Nhật" + "Vượng".
    """
    spans = _locate_entity_spans(text, entities)
    if not spans:
        return entities

    merged: list[dict] = []
    i = 0
    while i < len(spans):
        cur = spans[i].copy()
        j = i + 1
        while j < len(spans):
            nxt = spans[j]
            if nxt["ner_type"] != cur["ner_type"]:
                break
            gap = text[cur["end"]:nxt["start"]]
            if not gap or not gap.isspace():
                break
            cur["text"] = f"{cur['text']} {nxt['text']}".strip()
            cur["end"] = nxt["end"]
            cur["words"] = (cur.get("words", []) or []) + (nxt.get("words", []) or [])
            cur["confidence"] = max(
                float(cur.get("confidence", 0.0)),
                float(nxt.get("confidence", 0.0)),
            )
            j += 1
        merged.append({
            "text": cur["text"],
            "ner_type": cur["ner_type"],
            "words": cur.get("words", []),
            "confidence": float(cur.get("confidence", 0.0)),
        })
        i = j
    return merged

def gazetteer_scan(text: str, ner_results: list[dict]) -> list[dict]:
    """
    Quét lại văn bản để tìm các entity KB bằng regex word-boundary.
    Track vị trí (overlap) chuẩn xác để không sinh ra rác (substring) hay phá vỡ model entities.
    """
    if not hasattr(kb, 'kb_ready') or not kb.kb_ready:
        return ner_results
        
    occupied = []
    for e in ner_results:
        start_idx = text.find(e["text"])
        if start_idx != -1:
            occupied.append((start_idx, start_idx + len(e["text"])))
            
    candidates = sorted(list(set(kb._all_subjects) | set(kb._all_objects)), key=len, reverse=True)
    
    for ent in candidates:
        if len(ent) <= 3:
            continue
        if _is_generic_gazetteer_candidate(ent):
            continue
            
        esc_ent = re.escape(ent)
        pattern = re.compile(rf"(?<!\w){esc_ent}(?!\w)", re.IGNORECASE | re.UNICODE)
        
        for match in pattern.finditer(text):
            m_start, m_end = match.start(), match.end()
            
            is_overlap = any(max(m_start, o[0]) < min(m_end, o[1]) for o in occupied)
            if not is_overlap:
                ner_type = kb.get_entity_type(ent) or "ORGANIZATION"
                ner_results.append({
                    "text": match.group(0),
                    "ner_type": ner_type,
                    "words": [],
                    "confidence": 1.0,
                })
                occupied.append((m_start, m_end))
                
    return ner_results

def _is_generic_gazetteer_candidate(ent: str) -> bool:
    """
    Chặn các candidate KB quá generic (từ phổ thông 1 token, không viết hoa).
    Tránh quét KB sinh LOCATION/ORG rác từ văn bản thường.
    """
    norm = _normalize_token(ent)
    if not norm:
        return True
    if norm in _GAZETTEER_BLOCKLIST:
        return True

    parts = norm.split()
    if len(parts) == 1:
        token = parts[0]
        # 1-token lowercase thuần chữ thường rất dễ là từ chức năng.
        if token.isalpha() and token == token.lower():
            return True
    return False

def _recover_priority_locations(text: str, entities: list[dict]) -> list[dict]:
    """
    Bổ sung một số địa danh quan trọng hay bị miss bởi model.
    Chỉ thêm khi chưa có span overlap để tránh duplicate.
    """
    if not text.strip():
        return entities

    occupied: list[tuple[int, int]] = []
    for ent in entities:
        ent_text = ent.get("text", "").strip()
        if not ent_text:
            continue
        start = text.find(ent_text)
        if start != -1:
            occupied.append((start, start + len(ent_text)))

    for loc in _PRIORITY_LOCATIONS:
        pattern = re.compile(rf"(?<!\w){re.escape(loc)}(?!\w)", re.IGNORECASE | re.UNICODE)
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            overlap = any(max(s, a) < min(e, b) for a, b in occupied)
            if overlap:
                continue
            entities.append({
                "text": m.group(0),
                "ner_type": "LOCATION",
                "words": [],
                "confidence": 0.95,
            })
            occupied.append((s, e))
    return entities

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
    probs = torch.softmax(logits, dim=2)[0].cpu().numpy()
    return _decode_bio(words, word_ids, predictions, probs)

def _decode_bio(
    words: list[str],
    word_ids: list[Optional[int]],
    predictions,
    probs,
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
        token_conf = float(probs[token_idx][int(predictions[token_idx])])
        prev_word_id = word_id

        if label.startswith("B-"):
            if current:
                confs = current.pop("_token_confidences", [])
                current["confidence"] = float(sum(confs) / max(1, len(confs)))
                entities.append(current)
            current = {
                "text": word,
                "ner_type": label[2:],
                "words": [word_id],
                "_token_confidences": [token_conf],
            }

        elif label.startswith("I-") and current:
            ner_type = label[2:]
            if ner_type == current["ner_type"]:
                current["text"] += " " + word
                current["words"].append(word_id)
                current["_token_confidences"].append(token_conf)
            else:
                confs = current.pop("_token_confidences", [])
                current["confidence"] = float(sum(confs) / max(1, len(confs)))
                entities.append(current)
                current = {
                    "text": word,
                    "ner_type": ner_type,
                    "words": [word_id],
                    "_token_confidences": [token_conf],
                }

        else:  # O
            if current:
                confs = current.pop("_token_confidences", [])
                current["confidence"] = float(sum(confs) / max(1, len(confs)))
                entities.append(current)
                current = None

    if current:
        confs = current.pop("_token_confidences", [])
        current["confidence"] = float(sum(confs) / max(1, len(confs)))
        entities.append(current)

    return entities
