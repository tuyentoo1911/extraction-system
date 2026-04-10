"""Load và quản lý NER model (HuggingFace SafeTensors format)."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "model" / "phobert-ner-final"

# Biến global — được set bởi load_model(), dùng ở ner.py
ner_model     = None
ner_tokenizer = None
id2label: dict[int, str] = {}   # {0: "O", 1: "B-DATE", ...}
device        = None
model_ready   = False
model_error   = None


def load_model() -> None:
    """Load RobertaForTokenClassification từ thư mục model/."""
    global ner_model, ner_tokenizer, id2label, device, model_ready, model_error
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        TOKENIZER_SOURCE = "vinai/phobert-base"
        logger.info(f"Loading tokenizer from {TOKENIZER_SOURCE}...")
        try:
            from transformers import PhobertTokenizer
            ner_tokenizer = PhobertTokenizer.from_pretrained(TOKENIZER_SOURCE)
        except Exception:
            ner_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)

        logger.info(f"Loading model from {MODEL_DIR}...")
        ner_model = AutoModelForTokenClassification.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        ner_model.to(device)
        ner_model.eval()

        id2label = {int(k): v for k, v in ner_model.config.id2label.items()}
        logger.info(f"Labels ({len(id2label)}): {list(id2label.values())}")

        model_ready = True
        logger.info(" Model loaded successfully!")

    except Exception as e:
        model_error = str(e)
        logger.error(f" Model load failed: {e}", exc_info=True)
