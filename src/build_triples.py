import json
from io import BytesIO
from pathlib import Path
import re
import sys

import pandas as pd
import requests
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


MODEL_NAME = "NlpHUST/ner-vietnamese-electra-base"
OUTPUT_PATH = Path("data/triples.json")
DATA_DIR = Path("data")
TEXT_COLUMNS = ("title", "content", "text")
LOCAL_DATASET_CANDIDATES = (
    Path("data/CSV/ner_labeled_data.csv"),
    Path("data/CSV/raw_data_2_pre.csv"),
    Path("data/ner_labeled_data.csv"),
    Path("data/raw_data_2_pre.csv"),
)
REMOTE_DATASETS = (
    ("ner_labeled_data.csv", "1z-8Bq17-Z4fmvoLJKRoCPH8PlQltCoUT"),
    ("raw_data_2_pre.csv", "1uNNF7slyL0d1LsyDMw6GNUlnB5WcFJug"),
)


def normalize_text(text: str) -> str:
    if any(
        marker in text
        for marker in (
            "\u00c3",
            "\u00c4",
            "\u00e1\u00ba",
            "\u00e1\u00bb",
            "\u00c6",
            "\u00e2",
        )
    ):
        for encoding in ("cp1252", "latin-1"):
            try:
                return text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
    return text


def clean_entity_text(text: str) -> str:
    cleaned = normalize_text(text).strip()
    cleaned = cleaned.replace("\u00d0", "\u0110").replace("\u00f0", "\u0111")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def safe_print(*args) -> None:
    text = " ".join(str(arg) for arg in args)

    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode(
            sys.stdout.encoding or "utf-8",
            errors="replace",
        ).decode(sys.stdout.encoding or "utf-8")
        print(safe_text)


def find_local_dataset() -> Path | None:
    for path in LOCAL_DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def get_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def extract_download_form(html: str) -> tuple[str, dict[str, str]] | None:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        html,
        re.DOTALL,
    )
    if not form_match:
        return None

    action = form_match.group(1)
    form_body = form_match.group(2)
    inputs = dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', form_body))
    return action, inputs


def download_drive_file_bytes(file_id: str) -> bytes:
    session = requests.Session()
    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    response = session.get(base_url, params=params, stream=True, timeout=120)
    response.raise_for_status()

    confirm_token = get_confirm_token(response)
    if confirm_token:
        response.close()
        params["confirm"] = confirm_token
        response = session.get(base_url, params=params, stream=True, timeout=120)
        response.raise_for_status()
    elif "text/html" in response.headers.get("content-type", ""):
        html = response.text
        response.close()
        form = extract_download_form(html)
        if form is None:
            raise RuntimeError("Unable to confirm Google Drive download")
        form_action, form_params = form
        response = session.get(form_action, params=form_params, stream=True, timeout=120)
        response.raise_for_status()

    chunks: list[bytes] = []
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            chunks.append(chunk)
    response.close()

    return b"".join(chunks)


def load_remote_dataframe(file_id: str) -> pd.DataFrame:
    csv_bytes = download_drive_file_bytes(file_id)
    return pd.read_csv(BytesIO(csv_bytes), encoding="utf-8-sig")


def load_dataframe() -> tuple[pd.DataFrame, str]:
    local_dataset = find_local_dataset()
    if local_dataset is not None:
        return pd.read_csv(local_dataset, encoding="utf-8-sig"), str(local_dataset)

    last_error: Exception | None = None
    for dataset_name, file_id in REMOTE_DATASETS:
        try:
            return load_remote_dataframe(file_id), f"google-drive:{dataset_name}"
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError("Unable to load dataset from local files or Google Drive") from last_error


def parse_bio_string(bio_text: str) -> list[tuple[str, str]]:
    entities: list[tuple[str, str]] = []
    current_tokens: list[str] = []
    current_type: str | None = None

    for raw_line in bio_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue

        token, tag = parts
        token = clean_entity_text(token)

        if tag == "O":
            if current_tokens and current_type is not None:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = []
            current_type = None
            continue

        if tag.startswith("B-"):
            if current_tokens and current_type is not None:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = tag[2:]
            continue

        if tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)
            continue

        if current_tokens and current_type is not None:
            entities.append((" ".join(current_tokens), current_type))
        current_tokens = []
        current_type = None

    if current_tokens and current_type is not None:
        entities.append((" ".join(current_tokens), current_type))

    return entities


def load_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    has_bio = {"title_bio", "content_bio"}.issubset(df.columns)

    for row_index, row in df.iterrows():
        source_id = int(row["id"]) if "id" in df.columns and pd.notna(row["id"]) else int(row_index)

        for column in TEXT_COLUMNS:
            value = row.get(column, "")
            if pd.isna(value):
                continue

            text = normalize_text(str(value).strip())
            if not text:
                continue

            record = {
                "source_id": source_id,
                "source_field": column,
                "text": text,
            }

            bio_column = f"{column}_bio"
            bio_value = row.get(bio_column, "") if has_bio else ""
            if isinstance(bio_value, str) and bio_value.strip():
                record["entities"] = parse_bio_string(bio_value)

            records.append(record)

    return records


def build_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )


def extract_relations(text: str, entities: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    relations: list[tuple[str, str, str]] = []
    orgs = [word for word, entity_type in entities if entity_type == "ORGANIZATION"]
    locs = [word for word, entity_type in entities if entity_type == "LOCATION"]

    if "\u0111\u1ea7u t\u01b0" in text and len(orgs) >= 2:
        relations.append((orgs[0], "invest_in", orgs[1]))

    if "h\u1ee3p t\u00e1c" in text and len(orgs) >= 2:
        relations.append((orgs[0], "partner_with", orgs[-1]))

    if "t\u1ea1i" in text and len(orgs) >= 2 and locs:
        relations.append((orgs[1], "located_in", locs[0]))

    return relations


def build_triples(records: list[dict]) -> list[dict]:
    triples: list[dict] = []
    ner = None

    for record in records:
        text = record["text"]
        entities = record.get("entities")

        if entities is None:
            if ner is None:
                ner = build_ner_pipeline()
            ner_result = ner(text)
            entities = [
                (clean_entity_text(entity["word"]), entity["entity_group"])
                for entity in ner_result
            ]

        relations = extract_relations(text, entities)

        for subject, relation, obj in relations:
            subject = clean_entity_text(subject)
            obj = clean_entity_text(obj)
            triples.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "source_id": record["source_id"],
                    "source_field": record["source_field"],
                    "source_text": text[:200],
                }
            )

    return triples


def clean_triples(triples: list[dict]) -> list[dict]:
    cleaned_triples: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for triple in triples:
        subject = clean_entity_text(triple["subject"])
        relation = triple["relation"].strip()
        obj = clean_entity_text(triple["object"])

        if not subject or not obj or subject == obj:
            continue

        triple_key = (subject, relation, obj)
        if triple_key in seen:
            continue
        seen.add(triple_key)

        cleaned_triples.append(
            {
                **triple,
                "subject": subject,
                "relation": relation,
                "object": obj,
            }
        )

    return cleaned_triples


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataframe, input_source = load_dataframe()
    records = load_records(dataframe)
    triples = clean_triples(build_triples(records))

    OUTPUT_PATH.write_text(
        json.dumps(triples, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    safe_print(f"Input source: {input_source}")
    safe_print(f"Loaded records: {len(records)}")
    safe_print(f"Generated triples: {len(triples)}")
    safe_print(f"Saved triples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
