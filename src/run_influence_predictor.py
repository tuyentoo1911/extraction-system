from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _load_bundle(bundle_dir: Path):
    meta_path = bundle_dir / "metadata.json"
    model_path = bundle_dir / "influence_predictor.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names: list[str] = metadata["feature_names"]
    int_to_label: dict[str, str] = metadata["label_encoder"]["int_to_label"]

    model = joblib.load(model_path)
    return model, metadata, feature_names, int_to_label


def _read_input_json(path: str | None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return json.loads(sys.stdin.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference using influence_predictor.joblib")
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default=None,
        help="Directory containing influence_predictor.joblib + metadata.json",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON. If omitted, read from stdin.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else (base_dir / "model" / "influence_predictor")

    model, _metadata, feature_names, int_to_label = _load_bundle(bundle_dir)
    payload = _read_input_json(args.input)

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError('Input JSON must contain non-empty list field "items".')

    rows: list[dict[str, Any]] = []
    ids: list[str] = []

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{i}] must be an object.")
        item_id = str(item.get("id", i))
        ids.append(item_id)

        if "vector" in item:
            vec = item["vector"]
            if not isinstance(vec, list) or len(vec) != len(feature_names):
                raise ValueError(
                    f'items[{i}].vector must be a list with length={len(feature_names)} (got {len(vec) if isinstance(vec, list) else "non-list"}).'
                )
            rows.append({fn: float(v) for fn, v in zip(feature_names, vec)})
        elif "features" in item:
            feats = item["features"]
            if not isinstance(feats, dict):
                raise ValueError(f"items[{i}].features must be an object.")
            # Fill missing features with 0.0
            row = {fn: float(feats.get(fn, 0.0)) for fn in feature_names}
            rows.append(row)
        else:
            raise ValueError(f'items[{i}] must contain either "vector" or "features".')

    X = pd.DataFrame(rows, columns=feature_names)

    y_pred_int = model.predict(X).astype(int)
    out: dict[str, Any] = {"predictions": []}

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None

    for idx, item_id in enumerate(ids):
        pred_int = int(y_pred_int[idx])
        pred_label = int_to_label.get(str(pred_int), str(pred_int))
        pred_obj: dict[str, Any] = {
            "id": item_id,
            "label_int": pred_int,
            "label": pred_label,
        }
        if proba is not None:
            p = proba[idx]
            # class order assumed 0..K-1
            pred_obj["proba"] = {
                int_to_label.get(str(k), str(k)): float(p[k]) for k in range(len(p))
            }
            pred_obj["proba_max"] = float(np.max(p))
        out["predictions"].append(pred_obj)

    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

