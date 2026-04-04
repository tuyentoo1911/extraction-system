from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EvalResult:
    macro_f1: float
    weighted_f1: float
    accuracy: float
    mae_class_distance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "mae_class_distance": self.mae_class_distance,
        }


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvalResult:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    return EvalResult(
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=accuracy,
        mae_class_distance=mae,
    )


def _rank_key(res: EvalResult) -> tuple[float, float]:
    # Prefer higher macro_f1, then lower MAE in ordinal distance.
    return (res.macro_f1, -res.mae_class_distance)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train influence-level classifier (LOW/MEDIUM/HIGH) from feature_engineering_output."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Path to feature_engineering_output directory.")
    parser.add_argument(
        "--model-out-dir",
        type=str,
        default=None,
        help="Where to save trained model/metadata. Default: extraction-system/model/influence_predictor",
    )
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite existing model outputs.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = Path(args.data_dir) if args.data_dir else (base_dir / "feature_engineering_output")
    model_out_dir = (
        Path(args.model_out_dir)
        if args.model_out_dir
        else (base_dir / "model" / "influence_predictor")
    )
    model_out_dir.mkdir(parents=True, exist_ok=True)

    label_encoder_path = data_dir / "label_encoder.json"
    label_encoder = json.loads(label_encoder_path.read_text(encoding="utf-8"))
    label_to_int: dict[str, int] = {k: int(v) for k, v in label_encoder["label_to_int"].items()}
    int_to_label: dict[str, str] = label_encoder["int_to_label"]

    # Load split data
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train_df = pd.read_csv(data_dir / "y_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    y_val_df = pd.read_csv(data_dir / "y_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test_df = pd.read_csv(data_dir / "y_test.csv")

    feature_names = list(X_train.columns)

    for name, x in [("X_val", X_val), ("X_test", X_test)]:
        if list(x.columns) != feature_names:
            raise ValueError(f"{name} columns differ from X_train. Ensure same feature order.")

    y_train = y_train_df["label_int"].astype(int).to_numpy()
    y_val = y_val_df["label_int"].astype(int).to_numpy()
    y_test = y_test_df["label_int"].astype(int).to_numpy()

    # Candidate models
    candidates: list[tuple[str, Any]] = [
        (
            "logreg_standard_scaler",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=3000,
                            solver="lbfgs",
                            class_weight="balanced",
                            n_jobs=None,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        (
            "rf_balanced_subsample",
            RandomForestClassifier(
                n_estimators=600,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=1,
            ),
        ),
        (
            "extratrees_balanced",
            ExtraTreesClassifier(
                n_estimators=900,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                min_samples_leaf=1,
            ),
        ),
        (
            "histgb",
            HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=500,
                random_state=42,
            ),
        ),
    ]

    # If there's already a model, allow re-use unless --force is provided.
    model_path = model_out_dir / "influence_predictor.joblib"
    meta_path = model_out_dir / "metadata.json"
    if model_path.exists() and meta_path.exists() and not args.force:
        print(f"Model outputs already exist at {model_out_dir}. Use --force to retrain.")
        return

    best_name: str | None = None
    best_model: Any | None = None
    best_val_res: EvalResult | None = None
    best_val_cm: np.ndarray | None = None

    # Model selection on validation set
    for name, model in candidates:
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        res = _evaluate(y_val, y_val_pred)
        cm = confusion_matrix(y_val, y_val_pred, labels=sorted(label_to_int.values()))

        print(f"[VAL] {name}: macro_f1={res.macro_f1:.4f}, weighted_f1={res.weighted_f1:.4f}, acc={res.accuracy:.4f}, mae={res.mae_class_distance:.4f}")

        if best_val_res is None or _rank_key(res) > _rank_key(best_val_res):
            best_name = name
            best_model = model
            best_val_res = res
            best_val_cm = cm

    assert best_name is not None and best_model is not None and best_val_res is not None and best_val_cm is not None

    # Train final model on train+val
    X_train_all = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_train_all = np.concatenate([y_train, y_val], axis=0)
    best_model.fit(X_train_all, y_train_all)

    y_test_pred = best_model.predict(X_test)
    test_res = _evaluate(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred, labels=sorted(label_to_int.values()))

    # Optional probabilities (if supported)
    proba_path = model_out_dir / "test_predictions.csv"
    pred_df = pd.DataFrame(
        {
            "y_true_int": y_test.astype(int),
            "y_true_label": [int_to_label[str(i)] for i in y_test.astype(int)],
            "y_pred_int": y_test_pred.astype(int),
            "y_pred_label": [int_to_label[str(i)] for i in y_test_pred.astype(int)],
        }
    )
    if hasattr(best_model, "predict_proba"):
        try:
            proba = best_model.predict_proba(X_test)
            pred_df["pred_proba_max"] = np.max(proba, axis=1)
        except Exception:
            pass
    pred_df.to_csv(proba_path, index=False, encoding="utf-8")

    # Save model + metadata
    joblib.dump(best_model, model_path)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "timestamp_utc": timestamp,
        "data_dir": str(data_dir),
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "selected_model": best_name,
        "val_metrics": best_val_res.to_dict(),
        "val_confusion_matrix": best_val_cm.tolist(),
        "test_metrics": test_res.to_dict(),
        "test_confusion_matrix": test_cm.tolist(),
        "splits": {
            "X_train": "X_train.csv",
            "y_train": "y_train.csv",
            "X_val": "X_val.csv",
            "y_val": "y_val.csv",
            "X_test": "X_test.csv",
            "y_test": "y_test.csv",
        },
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"Saved test predictions: {proba_path}")
    print(f"[TEST] macro_f1={test_res.macro_f1:.4f}, weighted_f1={test_res.weighted_f1:.4f}, acc={test_res.accuracy:.4f}, mae={test_res.mae_class_distance:.4f}")


if __name__ == "__main__":
    main()

