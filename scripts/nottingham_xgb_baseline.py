#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import OrdinalEncoder

import xgboost as xgb
from preprocessing.nottingham_rgb_preprocess import (
    NOTTINGHAM_FEATURES,
    _decode,
    get_nottingham_grade_value,
    parse_clinical_xlsx,
    read_validation_patients,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "xgboost is not installed. Install it with `python -m pip install --user xgboost` and rerun."
    ) from exc


def _feature_key(display_name: str) -> str:
    return re.sub(r"\s+", "_", display_name.strip().lower())


def _select_clinical_workbook(base_dir: Path) -> Path:
    preferred = base_dir / "Clinical_and_Other_Features_full_label.xlsx"
    if preferred.exists():
        return preferred
    fallback = base_dir / "Clinical_and_Other_Features.xlsx"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find a clinical workbook under the provided base directory")


def _build_dataset(
    base_dir: Path,
) -> Tuple[Path, List[str], List[str], OrdinalEncoder, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    clinical_xlsx = _select_clinical_workbook(base_dir)
    eval_ids = read_validation_patients(base_dir / "validation_dataset_patients_list.csv")
    headers, maps, records = parse_clinical_xlsx(clinical_xlsx)
    clinical_by_id = {r["Patient ID"]: r for r in records}
    header_index = {h: i for i, h in enumerate(headers) if h}

    feature_names = [_feature_key(display_name) for _, display_name, _ in NOTTINGHAM_FEATURES]

    def encode_row(record: Dict[str, str]) -> List[str]:
        values: List[str] = []
        for src_col, display_name, _ in NOTTINGHAM_FEATURES:
            col_idx = header_index.get(src_col)
            dec_map = maps.get(col_idx) if col_idx is not None else None
            values.append(_decode(record.get(src_col, ""), dec_map))
        return values

    train_ids: List[str] = []
    x_train_rows: List[List[str]] = []
    y_train_raw: List[int] = []
    eval_set = set(eval_ids)

    for pid, record in clinical_by_id.items():
        label = get_nottingham_grade_value(record)
        if not label:
            continue
        label_int = int(float(label))
        features = encode_row(record)
        if pid not in eval_set:
            train_ids.append(pid)
            x_train_rows.append(features)
            y_train_raw.append(label_int)

    x_eval_rows: List[List[str]] = []
    y_eval_raw: List[int] = []
    missing_eval: List[str] = []
    for pid in eval_ids:
        record = clinical_by_id.get(pid)
        if record is None:
            missing_eval.append(pid)
            continue
        label = get_nottingham_grade_value(record)
        if not label:
            missing_eval.append(pid)
            continue
        x_eval_rows.append(encode_row(record))
        y_eval_raw.append(int(float(label)))

    if missing_eval:
        raise RuntimeError(
            f"Validation patients missing usable Nottingham labels in {clinical_xlsx.name}: {missing_eval[:10]}"
        )

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.float32,
    )
    x_train = encoder.fit_transform(np.asarray(x_train_rows, dtype=object))
    x_eval = encoder.transform(np.asarray(x_eval_rows, dtype=object))

    y_train = np.asarray([label - 1 for label in y_train_raw], dtype=np.int64)
    y_eval = np.asarray([label - 1 for label in y_eval_raw], dtype=np.int64)

    return clinical_xlsx, feature_names, train_ids, encoder, x_train, y_train, x_eval, y_eval


def run_baseline(args: argparse.Namespace) -> Dict[str, object]:
    base_dir = Path(args.base_dir)
    clinical_xlsx, feature_names, train_ids, encoder, x_train, y_train, x_eval, y_eval = _build_dataset(base_dir)

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=args.random_seed,
        tree_method="hist",
        device=args.device,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    eval_matrix = xgb.DMatrix(x_eval, feature_names=feature_names)
    pred_prob = model.get_booster().predict(eval_matrix)
    pred_idx = np.asarray(np.argmax(pred_prob, axis=1), dtype=int)
    y_true = (y_eval + 1).astype(int)
    y_pred = (pred_idx + 1).astype(int)

    prediction_path = Path(args.predictions_out)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    eval_ids = read_validation_patients(base_dir / "validation_dataset_patients_list.csv")
    with prediction_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "PatientID",
            "target_nottingham_grade",
            "predicted_nottingham_grade",
            "prob_grade_1",
            "prob_grade_2",
            "prob_grade_3",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, pid in enumerate(eval_ids):
            writer.writerow(
                {
                    "PatientID": pid,
                    "target_nottingham_grade": int(y_true[i]),
                    "predicted_nottingham_grade": int(y_pred[i]),
                    "prob_grade_1": f"{pred_prob[i][0]:.6f}",
                    "prob_grade_2": f"{pred_prob[i][1]:.6f}",
                    "prob_grade_3": f"{pred_prob[i][2]:.6f}",
                }
            )

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "encoder": encoder,
                "feature_names": feature_names,
                "clinical_xlsx": str(clinical_xlsx),
            },
            f,
        )

    feature_importances = []
    for name, importance in zip(feature_names, model.feature_importances_):
        feature_importances.append({"feature": name, "importance": float(importance)})
    feature_importances.sort(key=lambda item: item["importance"], reverse=True)

    metrics = {
        "clinical_xlsx": str(clinical_xlsx),
        "train_patients": len(train_ids),
        "eval_patients": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": [1, 2, 3],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 2, 3]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[1, 2, 3], output_dict=True, zero_division=0
        ),
        "feature_importances": feature_importances,
        "predictions_csv": str(prediction_path),
        "model_path": str(model_path),
    }

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simple tabular XGBoost baseline for Nottingham grade and evaluate on the 100-patient validation cohort."
    )
    parser.add_argument("--base-dir", default="/mnt/e/BreastCancerData", help="Directory containing the DUKE clinical workbook")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--model-out", default="data/models/nottingham_xgb_baseline.pkl")
    parser.add_argument("--predictions-out", default="data/results/nottingham_xgb_eval_predictions.csv")
    parser.add_argument("--metrics-out", default="data/results/nottingham_xgb_eval_metrics.json")
    return parser


def main(argv: Sequence[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    metrics = run_baseline(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
