# Copyright (c) 2026 Qineng Song

"""Evaluate frozen raw predictions with the protected P/S matching protocol."""

import argparse
import csv
import json
from pathlib import Path

import config as cfg
from lppn.metrics import phase_metrics


def require_absolute_file(path_value, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute Windows path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Prediction file contains no rows: {path}")
    required = {
        "trace_name",
        "true_p",
        "true_s",
        "original_length",
        "p_confidence",
        "p_sample_raw",
        "s_confidence",
        "s_sample_raw",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    return rows


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate frozen LPPN predictions.")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Evaluation split; test is the default, validation maps to valid/.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional absolute override; config.py otherwise selects current raw predictions.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print and validate the resolved configuration, then exit without writing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions_path = require_absolute_file(
        cfg.resolve_evaluation_predictions(args.predictions, split=args.split),
        "Raw predictions",
    )
    prediction_config_path = predictions_path.parent / "prediction_config.json"
    prediction_config_path = require_absolute_file(
        prediction_config_path, "Prediction configuration"
    )
    prediction_config = json.loads(prediction_config_path.read_text(encoding="utf-8"))
    cfg.validate_artifact_identity(prediction_config, "Prediction")
    cfg.validate_kd_artifact_identity(prediction_config, "Prediction")
    recorded_split = prediction_config.get("prediction_split", "test")
    if recorded_split != args.split:
        raise ValueError(
            f"Prediction split mismatch: recorded={recorded_split!r}, requested={args.split!r}."
        )
    recorded_checkpoint = prediction_config.get("checkpoint")
    cfg.print_entry_config(
        "evaluate",
        checkpoint=Path(recorded_checkpoint) if recorded_checkpoint else None,
        predictions=predictions_path,
        split=args.split,
    )
    if args.show_config:
        print("Prediction path and experiment identity: OK")
        return

    output_dir = cfg.evaluation_output_directory(args.split)
    if not output_dir.is_absolute():
        raise ValueError(f"Evaluation output must be absolute: {output_dir}")
    metrics_path = output_dir / "metrics.json"
    details_path = output_dir / "phase_predictions.csv"
    existing = [path for path in (metrics_path, details_path) if path.exists()]
    if existing and not (args.allow_overwrite or cfg.ALLOW_OVERWRITE):
        raise FileExistsError(
            "Refusing to overwrite evaluation artifacts:\n- "
            + "\n- ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(predictions_path)
    p_metrics, p_rows = phase_metrics(
        rows, "p", cfg.THRESHOLD_P, cfg.TOLERANCE_SECONDS, cfg.SAMPLE_RATE_HZ
    )
    s_metrics, s_rows = phase_metrics(
        rows, "s", cfg.THRESHOLD_S, cfg.TOLERANCE_SECONDS, cfg.SAMPLE_RATE_HZ
    )
    metrics = {
        "dataset_name": cfg.DATASET_NAME,
        "model_variant": cfg.MODEL_VARIANT,
        "seed": cfg.SEED,
        "model_n": cfg.MODEL_N,
        "model_s": cfg.MODEL_S,
        "experiment_id": cfg.EXPERIMENT_ID,
        "evaluation_split": args.split,
        f"{args.split}_samples": len(rows),
        "matching_rule": "global phase argmax, threshold, +/- tolerance",
        "P": p_metrics,
        "S": s_metrics,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(details_path, p_rows + s_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
