# Copyright (c) 2026 Qineng Song

"""Run the configured LPPN variant and save raw P/S picks without scoring."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from lppn import LPPNBase, LPPNDataset


def require_absolute_file(path_value, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute Windows path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must be a new LPPN experiment checkpoint.")
    saved_config = checkpoint.get("config", {})
    cfg.validate_artifact_identity(saved_config, "Checkpoint")
    cfg.validate_kd_artifact_identity(saved_config, "Checkpoint")
    model = LPPNBase(model_n=cfg.MODEL_N, n_stride=cfg.MODEL_S).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def collect_predictions(model, loader, device):
    rows = []
    inference_seconds = 0.0
    with torch.no_grad():
        for waveform, _, info in loader:
            waveform = waveform.to(device, dtype=torch.float32, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            probability, offset = model(waveform)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started

            probability = probability.detach().cpu().numpy()
            offset = offset.detach().cpu().numpy()
            if offset.ndim == 1:
                offset = offset[None, :]
            names = list(info["trace_name"])
            true_p = as_list(info["p_arrival_sample"])
            true_s = as_list(info["s_arrival_sample"])
            lengths = as_list(info["original_length"])
            categories = list(info["trace_category"])
            for index, name in enumerate(names):
                row = {
                    "trace_name": name,
                    "true_p": int(true_p[index]),
                    "true_s": int(true_s[index]),
                    "original_length": int(lengths[index]),
                    "trace_category": str(categories[index]),
                }
                for phase, class_index in (("p", 1), ("s", 2)):
                    phase_probability = probability[index, class_index, :]
                    cell = int(np.argmax(phase_probability))
                    row[f"{phase}_confidence"] = float(phase_probability[cell])
                    row[f"{phase}_sample_raw"] = float(
                        cell * cfg.MODEL_S + offset[index, cell]
                    )
                rows.append(row)
    return rows, inference_seconds


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with the configured LPPN checkpoint.")
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Prediction split; test is the default, validation writes to valid/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional absolute override; config.py otherwise selects the best checkpoint.",
    )
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional bounded smoke-test sample count for the selected split.",
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
    if args.max_samples is not None and args.max_test_samples is not None:
        raise ValueError("Use only one of --max-samples and --max-test-samples.")
    if args.split != "test" and args.max_test_samples is not None:
        raise ValueError("--max-test-samples is a legacy test-only option.")
    max_samples = args.max_samples or args.max_test_samples
    if max_samples is not None and max_samples <= 0:
        raise ValueError("The smoke-test sample limit must be positive.")
    cfg.validate_paths((args.split,))
    checkpoint_path = require_absolute_file(
        cfg.resolve_prediction_checkpoint(args.checkpoint), "Checkpoint"
    )
    cfg.print_entry_config("predict", checkpoint=checkpoint_path, split=args.split)
    if args.show_config:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        load_model(checkpoint_path, device)
        print("Checkpoint path and experiment identity: OK")
        return
    output_dir = cfg.prediction_output_directory(args.split)
    if not output_dir.is_absolute():
        raise ValueError(f"Prediction output must be absolute: {output_dir}")
    raw_path = output_dir / "raw_predictions.csv"
    config_path = output_dir / "prediction_config.json"
    summary_path = output_dir / "prediction_summary.json"
    existing = [path for path in (raw_path, config_path, summary_path) if path.exists()]
    if existing and not (args.allow_overwrite or cfg.ALLOW_OVERWRITE):
        raise FileExistsError(
            "Refusing to overwrite prediction artifacts:\n- "
            + "\n- ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(checkpoint_path, device)
    split_hdf5, split_csv = cfg.prediction_data_paths(args.split)
    dataset = LPPNDataset(
        split_hdf5,
        split_csv,
        target_length=cfg.MODEL_INPUT_LENGTH,
        stride=cfg.MODEL_S,
        normalization=cfg.NORMALIZATION,
        max_samples=max_samples,
        seed=cfg.SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.PREDICTION_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    rows, inference_seconds = collect_predictions(model, loader, device)
    if not rows:
        raise RuntimeError("Prediction produced no rows.")
    write_csv(raw_path, rows)

    prediction_config = cfg.experiment_snapshot()
    prediction_config.update(
        {
            "prediction_split": args.split,
            "prediction_split_output_dir": str(output_dir),
            "raw_predictions": str(raw_path),
            "evaluation_predictions": str(raw_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "device": str(device),
            "torch_version": torch.__version__,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            f"{args.split}_samples": len(rows),
        }
    )
    config_path.write_text(
        json.dumps(prediction_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment_id": cfg.EXPERIMENT_ID,
        "prediction_split": args.split,
        f"{args.split}_samples": len(rows),
        "inference_seconds_model_only": inference_seconds,
        "samples_per_second_model_only": len(rows) / max(inference_seconds, 1e-9),
        "milliseconds_per_sample_model_only": inference_seconds * 1000 / len(rows),
        "raw_predictions": str(raw_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dataset.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
