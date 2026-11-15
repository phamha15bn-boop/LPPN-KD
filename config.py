# Copyright (c) 2026 Qineng Song

"""Single source of truth for LPPN training, prediction, and evaluation.

Routine experiments use a JSON profile shared with the teacher.
Importing this module never creates data,
splits, checkpoints, or result folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


from release_config import (
    DATA_PATHS, DATASET_NAME, MODEL_VARIANT, PROFILE, ROOT as PROJECT_ROOT,
    RESULTS_ROOT, SEED, TEACHER, TEACHER_SHA256 as EQT_TEACHER_SHA256,
    TEACHER_WEIGHT as EQT_TEACHER_WEIGHT, cache_root, resolve_path,
)

MODEL_N = 8
MODEL_S = 8
BATCH_SIZE = PROFILE["batch_size"]
EPOCHS = PROFILE["epochs"]
LEARNING_RATE = PROFILE["learning_rate"]
WEIGHT_DECAY = 0.0
NUM_WORKERS = 0
GRAD_CLIP = 0.0

TEACHER_PROTOCOL_ID = TEACHER["p_protocol"]
TEACHER_FEATURE_PROTOCOL_ID = TEACHER["feature_protocol"]
EQT_TEACHER_ROOT = PROJECT_ROOT / "teacher" / "eqtransformer"
KD_LAMBDA_C = PROFILE["lambda_c"]
KD_LAMBDA_FP = PROFILE["lambda_fp"]
KD_TEACHER_MIN_CONFIDENCE = TEACHER["min_confidence"]
KD_TEACHER_MAX_ERROR_SECONDS = TEACHER["max_error_seconds"]
KD_VERIFY_CACHE_SHA256 = True
KD_FEATURE_NORMALIZATION_EPSILON = 1e-6

PREDICTION_BATCH_SIZE = 128
THRESHOLD_P = 0.3
THRESHOLD_S = 0.3
TOLERANCE_SECONDS = 0.5
PREDICTION_CHECKPOINT_OVERRIDE = None
EVALUATION_PREDICTIONS_OVERRIDE = None

# =============================================================================
# OPTIONAL ADVANCED TRAINING CONFIGURATION
# =============================================================================
NORMALIZATION = "eqt"
AMPLITUDE_SCALE = False
GAUSSIAN_NOISE_STD = 0.0
ALLOW_OVERWRITE = False


# =============================================================================
# FIXED LPPN-BASE MODEL AND EVALUATION PROTOCOL -- do not tune routinely
# =============================================================================
SAMPLE_RATE_HZ = 100.0
SOURCE_TRACE_LENGTH = 6000
MODEL_INPUT_LENGTH = 6144
CHANNEL_ORDER = ("E", "N", "Z")
REFERENCE_BASE_PARAMETER_COUNT = 43_296  # BASE check, not a future constraint


# =============================================================================
# Derived configuration -- normally do not edit below this line
# =============================================================================
IMPLEMENTED_MODEL_VARIANTS = {
    "LPPN-BASE",
    "LPPN-KD-C",
    "LPPN-KD-FP",
    "LPPN-KD-CFP",
}
RESERVED_MODEL_VARIANTS = {
    "LPPN-KD",
    "LPPN-OPT",
}
KD_CLASSIFICATION_VARIANTS = {"LPPN-KD-C", "LPPN-KD-CFP"}
KD_FEATURE_PROJECTION_VARIANTS = {"LPPN-KD-FP", "LPPN-KD-CFP"}
KD_FEATURE_VARIANTS = KD_FEATURE_PROJECTION_VARIANTS
KD_MODEL_VARIANTS = KD_CLASSIFICATION_VARIANTS | KD_FEATURE_VARIANTS

DATASET_PATHS = {DATASET_NAME: DATA_PATHS}

DATASET_NAME = DATASET_NAME.strip().lower()
MODEL_VARIANT = MODEL_VARIANT.strip().upper()


def dataset_paths(dataset_name: str) -> dict[str, Path]:
    """Return one fixed split mapping without creating or repartitioning data."""

    name = dataset_name.strip().lower()
    if name not in DATASET_PATHS:
        raise ValueError(
            f"Unsupported DATASET_NAME={name!r}; choose from {sorted(DATASET_PATHS)}."
        )
    return dict(DATASET_PATHS[name])


def _validate_configuration() -> None:
    if DATASET_NAME not in DATASET_PATHS:
        raise ValueError(
            f"Unsupported DATASET_NAME={DATASET_NAME!r}; "
            f"choose from {sorted(DATASET_PATHS)}."
        )
    if MODEL_VARIANT in RESERVED_MODEL_VARIANTS:
        raise ValueError(
            f"{MODEL_VARIANT} is reserved for a later stage and is not implemented."
        )
    if MODEL_VARIANT not in IMPLEMENTED_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported MODEL_VARIANT={MODEL_VARIANT!r}; "
            f"implemented: {sorted(IMPLEMENTED_MODEL_VARIANTS)}."
        )
    if not isinstance(SEED, int) or SEED < 0:
        raise ValueError("SEED must be a non-negative integer.")
    if not isinstance(MODEL_N, int) or MODEL_N <= 0:
        raise ValueError("MODEL_N must be a positive integer.")
    if not isinstance(MODEL_S, int) or MODEL_S <= 0:
        raise ValueError("MODEL_S must be a positive integer.")
    if MODEL_VARIANT in IMPLEMENTED_MODEL_VARIANTS and (MODEL_N, MODEL_S) != (8, 8):
        raise ValueError(
            "Implemented LPPN BASE/KD variants are verified only for "
            "MODEL_N=8 and MODEL_S=8. "
            "Other N/S combinations require a separately implemented and named "
            "model variant so labels, offsets, coordinates, and checkpoints cannot drift."
        )
    for name, value, lower_bound, strict in (
        ("BATCH_SIZE", BATCH_SIZE, 0, True),
        ("EPOCHS", EPOCHS, 0, True),
        ("LEARNING_RATE", LEARNING_RATE, 0, True),
        ("WEIGHT_DECAY", WEIGHT_DECAY, 0, False),
        ("NUM_WORKERS", NUM_WORKERS, 0, False),
        ("GRAD_CLIP", GRAD_CLIP, 0, False),
        ("PREDICTION_BATCH_SIZE", PREDICTION_BATCH_SIZE, 0, True),
        ("GAUSSIAN_NOISE_STD", GAUSSIAN_NOISE_STD, 0, False),
    ):
        invalid = value <= lower_bound if strict else value < lower_bound
        if invalid:
            relation = ">" if strict else ">="
            raise ValueError(f"{name} must be {relation} {lower_bound}; got {value}.")
    for name, threshold in (("THRESHOLD_P", THRESHOLD_P), ("THRESHOLD_S", THRESHOLD_S)):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {threshold}.")
    if TOLERANCE_SECONDS <= 0:
        raise ValueError("TOLERANCE_SECONDS must be positive.")
    if MODEL_INPUT_LENGTH % MODEL_S != 0:
        raise ValueError("MODEL_INPUT_LENGTH must be divisible by MODEL_S.")
    if MODEL_INPUT_LENGTH % 64 != 0:
        raise ValueError("LPPN-BASE MODEL_INPUT_LENGTH must be divisible by 64.")
    if NORMALIZATION not in {"eqt", "lppn", "none"}:
        raise ValueError("NORMALIZATION must be 'eqt', 'lppn', or 'none'.")

    if MODEL_VARIANT in KD_MODEL_VARIANTS:
        required = {
            "KD_TEACHER_MIN_CONFIDENCE": KD_TEACHER_MIN_CONFIDENCE,
            "KD_TEACHER_MAX_ERROR_SECONDS": KD_TEACHER_MAX_ERROR_SECONDS,
        }
        if MODEL_VARIANT in KD_CLASSIFICATION_VARIANTS:
            required["KD_LAMBDA_C"] = KD_LAMBDA_C
        if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS:
            required["KD_LAMBDA_FP"] = KD_LAMBDA_FP
        unset = [name for name, value in required.items() if value is None]
        if unset:
            raise ValueError(
                "Formal KD hyperparameters are intentionally unset; approve and set: "
                + ", ".join(unset)
            )
        for name, value in required.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative; got {value}.")
        if not 0.0 <= float(KD_TEACHER_MIN_CONFIDENCE) <= 1.0:
            raise ValueError("KD_TEACHER_MIN_CONFIDENCE must be in [0, 1].")

    for label, path in (("PROJECT_ROOT", PROJECT_ROOT), ("RESULTS_ROOT", RESULTS_ROOT)):
        if not path.is_absolute():
            raise ValueError(f"{label} must be an absolute path: {path}")
    for dataset, paths in DATASET_PATHS.items():
        for key, path in paths.items():
            if not path.is_absolute():
                raise ValueError(f"{dataset}.{key} must be absolute: {path}")
    for label, override in (
        ("PREDICTION_CHECKPOINT_OVERRIDE", PREDICTION_CHECKPOINT_OVERRIDE),
        ("EVALUATION_PREDICTIONS_OVERRIDE", EVALUATION_PREDICTIONS_OVERRIDE),
    ):
        if override is not None and not Path(override).is_absolute():
            raise ValueError(f"{label} must be None or an absolute path: {override}")


_validate_configuration()

_paths = dataset_paths(DATASET_NAME)
TRAIN_HDF5 = _paths["train_hdf5"]
TRAIN_CSV = _paths["train_csv"]
VALIDATION_HDF5 = _paths["validation_hdf5"]
VALIDATION_CSV = _paths["validation_csv"]
TEST_HDF5 = _paths["test_hdf5"]
TEST_CSV = _paths["test_csv"]

def experiment_directory(dataset_name: str, model_variant: str, seed: int) -> Path:
    """Resolve an isolated experiment directory without creating it."""

    return RESULTS_ROOT / dataset_name.strip().lower() / model_variant.strip().upper() / f"seed{seed}"


EXPERIMENT_ID = f"{DATASET_NAME}_{MODEL_VARIANT}_seed{SEED}"
EXPERIMENT_DIR = experiment_directory(DATASET_NAME, MODEL_VARIANT, SEED)
TRAIN_OUTPUT_DIR = EXPERIMENT_DIR / "training"
PREDICTION_OUTPUT_DIR = EXPERIMENT_DIR / "prediction"
EVALUATION_OUTPUT_DIR = EXPERIMENT_DIR / "evaluation"
TEACHER_CACHE_ROOT = cache_root("teacher_cache", TEACHER_PROTOCOL_ID)
TEACHER_CACHE_MANIFEST = TEACHER_CACHE_ROOT / "cache_manifest.json"
TEACHER_FEATURE_CACHE_ROOT = cache_root("teacher_feature_cache", TEACHER_FEATURE_PROTOCOL_ID)
TEACHER_FEATURE_CACHE_MANIFEST = TEACHER_FEATURE_CACHE_ROOT / "cache_manifest.json"
BEST_CHECKPOINT = TRAIN_OUTPUT_DIR / f"{EXPERIMENT_ID}_best.pt"
LAST_CHECKPOINT = TRAIN_OUTPUT_DIR / f"{EXPERIMENT_ID}_last.pt"
RAW_PREDICTIONS = PREDICTION_OUTPUT_DIR / "test" / "raw_predictions.csv"
PREDICTION_CHECKPOINT = (
    Path(PREDICTION_CHECKPOINT_OVERRIDE)
    if PREDICTION_CHECKPOINT_OVERRIDE is not None
    else BEST_CHECKPOINT
)
EVALUATION_PREDICTIONS = (
    Path(EVALUATION_PREDICTIONS_OVERRIDE)
    if EVALUATION_PREDICTIONS_OVERRIDE is not None
    else RAW_PREDICTIONS
)


def prediction_output_directory(split: str) -> Path:
    """Map CLI split names to the parallel on-disk test/valid layout."""

    name = split.strip().lower()
    if name == "test":
        return PREDICTION_OUTPUT_DIR / "test"
    if name == "validation":
        return PREDICTION_OUTPUT_DIR / "valid"
    raise ValueError(f"Prediction split must be validation or test; got {split!r}.")


def evaluation_output_directory(split: str) -> Path:
    name = split.strip().lower()
    if name == "test":
        return EVALUATION_OUTPUT_DIR / "test"
    if name == "validation":
        return EVALUATION_OUTPUT_DIR / "valid"
    raise ValueError(f"Evaluation split must be validation or test; got {split!r}.")


def prediction_data_paths(split: str) -> tuple[Path, Path]:
    name = split.strip().lower()
    if name == "test":
        return TEST_HDF5, TEST_CSV
    if name == "validation":
        return VALIDATION_HDF5, VALIDATION_CSV
    raise ValueError(f"Prediction split must be validation or test; got {split!r}.")


def raw_predictions_for_split(split: str) -> Path:
    return prediction_output_directory(split) / "raw_predictions.csv"

IDENTITY_FIELDS = {
    "dataset_name": DATASET_NAME,
    "model_variant": MODEL_VARIANT,
    "seed": SEED,
    "model_n": MODEL_N,
    "model_s": MODEL_S,
}


def validate_paths(required_splits: Iterable[str]) -> None:
    """Require configured split files to be existing absolute paths."""

    split_paths = {
        "train": (TRAIN_HDF5, TRAIN_CSV),
        "validation": (VALIDATION_HDF5, VALIDATION_CSV),
        "test": (TEST_HDF5, TEST_CSV),
    }
    problems = []
    for split in required_splits:
        if split not in split_paths:
            raise ValueError(f"Unknown split: {split}")
        for path in split_paths[split]:
            if not path.is_absolute():
                problems.append(f"{split} path is not absolute: {path}")
            elif not path.is_file():
                problems.append(f"{split} file does not exist: {path}")
    if problems:
        raise FileNotFoundError("Configured input validation failed:\n- " + "\n- ".join(problems))


def validate_artifact_identity(metadata: Mapping[str, object], artifact: str) -> None:
    """Reject checkpoints/predictions from a different experiment identity."""

    problems = []
    for key, expected in IDENTITY_FIELDS.items():
        actual = metadata.get(key)
        if actual != expected:
            problems.append(f"{key}={actual!r}, expected {expected!r}")
    if problems:
        raise ValueError(f"{artifact} identity mismatch:\n- " + "\n- ".join(problems))


def validate_kd_artifact_identity(metadata: Mapping[str, object], artifact: str) -> None:
    """Reject a KD artifact with a different semantic teacher/projection protocol.

    Non-operative fields in older checkpoints are not part of the active
    classification/feature-distillation identity.
    """

    if MODEL_VARIANT not in KD_MODEL_VARIANTS:
        return
    expected = experiment_snapshot()["kd"]
    actual = metadata.get("kd")
    if not isinstance(actual, Mapping):
        raise ValueError(f"{artifact} is missing KD protocol metadata.")
    identity_keys = [
        "enabled",
        "classification_enabled",
        "feature_enabled",
        "teacher_min_confidence",
        "teacher_max_error_seconds",
        "teacher_protocol_id",
        "teacher_weight_sha256",
    ]
    if MODEL_VARIANT in KD_CLASSIFICATION_VARIANTS:
        identity_keys.append("lambda_c")
    if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS:
        identity_keys.extend(
            (
                "lambda_fp",
                "feature_protocol_id",
                "feature_normalization_epsilon",
                "projection_type",
                "projection_in_features",
                "projection_out_features",
                "projection_bias",
            )
        )
    differences = {}
    for key in identity_keys:
        saved = actual.get(key)
        # Feature KD did not exist when the frozen KD-C checkpoint was made;
        # absence therefore means disabled for that one backward-compatible key.
        if key == "feature_enabled" and key not in actual:
            saved = False
        if saved != expected.get(key):
            differences[key] = {"saved": saved, "expected": expected.get(key)}
    if differences:
        raise ValueError(
            f"{artifact} KD protocol mismatch.\n"
            f"differences={json.dumps(differences, ensure_ascii=False, sort_keys=True)}"
        )


def resolve_prediction_checkpoint(command_line_override: Path | None = None) -> Path:
    path = resolve_path(command_line_override) if command_line_override else PREDICTION_CHECKPOINT
    if not path.is_absolute():
        raise ValueError(f"Prediction checkpoint must be absolute: {path}")
    return path


def resolve_evaluation_predictions(
    command_line_override: Path | None = None, *, split: str = "test"
) -> Path:
    if command_line_override is not None:
        path = resolve_path(command_line_override)
    elif split.strip().lower() == "test":
        path = EVALUATION_PREDICTIONS
    else:
        path = raw_predictions_for_split(split)
    if not path.is_absolute():
        raise ValueError(f"Evaluation predictions must be absolute: {path}")
    return path


def checkpoint_paths(output_dir: Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    return (
        directory / f"{EXPERIMENT_ID}_best.pt",
        directory / f"{EXPERIMENT_ID}_last.pt",
    )


def experiment_snapshot() -> dict[str, object]:
    """Return a JSON-safe, complete record for a run."""

    return {
        **IDENTITY_FIELDS,
        "experiment_id": EXPERIMENT_ID,
        "train_hdf5": str(TRAIN_HDF5),
        "train_csv": str(TRAIN_CSV),
        "validation_hdf5": str(VALIDATION_HDF5),
        "validation_csv": str(VALIDATION_CSV),
        "test_hdf5": str(TEST_HDF5),
        "test_csv": str(TEST_CSV),
        "batch_size": BATCH_SIZE,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "normalization": NORMALIZATION,
        "amplitude_scale": AMPLITUDE_SCALE,
        "gaussian_noise_std": GAUSSIAN_NOISE_STD,
        "grad_clip": GRAD_CLIP,
        "threshold_p": THRESHOLD_P,
        "threshold_s": THRESHOLD_S,
        "tolerance_seconds": TOLERANCE_SECONDS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "source_trace_length": SOURCE_TRACE_LENGTH,
        "model_input_length": MODEL_INPUT_LENGTH,
        "stride": MODEL_S,
        "channel_order": list(CHANNEL_ORDER),
        "reference_base_parameter_count": REFERENCE_BASE_PARAMETER_COUNT,
        "project_root": str(PROJECT_ROOT),
        "results_root": str(RESULTS_ROOT),
        "experiment_dir": str(EXPERIMENT_DIR),
        "train_output_dir": str(TRAIN_OUTPUT_DIR),
        "prediction_output_dir": str(PREDICTION_OUTPUT_DIR),
        "evaluation_output_dir": str(EVALUATION_OUTPUT_DIR),
        "test_prediction_output_dir": str(prediction_output_directory("test")),
        "test_evaluation_output_dir": str(evaluation_output_directory("test")),
        "validation_prediction_output_dir": str(
            prediction_output_directory("validation")
        ),
        "validation_evaluation_output_dir": str(
            evaluation_output_directory("validation")
        ),
        "best_checkpoint": str(BEST_CHECKPOINT),
        "last_checkpoint": str(LAST_CHECKPOINT),
        "prediction_checkpoint": str(PREDICTION_CHECKPOINT),
        "raw_predictions": str(RAW_PREDICTIONS),
        "evaluation_predictions": str(EVALUATION_PREDICTIONS),
        "allow_overwrite": ALLOW_OVERWRITE,
        "kd": {
            "enabled": MODEL_VARIANT in KD_MODEL_VARIANTS,
            "classification_enabled": MODEL_VARIANT in KD_CLASSIFICATION_VARIANTS,
            "feature_enabled": MODEL_VARIANT in KD_FEATURE_VARIANTS,
            "projection_enabled": MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS,
            "lambda_c": KD_LAMBDA_C if MODEL_VARIANT in KD_CLASSIFICATION_VARIANTS else None,
            "lambda_fp": KD_LAMBDA_FP if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS else None,
            "projection_type": "Linear" if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS else None,
            "projection_in_features": 32 if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS else None,
            "projection_out_features": 32 if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS else None,
            "projection_bias": False if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS else None,
            "teacher_min_confidence": KD_TEACHER_MIN_CONFIDENCE,
            "teacher_max_error_seconds": KD_TEACHER_MAX_ERROR_SECONDS,
            "teacher_protocol_id": TEACHER_PROTOCOL_ID,
            "feature_protocol_id": TEACHER_FEATURE_PROTOCOL_ID,
            "feature_normalization_epsilon": KD_FEATURE_NORMALIZATION_EPSILON,
            "teacher_weight": str(EQT_TEACHER_WEIGHT),
            "teacher_weight_sha256": EQT_TEACHER_SHA256,
            "teacher_cache_manifest": str(TEACHER_CACHE_MANIFEST),
            "teacher_feature_cache_manifest": str(TEACHER_FEATURE_CACHE_MANIFEST),
            "verify_cache_sha256": KD_VERIFY_CACHE_SHA256,
        },
    }


def entry_config(
    entry: str,
    *,
    checkpoint: Path | None = None,
    predictions: Path | None = None,
    split: str = "test",
) -> dict[str, object]:
    """Build a concise, read-only terminal view for one entry point."""

    common: dict[str, object] = {
        "entry": entry,
        **IDENTITY_FIELDS,
        "experiment_id": EXPERIMENT_ID,
    }
    if entry == "train":
        common.update(
            {
                "input": {
                    "train_hdf5": str(TRAIN_HDF5),
                    "train_csv": str(TRAIN_CSV),
                    "validation_hdf5": str(VALIDATION_HDF5),
                    "validation_csv": str(VALIDATION_CSV),
                },
                "output": str(TRAIN_OUTPUT_DIR),
                "batch_size": BATCH_SIZE,
                "model_weight": str(checkpoint) if checkpoint else "FROM_SCRATCH",
            }
        )
        if MODEL_VARIANT in KD_MODEL_VARIANTS:
            common["kd"] = {
                "classification_enabled": MODEL_VARIANT in KD_CLASSIFICATION_VARIANTS,
                "feature_enabled": MODEL_VARIANT in KD_FEATURE_VARIANTS,
                "projection_enabled": MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS,
                "lambda_c": KD_LAMBDA_C if MODEL_VARIANT in KD_CLASSIFICATION_VARIANTS else None,
                "lambda_fp": KD_LAMBDA_FP if MODEL_VARIANT in KD_FEATURE_PROJECTION_VARIANTS else None,
                "teacher_min_confidence": KD_TEACHER_MIN_CONFIDENCE,
                "teacher_max_error_seconds": KD_TEACHER_MAX_ERROR_SECONDS,
                "teacher_protocol_id": TEACHER_PROTOCOL_ID,
                "teacher_cache_manifest": str(TEACHER_CACHE_MANIFEST),
                "feature_protocol_id": TEACHER_FEATURE_PROTOCOL_ID,
                "teacher_feature_cache_manifest": str(TEACHER_FEATURE_CACHE_MANIFEST),
            }
    elif entry == "predict":
        split_name = split.strip().lower()
        split_hdf5, split_csv = prediction_data_paths(split_name)
        selected_checkpoint = checkpoint or PREDICTION_CHECKPOINT
        common.update(
            {
                "split": split_name,
                "input": {
                    f"{split_name}_hdf5": str(split_hdf5),
                    f"{split_name}_csv": str(split_csv),
                },
                "output": str(prediction_output_directory(split_name)),
                "batch_size": PREDICTION_BATCH_SIZE,
                "model_weight": str(selected_checkpoint),
            }
        )
    elif entry == "evaluate":
        split_name = split.strip().lower()
        selected_predictions = predictions or resolve_evaluation_predictions(split=split_name)
        common.update(
            {
                "split": split_name,
                "input": {"raw_predictions": str(selected_predictions)},
                "output": str(evaluation_output_directory(split_name)),
                "threshold_p": THRESHOLD_P,
                "threshold_s": THRESHOLD_S,
                "tolerance_seconds": TOLERANCE_SECONDS,
                "model_weight": str(checkpoint) if checkpoint else "RECORDED_IN_PREDICTION_CONFIG",
            }
        )
    else:
        raise ValueError(f"Unknown entry: {entry}")
    return common


def print_entry_config(
    entry: str,
    *,
    checkpoint: Path | None = None,
    predictions: Path | None = None,
    split: str = "test",
) -> None:
    print(
        json.dumps(
            entry_config(
                entry,
                checkpoint=checkpoint,
                predictions=predictions,
                split=split,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
