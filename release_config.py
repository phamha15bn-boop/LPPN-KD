# Copyright (c) 2026 Qineng Song

"""Portable settings shared by the TensorFlow and PyTorch environments."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    return (ROOT / value).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


PROFILE_PATH = resolve_path(os.environ.get("LPPN_CONFIG", "configs/smoke_test/stead.json"))
with PROFILE_PATH.open(encoding="utf-8") as stream:
    PROFILE = json.load(stream)
DATASET_NAME = PROFILE["dataset"]
if DATASET_NAME not in {"shanxi", "stead_demo"}:
    raise ValueError(f"Unsupported profile dataset: {DATASET_NAME!r}")
DEMO = DATASET_NAME == "stead_demo"
MODEL_VARIANT = os.environ.get("LPPN_VARIANT", PROFILE["variant"])
SEED = int(os.environ.get("LPPN_SEED", PROFILE["seed"]))
RESULTS_ROOT = resolve_path(os.environ.get("LPPN_OUTPUT_ROOT", PROFILE["output_root"]))
DATA_PATHS = {key: resolve_path(value) for key, value in PROFILE["data"].items()}
TEACHER = PROFILE["teacher"]
TEACHER_WEIGHT = resolve_path(
    os.environ.get("LPPN_TEACHER_WEIGHT",
                   TEACHER.get("weight", RESULTS_ROOT / "eqt" / "demo.weights.h5"))
)
TEACHER_SHA256 = TEACHER.get("sha256")
if DEMO:
    provenance = TEACHER_WEIGHT.with_suffix(".json")
    if provenance.is_file():
        with provenance.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("dataset") != "stead_demo" or metadata.get("trained_from_scratch") is not True:
            raise ValueError("Demo teacher requires a from-scratch STEAD checkpoint.")
        TEACHER_SHA256 = metadata["sha256"]

if DEMO and TEACHER_SHA256 and len(TEACHER_SHA256) != 64:
    raise ValueError("Invalid demo teacher checkpoint hash.")


def cache_root(kind, protocol):
    return (RESULTS_ROOT / kind / DATASET_NAME / TEACHER["id"]
            / (TEACHER_SHA256 or "untrained")[:16] / protocol)
