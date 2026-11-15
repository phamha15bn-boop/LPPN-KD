# Copyright (c) 2026 Qineng Song

"""Generate and verify resumable deterministic EQT feature caches.

The cache is intentionally independent from the existing P-probability cache.
It uses the same frozen split ordering and verifies that ordering against the
probability cache before a feature shard is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cache import (
    atomic_json,
    canonical_waveform,
    data_group,
    int_attr,
    read_csv_rows,
    sha256_file,
    utc_now,
)
from .protocol import (
    NORMALIZATION,
    PROTOCOL_ID as P_PROTOCOL_ID,
    CACHE_SPLITS,
    TEACHER_ROOT,
    TEACHER_WEIGHT,
    TEACHER_WEIGHT_SHA256,
)


from release_config import TEACHER, cache_root

SCHEMA_VERSION = 1
FEATURE_PROTOCOL_ID = TEACHER["feature_protocol"]
FEATURE_SEMANTIC_ID = "picker_P.decoder.stage4.post_conv"
FEATURE_SHAPE = (750, 32)
FEATURE_DTYPE = "float32"
DEFAULT_FEATURE_CACHE_ROOT = cache_root("teacher_feature_cache", FEATURE_PROTOCOL_ID)
DEFAULT_PROBABILITY_MANIFEST = cache_root("teacher_cache", P_PROTOCOL_ID) / "cache_manifest.json"


def _decode_strings(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _probability_split_identity(
    manifest_path: Path,
    split: str,
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the matching prefix in the existing probability cache."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Probability-cache manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != P_PROTOCOL_ID:
        raise RuntimeError("Probability-cache protocol identity mismatch.")
    if manifest.get("teacher", {}).get("weight_sha256") != TEACHER_WEIGHT_SHA256:
        raise RuntimeError("Probability-cache teacher SHA256 mismatch.")
    info = manifest.get("splits", {}).get(split)
    if not info or info.get("status") != "complete":
        raise RuntimeError(f"Probability-cache split is not complete: {split}")

    names: list[str] = []
    indices: list[int] = []
    needed = len(expected_rows)
    for shard in info.get("shards", []):
        if len(names) >= needed:
            break
        path = manifest_path.parent / shard["file"]
        if sha256_file(path) != shard.get("sha256"):
            raise RuntimeError(f"Probability-cache shard SHA256 mismatch: {path}")
        with h5py.File(path, "r") as h5:
            names.extend(_decode_strings(h5["trace_name"][...]))
            indices.extend(h5["csv_row_index"][...].astype(np.int64).tolist())
    names = names[:needed]
    indices = indices[:needed]
    if names != [row["trace_name"] for row in expected_rows]:
        raise RuntimeError(f"Probability/feature trace order mismatch for {split}.")
    if indices != [row["csv_row_index"] for row in expected_rows]:
        raise RuntimeError(f"Probability/feature CSV-row order mismatch for {split}.")
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_id": manifest["protocol_id"],
        "split_sample_count": int(info["sample_count"]),
        "verified_prefix_count": needed,
    }


def _identity_files() -> dict[str, str]:
    digest = hashlib.sha256()
    source_files = sorted(
        TEACHER_ROOT.rglob("*.py"), key=lambda item: item.as_posix().lower()
    )
    if not source_files:
        raise FileNotFoundError(f"No Python sources under teacher root: {TEACHER_ROOT}")
    for path in source_files:
        relative = path.relative_to(TEACHER_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())

    return {
        "teacher_source_tree_sha256": digest.hexdigest().upper(),
        "base_inference_adapter_sha256": sha256_file(Path(__file__).with_name("inference.py")),
        "feature_adapter_sha256": sha256_file(Path(__file__).with_name("feature_inference.py")),
        "cache_adapter_sha256": sha256_file(Path(__file__)),
    }


def _new_manifest(
    cache_root: Path,
    batch_size: int,
    shard_size: int,
    runtime: dict[str, Any],
    identities: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": FEATURE_PROTOCOL_ID,
        "status": "in_progress",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "cache_root": str(cache_root),
        "teacher": {
            "root": str(TEACHER_ROOT),
            "weight": str(TEACHER_WEIGHT),
            "weight_sha256": TEACHER_WEIGHT_SHA256,
            **identities,
            "base_probability_protocol_id": P_PROTOCOL_ID,
        },
        "runtime": runtime,
        "input": {
            "shape": [None, 6000, 3],
            "normalization": NORMALIZATION,
            "channel_order": ["E", "N", "Z"],
        },
        "output": {
            "name": "p_decoder_stage4_post_conv",
            "semantic_id": FEATURE_SEMANTIC_ID,
            "shape_per_trace": list(FEATURE_SHAPE),
            "layout": "time,channel",
            "dtype": FEATURE_DTYPE,
        },
        "generation": {"batch_size": batch_size, "shard_size": shard_size},
        "splits": {},
    }


def _identity_problems(
    manifest: dict[str, Any],
    cache_root: Path,
    batch_size: int,
    shard_size: int,
    identities: dict[str, str],
) -> list[str]:
    checks = {
        "schema_version": (manifest.get("schema_version"), SCHEMA_VERSION),
        "protocol_id": (manifest.get("protocol_id"), FEATURE_PROTOCOL_ID),
        "cache_root": (manifest.get("cache_root"), str(cache_root)),
        "teacher.weight_sha256": (
            manifest.get("teacher", {}).get("weight_sha256"),
            TEACHER_WEIGHT_SHA256,
        ),
        "teacher.source_tree_sha256": (
            manifest.get("teacher", {}).get("teacher_source_tree_sha256"),
            identities["teacher_source_tree_sha256"],
        ),
        "teacher.base_inference_adapter_sha256": (
            manifest.get("teacher", {}).get("base_inference_adapter_sha256"),
            identities["base_inference_adapter_sha256"],
        ),
        "teacher.feature_adapter_sha256": (
            manifest.get("teacher", {}).get("feature_adapter_sha256"),
            identities["feature_adapter_sha256"],
        ),
        "teacher.cache_adapter_sha256": (
            manifest.get("teacher", {}).get("cache_adapter_sha256"),
            identities["cache_adapter_sha256"],
        ),
        "output.semantic_id": (
            manifest.get("output", {}).get("semantic_id"),
            FEATURE_SEMANTIC_ID,
        ),
        "output.shape_per_trace": (
            manifest.get("output", {}).get("shape_per_trace"),
            list(FEATURE_SHAPE),
        ),
        "output.dtype": (manifest.get("output", {}).get("dtype"), FEATURE_DTYPE),
        "generation.batch_size": (
            manifest.get("generation", {}).get("batch_size"), batch_size
        ),
        "generation.shard_size": (
            manifest.get("generation", {}).get("shard_size"), shard_size
        ),
    }
    return [
        f"{name}: cached={actual!r}, requested={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]


def verify_shard(
    cache_root: Path,
    shard: dict[str, Any],
    csv_rows: list[dict[str, Any]],
    verify_sha256: bool = True,
) -> None:
    path = cache_root / shard["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Completed feature shard is missing: {path}")
    if verify_sha256 and sha256_file(path) != shard.get("sha256"):
        raise RuntimeError(f"Feature shard SHA256 mismatch: {path}")
    start, end = int(shard["start"]), int(shard["end"])
    expected = csv_rows[start:end]
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("protocol_id") != FEATURE_PROTOCOL_ID:
            raise RuntimeError(f"Feature shard protocol mismatch: {path}")
        features = h5["feature"]
        if features.shape != (len(expected), *FEATURE_SHAPE) or features.dtype != np.float32:
            raise RuntimeError(f"Feature shape/dtype mismatch: {features.shape}/{features.dtype}")
        names = _decode_strings(h5["trace_name"][...])
        rows = h5["csv_row_index"][...].astype(np.int64).tolist()
        if names != [item["trace_name"] for item in expected]:
            raise RuntimeError(f"trace_name order mismatch in {path}")
        if rows != [item["csv_row_index"] for item in expected]:
            raise RuntimeError(f"CSV row index mismatch in {path}")
        for start_index in range(0, len(expected), 32):
            if not np.isfinite(features[start_index : start_index + 32]).all():
                raise RuntimeError(f"Non-finite teacher feature in {path}")


def verify_split(manifest: dict[str, Any], split: str, verify_sha256: bool = True) -> dict[str, Any]:
    info = manifest.get("splits", {}).get(split)
    if not info:
        raise KeyError(f"Split is absent from manifest: {split}")
    if info.get("status") != "complete":
        raise RuntimeError(f"Feature cache split is not complete: {split}")
    csv_path = Path(info["csv"])
    if sha256_file(csv_path) != info.get("csv_sha256"):
        raise RuntimeError(f"CSV SHA256 changed for {split}.")
    rows = read_csv_rows(csv_path)[: int(info["sample_count"])]
    for shard in info.get("shards", []):
        if shard.get("status") != "complete":
            raise RuntimeError(f"Incomplete feature shard in manifest: {shard}")
        verify_shard(Path(manifest["cache_root"]), shard, rows, verify_sha256)
    covered = sum(int(shard["count"]) for shard in info.get("shards", []))
    if covered != len(rows):
        raise RuntimeError(f"Feature coverage mismatch: {covered} vs {len(rows)}")
    probability = _probability_split_identity(
        Path(info["probability_cache"]["manifest"]), split, rows
    )
    if probability["manifest_sha256"] != info["probability_cache"]["manifest_sha256"]:
        raise RuntimeError("Referenced probability-cache manifest SHA256 changed.")
    return {"split": split, "samples": len(rows), "shards": len(info["shards"])}


def _make_shard(
    *,
    teacher,
    source_h5: h5py.File,
    rows: list[dict[str, Any]],
    split: str,
    start: int,
    end: int,
    batch_size: int,
    path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    from .inference import normalize_waveforms

    count = end - start
    features = np.empty((count, *FEATURE_SHAPE), dtype=np.float32)
    peaks = np.empty(count, dtype=np.int32)
    peak_probabilities = np.empty(count, dtype=np.float32)
    manual_p = np.empty(count, dtype=np.int32)
    manual_s = np.empty(count, dtype=np.int32)
    csv_indices = np.empty(count, dtype=np.int64)
    names: list[str] = []
    group = data_group(source_h5)

    for output_start in range(0, count, batch_size):
        output_end = min(count, output_start + batch_size)
        batch_rows = rows[start + output_start : start + output_end]
        waveforms = []
        for row in batch_rows:
            name = row["trace_name"]
            if name not in group:
                raise KeyError(f"CSV trace is absent from HDF5: {name}")
            waveforms.append(canonical_waveform(group[name][...]))
        normalized = normalize_waveforms(np.stack(waveforms))
        p_probability, feature = teacher.predict_p_and_feature(normalized)
        features[output_start:output_end] = feature
        for local, row in enumerate(batch_rows, start=output_start):
            batch_index = local - output_start
            name = row["trace_name"]
            dataset = group[name]
            names.append(name)
            csv_indices[local] = int(row["csv_row_index"])
            manual_p[local] = int_attr(dataset, ("p_arrival_sample", "p_pick"))
            s_value = int_attr(dataset, ("s_arrival_sample", "s_pick"))
            manual_s[local] = s_value if 0 <= s_value < 6000 else -1
            peaks[local] = int(np.argmax(p_probability[batch_index]))
            peak_probabilities[local] = float(p_probability[batch_index, peaks[local]])

    if not np.isfinite(features).all():
        raise FloatingPointError("Teacher generated non-finite features.")
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(temporary, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["protocol_id"] = FEATURE_PROTOCOL_ID
        output.attrs["split"] = split
        output.attrs["start"] = start
        output.attrs["end"] = end
        output.create_dataset("trace_name", data=np.asarray(names, dtype=object), dtype=string_type)
        output.create_dataset("csv_row_index", data=csv_indices)
        output.create_dataset(
            "feature",
            data=features,
            dtype=np.float32,
            chunks=(min(count, 4), *FEATURE_SHAPE),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        output.create_dataset("p_peak_sample", data=peaks)
        output.create_dataset("p_peak_probability", data=peak_probabilities)
        output.create_dataset("manual_p_sample", data=manual_p)
        output.create_dataset("manual_s_sample", data=manual_s)
        output.create_dataset("s_missing", data=manual_s == -1)
    os.replace(temporary, path)
    return {
        "file": path.relative_to(cache_root).as_posix(),
        "split": split,
        "start": start,
        "end": end,
        "count": count,
        "status": "complete",
        "sha256": sha256_file(path),
        "first_trace_name": names[0],
        "last_trace_name": names[-1],
        "first_csv_row_index": int(csv_indices[0]),
        "last_csv_row_index": int(csv_indices[-1]),
        "feature_min": float(features.min()),
        "feature_max": float(features.max()),
        "s_missing_count": int(np.count_nonzero(manual_s == -1)),
    }


def generate(args) -> dict[str, Any]:
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size and shard-size must be positive.")
    if args.split not in CACHE_SPLITS:
        raise ValueError("Only train and validation feature caches are allowed.")
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.json"
    identities = _identity_files()

    source = CACHE_SPLITS[args.split]
    hdf5_path, csv_path = source["hdf5"], source["csv"]
    if not hdf5_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Configured {args.split} data is missing.")
    all_rows = read_csv_rows(csv_path)
    count = len(all_rows)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive.")
        count = min(count, args.max_samples)
    rows = all_rows[:count]
    probability_identity = _probability_split_identity(
        args.probability_manifest.resolve(), args.split, rows
    )

    # TensorFlow is imported only for generation, never for verification or training.
    from .feature_inference import DeterministicEQTFeatureTeacher

    teacher = DeterministicEQTFeatureTeacher(require_gpu=not args.allow_cpu)
    runtime = teacher.teacher.runtime_metadata()
    runtime["feature_identity"] = teacher.identity_metadata()
    if "GPU:0" not in str(runtime.get("output_device")) and not args.allow_cpu:
        raise RuntimeError(f"Teacher feature did not execute on GPU: {runtime.get('output_device')}")

    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Feature-cache manifest exists; use --resume for the same identity: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = _identity_problems(
            manifest, cache_root, args.batch_size, args.shard_size, identities
        )
        if problems:
            raise RuntimeError("Feature-cache identity mismatch:\n- " + "\n- ".join(problems))
    else:
        manifest = _new_manifest(
            cache_root, args.batch_size, args.shard_size, runtime, identities
        )
        atomic_json(manifest_path, manifest)

    requested = {
        "hdf5": str(hdf5_path),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "csv_total_rows": len(all_rows),
        "sample_count": count,
        "max_samples": args.max_samples,
        "shard_size": args.shard_size,
        "probability_cache": probability_identity,
    }
    info = manifest["splits"].get(args.split)
    if info:
        for key, expected in requested.items():
            if info.get(key) != expected:
                raise RuntimeError(
                    f"Resume split identity mismatch for {key}: "
                    f"cached={info.get(key)!r}, requested={expected!r}"
                )
        for shard in info.get("shards", []):
            verify_shard(cache_root, shard, rows, verify_sha256=True)
    else:
        info = {**requested, "status": "in_progress", "shards": [], "started_utc": utc_now()}
        manifest["splits"][args.split] = info
        atomic_json(manifest_path, manifest)

    completed = sum(int(shard["count"]) for shard in info["shards"])
    initial_completed = completed
    if completed > count:
        raise RuntimeError("Completed feature coverage exceeds requested samples.")
    split_dir = cache_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated_shards = 0

    import tensorflow as tf

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass
    with h5py.File(hdf5_path, "r") as source_h5:
        while completed < count:
            if args.max_shards_this_run is not None and generated_shards >= args.max_shards_this_run:
                break
            end = min(count, completed + args.shard_size)
            path = split_dir / f"part-{len(info['shards']):05d}.h5"
            if path.exists():
                raise FileExistsError(f"Unregistered feature shard exists: {path}")
            shard = _make_shard(
                teacher=teacher,
                source_h5=source_h5,
                rows=rows,
                split=args.split,
                start=completed,
                end=end,
                batch_size=args.batch_size,
                path=path,
                cache_root=cache_root,
            )
            info["shards"].append(shard)
            completed = end
            generated_shards += 1
            manifest["updated_utc"] = utc_now()
            atomic_json(manifest_path, manifest)

    elapsed = time.perf_counter() - started
    peak_bytes = None
    try:
        peak_bytes = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        pass
    info["status"] = "complete" if completed == count else "in_progress"
    info["completed_samples"] = completed
    info["completed_utc"] = utc_now() if completed == count else None
    info["last_run"] = {
        "generated_samples": completed - initial_completed,
        "seconds": elapsed,
        "samples_per_second": (completed - initial_completed) / max(elapsed, 1e-9),
        "gpu_peak_bytes": peak_bytes,
        "batch_size": args.batch_size,
    }
    manifest["status"] = (
        "complete"
        if {"train", "validation"}.issubset(manifest["splits"])
        and all(manifest["splits"][name]["status"] == "complete" for name in ("train", "validation"))
        else "in_progress"
    )
    manifest["updated_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "split": args.split,
        "status": info["status"],
        "completed_samples": completed,
        "total_samples": count,
        "last_run": info["last_run"],
        "feature_identity": teacher.identity_metadata(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create a deterministic EQT feature cache.")
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--max-shards-this-run", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_FEATURE_CACHE_ROOT)
    parser.add_argument("--probability-manifest", type=Path, default=DEFAULT_PROBABILITY_MANIFEST)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.cache_root.resolve() / "cache_manifest.json"
    if args.verify_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identities = _identity_files()
        problems = _identity_problems(
            manifest,
            args.cache_root.resolve(),
            int(manifest["generation"]["batch_size"]),
            int(manifest["generation"]["shard_size"]),
            identities,
        )
        if problems:
            raise RuntimeError("Feature-cache identity mismatch:\n- " + "\n- ".join(problems))
        result = verify_split(manifest, args.split, verify_sha256=True)
    else:
        result = generate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
