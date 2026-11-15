# Copyright (c) 2026 Qineng Song

"""Generate and verify resumable deterministic EQT P-probability caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .protocol import (
    DEFAULT_CACHE_ROOT,
    DROPOUT_PROTOCOL,
    EXECUTION_MODE,
    NORMALIZATION,
    PROTOCOL_ID,
    CACHE_SPLITS,
    TEACHER_ROOT,
    TEACHER_WEIGHT,
    TEACHER_WEIGHT_SHA256,
)


SCHEMA_VERSION = 1
CACHE_DTYPE = "float32"
TRACE_COLUMNS = ("trace_name", "trace_id", "trace", "id", "name")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        lookup = {item.strip().lower(): item for item in reader.fieldnames}
        column = next((lookup[item] for item in TRACE_COLUMNS if item in lookup), None)
        if column is None:
            raise ValueError(f"CSV has no trace-name column: {reader.fieldnames}")
        rows = []
        for row_number, row in enumerate(reader):
            name = (row.get(column) or "").strip()
            if not name:
                raise ValueError(f"Empty trace_name at CSV data row {row_number}.")
            rows.append({"trace_name": name, "csv_row_index": row_number})
    names = [row["trace_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate trace_name in {path}.")
    return rows


def data_group(h5: h5py.File):
    return h5["data"] if "data" in h5 and isinstance(h5["data"], h5py.Group) else h5


def int_attr(dataset: h5py.Dataset, names: tuple[str, ...], default: int = -1) -> int:
    for name in names:
        if name in dataset.attrs:
            try:
                return int(float(dataset.attrs[name]))
            except (TypeError, ValueError, OverflowError):
                pass
    return default


def canonical_waveform(value: np.ndarray) -> np.ndarray:
    waveform = np.asarray(value, dtype=np.float32)
    if waveform.shape == (6000, 3):
        return waveform
    if waveform.shape == (3, 6000):
        return waveform.T
    raise ValueError(f"Teacher waveform must be (6000,3), got {waveform.shape}.")


def initial_manifest(
    *,
    cache_root: Path,
    batch_size: int,
    shard_size: int,
    adapter_sha256: str,
    source_sha256: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "in_progress",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "cache_root": str(cache_root),
        "teacher": {
            "root": str(TEACHER_ROOT),
            "weight": str(TEACHER_WEIGHT),
            "weight_sha256": TEACHER_WEIGHT_SHA256,
            "source_tree_sha256": source_sha256,
            "adapter_sha256": adapter_sha256,
            "dropout_protocol": DROPOUT_PROTOCOL,
            "execution_mode": EXECUTION_MODE,
        },
        "runtime": runtime,
        "input": {
            "shape": [None, 6000, 3],
            "normalization": NORMALIZATION,
            "channel_order": ["E", "N", "Z"],
        },
        "output": {
            "name": "picker_P",
            "shape_per_trace": [6000],
            "dtype": CACHE_DTYPE,
        },
        "generation": {"batch_size": batch_size, "shard_size": shard_size},
        "splits": {},
    }


def identity_problems(
    manifest: dict[str, Any],
    *,
    cache_root: Path,
    batch_size: int,
    shard_size: int,
    adapter_sha256: str,
    source_sha256: str,
) -> list[str]:
    checks = {
        "schema_version": (manifest.get("schema_version"), SCHEMA_VERSION),
        "protocol_id": (manifest.get("protocol_id"), PROTOCOL_ID),
        "cache_root": (manifest.get("cache_root"), str(cache_root)),
        "teacher.weight_sha256": (
            manifest.get("teacher", {}).get("weight_sha256"),
            TEACHER_WEIGHT_SHA256,
        ),
        "teacher.source_tree_sha256": (
            manifest.get("teacher", {}).get("source_tree_sha256"),
            source_sha256,
        ),
        "teacher.adapter_sha256": (
            manifest.get("teacher", {}).get("adapter_sha256"),
            adapter_sha256,
        ),
        "teacher.execution_mode": (
            manifest.get("teacher", {}).get("execution_mode"),
            EXECUTION_MODE,
        ),
        "output.shape_per_trace": (
            manifest.get("output", {}).get("shape_per_trace"),
            [6000],
        ),
        "output.dtype": (manifest.get("output", {}).get("dtype"), CACHE_DTYPE),
        "generation.batch_size": (
            manifest.get("generation", {}).get("batch_size"),
            batch_size,
        ),
        "generation.shard_size": (
            manifest.get("generation", {}).get("shard_size"),
            shard_size,
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
        raise FileNotFoundError(f"Completed cache shard is missing: {path}")
    if verify_sha256:
        actual_hash = sha256_file(path)
        if actual_hash != shard["sha256"]:
            raise RuntimeError(f"Cache shard SHA256 mismatch: {path}")
    start, end = int(shard["start"]), int(shard["end"])
    expected_rows = csv_rows[start:end]
    with h5py.File(path, "r") as h5:
        curves = h5["p_probability"]
        if curves.shape != (len(expected_rows), 6000) or curves.dtype != np.float32:
            raise RuntimeError(
                f"Cache shape/dtype mismatch in {path}: {curves.shape}/{curves.dtype}"
            )
        names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in h5["trace_name"][...]
        ]
        rows = h5["csv_row_index"][...].astype(np.int64).tolist()
        if names != [item["trace_name"] for item in expected_rows]:
            raise RuntimeError(f"trace_name order mismatch in {path}")
        if rows != [item["csv_row_index"] for item in expected_rows]:
            raise RuntimeError(f"CSV row index mismatch in {path}")
        if not np.isfinite(curves[...]).all():
            raise RuntimeError(f"Non-finite P probability in {path}")


def verify_split(
    manifest: dict[str, Any], split: str, verify_sha256: bool = True
) -> dict[str, Any]:
    if split not in manifest.get("splits", {}):
        raise KeyError(f"Split is absent from manifest: {split}")
    split_info = manifest["splits"][split]
    csv_path = Path(split_info["csv"])
    if sha256_file(csv_path) != split_info["csv_sha256"]:
        raise RuntimeError(f"CSV SHA256 changed for split {split}.")
    rows = read_csv_rows(csv_path)[: int(split_info["sample_count"])]
    for shard in split_info["shards"]:
        if shard.get("status") != "complete":
            raise RuntimeError(f"Incomplete shard listed for {split}: {shard}")
        verify_shard(Path(manifest["cache_root"]), shard, rows, verify_sha256)
    covered = sum(int(item["count"]) for item in split_info["shards"])
    if covered != len(rows):
        raise RuntimeError(f"Split coverage mismatch: {covered} cached, {len(rows)} rows.")
    return {"split": split, "samples": len(rows), "shards": len(split_info["shards"])}


def make_shard(
    *,
    teacher,
    source_h5: h5py.File,
    rows: list[dict[str, Any]],
    split: str,
    start: int,
    end: int,
    batch_size: int,
    path: Path,
) -> dict[str, Any]:
    from .inference import normalize_waveforms

    group = data_group(source_h5)
    count = end - start
    probabilities = np.empty((count, 6000), dtype=np.float32)
    peaks = np.empty(count, dtype=np.int32)
    peak_probabilities = np.empty(count, dtype=np.float32)
    manual_p = np.empty(count, dtype=np.int32)
    manual_s = np.empty(count, dtype=np.int32)
    names = []
    csv_indices = np.empty(count, dtype=np.int64)

    for output_start in range(0, count, batch_size):
        output_end = min(count, output_start + batch_size)
        batch_rows = rows[start + output_start : start + output_end]
        waveforms = []
        for row in batch_rows:
            name = row["trace_name"]
            if name not in group:
                raise KeyError(f"CSV trace is absent from HDF5: {name}")
            waveforms.append(canonical_waveform(group[name][...]))
        batch = normalize_waveforms(np.stack(waveforms))
        p_probability = teacher.predict_p(batch)
        probabilities[output_start:output_end] = p_probability

        for local, row in enumerate(batch_rows, start=output_start):
            name = row["trace_name"]
            dataset = group[name]
            names.append(name)
            csv_indices[local] = int(row["csv_row_index"])
            manual_p[local] = int_attr(dataset, ("p_arrival_sample", "p_pick"))
            s_value = int_attr(dataset, ("s_arrival_sample", "s_pick"))
            manual_s[local] = s_value if 0 <= s_value < 6000 else -1
            peaks[local] = int(np.argmax(p_probability[local - output_start]))
            peak_probabilities[local] = float(
                p_probability[local - output_start, peaks[local]]
            )

    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Teacher generated non-finite P probabilities.")
    string_type = h5py.string_dtype(encoding="utf-8")
    temporary = path.with_suffix(path.suffix + ".partial")
    with h5py.File(temporary, "w") as output:
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["protocol_id"] = PROTOCOL_ID
        output.attrs["split"] = split
        output.attrs["start"] = start
        output.attrs["end"] = end
        output.create_dataset("trace_name", data=np.asarray(names, dtype=object), dtype=string_type)
        output.create_dataset("csv_row_index", data=csv_indices)
        output.create_dataset(
            "p_probability",
            data=probabilities,
            dtype=np.float32,
            chunks=(min(count, 16), 6000),
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
        "file": path.relative_to(path.parents[1]).as_posix(),
        "split": split,
        "start": start,
        "end": end,
        "count": count,
        "status": "complete",
        "sha256": sha256_file(path),
        "p_min": float(probabilities.min()),
        "p_max": float(probabilities.max()),
        "s_missing_count": int(np.count_nonzero(manual_s == -1)),
    }


def generate(args) -> dict[str, Any]:
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size and shard-size must be positive.")
    if args.split not in CACHE_SPLITS:
        raise ValueError("Only train and validation teacher caches are allowed.")
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.json"

    from .inference import DeterministicEQTTeacher, sha256_source_tree

    adapter_sha = sha256_file(Path(__file__).with_name("inference.py"))
    source_sha = sha256_source_tree(TEACHER_ROOT)
    teacher = DeterministicEQTTeacher(require_gpu=not args.allow_cpu)
    runtime = teacher.runtime_metadata()
    if "GPU:0" not in str(runtime["output_device"]) and not args.allow_cpu:
        raise RuntimeError(f"Teacher output did not execute on GPU: {runtime['output_device']}")

    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Cache manifest already exists; use --resume only for the same identity: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems = identity_problems(
            manifest,
            cache_root=cache_root,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            adapter_sha256=adapter_sha,
            source_sha256=source_sha,
        )
        if problems:
            raise RuntimeError("Cache identity mismatch:\n- " + "\n- ".join(problems))
    else:
        manifest = initial_manifest(
            cache_root=cache_root,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            adapter_sha256=adapter_sha,
            source_sha256=source_sha,
            runtime=runtime,
        )
        atomic_json(manifest_path, manifest)

    source = CACHE_SPLITS[args.split]
    hdf5_path, csv_path = source["hdf5"], source["csv"]
    if not hdf5_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Configured {args.split} data is missing.")
    all_rows = read_csv_rows(csv_path)
    count = len(all_rows)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("max-samples must be positive when supplied.")
        count = min(count, args.max_samples)
    rows = all_rows[:count]
    requested_identity = {
        "hdf5": str(hdf5_path),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "csv_total_rows": len(all_rows),
        "sample_count": count,
        "max_samples": args.max_samples,
        "shard_size": args.shard_size,
    }
    existing = manifest["splits"].get(args.split)
    if existing:
        for key, expected in requested_identity.items():
            if existing.get(key) != expected:
                raise RuntimeError(
                    f"Resume split identity mismatch for {key}: "
                    f"cached={existing.get(key)!r}, requested={expected!r}"
                )
        for shard in existing.get("shards", []):
            verify_shard(cache_root, shard, rows, verify_sha256=True)
    else:
        existing = {
            **requested_identity,
            "status": "in_progress",
            "shards": [],
            "started_utc": utc_now(),
        }
        manifest["splits"][args.split] = existing
        atomic_json(manifest_path, manifest)

    completed = sum(int(item["count"]) for item in existing["shards"])
    initial_completed = completed
    if completed > count:
        raise RuntimeError("Completed shard coverage exceeds requested samples.")
    split_dir = cache_root / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generated = 0
    shard_counter = len(existing["shards"])
    import tensorflow as tf

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        pass
    with h5py.File(hdf5_path, "r") as source_h5:
        while completed < count:
            if args.max_shards_this_run is not None and generated >= args.max_shards_this_run:
                break
            end = min(count, completed + args.shard_size)
            path = split_dir / f"part-{shard_counter:05d}.h5"
            if path.exists():
                raise FileExistsError(f"Unregistered shard already exists: {path}")
            shard = make_shard(
                teacher=teacher,
                source_h5=source_h5,
                rows=rows,
                split=args.split,
                start=completed,
                end=end,
                batch_size=args.batch_size,
                path=path,
            )
            existing["shards"].append(shard)
            completed = end
            shard_counter += 1
            generated += 1
            manifest["updated_utc"] = utc_now()
            atomic_json(manifest_path, manifest)

    elapsed = time.perf_counter() - started
    peak_bytes = None
    try:
        peak_bytes = int(tf.config.experimental.get_memory_info("GPU:0")["peak"])
    except Exception:
        pass
    existing["status"] = "complete" if completed == count else "in_progress"
    existing["completed_samples"] = completed
    existing["completed_utc"] = utc_now() if completed == count else None
    existing["last_run"] = {
        "generated_samples": completed - initial_completed,
        "seconds": elapsed,
        "samples_per_second": (
            (completed - initial_completed) / max(elapsed, 1e-9)
        ),
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
        "status": existing["status"],
        "completed_samples": completed,
        "total_samples": count,
        "last_run": existing["last_run"],
        "runtime": runtime,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a resumable deterministic EQT P-probability cache."
    )
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--max-shards-this-run", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.cache_root.resolve() / "cache_manifest.json"
    if args.verify_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = verify_split(manifest, args.split, verify_sha256=True)
    else:
        result = generate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
