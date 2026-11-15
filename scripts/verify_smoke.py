# Copyright (c) 2026 Qineng Song

"""Check demo artifacts, labels, split order, tensor metadata and checkpoint states."""

import csv
import json
import math

import h5py
import numpy as np
import torch

import config as cfg
from lppn import LPPNBase, LPPNDataset
from release_config import DATA_PATHS, DEMO, RESULTS_ROOT, ROOT, TEACHER_SHA256, sha256
from teacher import cache, feature_cache


VARIANTS = ("LPPN-BASE", "LPPN-KD-C", "LPPN-KD-FP", "LPPN-KD-CFP")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_samples():
    counts = {"train": (8, 1), "validation": (2, 0), "test": (2, 1)}
    source = ROOT / "data" / "sample_stead"
    manifest = read_json(source / "sample_manifest.json")
    for name, digest in manifest["files"].items():
        assert sha256(source / name) == digest.upper(), name
    rows, names, events = {}, set(), set()
    traced = {row["trace_name"]: row for row in read_csv(source / "sample_manifest.csv")}
    combined = read_csv(source / "sample.csv")
    assert len(traced) == len(combined) == 14
    with h5py.File(source / "sample.hdf5", "r") as h5:
        waveforms = h5["data"]
        for split, (expected_eq, expected_noise) in counts.items():
            records = read_csv(DATA_PATHS[split + "_csv"])
            rows[split] = records
            assert records == [row for row in combined if row["split"] == split]
            assert len(records) == expected_eq + expected_noise
            assert sum(r["trace_category"] == "noise" for r in records) == expected_noise
            for row in records:
                name = row["trace_name"]
                assert name not in names
                names.add(name)
                signal = waveforms[name]
                assert signal.shape == (6000, 3) and signal.dtype == np.float32
                assert np.isfinite(signal[...]).all()
                p, s = (int(row[k + "_arrival_sample"]) for k in ("p", "s"))
                for key in ("trace_name", "trace_category", "source_id", "p_arrival_sample", "s_arrival_sample"):
                    assert str(signal.attrs[key]) == row[key], (name, key)
                if row["trace_category"] == "noise":
                    assert (p, s) == (-1, -1)
                else:
                    assert 0 <= p < s < 6000 and row["source_id"] not in events
                    events.add(row["source_id"])
        assert names == set(waveforms) == set(traced) and len(events) == 12
    for split in ("train", "test"):
        dataset = LPPNDataset(DATA_PATHS[split + "_hdf5"], csv_path=DATA_PATHS[split + "_csv"])
        try:
            index = next(i for i, meta in enumerate(dataset.metadata) if meta.trace_category == "noise")
            wave, target, info = dataset[index]
            assert tuple(wave.shape) == (3, 6144)
            assert info["p_arrival_sample"] == info["s_arrival_sample"] == -1
            assert torch.all(target[0] == 0) and torch.all(target[1] == 0)
        finally:
            dataset.close()
    return rows


def verify_teacher(rows):
    teacher_dir = RESULTS_ROOT / "eqt"
    weight = teacher_dir / "demo.weights.h5"
    metadata = read_json(weight.with_suffix(".json"))
    assert metadata["sha256"] == sha256(weight) == TEACHER_SHA256
    assert metadata["trained_from_scratch"] and metadata["parameters"] == 373495
    for split in ("validation", "test"):
        summary = read_json(teacher_dir / split / "summary.json")
        peaks = read_csv(teacher_dir / split / "peaks.csv")
        expected = [r["trace_name"] for r in rows[split]]
        assert summary["teacher_sha256"] == metadata["sha256"] and summary["all_finite"]
        assert [r["trace_name"] for r in peaks] == expected
        with np.load(teacher_dir / split / "outputs.npz", allow_pickle=False) as output:
            assert output["trace_name"].tolist() == expected
            for key in ("detector", "picker_P", "picker_S"):
                value = output[key]
                assert value.shape == (len(expected), 6000, 1) and value.dtype == np.float32
                assert np.isfinite(value).all() and ((value >= 0) & (value <= 1)).all()
        for actual, row in zip(peaks, rows[split]):
            for key in ("trace_category", "p_arrival_sample", "s_arrival_sample"):
                assert actual[key] == row[key]
    return {"checkpoint_sha256": metadata["sha256"], "parameters": metadata["parameters"],
            "history": metadata["history"], "gpu": metadata["gpu"]}


def verify_caches(rows):
    result = {}
    for kind, module, path, key, shape in (
        ("probability", cache, cfg.TEACHER_CACHE_MANIFEST, "p_probability", [6000]),
        ("feature", feature_cache, cfg.TEACHER_FEATURE_CACHE_MANIFEST, "feature", [750, 32]),
    ):
        manifest = read_json(path)
        assert set(manifest["splits"]) == {"train", "validation"}
        assert manifest["teacher"]["weight_sha256"] == TEACHER_SHA256
        assert manifest["output"]["shape_per_trace"] == shape
        assert manifest["output"]["dtype"] == "float32"
        splits = {}
        for split, info in manifest["splits"].items():
            verified = module.verify_split(manifest, split)
            assert verified["samples"] == len(rows[split])
            for shard in info["shards"]:
                with h5py.File(path.parent / shard["file"], "r") as h5:
                    actual = h5[key]
                    assert list(actual.shape[1:]) == shape and actual.dtype == np.float32
                    for i, row in enumerate(rows[split][shard["start"]:shard["end"]]):
                        assert int(h5["manual_p_sample"][i]) == int(row["p_arrival_sample"])
                        assert int(h5["manual_s_sample"][i]) == int(row["s_arrival_sample"])
            splits[split] = verified
        result[kind] = {"protocol": manifest["protocol_id"], "shape_per_trace": shape,
                        "sha256": sha256(path), "splits": splits}
    return result


def verify_students(rows):
    records = {}
    for variant in VARIANTS:
        folder = RESULTS_ROOT / "stead_demo" / variant / "seed42"
        training = folder / "training"
        config = read_json(training / "run_config.json")
        history = read_csv(training / "training_log.csv")
        assert len(history) == 1
        assert all(math.isfinite(float(value)) for value in history[0].values())
        needs_c = variant in {"LPPN-KD-C", "LPPN-KD-CFP"}
        needs_fp = variant in {"LPPN-KD-FP", "LPPN-KD-CFP"}
        assert config["student_inference_parameters"] == 43296
        assert config["training_only_projection_parameters"] == (1024 if needs_fp else 0)
        assert config["train_samples"] == 9 and config["validation_samples"] == 2
        for component, required in (("kd_c_raw", needs_c), ("kd_fp_raw", needs_fp)):
            for split in ("train", "validation"):
                key = f"{split}_{component}_per_sample"
                assert (key in history[0]) == required
                if required:
                    assert float(history[0][key]) > 0, (variant, key)
        hashes = {}
        for name in ("best", "last"):
            path = training / f"stead_demo_{variant}_seed42_{name}.pt"
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = LPPNBase()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            assert all(torch.isfinite(value).all() for value in model.state_dict().values())
            assert ("projection_state" in checkpoint) == needs_fp
            hashes[name] = sha256(path)
        pred = read_csv(folder / "prediction/test/raw_predictions.csv")
        test = rows["test"]
        assert [r["trace_name"] for r in pred] == [r["trace_name"] for r in test]
        for result, row in zip(pred, test):
            assert result["trace_category"] == row["trace_category"]
            for phase in ("p", "s"):
                assert int(result[f"true_{phase}"]) == int(row[f"{phase}_arrival_sample"])
                assert math.isfinite(float(result[f"{phase}_sample_raw"]))
                assert 0 <= float(result[f"{phase}_confidence"]) <= 1
        metrics = read_json(folder / "evaluation/test/metrics.json")
        assert metrics["test_samples"] == 3 and metrics["model_variant"] == variant
        assert len(read_csv(folder / "evaluation/test/phase_predictions.csv")) == 6
        records[variant] = {"checkpoint_sha256": hashes, "train": history[0],
                            "device": config["device"], "test_trace_count": len(pred),
                            "student_parameters": 43296,
                            "training_only_projection_parameters": config["training_only_projection_parameters"]}
    return records


def main():
    if not DEMO:
        raise ValueError("Select the STEAD demo profile")
    rows = verify_samples()
    records = {
        "status": "PASS", "demo_only": True,
        "split_counts": {split: len(value) for split, value in rows.items()},
        "teacher": verify_teacher(rows), "caches": verify_caches(rows),
        "students": verify_students(rows),
    }
    with (RESULTS_ROOT / "smoke_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2)
    print("PASS: sample identities and labels, teacher outputs, cache shards, student checkpoints/predictions/evaluations.")


if __name__ == "__main__":
    main()
