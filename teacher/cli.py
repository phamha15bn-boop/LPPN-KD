# Copyright (c) 2026 Qineng Song

"""Fixed-split EQTransformer training and deterministic prediction."""

import argparse
import contextlib
import csv
import io
import json

import h5py
import numpy as np

from release_config import DATA_PATHS, DATASET_NAME, DEMO, RESULTS_ROOT, SEED, TEACHER, sha256
from .cache import atomic_json, canonical_waveform, data_group, read_csv_rows
from .protocol import OUTPUT_NAMES, TEACHER_WEIGHT


def split_rows(split):
    path = DATA_PATHS[split + "_csv"]
    rows = read_csv_rows(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{**metadata, **identity}
                for metadata, identity in zip(csv.DictReader(stream), rows)]


def require_disjoint_splits(rows):
    names = {split: {row["trace_name"] for row in records} for split, records in rows.items()}
    if sum(map(len, names.values())) != len(set().union(*names.values())):
        raise ValueError("Teacher data splits must have disjoint trace names")


def train(batch_size, epochs):
    import tensorflow as tf
    from .eqtransformer.EQTransformer.core.EqT_utils import DataGenerator, _phase_arrivals, f1
    from .eqtransformer.EQTransformer.core.trainer import _build_model

    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(SEED)
    rows = {split: split_rows(split) for split in ("train", "validation", "test")}
    require_disjoint_splits(rows)
    target = RESULTS_ROOT / "eqt" / ("demo.weights.h5" if DEMO else "trained.weights.h5")
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"Teacher output exists: {target}")
    if DEMO and target != TEACHER_WEIGHT:
        raise ValueError("Demo quality checks require the checkpoint in the demo output root")

    def generator(split, shuffle):
        return DataGenerator(
            [row["trace_name"] for row in rows[split]],
            str(DATA_PATHS[split + "_hdf5"]), batch_size=batch_size, shuffle=shuffle,
        )

    if DEMO:
        for invalid in ("", None, float("nan"), b"", -1):
            attrs = dict(trace_category="noise", p_arrival_sample=invalid, s_arrival_sample=invalid)
            if _phase_arrivals(attrs, "noise-check") != ("noise", -1, -1):
                raise AssertionError("Incorrect noise labels")
        try:
            _phase_arrivals(dict(trace_category="earthquake_local", p_arrival_sample=""),
                            "missing-P-check")
        except ValueError as error:
            if "missing-P-check" not in str(error):
                raise
        else:
            raise AssertionError("Earthquake with invalid P was accepted")
        for split in ("train", "test"):
            noise = [row["trace_name"] for row in rows[split] if row["trace_category"] == "noise"]
            check = DataGenerator(noise, str(DATA_PATHS[split + "_hdf5"]), shuffle=False)
            inputs, labels = check[0]
            if inputs.shape != (len(noise), 6000, 3) or any(np.any(x) for x in labels.values()):
                raise AssertionError(f"Noise labels are not all zero for {split}")
        print("Noise label checks PASS, including both real noise traces and five missing encodings.")

    args = {
        "input_dimention": (6000, 3), "cnn_blocks": 5, "lstm_blocks": 2,
        "padding": "same", "activation": "relu", "drop_rate": 0.1,
        "loss_weights": [0.05, 0.40, 0.55],
        "loss_types": ["binary_crossentropy"] * 3,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        model = _build_model(args)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss=args["loss_types"], loss_weights=args["loss_weights"], metrics=[f1])
    target.parent.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        generator("train", True), validation_data=generator("validation", False),
        epochs=epochs, workers=0, verbose=2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(target), monitor="val_loss",
                                               save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.7,
                                                patience=5, min_lr=1e-7),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                             restore_best_weights=True),
            tf.keras.callbacks.TerminateOnNaN(),
        ],
    ).history
    if not target.is_file() or not all(np.isfinite(values).all() for values in history.values()):
        raise FloatingPointError("Teacher history contains non-finite values or lacks checkpoint")
    checkpoint = {
        "dataset": DATASET_NAME, "trained_from_scratch": True, "sha256": sha256(target),
        "demo_only": DEMO, "seed": SEED, "batch_size": batch_size, "epochs": epochs,
        "parameters": model.count_params(), "architecture": args, "initial_lr": 0.0005,
        "tensorflow": tf.__version__, "keras": tf.keras.__version__,
        "gpu": [tf.config.experimental.get_device_details(gpu).get("device_name")
                for gpu in tf.config.list_physical_devices("GPU")],
        "build_info": {key: value for key, value in tf.sysconfig.get_build_info().items()
                       if key in {"is_cuda_build", "cuda_version", "cudnn_version"}},
        "split_counts": {key: len(value) for key, value in rows.items()},
        "history": {key: [float(value) for value in values] for key, values in history.items()},
    }
    atomic_json(target.with_suffix(".json"), checkpoint)
    print(json.dumps(checkpoint, indent=2))


def predict(split, batch_size, allow_cpu):
    from .inference import DeterministicEQTTeacher, normalize_waveforms

    rows = split_rows(split)
    teacher = DeterministicEQTTeacher(require_gpu=not allow_cpu)
    destination = RESULTS_ROOT / "eqt" / split
    destination.mkdir(parents=True, exist_ok=False)
    parts = {key: [] for key in OUTPUT_NAMES}
    source = DATA_PATHS[split + "_hdf5"]
    with h5py.File(source, "r") as h5:
        waves = data_group(h5)
        for start in range(0, len(rows), batch_size):
            batch = [canonical_waveform(waves[row["trace_name"]][...])
                     for row in rows[start:start + batch_size]]
            outputs = teacher.predict_all(normalize_waveforms(np.stack(batch)))
            for key, output in zip(OUTPUT_NAMES, outputs):
                parts[key].append(output)
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    names = [row["trace_name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Teacher prediction encountered duplicate trace names")
    np.savez_compressed(destination / "outputs.npz", trace_name=np.asarray(names), **arrays)
    with (destination / "peaks.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["trace_name", "trace_category", "p_arrival_sample", "s_arrival_sample",
                  "detector_peak", "p_pick_sample", "p_peak", "s_pick_sample", "s_peak"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            p, s = arrays["picker_P"][i, :, 0], arrays["picker_S"][i, :, 0]
            writer.writerow({
                "trace_name": names[i], "trace_category": row.get("trace_category", ""),
                "p_arrival_sample": row.get("p_arrival_sample", ""),
                "s_arrival_sample": row.get("s_arrival_sample", ""),
                "detector_peak": float(arrays["detector"][i].max()),
                "p_pick_sample": int(p.argmax()), "p_peak": float(p.max()),
                "s_pick_sample": int(s.argmax()), "s_peak": float(s.max()),
            })
    atomic_json(destination / "summary.json", {
        "dataset": DATASET_NAME, "demo_only": DEMO, "split": split, "count": len(rows),
        "teacher_sha256": sha256(TEACHER_WEIGHT),
        "output_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "all_finite": all(np.isfinite(value).all() for value in arrays.values()),
        "runtime": teacher.runtime_metadata(),
    })
    print(f"Wrote {len(rows)} {split} predictions to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    train_parser = actions.add_parser("train")
    train_parser.add_argument("--batch-size", type=int, default=TEACHER.get("batch_size"))
    train_parser.add_argument("--epochs", type=int, default=TEACHER.get("epochs"))
    predict_parser = actions.add_parser("predict")
    predict_parser.add_argument("--split", choices=("validation", "test"), required=True)
    predict_parser.add_argument("--batch-size", type=int, default=3)
    predict_parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if not args.batch_size or args.batch_size <= 0:
        parser.error("A positive --batch-size is required.")
    if args.command == "train":
        if not args.epochs or args.epochs <= 0:
            parser.error("A positive --epochs is required.")
        train(args.batch_size, args.epochs)
    else:
        predict(args.split, args.batch_size, args.allow_cpu)


if __name__ == "__main__":
    main()
