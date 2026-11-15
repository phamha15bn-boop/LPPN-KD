# Copyright (c) 2026 Qineng Song

"""Deterministic, GPU-only inference adapter for the fixed EQT teacher.

This module never edits the original ``1EQT`` tree.  It replaces only the
symbol used to construct the two forced SpatialDropout1D calls, before the
original model factory is invoked.  The replacement discards the factory's
hard-coded ``training=True`` and always delegates with ``training=False``.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from .protocol import (
    EXECUTION_MODE,
    INPUT_SHAPE,
    OUTPUT_NAMES,
    PROTOCOL_ID,
    TEACHER_ROOT,
    TEACHER_WEIGHT,
    TEACHER_WEIGHT_SHA256,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def sha256_source_tree(root: Path) -> str:
    """Hash relative paths and bytes of every teacher Python source file."""

    digest = hashlib.sha256()
    files = sorted(Path(root).rglob("*.py"), key=lambda item: item.as_posix().lower())
    if not files:
        raise FileNotFoundError(f"No Python sources under teacher root: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def normalize_waveforms(waveforms: np.ndarray) -> np.ndarray:
    array = np.asarray(waveforms, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or tuple(array.shape[1:]) != INPUT_SHAPE:
        raise ValueError(f"Teacher input must be [B,6000,3], got {array.shape}.")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, keepdims=True)
    return ((array - mean) / (std + np.float32(1e-6))).astype(np.float32)


class DeterministicEQTTeacher:
    """Strictly loaded EQT teacher with a single compiled execution route."""

    def __init__(self, require_gpu: bool = True):
        actual_hash = sha256_file(TEACHER_WEIGHT)
        if actual_hash != TEACHER_WEIGHT_SHA256:
            raise RuntimeError(
                "Teacher weight identity mismatch: "
                f"{actual_hash}, expected {TEACHER_WEIGHT_SHA256}."
            )

        gpus = tf.config.list_physical_devices("GPU")
        if require_gpu and not gpus:
            raise RuntimeError(
                "The formal teacher protocol requires a TensorFlow GPU, but none is visible. "
                "Activate the EQT environment before running this command."
            )
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        tf.keras.utils.set_random_seed(42)

        from .eqtransformer.EQTransformer.core import EqT_utils
        from .eqtransformer.EQTransformer.core.trainer import _build_model

        eqt_utils_source = Path(EqT_utils.__file__).read_text(encoding="utf-8")
        forced_sites = eqt_utils_source.count("SpatialDropout1D(drop_rate)(act_1, training=True)")
        forced_sites += eqt_utils_source.count("SpatialDropout1D(drop_rate)(act_2, training=True)")
        if forced_sites != 2:
            raise RuntimeError(
                f"Expected two forced SpatialDropout source call sites, found {forced_sites}."
            )

        base_dropout = tf.keras.layers.SpatialDropout1D

        class InferenceOnlySpatialDropout1D(base_dropout):
            def call(self, inputs, training=None):
                del training
                return super().call(inputs, training=False)

        args = {
            "input_dimention": INPUT_SHAPE,
            "cnn_blocks": 5,
            "lstm_blocks": 2,
            "padding": "same",
            "activation": "relu",
            "drop_rate": 0.1,
            "loss_weights": [0.05, 0.40, 0.55],
            "loss_types": [
                "binary_crossentropy",
                "binary_crossentropy",
                "binary_crossentropy",
            ],
        }
        original = EqT_utils.SpatialDropout1D
        try:
            EqT_utils.SpatialDropout1D = InferenceOnlySpatialDropout1D
            with contextlib.redirect_stdout(io.StringIO()):
                self.model = _build_model(args)
        finally:
            EqT_utils.SpatialDropout1D = original
        patched = [
            layer
            for layer in self.model.layers
            if isinstance(layer, InferenceOnlySpatialDropout1D)
        ]
        if not patched:
            raise RuntimeError(
                "No inference-only SpatialDropout adapter instances were constructed."
            )
        self.model.load_weights(str(TEACHER_WEIGHT))
        if self.model.count_params() != 373_495:
            raise RuntimeError(f"Unexpected teacher parameter count: {self.model.count_params()}")
        if tuple(self.model.input_shape[1:]) != INPUT_SHAPE:
            raise RuntimeError(f"Unexpected teacher input shape: {self.model.input_shape}")
        output_shapes = [tuple(shape[1:]) for shape in self.model.output_shape]
        if output_shapes != [(6000, 1), (6000, 1), (6000, 1)]:
            raise RuntimeError(f"Unexpected teacher output shapes: {output_shapes}")
        self.patched_spatial_dropout_count = len(patched)
        self.forced_spatial_dropout_source_sites = forced_sites

        @tf.function(
            input_signature=[tf.TensorSpec([None, 6000, 3], tf.float32)],
            autograph=False,
        )
        def compiled_predict(batch):
            return self.model(batch, training=False)

        self._compiled_predict = compiled_predict

    def predict_all(self, normalized_batch: np.ndarray):
        array = np.asarray(normalized_batch, dtype=np.float32)
        if array.ndim != 3 or tuple(array.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Normalized input must be [B,6000,3], got {array.shape}.")
        outputs = self._compiled_predict(tf.convert_to_tensor(array, tf.float32))
        numpy_outputs = tuple(np.asarray(item.numpy(), dtype=np.float32) for item in outputs)
        for name, output in zip(OUTPUT_NAMES, numpy_outputs):
            if output.shape != (array.shape[0], 6000, 1):
                raise RuntimeError(f"{name} shape changed: {output.shape}")
            if not np.isfinite(output).all():
                raise FloatingPointError(f"{name} contains non-finite values.")
        return numpy_outputs

    def predict_p(self, normalized_batch: np.ndarray) -> np.ndarray:
        return self.predict_all(normalized_batch)[1][..., 0]

    @property
    def output_device(self) -> str:
        probe = np.zeros((1, 6000, 3), dtype=np.float32)
        return str(self._compiled_predict(tf.convert_to_tensor(probe))[1].device)

    def runtime_metadata(self) -> dict[str, object]:
        build = tf.sysconfig.get_build_info()
        gpus = tf.config.list_physical_devices("GPU")
        logical = tf.config.list_logical_devices("GPU")
        gpu_name = None
        if gpus:
            gpu_name = tf.config.experimental.get_device_details(gpus[0]).get(
                "device_name"
            )
        return {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "cuda_build": build.get("cuda_version"),
            "cudnn_build": build.get("cudnn_version"),
            "physical_gpus": [str(item) for item in gpus],
            "logical_gpus": [str(item) for item in logical],
            "gpu_name": gpu_name,
            "output_device": self.output_device,
            "execution_mode": EXECUTION_MODE,
            "protocol_id": PROTOCOL_ID,
            "patched_spatial_dropout_count": self.patched_spatial_dropout_count,
            "forced_spatial_dropout_source_sites": self.forced_spatial_dropout_source_sites,
            "model_parameters": self.model.count_params(),
        }


def curve_digest(curves: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for curve in curves:
        array = np.ascontiguousarray(curve, dtype=np.float32)
        digest.update(array.tobytes())
    return digest.hexdigest().upper()
