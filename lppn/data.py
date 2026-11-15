# Copyright (c) 2026 Qineng Song

"""HDF5 waveform reading and original stride-8 LPPN label generation."""

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


TRACE_COLUMN_CANDIDATES = ("trace_name", "trace_id", "trace", "id", "name")


@dataclass(frozen=True)
class TraceMeta:
    trace_name: str
    p_arrival_sample: int
    s_arrival_sample: int
    n_samples: int
    trace_category: str


def _to_int(value: object, default: int = -1) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def discover_data_group(h5: h5py.File) -> h5py.Group:
    if "data" in h5 and isinstance(h5["data"], h5py.Group):
        return h5["data"]
    return h5


def read_manifest_trace_names(csv_path: str | os.PathLike[str]) -> list[str]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        lookup = {name.strip().lower(): name for name in reader.fieldnames}
        column = next((lookup[c] for c in TRACE_COLUMN_CANDIDATES if c in lookup), None)
        if column is None:
            raise ValueError(
                f"Cannot find a trace-name column in {path}; got {reader.fieldnames}."
            )
        names = [(row.get(column) or "").strip() for row in reader]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"Manifest contains no trace names: {path}")
    return names


def _read_arrivals(dataset: h5py.Dataset) -> tuple[int, int]:
    attrs = dataset.attrs
    p_arrival = _to_int(attrs.get("p_arrival_sample", attrs.get("p_pick", -1)))
    s_arrival = _to_int(attrs.get("s_arrival_sample", attrs.get("s_pick", -1)))
    return p_arrival, s_arrival


def _read_trace_category(dataset: h5py.Dataset) -> str:
    value = dataset.attrs.get("trace_category", "")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip().lower()


def _canonical_waveform(data: np.ndarray) -> np.ndarray:
    waveform = np.asarray(data)
    if waveform.ndim != 2:
        raise ValueError(f"Waveform must be 2-D, got {waveform.shape}")
    if waveform.shape[1] == 3:
        return waveform
    if waveform.shape[0] == 3:
        return waveform.T
    raise ValueError(f"Waveform must have three components, got {waveform.shape}")


def normalize_waveform(waveform: np.ndarray, mode: str) -> np.ndarray:
    output = np.asarray(waveform, dtype=np.float32).copy()
    if not np.isfinite(output).all():
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "eqt":
        mean = output.mean(axis=0, keepdims=True)
        std = output.std(axis=0, keepdims=True)
        output = (output - mean) / (std + 1e-6)
    elif mode == "lppn":
        output -= output.mean(axis=0, keepdims=True)
        output /= np.max(np.abs(output)) + 1e-6
    elif mode != "none":
        raise ValueError(f"Unknown normalization mode: {mode}")
    return output.astype(np.float32, copy=False)


def pad_or_crop_right(waveform: np.ndarray, target_length: int) -> np.ndarray:
    length = waveform.shape[0]
    if length == target_length:
        return waveform
    if length > target_length:
        return waveform[:target_length]
    return np.pad(waveform, ((0, target_length - length), (0, 0)), mode="constant")


def make_lppn_target(
    p_arrival_sample: int,
    s_arrival_sample: int,
    target_length: int = 6144,
    stride: int = 8,
) -> np.ndarray:
    """Build [class, offset] labels; a missing phase produces no phase target."""

    if target_length % stride != 0:
        raise ValueError("target_length must be divisible by stride")
    target = np.zeros((2, target_length // stride), dtype=np.float32)
    for phase_class, arrival in ((1, p_arrival_sample), (2, s_arrival_sample)):
        if not (0 <= arrival < target_length):
            continue
        cell = arrival // stride
        lower = max(0, cell - 1)
        upper = min(target.shape[1], cell + 2)
        target[:, lower:upper] = -1.0
        target[0, cell] = float(phase_class)
        target[1, cell] = float(arrival % stride)
    return target


class LPPNDataset(Dataset):
    """Read fixed EQTransformer-style splits without creating new splits."""

    def __init__(
        self,
        h5_path: str | os.PathLike[str],
        csv_path: Optional[str | os.PathLike[str]] = None,
        target_length: int = 6144,
        stride: int = 8,
        normalization: str = "eqt",
        amplitude_scale: bool = False,
        gaussian_noise_std: float = 0.0,
        max_samples: Optional[int] = None,
        trace_names: Optional[list[str]] = None,
        seed: int = 42,
        strict: bool = True,
    ):
        self.h5_path = str(Path(h5_path).expanduser().resolve())
        self.csv_path = str(Path(csv_path).expanduser().resolve()) if csv_path else None
        self.target_length = int(target_length)
        self.stride = int(stride)
        self.normalization = normalization
        self.amplitude_scale = amplitude_scale
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.strict = bool(strict)
        self._h5 = None
        self._group = None

        if self.target_length % 64 != 0:
            raise ValueError("LPPN input length must be divisible by 64; use 6144.")
        if not Path(self.h5_path).is_file():
            raise FileNotFoundError(self.h5_path)

        with h5py.File(self.h5_path, "r") as h5:
            group = discover_data_group(h5)
            available = set(group.keys())
            requested = list(trace_names) if trace_names is not None else (
                read_manifest_trace_names(self.csv_path)
                if self.csv_path
                else sorted(available)
            )
            missing = [name for name in requested if name not in available]
            if missing and self.strict:
                raise KeyError(
                    f"{len(missing)} manifest traces are absent from HDF5; first: "
                    + ", ".join(missing[:5])
                )
            metadata = []
            skipped = 0
            for name in (name for name in requested if name in available):
                obj = group[name]
                if not isinstance(obj, h5py.Dataset):
                    skipped += 1
                    continue
                try:
                    shape = obj.shape
                    if len(shape) != 2 or 3 not in shape:
                        raise ValueError(f"invalid waveform shape {shape}")
                    n_samples = int(shape[0] if shape[1] == 3 else shape[1])
                    p_arrival, s_arrival = _read_arrivals(obj)
                    trace_category = _read_trace_category(obj)
                    is_noise = trace_category == "noise"
                    if is_noise:
                        if p_arrival != -1 or s_arrival != -1:
                            raise ValueError(
                                "noise trace must not contain P/S arrival labels: "
                                f"P={p_arrival}, S={s_arrival}"
                            )
                    elif not (0 <= p_arrival < n_samples):
                        raise ValueError(f"invalid P label {p_arrival}")
                    if is_noise or not (0 <= s_arrival < n_samples):
                        s_arrival = -1
                    metadata.append(
                        TraceMeta(name, p_arrival, s_arrival, n_samples, trace_category)
                    )
                except Exception:
                    if self.strict:
                        raise
                    skipped += 1

        if not metadata:
            raise ValueError("No valid traces found")
        if max_samples is not None and max_samples > 0 and len(metadata) > max_samples:
            metadata = random.Random(seed).sample(metadata, int(max_samples))
        self.metadata = metadata
        self.skipped = skipped

    def __len__(self):
        return len(self.metadata)

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._group = discover_data_group(self._h5)
        return self._group

    def close(self):
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None
        self._group = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_group"] = None
        return state

    def __getitem__(self, index):
        meta = self.metadata[index]
        waveform = _canonical_waveform(self._ensure_open()[meta.trace_name][...])
        original_length = waveform.shape[0]
        waveform = normalize_waveform(waveform, self.normalization)
        if self.amplitude_scale:
            waveform *= np.float32(np.random.uniform(0.5, 2.0))
        if self.gaussian_noise_std > 0:
            waveform += np.random.normal(
                0.0, self.gaussian_noise_std, waveform.shape
            ).astype(np.float32)
        waveform = pad_or_crop_right(waveform, self.target_length)
        target = make_lppn_target(
            meta.p_arrival_sample,
            meta.s_arrival_sample,
            target_length=self.target_length,
            stride=self.stride,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(waveform.T)),
            torch.from_numpy(target),
            {
                "trace_name": meta.trace_name,
                "p_arrival_sample": meta.p_arrival_sample,
                "s_arrival_sample": meta.s_arrival_sample,
                "original_length": original_length,
                "trace_category": meta.trace_category,
            },
        )
