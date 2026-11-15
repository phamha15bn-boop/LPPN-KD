# Copyright (c) 2026 Qineng Song

"""Feature-cache dataset shared by the active projected KD variants."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .kd import sha256_file


FEATURE_PROTOCOL_ID = "EQT082_PDEC4_F32_DETERMINISTIC_OFF_V1"
FEATURE_SHAPE = (750, 32)


def _base_metadata(dataset: Dataset):
    current = dataset
    while not hasattr(current, "metadata") and hasattr(current, "base_dataset"):
        current = current.base_dataset
    metadata = getattr(current, "metadata", None)
    if metadata is None:
        raise TypeError("FeatureCacheDataset requires an LPPNDataset-compatible base.")
    return metadata


class FeatureCacheDataset(Dataset):
    """Attach one exact cached P-decoder feature to an LPPN training item."""

    def __init__(
        self,
        base_dataset: Dataset,
        manifest_path: Path,
        split: str,
        *,
        protocol_id: str,
        teacher_sha256: str,
        verify_sha256: bool = True,
    ):
        self.base_dataset = base_dataset
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Feature-cache manifest is missing: {manifest_path}")
        self.cache_root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("protocol_id") != protocol_id:
            raise RuntimeError("Feature-cache protocol ID mismatch.")
        if self.manifest.get("teacher", {}).get("weight_sha256") != teacher_sha256:
            raise RuntimeError("Feature-cache teacher SHA256 mismatch.")
        output = self.manifest.get("output", {})
        if output.get("shape_per_trace") != list(FEATURE_SHAPE):
            raise RuntimeError("Feature-cache shape is not [750,32].")
        if output.get("dtype") != "float32":
            raise RuntimeError("Feature-cache dtype is not float32.")
        info = self.manifest.get("splits", {}).get(split)
        if not info or info.get("status") != "complete":
            raise RuntimeError(f"Feature-cache split is not complete: {split}")

        self.split = split
        self._handles: dict[str, h5py.File] = {}
        self.index: dict[str, tuple[str, int, int]] = {}
        for shard in info.get("shards", []):
            path = self.cache_root / shard["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_sha256 and sha256_file(path) != shard.get("sha256"):
                raise RuntimeError(f"Feature cache shard SHA256 mismatch: {path}")
            with h5py.File(path, "r") as h5:
                if h5["feature"].shape[1:] != FEATURE_SHAPE:
                    raise RuntimeError(f"Invalid feature shape in {path}")
                if h5["feature"].dtype != np.float32:
                    raise RuntimeError(f"Invalid feature dtype in {path}")
                names = [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in h5["trace_name"][...]
                ]
                rows = h5["csv_row_index"][...].astype(np.int64).tolist()
            for local_index, (name, csv_row) in enumerate(zip(names, rows)):
                if name in self.index:
                    raise RuntimeError(f"Duplicate feature-cache trace_name: {name}")
                self.index[name] = (str(path), local_index, int(csv_row))
        missing = [
            meta.trace_name
            for meta in _base_metadata(base_dataset)
            if meta.trace_name not in self.index
        ]
        if missing:
            raise RuntimeError(
                f"Feature cache lacks {len(missing)} requested {split} traces; "
                f"first: {missing[:3]}"
            )

    def __len__(self):
        return len(self.base_dataset)

    def _handle(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, index):
        item = self.base_dataset[index]
        if len(item) == 3:
            waveform, target, info = item
            teacher: dict[str, object] = {}
        elif len(item) == 4:
            waveform, target, info, teacher_input = item
            teacher = dict(teacher_input)
        else:
            raise RuntimeError(f"Unexpected base item structure: {len(item)}")

        path, local_index, csv_row = self.index[info["trace_name"]]
        h5 = self._handle(path)
        stored_name = h5["trace_name"][local_index]
        if isinstance(stored_name, bytes):
            stored_name = stored_name.decode("utf-8")
        stored_row = int(h5["csv_row_index"][local_index])
        if stored_name != info["trace_name"] or stored_row != csv_row:
            raise RuntimeError("Feature-cache trace identity changed after indexing.")
        feature = np.asarray(h5["feature"][local_index], dtype=np.float32)
        if feature.shape != FEATURE_SHAPE or not np.isfinite(feature).all():
            raise RuntimeError(f"Invalid cached feature for {stored_name}")
        feature_peak_sample = int(h5["p_peak_sample"][local_index])
        feature_peak_probability = float(h5["p_peak_probability"][local_index])
        if "p_peak_sample" in teacher:
            # Joint KD-CFP keeps the probability-cache peak for KD-C and the
            # independently audited feature-cache peak for projected feature KD.
            teacher["feature_p_peak_sample"] = feature_peak_sample
            teacher["feature_p_peak_probability"] = feature_peak_probability
        stored_p = int(h5["manual_p_sample"][local_index])
        stored_s = int(h5["manual_s_sample"][local_index])
        if stored_p != int(info["p_arrival_sample"]) or stored_s != int(
            info["s_arrival_sample"]
        ):
            raise RuntimeError(f"Feature-cache manual label mismatch for {stored_name}")
        teacher["feature"] = torch.from_numpy(feature)
        teacher["csv_row_index"] = stored_row
        if "p_peak_sample" not in teacher:
            teacher["p_peak_sample"] = feature_peak_sample
            teacher["p_peak_probability"] = feature_peak_probability
        return waveform, target, info, teacher

    def close(self):
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        if hasattr(self, "_handles"):
            self._handles.clear()
        close = getattr(self.base_dataset, "close", None)
        if close is not None:
            close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self):
        self.close()
