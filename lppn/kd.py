# Copyright (c) 2026 Qineng Song

"""Offline teacher-cache access and additive KD losses for LPPN variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .loss import LPPNLoss


KD_VARIANTS = {"LPPN-KD-C"}


def teacher_trace_gate(
    peak_probability: torch.Tensor,
    peak_sample: torch.Tensor,
    human_p: torch.Tensor,
    *,
    minimum_confidence: float,
    maximum_error_samples: float,
    trace_is_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gate KD without ever treating a missing manual P as an arrival."""

    valid_manual_p = (human_p >= 0) & (human_p < 6000)
    safe_manual_p = torch.where(valid_manual_p, human_p, peak_sample)
    gate = (
        (peak_probability >= minimum_confidence)
        & valid_manual_p
        & ((peak_sample - safe_manual_p).abs() <= maximum_error_samples)
    )
    if trace_is_noise is not None:
        noise = trace_is_noise.to(gate.device, dtype=torch.bool)
        if noise.shape != gate.shape:
            raise ValueError(
                f"trace_is_noise must have shape {tuple(gate.shape)}, got {tuple(noise.shape)}"
            )
        gate = gate & ~noise
    return gate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def pool_teacher_p_to_cells(
    teacher_p: torch.Tensor, student_cells: int = 768, stride: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Max-pool 6000 teacher samples and return a 750/768 validity mask."""

    if teacher_p.ndim != 2 or teacher_p.shape[1] != 6000:
        raise ValueError(f"teacher_p must have shape [B,6000], got {teacher_p.shape}")
    if 6000 % stride != 0:
        raise ValueError("The fixed teacher length must be divisible by stride.")
    real_cells = 6000 // stride
    if student_cells < real_cells:
        raise ValueError("Student output is shorter than the real teacher time range.")
    pooled = teacher_p.reshape(teacher_p.shape[0], real_cells, stride).amax(dim=2)
    if student_cells > real_cells:
        pooled = F.pad(pooled, (0, student_cells - real_cells))
    mask = torch.zeros(student_cells, dtype=torch.bool, device=teacher_p.device)
    mask[:real_cells] = True
    return pooled, mask


class TeacherCacheDataset(Dataset):
    """Attach one exact cached P curve to every item in an LPPN dataset."""

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
            raise FileNotFoundError(f"Teacher cache manifest is missing: {manifest_path}")
        self.cache_root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("protocol_id") != protocol_id:
            raise RuntimeError("Teacher cache protocol ID mismatch.")
        if self.manifest.get("teacher", {}).get("weight_sha256") != teacher_sha256:
            raise RuntimeError("Teacher cache weight SHA256 mismatch.")
        if self.manifest.get("output", {}).get("shape_per_trace") != [6000]:
            raise RuntimeError("Teacher cache curve shape is not [6000].")
        if self.manifest.get("output", {}).get("dtype") != "float32":
            raise RuntimeError("Teacher cache dtype is not float32.")
        split_info = self.manifest.get("splits", {}).get(split)
        if not split_info or split_info.get("status") != "complete":
            raise RuntimeError(f"Teacher cache split is not complete: {split}")

        self.split = split
        self._handles: dict[str, h5py.File] = {}
        self.index: dict[str, tuple[str, int, int]] = {}
        for shard in split_info.get("shards", []):
            path = self.cache_root / shard["file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_sha256 and sha256_file(path) != shard.get("sha256"):
                raise RuntimeError(f"Teacher cache shard SHA256 mismatch: {path}")
            with h5py.File(path, "r") as h5:
                if h5["p_probability"].shape[1:] != (6000,):
                    raise RuntimeError(f"Invalid teacher curve shape in {path}")
                if h5["p_probability"].dtype != np.float32:
                    raise RuntimeError(f"Invalid teacher curve dtype in {path}")
                names = [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in h5["trace_name"][...]
                ]
                rows = h5["csv_row_index"][...].astype(np.int64).tolist()
            for local_index, (name, csv_row) in enumerate(zip(names, rows)):
                if name in self.index:
                    raise RuntimeError(f"Duplicate cached trace_name: {name}")
                self.index[name] = (str(path), local_index, int(csv_row))
        missing = [meta.trace_name for meta in base_dataset.metadata if meta.trace_name not in self.index]
        if missing:
            raise RuntimeError(
                f"Teacher cache lacks {len(missing)} requested {split} traces; first: {missing[:3]}"
            )

    def __len__(self):
        return len(self.base_dataset)

    def _handle(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, index):
        waveform, target, info = self.base_dataset[index]
        path, local_index, csv_row = self.index[info["trace_name"]]
        h5 = self._handle(path)
        stored_name = h5["trace_name"][local_index]
        if isinstance(stored_name, bytes):
            stored_name = stored_name.decode("utf-8")
        stored_row = int(h5["csv_row_index"][local_index])
        if stored_name != info["trace_name"] or stored_row != csv_row:
            raise RuntimeError("Teacher cache trace identity changed after indexing.")
        curve = np.asarray(h5["p_probability"][local_index], dtype=np.float32)
        if curve.shape != (6000,) or not np.isfinite(curve).all():
            raise RuntimeError(f"Invalid cached P curve for {stored_name}")
        teacher = {
            "p_probability": torch.from_numpy(curve),
            "p_peak_sample": int(h5["p_peak_sample"][local_index]),
            "p_peak_probability": float(h5["p_peak_probability"][local_index]),
            "csv_row_index": stored_row,
        }
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


class LPPNKDLoss(nn.Module):
    """Original sum-reduction supervision plus P/non-P classification KD."""

    def __init__(
        self,
        variant: str,
        *,
        lambda_c: float,
        teacher_min_confidence: float,
        teacher_max_error_samples: float,
        stride: int = 8,
    ):
        super().__init__()
        if variant not in KD_VARIANTS:
            raise ValueError(f"Not a KD variant: {variant}")
        self.variant = variant
        self.lambda_c = float(lambda_c)
        self.teacher_min_confidence = float(teacher_min_confidence)
        self.teacher_max_error_samples = float(teacher_max_error_samples)
        self.stride = int(stride)
        self.supervised = LPPNLoss()

    def forward(
        self,
        prediction,
        label,
        teacher_p,
        human_p,
        trace_is_noise: torch.Tensor | None = None,
    ):
        logits, _ = prediction
        supervised = self.supervised(prediction, label)
        teacher_p = teacher_p.to(logits.device, dtype=logits.dtype)
        human_p = human_p.to(logits.device, dtype=torch.long)
        pooled, cell_mask = pool_teacher_p_to_cells(
            teacher_p, student_cells=logits.shape[-1], stride=self.stride
        )
        teacher_peak_probability, teacher_peak = teacher_p.max(dim=1)
        trace_gate = teacher_trace_gate(
            teacher_peak_probability,
            teacher_peak,
            human_p,
            minimum_confidence=self.teacher_min_confidence,
            maximum_error_samples=self.teacher_max_error_samples,
            trace_is_noise=trace_is_noise,
        )

        kd_c = logits.sum() * 0.0
        active_c_cells = 0
        if trace_gate.any():
            student_p = logits.softmax(dim=1)[:, 1, :]
            mask = trace_gate[:, None] & cell_mask[None, :]
            kd_c = F.binary_cross_entropy(
                student_p[mask].clamp(1e-7, 1 - 1e-7),
                pooled[mask],
                reduction="sum",
            )
            active_c_cells = int(mask.sum().detach().cpu())

        total = supervised + self.lambda_c * kd_c
        components = {
            "supervised": supervised.detach(),
            "kd_c_raw": kd_c.detach(),
            "active_c_traces": trace_gate.sum().detach(),
            "active_c_cells": logits.new_tensor(active_c_cells).detach(),
        }
        return total, components
