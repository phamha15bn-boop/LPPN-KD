# Copyright (c) 2026 Qineng Song

"""Joint frozen KD-C and projected feature distillation for LPPN-KD-CFP."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kd import pool_teacher_p_to_cells, teacher_trace_gate
from .feature_cache import FEATURE_SHAPE
from .kdfp import TrainingOnlyFeatureProjection
from .loss import LPPNLoss


class LPPNClassificationFeatureProjectionDistillationLoss(nn.Module):
    """Original supervision + frozen KD-C + frozen identity-projected KD-FP."""

    requires_student_feature = True

    def __init__(
        self,
        projection: TrainingOnlyFeatureProjection,
        *,
        lambda_c: float,
        lambda_fp: float,
        teacher_min_confidence: float,
        teacher_max_error_samples: float,
        stride: int = 8,
        normalization_epsilon: float = 1e-6,
    ):
        super().__init__()
        if lambda_c < 0 or lambda_fp < 0:
            raise ValueError("KD-CFP loss weights must be non-negative.")
        self.projection = projection
        self.lambda_c = float(lambda_c)
        self.lambda_fp = float(lambda_fp)
        self.teacher_min_confidence = float(teacher_min_confidence)
        self.teacher_max_error_samples = float(teacher_max_error_samples)
        self.stride = int(stride)
        self.normalization_epsilon = float(normalization_epsilon)
        self.supervised = LPPNLoss()

    def _trace_gate(
        self,
        peak_probability: torch.Tensor,
        peak_sample: torch.Tensor,
        human_p: torch.Tensor,
        trace_is_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return teacher_trace_gate(
            peak_probability,
            peak_sample,
            human_p,
            minimum_confidence=self.teacher_min_confidence,
            maximum_error_samples=self.teacher_max_error_samples,
            trace_is_noise=trace_is_noise,
        )

    def forward(
        self,
        prediction,
        label: torch.Tensor,
        teacher: Mapping[str, torch.Tensor],
        human_p: torch.Tensor,
        student_feature: torch.Tensor,
        trace_is_noise: torch.Tensor | None = None,
    ):
        logits, _ = prediction
        supervised = self.supervised(prediction, label)
        human_p = human_p.to(logits.device, dtype=torch.long)

        # Frozen KD-C definition: full P curve -> stride pooling, probability-
        # curve peak gate, real 750-cell mask, BCE sum reduction.
        teacher_p = teacher["p_probability"].to(logits.device, dtype=logits.dtype)
        pooled_p, cell_mask = pool_teacher_p_to_cells(
            teacher_p, student_cells=logits.shape[-1], stride=self.stride
        )
        c_peak_probability, c_peak_sample = teacher_p.max(dim=1)
        c_gate = self._trace_gate(
            c_peak_probability, c_peak_sample, human_p, trace_is_noise
        )
        kd_c = logits.sum() * 0.0
        active_c_cells = 0
        if c_gate.any():
            student_p = logits.softmax(dim=1)[:, 1, :]
            c_mask = c_gate[:, None] & cell_mask[None, :]
            kd_c = F.binary_cross_entropy(
                student_p[c_mask].clamp(1e-7, 1 - 1e-7),
                pooled_p[c_mask],
                reduction="sum",
            )
            active_c_cells = int(c_mask.sum().detach().cpu())

        # Frozen KD-FP definition.  The feature cache retains its own audited
        # peak under feature_p_peak_* when composed after the probability cache.
        teacher_feature = teacher["feature"].to(
            logits.device, dtype=logits.dtype
        ).detach()
        if teacher_feature.ndim != 3 or tuple(teacher_feature.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"Teacher feature must be [B,750,32], got {teacher_feature.shape}")
        if student_feature.ndim != 3 or student_feature.shape[1:] != (768, 32):
            raise ValueError(f"Student feature must be [B,768,32], got {student_feature.shape}")
        if student_feature.shape[0] != teacher_feature.shape[0]:
            raise ValueError("Teacher/student feature batch size mismatch.")
        fp_peak_probability = teacher["feature_p_peak_probability"].to(
            logits.device, dtype=logits.dtype
        )
        fp_peak_sample = teacher["feature_p_peak_sample"].to(
            logits.device, dtype=torch.long
        )
        fp_gate = self._trace_gate(
            fp_peak_probability, fp_peak_sample, human_p, trace_is_noise
        )
        student_real = student_feature[:, : FEATURE_SHAPE[0], :]
        projected_student = self.projection(student_real)
        teacher_normalized = F.normalize(
            teacher_feature, p=2, dim=-1, eps=self.normalization_epsilon
        )
        student_normalized = F.normalize(
            projected_student, p=2, dim=-1, eps=self.normalization_epsilon
        )
        if fp_gate.any():
            difference = student_normalized[fp_gate] - teacher_normalized[fp_gate]
            kd_fp = difference.square().mean()
        else:
            kd_fp = projected_student.sum() * 0.0

        total = supervised + self.lambda_c * kd_c + self.lambda_fp * kd_fp
        active_c_traces = c_gate.sum()
        active_fp_traces = fp_gate.sum()
        active_fp_cells = active_fp_traces * FEATURE_SHAPE[0]
        components = {
            "supervised": supervised.detach(),
            "kd_c_raw": kd_c.detach(),
            "kd_fp_raw": kd_fp.detach(),
            "active_c_traces": active_c_traces.detach(),
            "active_c_cells": logits.new_tensor(active_c_cells).detach(),
            "active_fp_traces": active_fp_traces.detach(),
            "active_fp_cells": active_fp_cells.detach(),
            "active_fp_elements": (active_fp_cells * FEATURE_SHAPE[1]).detach(),
        }
        return total, components
