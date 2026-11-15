# Copyright (c) 2026 Qineng Song

"""Training-only channel projection for the KD-FP experiment."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_cache import FEATURE_SHAPE
from .kd import teacher_trace_gate
from .loss import LPPNLoss


class TrainingOnlyFeatureProjection(nn.Module):
    """A bias-free 32 -> 32 mapping initialized to the exact identity."""

    def __init__(self, channels: int = 32):
        super().__init__()
        if channels != FEATURE_SHAPE[1]:
            raise ValueError(f"KD-FP projection requires {FEATURE_SHAPE[1]} channels.")
        self.linear = nn.Linear(channels, channels, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(channels))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 3 or feature.shape[-1] != self.linear.in_features:
            raise ValueError(
                f"Projection input must be [B,T,{self.linear.in_features}], got {feature.shape}."
            )
        return self.linear(feature)


class LPPNFeatureProjectionDistillationLoss(nn.Module):
    """Original supervision plus gated normalized feature MSE after projection."""

    requires_student_feature = True

    def __init__(
        self,
        projection: TrainingOnlyFeatureProjection,
        *,
        lambda_fp: float,
        teacher_min_confidence: float,
        teacher_max_error_samples: float,
        normalization_epsilon: float = 1e-6,
    ):
        super().__init__()
        if lambda_fp < 0:
            raise ValueError("KD-FP loss weight must be non-negative.")
        self.projection = projection
        self.lambda_fp = float(lambda_fp)
        self.teacher_min_confidence = float(teacher_min_confidence)
        self.teacher_max_error_samples = float(teacher_max_error_samples)
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
        peak_probability = teacher["p_peak_probability"].to(
            logits.device, dtype=logits.dtype
        )
        peak_sample = teacher["p_peak_sample"].to(logits.device, dtype=torch.long)
        trace_gate = self._trace_gate(
            peak_probability, peak_sample, human_p, trace_is_noise
        )

        teacher_feature = teacher["feature"].to(
            logits.device, dtype=logits.dtype
        ).detach()
        if teacher_feature.ndim != 3 or tuple(teacher_feature.shape[1:]) != FEATURE_SHAPE:
            raise ValueError(f"Teacher feature must be [B,750,32], got {teacher_feature.shape}")
        if student_feature.ndim != 3 or student_feature.shape[1:] != (768, 32):
            raise ValueError(f"Student feature must be [B,768,32], got {student_feature.shape}")
        if student_feature.shape[0] != teacher_feature.shape[0]:
            raise ValueError("Teacher/student feature batch size mismatch.")

        student_real = student_feature[:, : FEATURE_SHAPE[0], :]
        projected_student = self.projection(student_real)
        teacher_normalized = F.normalize(
            teacher_feature, p=2, dim=-1, eps=self.normalization_epsilon
        )
        student_normalized = F.normalize(
            projected_student, p=2, dim=-1, eps=self.normalization_epsilon
        )
        if trace_gate.any():
            difference = student_normalized[trace_gate] - teacher_normalized[trace_gate]
            kd_fp = difference.square().mean()
        else:
            # Retain graph connections to both student and projection parameters.
            kd_fp = projected_student.sum() * 0.0

        total = supervised + self.lambda_fp * kd_fp
        active_traces = trace_gate.sum()
        active_cells = active_traces * FEATURE_SHAPE[0]
        components = {
            "supervised": supervised.detach(),
            "kd_fp_raw": kd_fp.detach(),
            "active_fp_traces": active_traces.detach(),
            "active_fp_cells": active_cells.detach(),
            "active_fp_elements": (active_cells * FEATURE_SHAPE[1]).detach(),
        }
        return total, components
