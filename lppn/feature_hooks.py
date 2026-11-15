# Copyright (c) 2026 Qineng Song

"""Training-only feature capture helpers that do not alter LPPN forward APIs."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassEncoderFeatureCapture:
    """Capture one ``class_encoder`` output and remove the hook on exit.

    The captured tensor is intentionally not detached so projected feature
    distillation can propagate gradients through the existing student graph.
    """

    def __init__(self, model: nn.Module):
        if not hasattr(model, "class_encoder"):
            raise TypeError("The student model has no class_encoder module.")
        self.model = model
        self._handle = None
        self._captured: torch.Tensor | None = None
        self._capture_count = 0

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("class_encoder must return one torch.Tensor.")
        self._capture_count += 1
        if self._capture_count != 1:
            raise RuntimeError("class_encoder ran more than once in one capture context.")
        self._captured = output

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("Feature capture context is already active.")
        self._captured = None
        self._capture_count = 0
        self._handle = self.model.class_encoder.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def raw_bchw(self) -> torch.Tensor:
        if self._captured is None:
            raise RuntimeError("No class_encoder feature has been captured.")
        expected = (32, 1, 768)
        if self._captured.ndim != 4 or tuple(self._captured.shape[1:]) != expected:
            raise RuntimeError(
                "Unexpected class_encoder feature shape: "
                f"{tuple(self._captured.shape)}; expected [B,32,1,768]."
            )
        return self._captured

    @property
    def feature_btc(self) -> torch.Tensor:
        return self.raw_bchw.squeeze(dim=2).transpose(1, 2).contiguous()
