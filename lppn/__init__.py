# Copyright (c) 2026 Qineng Song

"""Core LPPN-BASE model, data, loss and evaluation utilities."""

from .data import LPPNDataset, make_lppn_target
from .loss import LPPNLoss
from .model import LPPNBase, forward_logits
from .kd import LPPNKDLoss, TeacherCacheDataset, pool_teacher_p_to_cells
from .feature_cache import FeatureCacheDataset
from .feature_hooks import ClassEncoderFeatureCapture
from .dataset_composition import compose_training_dataset
from .kdfp import TrainingOnlyFeatureProjection, LPPNFeatureProjectionDistillationLoss
from .kdcfp import LPPNClassificationFeatureProjectionDistillationLoss

__all__ = [
    "LPPNBase",
    "LPPNDataset",
    "LPPNLoss",
    "forward_logits",
    "make_lppn_target",
    "LPPNKDLoss",
    "TeacherCacheDataset",
    "pool_teacher_p_to_cells",
    "FeatureCacheDataset",
    "ClassEncoderFeatureCapture",
    "compose_training_dataset",
    "TrainingOnlyFeatureProjection",
    "LPPNFeatureProjectionDistillationLoss",
    "LPPNClassificationFeatureProjectionDistillationLoss",
]
