# Copyright (c) 2026 Qineng Song

"""Explicitly compose the cache-backed datasets required by each LPPN variant."""

from pathlib import Path

from torch.utils.data import Dataset

from .kd import TeacherCacheDataset
from .feature_cache import FeatureCacheDataset


CLASSIFICATION_CACHE_VARIANTS = frozenset({"LPPN-KD-C", "LPPN-KD-CFP"})
FEATURE_CACHE_VARIANTS = frozenset({"LPPN-KD-FP", "LPPN-KD-CFP"})
SUPPORTED_VARIANTS = frozenset(
    {"LPPN-BASE"} | CLASSIFICATION_CACHE_VARIANTS | FEATURE_CACHE_VARIANTS
)


def compose_training_dataset(
    base_dataset: Dataset,
    *,
    variant: str,
    split: str,
    probability_manifest: Path,
    feature_manifest: Path,
    probability_protocol_id: str,
    feature_protocol_id: str,
    teacher_sha256: str,
    verify_sha256: bool = True,
) -> Dataset:
    """Return the exact dataset stack required by ``variant``.

    The order is intentional.  For KD-CFP the probability wrapper always sees
    the original LPPNDataset, and the feature wrapper merges its data into the
    existing teacher dictionary.  TeacherCacheDataset therefore never relies
    on another wrapper exposing an incidental ``metadata`` attribute.
    """

    normalized_variant = variant.strip().upper()
    if normalized_variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported dataset variant: {variant!r}")
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported training split: {split!r}")

    dataset = base_dataset
    if normalized_variant in CLASSIFICATION_CACHE_VARIANTS:
        dataset = TeacherCacheDataset(
            dataset,
            probability_manifest,
            split,
            protocol_id=probability_protocol_id,
            teacher_sha256=teacher_sha256,
            verify_sha256=verify_sha256,
        )
    if normalized_variant in FEATURE_CACHE_VARIANTS:
        dataset = FeatureCacheDataset(
            dataset,
            feature_manifest,
            split,
            protocol_id=feature_protocol_id,
            teacher_sha256=teacher_sha256,
            verify_sha256=verify_sha256,
        )
    return dataset
