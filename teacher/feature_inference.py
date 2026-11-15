# Copyright (c) 2026 Qineng Song

"""Read-only access to the fixed EQT P-decoder stage-4 feature.

This module builds a view over the already validated deterministic teacher. It
does not edit the original 1EQT source, weights, or probability cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .inference import DeterministicEQTTeacher, sha256_file
from .protocol import (
    INPUT_SHAPE,
    PROTOCOL_ID,
    TEACHER_WEIGHT,
    TEACHER_WEIGHT_SHA256,
)


EXPECTED_FEATURE_SHAPE = (None, 750, 32)
FEATURE_SEMANTIC_ID = "picker_P.decoder.stage4.post_conv"


@dataclass(frozen=True)
class FeatureLayerIdentity:
    semantic_id: str
    keras_name: str
    output_shape: tuple[int | None, ...]
    upstream_anchor: str
    downstream_anchor: str


def _history_parts(tensor):
    history = getattr(tensor, "_keras_history", None)
    if history is None:
        return None
    if hasattr(history, "layer"):
        return history.layer, int(history.node_index), int(history.tensor_index)
    return history[0], int(history[1]), int(history[2])


def _ancestor_layers(tensor) -> set[tf.keras.layers.Layer]:
    """Return exact graph ancestors for one Keras tensor."""

    ancestors: set[tf.keras.layers.Layer] = set()
    visited_nodes: set[tuple[int, int]] = set()

    def visit(current):
        parts = _history_parts(current)
        if parts is None:
            return
        layer, node_index, _ = parts
        key = (id(layer), node_index)
        if key in visited_nodes:
            return
        visited_nodes.add(key)
        ancestors.add(layer)
        node = layer._inbound_nodes[node_index]
        for inbound in tf.nest.flatten(node.input_tensors):
            visit(inbound)

    visit(tensor)
    return ancestors


def _single_output_tensor(layer: tf.keras.layers.Layer):
    outputs = tf.nest.flatten(layer.output)
    if len(outputs) != 1:
        return None
    return outputs[0]


def _shape_tuple(tensor) -> tuple[int | None, ...]:
    return tuple(None if dim is None else int(dim) for dim in tensor.shape)


def locate_p_decoder_stage4(model: tf.keras.Model) -> tuple[tf.keras.layers.Layer, FeatureLayerIdentity]:
    """Select the feature by topology and shape, not by an auto-generated name."""

    attention = model.get_layer("attentionP")
    picker = model.get_layer("picker_P")
    picker_ancestors = _ancestor_layers(_single_output_tensor(picker))

    candidates = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Conv1D):
            continue
        output = _single_output_tensor(layer)
        if output is None or _shape_tuple(output) != EXPECTED_FEATURE_SHAPE:
            continue
        ancestors = _ancestor_layers(output)
        if attention in ancestors and layer in picker_ancestors:
            candidates.append(layer)

    if len(candidates) != 1:
        details = [(layer.name, _shape_tuple(_single_output_tensor(layer))) for layer in candidates]
        raise RuntimeError(
            "Expected exactly one Conv1D [None,750,32] layer between attentionP and "
            f"picker_P, found {details}."
        )

    layer = candidates[0]
    identity = FeatureLayerIdentity(
        semantic_id=FEATURE_SEMANTIC_ID,
        keras_name=layer.name,
        output_shape=_shape_tuple(_single_output_tensor(layer)),
        upstream_anchor=attention.name,
        downstream_anchor=picker.name,
    )
    return layer, identity


class DeterministicEQTFeatureTeacher:
    """Deterministic P probability and P-decoder feature view."""

    def __init__(self, require_gpu: bool = True):
        actual_hash = sha256_file(TEACHER_WEIGHT)
        if actual_hash != TEACHER_WEIGHT_SHA256:
            raise RuntimeError(
                f"Teacher weight SHA256 mismatch: {actual_hash}; "
                f"expected {TEACHER_WEIGHT_SHA256}."
            )

        self.teacher = DeterministicEQTTeacher(require_gpu=require_gpu)
        feature_layer, self.feature_identity = locate_p_decoder_stage4(
            self.teacher.model
        )
        self.model = tf.keras.Model(
            inputs=self.teacher.model.input,
            outputs=[self.teacher.model.get_layer("picker_P").output, feature_layer.output],
            name="eqt082_p_feature_view",
        )

        @tf.function(
            input_signature=[tf.TensorSpec([None, 6000, 3], tf.float32)],
            autograph=False,
        )
        def compiled_predict(batch):
            return self.model(batch, training=False)

        self._compiled_predict = compiled_predict

    def predict_p_and_feature(
        self, normalized_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(normalized_batch, dtype=np.float32)
        if array.ndim != 3 or tuple(array.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Normalized input must be [B,6000,3], got {array.shape}.")
        p_tensor, feature_tensor = self._compiled_predict(
            tf.convert_to_tensor(array, tf.float32)
        )
        p_probability = np.asarray(p_tensor.numpy(), dtype=np.float32)
        feature = np.asarray(feature_tensor.numpy(), dtype=np.float32)
        expected_p = (array.shape[0], 6000, 1)
        expected_feature = (array.shape[0], 750, 32)
        if p_probability.shape != expected_p:
            raise RuntimeError(
                f"Teacher P output shape changed: {p_probability.shape}, expected {expected_p}."
            )
        if feature.shape != expected_feature:
            raise RuntimeError(
                f"Teacher feature shape changed: {feature.shape}, expected {expected_feature}."
            )
        if not np.isfinite(p_probability).all() or not np.isfinite(feature).all():
            raise FloatingPointError("Teacher P output or feature contains non-finite values.")
        return p_probability[..., 0], feature

    def predict_feature(self, normalized_batch: np.ndarray) -> np.ndarray:
        return self.predict_p_and_feature(normalized_batch)[1]

    def identity_metadata(self) -> dict[str, object]:
        return {
            "base_protocol_id": PROTOCOL_ID,
            "teacher_weight": str(TEACHER_WEIGHT),
            "teacher_weight_sha256": TEACHER_WEIGHT_SHA256,
            "feature_semantic_id": self.feature_identity.semantic_id,
            "feature_keras_name": self.feature_identity.keras_name,
            "feature_shape": list(self.feature_identity.output_shape),
            "upstream_anchor": self.feature_identity.upstream_anchor,
            "downstream_anchor": self.feature_identity.downstream_anchor,
            "execution": "tf.function(feature_model(x, training=False))",
        }
