# Copyright (c) 2026 Qineng Song

"""Original phase matching and arrival-time error statistics."""

import math

import numpy as np


def phase_metrics(rows, phase, threshold, tolerance_seconds, sample_rate=100.0):
    true_key = f"true_{phase}"
    confidence_key = f"{phase}_confidence"
    prediction_key = f"{phase}_sample_raw"
    tolerance_samples = tolerance_seconds * sample_rate
    true_positive = false_positive = false_negative = 0
    residuals = []
    output_rows = []

    for row in rows:
        true_sample = int(float(row[true_key]))
        original_length = int(float(row["original_length"]))
        has_true = 0 <= true_sample < original_length
        confidence = float(row[confidence_key])
        has_prediction = confidence >= threshold
        trace_category = str(row.get("trace_category", "")).strip().lower()
        is_noise = trace_category == "noise"
        predicted_sample = (
            float(row[prediction_key]) if has_prediction else math.nan
        )
        hit = False
        residual = math.nan

        if has_true and has_prediction:
            residual = (predicted_sample - true_sample) / sample_rate
            hit = abs(predicted_sample - true_sample) <= tolerance_samples
            if hit:
                true_positive += 1
                residuals.append(residual)
            else:
                false_positive += 1
                false_negative += 1
        elif is_noise and has_prediction:
            false_positive += 1
        elif is_noise:
            pass
        elif has_prediction and not has_true:
            # Preserve the established Shanxi missing-phase rule while keeping
            # it semantically distinct from an explicit noise negative.
            false_positive += 1
        elif has_true and not has_prediction:
            false_negative += 1

        if has_true:
            label_status = "positive"
        elif is_noise:
            label_status = "known_negative_noise"
        else:
            label_status = "missing_non_noise_legacy"

        output_rows.append(
            {
                "trace_name": row["trace_name"],
                "trace_category": trace_category,
                "phase": phase.upper(),
                "true_sample": true_sample if has_true else -1,
                "pred_sample": predicted_sample,
                "confidence": confidence,
                "threshold": threshold,
                "residual_seconds": residual,
                "hit": int(hit),
                "label_status": label_status,
            }
        )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    residual_array = np.asarray(residuals, dtype=float)
    return (
        {
            "threshold": threshold,
            "tolerance_seconds": tolerance_seconds,
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "residual_count_tp": int(residual_array.size),
            "residual_mean_seconds": (
                float(residual_array.mean()) if residual_array.size else None
            ),
            "residual_std_seconds": (
                float(residual_array.std()) if residual_array.size else None
            ),
            "mae_seconds": (
                float(np.abs(residual_array).mean()) if residual_array.size else None
            ),
        },
        output_rows,
    )
