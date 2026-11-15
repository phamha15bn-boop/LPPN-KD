# Copyright (c) 2026 Qineng Song

"""Train LPPN BASE or an offline-cache KD variant from scratch."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from release_config import DEMO
from lppn import (
    ClassEncoderFeatureCapture,
    LPPNBase,
    LPPNDataset,
    LPPNFeatureProjectionDistillationLoss,
    LPPNClassificationFeatureProjectionDistillationLoss,
    LPPNKDLoss,
    LPPNLoss,
    TrainingOnlyFeatureProjection,
    compose_training_dataset,
    forward_logits,
)
from lppn.kd import sha256_file


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def load_checkpoint(path: Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def require_absolute_file(path_value, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute Windows path: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def run_epoch(model, loader, loss_function, device, optimizer=None, grad_clip=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    component_totals = {}
    started = time.perf_counter()

    for batch in loader:
        if len(batch) == 3:
            waveform, target, info = batch
            teacher = None
        elif len(batch) == 4:
            waveform, target, info, teacher = batch
        else:
            raise RuntimeError(f"Unexpected training batch structure: {len(batch)}")
        waveform = waveform.to(device, dtype=torch.float32, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        batch_size = waveform.shape[0]
        categories = info.get("trace_category")
        trace_is_noise = None
        if categories is not None:
            trace_is_noise = torch.tensor(
                [str(value).strip().lower() == "noise" for value in categories],
                dtype=torch.bool,
                device=device,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if getattr(loss_function, "requires_student_feature", False):
                with ClassEncoderFeatureCapture(model) as capture:
                    prediction = model(waveform) if training else forward_logits(model, waveform)
                student_feature = capture.feature_btc
            else:
                prediction = model(waveform) if training else forward_logits(model, waveform)
                student_feature = None
            if teacher is None:
                loss = loss_function(prediction, target)
                components = {}
            elif student_feature is not None:
                loss, components = loss_function(
                    prediction,
                    target,
                    teacher,
                    info["p_arrival_sample"],
                    student_feature,
                    trace_is_noise,
                )
            else:
                loss, components = loss_function(
                    prediction,
                    target,
                    teacher["p_probability"],
                    info["p_arrival_sample"],
                    trace_is_noise,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            if training:
                loss.backward()
                if DEMO:
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError("Missing or non-finite demo gradient")
                if grad_clip > 0:
                    optimized_parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                    ]
                    torch.nn.utils.clip_grad_norm_(optimized_parameters, grad_clip)
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_samples += batch_size
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + float(
                value.detach().cpu()
            )

    elapsed = time.perf_counter() - started
    result = {
        "loss": total_loss / max(total_samples, 1),
        "seconds": elapsed,
        "samples_per_second": total_samples / max(elapsed, 1e-9),
        "samples": total_samples,
    }
    result.update(
        {f"component_{name}": value / max(total_samples, 1) for name, value in component_totals.items()}
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train the configured LPPN variant.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print and validate the resolved configuration, then exit without writing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg.validate_paths(("train", "validation"))
    require_absolute_file(cfg.TRAIN_HDF5, "Training HDF5")
    require_absolute_file(cfg.TRAIN_CSV, "Training CSV")
    require_absolute_file(cfg.VALIDATION_HDF5, "Validation HDF5")
    require_absolute_file(cfg.VALIDATION_CSV, "Validation CSV")
    resume_path = (
        require_absolute_file(args.resume, "Resume checkpoint")
        if args.resume is not None
        else None
    )
    cfg.print_entry_config("train", checkpoint=resume_path)
    if args.show_config:
        return

    output_dir = cfg.TRAIN_OUTPUT_DIR
    if not output_dir.is_absolute():
        raise ValueError(f"Training output must be absolute: {output_dir}")
    best_path, last_path = cfg.checkpoint_paths(output_dir)
    run_config_path = output_dir / "run_config.json"
    log_path = output_dir / "training_log.csv"
    protected = (best_path, last_path, run_config_path, log_path)
    existing = [path for path in protected if path.exists()]
    if existing and resume_path is None and not (args.allow_overwrite or cfg.ALLOW_OVERWRITE):
        raise FileExistsError(
            "Refusing to overwrite an existing experiment:\n- "
            + "\n- ".join(str(path) for path in existing)
        )
    seed_everything(cfg.SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_dataset = LPPNDataset(
        cfg.TRAIN_HDF5,
        cfg.TRAIN_CSV,
        target_length=cfg.MODEL_INPUT_LENGTH,
        stride=cfg.MODEL_S,
        normalization=cfg.NORMALIZATION,
        amplitude_scale=cfg.AMPLITUDE_SCALE,
        gaussian_noise_std=cfg.GAUSSIAN_NOISE_STD,
        max_samples=args.max_train_samples,
        seed=cfg.SEED,
    )
    validation_dataset = LPPNDataset(
        cfg.VALIDATION_HDF5,
        cfg.VALIDATION_CSV,
        target_length=cfg.MODEL_INPUT_LENGTH,
        stride=cfg.MODEL_S,
        normalization=cfg.NORMALIZATION,
        max_samples=args.max_validation_samples,
        seed=cfg.SEED,
    )
    cache_manifest_sha256 = None
    feature_cache_manifest_sha256 = None
    if cfg.MODEL_VARIANT in cfg.KD_CLASSIFICATION_VARIANTS:
        cache_manifest_sha256 = sha256_file(cfg.TEACHER_CACHE_MANIFEST)
    if cfg.MODEL_VARIANT in cfg.KD_FEATURE_VARIANTS:
        feature_cache_manifest_sha256 = sha256_file(cfg.TEACHER_FEATURE_CACHE_MANIFEST)
    train_dataset = compose_training_dataset(
        train_dataset,
        variant=cfg.MODEL_VARIANT,
        split="train",
        probability_manifest=cfg.TEACHER_CACHE_MANIFEST,
        feature_manifest=cfg.TEACHER_FEATURE_CACHE_MANIFEST,
        probability_protocol_id=cfg.TEACHER_PROTOCOL_ID,
        feature_protocol_id=cfg.TEACHER_FEATURE_PROTOCOL_ID,
        teacher_sha256=cfg.EQT_TEACHER_SHA256,
        verify_sha256=cfg.KD_VERIFY_CACHE_SHA256,
    )
    validation_dataset = compose_training_dataset(
        validation_dataset,
        variant=cfg.MODEL_VARIANT,
        split="validation",
        probability_manifest=cfg.TEACHER_CACHE_MANIFEST,
        feature_manifest=cfg.TEACHER_FEATURE_CACHE_MANIFEST,
        probability_protocol_id=cfg.TEACHER_PROTOCOL_ID,
        feature_protocol_id=cfg.TEACHER_FEATURE_PROTOCOL_ID,
        teacher_sha256=cfg.EQT_TEACHER_SHA256,
        verify_sha256=cfg.KD_VERIFY_CACHE_SHA256,
    )
    loader_options = {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": pin_memory,
        "drop_last": False,
        "worker_init_fn": worker_init_fn if cfg.NUM_WORKERS else None,
        "persistent_workers": cfg.NUM_WORKERS > 0,
    }
    generator = torch.Generator().manual_seed(cfg.SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        **loader_options,
    )

    model = LPPNBase(model_n=cfg.MODEL_N, n_stride=cfg.MODEL_S).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    projection = None
    projection_parameter_count = 0
    if cfg.MODEL_VARIANT in cfg.KD_FEATURE_PROJECTION_VARIANTS:
        projection = TrainingOnlyFeatureProjection().to(device)
        projection_parameter_count = sum(
            parameter.numel() for parameter in projection.parameters()
        )
        projection_loss_arguments = {
            "teacher_min_confidence": float(cfg.KD_TEACHER_MIN_CONFIDENCE),
            "teacher_max_error_samples": (
                float(cfg.KD_TEACHER_MAX_ERROR_SECONDS) * cfg.SAMPLE_RATE_HZ
            ),
            "normalization_epsilon": cfg.KD_FEATURE_NORMALIZATION_EPSILON,
        }
        if cfg.MODEL_VARIANT == "LPPN-KD-CFP":
            loss_function = LPPNClassificationFeatureProjectionDistillationLoss(
                projection,
                lambda_c=float(cfg.KD_LAMBDA_C),
                lambda_fp=float(cfg.KD_LAMBDA_FP),
                stride=cfg.MODEL_S,
                **projection_loss_arguments,
            ).to(device)
        else:
            loss_function = LPPNFeatureProjectionDistillationLoss(
                projection,
                lambda_fp=float(cfg.KD_LAMBDA_FP),
                **projection_loss_arguments,
            ).to(device)
    elif cfg.MODEL_VARIANT in cfg.KD_CLASSIFICATION_VARIANTS:
        loss_function = LPPNKDLoss(
            cfg.MODEL_VARIANT,
            lambda_c=cfg.KD_LAMBDA_C or 0.0,
            teacher_min_confidence=cfg.KD_TEACHER_MIN_CONFIDENCE,
            teacher_max_error_samples=(
                cfg.KD_TEACHER_MAX_ERROR_SECONDS * cfg.SAMPLE_RATE_HZ
            ),
            stride=cfg.MODEL_S,
        ).to(device)
    else:
        loss_function = LPPNLoss().to(device)
    optimizer_parameters = list(model.parameters())
    if projection is not None:
        optimizer_parameters.extend(projection.parameters())
    optimizer = torch.optim.Adam(
        optimizer_parameters, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        old_config = checkpoint.get("config", {})
        cfg.validate_artifact_identity(old_config, "Resume checkpoint")
        cfg.validate_kd_artifact_identity(old_config, "Resume checkpoint")
        if cfg.MODEL_VARIANT in cfg.KD_MODEL_VARIANTS and old_config.get(
            "teacher_cache_manifest_sha256"
        ) != cache_manifest_sha256:
            if cfg.MODEL_VARIANT in cfg.KD_CLASSIFICATION_VARIANTS:
                raise ValueError("Resume checkpoint teacher-cache manifest SHA256 mismatch.")
        if cfg.MODEL_VARIANT in cfg.KD_FEATURE_VARIANTS and old_config.get(
            "teacher_feature_cache_manifest_sha256"
        ) != feature_cache_manifest_sha256:
            raise ValueError("Resume checkpoint feature-cache manifest SHA256 mismatch.")
        model.load_state_dict(checkpoint["model_state"])
        if projection is not None:
            projection_state = checkpoint.get("projection_state")
            if not isinstance(projection_state, dict):
                raise ValueError("KD-FP resume checkpoint is missing projection_state.")
            projection.load_state_dict(projection_state, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint.get("best_validation_loss", float("inf")))

    run_config = cfg.experiment_snapshot()
    run_config.update(
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "model_parameters": parameter_count,
            "student_inference_parameters": parameter_count,
            "training_only_projection_parameters": projection_parameter_count,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "teacher_cache_manifest_sha256": cache_manifest_sha256,
            "teacher_feature_cache_manifest_sha256": feature_cache_manifest_sha256,
        }
    )
    run_config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log_mode = "a" if resume_path is not None else "w"
    with log_path.open(log_mode, encoding="utf-8", newline="") as log_file:
        fields = [
            "epoch",
            "train_loss",
            "validation_loss",
            "train_seconds",
            "validation_seconds",
            "train_samples_per_second",
            "validation_samples_per_second",
            "learning_rate",
        ]
        if cfg.MODEL_VARIANT in cfg.KD_MODEL_VARIANTS:
            fields.extend([
                    "train_supervised_per_sample",
                    "validation_supervised_per_sample",
            ])
        if cfg.MODEL_VARIANT in cfg.KD_CLASSIFICATION_VARIANTS:
            fields.extend(["train_kd_c_raw_per_sample", "validation_kd_c_raw_per_sample"])
        if cfg.MODEL_VARIANT in cfg.KD_FEATURE_PROJECTION_VARIANTS:
            fields.extend(
                [
                    "train_kd_fp_raw_per_sample",
                    "validation_kd_fp_raw_per_sample",
                    "train_active_fp_traces_per_sample",
                    "validation_active_fp_traces_per_sample",
                ]
            )
        writer = csv.DictWriter(log_file, fieldnames=fields)
        if log_mode == "w" or log_path.stat().st_size == 0:
            writer.writeheader()

        print(f"Experiment: {cfg.EXPERIMENT_ID}")
        print(f"Device: {device}")
        print(f"Parameters: {parameter_count:,}")
        if projection is not None:
            print(f"Training-only projection parameters: {projection_parameter_count:,}")
        print(f"Train/validation: {len(train_dataset):,}/{len(validation_dataset):,}")

        for epoch in range(start_epoch, cfg.EPOCHS + 1):
            train_stats = run_epoch(
                model,
                train_loader,
                loss_function,
                device,
                optimizer,
                cfg.GRAD_CLIP,
            )
            with torch.no_grad():
                validation_stats = run_epoch(
                    model, validation_loader, loss_function, device
                )
            log_row = {
                    "epoch": epoch,
                    "train_loss": train_stats["loss"],
                    "validation_loss": validation_stats["loss"],
                    "train_seconds": train_stats["seconds"],
                    "validation_seconds": validation_stats["seconds"],
                    "train_samples_per_second": train_stats["samples_per_second"],
                    "validation_samples_per_second": validation_stats[
                        "samples_per_second"
                    ],
                    "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if cfg.MODEL_VARIANT in cfg.KD_MODEL_VARIANTS:
                log_row.update(
                    {
                    "train_supervised_per_sample": train_stats.get("component_supervised", ""),
                    "validation_supervised_per_sample": validation_stats.get("component_supervised", ""),
                    }
                )
            if cfg.MODEL_VARIANT in cfg.KD_CLASSIFICATION_VARIANTS:
                log_row.update(
                    {
                        "train_kd_c_raw_per_sample": train_stats.get("component_kd_c_raw", ""),
                        "validation_kd_c_raw_per_sample": validation_stats.get("component_kd_c_raw", ""),
                    }
                )
            if cfg.MODEL_VARIANT in cfg.KD_FEATURE_PROJECTION_VARIANTS:
                log_row.update(
                    {
                        "train_kd_fp_raw_per_sample": train_stats.get("component_kd_fp_raw", ""),
                        "validation_kd_fp_raw_per_sample": validation_stats.get("component_kd_fp_raw", ""),
                        "train_active_fp_traces_per_sample": train_stats.get("component_active_fp_traces", ""),
                        "validation_active_fp_traces_per_sample": validation_stats.get("component_active_fp_traces", ""),
                    }
                )
            writer.writerow(log_row)
            log_file.flush()
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_validation_loss": min(
                    best_validation_loss, validation_stats["loss"]
                ),
                "config": run_config,
            }
            if projection is not None:
                checkpoint["projection_state"] = projection.state_dict()
            torch.save(checkpoint, last_path)
            if validation_stats["loss"] < best_validation_loss:
                best_validation_loss = validation_stats["loss"]
                checkpoint["best_validation_loss"] = best_validation_loss
                torch.save(checkpoint, best_path)
            print(
                f"epoch={epoch:03d}/{cfg.EPOCHS} "
                f"train={train_stats['loss']:.6f} "
                f"validation={validation_stats['loss']:.6f}"
            )

    train_dataset.close()
    validation_dataset.close()
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
