# Copyright (c) 2026 Qineng Song

"""Extract inference state_dicts from existing trusted Shanxi runs."""

import argparse
import json
from pathlib import Path

import torch

from lppn.model import LPPNBase
from release_config import ROOT, sha256

VARIANTS = ("LPPN-BASE", "LPPN-KD-C", "LPPN-KD-FP", "LPPN-KD-CFP")
SEEDS = (42, 43, 44)
FORMAL_KEYS = ("dataset_name", "model_variant", "seed", "batch_size", "epochs",
               "learning_rate", "weight_decay", "num_workers", "grad_clip", "normalization",
               "amplitude_scale", "gaussian_noise_std", "model_n", "model_s",
               "threshold_p", "threshold_s", "tolerance_seconds",
               "student_inference_parameters", "training_only_projection_parameters")
KD_KEYS = ("lambda_c", "lambda_fp", "teacher_min_confidence", "teacher_max_error_seconds",
           "teacher_protocol_id", "teacher_weight_sha256", "feature_protocol_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shanxi-results", required=True, type=Path)
    args = parser.parse_args()
    dest = ROOT / "checkpoints" / "release_candidates"
    manifest_path = dest / "manifest.json"
    if any(dest.iterdir()):
        raise FileExistsError(f"Candidate directory is not empty: {dest}")
    records = []
    torch.set_num_threads(1)
    model = LPPNBase()
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 43296
    formal = []
    for variant in VARIANTS:
        for seed in SEEDS:
            experiment = f"shanxi_{variant}_seed{seed}"
            source = args.shanxi_results / variant / f"seed{seed}" / "training" / f"{experiment}_best.pt"
            before = sha256(source)
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            cfg = checkpoint["config"]
            identity = (cfg["dataset_name"], cfg["model_variant"], cfg["seed"])
            if identity != ("shanxi", variant, seed):
                raise ValueError(f"Checkpoint identity mismatch: {experiment}")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            target = dest / f"{experiment}_student.pt"
            torch.save(model.state_dict(), target)
            state = torch.load(target, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            if sha256(source) != before:
                raise RuntimeError(f"Source changed while reading {experiment}")
            settings = {key: cfg[key] for key in FORMAL_KEYS if key in cfg}
            settings["kd"] = {key: cfg.get("kd", {}).get(key) for key in KD_KEYS
                              if key in cfg.get("kd", {})}
            if (settings["batch_size"], settings["epochs"], settings["learning_rate"]) != (128, 100, 0.0001):
                raise ValueError(f"Formal settings differ: {experiment}")
            formal.append(settings)
            records.append({
                "dataset": "shanxi", "variant": variant, "seed": seed,
                "parameters": parameters, "checkpoint_epoch": checkpoint["epoch"],
                "original_name": source.name, "original_bytes": source.stat().st_size,
                "original_sha256": before, "candidate_name": target.name,
                "candidate_bytes": target.stat().st_size, "candidate_sha256": sha256(target),
                "settings": settings, "strict_reload": True,
            })
            print(f"Extracted and strictly reloaded {target.name}")
    data = {"status": "LOCAL_CANDIDATES_RELEASE_PERMISSION_PENDING",
            "format": "PyTorch state_dict containing only Student tensors",
            "records": records}
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    with (ROOT / "configs/formal/shanxi_runs.json").open("x", encoding="utf-8") as stream:
        json.dump(formal, stream, indent=2)


if __name__ == "__main__":
    main()
