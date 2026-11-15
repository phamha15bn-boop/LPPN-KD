# Copyright (c) 2026 Qineng Song

"""Run the STEAD Teacher-to-Student demo with two existing Python environments."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from release_config import ROOT, sha256

VARIANTS = ("LPPN-BASE", "LPPN-KD-C", "LPPN-KD-FP", "LPPN-KD-CFP")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eqt-python", required=True, type=Path)
    parser.add_argument("--student-python", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/smoke_test"))
    parser.add_argument("--allow-teacher-cpu", action="store_true")
    parser.add_argument("--reuse-trained-demo-teacher", action="store_true",
                        help="Use a previously trained, identity-checked teacher in this output root.")
    args = parser.parse_args()

    output = (ROOT / args.output_root).resolve()
    eqt, student = args.eqt_python.resolve(), args.student_python.resolve()
    for executable in (eqt, student):
        if not executable.is_file():
            parser.error(f"Python executable is missing: {executable}")
    if output.exists() and not args.reuse_trained_demo_teacher:
        parser.error(f"Choose a new output root: {output}")
    checkpoint = output / "eqt" / "demo.weights.h5"
    if args.reuse_trained_demo_teacher:
        provenance = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        identity = tuple(provenance[k] for k in
                         ("dataset", "trained_from_scratch", "seed", "epochs", "batch_size"))
        if identity != ("stead_demo", True, 42, 1, 3) or sha256(checkpoint) != provenance["sha256"]:
            parser.error("Cannot reuse teacher: checkpoint provenance/identity mismatch.")
        if {path.name for path in output.iterdir()} - {"eqt", "logs"} or any(
                (output / "eqt" / split).exists() for split in ("validation", "test")):
            parser.error("Teacher reuse needs an output root without downstream demo results.")

    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=args.reuse_trained_demo_teacher)
    base = dict(os.environ)
    for name in ("LPPN_VARIANT", "LPPN_SEED", "LPPN_TEACHER_WEIGHT"):
        base.pop(name, None)
    base.update(
        LPPN_CONFIG="configs/smoke_test/stead.json", LPPN_OUTPUT_ROOT=str(output),
        PYTHONDONTWRITEBYTECODE="1", TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2", TF_CPP_MIN_LOG_LEVEL="2", PYTHONIOENCODING="utf-8",
    )
    cpu = ["--allow-cpu"] if args.allow_teacher_cpu else []
    stages = []

    def run(name, executable, args, variant=None):
        env = base.copy()
        prefix = executable.parent
        # Windows Conda DLL search path for explicit interpreter invocation.
        if os.name == "nt":
            env["PATH"] = os.pathsep.join(
                map(str, [prefix, prefix / "Library" / "bin", prefix / "Scripts", env["PATH"]])
            )
        if variant:
            env["LPPN_VARIANT"] = variant
        command = [str(executable), "-u", *args]
        record = {"stage": name, "args": args, "variant": variant,
                  "log": f"logs/{name}.log"}
        print(f"Running {name}", flush=True)
        log = output / record["log"]
        with log.open("xb") as stream:
            status = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT).returncode
        record["exit_code"] = status
        stages.append(record)
        (output / "stages.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
        if status:
            for line in log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]:
                print(line)
            raise SystemExit(f"{name} failed (exit {status}). See {log}.")

    if args.reuse_trained_demo_teacher:
        print("Using the existing, identity-checked from-scratch STEAD teacher.")
        stages.append({"stage": "eqt-train", "teacher_sha256": sha256(checkpoint),
                       "existing_teacher_verified": True, "log": "logs/eqt-train.log",
                       "exit_code": 0})
    else:
        run("eqt-train", eqt, ["-m", "teacher.cli", "train"])
    for split in ("validation", "test"):
        run(f"eqt-predict-{split}", eqt, ["-m", "teacher.cli", "predict", "--split", split, *cpu])
    for kind, module in (("probability", "teacher.cache"), ("feature", "teacher.feature_cache")):
        for split in ("train", "validation"):
            resume = ["--resume"] if split == "validation" else []
            run(f"{kind}-cache-{split}", eqt,
                ["-m", module, "--split", split, "--batch-size", "3", "--shard-size", "3",
                 *resume, *cpu])
        for split in ("train", "validation"):
            run(f"{kind}-verify-{split}", student,
                ["-m", module, "--split", split, "--batch-size", "3", "--verify-only"])
    for variant in VARIANTS:
        for action in ("train", "predict", "evaluate"):
            run(f"{variant.lower()}-{action}", student, [f"{action}.py"], variant)
    run("verify-demo", student, ["-m", "scripts.verify_smoke"])
    print(f"Demo verification passed. Outputs: {output}")


if __name__ == "__main__":
    main()
