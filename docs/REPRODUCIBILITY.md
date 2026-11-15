# Reproducibility

## Environments

Use existing compatible environments if available. The checked Windows GPU environments use these versions.

| Stage | Python | Framework | GPU build/runtime family | Other direct packages |
| --- | --- | --- | --- | --- |
| EQT | 3.8.20 | TensorFlow/Keras 2.10.0 | CUDA 11.2.2 runtime package, cuDNN 8.1.0.77 package | NumPy 1.23.5, h5py 3.7.0, SciPy 1.10.1, ObsPy 1.3.0, Matplotlib 3.5.3, pandas 1.4.4, tqdm 4.67.1 |
| Student | 3.11.16 | PyTorch 2.12.1+cu130 | CUDA 13.0 build, cuDNN DLL 9.20.0.48, runtime reported 92000 | NumPy 2.4.6, h5py 3.16.0 |

The environment YAML files can be used separately from the repository root. The EQT recipe targets native Windows with an NVIDIA GPU and a compatible host driver. Its runtime packages follow the TensorFlow 2.10 setup. The Student recipe installs the Python dependencies. To select the GPU build used in the demo, install the CUDA 13.0 PyTorch wheel in the Student environment. These commands are provided for a fresh installation and were checked against the installed package metadata. A new installation and other operating systems remain untested. [TensorFlow Windows setup](https://www.tensorflow.org/install/pip#windows-native) · [PyTorch CUDA 13.0 index](https://download.pytorch.org/whl/cu130/torch/)

```powershell
conda env create -f environment-eqt.yml
conda env create -f environment-student.yml
conda run -n lppn-kd-student python -m pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

## One command

Set `$EqtPython` and `$StudentPython` to your respective Python executables. Use a fresh PowerShell session without inherited `LPPN_CONFIG` or other `LPPN_*` experiment overrides. From the repository root:

```powershell
$OutputRoot = 'outputs/smoke_test'
./scripts/run_full_smoke_test.ps1 -EqtPython $EqtPython -StudentPython $StudentPython -OutputRoot $OutputRoot
```

Use a new output path for each run, for example `outputs/smoke_test_repeat`. The runner rejects a path that already exists, including an empty directory. On a host without a TensorFlow GPU, the explicit `-AllowTeacherCpu` option permits CPU cache generation. The equivalent Python runner is `python -m scripts.run_smoke --eqt-python <path> --student-python <path> --output-root <new-path>`. Each stage logs separately and reports its exit code. The explicit continuation option `-ReuseTrainedDemoTeacher` requires the expected one-epoch checkpoint, a matching provenance hash, and an output root without downstream demo artifacts. The historical run used this continuation after training the teacher separately.

## Individual entry points

These are PowerShell examples with the demo selected. Keep the same `LPPN_OUTPUT_ROOT` for every command. Use separate output roots for different smoke runs.

```powershell
$env:LPPN_CONFIG = 'configs/smoke_test/stead.json'
$env:LPPN_OUTPUT_ROOT = 'outputs/manual_demo'
& $EqtPython -m teacher.cli train
& $EqtPython -m teacher.cli predict --split validation
& $EqtPython -m teacher.cli predict --split test
& $EqtPython -m teacher.cache --split train --batch-size 3 --shard-size 3
& $EqtPython -m teacher.cache --split validation --batch-size 3 --shard-size 3 --resume
& $EqtPython -m teacher.feature_cache --split train --batch-size 3 --shard-size 3
& $EqtPython -m teacher.feature_cache --split validation --batch-size 3 --shard-size 3 --resume
& $StudentPython -m teacher.cache --split train --batch-size 3 --verify-only
& $StudentPython -m teacher.cache --split validation --batch-size 3 --verify-only
& $StudentPython -m teacher.feature_cache --split train --verify-only
& $StudentPython -m teacher.feature_cache --split validation --verify-only
foreach ($Variant in @('LPPN-BASE', 'LPPN-KD-C', 'LPPN-KD-FP', 'LPPN-KD-CFP')) {
    $env:LPPN_VARIANT = $Variant
    & $StudentPython train.py
    & $StudentPython predict.py --split test
    & $StudentPython evaluate.py --split test
}
& $StudentPython -m scripts.verify_smoke
```

On Windows, individual invocations of the Teacher Python need Conda activation or the environment's `Library/bin` on the process PATH. The runner sets that search path for each child process.

## Formal Shanxi profile

Paths in `configs/formal/shanxi.json` resolve relative to this repository. Copy the profile to an ignored local directory and substitute the authorised split paths where necessary. Select it using `LPPN_CONFIG` and set `LPPN_VARIANT` and `LPPN_SEED` before running. `configs/formal/shanxi_runs.json` records the historical seeds and safe settings. The fixed teacher hash is in the profile. A trainer run on authorised splits can also generate a new EQT checkpoint via `python -m teacher.cli train --epochs 100 --batch-size 64`. To use such a separate teacher in an experiment, set its checkpoint path, expected SHA, identity and distinct protocol IDs in a separate profile. This produces a new run rather than certifying a reconstruction of the historical teacher.

Use the cache commands above with the formal profile and authorised data paths. Generate both probability splits before generating features. Use identical batch and shard settings when resuming a cache and verify both splits before Student training. The teacher KD gate compares teacher P peaks with manual P at confidence 0.5 and time error 0.5 s. Student evaluation uses detection thresholds of 0.3 for P and S and a separate P/S matching tolerance of 0.5 s. Keep an immutable waveform snapshot and use the exact teacher identity for each run.

## Checkpoint loading

`predict.py` validates experiment metadata and strictly loads `model_state` from a training checkpoint. Supply only training checkpoints from trusted sources to `predict.py` or `train.py --resume` because these entry points use PyTorch pickle loading. `--split validation` and `--split test` write separately. The local formal candidates are pure student state dictionaries under `checkpoints/release_candidates/` and can be loaded as follows after checking the matching hash in their manifest.

```python
import torch
from lppn.model import LPPNBase

model = LPPNBase(model_n=8, n_stride=8)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
model.eval()
```

Confirm the applicable weight and data distribution rights before redistributing formal Shanxi weights. The formal checkpoint manifest is kept with those local candidates.
