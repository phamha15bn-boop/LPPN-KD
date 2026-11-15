# LPPN-KD

This repository contains the EQTransformer-to-LPPN knowledge distillation pipeline for seismic P-wave picking. Four student variants share the same 43,296-parameter deployment network. LPPN-BASE uses supervised training. KD-C transfers P-wave probabilities. KD-FP transfers projected intermediate features. KD-CFP combines both losses.

## Run the STEAD demonstration

The bundled [STEAD sample](data/sample_stead/README.md) has 9 training, 2 validation, and 3 test traces. Use separate Python environments for the TensorFlow teacher and the PyTorch student. See [environment and platform details](docs/REPRODUCIBILITY.md).

From the repository root in PowerShell, set the two Python executable variables and choose an output path that does not exist yet.

```powershell
$OutputRoot = 'outputs/smoke_test'
./scripts/run_full_smoke_test.ps1 -EqtPython $EqtPython -StudentPython $StudentPython -OutputRoot $OutputRoot
```

The runner is configured to train EQTransformer from scratch for one epoch, reload it for validation and test prediction, and generate and verify both teacher caches on the training and validation splits. It then trains each student for one epoch, strictly reloads each student checkpoint, predicts on the test split, scores the picks, and verifies the artifacts. Each stage records its output and exit status under the selected output root. A failed stage stops the run. The historical GPU smoke trained the teacher in a separate invocation and continued through the runner. See the [release checklist](docs/RELEASE_CHECKLIST.md) for the status of the single-invocation check.

The demo uses seed 42 and batch size 3. Its gate accepts the full probability range and arrival-time differences of up to 60 s so that one-epoch teacher predictions exercise the KD losses. Valid manual P remains required and noise is excluded. Demo predictions and scores check program behaviour on 14 traces.

For manual commands and the equivalent Python runner, see [reproducibility](docs/REPRODUCIBILITY.md). For tensor layouts and teacher layers, see [architecture](docs/ARCHITECTURE.md) and [distillation](docs/DISTILLATION.md).

## Run the configured Shanxi experiments

Copy `configs/formal/shanxi.json` to the ignored `local/` directory and select the copy with `LPPN_CONFIG`. Set the six split paths to authorised HDF5 and CSV inputs, and point `LPPN_TEACHER_WEIGHT` to the authorised teacher file with the configured SHA-256. The formal student profile uses batch size 128, 100 epochs and learning rate 0.0001. The **teacher KD gate** uses a minimum P-peak confidence of 0.5 and a maximum teacher-to-manual P error of 0.5 s. **Student evaluation** uses separate P and S detection thresholds of 0.3 and a pick-matching tolerance of 0.5 s. Choose a student with `LPPN_VARIANT` and a seed with `LPPN_SEED`. The [data format](docs/DATA_FORMAT.md) documents the expected files. Shanxi waveforms, split manifests, teacher weights, and caches must be supplied by the researcher.

Run `python train.py`, `python predict.py --split test`, and `python evaluate.py --split test` with the Student interpreter after selecting the experiment. The standard prediction entry accepts an identity-bearing training checkpoint and strictly loads its student state. For pure inference candidate weights, see [checkpoint loading](docs/REPRODUCIBILITY.md).

## Author and Attribution

**Qineng Song**, a master's student at Taiyuan University of Technology, is the sole software author, contributor and maintainer of LPPN-KD.

Qineng Song independently developed the project adaptations, distillation workflow, demonstration, release tooling and documentation. LPPN-KD integrates EQTransformer and the csnbench LPPNM implementation under their respective licences, with credit to the original authors. Cite LPPN-KD using the metadata in the citation file. [Third-party notices](THIRD_PARTY_NOTICES.md) · [CITATION.cff](CITATION.cff)

## Licences

The Student and project code use GPL-3.0. The vendored EQTransformer code retains its own MIT licence. The redistributed STEAD sample is under CC BY 4.0. See [third-party notices](THIRD_PARTY_NOTICES.md) and [data licence](DATA_LICENSE.md). Formal Shanxi release-candidate weights are excluded from version control pending a separate distribution decision.
