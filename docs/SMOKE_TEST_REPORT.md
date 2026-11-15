# STEAD end-to-end smoke test

The completed run used the bundled 14-trace STEAD subset, seed 42, and an NVIDIA GeForce RTX 3080. The one-epoch screening run checks execution and artifact integrity. Its scores have no statistical role in the Shanxi experiments.

## Acceptance results

| Check | Status | Evidence |
| --- | --- | --- |
| STEAD HDF5 | PASS | 14 finite float32 waveform arrays, each `[6000,3]` |
| STEAD CSV | PASS | Train/validation/test 9/2/3; all waveform keys, labels, and split rows match; 12 distinct earthquake source IDs |
| Noise labels | PASS | Both real noise traces have P/S −1 and all-zero EQT labels and Student targets; five absent-pick encodings accepted; earthquake without P rejected |
| EQT train | PASS | From-scratch one epoch, batch 3; validation retained both traces in a partial batch |
| EQT checkpoint | PASS | HDF5 weight file saved and hash checked, then loaded by the reconstructed deterministic graph |
| EQT prediction | PASS | Validation and test emit finite float32 D/P/S tensors `[2,6000,1]` and `[3,6000,1]`; noise metadata retained |
| P probability | PASS | `picker_P` produces finite float32 `[6000]` per trace |
| P feature | PASS | `picker_P.decoder.stage4.post_conv`, Keras layer `conv1d_33`, produces finite float32 `[750,32]` per trace |
| Probability cache | PASS | 9 train and 2 validation traces, 3/1 shards; strictly verified |
| Feature cache | PASS | 9 train and 2 validation traces, 3/1 shards; strictly verified against P-cache trace order and manifest |
| BASE | PASS | One epoch, strict reload, 3 test predictions, evaluation JSON/CSV |
| KD-C | PASS | Same path with active, finite probability loss and finite training gradients |
| KD-FP | PASS | Same path with 32-channel captured feature, 1,024-parameter projection, active finite feature loss and finite training gradients |
| KD-CFP | PASS | Both active finite losses, finite joint gradients, strict reload, prediction and evaluation |
| Student checkpoints | PASS | All four best and last checkpoint states strictly reload as the common Student |
| README | PASS | Running commands, dependencies, input format, and checkpoint formats documented |
| Presence of source and data notices | PASS | Retained GPL and MIT texts, upstream attribution and CC BY 4.0 data notice. Exact source revisions and project authorship are tracked in the release checklist. |
| Formal Shanxi weight redistribution | BLOCKED | Distribution permission for derived weights is unverified |
| Fresh environment provisioning and full Shanxi rerun | NOT TESTED | Test scope used existing Windows environments and the 14-trace demo |

## Data and execution

The extraction script used seed 42, rejected incomplete earthquake picks, preserved source split membership, and checked each selected name and pick against the original HDF5. Train includes 8 earthquakes and 1 noise trace, validation 2 earthquakes, and test 2 earthquakes and 1 noise trace. [STEAD attribution](../data/sample_stead/DATA_LICENSE.md) · [Selection and file hashes](../data/sample_stead/sample_manifest.json) · [Exact selected rows](../data/sample_stead/sample_manifest.csv).

All 14 bundled waveform arrays were independently compared sample-for-sample with their source arrays and checked against the selection manifest's waveform hashes.

The teacher was built with the original model factory and trained on the fixed split via `python -m teacher.cli train`. The final epoch recorded train loss **0.678203** and validation loss **0.645168**. The release's EQT generator handles the partial validation batch, rejects missing named waveforms and diagnoses invalid earthquake P labels. Model weights and network code came from the release pipeline and the upstream-derived code, respectively. The training topology and loss definitions were retained.

The inference adapter checked the two forced SpatialDropout source sites and constructed 14 inference-only SpatialDropout instances. Strict weight reload and fixed-signature `tf.function` inference ran on `GPU:0`. Prediction and caching used per-component standardisation. The cache fixtures include only train and validation. The verifier checked SHA-256, CSV row index, trace name, shapes, dtype, finite values and original P/S arrival metadata. The feature cache verifies the probability manifest and split ordering before import by a student.

- Probability manifest SHA-256 `4762EF1B86ABA9DF0F89300F69B65AB09CB1089CF7B93098189F4944C19C38ED`
- Feature manifest SHA-256 `8D2932AB697C859FE2037492568AB8ADF503E013651F89997A0121F0BF2CAB71`
- Teacher source tree SHA-256 `9A290A59DF188E26173FF14612CC2537D0ED7074437CC51F12FFA40A4965C254`
- Sample HDF5 SHA-256 `738636F2DA415F841C2AB0EB1CB48736179918F31087E33E4155F4F5EE788A1C`

Both cache families used batch size 3 and shard size 3. Their demo protocol IDs are `EQT_STEAD_DEMO_P_DETERMINISTIC_OFF_V1` and `EQT_STEAD_DEMO_PDEC4_F32_DETERMINISTIC_OFF_V1`. Validation append used the same manifest identity.

## Student loss and checkpoint records

Each student trained on the same nine traces for one epoch, with validation on the same two traces. The demo gate thresholds were 0 confidence and 60 s deviation. The FP gate admitted eight of nine train traces and both validation traces. Noise remained in supervised learning. For FP and CFP, the captured student feature had 768 cells × 32 channels and the first 750 were matched. The demo checks every optimised parameter's gradient, including the projection. The prediction entry strictly loads only the 43,296-parameter Student state for inference.

| Variant | Train total / trace | Validation total / trace | Train raw C / trace | Train raw FP / trace | Projection parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| LPPN-BASE | 1668.121745 | 1426.252075 | — | — | 0 |
| LPPN-KD-C | 2062.070855 | 1826.051025 | 788.181885 | — | 0 |
| LPPN-KD-FP | 9474.140191 | 14840.133789 | — | 0.021380251 | 1024 |
| LPPN-KD-CFP | 9868.955078 | 15249.618164 | 788.900160 | 0.021381410 | 1024 |

Teacher `demo.weights.h5` SHA-256 `453BEAA396A765FD654425858027AFA82E4E0367C25711E38291DDE3CC084EC9`.

| Student demo model | Best SHA-256 | Last SHA-256 |
| --- | --- | --- |
| LPPN-BASE | `64FE1B435A749650CB9964E0C615DE010CFD15444D9153E9786AB54DF5C4A99E` | `EC1DCC74AE8507BB5FE63301F933D4A6897CF5E8EED563D6F82C0B8FB6CFA7BA` |
| LPPN-KD-C | `331CFB5FC9D53E86ED112C507A0F9BC8B20B6A0783E00D71F2E1BBC8AA71B149` | `41BDFAE15C1FED4EADE848EB2F61B320976CC27E8AF36EB66D6E52F4BB83F867` |
| LPPN-KD-FP | `5CDCED8144ED7EAB47C0F5F519290DDC238DF8BAFD153B2AD0D533FF52003734` | `0F7724E822E91D50DEE3449A66AFDAF55BFCCE4D21EAD37938E67B3FEC324D7C` |
| LPPN-KD-CFP | `6425B02EDC5550EA6C5A74DDCD7E45718C82D5CD60947E0571693D452EAE424C` | `5615679CD987D4952A92BEF1153AD094997E80DFD2761178BCA079A5AD9E9B4D` |

The run recorded 24 successful stages in `outputs/smoke_test/stages.json`. `smoke_manifest.json` includes the machine-readable checks and losses. Per-stage logs are under `logs/`. The EqT epoch was run first using the same entry point as the runner. The PowerShell runner then resumed with `-ReuseTrainedDemoTeacher` after verifying that checkpoint's provenance and hash.

## Environment and recovery

TensorFlow/Keras 2.10.0 on Python 3.8.20 used the CUDA 11.2 build with cuDNN 8.1. PyTorch 2.12.1+cu130 on Python 3.11.16 used the CUDA 13.0 build. The recorded cuDNN value was 92000 and the installed DLL version is 9.20.0.48. Both used the RTX 3080. [Exact direct dependencies and commands](REPRODUCIBILITY.md).

Direct TensorFlow invocation initially missed the Conda runtime DLLs. Targeting the existing environment's `Library/bin` through the child process PATH restored GPU detection. The first validation prediction attempt reached the GPU forward pass but failed while writing runtime metadata because the adapted file had dropped the `sys` import. After restoring the import and clearing only the two partial demo output files, the runner continued from the same verified trained teacher. The original failed log is retained locally. The formal checkpoint export initially rejected the legacy `TorchVersion` metadata class. The final export used weights-only loading with that single class explicitly allowlisted.

## Formal Shanxi candidates

The local-only candidate manifest records 12 original-to-exported identities and hashes. All are 43,296-parameter pure Student state dictionaries. The manifest holds the actual best epoch and settings for each run. The candidate files range from 268,219 to 268,683 bytes. Formal configurations keep batch 128, 100 epochs, learning rate 0.0001, thresholds 0.3/0.3 and matching tolerance 0.5 s. The demo settings are stored separately.

## Initial candidate source inventory

This table records the initial smoke snapshot. Use the [current release file manifest](RELEASE_FILE_MANIFEST.csv) for the exact allowlist and exclusions.

| Location | Files |
| --- | --- |
| Repository root | `.gitignore`, `CITATION.cff`, `config.py`, `DATA_LICENSE.md`, `environment-eqt.yml`, `environment-student.yml`, `evaluate.py`, `LICENSE`, `predict.py`, `README.md`, `release_config.py`, `requirements-eqt.txt`, `requirements-student.txt`, `THIRD_PARTY_NOTICES.md`, `train.py` |
| `configs/` | `formal/shanxi_runs.json`, `formal/shanxi.json`, `smoke_test/stead.json` |
| `data/` | `sample_stead/DATA_LICENSE.md`, `sample_stead/README.md`, `sample_stead/sample_manifest.csv`, `sample_stead/sample_manifest.json`, `sample_stead/sample.csv`, `sample_stead/sample.hdf5`, `sample_stead/test.csv`, `sample_stead/train.csv`, `sample_stead/validation.csv` |
| `docs/` | `ARCHITECTURE.md`, `DATA_FORMAT.md`, `DISTILLATION.md`, `REPRODUCIBILITY.md`, `SMOKE_TEST_REPORT.md` |
| `lppn/` | `__init__.py`, `data.py`, `dataset_composition.py`, `feature_cache.py`, `feature_hooks.py`, `kd.py`, `kdcfp.py`, `kdfp.py`, `loss.py`, `metrics.py`, `model.py` |
| `scripts/` | `__init__.py`, `export_student_candidates.py`, `extract_stead_sample.py`, `run_full_smoke_test.ps1`, `run_smoke.py`, `verify_smoke.py` |
| `teacher/` | `__init__.py`, `cache.py`, `cli.py`, `feature_cache.py`, `feature_inference.py`, `inference.py`, `protocol.py`, `eqtransformer/LICENSE`, `eqtransformer/EQTransformer/__init__.py`, `eqtransformer/EQTransformer/core/__init__.py`, `eqtransformer/EQTransformer/core/EqT_utils.py`, `eqtransformer/EQTransformer/core/trainer.py` |

Local-only outputs are `outputs/` (including logs, demo checkpoints and caches) and `checkpoints/release_candidates/`. The `.gitignore` keeps these and local data/configuration out of the candidate source set.

## Before release at the time of the smoke runs

Confirm project maintainer authorship, the precise csnbench snapshot and the distribution rights for any formal weights before publishing them. Keep formal Shanxi waveform data, split records, caches, checkpoints and internal logs local. The tested smoke outputs are machine-local evidence; generate them in the reader's checkout with the included command. No external baseline or benchmark is part of this candidate.

## Task 54 fresh one-command GPU run

**PASS** The documented PowerShell runner completed all 24 stages in a single invocation on the RTX 3080. The output root `outputs/release_audit_fresh/` was absent at launch. `stages.json` records `python -m teacher.cli train` as the first stage with exit code 0. The teacher started from random initialization and used the 14-trace STEAD demo, seed 42, batch size 3 and one epoch, with 9/2/3 train/validation/test traces. Each Student trained for one epoch. The run used the existing TensorFlow 2.10.0 and PyTorch 2.12.1+cu130 environments.

The new `eqt/demo.weights.h5` was created at 2026-11-14 23:35:42.531559 UTC. SHA-256 `384F36ABB38A0EC9C78782D9B02A82E627D5BD16C1599E359626BAB404283392`. The provenance file records `trained_from_scratch: true`. Teacher train and validation losses were 0.678203 and 0.645162. Validation and test prediction strictly reloaded the same weights and returned finite D/P/S arrays of shape `[2,6000,1]` and `[3,6000,1]` per output.

| Check | Status | Result |
| --- | --- | --- |
| GPU and output root | PASS | Both frameworks detected the RTX 3080. The code used the new output root and CUDA devices with neither teacher reuse nor the CPU fallback flag enabled. |
| Probability cache | PASS | 9 training and 2 validation traces, 3/1 shards, finite float32 P curves `[6000]` per trace. |
| Feature cache | PASS | Same trace identities and counts, 3/1 shards, finite float32 `[750,32]` per trace. Both caches match the new teacher SHA-256. |
| Teacher feature recheck | PASS | `picker_P.decoder.stage4.post_conv` (`conv1d_33`), P head `[1,6000,1]`, P curves `[1,6000]`, features `[1,750,32]`, all finite on `GPU:0`. |
| Student feature recheck | PASS | `class_encoder` `[1,32,1,768]` → `[1,768,32]`. A hook on the projection during the actual CFP loss call verified equality with the first 750 feature positions and output `[1,750,32]`. |
| Student deployment and reload | PASS | Four CUDA Students with 43,296 parameters each completed train, strict reload, test prediction and evaluation. FP and CFP keep 1,024 projection parameters separately in training checkpoints. The deployed Student state excludes projection. |
| Gradients and KD losses | PASS | The demo gradient guard passed in each training stage. All required raw KD components are finite and nonzero on train and validation. |
| Output integrity | PASS | 24 stage logs, `smoke_manifest.json`, 9 checkpoint hashes and the expected eight cache shards verified. The run used only bundled STEAD inputs. The new binaries match the expected artifact list and none matched any audited formal weight hash. |
| Historical preservation | PASS | 141 protected path hashes matched, including 82 historical demo files and 13 formal candidate artifacts. |
| Clean installs and other platforms | NOT TESTED | The run used the two existing Windows environments. |
| Publication and formal Shanxi asset distribution | BLOCKED | Source provenance, authorship and distribution clearance remain open. |

| Student variant | Train total / trace | Validation total / trace | Train raw C / trace | Train raw FP / trace |
| --- | ---: | ---: | ---: | ---: |
| LPPN-BASE | 1668.116645 | 1426.255005 | — | — |
| LPPN-KD-C | 2062.052246 | 1826.107910 | 788.218777 | — |
| LPPN-KD-FP | 9474.195095 | 14839.366211 | — | 0.021380520 |
| LPPN-KD-CFP | 9869.133464 | 15248.667969 | 788.991184 | 0.021381740 |

| Student variant | Best checkpoint SHA-256 | Last checkpoint SHA-256 |
| --- | --- | --- |
| LPPN-BASE | `41DB0B7325BF7F667AE09D1B01483E53F5A3198105D12C95545F95B71C4BF554` | `E4D4C74864BB27DC348D4FEF3A6B5FB202F53BFF25AE0FC89E19BDFC4DA5380C` |
| LPPN-KD-C | `3CCCE2CF0C759A6A376A1995A8D70418BE088BB7723295EE88DA041469C8B4FE` | `FAA064B1FB005B2BE7A385FE1F4210980994752A1F59342602071DEA758ED76C` |
| LPPN-KD-FP | `3433BFABA2F24E7A7616E85F5BFC43376966C17A68D2DA1CA4E5FAAAE761B952` | `C194D102A56F738E1716B6ACF2FAE82CD0FF11103911EEACAD7527CB4E7849CA` |
| LPPN-KD-CFP | `E921F3EF4070FE1336A2E74936D5A3760C44D7516145FB220F0143CA392BFD3A` | `28631FF5051D4CA55E6B14F59B1FC80AC01E2E9328BA9E79A3A2EB9AB7B235AB` |

The fresh probability and feature manifest SHA-256 values are `A7DCC131C30FC492741A17D4B38D593039A13F3655BB55C30D368D3883CD93B8` and `EE00453768F42B43BB90F2FEC31C06EA4A2B1F730FEE4AEB18186726F6CA57A8`. The local run also holds `task54_teacher_hook.log` and `task54_student_hook.log` for the two direct GPU probes. Demo losses serve execution checks and are outside accuracy and benchmark claims.

## Current attribution and release status

Task 55 pinned the csnbench LPPNM source file. Task 56 confirmed Qineng Song as the sole LPPN-KD software author and maintainer, added the dated GPL-3.0 modification notices, preserved the EQTransformer MIT licence and supplied the complete MIT notices for the three Keras helpers and the BMC peak detector. The historical rows above record the status at the time of each run. [Third-party notices](../THIRD_PARTY_NOTICES.md)

Task 58 verified the release allowlist and the exclusion of the local audit records. The exact EQTransformer and Keras integration revisions and Taiyuan University of Technology's ownership and release authority still require confirmation. Formal Shanxi waveforms, split files and weights remain outside the public distribution scope. Task 58 performed static checks and retained the Task 54 run as historical evidence. [Current checklist](RELEASE_CHECKLIST.md)
