# LPPN-KD final release audit

**Release status remains BLOCKED.** Task 54 passed the from-scratch RTX 3080 smoke and both runtime hook checks. Task 56 confirmed Qineng Song as the sole software author and maintainer and repaired the third-party notices. Task 58 repaired the file manifest and repository exclusions. Exact EQTransformer and Keras source revisions, institutional IP clearance and formal Shanxi distribution permission remain open.

[Release checklist](RELEASE_CHECKLIST.md) · [File manifest](RELEASE_FILE_MANIFEST.csv) · [Historical smoke evidence](SMOKE_TEST_REPORT.md)

## 1. Method and implementation

| Check | Status | Source and evidence |
| --- | --- | --- |
| Teacher graph and weight load | PASS | `teacher/eqtransformer/EQTransformer/core/trainer.py:396` calls `cred2` with input `[B,6000,3]`, 5 residual blocks and 2 BiLSTM blocks. `teacher/inference.py:73–164` verifies SHA-256, uses default topological HDF5 loading, and checks 373,495 parameters and three outputs with shape `[B,6000,1]`. |
| Teacher P and FP extraction | PASS | `EqT_utils.py:2335–2349,2492–2505` exposes `picker_P` and the decoder convolution following the fourth upsampling/cropping stage. `teacher/feature_inference.py:81–113,124–170` finds the unique `[B,750,32]` Conv1D descended from `attentionP` and upstream of `picker_P`, then returns P as `[B,6000]` and the feature as `[B,750,32]`. The historical manifest confirms semantic ID `picker_P.decoder.stage4.post_conv` and the Keras-generated name `conv1d_33`. |
| Teacher execution mode | PASS | The adapter substitutes inference-only SpatialDropout while building the graph and restores the original constructor in `finally`. The resulting models call `training=False` inside `tf.function`. Cache generation uses no gradient tape and converts the returned tensors to NumPy. The train CLI calls the original factory with the original dropout behaviour. |
| Cache identity and alignment | PASS | Cache generators convert input to float32 `[6000,3]` ENZ data, replace non-finite values with zero before per-component `(x−mean)/(std+1e-6)`, and record teacher weight/protocol and source/adapter identity. Manifests bind split CSV hashes, shard hashes, row indices and trace names. The feature generator checks probability-cache identity and ordering. Student wrappers join by trace name and validate the relevant cache hashes and identity. The FP wrapper also compares stored and current manual P/S samples. |
| Waveform content identity on general cache reuse | FAIL | The source HDF5 is recorded by path rather than by content hash. A same-path replacement with the same trace names, CSV and pick labels can pass the current cache validation. Documented immutable input snapshots and fresh cache generation after changes. |
| Student feature source and alignment | PASS | `lppn/model.py:155–178` uses the `class_encoder` output for classification. `lppn/feature_hooks.py` captures `[B,32,1,768]` once per forward without detaching and transforms it to `[B,768,32]`. The loss slices the first 750 cells before projection. Historical FP/CFP losses ran with the runtime shape assertions enabled. |
| Projection and deployment | PASS | `lppn/kdfp.py:16–32` defines a bias-free `Linear(32,32)` with identity initialization, 1,024 parameters. `train.py:271–317` adds those parameters to Adam for FP/CFP. Training checkpoints store projection state separately. `predict.py:32–42` constructs the 43,296-parameter Student, loads only `model_state` with strict state-key checking, and runs in eval/no-grad. The 12 exported state dictionaries are local-only Student artifacts. |
| Fresh runtime hook recheck | PASS | Task 54 confirmed the new Teacher P shape `[1,6000]` and `conv1d_33` feature `[1,750,32]` on the RTX 3080. With the new CFP checkpoint, `class_encoder` produced `[1,32,1,768]` and `[1,768,32]` after transformation. The actual CFP loss projected the first 750 cells to `[1,750,32]`. Projection state was separate from the 43,296-parameter Student. |

## 2. Targets, objectives and frozen settings

`lppn/data.py` maps manual phase picks to stride-8 cells and offsets. Missing S creates no S target. EQT and Student require valid P for earthquake traces, and explicitly labelled noise lacks both manual picks and uses supervised background targets. Distillation eligibility requires non-noise, a manual P in the original 6,000 samples, sufficient teacher peak confidence and teacher-to-manual P error within the time threshold. It filters the auxiliary terms while preserving the manual targets.

| Variant | Status | Actual objective |
| --- | --- | --- |
| BASE | PASS | `lppn/loss.py` sums seven-class cross-entropy over non-ignored cells (ignore index −1), then adds the sum of squared offset errors on manual phase cells divided by 10. Uses raw logits for supervision, including validation through `forward_logits`. |
| KD-C | PASS | BASE + `0.5 × C`. `lppn/kd.py` max-pools consecutive groups of 8 in the teacher P curve to 750 targets, pads to the 768-cell Student axis and masks the last 18 cells. The P-class softmax and the pooled teacher probabilities enter BCE with sum reduction over eligible real cells. The unpooled P curve supplies the gate peak. |
| KD-FP | PASS | BASE + `365000 × FP`. `lppn/kdfp.py` projects the first 750 Student time cells, L2-normalizes each 32-channel Student and detached Teacher vector with epsilon 1e-6 and computes elementwise mean squared difference over eligible traces, cells and channels. Gradients flow through the Student hook and projection. |
| KD-CFP | PASS | BASE + `0.5 × C + 365000 × FP`. `lppn/kdcfp.py` uses distinct cached peak sources for C and FP, even when both use the same gate thresholds. Ineligible batches give differentiable zero auxiliary terms. |

Both KD terms exclude padded cells. The legacy supervised loss follows the target arrays on the full axis. `predict.py` keeps its full-axis phase argmax and `cell × 8 + offset` position. `evaluate.py` and `lppn/metrics.py` apply phase-specific score thresholds and time matching. If a true and a detected pick are outside the time tolerance, evaluation counts both a false positive and a false negative.

| Setting | STEAD demo | Formal Shanxi |
| --- | ---: | ---: |
| Student train batch size | 3 | 128 |
| Student epochs | 1 | 100 |
| Student learning rate | 0.0001 | 0.0001 |
| Seeds | 42 | 42, 43, 44 |
| Lambda C when enabled | 0.5 | 0.5 |
| Lambda FP when enabled | 365000 | 365000 |
| **Teacher KD** minimum P-peak confidence | 0.0 | 0.5 |
| **Teacher KD** maximum teacher-to-manual P error | 60 s | 0.5 s |
| **Student evaluation** P detection threshold | 0.3 | 0.3 |
| **Student evaluation** S detection threshold | 0.3 | 0.3 |
| **Student evaluation** P/S pick-matching tolerance | 0.5 s | 0.5 s |

All 12 supplied formal run records agree with the formal settings relevant to their variants. The two JSON profiles supply the training values and the KD gate. The evaluation constants are shared in `config.py:40–43`. Demo training of the Teacher uses batch 3 and one epoch. The demo exists to exercise the pipeline and the gate, with no accuracy pass criterion.

## 3. One-command runner

| Check | Status | Finding |
| --- | --- | --- |
| New output path and explicit start | PASS | `scripts/run_full_smoke_test.ps1` invokes the Python runner in the repository. `scripts/run_smoke.py:30–53,83–90` rejects an existing output path, including an empty folder, for a fresh run. Without the reuse flag, the first stage is `python -m teacher.cli train`. The reuse branch is opt-in and checks the existing demo checkpoint and provenance. |
| Output and isolation | PASS | Child stages select the STEAD demo profile. They clear inherited variant, seed and teacher weight overrides and use the selected output root. The first split for each cache starts a new manifest. Validation appends to the cache generated in that same run. Student data, caches and checkpoints are under distinct demo identities. |
| Failure and logs | PASS | The source shows exclusive creation of per-stage logs and immediate termination on a nonzero child exit code. Task 54 recorded all 24 successful stages and their logs, including `verify_smoke`. Fault injection remains untested. |
| Reusing a shell with invalid inherited settings | FAIL | `scripts/run_smoke.py:9` imports `release_config` before it constructs the child environment. An invalid inherited profile or seed can abort before stage setup. The README procedure uses a fresh shell. A source fix would require separate approval. |
| Complete from-scratch GPU run in a single invocation | PASS | The one-command Task 54 run began with `python -m teacher.cli train` in a previously nonexistent output root. All 24 stages completed with CUDA Teacher and Student runs, new cache identity, strict reloads and evaluation. |

Task 54 used the two existing Python environments and the bundled STEAD example. New files are under `outputs/release_audit_fresh`. The Teacher checkpoint was created at 2026-11-14 23:35:42.531559 UTC, and its SHA-256 `384F36ABB38A0EC9C78782D9B02A82E627D5BD16C1599E359626BAB404283392` matches both new cache manifests and both Teacher prediction splits. All 141 protected file hashes matched, including historical outputs and formal candidate weights. [Run evidence](SMOKE_TEST_REPORT.md#task-54-fresh-one-command-gpu-run).

## 4. Environments and dependencies

- **PASS** Both independent YAML files parsed with PyYAML. AST import review was cross-checked with the two requirements files and installed distribution metadata. The installed pins match and both environments passed `pip check`. The runner uses stdlib orchestration and isolates TensorFlow and PyTorch in separate processes.
- **PASS** EQT uses Python 3.8.20, TensorFlow/Keras 2.10.0, CUDA 11.2.2 and cuDNN 8.1.0.77 runtime packages. The native-Windows GPU YAML now declares compatible CUDA and cuDNN versions. [TensorFlow native Windows instructions](https://www.tensorflow.org/install/pip#windows-native).
- **PASS** Student uses Python 3.11.16 and PyTorch 2.12.1+cu130. Its CUDA build is 13.0 and the installed cuDNN DLL is 9.20.0.48. The historical runtime integer is 92000. The docs now provide the separate wheel-selection command for Windows CPython 3.11. [Official PyTorch wheel index](https://download.pytorch.org/whl/cu130/torch/).
- **NOT TESTED** Solving and installing from empty Conda environments, CPU-only and alternate OS execution. The Task 54 GPU run exercised the existing Windows environments.

## 5. Source, licence and authorship

| Item | Status | Evidence or outstanding requirement |
| --- | --- | --- |
| LPPN-derived Student and GPL text | PASS | The archived `models/LPPNM.py` identity matches the locally retained source snapshot. Its SHA-256 is recorded in the notices. The archived GPL-3.0 text matches the root licence. The separate LPPN author's repository credits Ziye Yu and Weitao Wang at CEA-IGP and links csnbench. [csnbench licence](https://github.com/cangyeone/csnbench/blob/master/LICENSE) · [LPPN authors and repository](https://github.com/cangyeone/LPPN#authors). |
| Historical csnbench model file | PASS | Task 55 matched `models/LPPNM.py` byte-for-byte to commit `66614c5c1ffc13fcabd0f4c0b7a599e17bebb839`. The original archive's overall checkout HEAD remains undetermined. [Pinned model](https://github.com/cangyeone/csnbench/blob/66614c5c1ffc13fcabd0f4c0b7a599e17bebb839/models/LPPNM.py) |
| EQTransformer subset and MIT notice | PASS | The independent S. Mostafa Mousavi MIT text and original headers are retained. Task 56 identifies local data and label adaptations in comments with their respective attribution. The earlier Task 55 comparison describes the functional changes to the vendored generator. [Upstream EQTransformer licence](https://github.com/smousavi05/EQTransformer/blob/2d1fa094a705798e68757e6b3c802361810cf8c3/LICENSE) |
| EQTransformer and Keras historical revisions | BLOCKED | The EQT package field and the comparable Keras commits do not identify the exact historical integrated snapshots. [Third-party sources and notices](../THIRD_PARTY_NOTICES.md) |
| Embedded Keras and BMC MIT notices | PASS | Component-level source comments and the full original MIT notices name Zhao HG, PoW and Marcos Duarte. [Third-party notices](../THIRD_PARTY_NOTICES.md) |
| Software author and GPL modifications | PASS | Qineng Song is the confirmed sole software author, contributor and maintainer. The GPL-derived model and loss record 2026 as the documented modification period. Official CFF 1.2.0 schema validation passed. [Software citation](../CITATION.cff) |
| Institutional rights and release authority | NEEDS_USER_CONFIRMATION | Taiyuan University of Technology ownership and authority to publish still require formal confirmation. |
| STEAD example | PASS | The packaged HDF5 is the 14-trace STEAD sample whose split counts, waveform hashes and origins are in the sample manifests. Both data notices credit the original authors and explain selection, repackaging and conversion of labels. The sample license remains CC BY 4.0. [Official STEAD licence](https://github.com/smousavi05/STEAD/blob/master/LICENSE). |
| Shanxi data and model assets | BLOCKED | Distribution permission for source waveforms, split data, weights and derivatives is unconfirmed. The 12 validated pure Student state dictionaries and their manifest remain excluded. |

## 6. Release file boundary and fixes

- **PASS** The candidate set includes the README, licence and data notices, citation file, two environment recipes, four variants, training/prediction/evaluation entries, both teacher cache generators, demo and formal template settings, and the documented STEAD sample. The sole candidate binary dataset is `data/sample_stead/sample.hdf5` at 1,048,272 bytes.
- **PASS** Task 58 verified all 64 allowlisted paths and 181 exclusions. The ignore rules cover nested run directories, private configuration and local audit records. The nine STEAD file exceptions are unchanged. Rule tests used a Git ignore parser. Native Git tracking and index behaviour remain untested.
- **PASS** Text scanning checked common credentials, private URLs, absolute local paths and user/environment path patterns. The allowlisted candidates contain no credentials or private absolute paths under those rules. Some excluded output logs and metadata have local paths. The scan is a pattern check and leaves excluded checkpoint payloads uninterpreted.
- **FAIL** Pre-existing Student checkpoint entry points use `weights_only=False`. They are suitable for locally generated or otherwise trusted checkpoint bundles. The documentation now specifies the trust boundary and separately describes weights-only loading of pure state dictionaries. Approval would be required for a source change. The appropriate compatibility test covers native demo checkpoints and local formal checkpoint formats, then rejection of unsupported serialized metadata, with no execution of untrusted input.
- **PASS** Task 53 changed documentation and release rules. Task 56 changed attribution documents and source comments. AST and non-comment tokens matched for all Python edits. Historical data, configurations and weights retained their fingerprints.
- **NOT TESTED** Task 56 performed static checks without another training or prediction run. Comments alter the existing byte-level source identity. Resume and the feature-cache CLI verification still require the source fingerprints stored when the cache was generated. The Task 54 evidence remains a record of its own source snapshot.

### Decision

**A. Release candidates.** The [file manifest](RELEASE_FILE_MANIFEST.csv) records the 64-file Task 58 allowlist, verified hashes and sizes, and exact exclusions for the four local audit documents. Its own row has empty hash and size fields.

**B. Exclusions.** Local output and checkpoint trees, Shanxi source and split assets, formal EQT and Student weights, teacher caches, internal logs and audit documents, local environments, benchmark data and generated files stay outside the allowlist.

**C. Confirmation pending.** Institutional rights and authority, exact vendored revisions, first-release metadata and any separate redistribution of formal Shanxi artifacts.

**D. Remaining work.** Complete the final rights and provenance review. First-release metadata remains open. Source hardening is a separate approval item. The existing workflow relies on immutable waveforms and a clean shell.

**E. GitHub status.** The static repository-boundary review passed. Public release remains blocked pending the source and authority checks.

**F. Claims still unverified.** Clean install, general offline cache content identity, cross-platform execution, full formal Shanxi replication, and precision or benchmark conclusions from the smoke data.
