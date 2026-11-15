# Release checklist

| Item | Status | Evidence or next step |
| --- | --- | --- |
| Teacher topology, P head, 750×32 feature path | PASS | Source and historical compiled GPU trace agree. |
| Student hook, 768→750 alignment, projection and deployment | PASS | Source and historical FP/CFP shape assertions agree. |
| BASE, C, FP and CFP objective and gradients | PASS | Source traces and historical finite-gradient smoke evidence reviewed. |
| Demo and formal hyperparameters and evaluation thresholds | PASS | Two profiles, shared evaluation constants and 12 frozen records checked. |
| Current runtime feature hook recheck | PASS | The new Teacher emitted P `[1,6000]` and FP `[1,750,32]`. The new Student hook gave `[1,32,1,768]` then `[1,768,32]`. The CFP loss passed the first 750 positions into Projection and produced `[1,750,32]`. |
| Static one-command start, new output root, stage failure logic | PASS | Read both entry points, reuse guard and per-stage logging. |
| Single-invocation fresh GPU smoke | PASS | All 24 stages succeeded on RTX 3080. The new EQT checkpoint has from-scratch provenance. Both caches and four Student runs passed identity, strict reload and output checks. |
| Same-path waveform mutation detection on cache reuse | FAIL | Source HDF5 content is not fingerprinted. Use immutable inputs and fresh caches for changed waveforms. |
| Invalid inherited LPPN profile handling | FAIL | Use a clean shell. Runner bootstrap imports the profile before resetting child settings. |
| Loading external untrusted checkpoint bundles | FAIL | Existing pickle load belongs to the trusted-checkpoint workflow. |
| Local environments and dependency declarations | PASS | Interpreter/package metadata, YAML parse, direct import audit, both `pip check` results and official platform requirements. |
| Clean provisioning or alternative platforms | NOT TESTED | May be tested separately after approval where needed. |
| GPL-3.0, EQT MIT and STEAD CC BY 4.0 notices | PASS | Compared local licence texts and consulted primary upstream sources. |
| LPPN derived source and dated modification notices | PASS | Pinned csnbench LPPNM source and documented 2026 changes in both derived files. |
| EQTransformer, Keras helper, BMC and STEAD notices | PASS | Preserved independent EQT MIT and STEAD CC BY notices. Added the complete four embedded MIT notices and component-level source comments. |
| Software author and CFF | PASS | Qineng Song is the confirmed sole author, contributor and maintainer. Official CFF 1.2.0 schema validation passed. |
| Exact EQTransformer and Keras historical revisions | BLOCKED | Comparison sources are recorded. The original integrated revisions remain unresolved. |
| Institutional IP and release authority | NEEDS_USER_CONFIRMATION | Taiyuan University of Technology ownership and authority to publish still require formal confirmation. |
| Formal Shanxi artifact distribution | BLOCKED | Obtain specific authorization before considering any distribution. |
| Public-file inventory and ignore-rule behavior | PASS | Task 58 verified the 64-path allowlist, all 181 exclusions and ignore rules for real and simulated paths. |
| Candidate secrets and machine-path pattern scan | PASS | The static scan found no credentials or private absolute paths in the public allowlist. |
| Native Git tracking/index | NOT TESTED | No Git repository exists in the candidate. |
| Historical outputs, checkpoint hashes and experiment definitions | PASS | Task 54 verified all 141 protected files, including 82 old demo outputs and 13 formal candidate artifacts. |
| Final review handoff | PASS | Authorship and notice repairs are ready for final review. |
| Release decision | BLOCKED | Resolve remaining source provenance, institutional release authority and formal asset permissions. |

Task 54 used the bundled STEAD demo and wrote its outputs under `outputs/release_audit_fresh`. The four Student variants used 43,296 deployment parameters each, and the FP projection state stayed separate. F1 and accuracy were outside the execution acceptance criteria. [Task 54 results and hashes](SMOKE_TEST_REPORT.md#task-54-fresh-one-command-gpu-run).

Task 56 verified the attribution edits against the Python AST and non-comment tokens. Task 57 found that the old Demo manifests use earlier source fingerprints, so both cache families fail the current resume identity checks. The FP CLI also checks the source fingerprints in verify-only mode. Task 58 was static and kept these artifacts as historical evidence. [Third-party notices](../THIRD_PARTY_NOTICES.md)

[Audit findings and limitations](RELEASE_FINAL_AUDIT.md) · [Per-file decisions](RELEASE_FILE_MANIFEST.csv)
