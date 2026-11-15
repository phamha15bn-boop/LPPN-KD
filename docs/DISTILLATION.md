# Distillation

`lppn/loss.py` computes the supervised objective: sum-reduced seven-class cross-entropy with ignore index −1, plus a sum of squared phase-cell offset errors divided by ten. KD leaves the supervised labels unchanged.

| Variant | Objective | Offline input |
| --- | --- | --- |
| LPPN-BASE | Supervised | Waveform and picks |
| LPPN-KD-C | Supervised + 0.5 × C | P curve |
| LPPN-KD-FP | Supervised + 365000 × FP | P decoder feature and corresponding peak |
| LPPN-KD-CFP | Supervised + 0.5 × C + 365000 × FP | Both caches |

`lppn/kd.py` reduces each group of 8 teacher P samples to its maximum and matches the resulting soft target to the Student softmax P channel using binary cross-entropy summed over eligible real cells. `lppn/kdfp.py` slices Student features to 750 cells, applies a bias-free 32-to-32 projection initialised as the identity matrix, and L2-normalises both projected Student and detached Teacher vectors along the channel dimension with epsilon 1e-6. Its MSE averages over the eligible traces, time cells, and channels. The Student and projection both receive FP gradients and are in the Student Adam optimiser. `lppn/kdcfp.py` uses the peak from each component's own cache for its gate, then adds the weighted losses to supervision. The last 18 cells are excluded from both KD terms. Existing supervised targets and Student prediction retain the full cell axis.

The **teacher KD gate** requires a valid manual P sample in the original 6,000-sample interval and a non-noise trace. Its two tests are P-peak confidence and the absolute difference between the teacher P peak and manual P. It leaves the manual targets in control of supervision. Noise retains supervised background targets. Missing S creates no S target. Empty KD gates yield graph-connected zero terms.

| Control | STEAD Student demo | Formal Shanxi Student |
| --- | ---: | ---: |
| Teacher KD minimum P-peak confidence | 0.0 | 0.5 |
| Teacher KD maximum teacher-to-manual P error | 60 s | 0.5 s |
| Student P detection threshold | 0.3 | 0.3 |
| Student S detection threshold | 0.3 | 0.3 |
| Student P/S pick-matching tolerance | 0.5 s | 0.5 s |
| Student epochs | 1 | 100 |
| Student training batch size | 3 | 128 |
| Student learning rate | 0.0001 | 0.0001 |
| Student seeds | 42 | 42, 43, 44 |

The demo gate is set to exercise KD gradients with a one-epoch teacher. The cache tooling records the teacher weight SHA-256, code identity, preprocessing, protocol, split CSV and trace names. Student training and validation use only the corresponding train and validation caches. Prediction loads only the Student.
