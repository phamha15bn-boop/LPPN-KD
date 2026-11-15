# Architecture

## EQTransformer teacher

`teacher/eqtransformer/EQTransformer/core/trainer.py` constructs the vendored EQT `cred2` network from `EqT_utils.py`. The configuration uses five convolutional blocks, two bidirectional LSTM blocks and 0.1 dropout, with seven encoder filter widths from 8 to 64. The input has 6000 time samples and three ENZ components. The 373,495-parameter network emits `detector`, `picker_P`, and `picker_S`, each with shape `[batch,6000,1]`.

`teacher/inference.py` checks the HDF5 checkpoint SHA-256 and loads weights by topology. The adapter substitutes an inference-only SpatialDropout constructor while building the graph, covering the two forced-dropout call sites, and restores the original constructor afterwards. The training CLI uses the original factory. The adapter runs with `training=False` inside `tf.function` and converts outputs to NumPy for caching. `teacher/feature_inference.py` finds the unique Conv1D tensor with shape `[batch,750,32]` on the P decoder between `attentionP` and `picker_P`. The semantic layer identifier is `picker_P.decoder.stage4.post_conv`. Keras generated `conv1d_33` in the historical demo. The same forward pass produces the P curve and feature.

## LPPN-derived student

`lppn/model.py` defines `LPPNBase(model_n=8,n_stride=8)`. A trace is right-padded to 6144 and passed as `[batch,3,6144]`. The student emits seven class channels and per-cell offsets across 768 cells. The first 750 cells cover the real waveform. The last 18 cover zero padding. The P and S classes are 1 and 2 respectively. The offset head returns `8 × sigmoid` in sample units.

`lppn/feature_hooks.py` registers a PyTorch forward hook on `class_encoder` for feature distillation during training and loss validation. It captures the differentiable tensor `[batch,32,1,768]`, removes the height dimension, and transposes it to `[batch,768,32]`. The loss slices the first 750 cells before projection. The 1,024-parameter projection has separate state in training checkpoints. The predictor loads only the Student's `model_state`, which contains 43,296 trainable parameters plus buffers.
