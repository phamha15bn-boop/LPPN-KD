# Copyright (c) 2026 Qineng Song

"""Teacher protocol shared by the demo and formal experiment profiles."""

from release_config import DATA_PATHS, ROOT, TEACHER, TEACHER_SHA256, TEACHER_WEIGHT, cache_root

PROTOCOL_ID = TEACHER["p_protocol"]
TEACHER_ROOT = ROOT / "teacher" / "eqtransformer"
TEACHER_WEIGHT_SHA256 = TEACHER_SHA256
INPUT_SHAPE = (6000, 3)
P_OUTPUT_LENGTH = 6000
OUTPUT_NAMES = ("detector", "picker_P", "picker_S")
NORMALIZATION = "per-component (x-mean)/(std+1e-6)"
EXECUTION_MODE = "tf.function(model(x, training=False), fixed [None,6000,3])"
DROPOUT_PROTOCOL = (
    "Both forced SpatialDropout1D calls are replaced at construction time by "
    "a local adapter whose call always uses training=False; all other dropout "
    "also receives model training=False. No MC dropout."
)

CACHE_SPLITS = {
    split: {kind: DATA_PATHS[f"{split}_{kind}"] for kind in ("hdf5", "csv")}
    for split in ("train", "validation")
}
DEFAULT_CACHE_ROOT = cache_root("teacher_cache", PROTOCOL_ID)
