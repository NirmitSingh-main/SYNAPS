"""
Configuration constants and paths for SYNAPS (SIH26147).
"""

from project_paths import (
    PROJECT_ROOT,
    DATA_ROOT,
    DATASET_CSV,
    MODELS_ROOT,
    AI_MODELS_ROOT,
    TRANSFORMER_CHECKPOINT,
    RESULT_ROOT,
    TRANSFORMER_RESULT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    PURE_CLASS_NAMES,
    normalize_modulation_name,
    resolve_sample_paths,
    get_class_iq_dir,
    get_class_metadata_dir,
    get_class_wav_dir,
)

DEFAULT_SAMPLING_RATE = 1_000_000.0
DEFAULT_SAMPLES_PER_SYMBOL = 10
DEFAULT_NUM_TOKENS = 256
DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

