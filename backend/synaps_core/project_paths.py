"""
Centralized project paths and canonical dataset resolution for SYNAPS (SIH26147).

This module serves as the single source of truth for repository directory paths,
supported modulation classes, and canonical filename/sample resolution across
the new dataset layout (v2).

Dataset layout:
    data/{CLASS}/single/iq/{CLASS}/     (pure: BPSK, QPSK, FSK, QAM16)
    data/{CLASS}/single/metadata/{CLASS}/
    data/{CLASS}/single/wav/{CLASS}/
    data/MIXED/mixed/iq/MIXED/
    data/MIXED/mixed/metadata/MIXED/
    data/MIXED/mixed/wav/MIXED/
    data/MIXED/dataset.csv              (master CSV with split column)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import re


# =====================================================================
# CANONICAL DIRECTORY PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"
DATASET_CSV = DATA_ROOT / "MIXED" / "dataset.csv"

IQ_ROOT = DATA_ROOT
WAV_ROOT = DATA_ROOT
METADATA_ROOT = DATA_ROOT

MODELS_ROOT = PROJECT_ROOT / "models"
AI_MODELS_ROOT = PROJECT_ROOT / "ai" / "models"
TRANSFORMER_CHECKPOINT = AI_MODELS_ROOT / "transformer.pth"

RESULT_ROOT = PROJECT_ROOT / "result"
TRANSFORMER_RESULT_DIR = RESULT_ROOT / "transformer"


# =====================================================================
# MODULATION CLASSES & MAPPINGS
# =====================================================================

CLASS_NAMES: List[str] = [
    "BPSK",
    "QPSK",
    "FSK",
    "QAM16",
    "MIXED",
]

NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_TO_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS: Dict[int, str] = {
    idx: name for idx, name in enumerate(CLASS_NAMES)
}

# Aliases for robust mapping
MODULATION_ALIASES: Dict[str, str] = {
    "BPSK": "BPSK",
    "2BPSK": "BPSK",
    "QPSK": "QPSK",
    "4QPSK": "QPSK",
    "4PSK": "QPSK",
    "FSK": "FSK",
    "2FSK": "FSK",
    "2-FSK": "FSK",
    "QAM16": "QAM16",
    "16QAM": "QAM16",
    "16-QAM": "QAM16",
    "16_QAM": "QAM16",
    "MIXED": "MIXED",
}

# Pure modulation classes (everything except MIXED)
PURE_CLASS_NAMES: List[str] = ["BPSK", "QPSK", "FSK", "QAM16"]


def normalize_modulation_name(mod_name: str) -> str:
    """
    Normalize any modulation string into canonical class name:
    'BPSK', 'QPSK', 'FSK', 'QAM16', or 'MIXED'.
    """
    clean = str(mod_name).strip().upper().replace("-", "").replace("_", "")
    if clean in MODULATION_ALIASES:
        return MODULATION_ALIASES[clean]
    if "MIXED" in clean:
        return "MIXED"
    if "16QAM" in clean or "QAM16" in clean:
        return "QAM16"
    if "2FSK" in clean or "FSK" in clean:
        return "FSK"
    if "QPSK" in clean:
        return "QPSK"
    if "BPSK" in clean:
        return "BPSK"
    return mod_name.upper()


# =====================================================================
# DATASET PATH HELPERS
# =====================================================================

def get_class_iq_dir(class_name: str) -> Path:
    """Return the IQ directory for a given class."""
    class_name = normalize_modulation_name(class_name)
    if class_name == "MIXED":
        return DATA_ROOT / "MIXED" / "mixed" / "iq" / "MIXED"
    return DATA_ROOT / class_name / "single" / "iq" / class_name


def get_class_metadata_dir(class_name: str) -> Path:
    """Return the metadata directory for a given class."""
    class_name = normalize_modulation_name(class_name)
    if class_name == "MIXED":
        return DATA_ROOT / "MIXED" / "mixed" / "metadata" / "MIXED"
    return DATA_ROOT / class_name / "single" / "metadata" / class_name


def get_class_wav_dir(class_name: str) -> Path:
    """Return the WAV directory for a given class."""
    class_name = normalize_modulation_name(class_name)
    if class_name == "MIXED":
        return DATA_ROOT / "MIXED" / "mixed" / "wav" / "MIXED"
    return DATA_ROOT / class_name / "single" / "wav" / class_name


# =====================================================================
# CANONICAL SAMPLE RESOLUTION
# =====================================================================

def resolve_sample_paths(
    identifier_or_path: Union[str, Path]
) -> Dict[str, Optional[Path]]:
    """
    Resolve any sample path, filename, or stem into its canonical:
      - 'iq_path': Path to .iq file
      - 'wav_path': Path to .wav file
      - 'metadata_path': Path to .json file
      - 'class_name': Canonical modulation class name
      - 'sample_id': Canonical sample ID (e.g. 'signal_00601')

    Handles naming variations seamlessly:
      signal_00001_bpsk.iq  <-> signal_00001_bpsk.json
      signal_03001_mixed.iq <-> signal_03001_mixed.json
    """
    path = Path(identifier_or_path)
    stem = path.stem.lower()

    # Extract sample number (e.g. '00601' from 'signal_00601_qpsk')
    match = re.search(r"signal_(\d+)", stem)
    if match:
        sample_num_str = match.group(1)
        sample_id = f"signal_{sample_num_str}"
    else:
        num_match = re.search(r"(\d+)", stem)
        if num_match:
            sample_num_str = num_match.group(1)
            sample_id = f"signal_{int(sample_num_str):05d}"
        else:
            sample_id = stem
            sample_num_str = None

    # Determine class from stem or parent directory
    class_name = None
    stem_lower = stem
    parent_lower = path.parent.name.lower() if path.parent else ""

    if "mixed" in stem_lower or parent_lower == "mixed":
        class_name = "MIXED"
    elif "bpsk" in stem_lower or parent_lower == "bpsk":
        class_name = "BPSK"
    elif "qpsk" in stem_lower or parent_lower == "qpsk":
        class_name = "QPSK"
    elif "fsk" in stem_lower or "2fsk" in stem_lower or parent_lower == "fsk":
        class_name = "FSK"
    elif "qam" in stem_lower or "16qam" in stem_lower or parent_lower == "qam16":
        class_name = "QAM16"

    # Search candidates in class directory
    search_classes = [class_name] if class_name else CLASS_NAMES

    iq_path = None
    wav_path = None
    meta_path = None

    # Direct check if path is already existing absolute/relative file
    if path.exists() and path.is_file():
        ext = path.suffix.lower()
        if ext == ".iq":
            iq_path = path.resolve()
        elif ext == ".wav":
            wav_path = path.resolve()
        elif ext == ".json":
            meta_path = path.resolve()

    # Build candidate filenames from stem
    candidate_suffixes = []
    if class_name:
        mod_lower = class_name.lower()
        candidate_suffixes.append(f"{sample_id}_{mod_lower}")
        if class_name == "FSK":
            candidate_suffixes.append(f"{sample_id}_2fsk")
            candidate_suffixes.append(f"{sample_id}_fsk")
        elif class_name == "QAM16":
            candidate_suffixes.append(f"{sample_id}_16qam")
            candidate_suffixes.append(f"{sample_id}_qam16")
    candidate_suffixes.append(stem)

    # Search in class folders
    for c in search_classes:
        iq_dir = get_class_iq_dir(c)
        wav_dir = get_class_wav_dir(c)
        meta_dir = get_class_metadata_dir(c)

        for base in candidate_suffixes:
            if iq_path is None and iq_dir.exists():
                candidate = iq_dir / f"{base}.iq"
                if candidate.exists():
                    iq_path = candidate.resolve()
                    if class_name is None:
                        class_name = c

            if wav_path is None and wav_dir.exists():
                candidate = wav_dir / f"{base}.wav"
                if candidate.exists():
                    wav_path = candidate.resolve()
                    if class_name is None:
                        class_name = c

            if meta_path is None and meta_dir.exists():
                candidate = meta_dir / f"{base}.json"
                if candidate.exists():
                    meta_path = candidate.resolve()
                    if class_name is None:
                        class_name = c

    return {
        "iq_path": iq_path,
        "wav_path": wav_path,
        "metadata_path": meta_path,
        "class_name": class_name,
        "sample_id": sample_id,
    }
