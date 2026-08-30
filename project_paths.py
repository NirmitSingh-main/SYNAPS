"""
Centralized project paths and canonical dataset resolution for SYNAPS (SIH26147).

This module serves as the single source of truth for repository directory paths,
supported modulation classes, and canonical filename/sample resolution across
varying naming conventions (e.g. signal_0601_qam16.iq <-> signal_0601_16qam.json).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re


# =====================================================================
# CANONICAL DIRECTORY PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"
IQ_ROOT = DATA_ROOT / "iq"
WAV_ROOT = DATA_ROOT / "wav"
METADATA_ROOT = DATA_ROOT / "metadata"
PROCESSED_ROOT = DATA_ROOT / "processed"

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
]

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
}


def normalize_modulation_name(mod_name: str) -> str:
    """
    Normalize any modulation string into canonical class name:
    'BPSK', 'QPSK', 'FSK', or 'QAM16'.
    """
    clean = str(mod_name).strip().upper().replace("-", "").replace("_", "")
    if clean in MODULATION_ALIASES:
        return MODULATION_ALIASES[clean]
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
      - 'class_name': Canonical modulation class name ('BPSK', 'QPSK', 'FSK', 'QAM16')
      - 'sample_id': Canonical sample ID (e.g. 'signal_0601')

    Handles naming variations seamlessly:
      signal_0601_qam16.iq <-> signal_0601_16qam.json <-> signal_0601_qam16.wav
      signal_0401_fsk.iq   <-> signal_0401_2fsk.json  <-> signal_0401_fsk.wav
      signal_0001_bpsk.iq  <-> signal_0001_bpsk.json  <-> signal_0001_bpsk.wav
      signal_0201_qpsk.iq  <-> signal_0201_qpsk.json  <-> signal_0201_qpsk.wav
    """
    path = Path(identifier_or_path)
    stem = path.stem.lower()

    # Extract sample number (e.g. '0601' from 'signal_0601_qam16' or '0601')
    match = re.search(r"signal_(\d+)", stem)
    if match:
        sample_num = int(match.group(1))
        sample_id = f"signal_{sample_num:04d}"
    else:
        num_match = re.search(r"(\d+)", stem)
        if num_match:
            sample_num = int(num_match.group(1))
            sample_id = f"signal_{sample_num:04d}"
        else:
            sample_num = None
            sample_id = stem

    # Determine class from stem or path
    class_name = None
    if "bpsk" in stem or (path.parent and "bpsk" in path.parent.name.lower()):
        class_name = "BPSK"
    elif "qpsk" in stem or (path.parent and "qpsk" in path.parent.name.lower()):
        class_name = "QPSK"
    elif "fsk" in stem or "2fsk" in stem or (path.parent and "fsk" in path.parent.name.lower()):
        class_name = "FSK"
    elif "qam" in stem or "16qam" in stem or (path.parent and "qam" in path.parent.name.lower()):
        class_name = "QAM16"
    elif sample_num is not None:
        # Range-based fallback based on standard dataset partitioning
        if 1 <= sample_num <= 200:
            class_name = "BPSK"
        elif 201 <= sample_num <= 400:
            class_name = "QPSK"
        elif 401 <= sample_num <= 600:
            class_name = "FSK"
        elif 601 <= sample_num <= 800:
            class_name = "QAM16"

    # Search candidates in class directory or flat root
    search_classes = [class_name] if class_name else CLASS_NAMES

    iq_path = None
    wav_path = None
    meta_path = None

    # Candidate file patterns for this sample
    iq_names = [
        f"{sample_id}_bpsk.iq", f"{sample_id}_qpsk.iq",
        f"{sample_id}_fsk.iq", f"{sample_id}_2fsk.iq",
        f"{sample_id}_qam16.iq", f"{sample_id}_16qam.iq",
        f"{stem}.iq", path.name if path.suffix.lower() == ".iq" else f"{path.name}.iq"
    ]
    wav_names = [
        f"{sample_id}_bpsk.wav", f"{sample_id}_qpsk.wav",
        f"{sample_id}_fsk.wav", f"{sample_id}_2fsk.wav",
        f"{sample_id}_qam16.wav", f"{sample_id}_16qam.wav",
        f"{stem}.wav", path.name if path.suffix.lower() == ".wav" else f"{path.name}.wav"
    ]
    json_names = [
        f"{sample_id}_bpsk.json", f"{sample_id}_qpsk.json",
        f"{sample_id}_2fsk.json", f"{sample_id}_fsk.json",
        f"{sample_id}_16qam.json", f"{sample_id}_qam16.json",
        f"{stem}.json", path.name if path.suffix.lower() == ".json" else f"{path.name}.json"
    ]

    # Direct check if path is already existing absolute/relative file
    if path.exists() and path.is_file():
        ext = path.suffix.lower()
        if ext == ".iq":
            iq_path = path.resolve()
        elif ext == ".wav":
            wav_path = path.resolve()
        elif ext == ".json":
            meta_path = path.resolve()

    # Search in class folders
    for c in search_classes:
        iq_dir = IQ_ROOT / c
        wav_dir = WAV_ROOT / c
        meta_dir = METADATA_ROOT / c

        if iq_path is None and iq_dir.exists():
            for name in iq_names:
                candidate = iq_dir / name
                if candidate.exists():
                    iq_path = candidate.resolve()
                    if class_name is None:
                        class_name = c
                    break

        if wav_path is None and wav_dir.exists():
            for name in wav_names:
                candidate = wav_dir / name
                if candidate.exists():
                    wav_path = candidate.resolve()
                    if class_name is None:
                        class_name = c
                    break

        if meta_path is None and meta_dir.exists():
            for name in json_names:
                candidate = meta_dir / name
                if candidate.exists():
                    meta_path = candidate.resolve()
                    if class_name is None:
                        class_name = c
                    break

    return {
        "iq_path": iq_path,
        "wav_path": wav_path,
        "metadata_path": meta_path,
        "class_name": class_name,
        "sample_id": sample_id,
    }
