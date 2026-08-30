"""
Unit tests for signal input, format detection, loader, metadata, and dataset validation.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal_processing.input.format_detection import detect_format
from signal_processing.input.loader import load_signal
from signal_processing.input.metadata import SignalMetadata, create_metadata
from signal_processing.input.validation import (
    validate_samples,
    validate_metadata,
    validate_dataset_sample,
)
from project_paths import IQ_ROOT, WAV_ROOT


def test_format_detection():
    assert detect_format("signal.iq") == "IQ"
    assert detect_format("path/to/signal.WAV") == "WAV"
    try:
        detect_format("invalid.mp3")
        assert False, "Should raise ValueError for unsupported format"
    except ValueError:
        pass
    print("[PASS] test_format_detection")


def test_signal_loader():
    iq_file = IQ_ROOT / "BPSK" / "signal_0001_bpsk.iq"
    samples, fs = load_signal(str(iq_file))
    assert len(samples) == 8192
    assert fs > 0
    assert np.iscomplexobj(samples)

    wav_file = WAV_ROOT / "QPSK" / "signal_0201_qpsk.wav"
    w_samples, w_fs = load_signal(str(wav_file))
    assert len(w_samples) == 8192
    assert w_fs > 0
    print("[PASS] test_signal_loader")


def test_metadata_dataclass():
    meta = create_metadata(
        signal_id="test_0001",
        file_name="test_0001.iq",
        file_format="IQ",
        sample_rate=1_000_000.0,
        number_of_samples=8192,
        modulation_type="BPSK",
        signal_to_noise_ratio=15.0,
    )
    d = meta.to_dict()
    assert d["signal_id"] == "test_0001"
    assert d["modulation_type"] == "BPSK"

    reconstructed = SignalMetadata.from_dict(d)
    assert reconstructed.signal_id == "test_0001"
    validate_metadata(reconstructed)
    print("[PASS] test_metadata_dataclass")


def test_dataset_sample_validation():
    iq_file = IQ_ROOT / "BPSK" / "signal_0001_bpsk.iq"
    res = validate_dataset_sample(str(iq_file))
    assert res["status"] == "VALID", f"Expected VALID sample, got: {res}"
    assert res["sample_count"] == 8192
    print("[PASS] test_dataset_sample_validation")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING INPUT MODULE TESTS")
    print("=" * 60)
    test_format_detection()
    test_signal_loader()
    test_metadata_dataclass()
    test_dataset_sample_validation()
    print("\nALL INPUT TESTS PASSED SUCCESSFULLY!")