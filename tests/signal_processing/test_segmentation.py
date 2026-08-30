"""
Unit tests for energy detection, signal segmentation, and active region extraction.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal_processing.detection.energy_detection import calculate_average_power, detect_energy
from signal_processing.detection.segmentation import find_signal_region, extract_signal
from signal_processing.detection.signal_detector import detect_signal


def test_energy_detection():
    # Signal with noise burst in the middle
    noise = (np.random.randn(100) + 1j * np.random.randn(100)) * 0.01
    burst = (np.random.randn(100) + 1j * np.random.randn(100)) * 2.0
    sig = np.concatenate([noise, burst, noise]).astype(np.complex64)

    pwr = calculate_average_power(sig)
    assert pwr > 0.0

    mask = detect_energy(sig, threshold=0.1)
    assert mask.shape == (300,)
    assert np.any(mask[100:200])
    print("[PASS] test_energy_detection")


def test_signal_region_segmentation():
    mask = np.zeros(200, dtype=bool)
    mask[50:150] = True

    region = find_signal_region(mask, minimum_samples=10)
    assert region == (50, 150), f"Expected (50, 150), got {region}"

    samples = np.ones(200, dtype=np.complex64)
    extracted = extract_signal(samples, mask, minimum_samples=10)
    assert len(extracted) == 100
    print("[PASS] test_signal_region_segmentation")


def test_detect_signal_full():
    samples = (np.random.randn(500) + 1j * np.random.randn(500)).astype(np.complex64)
    det_sig, region = detect_signal(samples)
    assert len(det_sig) > 0
    assert region[0] >= 0 and region[1] <= len(samples)
    print("[PASS] test_detect_signal_full")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING SIGNAL DETECTION & SEGMENTATION TESTS")
    print("=" * 60)
    test_energy_detection()
    test_signal_region_segmentation()
    test_detect_signal_full()
    print("\nALL SEGMENTATION TESTS PASSED SUCCESSFULLY!")