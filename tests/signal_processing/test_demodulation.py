"""
Demodulation unit tests and dataset integration tests for BPSK, QPSK, FSK, and 16QAM.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import resolve_sample_paths
from ai.preprocessing.iq_loader import load_iq_file
from signal_processing.synchronization.frequency_sync import correct_frequency_offset
from signal_processing.synchronization.phase_sync import correct_phase_offset
from signal_processing.synchronization.timing_sync import estimate_timing_offset, sample_symbols
from signal_processing.demodulation.bpsk import demodulate_bpsk
from signal_processing.demodulation.qpsk import demodulate_qpsk
from signal_processing.demodulation.fsk import demodulate_fsk
from signal_processing.demodulation.qam import demodulate_16qam, demodulate_qam


# =====================================================================
# SYNTHETIC DETERMINISTIC UNIT TESTS
# =====================================================================

def test_synthetic_bpsk():
    """
    Deterministic synthetic test for BPSK demodulation.
    """
    tx_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    symbols = np.where(tx_bits == 1, 1.0 + 0j, -1.0 + 0j)
    
    rx_bits = demodulate_bpsk(symbols)
    assert np.array_equal(tx_bits, rx_bits), f"BPSK mismatch: {tx_bits} != {rx_bits}"
    print("[PASS] test_synthetic_bpsk: 100% match on deterministic bitstream")


def test_synthetic_qpsk():
    """
    Deterministic synthetic test for QPSK demodulation (Gray mapped).
    """
    # Mapping: +I,+Q->00, -I,+Q->01, -I,-Q->11, +I,-Q->10
    symbols = np.array([
        1.0 + 1.0j,   # 00
        -1.0 + 1.0j,  # 01
        -1.0 - 1.0j,  # 11
        1.0 - 1.0j,   # 10
    ], dtype=np.complex128) / np.sqrt(2.0)
    
    expected_bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    rx_bits = demodulate_qpsk(symbols)
    assert np.array_equal(expected_bits, rx_bits), f"QPSK mismatch: {expected_bits} != {rx_bits}"
    print("[PASS] test_synthetic_qpsk: 100% match on deterministic constellation")


def test_synthetic_fsk():
    """
    Deterministic synthetic test for 2-FSK demodulation.
    """
    sps = 16
    fs = 1_000_000.0
    f0 = 20_000.0  # bit 0
    f1 = 80_000.0  # bit 1
    tx_bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    
    t_sym = np.arange(sps) / fs
    signal_chunks = []
    phase = 0.0
    for b in tx_bits:
        freq = f1 if b == 1 else f0
        chunk = np.exp(1j * (2 * np.pi * freq * t_sym + phase))
        phase = np.angle(chunk[-1])
        signal_chunks.append(chunk)
        
    signal = np.concatenate(signal_chunks)
    rx_bits = demodulate_fsk(signal, samples_per_symbol=sps)
    assert np.array_equal(tx_bits, rx_bits), f"FSK mismatch: {tx_bits} != {rx_bits}"
    print("[PASS] test_synthetic_fsk: 100% match on continuous-phase FSK signal")


def test_synthetic_16qam():
    """
    Deterministic synthetic test for 16QAM demodulation (Gray mapped).
    """
    norm = np.sqrt(10.0)
    levels = [-3.0, -1.0, 1.0, 3.0]
    gray_map = {
        (-3, -3): [0, 0, 0, 0],
        (-1, 1):  [0, 1, 1, 1],
        (3, -1):  [1, 0, 0, 1],
        (1, 3):   [1, 1, 1, 0],
    }
    symbols = []
    expected_bits = []
    for (i_val, q_val), bits in gray_map.items():
        symbols.append((i_val + 1j * q_val) / norm)
        expected_bits.extend(bits)
        
    rx_bits = demodulate_16qam(np.array(symbols))
    assert np.array_equal(expected_bits, rx_bits), f"16QAM mismatch: {expected_bits} != {rx_bits}"
    print("[PASS] test_synthetic_16qam: 100% match on deterministic 16QAM constellation")


# =====================================================================
# DATASET INTEGRATION TESTS
# =====================================================================

def _test_dataset_demodulation(class_name: str, sample_identifier: str, demod_func, uses_timing: bool = True):
    resolved = resolve_sample_paths(sample_identifier)
    iq_path = resolved["iq_path"]
    meta_path = resolved["metadata_path"]
    
    assert iq_path and iq_path.exists(), f"IQ file missing: {iq_path}"
    assert meta_path and meta_path.exists(), f"Metadata missing: {meta_path}"
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    iq = load_iq_file(iq_path)
    sps = meta.get("samples_per_symbol", 10)
    fo = meta.get("frequency_offset_hz", 0.0)
    po = meta.get("phase_offset_degrees", 0.0)
    fs = meta.get("sampling_frequency_hz", 1_000_000.0)
    
    # Synchronize
    freq_sync = correct_frequency_offset(iq, fo, fs)
    phase_sync = correct_phase_offset(freq_sync, po)
    
    if uses_timing:
        timing_offset = estimate_timing_offset(phase_sync, sps)
        symbols = sample_symbols(phase_sync, sps, timing_offset)
        recovered_bits = demod_func(symbols)
    else:
        recovered_bits = demod_func(phase_sync, sps)
        
    assert len(recovered_bits) > 0, f"No bits recovered for {class_name}"
    assert np.all((recovered_bits == 0) | (recovered_bits == 1)), f"Non-binary bits for {class_name}"
    print(f"[PASS] Dataset integration {class_name}: successfully recovered {len(recovered_bits)} binary bits from {iq_path.name}")


def test_dataset_bpsk():
    _test_dataset_demodulation("BPSK", "signal_0001", demodulate_bpsk, uses_timing=True)


def test_dataset_qpsk():
    _test_dataset_demodulation("QPSK", "signal_0201", demodulate_qpsk, uses_timing=True)


def test_dataset_fsk():
    _test_dataset_demodulation("FSK", "signal_0401", demodulate_fsk, uses_timing=False)


def test_dataset_16qam():
    _test_dataset_demodulation("16QAM", "signal_0601", demodulate_16qam, uses_timing=True)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DEMODULATION TESTS")
    print("=" * 60)
    
    test_synthetic_bpsk()
    test_synthetic_qpsk()
    test_synthetic_fsk()
    test_synthetic_16qam()
    
    test_dataset_bpsk()
    test_dataset_qpsk()
    test_dataset_fsk()
    test_dataset_16qam()
    
    print("\nALL DEMODULATION TESTS PASSED SUCCESSFULLY!")