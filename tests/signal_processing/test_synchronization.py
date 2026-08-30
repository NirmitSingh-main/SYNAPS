import sys
from pathlib import Path

import numpy as np

# Ensure direct execution can import project packages.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signal_processing.synchronization.frequency_sync import (
    correct_frequency_offset,
)

from signal_processing.synchronization.phase_sync import (
    correct_phase_offset,
)

from signal_processing.synchronization.timing_sync import (
    estimate_timing_offset,
    sample_symbols,
)

from signal_processing.synchronization.carrier_recovery import (
    recover_carrier,
)


def generate_test_signal():
    """
    Generate a simple synthetic BPSK-like IQ signal
    with a known frequency and phase offset.
    """

    sampling_frequency_hz = 1_000_000
    samples_per_symbol = 10

    bits = np.array(
        [0, 1, 1, 0, 1, 0, 0, 1],
        dtype=np.uint8,
    )

    # BPSK symbols: 0 -> -1, 1 -> +1
    symbols = 2 * bits.astype(np.float64) - 1

    # Repeat each symbol for 10 samples
    iq = np.repeat(
        symbols,
        samples_per_symbol,
    ).astype(np.complex64)

    # Known impairments
    frequency_offset_hz = -10_000
    phase_offset_degrees = 30

    n = np.arange(len(iq))

    # Add frequency offset
    frequency_rotation = np.exp(
        1j
        * 2
        * np.pi
        * frequency_offset_hz
        * n
        / sampling_frequency_hz
    )

    # Add phase offset
    phase_rotation = np.exp(
        1j * np.deg2rad(phase_offset_degrees)
    )

    impaired_iq = (
        iq
        * frequency_rotation
        * phase_rotation
    )

    return (
        impaired_iq,
        sampling_frequency_hz,
        samples_per_symbol,
        frequency_offset_hz,
        phase_offset_degrees,
    )


def test_frequency_synchronization():
    """
    Test frequency offset correction.
    """

    (
        iq,
        sampling_frequency_hz,
        _,
        frequency_offset_hz,
        _,
    ) = generate_test_signal()

    corrected = correct_frequency_offset(
        iq,
        frequency_offset_hz,
        sampling_frequency_hz,
    )

    assert len(corrected) == len(iq)

    assert np.all(
        np.isfinite(corrected.real)
    )

    assert np.all(
        np.isfinite(corrected.imag)
    )


def test_phase_synchronization():
    """
    Test phase offset correction.
    """

    (
        iq,
        sampling_frequency_hz,
        _,
        frequency_offset_hz,
        phase_offset_degrees,
    ) = generate_test_signal()

    frequency_corrected = correct_frequency_offset(
        iq,
        frequency_offset_hz,
        sampling_frequency_hz,
    )

    phase_corrected = correct_phase_offset(
        frequency_corrected,
        phase_offset_degrees,
    )

    assert len(phase_corrected) == len(iq)

    assert np.all(
        np.isfinite(phase_corrected.real)
    )

    assert np.all(
        np.isfinite(phase_corrected.imag)
    )


def test_timing_synchronization():
    """
    Test timing offset estimation and symbol sampling.
    """

    (
        iq,
        sampling_frequency_hz,
        samples_per_symbol,
        frequency_offset_hz,
        phase_offset_degrees,
    ) = generate_test_signal()

    frequency_corrected = correct_frequency_offset(
        iq,
        frequency_offset_hz,
        sampling_frequency_hz,
    )

    phase_corrected = correct_phase_offset(
        frequency_corrected,
        phase_offset_degrees,
    )

    timing_offset = estimate_timing_offset(
        phase_corrected,
        samples_per_symbol,
    )

    symbols = sample_symbols(
        phase_corrected,
        samples_per_symbol,
        timing_offset,
    )

    assert 0 <= timing_offset < samples_per_symbol

    assert len(symbols) > 0

    assert np.all(
        np.isfinite(symbols.real)
    )

    assert np.all(
        np.isfinite(symbols.imag)
    )


def test_carrier_recovery():
    """
    Test the complete carrier recovery operation.

    Carrier recovery performs:
        frequency correction
        +
        phase correction
    """

    (
        iq,
        sampling_frequency_hz,
        _,
        frequency_offset_hz,
        phase_offset_degrees,
    ) = generate_test_signal()

    recovered = recover_carrier(
        iq,
        frequency_offset_hz,
        phase_offset_degrees,
        sampling_frequency_hz,
    )

    assert len(recovered) == len(iq)

    assert np.all(
        np.isfinite(recovered.real)
    )

    assert np.all(
        np.isfinite(recovered.imag)
    )


def test_complete_synchronization():
    """
    Test the complete synchronization pipeline.
    """

    (
        iq,
        sampling_frequency_hz,
        samples_per_symbol,
        frequency_offset_hz,
        phase_offset_degrees,
    ) = generate_test_signal()

    # ---------------------------------------------------------
    # 1. Carrier recovery
    # ---------------------------------------------------------

    recovered = recover_carrier(
        iq,
        frequency_offset_hz,
        phase_offset_degrees,
        sampling_frequency_hz,
    )

    # ---------------------------------------------------------
    # 2. Timing synchronization
    # ---------------------------------------------------------

    timing_offset = estimate_timing_offset(
        recovered,
        samples_per_symbol,
    )

    symbols = sample_symbols(
        recovered,
        samples_per_symbol,
        timing_offset,
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    assert len(recovered) == len(iq)

    assert len(symbols) > 0

    assert (
        0
        <= timing_offset
        < samples_per_symbol
    )

    assert np.all(
        np.isfinite(symbols.real)
    )

    assert np.all(
        np.isfinite(symbols.imag)
    )