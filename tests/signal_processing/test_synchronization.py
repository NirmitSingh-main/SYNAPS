import numpy as np
import matplotlib.pyplot as plt

from signal_processing.synchronization.frequency_sync import (
    correct_frequency_offset
)

from signal_processing.synchronization.phase_sync import (
    correct_phase_offset
)

from signal_processing.synchronization.timing_sync import (
    estimate_timing_offset,
    sample_symbols
)


def test_synchronization():

    # Load IQ signal
    filepath = "data/iq/signal_0004_bpsk.iq"

    data = np.fromfile(filepath, dtype="<f4")

    i = data[0::2]
    q = data[1::2]

    iq = i + 1j * q

    # Signal metadata
    sampling_frequency_hz = 1_000_000
    frequency_offset_hz = -10_000
    phase_offset_degrees = 0
    samples_per_symbol = 10

    # 1. Frequency synchronization
    frequency_corrected = correct_frequency_offset(
        iq,
        frequency_offset_hz,
        sampling_frequency_hz
    )

    # 2. Phase synchronization
    phase_corrected = correct_phase_offset(
        frequency_corrected,
        phase_offset_degrees
    )

    # 3. Automatic timing synchronization
    timing_offset = estimate_timing_offset(
        phase_corrected,
        samples_per_symbol
    )

    symbols = sample_symbols(
        phase_corrected,
        samples_per_symbol,
        timing_offset
    )

    # Validation
    assert len(frequency_corrected) == len(iq)
    assert len(phase_corrected) == len(iq)
    assert len(symbols) > 0
    assert 0 <= timing_offset < samples_per_symbol
    assert np.all(np.isfinite(symbols))

    print("\nSynchronization successful.")
    print("Original samples:", len(iq))
    print("Samples per symbol:", samples_per_symbol)
    print("Estimated timing offset:", timing_offset)
    print("Recovered symbol samples:", len(symbols))

    # Symbol constellation
    plt.figure(figsize=(7, 7))

    plt.scatter(
        symbols.real,
        symbols.imag,
        s=8
    )

    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title("BPSK Symbol Constellation After Synchronization")

    plt.grid(True)
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    test_synchronization()


