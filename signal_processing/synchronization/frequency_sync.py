import numpy as np


def correct_frequency_offset(iq, frequency_offset_hz, sampling_frequency_hz):
    """
    Correct a known frequency offset in a complex IQ signal.

    Parameters:
        iq: Complex IQ samples
        frequency_offset_hz: Frequency offset from metadata
        sampling_frequency_hz: Sampling frequency

    Returns:
        Frequency-corrected IQ samples
    """

    n = np.arange(len(iq))

    correction = np.exp(
        -1j * 2 * np.pi * frequency_offset_hz * n / sampling_frequency_hz
    )

    corrected_iq = iq * correction

    return corrected_iq


if __name__ == "__main__":

    # Load IQ data
    filepath = "data/iq/signal_0004_bpsk.iq"

    data = np.fromfile(filepath, dtype="<f4")

    i = data[0::2]
    q = data[1::2]

    iq = i + 1j * q

    # Values from signal_0004_bpsk.json
    frequency_offset_hz = -10000
    sampling_frequency_hz = 1000000

    corrected_iq = correct_frequency_offset(
        iq,
        frequency_offset_hz,
        sampling_frequency_hz
    )

    print("Frequency synchronization completed.")
    print("Original samples:", len(iq))
    print("Corrected samples:", len(corrected_iq))
    print("First 5 corrected samples:")
    print(corrected_iq[:5])