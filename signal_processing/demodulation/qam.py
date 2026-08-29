import numpy as np


def demodulate_16qam(symbols):
    """
    Demodulate normalized 16QAM symbols into bits.

    Gray mapping:
        00 -> -3
        01 -> -1
        11 -> +1
        10 -> +3

    The constellation is normalized by sqrt(10).
    """

    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.size == 0:
        return np.array([], dtype=np.uint8)

    # Remove arbitrary amplitude scaling.
    # For 16QAM, RMS amplitude is approximately the signal amplitude.
    rms = np.sqrt(np.mean(np.abs(symbols) ** 2))

    if rms < 1e-12:
        raise ValueError("Input symbols have zero amplitude.")

    normalized = symbols / rms

    # Normalized 16QAM levels
    levels = np.array([
        -3 / np.sqrt(10),
        -1 / np.sqrt(10),
         1 / np.sqrt(10),
         3 / np.sqrt(10)
    ])

    def decide(value):
        distances = np.abs(value - levels)
        return int(np.argmin(distances))

    # Gray-coded mapping:
    #
    # index 0 = -3 -> 00
    # index 1 = -1 -> 01
    # index 2 = +1 -> 11
    # index 3 = +3 -> 10

    gray_bits = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0)
    }

    bits = []

    for symbol in normalized:
        i_index = decide(symbol.real)
        q_index = decide(symbol.imag)

        bits.extend(gray_bits[i_index])
        bits.extend(gray_bits[q_index])

    return np.array(bits, dtype=np.uint8)


# Generic name for compatibility with the demodulation pipeline
def demodulate_qam(symbols, order=16):
    """
    Generic QAM demodulator entry point.

    Currently supports 16QAM.
    """

    if order == 16:
        return demodulate_16qam(symbols)

    raise ValueError(
        f"Unsupported QAM order: {order}. "
        "Currently only 16QAM is supported."
    )