import numpy as np


def demodulate_16qam(symbols, auto_normalize: bool = True):
    """
    Demodulate 16QAM symbols into binary bits using standard Gray mapping.

    Standard Gray mapping per axis (I and Q):
        -3 / sqrt(10) -> 00
        -1 / sqrt(10) -> 01
        +1 / sqrt(10) -> 11
        +3 / sqrt(10) -> 10

    Parameters
    ----------
    symbols : array_like
        Complex symbols.
    auto_normalize : bool
        Whether to scale symbols by RMS amplitude.

    Returns
    -------
    bits : np.ndarray of dtype uint8 (4 bits per symbol: 2 for I, 2 for Q)
    """
    symbols = np.asarray(symbols, dtype=np.complex128)

    if symbols.size == 0:
        return np.array([], dtype=np.uint8)

    if auto_normalize:
        rms = np.sqrt(np.mean(np.abs(symbols) ** 2))
        if rms > 1e-12:
            symbols = symbols / rms
        else:
            raise ValueError("Input symbols have zero amplitude.")

    # Standard levels normalized by sqrt(10)
    norm_factor = np.sqrt(10.0)
    levels = np.array([-3.0, -1.0, 1.0, 3.0]) / norm_factor

    gray_bits = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 1),
        3: (1, 0),
    }

    def decide(val):
        dists = np.abs(val - levels)
        return int(np.argmin(dists))

    bits = []
    for s in symbols:
        i_idx = decide(s.real)
        q_idx = decide(s.imag)
        bits.extend(gray_bits[i_idx])
        bits.extend(gray_bits[q_idx])

    return np.array(bits, dtype=np.uint8)


def demodulate_qam(symbols, order=16):
    """
    Generic QAM demodulator entry point.
    """
    if order == 16:
        return demodulate_16qam(symbols)

    raise ValueError(
        f"Unsupported QAM order: {order}. Currently only 16QAM is supported."
    )