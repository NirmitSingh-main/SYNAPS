import numpy as np


def demodulate_bpsk(symbols):
    """
    Demodulate BPSK symbols into bits.

    Positive in-phase component → 1
    Negative in-phase component → 0
    """

    symbols = np.asarray(symbols)

    bits = (symbols.real >= 0).astype(np.uint8)

    return bits