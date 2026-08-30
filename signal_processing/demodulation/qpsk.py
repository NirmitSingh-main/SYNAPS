import numpy as np


def demodulate_qpsk(symbols):
    """
    Demodulate QPSK symbols using Gray coding.

    Mapping:
        +I, +Q -> 00
        -I, +Q -> 01
        -I, -Q -> 11
        +I, -Q -> 10
    """

    symbols = np.asarray(symbols)

    bits = []

    for symbol in symbols:

        if symbol.real >= 0 and symbol.imag >= 0:
            bits.extend([0, 0])

        elif symbol.real < 0 and symbol.imag >= 0:
            bits.extend([0, 1])

        elif symbol.real < 0 and symbol.imag < 0:
            bits.extend([1, 1])

        else:
            bits.extend([1, 0])

    return np.array(bits, dtype=np.uint8)