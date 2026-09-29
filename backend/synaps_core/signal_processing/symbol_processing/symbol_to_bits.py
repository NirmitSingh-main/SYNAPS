import numpy as np


def bpsk_symbols_to_bits(
    symbols: np.ndarray,
) -> np.ndarray:
    """
    Convert decided BPSK symbols to bits.

    Mapping:
        -1 -> 0
        +1 -> 1

    Parameters
    ----------
    symbols : numpy.ndarray
        Decided BPSK symbols.

    Returns
    -------
    numpy.ndarray
        Bits as uint8 values.
    """

    _validate_symbols(symbols)

    bits = np.where(
        np.real(symbols) >= 0,
        1,
        0,
    )

    return bits.astype(np.uint8)


def qpsk_symbols_to_bits(
    symbols: np.ndarray,
) -> np.ndarray:
    """
    Convert decided QPSK symbols to bits.

    Mapping:

        +1 + 1j -> 00
        -1 + 1j -> 01
        -1 - 1j -> 11
        +1 - 1j -> 10
    """

    _validate_symbols(symbols)

    bits = []

    for symbol in symbols:

        real_positive = symbol.real >= 0
        imag_positive = symbol.imag >= 0

        if real_positive and imag_positive:
            bits.extend([0, 0])

        elif not real_positive and imag_positive:
            bits.extend([0, 1])

        elif not real_positive and not imag_positive:
            bits.extend([1, 1])

        else:
            bits.extend([1, 0])

    return np.asarray(
        bits,
        dtype=np.uint8,
    )


def symbols_to_bits(
    symbols: np.ndarray,
    modulation: str,
) -> np.ndarray:
    """
    Convert decided symbols into bits.

    Supported modulation types:
        BPSK
        QPSK
    """

    if not isinstance(modulation, str):
        raise TypeError(
            "modulation must be a string."
        )

    modulation = modulation.upper()

    if modulation == "BPSK":
        return bpsk_symbols_to_bits(symbols)

    if modulation == "QPSK":
        return qpsk_symbols_to_bits(symbols)

    raise ValueError(
        f"Unsupported modulation: {modulation}. "
        "Supported modulations are BPSK and QPSK."
    )


def _validate_symbols(
    symbols: np.ndarray,
) -> None:
    """
    Validate symbol input.

    Both real-valued and complex-valued symbol
    arrays are accepted because BPSK decisions
    may be represented as -1/+1 real values.
    """

    if not isinstance(symbols, np.ndarray):
        raise TypeError(
            "symbols must be a NumPy array."
        )

    if symbols.size == 0:
        raise ValueError(
            "symbols cannot be empty."
        )

    if not np.isfinite(symbols).all():
        raise ValueError(
            "symbols contain invalid values."
        )