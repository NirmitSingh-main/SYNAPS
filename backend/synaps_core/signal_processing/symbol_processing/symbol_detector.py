import numpy as np


def detect_symbols(
    iq: np.ndarray,
    samples_per_symbol: int,
    timing_offset: int = 0,
) -> np.ndarray:
    """
    Extract one representative IQ sample for each symbol.

    Parameters
    ----------
    iq : numpy.ndarray
        Synchronized complex IQ samples.

    samples_per_symbol : int
        Number of samples representing one symbol.

    timing_offset : int, optional
        Starting sample position for symbol extraction.

    Returns
    -------
    numpy.ndarray
        Symbol-spaced IQ samples.
    """

    if not isinstance(iq, np.ndarray):
        raise TypeError(
            "iq must be a NumPy array."
        )

    if iq.size == 0:
        raise ValueError(
            "IQ signal cannot be empty."
        )

    if not np.iscomplexobj(iq):
        raise TypeError(
            "iq must contain complex-valued samples."
        )

    if samples_per_symbol <= 0:
        raise ValueError(
            "samples_per_symbol must be greater than zero."
        )

    if timing_offset < 0:
        raise ValueError(
            "timing_offset cannot be negative."
        )

    if timing_offset >= samples_per_symbol:
        raise ValueError(
            "timing_offset must be smaller than "
            "samples_per_symbol."
        )

    if timing_offset >= len(iq):
        raise ValueError(
            "timing_offset is outside the IQ signal."
        )

    symbols = iq[
        timing_offset::samples_per_symbol
    ]

    if symbols.size == 0:
        raise ValueError(
            "No symbols could be extracted."
        )

    return symbols