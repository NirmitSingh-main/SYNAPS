import numpy as np


def decide_bpsk(symbols: np.ndarray) -> np.ndarray:
    """
    Make hard symbol decisions for BPSK.

    Decision rule:
        Real part >= 0  -> +1
        Real part < 0   -> -1

    Parameters
    ----------
    symbols : numpy.ndarray
        Complex-valued synchronized symbol samples.

    Returns
    -------
    numpy.ndarray
        BPSK symbol decisions as -1.0 or +1.0.
    """

    _validate_symbols(symbols)

    decisions = np.where(
        symbols.real >= 0,
        1.0,
        -1.0,
    )

    return decisions.astype(np.float32)


def decide_qpsk(symbols: np.ndarray) -> np.ndarray:
    """
    Make hard symbol decisions for QPSK.

    The sign of the I and Q components determines
    the quadrant.

    Returns:
        Complex-valued QPSK decisions:
            +1 + 1j
            -1 + 1j
            -1 - 1j
            +1 - 1j
    """

    _validate_symbols(symbols)

    real_decision = np.where(
        symbols.real >= 0,
        1.0,
        -1.0,
    )

    imag_decision = np.where(
        symbols.imag >= 0,
        1.0,
        -1.0,
    )

    decisions = (
        real_decision
        + 1j * imag_decision
    )

    return decisions.astype(np.complex64)


def decide(
    symbols: np.ndarray,
    modulation: str,
) -> np.ndarray:
    """
    Make hard symbol decisions according to modulation type.

    Supported modulation types:
        BPSK
        QPSK

    Parameters
    ----------
    symbols : numpy.ndarray
        Complex-valued symbol samples.

    modulation : str
        Modulation type.

    Returns
    -------
    numpy.ndarray
        Decided symbols.
    """

    if not isinstance(modulation, str):
        raise TypeError(
            "modulation must be a string."
        )

    modulation = modulation.upper()

    if modulation == "BPSK":
        return decide_bpsk(symbols)

    if modulation == "QPSK":
        return decide_qpsk(symbols)

    raise ValueError(
        f"Unsupported modulation: {modulation}. "
        "Supported modulations are BPSK and QPSK."
    )


def _validate_symbols(
    symbols: np.ndarray,
) -> None:
    """
    Validate the input symbol array.
    """

    if not isinstance(symbols, np.ndarray):
        raise TypeError(
            "symbols must be a NumPy array."
        )

    if symbols.size == 0:
        raise ValueError(
            "symbols cannot be empty."
        )

    if not np.iscomplexobj(symbols):
        raise TypeError(
            "symbols must contain complex-valued samples."
        )

    if not np.isfinite(symbols.real).all():
        raise ValueError(
            "symbols contain invalid real values."
        )

    if not np.isfinite(symbols.imag).all():
        raise ValueError(
            "symbols contain invalid imaginary values."
        )