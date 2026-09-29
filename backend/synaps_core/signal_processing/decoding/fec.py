import numpy as np


SUPPORTED_FEC = {
    "NONE",
    "NO_FEC",
}


def decode_fec(
    bits: np.ndarray,
    fec_type: str = "NONE",
) -> np.ndarray:
    """
    Apply forward-error-correction decoding.

    Currently supported:
        NONE
        NO_FEC

    Parameters
    ----------
    bits : numpy.ndarray
        Received binary bits.

    fec_type : str
        FEC scheme specified by the signal metadata.

    Returns
    -------
    numpy.ndarray
        Decoded bits.

    Notes
    -----
    Actual FEC algorithms such as convolutional,
    Reed-Solomon, or LDPC should be added here once
    the exact coding schemes used by the dataset
    are defined.
    """

    bits = _validate_bits(bits)

    if not isinstance(fec_type, str):
        raise TypeError(
            "fec_type must be a string."
        )

    fec_type = fec_type.upper().strip()

    if fec_type not in SUPPORTED_FEC:
        raise ValueError(
            f"Unsupported FEC type: {fec_type}. "
            f"Currently supported types are: "
            f"{', '.join(sorted(SUPPORTED_FEC))}"
        )

    # No FEC means the received bits are already
    # the decoded bits.
    return bits.copy()


def has_fec(fec_type: str) -> bool:
    """
    Check whether a signal uses a supported FEC scheme.

    Returns
    -------
    bool
        True if an actual FEC scheme is specified.
    """

    if not isinstance(fec_type, str):
        raise TypeError(
            "fec_type must be a string."
        )

    normalized = fec_type.upper().strip()

    return normalized not in {
        "NONE",
        "NO_FEC",
        "",
    }


def _validate_bits(
    bits: np.ndarray,
) -> np.ndarray:
    """
    Validate a binary bit array.
    """

    if not isinstance(bits, np.ndarray):
        raise TypeError(
            "bits must be a NumPy array."
        )

    if bits.size == 0:
        raise ValueError(
            "bits cannot be empty."
        )

    bits = bits.reshape(-1)

    if not np.all(
        (bits == 0) | (bits == 1)
    ):
        raise ValueError(
            "bits must contain only 0 and 1."
        )

    return bits.astype(np.uint8)