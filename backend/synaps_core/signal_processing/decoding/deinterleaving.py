import numpy as np


def deinterleave_bits(
    bits: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """
    Reverse a known bit interleaving operation.

    Parameters
    ----------
    bits : numpy.ndarray
        Interleaved binary bits.

    permutation : numpy.ndarray
        Permutation used during interleaving.

        If the original bits were interleaved using:

            interleaved = original[permutation]

        then this function reconstructs:

            original = interleaved[inverse_permutation]

    Returns
    -------
    numpy.ndarray
        Deinterleaved bits.
    """

    bits = _validate_bits(bits)

    permutation = np.asarray(permutation)

    if permutation.ndim != 1:
        raise ValueError(
            "permutation must be a one-dimensional array."
        )

    if len(permutation) != len(bits):
        raise ValueError(
            "permutation length must match the number of bits."
        )

    if not np.issubdtype(
        permutation.dtype,
        np.integer,
    ):
        raise TypeError(
            "permutation must contain integers."
        )

    expected = np.arange(len(bits))

    if not np.array_equal(
        np.sort(permutation),
        expected,
    ):
        raise ValueError(
            "permutation must contain every index exactly once."
        )

    inverse_permutation = np.argsort(
        permutation
    )

    return bits[
        inverse_permutation
    ]


def identity_deinterleave(
    bits: np.ndarray,
) -> np.ndarray:
    """
    Return bits unchanged.

    Useful when metadata specifies that
    no interleaving was applied.
    """

    bits = _validate_bits(bits)

    return bits.copy()


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