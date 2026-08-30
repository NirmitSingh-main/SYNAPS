import numpy as np


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert a sequence of bits into bytes.

    Bits are interpreted MSB-first.

    Example:
        [0, 1, 0, 0, 0, 0, 0, 1]
        -> b"A"

    Parameters
    ----------
    bits : numpy.ndarray
        Array containing only 0 and 1.

    Returns
    -------
    bytes
        Converted byte sequence.
    """

    bits = _validate_bits(bits)

    # A complete byte requires 8 bits.
    if len(bits) % 8 != 0:
        raise ValueError(
            "Number of bits must be a multiple of 8."
        )

    if len(bits) == 0:
        return b""

    # Group bits into groups of 8.
    bit_groups = bits.reshape(-1, 8)

    # Convert each group to one byte.
    weights = np.array(
        [128, 64, 32, 16, 8, 4, 2, 1],
        dtype=np.uint16,
    )

    byte_values = (
        bit_groups.astype(np.uint16) * weights
    ).sum(axis=1)

    return bytes(byte_values.tolist())


def bits_to_ascii(bits: np.ndarray) -> str:
    """
    Convert bits directly into an ASCII string.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary bit array.

    Returns
    -------
    str
        Decoded ASCII text.
    """

    data = bits_to_bytes(bits)

    try:
        return data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Recovered bits do not represent valid ASCII data."
        ) from exc


def bits_to_utf8(bits: np.ndarray) -> str:
    """
    Convert bits directly into a UTF-8 string.

    Parameters
    ----------
    bits : numpy.ndarray
        Binary bit array.

    Returns
    -------
    str
        Decoded UTF-8 text.
    """

    data = bits_to_bytes(bits)

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Recovered bits do not represent valid UTF-8 data."
        ) from exc


def bits_to_data(
    bits: np.ndarray,
    encoding: str = "ASCII",
):
    """
    Convert recovered bits into application-level data.

    Supported encodings:

        ASCII
        UTF-8
        BINARY / RAW

    Parameters
    ----------
    bits : numpy.ndarray
        Recovered binary bits.

    encoding : str
        Encoding used by the original message.

    Returns
    -------
    str or bytes
        Decoded data.
    """

    if not isinstance(encoding, str):
        raise TypeError(
            "encoding must be a string."
        )

    encoding = encoding.upper().replace("_", "-")

    if encoding == "ASCII":
        return bits_to_ascii(bits)

    if encoding in {"UTF-8", "UTF8"}:
        return bits_to_utf8(bits)

    if encoding in {"BINARY", "RAW", "BYTES"}:
        return bits_to_bytes(bits)

    raise ValueError(
        f"Unsupported encoding: {encoding}. "
        "Supported encodings are ASCII, UTF-8, "
        "and BINARY."
    )


def _validate_bits(
    bits: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize a bit array.

    Returns
    -------
    numpy.ndarray
        Validated uint8 bit array.
    """

    if not isinstance(bits, np.ndarray):
        raise TypeError(
            "bits must be a NumPy array."
        )

    if bits.size == 0:
        raise ValueError(
            "bits cannot be empty."
        )

    # Flatten in case a column/vector array is supplied.
    bits = bits.reshape(-1)

    if not np.isfinite(bits).all():
        raise ValueError(
            "bits contain invalid values."
        )

    if not np.all(
        (bits == 0) | (bits == 1)
    ):
        raise ValueError(
            "bits must contain only 0 and 1."
        )

    return bits.astype(np.uint8)