import numpy as np

from signal_processing.decoding.bit_to_data import (
    bits_to_bytes,
    bits_to_ascii,
    bits_to_utf8,
    bits_to_data,
)

from signal_processing.decoding.deinterleaving import (
    deinterleave_bits,
    identity_deinterleave,
)

from signal_processing.decoding.fec import (
    decode_fec,
    has_fec,
)


# ============================================================
# BIT TO BYTE TEST
# ============================================================

def test_bits_to_bytes():
    """
    Test conversion of 8 bits into one byte.

    01000001 = ASCII 'A'
    """

    bits = np.array(
        [0, 1, 0, 0, 0, 0, 0, 1],
        dtype=np.uint8,
    )

    result = bits_to_bytes(bits)

    assert result == b"A"


# ============================================================
# ASCII TEST
# ============================================================

def test_bits_to_ascii():
    """
    Test conversion of bits into ASCII text.

    ABC:
        A = 01000001
        B = 01000010
        C = 01000011
    """

    bits = np.array(
        [
            0, 1, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 0, 1, 1,
        ],
        dtype=np.uint8,
    )

    result = bits_to_ascii(bits)

    assert result == "ABC"


# ============================================================
# UTF-8 TEST
# ============================================================

def test_bits_to_utf8():
    """
    Test conversion of bits into UTF-8 text.
    """

    bits = np.array(
        [
            0, 1, 0, 0, 0, 0, 0, 1,
        ],
        dtype=np.uint8,
    )

    result = bits_to_utf8(bits)

    assert result == "A"


# ============================================================
# GENERIC BIT TO DATA TEST
# ============================================================

def test_bits_to_data_ascii():
    """
    Test generic bits_to_data() using ASCII.
    """

    bits = np.array(
        [
            0, 1, 0, 0, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 0, 1, 0,
        ],
        dtype=np.uint8,
    )

    result = bits_to_data(
        bits,
        encoding="ASCII",
    )

    assert result == "AB"


# ============================================================
# BINARY DATA TEST
# ============================================================

def test_bits_to_data_binary():
    """
    Test conversion of bits into raw binary data.
    """

    bits = np.array(
        [
            1, 0, 1, 0, 1, 0, 1, 0,
        ],
        dtype=np.uint8,
    )

    result = bits_to_data(
        bits,
        encoding="BINARY",
    )

    assert result == bytes([170])


# ============================================================
# DEINTERLEAVING TEST
# ============================================================

def test_deinterleave_bits():
    """
    Test reversal of a known interleaving permutation.
    """

    original_bits = np.array(
        [0, 1, 1, 0, 1, 0],
        dtype=np.uint8,
    )

    permutation = np.array(
        [2, 0, 4, 1, 5, 3],
        dtype=np.int64,
    )

    interleaved_bits = original_bits[
        permutation
    ]

    recovered_bits = deinterleave_bits(
        interleaved_bits,
        permutation,
    )

    assert np.array_equal(
        recovered_bits,
        original_bits,
    )


# ============================================================
# IDENTITY DEINTERLEAVING TEST
# ============================================================

def test_identity_deinterleave():
    """
    Test the no-interleaving case.
    """

    bits = np.array(
        [0, 1, 1, 0, 1, 1],
        dtype=np.uint8,
    )

    result = identity_deinterleave(bits)

    assert np.array_equal(
        result,
        bits,
    )

    # Make sure a copy was returned.
    assert result is not bits


# ============================================================
# FEC TEST
# ============================================================

def test_decode_fec_none():
    """
    Test FEC decoding when no FEC is used.
    """

    bits = np.array(
        [0, 1, 1, 0, 1, 0],
        dtype=np.uint8,
    )

    result = decode_fec(
        bits,
        fec_type="NONE",
    )

    assert np.array_equal(
        result,
        bits,
    )


# ============================================================
# FEC NO_FEC TEST
# ============================================================

def test_decode_fec_no_fec():
    """
    Test the NO_FEC alias.
    """

    bits = np.array(
        [1, 0, 0, 1, 1, 0],
        dtype=np.uint8,
    )

    result = decode_fec(
        bits,
        fec_type="NO_FEC",
    )

    assert np.array_equal(
        result,
        bits,
    )


# ============================================================
# HAS FEC TEST
# ============================================================

def test_has_fec():
    """
    Test detection of whether an FEC scheme is specified.
    """

    assert has_fec("NONE") is False

    assert has_fec("NO_FEC") is False

    assert has_fec("") is False

    # These are not currently supported for decoding,
    # but the function should recognize that an FEC
    # scheme has been specified.
    assert has_fec("CONVOLUTIONAL") is True
    assert has_fec("LDPC") is True


# ============================================================
# COMPLETE DECODING PIPELINE TEST
# ============================================================

def test_complete_decoding_pipeline():
    """
    Test the complete decoding pipeline:

        Bits
          ↓
        Deinterleaving
          ↓
        FEC
          ↓
        Bytes
          ↓
        ASCII
          ↓
        Recovered message
    """

    # Original message = "HI"

    original_bits = np.array(
        [
            # H = 01001000
            0, 1, 0, 0, 1, 0, 0, 0,

            # I = 01001001
            0, 1, 0, 0, 1, 0, 0, 1,
        ],
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Create a known interleaving
    # --------------------------------------------------------

    permutation = np.array(
        [
            8, 0, 9, 1,
            10, 2, 11, 3,
            12, 4, 13, 5,
            14, 6, 15, 7,
        ],
        dtype=np.int64,
    )

    interleaved_bits = original_bits[
        permutation
    ]

    # --------------------------------------------------------
    # Step 1: Deinterleave
    # --------------------------------------------------------

    deinterleaved_bits = deinterleave_bits(
        interleaved_bits,
        permutation,
    )

    # --------------------------------------------------------
    # Step 2: FEC decoding
    # --------------------------------------------------------

    decoded_bits = decode_fec(
        deinterleaved_bits,
        fec_type="NONE",
    )

    # --------------------------------------------------------
    # Step 3: Convert bits to data
    # --------------------------------------------------------

    recovered_message = bits_to_data(
        decoded_bits,
        encoding="ASCII",
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert np.array_equal(
        deinterleaved_bits,
        original_bits,
    )

    assert np.array_equal(
        decoded_bits,
        original_bits,
    )

    assert recovered_message == "HI"