import numpy as np

from signal_processing.symbol_processing.symbol_detector import (
    detect_symbols,
)

from signal_processing.symbol_processing.decision import (
    decide_bpsk,
    decide_qpsk,
    decide,
)

from signal_processing.symbol_processing.symbol_to_bits import (
    bpsk_symbols_to_bits,
    qpsk_symbols_to_bits,
    symbols_to_bits,
)


# ============================================================
# SYMBOL DETECTOR TESTS
# ============================================================

def test_symbol_detector():
    """
    Test extraction of one sample per symbol.
    """

    iq = np.array(
        [
            1 + 1j,
            1 + 1j,
            1 + 1j,
            -1 - 1j,
            -1 - 1j,
            -1 - 1j,
            1 + 1j,
            1 + 1j,
            1 + 1j,
        ],
        dtype=np.complex64,
    )

    symbols = detect_symbols(
        iq,
        samples_per_symbol=3,
        timing_offset=0,
    )

    assert len(symbols) == 3

    assert np.allclose(
        symbols,
        [
            1 + 1j,
            -1 - 1j,
            1 + 1j,
        ],
    )


# ============================================================
# BPSK DECISION TEST
# ============================================================

def test_bpsk_decision():
    """
    Test BPSK hard symbol decisions.
    """

    symbols = np.array(
        [
            -0.8 + 0.1j,
            0.7 + 0.2j,
            -0.5 - 0.3j,
            1.2 - 0.1j,
        ],
        dtype=np.complex64,
    )

    decisions = decide_bpsk(symbols)

    expected = np.array(
        [
            -1.0,
            1.0,
            -1.0,
            1.0,
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        decisions,
        expected,
    )


# ============================================================
# QPSK DECISION TEST
# ============================================================

def test_qpsk_decision():
    """
    Test QPSK quadrant decisions.
    """

    symbols = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex64,
    )

    decisions = decide_qpsk(symbols)

    expected = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex64,
    )

    assert np.allclose(
        decisions,
        expected,
    )


# ============================================================
# GENERIC DECISION TEST
# ============================================================

def test_generic_decision():
    """
    Test the generic decide() function.
    """

    bpsk_symbols = np.array(
        [
            -1 + 0j,
            1 + 0j,
        ],
        dtype=np.complex64,
    )

    bpsk_result = decide(
        bpsk_symbols,
        "BPSK",
    )

    assert np.allclose(
        bpsk_result,
        [-1.0, 1.0],
    )

    qpsk_symbols = np.array(
        [
            1 + 1j,
            -1 - 1j,
        ],
        dtype=np.complex64,
    )

    qpsk_result = decide(
        qpsk_symbols,
        "QPSK",
    )

    assert np.allclose(
        qpsk_result,
        [
            1 + 1j,
            -1 - 1j,
        ],
    )


# ============================================================
# BPSK SYMBOL → BITS
# ============================================================

def test_bpsk_symbols_to_bits():
    """
    Test BPSK symbol-to-bit conversion.

    Mapping:
        -1 → 0
        +1 → 1
    """

    symbols = np.array(
        [
            -1 + 0j,
            1 + 0j,
            1 + 0j,
            -1 + 0j,
        ],
        dtype=np.complex64,
    )

    bits = bpsk_symbols_to_bits(
        symbols
    )

    expected = np.array(
        [0, 1, 1, 0],
        dtype=np.uint8,
    )

    assert np.array_equal(
        bits,
        expected,
    )


# ============================================================
# QPSK SYMBOL → BITS
# ============================================================

def test_qpsk_symbols_to_bits():
    """
    Test QPSK symbol-to-bit conversion.
    """

    symbols = np.array(
        [
            1 + 1j,
            -1 + 1j,
            -1 - 1j,
            1 - 1j,
        ],
        dtype=np.complex64,
    )

    bits = qpsk_symbols_to_bits(
        symbols
    )

    expected = np.array(
        [
            0, 0,
            0, 1,
            1, 1,
            1, 0,
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(
        bits,
        expected,
    )


# ============================================================
# GENERIC SYMBOL → BITS
# ============================================================

def test_generic_symbols_to_bits():
    """
    Test the generic symbols_to_bits() function.
    """

    symbols = np.array(
        [
            -1 + 0j,
            1 + 0j,
            -1 + 0j,
        ],
        dtype=np.complex64,
    )

    bits = symbols_to_bits(
        symbols,
        "BPSK",
    )

    expected = np.array(
        [0, 1, 0],
        dtype=np.uint8,
    )

    assert np.array_equal(
        bits,
        expected,
    )


# ============================================================
# COMPLETE SYMBOL PROCESSING PIPELINE
# ============================================================

def test_complete_symbol_processing():
    """
    Test the complete symbol-processing pipeline:

        IQ samples
             ↓
        Symbol detection
             ↓
        BPSK decision
             ↓
        Symbol → bits
    """

    # Four BPSK symbols.
    original_bits = np.array(
        [0, 1, 1, 0],
        dtype=np.uint8,
    )

    # BPSK mapping:
    # 0 → -1
    # 1 → +1
    bpsk_symbols = np.where(
        original_bits == 0,
        -1.0,
        1.0,
    )

    # Each symbol contains 4 samples.
    iq = np.repeat(
        bpsk_symbols,
        4,
    ).astype(np.complex64)

    # --------------------------------------------------------
    # Step 1: Detect symbols
    # --------------------------------------------------------

    detected_symbols = detect_symbols(
        iq,
        samples_per_symbol=4,
        timing_offset=0,
    )

    # --------------------------------------------------------
    # Step 2: Make symbol decisions
    # --------------------------------------------------------

    decided_symbols = decide_bpsk(
        detected_symbols
    )

    # --------------------------------------------------------
    # Step 3: Convert symbols to bits
    # --------------------------------------------------------

    recovered_bits = bpsk_symbols_to_bits(
        decided_symbols
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert len(detected_symbols) == 4

    assert np.allclose(
        decided_symbols,
        bpsk_symbols,
    )

    assert np.array_equal(
        recovered_bits,
        original_bits,
    )