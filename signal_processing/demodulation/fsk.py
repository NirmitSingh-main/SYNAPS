import numpy as np


def demodulate_fsk(signal, samples_per_symbol):
    """
    Demodulate binary Frequency Shift Keying.

    The bit is determined from the instantaneous
    frequency of each symbol.

    Returns:
        NumPy array of recovered bits.
    """

    signal = np.asarray(signal)

    # Calculate phase
    phase = np.unwrap(np.angle(signal))

    # Calculate instantaneous frequency (padded to maintain full length)
    frequency = np.pad(np.diff(phase), (0, 1), mode="edge")

    # One frequency estimate per symbol
    symbol_frequencies = []

    for start in range(
        0,
        len(frequency),
        samples_per_symbol
    ):
        symbol = frequency[
            start:start + samples_per_symbol
        ]

        if len(symbol) == samples_per_symbol:
            symbol_frequencies.append(
                np.mean(symbol)
            )

    symbol_frequencies = np.asarray(
        symbol_frequencies
    )

    # For binary FSK:
    # lower frequency -> 0
    # higher frequency -> 1
    threshold = (
        np.min(symbol_frequencies)
        + np.max(symbol_frequencies)
    ) / 2

    bits = (
        symbol_frequencies > threshold
    ).astype(np.uint8)

    return bits