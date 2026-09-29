import numpy as np


def estimate_timing_offset(iq, samples_per_symbol):
    """
    Estimate the best symbol sampling position.

    Tests every possible sample position within one symbol
    period and selects the position with the strongest
    average signal magnitude.

    Parameters:
        iq: Complex IQ samples
        samples_per_symbol: Samples per symbol

    Returns:
        Best timing offset
    """

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be greater than zero")

    best_offset = 0
    best_metric = -np.inf

    for offset in range(samples_per_symbol):

        samples = iq[offset::samples_per_symbol]

        if len(samples) == 0:
            continue

        # Measure how concentrated/strong the sampled symbols are
        metric = np.mean(np.abs(samples) ** 2)

        if metric > best_metric:
            best_metric = metric
            best_offset = offset

    return best_offset


def sample_symbols(iq, samples_per_symbol, timing_offset=None):
    """
    Extract one sample per symbol using the estimated timing offset.

    Parameters:
        iq: Complex IQ samples
        samples_per_symbol: Samples per symbol
        timing_offset: Optional manually specified offset.
                       If None, estimate automatically.

    Returns:
        Symbol-spaced IQ samples
    """

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be greater than zero")

    if timing_offset is None:
        timing_offset = estimate_timing_offset(
            iq,
            samples_per_symbol
        )

    if timing_offset < 0 or timing_offset >= samples_per_symbol:
        raise ValueError(
            "timing_offset must be between "
            "0 and samples_per_symbol - 1"
        )

    symbols = iq[timing_offset::samples_per_symbol]

    return symbols