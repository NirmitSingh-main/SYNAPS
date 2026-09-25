import numpy as np


def estimate_timing_offset(
    iq: np.ndarray,
    samples_per_symbol: float,
    return_fractional: bool = False,
) -> int | float:
    """
    Estimate the optimal symbol sampling position using Oerder-Meyr
    cyclostationary phase recovery and eye-opening energy variance.

    Parameters:
        iq: Complex IQ samples
        samples_per_symbol: Samples per symbol (SPS)
        return_fractional: If True, returns continuous offset in [0, SPS)

    Returns:
        Best timing offset (int if return_fractional=False, else float)
    """

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be greater than zero")

    iq = np.asarray(iq, dtype=np.complex128)
    N = len(iq)
    sps = float(samples_per_symbol)

    if N < 2 * sps:
        return 0.0 if return_fractional else 0

    # 1. Feedforward Oerder-Meyr estimator
    z = np.abs(iq) ** 2
    z = z - np.mean(z)

    n = np.arange(N)
    omega = 2.0 * np.pi / sps
    X = np.sum(z * np.exp(-1j * omega * n))

    if abs(X) > 1e-12:
        raw_phase = np.angle(X)
        tau = - (sps / (2.0 * np.pi)) * raw_phase
        tau = float(tau % sps)
    else:
        tau = 0.0

    # 2. Discrete scan verification (variance of power / eye opening)
    int_sps = max(1, int(round(sps)))
    best_int_offset = 0
    best_metric = -np.inf

    for offset in range(int_sps):
        sub = iq[offset::int_sps]
        if len(sub) < 4:
            continue
        pwr = np.abs(sub) ** 2
        # Variance of power peaks at maximum eye opening (symbol center)
        metric = float(np.var(pwr))
        if metric > best_metric:
            best_metric = metric
            best_int_offset = offset

    if return_fractional:
        return float(tau)

    # Return integer offset closest to Oerder-Meyr phase, or best eye opening
    om_int = int(round(tau)) % int_sps
    return int(om_int if best_metric <= 0.0 else best_int_offset)


def sample_symbols(
    iq: np.ndarray,
    samples_per_symbol: float,
    timing_offset: float | None = None,
    use_fractional: bool = True,
) -> np.ndarray:
    """
    Extract symbol-spaced samples at the optimal symbol-center phase.

    Supports both integer decimation and fractional cubic interpolation.

    Parameters:
        iq: Complex IQ samples
        samples_per_symbol: Samples per symbol
        timing_offset: Optional timing offset. If None, estimated automatically.
        use_fractional: If True and timing_offset is float, uses 4-point cubic interpolation.

    Returns:
        Symbol-spaced IQ samples
    """

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be greater than zero")

    iq = np.asarray(iq, dtype=np.complex128)
    N = len(iq)
    sps = float(samples_per_symbol)

    if timing_offset is None:
        timing_offset = estimate_timing_offset(
            iq,
            sps,
            return_fractional=use_fractional,
        )

    if timing_offset < 0 or timing_offset >= sps:
        timing_offset = float(timing_offset % sps)

    if not use_fractional or isinstance(timing_offset, (int, np.integer)):
        int_off = int(round(timing_offset)) % max(1, int(round(sps)))
        return iq[int_off::int(round(sps))]

    # Fractional cubic Hermite interpolation
    num_symbols = int(np.floor((N - timing_offset - 1) / sps))
    if num_symbols <= 0:
        return iq[::max(1, int(round(sps)))]

    sample_times = timing_offset + np.arange(num_symbols) * sps
    symbols = np.zeros(num_symbols, dtype=np.complex128)

    for i, t in enumerate(sample_times):
        idx = int(np.floor(t))
        frac = t - idx
        if 1 <= idx < N - 2:
            y0, y1, y2, y3 = iq[idx-1], iq[idx], iq[idx+1], iq[idx+2]
            c0 = y1
            c1 = 0.5 * (y2 - y0)
            c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
            c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)
            symbols[i] = ((c3 * frac + c2) * frac + c1) * frac + c0
        elif 0 <= idx < N:
            symbols[i] = iq[idx]

    return symbols