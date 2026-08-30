"""
Symbol-rate estimation for digital communication signals.

Supported inputs:
    - Real-valued NumPy arrays
    - Complex IQ NumPy arrays

The estimator uses the spectral content of the signal envelope.
For digitally modulated signals, the magnitude-squared signal often
contains a strong component at the symbol rate and/or harmonics of
the symbol rate.

Main functions:
    estimate_symbol_rate()
    estimate_symbol_rate_from_signal()
    get_symbol_rate()
"""


from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _validate_signal(
    signal: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Validate the input signal and sampling rate.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError(
            "Signal cannot be empty."
        )

    if signal.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not np.isfinite(
        sampling_rate
    ) or sampling_rate <= 0.0:
        raise ValueError(
            "sampling_rate must be greater than zero."
        )

    if not np.all(
        np.isfinite(signal)
    ):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if np.iscomplexobj(signal):

        return np.asarray(
            signal,
            dtype=np.complex128,
        )

    return np.asarray(
        signal,
        dtype=np.float64,
    )


# ---------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------

def _prepare_timing_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convert the input into a real-valued timing-sensitive signal.

    For complex IQ:
        |x[n]|^2 is used.

    For real-valued signals:
        x[n]^2 is used.

    The mean is removed afterward.
    """

    signal = np.asarray(
        signal
    )

    if np.iscomplexobj(signal):

        timing_signal = (
            np.abs(signal) ** 2
        )

    else:

        timing_signal = (
            np.asarray(
                signal,
                dtype=np.float64,
            )
            ** 2
        )

    timing_signal = np.asarray(
        timing_signal,
        dtype=np.float64,
    )

    timing_signal -= np.mean(
        timing_signal
    )

    return timing_signal


# ---------------------------------------------------------------------
# FFT-based symbol-rate spectrum
# ---------------------------------------------------------------------

def _calculate_timing_spectrum(
    timing_signal: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the one-sided FFT spectrum of the timing signal.
    """

    number_of_samples = (
        timing_signal.size
    )

    if number_of_samples < 4:
        raise ValueError(
            "At least four samples are required "
            "for symbol-rate estimation."
        )

    window = np.hanning(
        number_of_samples
    )

    windowed_signal = (
        timing_signal * window
    )

    spectrum = np.fft.rfft(
        windowed_signal
    )

    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )

    magnitude = np.abs(
        spectrum
    )

    # Remove DC.
    if magnitude.size > 0:
        magnitude[0] = 0.0

    return (
        frequencies,
        magnitude,
    )


# ---------------------------------------------------------------------
# Candidate symbol rates
# ---------------------------------------------------------------------

def _find_symbol_rate_candidates(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    maximum_candidates: int = 10,
) -> list[tuple[float, float]]:
    """
    Find strong spectral peaks that may correspond to symbol rate
    or symbol-rate harmonics.
    """

    mask = (
        (frequencies >= minimum_frequency_hz)
        & (
            frequencies
            <= maximum_frequency_hz
        )
    )

    if not np.any(mask):
        return []

    candidate_frequencies = (
        frequencies[mask]
    )

    candidate_magnitude = (
        magnitude[mask]
    )

    if candidate_magnitude.size == 0:
        return []

    maximum_magnitude = float(
        np.max(candidate_magnitude)
    )

    if maximum_magnitude <= 0.0:
        return []

    # Peaks need some separation so adjacent FFT bins are not
    # interpreted as separate symbol-rate candidates.
    distance = max(
        1,
        candidate_magnitude.size // 200,
    )

    peaks, properties = find_peaks(
        candidate_magnitude,
        height=(
            maximum_magnitude * 0.05
        ),
        distance=distance,
    )

    if peaks.size == 0:
        # Fall back to the strongest bin.
        strongest = int(
            np.argmax(
                candidate_magnitude
            )
        )

        return [
            (
                float(
                    candidate_frequencies[
                        strongest
                    ]
                ),
                float(
                    candidate_magnitude[
                        strongest
                    ]
                ),
            )
        ]

    heights = properties.get(
        "peak_heights",
        candidate_magnitude[peaks],
    )

    order = np.argsort(
        heights
    )[::-1]

    candidates: list[
        tuple[float, float]
    ] = []

    for index in order[
        :maximum_candidates
    ]:

        peak_index = int(
            peaks[index]
        )

        candidates.append(
            (
                float(
                    candidate_frequencies[
                        peak_index
                    ]
                ),
                float(
                    candidate_magnitude[
                        peak_index
                    ]
                ),
            )
        )

    return candidates


# ---------------------------------------------------------------------
# Harmonic correction
# ---------------------------------------------------------------------

def _select_fundamental(
    candidates: list[tuple[float, float]],
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    tolerance: float = 0.03,
) -> tuple[float, float]:
    """
    Select the most plausible fundamental symbol rate.

    A strong peak can sometimes occur at 2x, 3x, etc. of the actual
    symbol rate. This routine checks whether lower-frequency
    candidates have harmonically related peaks.

    Returns:
        selected symbol rate
        confidence score
    """

    if not candidates:
        raise ValueError(
            "Unable to find a symbol-rate candidate."
        )

    # Normalize candidate strengths.
    maximum_strength = max(
        strength
        for _, strength in candidates
    )

    if maximum_strength <= 0.0:
        return (
            candidates[0][0],
            0.0,
        )

    scored_candidates: list[
        tuple[float, float]
    ] = []

    candidate_rates = [
        frequency
        for frequency, _ in candidates
    ]

    for rate, strength in candidates:

        score = (
            strength
            / maximum_strength
        )

        # If this candidate has lower harmonics, prefer the
        # lower frequency when those harmonics are present.
        for divisor in (2, 3, 4):

            possible_fundamental = (
                rate / divisor
            )

            if possible_fundamental <= 0.0:
                continue

            frequency_difference = (
                np.abs(
                    frequencies
                    - possible_fundamental
                )
            )

            nearest_index = int(
                np.argmin(
                    frequency_difference
                )
            )

            nearest_frequency = float(
                frequencies[
                    nearest_index
                ]
            )

            relative_error = (
                abs(
                    nearest_frequency
                    - possible_fundamental
                )
                / possible_fundamental
            )

            if relative_error <= tolerance:

                harmonic_strength = (
                    magnitude[
                        nearest_index
                    ]
                )

                if harmonic_strength > (
                    0.10 * strength
                ):

                    # Give a small preference to the lower
                    # fundamental if a harmonic is confirmed.
                    score = max(
                        score,
                        0.85
                        * (
                            float(
                                harmonic_strength
                            )
                            / maximum_strength
                        ),
                    )

        scored_candidates.append(
            (
                rate,
                score,
            )
        )

    scored_candidates.sort(
        key=lambda item: (
            item[1],
            -item[0],
        ),
        reverse=True,
    )

    selected_rate = (
        scored_candidates[0][0]
    )

    selected_score = (
        scored_candidates[0][1]
    )

    return (
        float(selected_rate),
        float(
            np.clip(
                selected_score,
                0.0,
                1.0,
            )
        ),
    )


# ---------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------

def estimate_symbol_rate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float = 1.0,
    maximum_symbol_rate_hz: Optional[
        float
    ] = None,
) -> dict[str, Any]:
    """
    Estimate the symbol rate of a digital communication signal.

    Parameters
    ----------
    signal:
        One-dimensional real-valued or complex IQ signal.

    sampling_rate:
        Sampling frequency in Hertz.

    minimum_symbol_rate_hz:
        Minimum symbol rate to consider.

    maximum_symbol_rate_hz:
        Maximum symbol rate to consider.

        If None, Nyquist frequency is used.

    Returns
    -------
    dict
        Contains:

            symbol_rate_hz
            samples_per_symbol
            confidence
            timing_peak_frequency_hz
            timing_peak_magnitude
            frequency_resolution_hz
            number_of_samples
            sampling_rate_hz
            is_complex
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if not np.isfinite(
        minimum_symbol_rate_hz
    ) or minimum_symbol_rate_hz <= 0.0:

        raise ValueError(
            "minimum_symbol_rate_hz must be greater than zero."
        )

    nyquist_frequency = (
        sampling_rate / 2.0
    )

    if maximum_symbol_rate_hz is None:

        maximum_symbol_rate_hz = (
            nyquist_frequency
        )

    else:

        if not np.isfinite(
            maximum_symbol_rate_hz
        ):

            raise ValueError(
                "maximum_symbol_rate_hz must be finite."
            )

        if maximum_symbol_rate_hz <= 0.0:

            raise ValueError(
                "maximum_symbol_rate_hz must be greater than zero."
            )

        maximum_symbol_rate_hz = min(
            float(
                maximum_symbol_rate_hz
            ),
            nyquist_frequency,
        )

    if minimum_symbol_rate_hz >= (
        maximum_symbol_rate_hz
    ):

        raise ValueError(
            "minimum_symbol_rate_hz must be "
            "less than maximum_symbol_rate_hz."
        )

    timing_signal = (
        _prepare_timing_signal(
            signal
        )
    )

    (
        frequencies,
        magnitude,
    ) = _calculate_timing_spectrum(
        timing_signal,
        sampling_rate,
    )

    frequency_resolution = float(
        sampling_rate
        / signal.size
    )

    candidates = (
        _find_symbol_rate_candidates(
            frequencies,
            magnitude,
            float(
                minimum_symbol_rate_hz
            ),
            float(
                maximum_symbol_rate_hz
            ),
        )
    )

    if not candidates:

        raise ValueError(
            "Unable to detect a symbol-rate spectral peak."
        )

    (
        symbol_rate,
        confidence,
    ) = _select_fundamental(
        candidates,
        magnitude,
        frequencies,
    )

    # Locate the actual FFT peak nearest to the selected rate.
    nearest_index = int(
        np.argmin(
            np.abs(
                frequencies
                - symbol_rate
            )
        )
    )

    peak_frequency = float(
        frequencies[
            nearest_index
        ]
    )

    peak_magnitude = float(
        magnitude[
            nearest_index
        ]
    )

    samples_per_symbol = float(
        sampling_rate
        / symbol_rate
    )

    result: dict[str, Any] = {
        "symbol_rate_hz": float(
            symbol_rate
        ),

        "samples_per_symbol": (
            samples_per_symbol
        ),

        "confidence": float(
            confidence
        ),

        "timing_peak_frequency_hz": (
            peak_frequency
        ),

        "timing_peak_magnitude": (
            peak_magnitude
        ),

        "frequency_resolution_hz": (
            frequency_resolution
        ),

        "number_of_samples": int(
            signal.size
        ),

        "sampling_rate_hz": float(
            sampling_rate
        ),

        "is_complex": bool(
            np.iscomplexobj(signal)
        ),
    }

    return result


# ---------------------------------------------------------------------
# Compatibility wrapper
# ---------------------------------------------------------------------

def estimate_symbol_rate_from_signal(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float = 1.0,
    maximum_symbol_rate_hz: Optional[
        float
    ] = None,
) -> dict[str, Any]:
    """
    Compatibility wrapper for estimate_symbol_rate().
    """

    return estimate_symbol_rate(
        signal,
        sampling_rate,
        minimum_symbol_rate_hz=(
            minimum_symbol_rate_hz
        ),
        maximum_symbol_rate_hz=(
            maximum_symbol_rate_hz
        ),
    )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def get_symbol_rate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float = 1.0,
    maximum_symbol_rate_hz: Optional[
        float
    ] = None,
) -> float:
    """
    Return only the estimated symbol rate in Hertz.
    """

    result = estimate_symbol_rate(
        signal,
        sampling_rate,
        minimum_symbol_rate_hz=(
            minimum_symbol_rate_hz
        ),
        maximum_symbol_rate_hz=(
            maximum_symbol_rate_hz
        ),
    )

    return float(
        result["symbol_rate_hz"]
    )