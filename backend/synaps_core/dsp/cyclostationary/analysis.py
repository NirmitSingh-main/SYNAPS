"""
Cyclostationary signal analysis.

This module provides lightweight cyclostationary analysis utilities
for communication signals.

Supported inputs:
    - Real-valued NumPy arrays
    - Complex IQ NumPy arrays

The implementation estimates the spectral correlation function (SCF)
using a frequency-smoothing approach.

Cyclostationary signals contain statistical features that repeat
periodically. In communication systems, these periodicities can be
caused by:
    - Symbol timing
    - Carrier modulation
    - Pulse shaping
    - Frame structure

Main functions:
    analyze_cyclostationarity()
    estimate_cyclic_frequencies()
    calculate_cyclic_autocorrelation()
    get_cyclic_frequencies()
"""


from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_signal(
    signal: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Validate and normalize the input signal.
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
# Basic signal utilities
# ---------------------------------------------------------------------

def _remove_mean(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Remove the DC component from a signal.
    """

    signal = np.asarray(
        signal
    )

    return signal - np.mean(
        signal
    )


def _normalize_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Normalize signal power to approximately one.
    """

    signal = np.asarray(
        signal
    )

    power = float(
        np.mean(
            np.abs(signal) ** 2
        )
    )

    if power <= 0.0:

        raise ValueError(
            "Signal has zero power."
        )

    return signal / np.sqrt(
        power
    )


# ---------------------------------------------------------------------
# Cyclic autocorrelation
# ---------------------------------------------------------------------

def calculate_cyclic_autocorrelation(
    signal: np.ndarray,
    sampling_rate: float,
    cyclic_frequency_hz: float,
    max_lag_samples: Optional[int] = None,
) -> dict[str, Any]:
    """
    Calculate the cyclic autocorrelation for a specified cyclic
    frequency.

    The cyclic autocorrelation is approximately:

        R_alpha(tau) =
            E[
                x(t + tau/2)
                x*(t - tau/2)
                exp(-j 2 pi alpha t)
            ]

    Parameters
    ----------
    signal:
        Real or complex signal.

    sampling_rate:
        Sampling rate in Hertz.

    cyclic_frequency_hz:
        Cyclic frequency alpha in Hertz.

    max_lag_samples:
        Maximum positive lag.

    Returns
    -------
    dict
        Contains lag values, cyclic autocorrelation and magnitude.
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if not np.isfinite(
        cyclic_frequency_hz
    ):
        raise ValueError(
            "cyclic_frequency_hz must be finite."
        )

    number_of_samples = (
        signal.size
    )

    if number_of_samples < 4:

        raise ValueError(
            "At least four samples are required "
            "for cyclostationary analysis."
        )

    if max_lag_samples is None:

        max_lag_samples = min(
            256,
            number_of_samples // 4,
        )

    max_lag_samples = int(
        max_lag_samples
    )

    if max_lag_samples < 0:

        raise ValueError(
            "max_lag_samples cannot be negative."
        )

    max_lag_samples = min(
        max_lag_samples,
        number_of_samples - 1,
    )

    # Time index.
    sample_indices = np.arange(
        number_of_samples,
        dtype=np.float64,
    )

    cyclic_exponential = np.exp(
        -1j
        * 2.0
        * np.pi
        * cyclic_frequency_hz
        * sample_indices
        / sampling_rate
    )

    lags = np.arange(
        -max_lag_samples,
        max_lag_samples + 1,
        dtype=np.int64,
    )

    autocorrelation = np.zeros(
        lags.size,
        dtype=np.complex128,
    )

    for index, lag in enumerate(
        lags
    ):

        if lag >= 0:

            x1 = signal[
                lag:
            ]

            x2 = signal[
                :number_of_samples - lag
            ]

        else:

            positive_lag = -lag

            x1 = signal[
                :number_of_samples
                - positive_lag
            ]

            x2 = signal[
                positive_lag:
            ]

        usable_length = x1.size

        if usable_length == 0:
            continue

        exponential = (
            cyclic_exponential[
                :usable_length
            ]
        )

        autocorrelation[index] = (
            np.mean(
                x1
                * np.conjugate(x2)
                * exponential
            )
        )

    magnitude = np.abs(
        autocorrelation
    )

    return {
        "lags_samples": lags,
        "lags_seconds": (
            lags.astype(np.float64)
            / sampling_rate
        ),
        "cyclic_autocorrelation": (
            autocorrelation
        ),
        "magnitude": magnitude,
        "cyclic_frequency_hz": float(
            cyclic_frequency_hz
        ),
        "sampling_rate_hz": float(
            sampling_rate
        ),
    }


# ---------------------------------------------------------------------
# Spectral correlation estimation
# ---------------------------------------------------------------------

def _calculate_spectral_correlation(
    signal: np.ndarray,
    sampling_rate: float,
    cyclic_frequencies_hz: np.ndarray,
    fft_size: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate spectral correlation for a set of cyclic frequencies.

    This uses the frequency-domain relationship:

        S_alpha(f) ≈ X(f + alpha/2)
                     X*(f - alpha/2)

    The implementation uses interpolation on the FFT spectrum.
    """

    number_of_samples = (
        signal.size
    )

    if fft_size is None:

        fft_size = number_of_samples

    fft_size = int(
        fft_size
    )

    if fft_size < 16:

        raise ValueError(
            "fft_size must be at least 16."
        )

    # Zero-pad or truncate to FFT size.
    if number_of_samples < fft_size:

        padded = np.zeros(
            fft_size,
            dtype=signal.dtype,
        )

        padded[
            :number_of_samples
        ] = signal

        working_signal = padded

    else:

        working_signal = signal[
            :fft_size
        ]

    window = np.hanning(
        fft_size
    )

    windowed_signal = (
        working_signal
        * window
    )

    spectrum = np.fft.fft(
        windowed_signal
    )

    spectrum = np.fft.fftshift(
        spectrum
    )

    frequency_axis = (
        np.fft.fftshift(
            np.fft.fftfreq(
                fft_size,
                d=1.0 / sampling_rate,
            )
        )
    )

    spectral_correlation = np.zeros(
        cyclic_frequencies_hz.size,
        dtype=np.complex128,
    )

    for index, alpha in enumerate(
        cyclic_frequencies_hz
    ):

        positive_frequency = (
            frequency_axis
            + alpha / 2.0
        )

        negative_frequency = (
            frequency_axis
            - alpha / 2.0
        )

        positive_real = np.interp(
            positive_frequency,
            frequency_axis,
            np.real(spectrum),
            left=0.0,
            right=0.0,
        )

        positive_imag = np.interp(
            positive_frequency,
            frequency_axis,
            np.imag(spectrum),
            left=0.0,
            right=0.0,
        )

        negative_real = np.interp(
            negative_frequency,
            frequency_axis,
            np.real(spectrum),
            left=0.0,
            right=0.0,
        )

        negative_imag = np.interp(
            negative_frequency,
            frequency_axis,
            np.imag(spectrum),
            left=0.0,
            right=0.0,
        )

        positive_spectrum = (
            positive_real
            + 1j * positive_imag
        )

        negative_spectrum = (
            negative_real
            + 1j * negative_imag
        )

        correlation = (
            positive_spectrum
            * np.conjugate(
                negative_spectrum
            )
        )

        spectral_correlation[index] = (
            np.mean(
                correlation
            )
        )

    return (
        cyclic_frequencies_hz,
        spectral_correlation,
        frequency_axis,
    )


# ---------------------------------------------------------------------
# Cyclic-frequency estimation
# ---------------------------------------------------------------------

def estimate_cyclic_frequencies(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_cyclic_frequency_hz: float = 1.0,
    maximum_cyclic_frequency_hz: Optional[
        float
    ] = None,
    number_of_frequencies: int = 256,
) -> dict[str, Any]:
    """
    Estimate cyclic-frequency peaks.

    Parameters
    ----------
    signal:
        Real or complex signal.

    sampling_rate:
        Sampling rate in Hertz.

    minimum_cyclic_frequency_hz:
        Minimum cyclic frequency to analyze.

    maximum_cyclic_frequency_hz:
        Maximum cyclic frequency.

        Defaults to Nyquist frequency.

    number_of_frequencies:
        Number of cyclic-frequency points to evaluate.

    Returns
    -------
    dict
        Cyclic-frequency grid and spectral-correlation magnitude.
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if not np.isfinite(
        minimum_cyclic_frequency_hz
    ) or minimum_cyclic_frequency_hz < 0.0:

        raise ValueError(
            "minimum_cyclic_frequency_hz must be "
            "non-negative."
        )

    nyquist = (
        sampling_rate / 2.0
    )

    if maximum_cyclic_frequency_hz is None:

        maximum_cyclic_frequency_hz = (
            nyquist
        )

    else:

        if not np.isfinite(
            maximum_cyclic_frequency_hz
        ):

            raise ValueError(
                "maximum_cyclic_frequency_hz must be finite."
            )

        maximum_cyclic_frequency_hz = min(
            float(
                maximum_cyclic_frequency_hz
            ),
            nyquist,
        )

    if minimum_cyclic_frequency_hz >= (
        maximum_cyclic_frequency_hz
    ):

        raise ValueError(
            "minimum_cyclic_frequency_hz must be "
            "less than maximum_cyclic_frequency_hz."
        )

    number_of_frequencies = int(
        number_of_frequencies
    )

    if number_of_frequencies < 2:

        raise ValueError(
            "number_of_frequencies must be at least 2."
        )

    cyclic_frequencies = np.linspace(
        minimum_cyclic_frequency_hz,
        maximum_cyclic_frequency_hz,
        number_of_frequencies,
    )

    (
        frequencies,
        spectral_correlation,
        _frequency_axis,
    ) = _calculate_spectral_correlation(
        signal,
        sampling_rate,
        cyclic_frequencies,
    )

    magnitude = np.abs(
        spectral_correlation
    )

    if magnitude.size == 0:

        raise ValueError(
            "Unable to calculate cyclic-frequency spectrum."
        )

    peak_index = int(
        np.argmax(
            magnitude
        )
    )

    peak_frequency = float(
        frequencies[
            peak_index
        ]
    )

    peak_magnitude = float(
        magnitude[
            peak_index
        ]
    )

    maximum_magnitude = float(
        np.max(
            magnitude
        )
    )

    if maximum_magnitude > 0.0:

        normalized_magnitude = (
            magnitude
            / maximum_magnitude
        )

    else:

        normalized_magnitude = (
            np.zeros_like(
                magnitude
            )
        )

    return {
        "cyclic_frequencies_hz": (
            frequencies
        ),

        "spectral_correlation": (
            spectral_correlation
        ),

        "spectral_correlation_magnitude": (
            magnitude
        ),

        "normalized_magnitude": (
            normalized_magnitude
        ),

        "peak_cyclic_frequency_hz": (
            peak_frequency
        ),

        "peak_magnitude": (
            peak_magnitude
        ),

        "sampling_rate_hz": float(
            sampling_rate
        ),

        "number_of_samples": int(
            signal.size
        ),

        "number_of_frequencies": int(
            number_of_frequencies
        ),
    }


# ---------------------------------------------------------------------
# Complete analysis
# ---------------------------------------------------------------------

def analyze_cyclostationarity(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_cyclic_frequency_hz: float = 1.0,
    maximum_cyclic_frequency_hz: Optional[
        float
    ] = None,
    number_of_frequencies: int = 256,
    max_lag_samples: Optional[int] = None,
) -> dict[str, Any]:
    """
    Perform complete cyclostationary analysis.

    Parameters
    ----------
    signal:
        Real-valued or complex IQ samples.

    sampling_rate:
        Sampling rate in Hertz.

    minimum_cyclic_frequency_hz:
        Minimum cyclic frequency.

    maximum_cyclic_frequency_hz:
        Maximum cyclic frequency.

    number_of_frequencies:
        Number of cyclic-frequency points.

    max_lag_samples:
        Maximum lag used for cyclic autocorrelation.

    Returns
    -------
    dict
        Complete cyclostationary analysis result.
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    normalized_signal = _normalize_signal(
        signal
    )

    cyclic_analysis = (
        estimate_cyclic_frequencies(
            normalized_signal,
            sampling_rate,
            minimum_cyclic_frequency_hz=(
                minimum_cyclic_frequency_hz
            ),
            maximum_cyclic_frequency_hz=(
                maximum_cyclic_frequency_hz
            ),
            number_of_frequencies=(
                number_of_frequencies
            ),
        )
    )

    peak_cyclic_frequency = float(
        cyclic_analysis[
            "peak_cyclic_frequency_hz"
        ]
    )

    autocorrelation = (
        calculate_cyclic_autocorrelation(
            normalized_signal,
            sampling_rate,
            peak_cyclic_frequency,
            max_lag_samples=max_lag_samples,
        )
    )

    cyclic_magnitude = np.asarray(
        cyclic_analysis[
            "spectral_correlation_magnitude"
        ],
        dtype=np.float64,
    )

    if cyclic_magnitude.size > 1:

        mean_magnitude = float(
            np.mean(
                cyclic_magnitude
            )
        )

        standard_deviation = float(
            np.std(
                cyclic_magnitude
            )
        )

        if standard_deviation > 0.0:

            detection_score = (
                (
                    float(
                        np.max(
                            cyclic_magnitude
                        )
                    )
                    - mean_magnitude
                )
                / standard_deviation
            )

        else:

            detection_score = 0.0

    else:

        detection_score = 0.0

    detection_score = float(
        max(
            0.0,
            detection_score,
        )
    )

    result: dict[str, Any] = {

        "cyclic_frequencies_hz": (
            cyclic_analysis[
                "cyclic_frequencies_hz"
            ]
        ),

        "spectral_correlation": (
            cyclic_analysis[
                "spectral_correlation"
            ]
        ),

        "spectral_correlation_magnitude": (
            cyclic_analysis[
                "spectral_correlation_magnitude"
            ]
        ),

        "normalized_magnitude": (
            cyclic_analysis[
                "normalized_magnitude"
            ]
        ),

        "peak_cyclic_frequency_hz": (
            peak_cyclic_frequency
        ),

        "peak_magnitude": (
            cyclic_analysis[
                "peak_magnitude"
            ]
        ),

        "detection_score": (
            detection_score
        ),

        "cyclic_autocorrelation": (
            autocorrelation[
                "cyclic_autocorrelation"
            ]
        ),

        "cyclic_autocorrelation_magnitude": (
            autocorrelation[
                "magnitude"
            ]
        ),

        "lags_samples": (
            autocorrelation[
                "lags_samples"
            ]
        ),

        "lags_seconds": (
            autocorrelation[
                "lags_seconds"
            ]
        ),

        "sampling_rate_hz": float(
            sampling_rate
        ),

        "number_of_samples": int(
            signal.size
        ),

        "is_complex": bool(
            np.iscomplexobj(signal)
        ),
    }

    return result


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------

def get_cyclic_frequencies(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_cyclic_frequency_hz: float = 1.0,
    maximum_cyclic_frequency_hz: Optional[
        float
    ] = None,
    number_of_frequencies: int = 256,
) -> np.ndarray:
    """
    Return the evaluated cyclic-frequency grid.
    """

    result = estimate_cyclic_frequencies(
        signal,
        sampling_rate,
        minimum_cyclic_frequency_hz=(
            minimum_cyclic_frequency_hz
        ),
        maximum_cyclic_frequency_hz=(
            maximum_cyclic_frequency_hz
        ),
        number_of_frequencies=(
            number_of_frequencies
        ),
    )

    return np.asarray(
        result[
            "cyclic_frequencies_hz"
        ],
        dtype=np.float64,
    )