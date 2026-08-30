"""
Power Spectral Density (PSD) estimation utilities.

This module calculates the Power Spectral Density of communication
signals.

Supported input:
    - Real-valued NumPy arrays
    - Complex IQ NumPy arrays

Main functions:
    estimate_psd()
    find_peak_frequency()
    calculate_total_power()

The PSD is calculated using Welch's method, which provides a more
stable spectral estimate than a single FFT periodogram.
"""

from typing import Any, Optional

import numpy as np
from scipy.signal import welch


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

    if not np.all(
        np.isfinite(signal)
    ):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if not np.isfinite(
        sampling_rate
    ):
        raise ValueError(
            "Sampling rate must be finite."
        )

    if sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    return signal


# ---------------------------------------------------------------------
# PSD calculation
# ---------------------------------------------------------------------

def estimate_psd(
    signal: np.ndarray,
    sampling_rate: float,
    segment_length: Optional[int] = None,
    overlap: float = 0.5,
) -> dict[str, Any]:
    """
    Estimate the Power Spectral Density of a signal.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    sampling_rate:
        Sampling rate in Hertz.

    segment_length:
        Number of samples in each Welch segment.

        If None, a suitable value is selected automatically.

    overlap:
        Fraction of each segment that overlaps with the next segment.

        Must be between 0 and less than 1.

        Default:
            0.5

    Returns
    -------
    dict
        Contains:

            frequencies_hz
            psd
            sampling_rate_hz
            segment_length
            overlap
            total_power
            peak_frequency_hz
            peak_psd
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if not np.isfinite(
        overlap
    ):
        raise ValueError(
            "overlap must be finite."
        )

    if (
        overlap < 0.0
        or overlap >= 1.0
    ):
        raise ValueError(
            "overlap must be between 0 and 1."
        )

    number_of_samples = int(
        signal.size
    )

    # -------------------------------------------------------------
    # Select segment length.
    # -------------------------------------------------------------

    if segment_length is None:

        if number_of_samples < 256:

            segment_length = number_of_samples

        else:

            segment_length = min(
                1024,
                number_of_samples,
            )

    else:

        segment_length = int(
            segment_length
        )

        if segment_length <= 0:
            raise ValueError(
                "segment_length must be greater than zero."
            )

        segment_length = min(
            segment_length,
            number_of_samples,
        )

    if segment_length < 2:
        raise ValueError(
            "Signal must contain at least two samples."
        )

    # -------------------------------------------------------------
    # Calculate number of overlapping samples.
    # -------------------------------------------------------------

    overlap_samples = int(
        segment_length * overlap
    )

    if overlap_samples >= segment_length:
        overlap_samples = (
            segment_length - 1
        )

    # -------------------------------------------------------------
    # Calculate PSD.
    # -------------------------------------------------------------

    if np.iscomplexobj(signal):

        frequencies, psd = welch(
            signal,
            fs=sampling_rate,
            window="hann",
            nperseg=segment_length,
            noverlap=overlap_samples,
            detrend="constant",
            return_onesided=False,
            scaling="density",
        )

        frequencies = np.fft.fftshift(
            frequencies
        )

        psd = np.fft.fftshift(
            psd
        )

    else:

        frequencies, psd = welch(
            signal,
            fs=sampling_rate,
            window="hann",
            nperseg=segment_length,
            noverlap=overlap_samples,
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    psd = np.asarray(
        np.real(psd),
        dtype=np.float64,
    )

    # -------------------------------------------------------------
    # Validate PSD result.
    # -------------------------------------------------------------

    if frequencies.size == 0:
        raise ValueError(
            "Unable to calculate PSD."
        )

    if psd.size == 0:
        raise ValueError(
            "PSD calculation returned no data."
        )

    # Numerical noise can occasionally create tiny negative values.
    psd = np.maximum(
        psd,
        0.0,
    )

    # -------------------------------------------------------------
    # Calculate total power.
    #
    # Integrating PSD over frequency gives signal power.
    # -------------------------------------------------------------

    total_power = float(
        np.trapezoid(
            psd,
            frequencies,
        )
    )

    # NumPy versions before 2.0 do not have trapezoid.
    # Fall back to trapz if necessary.
    if not np.isfinite(
        total_power
    ):
       if frequencies.size < 2:
            total_power = 0.0
       else:
            total_power = float(
        np.sum(
            0.5
            * (
                psd[:-1]
                + psd[1:]
            )
            * np.diff(frequencies)
        )
    )

    # -------------------------------------------------------------
    # Find peak frequency.
    # -------------------------------------------------------------

    peak_index = int(
        np.argmax(psd)
    )

    peak_frequency = float(
        frequencies[peak_index]
    )

    peak_psd = float(
        psd[peak_index]
    )

    result: dict[str, Any] = {
        "frequencies_hz": frequencies,
        "psd": psd,
        "sampling_rate_hz": float(
            sampling_rate
        ),
        "segment_length": int(
            segment_length
        ),
        "overlap": float(
            overlap
        ),
        "overlap_samples": int(
            overlap_samples
        ),
        "total_power": float(
            total_power
        ),
        "peak_frequency_hz": float(
            peak_frequency
        ),
        "peak_psd": float(
            peak_psd
        ),
    }

    return result


# ---------------------------------------------------------------------
# Peak frequency
# ---------------------------------------------------------------------

def find_peak_frequency(
    frequencies: np.ndarray,
    psd: np.ndarray,
) -> float:
    """
    Find the frequency corresponding to the maximum PSD.

    Parameters
    ----------
    frequencies:
        Frequency values in Hertz.

    psd:
        Power spectral density values.

    Returns
    -------
    float
        Peak frequency in Hertz.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    psd = np.asarray(
        psd,
        dtype=np.float64,
    )

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if psd.ndim != 1:
        raise ValueError(
            "psd must be one-dimensional."
        )

    if frequencies.size == 0:
        raise ValueError(
            "frequencies cannot be empty."
        )

    if psd.size == 0:
        raise ValueError(
            "psd cannot be empty."
        )

    if frequencies.size != psd.size:
        raise ValueError(
            "frequencies and psd must have the same length."
        )

    if not np.all(
        np.isfinite(frequencies)
    ):
        raise ValueError(
            "frequencies contain NaN or infinite values."
        )

    if not np.all(
        np.isfinite(psd)
    ):
        raise ValueError(
            "psd contains NaN or infinite values."
        )

    peak_index = int(
        np.argmax(psd)
    )

    return float(
        frequencies[peak_index]
    )


# ---------------------------------------------------------------------
# Total power
# ---------------------------------------------------------------------

def calculate_total_power(
    frequencies: np.ndarray,
    psd: np.ndarray,
) -> float:
    """
    Calculate total signal power from a PSD.

    Parameters
    ----------
    frequencies:
        Frequency values in Hertz.

    psd:
        Power spectral density values.

    Returns
    -------
    float
        Total signal power.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    psd = np.asarray(
        psd,
        dtype=np.float64,
    )

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if psd.ndim != 1:
        raise ValueError(
            "psd must be one-dimensional."
        )

    if frequencies.size == 0:
        raise ValueError(
            "frequencies cannot be empty."
        )

    if psd.size == 0:
        raise ValueError(
            "psd cannot be empty."
        )

    if frequencies.size != psd.size:
        raise ValueError(
            "frequencies and psd must have the same length."
        )

    if not np.all(
        np.isfinite(frequencies)
    ):
        raise ValueError(
            "frequencies contain NaN or infinite values."
        )

    if not np.all(
        np.isfinite(psd)
    ):
        raise ValueError(
            "psd contains NaN or infinite values."
        )

    if np.any(
        psd < 0.0
    ):
        raise ValueError(
            "PSD values cannot be negative."
        )

    # Ensure frequencies are ordered.
    sort_indices = np.argsort(
        frequencies
    )

    sorted_frequencies = frequencies[
        sort_indices
    ]

    sorted_psd = psd[
        sort_indices
    ]

    if sorted_frequencies.size < 2:
        total_power = 0.0
    else:
        total_power = float(
            np.sum(
            0.5
            * (
                sorted_psd[:-1]
                + sorted_psd[1:]
            )
            * np.diff(sorted_frequencies)
        )
    )
    if not np.isfinite(
        total_power
    ):
        raise ValueError(
            "Unable to calculate total signal power."
        )

    return total_power


# ---------------------------------------------------------------------
# PSD in decibels
# ---------------------------------------------------------------------

def psd_to_db(
    psd: np.ndarray,
    reference: float = 1.0,
) -> np.ndarray:
    """
    Convert PSD values to decibels.

    Formula:

        PSD_dB = 10 * log10(PSD / reference)

    Parameters
    ----------
    psd:
        Power spectral density values.

    reference:
        Reference power value.

    Returns
    -------
    np.ndarray
        PSD expressed in decibels.
    """

    psd = np.asarray(
        psd,
        dtype=np.float64,
    )

    if psd.size == 0:
        raise ValueError(
            "PSD cannot be empty."
        )

    if not np.all(
        np.isfinite(psd)
    ):
        raise ValueError(
            "PSD contains NaN or infinite values."
        )

    if np.any(
        psd < 0.0
    ):
        raise ValueError(
            "PSD values cannot be negative."
        )

    if not np.isfinite(
        reference
    ):
        raise ValueError(
            "reference must be finite."
        )

    if reference <= 0.0:
        raise ValueError(
            "reference must be greater than zero."
        )

    # Avoid log10(0).
    safe_psd = np.maximum(
        psd,
        np.finfo(np.float64).tiny,
    )

    return np.asarray(
        10.0
        * np.log10(
            safe_psd / reference
        ),
        dtype=np.float64,
    )