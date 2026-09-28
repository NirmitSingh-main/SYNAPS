"""
Bandwidth estimation utilities for communication signals.

This module estimates the occupied bandwidth of a signal using its
power spectral density.

Supported input:
    - Real-valued NumPy arrays
    - Complex IQ NumPy arrays

Main function:
    estimate_bandwidth()

The occupied bandwidth is calculated using a cumulative-power method.
By default, 99% of the total signal power is considered occupied.
"""

from typing import Any

import numpy as np
from scipy.signal import periodogram


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

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if not np.isfinite(sampling_rate):
        raise ValueError(
            "Sampling rate must be finite."
        )

    if sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    return signal


def _calculate_psd(
    signal: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the power spectral density.

    Returns:
        frequencies:
            Frequency values in Hz.

        power:
            Power spectral density values.
    """

    if np.iscomplexobj(signal):

        frequencies, power = periodogram(
            signal,
            fs=sampling_rate,
            window="hann",
            detrend="constant",
            return_onesided=False,
            scaling="density",
        )

        frequencies = np.fft.fftshift(
            frequencies
        )

        power = np.fft.fftshift(
            power
        )

    else:

        frequencies, power = periodogram(
            signal,
            fs=sampling_rate,
            window="hann",
            detrend="constant",
            return_onesided=True,
            scaling="density",
        )

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    power = np.asarray(
        power,
        dtype=np.float64,
    )

    return frequencies, power


def _occupied_band_edges(
    frequencies: np.ndarray,
    power: np.ndarray,
    occupied_fraction: float,
) -> tuple[float, float]:
    """
    Find the lower and upper frequency containing the requested
    fraction of total power.
    """

    if frequencies.size == 0:
        raise ValueError(
            "Unable to calculate frequency spectrum."
        )

    total_power = float(
        np.sum(power)
    )

    if total_power <= 0.0:
        raise ValueError(
            "Signal has zero spectral power."
        )

    target_power = (
        occupied_fraction
        * total_power
    )

    lower_power = (
        (1.0 - occupied_fraction)
        / 2.0
        * total_power
    )

    upper_power = (
        total_power
        - lower_power
    )

    cumulative_power = np.cumsum(
        power
    )

    lower_index = int(
        np.searchsorted(
            cumulative_power,
            lower_power,
            side="left",
        )
    )

    upper_index = int(
        np.searchsorted(
            cumulative_power,
            upper_power,
            side="left",
        )
    )

    lower_index = min(
        lower_index,
        frequencies.size - 1,
    )

    upper_index = min(
        upper_index,
        frequencies.size - 1,
    )

    lower_frequency = float(
        frequencies[lower_index]
    )

    upper_frequency = float(
        frequencies[upper_index]
    )

    if lower_frequency > upper_frequency:

        lower_frequency, upper_frequency = (
            upper_frequency,
            lower_frequency,
        )

    return (
        lower_frequency,
        upper_frequency,
    )


def estimate_bandwidth(
    signal: np.ndarray,
    sampling_rate: float,
    occupied_fraction: float = 0.99,
) -> dict[str, Any]:
    """
    Estimate the occupied bandwidth of a signal.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    sampling_rate:
        Sampling rate in Hertz.

    occupied_fraction:
        Fraction of total signal power considered occupied.

        Default:
            0.99 = 99%

        Must be greater than 0 and less than or equal to 1.

    Returns
    -------
    dict
        Dictionary containing:

            bandwidth_hz
            lower_frequency_hz
            upper_frequency_hz
            occupied_fraction
            sampling_rate_hz
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if not np.isfinite(
        occupied_fraction
    ):
        raise ValueError(
            "occupied_fraction must be finite."
        )

    if (
        occupied_fraction <= 0.0
        or occupied_fraction > 1.0
    ):
        raise ValueError(
            "occupied_fraction must be greater than "
            "0 and less than or equal to 1."
        )

    frequencies, power = _calculate_psd(
        signal,
        sampling_rate,
    )

    (
        lower_frequency,
        upper_frequency,
    ) = _occupied_band_edges(
        frequencies,
        power,
        occupied_fraction,
    )

    bandwidth = (
        upper_frequency
        - lower_frequency
    )

    result: dict[str, Any] = {
        "bandwidth_hz": float(
            bandwidth
        ),
        "lower_frequency_hz": float(
            lower_frequency
        ),
        "upper_frequency_hz": float(
            upper_frequency
        ),
        "occupied_fraction": float(
            occupied_fraction
        ),
        "sampling_rate_hz": float(
            sampling_rate
        ),
    }

    return result


def estimate_bandwidth_from_psd(
    frequencies: np.ndarray,
    power: np.ndarray,
    occupied_fraction: float = 0.99,
) -> dict[str, Any]:
    """
    Estimate occupied bandwidth from an already calculated PSD.

    Parameters
    ----------
    frequencies:
        One-dimensional frequency array in Hz.

    power:
        One-dimensional PSD array.

    occupied_fraction:
        Fraction of total power considered occupied.

    Returns
    -------
    dict
        Bandwidth estimation results.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    power = np.asarray(
        power,
        dtype=np.float64,
    )

    if frequencies.size == 0:
        raise ValueError(
            "Frequency array cannot be empty."
        )

    if power.size == 0:
        raise ValueError(
            "Power array cannot be empty."
        )

    if frequencies.ndim != 1:
        raise ValueError(
            "Frequencies must be one-dimensional."
        )

    if power.ndim != 1:
        raise ValueError(
            "Power must be one-dimensional."
        )

    if frequencies.size != power.size:
        raise ValueError(
            "Frequencies and power must have the same length."
        )

    if not np.all(
        np.isfinite(frequencies)
    ):
        raise ValueError(
            "Frequencies contain NaN or infinite values."
        )

    if not np.all(
        np.isfinite(power)
    ):
        raise ValueError(
            "Power contains NaN or infinite values."
        )

    if np.any(power < 0.0):
        raise ValueError(
            "Power values cannot be negative."
        )

    if not np.isfinite(
        occupied_fraction
    ):
        raise ValueError(
            "occupied_fraction must be finite."
        )

    if (
        occupied_fraction <= 0.0
        or occupied_fraction > 1.0
    ):
        raise ValueError(
            "occupied_fraction must be greater than "
            "0 and less than or equal to 1."
        )

    # Sort frequencies so cumulative integration is performed
    # from low to high frequency.
    sort_indices = np.argsort(
        frequencies
    )

    frequencies = frequencies[
        sort_indices
    ]

    power = power[
        sort_indices
    ]

    total_power = float(
        np.sum(power)
    )

    if total_power <= 0.0:
        raise ValueError(
            "Power spectrum contains no energy."
        )

    cumulative_power = np.cumsum(
        power
    )

    lower_power = (
        (1.0 - occupied_fraction)
        / 2.0
        * total_power
    )

    upper_power = (
        total_power
        - lower_power
    )

    lower_index = int(
        np.searchsorted(
            cumulative_power,
            lower_power,
            side="left",
        )
    )

    upper_index = int(
        np.searchsorted(
            cumulative_power,
            upper_power,
            side="left",
        )
    )

    lower_index = min(
        lower_index,
        frequencies.size - 1,
    )

    upper_index = min(
        upper_index,
        frequencies.size - 1,
    )

    lower_frequency = float(
        frequencies[lower_index]
    )

    upper_frequency = float(
        frequencies[upper_index]
    )

    if lower_frequency > upper_frequency:

        lower_frequency, upper_frequency = (
            upper_frequency,
            lower_frequency,
        )

    bandwidth = (
        upper_frequency
        - lower_frequency
    )

    return {
        "bandwidth_hz": float(
            bandwidth
        ),
        "lower_frequency_hz": float(
            lower_frequency
        ),
        "upper_frequency_hz": float(
            upper_frequency
        ),
        "occupied_fraction": float(
            occupied_fraction
        ),
    }