"""
FFT utilities for communication signals.

This module provides FFT-based spectral analysis for:

    - Real-valued signals
    - Complex IQ signals

The implementation uses NumPy only.

Main function:
    compute_fft()
"""

from typing import Any

import numpy as np


def _validate_signal(
    signal: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Validate signal and sampling rate.
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

    if sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    return signal


def _calculate_frequency_axis(
    number_of_samples: int,
    sampling_rate: float,
) -> np.ndarray:
    """
    Create a centered FFT frequency axis.

    Frequencies are returned in the range:

        -Fs/2 ... +Fs/2
    """

    frequencies = np.fft.fftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )

    return np.fft.fftshift(
        frequencies
    )


def _calculate_fft(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Calculate a centered FFT.

    fftshift is applied so that the FFT and frequency
    axis have matching positions.
    """

    spectrum = np.fft.fft(
        signal
    )

    return np.fft.fftshift(
        spectrum
    )


def _find_peak_frequency(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    signal_is_complex: bool,
) -> tuple[float, float]:
    """
    Find the dominant frequency.

    For complex IQ signals:
        Search the complete frequency range.

    For real signals:
        Search only positive frequencies so that the
        negative-frequency mirror is not selected.
    """

    magnitude = np.abs(
        spectrum
    )

    if signal_is_complex:

        peak_index = int(
            np.argmax(magnitude)
        )

    else:

        positive_mask = frequencies >= 0

        if not np.any(
            positive_mask
        ):
            peak_index = int(
                np.argmax(magnitude)
            )

        else:

            positive_indices = np.flatnonzero(
                positive_mask
            )

            positive_magnitude = magnitude[
                positive_mask
            ]

            relative_index = int(
                np.argmax(
                    positive_magnitude
                )
            )

            peak_index = int(
                positive_indices[
                    relative_index
                ]
            )

    return (
        float(frequencies[peak_index]),
        float(magnitude[peak_index]),
    )


def compute_fft(
    signal: np.ndarray,
    sampling_rate: float,
    remove_dc: bool = False,
) -> dict[str, Any]:
    """
    Compute the FFT of a signal.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    sampling_rate:
        Sampling rate in Hertz.

    remove_dc:
        If True, subtract the mean before computing
        the FFT.

    Returns
    -------
    dict
        Dictionary containing:

            spectrum
            fft
            frequencies_hz
            magnitude
            phase_radians
            real
            imaginary
            number_of_samples
            sampling_rate_hz
            frequency_resolution_hz
            peak_frequency_hz
            peak_magnitude
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    number_of_samples = int(
        signal.size
    )

    signal_is_complex = bool(
        np.iscomplexobj(signal)
    )

    # Convert to a predictable numeric type.
    if signal_is_complex:

        working_signal = np.asarray(
            signal,
            dtype=np.complex128,
        )

    else:

        working_signal = np.asarray(
            signal,
            dtype=np.float64,
        )

    # Optional DC removal.
    if remove_dc:

        working_signal = (
            working_signal
            - np.mean(working_signal)
        )

    # Calculate centered FFT.
    spectrum = _calculate_fft(
        working_signal
    )

    # Frequency axis corresponding exactly
    # to the centered FFT.
    frequencies_hz = _calculate_frequency_axis(
        number_of_samples,
        float(sampling_rate),
    )

    magnitude = np.abs(
        spectrum
    )

    phase_radians = np.angle(
        spectrum
    )

    peak_frequency_hz, peak_magnitude = (
        _find_peak_frequency(
            spectrum,
            frequencies_hz,
            signal_is_complex,
        )
    )

    frequency_resolution_hz = (
        float(sampling_rate)
        / float(number_of_samples)
    )

    result: dict[str, Any] = {
        # Main spectrum name expected by the project.
        "spectrum": np.asarray(
            spectrum,
            dtype=np.complex128,
        ),

        # Compatibility alias.
        "fft": np.asarray(
            spectrum,
            dtype=np.complex128,
        ),

        "frequencies_hz": np.asarray(
            frequencies_hz,
            dtype=np.float64,
        ),

        "magnitude": np.asarray(
            magnitude,
            dtype=np.float64,
        ),

        "phase_radians": np.asarray(
            phase_radians,
            dtype=np.float64,
        ),

        "real": np.asarray(
            np.real(spectrum),
            dtype=np.float64,
        ),

        "imaginary": np.asarray(
            np.imag(spectrum),
            dtype=np.float64,
        ),

        "number_of_samples": number_of_samples,

        "sampling_rate_hz": float(
            sampling_rate
        ),

        "frequency_resolution_hz": (
            frequency_resolution_hz
        ),

        "peak_frequency_hz": (
            peak_frequency_hz
        ),

        "peak_magnitude": (
            peak_magnitude
        ),
    }

    return result


def fft(
    signal: np.ndarray,
    sampling_rate: float,
    remove_dc: bool = False,
) -> dict[str, Any]:
    """
    Compatibility wrapper around compute_fft().
    """

    return compute_fft(
        signal,
        sampling_rate,
        remove_dc=remove_dc,
    )


def calculate_fft(
    signal: np.ndarray,
    sampling_rate: float,
    remove_dc: bool = False,
) -> dict[str, Any]:
    """
    Compatibility wrapper around compute_fft().
    """

    return compute_fft(
        signal,
        sampling_rate,
        remove_dc=remove_dc,
    )