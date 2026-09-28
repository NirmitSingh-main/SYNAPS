"""
Signal-to-Noise Ratio estimation utilities for communication signals.

Supported input:
    - Real-valued NumPy signals
    - Complex IQ signals

SNR can be estimated using:
    - spectral: PSD-based noise-floor estimation (default)
    - difference: sample-to-sample difference method
    - variance: variance-based estimation

The spectral method is preferred for communication signals because
modulation changes in the signal should not automatically be treated
as noise.
"""

from typing import Any

import numpy as np
from scipy.signal import periodogram


def _validate_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Validate and return the input signal.
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

    return signal


def calculate_signal_power(
    signal: np.ndarray,
) -> float:
    """
    Calculate average signal power.

    For complex IQ:

        P = mean(|x|^2)

    For real signals this reduces to:

        P = mean(x^2)
    """

    signal = _validate_signal(signal)

    power = np.mean(
        np.abs(signal) ** 2
    )

    return float(power)


def estimate_noise_power(
    signal: np.ndarray,
    method: str = "spectral",
    sampling_rate: float | None = None,
) -> float:
    """
    Estimate noise power.

    Methods:

        spectral:
            Estimate the noise floor from the PSD.
            Recommended for communication signals.

        difference:
            Estimate noise from sample-to-sample differences.
            Preserved for compatibility with the original
            implementation.

        variance:
            Estimate noise using signal variance.
            Preserved for compatibility.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    method:
        Noise estimation method.

    sampling_rate:
        Sampling rate in Hertz.
        Required for the spectral method.
    """

    signal = _validate_signal(signal)

    method = method.lower().strip()

    if method == "difference":

        if signal.size < 2:
            raise ValueError(
                "At least two samples are required "
                "for difference-based noise estimation."
            )

        differences = (
            signal[1:] - signal[:-1]
        )

        noise_power = (
            np.mean(
                np.abs(differences) ** 2
            )
            / 2.0
        )

        return float(noise_power)

    if method == "variance":

        centered_signal = (
            signal - np.mean(signal)
        )

        noise_power = np.mean(
            np.abs(centered_signal) ** 2
        )

        return float(noise_power)

    if method == "spectral":

        if sampling_rate is None:
            raise ValueError(
                "sampling_rate is required "
                "for spectral noise estimation."
            )

        if not np.isfinite(
            sampling_rate
        ):
            raise ValueError(
                "sampling_rate must be finite."
            )

        if sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be greater than zero."
            )

        if signal.size < 8:
            raise ValueError(
                "At least 8 samples are required "
                "for spectral noise estimation."
            )

        # ---------------------------------------------------------
        # Calculate a two-sided PSD for complex IQ and a one-sided
        # PSD for real-valued signals.
        # ---------------------------------------------------------
        if np.iscomplexobj(signal):

            frequencies, psd = periodogram(
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

            psd = np.fft.fftshift(
                psd
            )

        else:

            frequencies, psd = periodogram(
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

        psd = np.asarray(
            psd,
            dtype=np.float64,
        )

        if psd.size == 0:
            raise ValueError(
                "Unable to calculate signal PSD."
            )

        # Guard against tiny numerical negative values.
        psd = np.maximum(
            psd,
            0.0,
        )

        # ---------------------------------------------------------
        # Estimate the noise floor.
        #
        # The median is deliberately used instead of the mean.
        # Strong signal components can raise the mean considerably,
        # while the median is more robust against those components.
        # ---------------------------------------------------------
        noise_psd = float(
            np.median(psd)
        )

        if noise_psd <= 0.0:
            return 0.0

        # ---------------------------------------------------------
        # Determine the approximate occupied signal region.
        #
        # Frequencies whose PSD is substantially above the noise
        # floor are treated as signal. The threshold is intentionally
        # conservative so that normal spectral leakage does not make
        # the entire spectrum look like signal.
        # ---------------------------------------------------------
        signal_threshold = (
            noise_psd * 3.0
        )

        signal_mask = (
            psd > signal_threshold
        )

        # If the threshold does not identify a meaningful region,
        # fall back to the complete available bandwidth.
        if np.count_nonzero(
            signal_mask
        ) < 2:

            if np.iscomplexobj(signal):
                occupied_bandwidth = float(
                    sampling_rate
                )
            else:
                occupied_bandwidth = float(
                    sampling_rate / 2.0
                )

        else:

            signal_frequencies = (
                frequencies[signal_mask]
            )

            occupied_bandwidth = float(
                np.max(signal_frequencies)
                - np.min(signal_frequencies)
            )

            # Avoid zero-width bands.
            if occupied_bandwidth <= 0.0:

                if np.iscomplexobj(signal):
                    occupied_bandwidth = float(
                        sampling_rate
                    )
                else:
                    occupied_bandwidth = float(
                        sampling_rate / 2.0
                    )

        # ---------------------------------------------------------
        # Noise power inside the estimated signal bandwidth.
        #
        # PSD has units of power/Hz, so:
        #
        #     noise power = noise PSD × bandwidth
        # ---------------------------------------------------------
        noise_power = (
            noise_psd
            * occupied_bandwidth
        )

        return float(
            max(noise_power, 0.0)
        )

    raise ValueError(
        "Unsupported noise estimation method. "
        "Use 'spectral', 'difference', or 'variance'."
    )


def snr_from_powers(
    signal_power: float,
    noise_power: float,
) -> float:
    """
    Calculate SNR in decibels from signal and noise powers.
    """

    if not np.isfinite(
        signal_power
    ):
        raise ValueError(
            "Signal power must be finite."
        )

    if not np.isfinite(
        noise_power
    ):
        raise ValueError(
            "Noise power must be finite."
        )

    if signal_power < 0.0:
        raise ValueError(
            "Signal power cannot be negative."
        )

    if noise_power < 0.0:
        raise ValueError(
            "Noise power cannot be negative."
        )

    if signal_power == 0.0:
        return float("-inf")

    if noise_power == 0.0:
        return float("inf")

    return float(
        10.0
        * np.log10(
            signal_power
            / noise_power
        )
    )


def estimate_snr(
    signal: np.ndarray,
    method: str = "spectral",
    sampling_rate: float | None = None,
) -> dict[str, Any]:
    """
    Estimate Signal-to-Noise Ratio.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    method:
        SNR estimation method.

        Default:
            "spectral"

        Other supported methods:
            "difference"
            "variance"

    sampling_rate:
        Sampling rate in Hertz.

        Required when using the spectral method.

    Returns
    -------
    dict
        Dictionary containing:

            snr_db
            signal_power
            noise_power
            method
    """

    signal = _validate_signal(
        signal
    )

    signal_power = (
        calculate_signal_power(
            signal
        )
    )

    noise_power = (
        estimate_noise_power(
            signal,
            method=method,
            sampling_rate=sampling_rate,
        )
    )

    snr_db = (
        snr_from_powers(
            signal_power,
            noise_power,
        )
    )

    return {
        "snr_db": float(snr_db),
        "signal_power": float(
            signal_power
        ),
        "noise_power": float(
            noise_power
        ),
        "method": str(
            method.lower().strip()
        ),
    }