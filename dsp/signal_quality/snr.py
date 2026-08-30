"""
Signal-to-Noise Ratio (SNR) estimation utilities.

This module estimates the Signal-to-Noise Ratio of communication
signals.

Supported input:
    - Real-valued NumPy arrays
    - Complex IQ NumPy arrays

The primary estimator uses the signal's power relative to an
estimated noise power.

Main functions:
    estimate_snr()
    calculate_signal_power()
    estimate_noise_power()
    snr_from_powers()
    snr()
"""


from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _validate_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Validate and convert a signal to a numeric NumPy array.
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
# Signal power
# ---------------------------------------------------------------------

def calculate_signal_power(
    signal: np.ndarray,
    remove_mean: bool = False,
) -> float:
    """
    Calculate average signal power.

    For real signals:

        P = mean(x^2)

    For complex IQ signals:

        P = mean(|x|^2)

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    remove_mean:
        If True, remove the DC component before calculating power.

    Returns
    -------
    float
        Average signal power.
    """

    signal = _validate_signal(
        signal
    )

    if remove_mean:

        signal = (
            signal
            - np.mean(signal)
        )

    power = float(
        np.mean(
            np.abs(signal) ** 2
        )
    )

    if not np.isfinite(
        power
    ):
        raise ValueError(
            "Unable to calculate signal power."
        )

    return power


# ---------------------------------------------------------------------
# Noise power
# ---------------------------------------------------------------------

def estimate_noise_power(
    signal: np.ndarray,
    method: str = "difference",
) -> float:
    """
    Estimate noise power.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    method:
        Noise estimation method.

        Supported methods:

            "difference"
                Estimates noise from sample-to-sample
                differences.

            "variance"
                Uses variance around the signal mean.

    Returns
    -------
    float
        Estimated noise power.
    """

    signal = _validate_signal(
        signal
    )

    method = str(
        method
    ).lower().strip()

    if signal.size < 2:

        raise ValueError(
            "At least two samples are required "
            "for noise estimation."
        )

    if method == "difference":

        # Difference consecutive samples.
        #
        # For white noise:
        #
        # Var(x[n] - x[n-1]) = 2 * noise_power
        #
        # Therefore divide by 2.

        differences = (
            signal[1:]
            - signal[:-1]
        )

        noise_power = float(
            np.mean(
                np.abs(differences) ** 2
            )
            / 2.0
        )

    elif method == "variance":

        centered = (
            signal
            - np.mean(signal)
        )

        noise_power = float(
            np.mean(
                np.abs(centered) ** 2
            )
        )

    else:

        raise ValueError(
            "method must be either "
            "'difference' or 'variance'."
        )

    if not np.isfinite(
        noise_power
    ):

        raise ValueError(
            "Unable to estimate noise power."
        )

    return noise_power


# ---------------------------------------------------------------------
# SNR from powers
# ---------------------------------------------------------------------

def snr_from_powers(
    signal_power: float,
    noise_power: float,
) -> float:
    """
    Calculate SNR in decibels from signal and noise power.

    Formula:

        SNR(dB) = 10 * log10(P_signal / P_noise)

    Parameters
    ----------
    signal_power:
        Signal power.

    noise_power:
        Noise power.

    Returns
    -------
    float
        SNR in decibels.
    """

    if not np.isfinite(
        signal_power
    ):
        raise ValueError(
            "signal_power must be finite."
        )

    if not np.isfinite(
        noise_power
    ):
        raise ValueError(
            "noise_power must be finite."
        )

    if signal_power < 0.0:

        raise ValueError(
            "signal_power cannot be negative."
        )

    if noise_power < 0.0:

        raise ValueError(
            "noise_power cannot be negative."
        )

    if noise_power == 0.0:

        if signal_power > 0.0:
            return float("inf")

        return 0.0

    if signal_power == 0.0:
        return float("-inf")

    snr_db = float(
        10.0
        * np.log10(
            signal_power
            / noise_power
        )
    )

    return snr_db


# ---------------------------------------------------------------------
# Main SNR estimator
# ---------------------------------------------------------------------

def estimate_snr(
    signal: np.ndarray,
    method: str = "difference",
    signal_power: Optional[float] = None,
) -> dict[str, Any]:
    """
    Estimate the Signal-to-Noise Ratio.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    method:
        Noise estimation method.

        Supported:

            "difference"
            "variance"

    signal_power:
        Optional externally calculated signal power.

        If None, signal power is calculated automatically.

    Returns
    -------
    dict
        Contains:

            snr_db
            signal_power
            noise_power
            signal_to_noise_ratio
            noise_estimation_method
            number_of_samples
            is_complex
    """

    signal = _validate_signal(
        signal
    )

    if signal_power is None:

        calculated_signal_power = (
            calculate_signal_power(
                signal
            )
        )

    else:

        calculated_signal_power = float(
            signal_power
        )

        if not np.isfinite(
            calculated_signal_power
        ):

            raise ValueError(
                "signal_power must be finite."
            )

        if calculated_signal_power < 0.0:

            raise ValueError(
                "signal_power cannot be negative."
            )

    noise_power = estimate_noise_power(
        signal,
        method=method,
    )

    snr_db = snr_from_powers(
        calculated_signal_power,
        noise_power,
    )

    result: dict[str, Any] = {
        "snr_db": float(
            snr_db
        ),

        "signal_power": float(
            calculated_signal_power
        ),

        "noise_power": float(
            noise_power
        ),

        "signal_to_noise_ratio": (
            float(
                calculated_signal_power
                / noise_power
            )
            if noise_power > 0.0
            else float("inf")
        ),

        "noise_estimation_method": (
            str(method).lower().strip()
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
# Compatibility wrapper
# ---------------------------------------------------------------------

def snr(
    signal: np.ndarray,
    method: str = "difference",
) -> dict[str, Any]:
    """
    Compatibility wrapper around estimate_snr().
    """

    return estimate_snr(
        signal,
        method=method,
    )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def get_snr_db(
    signal: np.ndarray,
    method: str = "difference",
) -> float:
    """
    Return only the SNR value in decibels.
    """

    result = estimate_snr(
        signal,
        method=method,
    )

    return float(
        result["snr_db"]
    )