from typing import Tuple

import numpy as np
from scipy.signal import resample_poly


def resample_signal(
    samples: np.ndarray,
    original_sample_rate: float,
    target_sample_rate: float,
) -> Tuple[np.ndarray, float]:
    """
    Resample a signal from its original sampling rate
    to a target sampling rate.

    Works with both real and complex IQ signals.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal samples.

    original_sample_rate : float
        Original sampling frequency in Hz.

    target_sample_rate : float
        Desired sampling frequency in Hz.

    Returns
    -------
    resampled_samples : numpy.ndarray
        Resampled signal.

    new_sample_rate : float
        Target sampling frequency.

    Raises
    ------
    TypeError
        If samples is not a NumPy array.

    ValueError
        If the signal or sampling rates are invalid.
    """

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

    if not isinstance(samples, np.ndarray):
        raise TypeError(
            "samples must be a NumPy array."
        )

    if samples.size == 0:
        raise ValueError(
            "Signal contains no samples."
        )

    if not np.isfinite(samples.real).all():
        raise ValueError(
            "Signal contains invalid real values."
        )

    if not np.isfinite(samples.imag).all():
        raise ValueError(
            "Signal contains invalid imaginary values."
        )

    if not np.isfinite(original_sample_rate):
        raise ValueError(
            "original_sample_rate must be finite."
        )

    if not np.isfinite(target_sample_rate):
        raise ValueError(
            "target_sample_rate must be finite."
        )

    if original_sample_rate <= 0:
        raise ValueError(
            "original_sample_rate must be greater than zero."
        )

    if target_sample_rate <= 0:
        raise ValueError(
            "target_sample_rate must be greater than zero."
        )

    # ---------------------------------------------------------
    # NO RESAMPLING REQUIRED
    # ---------------------------------------------------------

    if np.isclose(
        original_sample_rate,
        target_sample_rate,
    ):
        return samples.copy(), float(target_sample_rate)

    # ---------------------------------------------------------
    # CALCULATE RATIONAL RESAMPLING RATIO
    # ---------------------------------------------------------

    from fractions import Fraction

    ratio = Fraction(
        target_sample_rate / original_sample_rate
    ).limit_denominator(100000)

    up = ratio.numerator
    down = ratio.denominator

    # ---------------------------------------------------------
    # RESAMPLE
    # ---------------------------------------------------------

    resampled_samples = resample_poly(
        samples,
        up,
        down,
    )

    return (
        resampled_samples.astype(samples.dtype),
        float(target_sample_rate),
    )


def downsample_signal(
    samples: np.ndarray,
    sample_rate: float,
    target_sample_rate: float,
) -> Tuple[np.ndarray, float]:
    """
    Downsample a signal to a lower sampling rate.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal.

    sample_rate : float
        Current sampling rate in Hz.

    target_sample_rate : float
        Desired lower sampling rate in Hz.

    Returns
    -------
    tuple
        Resampled signal and new sampling rate.
    """

    if target_sample_rate >= sample_rate:
        raise ValueError(
            "target_sample_rate must be lower than "
            "sample_rate for downsampling."
        )

    return resample_signal(
        samples,
        sample_rate,
        target_sample_rate,
    )


def upsample_signal(
    samples: np.ndarray,
    sample_rate: float,
    target_sample_rate: float,
) -> Tuple[np.ndarray, float]:
    """
    Upsample a signal to a higher sampling rate.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal.

    sample_rate : float
        Current sampling rate in Hz.

    target_sample_rate : float
        Desired higher sampling rate.

    Returns
    -------
    tuple
        Resampled signal and new sampling rate.
    """

    if target_sample_rate <= sample_rate:
        raise ValueError(
            "target_sample_rate must be higher than "
            "sample_rate for upsampling."
        )

    return resample_signal(
        samples,
        sample_rate,
        target_sample_rate,
    )