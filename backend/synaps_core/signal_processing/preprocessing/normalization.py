import numpy as np


def normalize_signal(
    samples: np.ndarray,
    target_amplitude: float = 1.0,
) -> np.ndarray:
    """
    Normalize a signal to a specified peak amplitude.

    For a complex IQ signal, the magnitude |I + jQ|
    is used to determine the peak amplitude.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal samples.

    target_amplitude : float
        Desired maximum signal magnitude.
        Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Normalized signal.

    Raises
    ------
    TypeError
        If samples is not a NumPy array.

    ValueError
        If the signal is empty, contains invalid values,
        or target_amplitude is not positive.
    """

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

    if not np.isfinite(target_amplitude):
        raise ValueError(
            "target_amplitude must be finite."
        )

    if target_amplitude <= 0:
        raise ValueError(
            "target_amplitude must be greater than zero."
        )

    # Find the maximum signal magnitude.
    peak_amplitude = np.max(np.abs(samples))

    # Avoid division by zero for an all-zero signal.
    if peak_amplitude == 0:
        return samples.copy()

    # Scale the signal.
    normalized = (
        samples / peak_amplitude
    ) * target_amplitude

    return normalized.astype(samples.dtype)


def normalize_power(
    samples: np.ndarray,
    target_power: float = 1.0,
) -> np.ndarray:
    """
    Normalize a signal to a specified average power.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal samples.

    target_power : float
        Desired average signal power.
        Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Power-normalized signal.
    """

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

    if not np.isfinite(target_power):
        raise ValueError(
            "target_power must be finite."
        )

    if target_power <= 0:
        raise ValueError(
            "target_power must be greater than zero."
        )

    # Calculate current average power.
    current_power = np.mean(
        np.abs(samples) ** 2
    )

    # Avoid division by zero for an all-zero signal.
    if current_power == 0:
        return samples.copy()

    # Calculate scaling factor.
    scale = np.sqrt(
        target_power / current_power
    )

    normalized = samples * scale

    return normalized.astype(samples.dtype)


def normalize_iq(
    samples: np.ndarray,
) -> np.ndarray:
    """
    Normalize an IQ signal so that its maximum magnitude
    is equal to 1.

    This is a convenience wrapper around normalize_signal().
    """

    return normalize_signal(
        samples,
        target_amplitude=1.0,
    )