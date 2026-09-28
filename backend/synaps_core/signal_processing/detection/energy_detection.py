import numpy as np


def calculate_signal_power(samples: np.ndarray) -> np.ndarray:
    """
    Calculate the instantaneous power of a complex signal.

    For a complex sample:

        x = I + jQ

    the power is:

        |x|^2 = I^2 + Q^2

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    Returns
    -------
    numpy.ndarray
        Instantaneous power of each sample.
    """

    if not isinstance(samples, np.ndarray):
        raise TypeError("samples must be a NumPy array.")

    if samples.size == 0:
        raise ValueError("Signal contains no samples.")

    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex64)

    if not np.isfinite(samples.real).all():
        raise ValueError("Signal contains invalid real values.")

    if not np.isfinite(samples.imag).all():
        raise ValueError("Signal contains invalid imaginary values.")

    return np.abs(samples) ** 2


def calculate_average_power(samples: np.ndarray) -> float:
    """
    Calculate the average power of a signal.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    Returns
    -------
    float
        Average signal power.
    """

    power = calculate_signal_power(samples)

    return float(np.mean(power))


def calculate_energy(samples: np.ndarray) -> float:
    """
    Calculate the total energy of a signal.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    Returns
    -------
    float
        Total signal energy.
    """

    power = calculate_signal_power(samples)

    return float(np.sum(power))


def detect_energy(
    samples: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Detect samples whose instantaneous power is
    greater than the specified threshold.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    threshold : float
        Power threshold used for detection.

    Returns
    -------
    numpy.ndarray
        Boolean detection mask.

        True  -> signal detected
        False -> signal not detected
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    if threshold < 0:
        raise ValueError("threshold cannot be negative.")

    power = calculate_signal_power(samples)

    return power > threshold


def detect_by_average_power(
    samples: np.ndarray,
    threshold_factor: float = 1.0,
) -> np.ndarray:
    """
    Perform simple energy detection using the average
    signal power as the reference.

    This is mainly intended for the current synthetic
    dataset and initial pipeline testing.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    threshold_factor : float
        Multiplier applied to average power.

    Returns
    -------
    numpy.ndarray
        Boolean detection mask.
    """

    if not np.isfinite(threshold_factor):
        raise ValueError(
            "threshold_factor must be finite."
        )

    if threshold_factor < 0:
        raise ValueError(
            "threshold_factor cannot be negative."
        )

    average_power = calculate_average_power(samples)

    threshold = average_power * threshold_factor

    return detect_energy(
        samples,
        threshold,
    )