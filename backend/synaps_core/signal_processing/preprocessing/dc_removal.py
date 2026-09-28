import numpy as np


def remove_dc(samples: np.ndarray) -> np.ndarray:
    """
    Remove the DC component from a signal.

    The DC component is the constant average value of the
    signal. For a complex IQ signal, the mean is calculated
    independently for the I and Q components.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    Returns
    -------
    numpy.ndarray
        Signal with its DC component removed.

    Raises
    ------
    TypeError
        If samples is not a NumPy array.

    ValueError
        If the signal is empty or contains invalid values.
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

    # Convert to complex representation if necessary.
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex64)

    # Calculate the DC component.
    dc_component = np.mean(samples)

    # Remove the DC component.
    corrected_signal = samples - dc_component

    return corrected_signal.astype(samples.dtype)