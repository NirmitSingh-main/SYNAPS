import numpy as np
from scipy.signal import butter, sosfilt


def lowpass_filter(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_frequency: float,
    filter_order: int = 5,
) -> np.ndarray:
    """
    Apply a low-pass Butterworth filter to a signal.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal samples. Can be real or complex.

    sample_rate : float
        Sampling frequency in Hz.

    cutoff_frequency : float
        Low-pass cutoff frequency in Hz.

    filter_order : int
        Order of the Butterworth filter.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    _validate_inputs(
        samples,
        sample_rate,
        cutoff_frequency,
        filter_order,
    )

    normalized_cutoff = cutoff_frequency / (
        sample_rate / 2.0
    )

    sos = butter(
        filter_order,
        normalized_cutoff,
        btype="lowpass",
        output="sos",
    )

    if np.iscomplexobj(samples):
        real_part = sosfilt(sos, samples.real)
        imag_part = sosfilt(sos, samples.imag)

        return (
            real_part + 1j * imag_part
        ).astype(samples.dtype)

    return sosfilt(
        sos,
        samples,
    ).astype(samples.dtype)


def highpass_filter(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_frequency: float,
    filter_order: int = 5,
) -> np.ndarray:
    """
    Apply a high-pass Butterworth filter to a signal.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal samples. Can be real or complex.

    sample_rate : float
        Sampling frequency in Hz.

    cutoff_frequency : float
        High-pass cutoff frequency in Hz.

    filter_order : int
        Order of the Butterworth filter.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    _validate_inputs(
        samples,
        sample_rate,
        cutoff_frequency,
        filter_order,
    )

    normalized_cutoff = cutoff_frequency / (
        sample_rate / 2.0
    )

    sos = butter(
        filter_order,
        normalized_cutoff,
        btype="highpass",
        output="sos",
    )

    if np.iscomplexobj(samples):
        real_part = sosfilt(sos, samples.real)
        imag_part = sosfilt(sos, samples.imag)

        return (
            real_part + 1j * imag_part
        ).astype(samples.dtype)

    return sosfilt(
        sos,
        samples,
    ).astype(samples.dtype)


def bandpass_filter(
    samples: np.ndarray,
    sample_rate: float,
    low_cutoff: float,
    high_cutoff: float,
    filter_order: int = 5,
) -> np.ndarray:
    """
    Apply a band-pass Butterworth filter to a signal.

    Parameters
    ----------
    samples : numpy.ndarray
        Input signal samples. Can be real or complex.

    sample_rate : float
        Sampling frequency in Hz.

    low_cutoff : float
        Lower cutoff frequency in Hz.

    high_cutoff : float
        Upper cutoff frequency in Hz.

    filter_order : int
        Order of the Butterworth filter.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    if low_cutoff >= high_cutoff:
        raise ValueError(
            "low_cutoff must be smaller than high_cutoff."
        )

    _validate_inputs(
        samples,
        sample_rate,
        low_cutoff,
        filter_order,
    )

    if high_cutoff >= sample_rate / 2.0:
        raise ValueError(
            "high_cutoff must be below the Nyquist frequency."
        )

    normalized_low = low_cutoff / (
        sample_rate / 2.0
    )

    normalized_high = high_cutoff / (
        sample_rate / 2.0
    )

    sos = butter(
        filter_order,
        [normalized_low, normalized_high],
        btype="bandpass",
        output="sos",
    )

    if np.iscomplexobj(samples):
        real_part = sosfilt(sos, samples.real)
        imag_part = sosfilt(sos, samples.imag)

        return (
            real_part + 1j * imag_part
        ).astype(samples.dtype)

    return sosfilt(
        sos,
        samples,
    ).astype(samples.dtype)


def _validate_inputs(
    samples: np.ndarray,
    sample_rate: float,
    cutoff_frequency: float,
    filter_order: int,
) -> None:
    """
    Validate common filter parameters.
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

    if not np.isfinite(sample_rate):
        raise ValueError(
            "sample_rate must be finite."
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be greater than zero."
        )

    nyquist = sample_rate / 2.0

    if cutoff_frequency <= 0:
        raise ValueError(
            "cutoff_frequency must be greater than zero."
        )

    if cutoff_frequency >= nyquist:
        raise ValueError(
            "cutoff_frequency must be below "
            "the Nyquist frequency."
        )

    if filter_order <= 0:
        raise ValueError(
            "filter_order must be greater than zero."
        )