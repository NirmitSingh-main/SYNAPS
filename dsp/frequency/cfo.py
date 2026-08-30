"""
Carrier Frequency Offset estimation.

Supported input types:
    - Real-valued NumPy signals
    - Complex In-phase/Quadrature signals
    - WAV files
    - Raw IQ files

Carrier Frequency Offset is calculated as:

    CFO = measured_frequency - reference_frequency

Positive CFO:
    Measured frequency is higher than the reference frequency.

Negative CFO:
    Measured frequency is lower than the reference frequency.
"""

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert


def _validate_signal(
    signal: np.ndarray,
    sampling_rate: float,
) -> np.ndarray:
    """
    Validate the input signal and sampling rate.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

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


def _prepare_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convert a real-valued signal into an analytic
    complex signal.

    Complex IQ signals are returned directly.

    Real signals are converted using the Hilbert
    transform.
    """

    signal = np.asarray(signal)

    if np.iscomplexobj(signal):
        return np.asarray(
            signal,
            dtype=np.complex128,
        )

    real_signal = np.asarray(
        signal,
        dtype=np.float64,
    )

    analytic_signal = np.asarray(
        hilbert(real_signal),
        dtype=np.complex128,
    )

    return analytic_signal
def _estimate_carrier_frequency(
    signal: np.ndarray,
    sampling_rate: float,
) -> float:
    """
    Estimate the dominant carrier frequency.

    The Fast Fourier Transform is used to locate
    the strongest positive-frequency component.
    """

    signal = _prepare_signal(signal)

    number_of_samples = signal.size

    # Remove the average value to reduce the
    # effect of the DC component.
    signal = signal - np.mean(signal)

    # Apply a Hann window to reduce spectral leakage.
    window = np.hanning(number_of_samples)

    windowed_signal = signal * window

    # Calculate the Fast Fourier Transform.
    spectrum = np.fft.fft(
        windowed_signal
    )

    frequencies = np.fft.fftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )

    # Keep only positive frequencies.
    positive_mask = frequencies >= 0

    positive_spectrum = np.abs(
        spectrum[positive_mask]
    )

    positive_frequencies = frequencies[
        positive_mask
    ]

    # Ignore the DC component.
    if positive_spectrum.size > 1:
        positive_spectrum[0] = 0.0

    # Find the strongest frequency component.
    peak_index = np.argmax(
        positive_spectrum
    )

    measured_frequency = (
        positive_frequencies[peak_index]
    )

    return float(measured_frequency)


def estimate_cfo(
    signal: np.ndarray,
    sampling_rate: float,
    reference_frequency_hz: float,
) -> dict[str, Any]:
    """
    Estimate Carrier Frequency Offset from a signal.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    sampling_rate:
        Signal sampling rate in Hertz.

    reference_frequency_hz:
        Expected/reference carrier frequency
        in Hertz.

    Returns
    -------
    dict
        Dictionary containing:

        measured_frequency_hz
        reference_frequency_hz
        cfo_hz
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if not np.isfinite(
        reference_frequency_hz
    ):
        raise ValueError(
            "Reference frequency must be finite."
        )

    measured_frequency_hz = (
        _estimate_carrier_frequency(
            signal,
            sampling_rate,
        )
    )

    cfo_hz = (
        measured_frequency_hz
        - reference_frequency_hz
    )

    return {
        "measured_frequency_hz": float(
            measured_frequency_hz
        ),
        "reference_frequency_hz": float(
            reference_frequency_hz
        ),
        "cfo_hz": float(cfo_hz),
    }


def _load_wav_file(
    file_path: Path,
) -> tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Stereo WAV files are converted to mono.

    Integer WAV samples are converted to floating
    point and normalized.

    Returns
    -------
    tuple
        signal, sampling_rate
    """

    sampling_rate, signal = wavfile.read(
        file_path
    )

    signal = np.asarray(signal)

    # Convert stereo audio to mono.
    if signal.ndim == 2:
        signal = np.mean(
            signal.astype(np.float64),
            axis=1,
        )

    # Convert integer samples to floating point.
    if np.issubdtype(
        signal.dtype,
        np.integer,
    ):
        signal = signal.astype(
            np.float64
        )

        maximum_amplitude = np.max(
            np.abs(signal)
        )

        if maximum_amplitude > 0:
            signal = (
                signal
                / maximum_amplitude
            )

    else:
        signal = signal.astype(
            np.float64
        )

    return (
        signal,
        float(sampling_rate),
    )


def _load_iq_file(
    file_path: Path,
    iq_dtype: np.dtype = np.dtype(np.float32),
    iq_order: str = "IQ",
) -> np.ndarray:
    """
    Load a raw interleaved IQ file.

    Expected format:

        I, Q, I, Q, I, Q, ...

    Parameters
    ----------
    file_path:
        Path to the IQ file.

    iq_dtype:
        NumPy data type of each I/Q sample.

    iq_order:
        Either "IQ" or "QI".

    Returns
    -------
    numpy.ndarray
        Complex IQ signal.
    """

    raw_data = np.fromfile(
        file_path,
        dtype=iq_dtype,
    )

    if raw_data.size == 0:
        raise ValueError(
            "IQ file is empty."
        )

    # Every IQ sample requires two values:
    # one In-phase value and one Quadrature value.
    if raw_data.size % 2 != 0:
        raise ValueError(
            "IQ file must contain an even number "
            "of values."
        )

    first_component = raw_data[0::2]
    second_component = raw_data[1::2]

    iq_order = iq_order.upper()

    if iq_order == "IQ":

        i_samples = first_component
        q_samples = second_component

    elif iq_order == "QI":

        q_samples = first_component
        i_samples = second_component

    else:
        raise ValueError(
            "iq_order must be either 'IQ' or 'QI'."
        )

    signal = (
        i_samples.astype(np.float64)
        + 1j * q_samples.astype(np.float64)
    )

    return signal


def estimate_cfo_from_file(
    file_path: str | Path,
    reference_frequency_hz: float,
    sampling_rate: float | None = None,
    iq_dtype: np.dtype = np.dtype(np.float32),
    iq_order: str = "IQ",
) -> dict[str, Any]:
    """
    Estimate Carrier Frequency Offset directly
    from a WAV or IQ file.

    Parameters
    ----------
    file_path:
        Path to a .wav or .iq file.

    reference_frequency_hz:
        Expected/reference carrier frequency
        in Hertz.

    sampling_rate:
        Sampling rate for raw IQ files.

        WAV files do not require this because
        their sampling rate is stored in the WAV
        file header.

    iq_dtype:
        Data type used by the raw IQ file.

    iq_order:
        IQ ordering.

        "IQ" means:
            I, Q, I, Q, ...

        "QI" means:
            Q, I, Q, I, ...

    Returns
    -------
    dict
        CFO estimation result and file metadata.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".wav":

        signal, file_sampling_rate = (
            _load_wav_file(file_path)
        )

        sampling_rate = file_sampling_rate

        result = estimate_cfo(
            signal,
            sampling_rate,
            reference_frequency_hz,
        )

    elif extension == ".iq":

        if sampling_rate is None:
            raise ValueError(
                "sampling_rate is required "
                "for raw IQ files."
            )

        signal = _load_iq_file(
            file_path,
            iq_dtype=iq_dtype,
            iq_order=iq_order,
        )

        result = estimate_cfo(
            signal,
            sampling_rate,
            reference_frequency_hz,
        )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Only .wav and .iq files are supported."
        )

    result["file"] = str(file_path)

    result["file_type"] = extension

    result["sampling_rate_hz"] = float(
        sampling_rate
    )

    return result