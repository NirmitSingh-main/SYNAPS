"""
Phase estimation for communication signals.

Supported inputs:
    - Real-valued signals
    - Complex In-phase/Quadrature signals
    - WAV files
    - Raw IQ files

The module estimates the carrier phase at t = 0.

For a signal of the form:

    x(t) = A cos(2*pi*f*t + phi)

the estimated phase is:

    phi = angle(sum(x(t) * exp(-j*2*pi*f*t)))

For real-valued signals, an analytic signal is first generated
using the Hilbert transform.

For complex IQ signals, the signal is used directly.
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


def _prepare_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convert the input into a complex analytic signal.

    Complex IQ signals are retained.

    Real-valued signals are converted into an analytic
    signal using the Hilbert transform.
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


def _estimate_frequency(
    signal: np.ndarray,
    sampling_rate: float,
) -> float:
    """
    Estimate the dominant positive frequency using
    the Fast Fourier Transform.

    This is used when the caller does not provide
    a known carrier frequency.
    """

    analytic_signal = _prepare_signal(signal)

    number_of_samples = analytic_signal.size

    # Remove the average value.
    analytic_signal = (
        analytic_signal
        - np.mean(analytic_signal)
    )

    # Window the signal to reduce spectral leakage.
    window = np.hanning(number_of_samples)

    windowed_signal = (
        analytic_signal * window
    )

    spectrum = np.fft.fft(
        windowed_signal
    )

    frequencies = np.fft.fftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )

    positive_mask = frequencies >= 0

    positive_spectrum = np.abs(
        spectrum[positive_mask]
    )

    positive_frequencies = frequencies[
        positive_mask
    ]

    # Ignore the direct-current component.
    if positive_spectrum.size > 1:
        positive_spectrum[0] = 0.0

    peak_index = np.argmax(
        positive_spectrum
    )

    return float(
        positive_frequencies[peak_index]
    )


def _wrap_phase(
    phase_radians: float,
) -> float:
    """
    Wrap a phase angle into the interval:

        [-pi, pi)
    """

    wrapped_phase = (
        (phase_radians + np.pi)
        % (2.0 * np.pi)
    ) - np.pi

    return float(wrapped_phase)


def _estimate_phase_at_frequency(
    signal: np.ndarray,
    sampling_rate: float,
    frequency_hz: float,
) -> float:
    """
    Estimate the carrier phase at t = 0.

    The signal is projected onto the complex exponential
    corresponding to the supplied carrier frequency.
    """

    analytic_signal = _prepare_signal(
        signal
    )

    number_of_samples = analytic_signal.size

    time = np.arange(
        number_of_samples,
        dtype=np.float64,
    ) / sampling_rate

    reference = np.exp(
        -1j
        * 2.0
        * np.pi
        * frequency_hz
        * time
    )

    projection = np.sum(
        analytic_signal * reference
    )

    if np.abs(projection) == 0:
        raise ValueError(
            "Unable to estimate phase from "
            "a zero-energy signal."
        )

    phase_radians = np.angle(
        projection
    )

    return _wrap_phase(
        float(phase_radians)
    )


def _calculate_phase_statistics(
    signal: np.ndarray,
) -> dict[str, float]:
    """
    Calculate useful phase statistics from the
    analytic signal.

    Returns:
        mean phase
        phase standard deviation
        minimum phase
        maximum phase
    """

    analytic_signal = _prepare_signal(
        signal
    )

    instantaneous_phase = np.unwrap(
        np.angle(analytic_signal)
    )

    mean_phase = float(
        np.mean(instantaneous_phase)
    )

    phase_standard_deviation = float(
        np.std(instantaneous_phase)
    )

    minimum_phase = float(
        np.min(instantaneous_phase)
    )

    maximum_phase = float(
        np.max(instantaneous_phase)
    )

    return {
        "mean_phase_radians": mean_phase,
        "phase_std_radians": phase_standard_deviation,
        "minimum_phase_radians": minimum_phase,
        "maximum_phase_radians": maximum_phase,
    }


def estimate_phase(
    signal: np.ndarray,
    sampling_rate: float,
    frequency_hz: float | None = None,
) -> dict[str, Any]:
    """
    Estimate carrier phase from a signal.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    sampling_rate:
        Sampling rate in Hertz.

    frequency_hz:
        Known carrier frequency in Hertz.

        If None, the dominant frequency is estimated
        automatically.

    Returns
    -------
    dict
        Phase estimation results.
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    if frequency_hz is None:

        frequency_hz = _estimate_frequency(
            signal,
            sampling_rate,
        )

    else:

        if not np.isfinite(
            frequency_hz
        ):
            raise ValueError(
                "Frequency must be finite."
            )

        if frequency_hz < 0:
            raise ValueError(
                "Frequency cannot be negative."
            )

    phase_radians = (
        _estimate_phase_at_frequency(
            signal,
            sampling_rate,
            frequency_hz,
        )
    )

    phase_degrees = float(
        np.degrees(phase_radians)
    )

    statistics = (
        _calculate_phase_statistics(
            signal
        )
    )

    result: dict[str, Any] = {
        "estimated_phase_radians": float(
            phase_radians
        ),
        "estimated_phase_degrees": phase_degrees,
        "frequency_hz": float(
            frequency_hz
        ),
        "sampling_rate_hz": float(
            sampling_rate
        ),
    }

    result.update(statistics)

    return result


def _load_wav_file(
    file_path: Path,
) -> tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Stereo WAV files are converted to mono.

    Integer WAV samples are converted to floating-point
    values and normalized.
    """

    sampling_rate, signal = wavfile.read(
        file_path
    )

    signal = np.asarray(signal)

    # Convert stereo to mono.
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
        Data type used by each I/Q component.

    iq_order:
        Either "IQ" or "QI".
    """

    raw_data = np.fromfile(
        file_path,
        dtype=iq_dtype,
    )

    if raw_data.size == 0:
        raise ValueError(
            "IQ file is empty."
        )

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
        + 1j
        * q_samples.astype(np.float64)
    )

    return signal


def estimate_phase_from_file(
    file_path: str | Path,
    frequency_hz: float | None = None,
    sampling_rate: float | None = None,
    iq_dtype: np.dtype = np.dtype(np.float32),
    iq_order: str = "IQ",
) -> dict[str, Any]:
    """
    Estimate phase directly from a WAV or IQ file.

    Parameters
    ----------
    file_path:
        Path to a .wav or .iq file.

    frequency_hz:
        Known carrier frequency.

        If None, the dominant frequency is estimated.

    sampling_rate:
        Required for raw IQ files.

        WAV files obtain their sampling rate
        from the file header.

    iq_dtype:
        Data type used by the raw IQ file.

    iq_order:
        Either "IQ" or "QI".

    Returns
    -------
    dict
        Phase estimation result and file metadata.
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension == ".wav":

        signal, file_sampling_rate = (
            _load_wav_file(
                file_path
            )
        )

        sampling_rate = (
            file_sampling_rate
        )

        result = estimate_phase(
            signal,
            sampling_rate,
            frequency_hz,
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

        result = estimate_phase(
            signal,
            sampling_rate,
            frequency_hz,
        )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Only .wav and .iq files are supported."
        )

    result["file"] = str(
        file_path
    )

    result["file_type"] = extension

    result["sampling_rate_hz"] = float(
        sampling_rate
    )

    return result