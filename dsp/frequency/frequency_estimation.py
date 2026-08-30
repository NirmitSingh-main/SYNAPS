"""
Frequency estimation for WAV and raw IQ signals.

This module:
1. Loads WAV and raw IQ files.
2. Converts them into a common NumPy representation.
3. Estimates the dominant signal frequency using FFT.
4. Refines the FFT peak using quadratic interpolation.
5. Returns structured frequency-analysis results.

CFO (Carrier Frequency Offset) estimation is intentionally NOT
implemented here. It belongs in cfo.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SUPPORTED_FILE_TYPES = {".wav", ".iq"}

DEFAULT_IQ_DTYPE = np.float32
DEFAULT_IQ_SCALE = 1.0

# Number of samples used when a very large signal is supplied.
DEFAULT_MAX_SAMPLES = 2_000_000

# Zero-padding factor.
# Larger values give a denser frequency grid but require more computation.
DEFAULT_ZERO_PADDING_FACTOR = 4


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _validate_file_path(file_path: str | Path) -> Path:
    """
    Validate the input file path.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported types: {sorted(SUPPORTED_FILE_TYPES)}"
        )

    return path


# ---------------------------------------------------------------------
# Signal normalization
# ---------------------------------------------------------------------

def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Convert a signal into a floating-point NumPy representation.

    Complex signals remain complex.
    Real signals remain real.
    """

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal contains no samples.")

    if np.iscomplexobj(signal):
        signal = signal.astype(np.complex128, copy=False)
    else:
        signal = signal.astype(np.float64, copy=False)

    # Remove DC component.
    signal = signal - np.mean(signal)

    # Normalize amplitude.
    max_amplitude = np.max(np.abs(signal))

    if max_amplitude > 0:
        signal = signal / max_amplitude

    return signal


# ---------------------------------------------------------------------
# WAV loading
# ---------------------------------------------------------------------

def _load_wav(
    file_path: Path,
) -> tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Supported WAV layouts:

    Mono:
        real-valued audio samples

    Stereo:
        channel 0 = I
        channel 1 = Q

    Returns:
        signal, sampling_rate
    """

    sampling_rate, data = wavfile.read(file_path)

    if sampling_rate <= 0:
        raise ValueError("WAV file has an invalid sampling rate.")

    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("WAV file contains no samples.")

    # Mono WAV -> real signal
    if data.ndim == 1:
        signal = data.astype(np.float64)

    # Stereo/multi-channel WAV
    elif data.ndim == 2:

        if data.shape[1] == 2:
            # Interpret stereo WAV as I/Q.
            i_component = data[:, 0].astype(np.float64)
            q_component = data[:, 1].astype(np.float64)

            signal = i_component + 1j * q_component

        else:
            # For more than two channels, use first channel.
            signal = data[:, 0].astype(np.float64)

    else:
        raise ValueError("Unsupported WAV data structure.")

    return signal, float(sampling_rate)


# ---------------------------------------------------------------------
# Raw IQ loading
# ---------------------------------------------------------------------

def _load_iq(
    file_path: Path,
    sampling_rate: float,
    dtype: np.dtype | type = DEFAULT_IQ_DTYPE,
    iq_order: str = "IQ",
    scale: float = DEFAULT_IQ_SCALE,
) -> tuple[np.ndarray, float]:
    """
    Load a raw IQ file.

    Expected raw layout:

        I, Q, I, Q, I, Q, ...

    Parameters
    ----------
    file_path:
        Path to the .iq file.

    sampling_rate:
        Sampling rate in samples/second.

    dtype:
        Raw numeric type stored in the file.
        Common choices:
            np.float32
            np.int16
            np.int8

    iq_order:
        "IQ" for I,Q,I,Q...
        "QI" for Q,I,Q,I...

    scale:
        Optional amplitude scaling factor.

    Returns
    -------
    complex_signal, sampling_rate
    """

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    iq_order = iq_order.upper()

    if iq_order not in {"IQ", "QI"}:
        raise ValueError("iq_order must be either 'IQ' or 'QI'.")

    raw_data = np.fromfile(file_path, dtype=dtype)

    if raw_data.size == 0:
        raise ValueError("IQ file contains no samples.")

    # I/Q data must contain pairs.
    if raw_data.size % 2 != 0:
        raise ValueError(
            "Raw IQ file contains an odd number of values. "
            "Expected interleaved I/Q pairs."
        )

    raw_data = raw_data.astype(np.float64)

    i_samples = raw_data[0::2]
    q_samples = raw_data[1::2]

    if iq_order == "QI":
        i_samples, q_samples = q_samples, i_samples

    signal = i_samples + 1j * q_samples

    if scale <= 0:
        raise ValueError("scale must be greater than zero.")

    signal = signal * scale

    return signal, float(sampling_rate)


# ---------------------------------------------------------------------
# Unified signal loader
# ---------------------------------------------------------------------

def load_signal(
    file_path: str | Path,
    sampling_rate: float | None = None,
    iq_dtype: np.dtype | type = DEFAULT_IQ_DTYPE,
    iq_order: str = "IQ",
    iq_scale: float = DEFAULT_IQ_SCALE,
) -> tuple[np.ndarray, float]:
    """
    Load either a WAV or IQ signal.

    For WAV:
        sampling_rate is obtained from the WAV header.

    For raw IQ:
        sampling_rate MUST be supplied.

    Returns:
        signal, sampling_rate
    """

    path = _validate_file_path(file_path)

    extension = path.suffix.lower()

    if extension == ".wav":
        signal, detected_sampling_rate = _load_wav(path)

        if sampling_rate is not None:
            if sampling_rate <= 0:
                raise ValueError("sampling_rate must be greater than zero.")

            # Explicit sampling rate may be used to override the WAV value.
            detected_sampling_rate = float(sampling_rate)

        return _normalize_signal(signal), detected_sampling_rate

    if extension == ".iq":

        if sampling_rate is None:
            raise ValueError(
                "sampling_rate must be provided when loading a raw .iq file."
            )

        signal, detected_sampling_rate = _load_iq(
            path,
            sampling_rate=sampling_rate,
            dtype=iq_dtype,
            iq_order=iq_order,
            scale=iq_scale,
        )

        return _normalize_signal(signal), detected_sampling_rate

    raise ValueError("Unsupported signal format.")


# ---------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------

def _limit_signal_length(
    signal: np.ndarray,
    max_samples: int,
) -> np.ndarray:
    """
    Limit the number of samples used for frequency estimation.

    The signal is taken from the beginning for deterministic behaviour.
    """

    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero.")

    if len(signal) <= max_samples:
        return signal

    return signal[:max_samples]


# ---------------------------------------------------------------------
# Quadratic peak interpolation
# ---------------------------------------------------------------------

def _quadratic_peak_interpolation(
    magnitude: np.ndarray,
    peak_index: int,
) -> float:
    """
    Refine the FFT peak location using quadratic interpolation.

    Returns:
        Fractional correction relative to peak_index.
    """

    if peak_index <= 0 or peak_index >= len(magnitude) - 1:
        return 0.0

    alpha = magnitude[peak_index - 1]
    beta = magnitude[peak_index]
    gamma = magnitude[peak_index + 1]

    denominator = alpha - 2.0 * beta + gamma

    if abs(denominator) < 1e-15:
        return 0.0

    correction = 0.5 * (alpha - gamma) / denominator

    # Prevent unreasonable interpolation.
    return float(np.clip(correction, -0.5, 0.5))


# ---------------------------------------------------------------------
# Frequency estimation
# ---------------------------------------------------------------------

def estimate_frequency(
    signal: np.ndarray,
    sampling_rate: float,
    zero_padding_factor: int = DEFAULT_ZERO_PADDING_FACTOR,
) -> dict[str, Any]:
    """
    Estimate the dominant frequency of a signal.

    Parameters
    ----------
    signal:
        Real or complex signal samples.

    sampling_rate:
        Sampling rate in samples/second.

    zero_padding_factor:
        Factor used to increase FFT length.

    Returns
    -------
    Dictionary containing:

        estimated_frequency_hz
        frequency_resolution_hz
        peak_magnitude
        peak_index
        signal_type
        number_of_samples
    """

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be greater than zero.")

    signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal contains no samples.")

    if zero_padding_factor < 1:
        raise ValueError("zero_padding_factor must be at least 1.")

    # Normalize and remove DC.
    signal = _normalize_signal(signal)

    number_of_samples = len(signal)

    # Use a Hann window to reduce spectral leakage.
    window = np.hanning(number_of_samples)

    windowed_signal = signal * window

    # Zero-padding.
    fft_length = int(
        2 ** np.ceil(
            np.log2(number_of_samples * zero_padding_factor)
        )
    )

    spectrum = np.fft.fft(windowed_signal, n=fft_length)

    # Complex IQ signals need the full signed spectrum.
    if np.iscomplexobj(signal):

        spectrum = np.fft.fftshift(spectrum)

        frequencies = np.fft.fftshift(
            np.fft.fftfreq(
                fft_length,
                d=1.0 / sampling_rate,
            )
        )

    else:

        # Real signals have redundant negative-frequency content.
        positive_length = fft_length // 2 + 1

        spectrum = spectrum[:positive_length]

        frequencies = np.fft.rfftfreq(
            fft_length,
            d=1.0 / sampling_rate,
        )

    magnitude = np.abs(spectrum)

    # Ignore the DC bin when searching for the dominant signal.
    if len(magnitude) > 1:
        search_magnitude = magnitude.copy()

        dc_index = np.argmin(np.abs(frequencies))

        search_magnitude[dc_index] = 0.0
    else:
        search_magnitude = magnitude

    peak_index = int(np.argmax(search_magnitude))

    peak_magnitude = float(magnitude[peak_index])

    # Refine peak position.
    correction = _quadratic_peak_interpolation(
        magnitude,
        peak_index,
    )

    frequency_resolution = sampling_rate / fft_length

    estimated_frequency = (
        frequencies[peak_index]
        + correction * frequency_resolution
    )

    signal_type = (
        "complex_iq"
        if np.iscomplexobj(signal)
        else "real"
    )

    return {
        "estimated_frequency_hz": float(estimated_frequency),
        "frequency_resolution_hz": float(frequency_resolution),
        "peak_magnitude": peak_magnitude,
        "peak_index": peak_index,
        "signal_type": signal_type,
        "number_of_samples": number_of_samples,
        "fft_length": fft_length,
    }


# ---------------------------------------------------------------------
# Complete file-based frequency estimation
# ---------------------------------------------------------------------

def estimate_frequency_from_file(
    file_path: str | Path,
    sampling_rate: float | None = None,
    iq_dtype: np.dtype | type = DEFAULT_IQ_DTYPE,
    iq_order: str = "IQ",
    iq_scale: float = DEFAULT_IQ_SCALE,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    zero_padding_factor: int = DEFAULT_ZERO_PADDING_FACTOR,
) -> dict[str, Any]:
    """
    Load a WAV/IQ file and estimate its dominant frequency.

    This is the main function that other parts of the project
    should call.
    """

    signal, detected_sampling_rate = load_signal(
        file_path=file_path,
        sampling_rate=sampling_rate,
        iq_dtype=iq_dtype,
        iq_order=iq_order,
        iq_scale=iq_scale,
    )

    signal = _limit_signal_length(
        signal,
        max_samples=max_samples,
    )

    result = estimate_frequency(
        signal=signal,
        sampling_rate=detected_sampling_rate,
        zero_padding_factor=zero_padding_factor,
    )

    # Add input metadata.
    result["file"] = str(file_path)
    result["file_type"] = Path(file_path).suffix.lower()
    result["sampling_rate_hz"] = float(detected_sampling_rate)

    return result


# ---------------------------------------------------------------------
# Simple command-line usage
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate dominant frequency from WAV or IQ signal."
    )

    parser.add_argument(
        "file",
        help="Path to .wav or .iq file",
    )

    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=None,
        help="Sampling rate for raw .iq files.",
    )

    parser.add_argument(
        "--iq-dtype",
        choices=["float32", "int16", "int8"],
        default="float32",
        help="Raw IQ data type.",
    )

    parser.add_argument(
        "--iq-order",
        choices=["IQ", "QI"],
        default="IQ",
        help="Interleaving order of raw IQ samples.",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": np.float32,
        "int16": np.int16,
        "int8": np.int8,
    }

    result = estimate_frequency_from_file(
        file_path=args.file,
        sampling_rate=args.sampling_rate,
        iq_dtype=dtype_map[args.iq_dtype],
        iq_order=args.iq_order,
    )

    print("\nFrequency Estimation Result")
    print("-" * 35)

    for key, value in result.items():
        print(f"{key}: {value}")