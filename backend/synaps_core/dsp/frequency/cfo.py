"""
Carrier Frequency Offset estimation.

Supported input types:
    - Real-valued NumPy signals
    - Complex In-phase/Quadrature signals
    - WAV files
    - Raw IQ files

Supported modulation-aware CFO estimation:
    - FSK / 2FSK
    - BPSK
    - QPSK
    - QAM / 16QAM

Carrier Frequency Offset is calculated as:

    CFO = measured_frequency - reference_frequency

For symmetric 2FSK signals, the carrier center is estimated
from the midpoint between the lower and upper FSK tones.

For PSK signals, a power-law estimator is used to suppress
the modulation phase pattern and estimate the carrier rotation.

For QAM signals, the fourth-power estimator is used as a
carrier-frequency estimator, with a generic spectral fallback
when the fourth-power spectrum does not contain a reliable peak.

If no modulation is supplied, the generic estimator is used.

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
        raise ValueError(
            "Signal cannot be empty."
        )

    if signal.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not np.isfinite(
        sampling_rate
    ):
        raise ValueError(
            "Sampling rate must be finite."
        )

    if sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    if not np.all(
        np.isfinite(signal)
    ):
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


def _frequency_axis(
    number_of_samples: int,
    sampling_rate: float,
) -> np.ndarray:
    """
    Return a centered FFT frequency axis.
    """

    return np.fft.fftshift(
        np.fft.fftfreq(
            number_of_samples,
            d=1.0 / sampling_rate,
        )
    )


def _estimate_carrier_frequency(
    signal: np.ndarray,
    sampling_rate: float,
) -> float:
    """
    Estimate the dominant positive-frequency component.

    This is retained as the generic fallback estimator.

    Important:
        For FSK, the strongest tone is not necessarily the
        carrier center. The FSK-specific estimator should be
        used when modulation information is available.
    """

    signal = _prepare_signal(
        signal
    )

    number_of_samples = signal.size

    signal = (
        signal
        - np.mean(signal)
    )

    window = np.hanning(
        number_of_samples
    )

    windowed_signal = (
        signal * window
    )

    spectrum = np.fft.fft(
        windowed_signal
    )

    frequencies = np.fft.fftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )

    positive_mask = (
        frequencies >= 0
    )

    positive_spectrum = np.abs(
        spectrum[positive_mask]
    )

    positive_frequencies = (
        frequencies[positive_mask]
    )

    if positive_spectrum.size > 1:
        positive_spectrum[0] = 0.0

    if positive_spectrum.size == 0:
        return 0.0

    peak_index = int(
        np.argmax(
            positive_spectrum
        )
    )

    return float(
        positive_frequencies[
            peak_index
        ]
    )


def _estimate_power_law_frequency(
    signal: np.ndarray,
    sampling_rate: float,
    power: int,
) -> tuple[float, float]:
    """
    Estimate carrier frequency using a power-law transform.

    Returns:
        measured_frequency_hz
        peak_strength_ratio

    The transformed signal is:

        y[n] = x[n] ** power

    For PSK this suppresses much of the data-dependent phase
    structure and produces a spectral component related to
    the carrier-frequency rotation.

    The returned frequency is divided by `power` to recover
    the original carrier frequency.
    """

    if power < 2:
        raise ValueError(
            "Power must be at least 2."
        )

    signal = _prepare_signal(
        signal
    )

    signal = (
        signal
        - np.mean(signal)
    )

    if np.allclose(
        signal,
        0.0,
    ):
        return 0.0, 0.0

    transformed = (
        signal ** power
    )

    number_of_samples = (
        transformed.size
    )

    window = np.hanning(
        number_of_samples
    )

    spectrum = np.abs(
        np.fft.fftshift(
            np.fft.fft(
                transformed * window
            )
        )
    )

    frequencies = _frequency_axis(
        number_of_samples,
        sampling_rate,
    )

    if spectrum.size == 0:
        return 0.0, 0.0

    # Ignore DC.
    dc_index = int(
        np.argmin(
            np.abs(frequencies)
        )
    )

    spectrum[
        max(0, dc_index - 1):
        min(spectrum.size, dc_index + 2)
    ] = 0.0

    peak_index = int(
        np.argmax(
            spectrum
        )
    )

    peak_frequency = float(
        frequencies[peak_index]
    )

    peak_value = float(
        spectrum[peak_index]
    )

    nonzero_spectrum = spectrum[
        spectrum > 0.0
    ]

    if nonzero_spectrum.size == 0:
        return 0.0, 0.0

    median_level = float(
        np.median(
            nonzero_spectrum
        )
    )

    if median_level <= 0.0:
        peak_strength_ratio = float(
            "inf"
        )
    else:
        peak_strength_ratio = (
            peak_value
            / median_level
        )

    measured_frequency = (
        peak_frequency / power
    )

    return (
        float(measured_frequency),
        float(peak_strength_ratio),
    )


def _estimate_psk_frequency(
    signal: np.ndarray,
    sampling_rate: float,
    modulation: str,
) -> dict[str, float]:
    """
    Estimate carrier frequency for PSK.

    BPSK uses a second-power estimator.

    QPSK uses a fourth-power estimator.

    The transformed frequency is divided by the corresponding
    power to recover the original carrier-frequency estimate.
    """

    modulation_name = (
        modulation.upper()
    )

    if modulation_name in {
        "BPSK",
        "2PSK",
        "2-PSK",
    }:
        power = 2

    elif modulation_name in {
        "QPSK",
        "4PSK",
        "4-PSK",
    }:
        power = 4

    else:
        power = 4

    frequency, strength = (
        _estimate_power_law_frequency(
            signal,
            sampling_rate,
            power,
        )
    )

    return {
        "measured_frequency_hz": float(
            frequency
        ),
        "peak_strength_ratio": float(
            strength
        ),
    }


def _estimate_qam_frequency(
    signal: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """
    Estimate carrier frequency for QAM.

    A fourth-power estimator is used first. If the transformed
    spectrum does not contain a meaningful peak, the generic
    spectral estimator is used as a fallback.

    The generic fallback is retained because QAM data does not
    always produce a strong fourth-power spectral line.
    """

    power_law_frequency, strength = (
        _estimate_power_law_frequency(
            signal,
            sampling_rate,
            4,
        )
    )

    # A strong transformed spectral peak is preferred.
    if np.isfinite(strength) and strength >= 3.0:

        return {
            "measured_frequency_hz": float(
                power_law_frequency
            ),
            "peak_strength_ratio": float(
                strength
            ),
        }

    # Fall back to the original generic estimator.
    generic_frequency = (
        _estimate_carrier_frequency(
            signal,
            sampling_rate,
        )
    )

    return {
        "measured_frequency_hz": float(
            generic_frequency
        ),
        "peak_strength_ratio": float(
            strength
        ),
    }


def estimate_fsk_carrier_center(
    signal: np.ndarray,
    sampling_rate: float,
    reference_frequency_hz: float = 0.0,
) -> dict[str, Any]:
    """
    Estimate carrier center frequency, CFO, tone separation,
    and frequency deviation for symmetric 2FSK signals.

    For symmetric 2FSK:

        f_center = (f_upper + f_lower) / 2

        cfo_hz = f_center - reference_frequency_hz

        tone_separation_hz = f_upper - f_lower

        frequency_deviation_hz = tone_separation_hz / 2

    The carrier center is used for CFO rather than the strongest
    individual FSK tone.
    """

    signal = _validate_signal(
        signal,
        sampling_rate,
    )

    signal = _prepare_signal(
        signal
    )

    signal = (
        signal
        - np.mean(signal)
    )

    num_samples = signal.size

    windowed = (
        signal
        * np.hanning(
            num_samples
        )
    )

    spec = np.abs(
        np.fft.fftshift(
            np.fft.fft(
                windowed
            )
        )
    )

    freqs = _frequency_axis(
        num_samples,
        sampling_rate,
    )

    neg_mask = (
        freqs < -5000.0
    )

    pos_mask = (
        freqs > 5000.0
    )

    if (
        np.any(neg_mask)
        and np.any(pos_mask)
    ):

        neg_spec = spec[
            neg_mask
        ]

        neg_freqs = freqs[
            neg_mask
        ]

        pos_spec = spec[
            pos_mask
        ]

        pos_freqs = freqs[
            pos_mask
        ]

        lower_index = int(
            np.argmax(
                neg_spec
            )
        )

        upper_index = int(
            np.argmax(
                pos_spec
            )
        )

        f_lower = float(
            neg_freqs[
                lower_index
            ]
        )

        f_upper = float(
            pos_freqs[
                upper_index
            ]
        )

        f_center = float(
            (f_upper + f_lower)
            / 2.0
        )

        cfo_hz = float(
            f_center
            - reference_frequency_hz
        )

        tone_sep = float(
            f_upper - f_lower
        )

        f_dev = float(
            tone_sep / 2.0
        )

    else:

        peak_idx = int(
            np.argmax(spec)
        )

        f_center = float(
            freqs[peak_idx]
        )

        cfo_hz = float(
            f_center
            - reference_frequency_hz
        )

        f_lower = f_center
        f_upper = f_center
        tone_sep = 0.0
        f_dev = 0.0

    return {
        "carrier_center_hz": float(
            f_center
        ),
        "cfo_hz": float(
            cfo_hz
        ),
        "reference_frequency_hz": float(
            reference_frequency_hz
        ),
        "dominant_lower_tone_hz": float(
            f_lower
        ),
        "dominant_upper_tone_hz": float(
            f_upper
        ),
        "tone_separation_hz": float(
            tone_sep
        ),
        "frequency_deviation_hz": float(
            f_dev
        ),
    }


def estimate_cfo(
    signal: np.ndarray,
    sampling_rate: float,
    reference_frequency_hz: float = 0.0,
    modulation: str | None = None,
) -> dict[str, Any]:
    """
    Estimate Carrier Frequency Offset.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    sampling_rate:
        Signal sampling rate in Hertz.

    reference_frequency_hz:
        Expected/reference carrier frequency in Hertz.

    modulation:
        Optional modulation type.

        Supported modulation-aware values include:

            FSK
            2FSK
            BPSK
            QPSK
            QAM
            16QAM

        If omitted or unknown, the generic estimator is used.

    Returns
    -------
    dict
        Dictionary containing:

            measured_frequency_hz
            reference_frequency_hz
            cfo_hz

        Additional modulation-specific information may also
        be returned.
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

    modulation_name = (
        str(modulation)
        .strip()
        .upper()
        if modulation is not None
        else ""
    )

    # ---------------------------------------------------------
    # FSK
    # ---------------------------------------------------------
    if modulation_name in {
        "FSK",
        "2FSK",
        "2-FSK",
    }:

        result = (
            estimate_fsk_carrier_center(
                signal,
                sampling_rate,
                reference_frequency_hz=(
                    reference_frequency_hz
                ),
            )
        )

        return {
            "measured_frequency_hz": float(
                result[
                    "carrier_center_hz"
                ]
            ),
            "reference_frequency_hz": float(
                reference_frequency_hz
            ),
            "cfo_hz": float(
                result["cfo_hz"]
            ),
            "carrier_center_hz": float(
                result[
                    "carrier_center_hz"
                ]
            ),
            "dominant_lower_tone_hz": float(
                result[
                    "dominant_lower_tone_hz"
                ]
            ),
            "dominant_upper_tone_hz": float(
                result[
                    "dominant_upper_tone_hz"
                ]
            ),
            "tone_separation_hz": float(
                result[
                    "tone_separation_hz"
                ]
            ),
            "frequency_deviation_hz": float(
                result[
                    "frequency_deviation_hz"
                ]
            ),
            "estimator": "fsk_carrier_center",
        }

    # ---------------------------------------------------------
    # BPSK / QPSK
    # ---------------------------------------------------------
    if modulation_name in {
        "BPSK",
        "2PSK",
        "2-PSK",
        "QPSK",
        "4PSK",
        "4-PSK",
    }:

        result = _estimate_psk_frequency(
            signal,
            sampling_rate,
            modulation_name,
        )

        measured_frequency_hz = float(
            result[
                "measured_frequency_hz"
            ]
        )

        cfo_hz = float(
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
            "cfo_hz": float(
                cfo_hz
            ),
            "estimator": "power_law",
            "power": (
                2
                if modulation_name in {
                    "BPSK",
                    "2PSK",
                    "2-PSK",
                }
                else 4
            ),
            "peak_strength_ratio": float(
                result[
                    "peak_strength_ratio"
                ]
            ),
        }

    # ---------------------------------------------------------
    # QAM
    # ---------------------------------------------------------
    if modulation_name in {
        "QAM",
        "16QAM",
        "64QAM",
        "256QAM",
    }:

        result = _estimate_qam_frequency(
            signal,
            sampling_rate,
        )

        measured_frequency_hz = float(
            result[
                "measured_frequency_hz"
            ]
        )

        cfo_hz = float(
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
            "cfo_hz": float(
                cfo_hz
            ),
            "estimator": (
                "qam_fourth_power"
                if (
                    np.isfinite(
                        result[
                            "peak_strength_ratio"
                        ]
                    )
                    and result[
                        "peak_strength_ratio"
                    ] >= 3.0
                )
                else "generic_fallback"
            ),
            "peak_strength_ratio": float(
                result[
                    "peak_strength_ratio"
                ]
            ),
        }

    # ---------------------------------------------------------
    # Generic fallback
    # ---------------------------------------------------------
    measured_frequency_hz = (
        _estimate_carrier_frequency(
            signal,
            sampling_rate,
        )
    )

    cfo_hz = float(
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
        "cfo_hz": float(
            cfo_hz
        ),
        "estimator": "generic_peak",
    }


def _load_wav_file(
    file_path: Path,
) -> tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Stereo WAV files are converted to mono.

    Integer WAV samples are converted to floating
    point and normalized.
    """

    sampling_rate, signal = wavfile.read(
        file_path
    )

    signal = np.asarray(
        signal
    )

    if signal.ndim == 2:

        signal = np.mean(
            signal.astype(
                np.float64
            ),
            axis=1,
        )

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
    iq_dtype: np.dtype = np.dtype(
        np.float32
    ),
    iq_order: str = "IQ",
) -> np.ndarray:
    """
    Load a raw interleaved IQ file.

    Expected format:

        I, Q, I, Q, I, Q, ...
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

    first_component = raw_data[
        0::2
    ]

    second_component = raw_data[
        1::2
    ]

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
        i_samples.astype(
            np.float64
        )
        + 1j
        * q_samples.astype(
            np.float64
        )
    )

    return signal


def estimate_cfo_from_file(
    file_path: str | Path,
    reference_frequency_hz: float,
    sampling_rate: float | None = None,
    iq_dtype: np.dtype = np.dtype(
        np.float32
    ),
    iq_order: str = "IQ",
    modulation: str | None = None,
) -> dict[str, Any]:
    """
    Estimate Carrier Frequency Offset directly
    from a WAV or IQ file.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = (
        file_path.suffix.lower()
    )

    if extension == ".wav":

        signal, file_sampling_rate = (
            _load_wav_file(
                file_path
            )
        )

        sampling_rate = (
            file_sampling_rate
        )

        result = estimate_cfo(
            signal,
            sampling_rate,
            reference_frequency_hz,
            modulation=modulation,
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
            modulation=modulation,
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