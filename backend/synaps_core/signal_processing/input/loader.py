from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile

from .format_detection import detect_format


def load_signal(
    file_path: str,
    iq_dtype: np.dtype = np.complex64,
    iq_sample_rate: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Load a synthetic WAV or IQ signal.

    Parameters
    ----------
    file_path : str
        Path to the input .wav or .iq file.

    iq_dtype : numpy dtype, optional
        Internal data type used for IQ samples.
        Default is complex64.

    iq_sample_rate : float, optional
        Sampling rate for raw .iq files.
        Required because raw IQ files do not contain
        sampling-rate information.

    Returns
    -------
    samples : numpy.ndarray
        Complex-valued signal samples.

    sample_rate : float
        Sampling frequency in samples per second.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    ValueError
        If the file format or IQ configuration is invalid.
    """

    path = Path(file_path)

    # Check file existence
    if not path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {file_path}"
        )

    # Check that it is actually a file
    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {file_path}"
        )

    # Use the central format detector
    signal_format = detect_format(file_path)

    if signal_format == "WAV":
        return _load_wav(path, iq_dtype)

    if signal_format == "IQ":
        if iq_sample_rate is None:
            # Try to resolve from metadata or fallback to standard 1MHz
            try:
                from project_paths import resolve_sample_paths
                import json
                resolved = resolve_sample_paths(path)
                if resolved.get("metadata_path") and resolved["metadata_path"].exists():
                    with open(resolved["metadata_path"], "r", encoding="utf-8") as f:
                        meta_dict = json.load(f)
                    iq_sample_rate = float(meta_dict.get("sampling_frequency_hz", meta_dict.get("sample_rate", 1_000_000.0)))
                else:
                    iq_sample_rate = 1_000_000.0
            except Exception:
                iq_sample_rate = 1_000_000.0

        return _load_iq(
            path,
            iq_dtype,
            iq_sample_rate,
        )

    # This should normally never be reached because
    # detect_format() already validates the format.
    raise ValueError(
        f"Unsupported signal format: {signal_format}"
    )


def _load_wav(
    path: Path,
    iq_dtype: np.dtype,
) -> Tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Expected synthetic WAV format:
        Channel 1 -> I
        Channel 2 -> Q

    A mono WAV file is also accepted and treated as
    a real-valued signal with Q = 0.
    """

    sample_rate, data = wavfile.read(path)

    data = np.asarray(data)

    # Mono WAV
    if data.ndim == 1:

        samples = (
            data.astype(np.float32)
            .astype(iq_dtype)
        )

    # Stereo WAV: I + Q
    elif data.ndim == 2 and data.shape[1] == 2:

        i = data[:, 0].astype(np.float32)
        q = data[:, 1].astype(np.float32)

        samples = (
            i + 1j * q
        ).astype(iq_dtype)

    else:
        raise ValueError(
            "WAV file must contain either one channel "
            "or exactly two channels (I and Q)."
        )

    return samples, float(sample_rate)


def _load_iq(
    path: Path,
    iq_dtype: np.dtype,
    sample_rate: float,
) -> Tuple[np.ndarray, float]:
    """
    Load a raw IQ file.

    Expected synthetic format:

        I0, Q0, I1, Q1, I2, Q2, ...

    Samples are assumed to be stored as float32.
    """

    raw_data = np.fromfile(
        path,
        dtype=np.float32,
    )

    # Empty IQ file
    if raw_data.size == 0:
        raise ValueError(
            "IQ file is empty."
        )

    # Every I value must have a corresponding Q value
    if raw_data.size % 2 != 0:
        raise ValueError(
            "Invalid IQ file: the number of values "
            "must be even because samples are stored "
            "as I/Q pairs."
        )

    i = raw_data[0::2]
    q = raw_data[1::2]

    samples = (
        i + 1j * q
    ).astype(iq_dtype)

    return samples, float(sample_rate)