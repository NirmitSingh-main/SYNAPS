from pathlib import Path
from typing import Optional

import numpy as np

from .format_detection import detect_format
from .metadata import SignalMetadata


def validate_file(file_path: str) -> None:
    """
    Validate the input signal file.

    Checks:
    - File exists
    - Path is a file
    - File is not empty
    - File has a supported extension

    Supported formats:
    - .wav
    - .iq

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.

    ValueError
        If the path is not a file, the file is empty,
        or the format is unsupported.
    """

    path = Path(file_path)

    # Check whether the path exists
    if not path.exists():
        raise FileNotFoundError(
            f"Signal file not found: {file_path}"
        )

    # Check whether the path is actually a file
    if not path.is_file():
        raise ValueError(
            f"Input path is not a file: {file_path}"
        )

    # Check whether the file is empty
    if path.stat().st_size == 0:
        raise ValueError(
            f"Signal file is empty: {file_path}"
        )

    # Check whether the format is supported
    detect_format(file_path)


def validate_samples(
    samples: np.ndarray,
    sample_rate: float,
) -> None:
    """
    Validate loaded signal samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    sample_rate : float
        Sampling frequency in samples per second.

    Raises
    ------
    TypeError
        If samples are not a NumPy array.

    ValueError
        If samples are empty, contain invalid values,
        or the sampling rate is invalid.
    """

    # Check data type
    if not isinstance(samples, np.ndarray):
        raise TypeError(
            "Signal samples must be a NumPy array."
        )

    # Check for empty signal
    if samples.size == 0:
        raise ValueError(
            "Signal contains no samples."
        )

    # Check real component
    if not np.isfinite(samples.real).all():
        raise ValueError(
            "Signal contains invalid real values."
        )

    # Check imaginary component
    if not np.isfinite(samples.imag).all():
        raise ValueError(
            "Signal contains invalid imaginary values."
        )

    # Check sampling rate
    if not np.isfinite(sample_rate):
        raise ValueError(
            "Sampling frequency must be finite."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sampling frequency must be greater than zero."
        )


def validate_metadata(
    metadata: SignalMetadata,
) -> None:
    """
    Validate signal metadata.

    Checks the basic physical and file-related
    parameters required by the signal-processing pipeline.

    Units
    -----
    sample_rate       : samples/second
    carrier_frequency : Hz
    symbol_rate       : symbols/second
    bandwidth         : Hz
    frequency_offset  : Hz
    phase_offset      : radians
    signal_to_noise_ratio : dB
    """

    # Check metadata object type
    if not isinstance(metadata, SignalMetadata):
        raise TypeError(
            "metadata must be a SignalMetadata object."
        )

    # ---------------------------------------------------------
    # FILE INFORMATION
    # ---------------------------------------------------------

    if not metadata.signal_id:
        raise ValueError(
            "signal_id cannot be empty."
        )

    if not metadata.file_name:
        raise ValueError(
            "file_name cannot be empty."
        )

    if metadata.file_format not in {"WAV", "IQ"}:
        raise ValueError(
            "file_format must be either 'WAV' or 'IQ'."
        )

    # ---------------------------------------------------------
    # BASIC SIGNAL INFORMATION
    # ---------------------------------------------------------

    if not np.isfinite(metadata.sample_rate):
        raise ValueError(
            "sample_rate must be finite."
        )

    if metadata.sample_rate <= 0:
        raise ValueError(
            "sample_rate must be greater than zero."
        )

    if metadata.number_of_samples <= 0:
        raise ValueError(
            "number_of_samples must be greater than zero."
        )

    # ---------------------------------------------------------
    # SIGNAL PARAMETERS
    # ---------------------------------------------------------

    # Symbol rate must be positive
    _validate_optional_positive(
        metadata.symbol_rate,
        "symbol_rate",
    )

    # Bandwidth must be positive
    _validate_optional_positive(
        metadata.bandwidth,
        "bandwidth",
    )

    # Carrier frequency can be positive or negative
    _validate_optional_finite(
        metadata.carrier_frequency,
        "carrier_frequency",
    )

    # Frequency offset can be positive or negative
    _validate_optional_finite(
        metadata.frequency_offset,
        "frequency_offset",
    )

    # Phase can be positive or negative
    _validate_optional_finite(
        metadata.phase_offset,
        "phase_offset",
    )

    # Signal-to-noise ratio can be negative
    _validate_optional_finite(
        metadata.signal_to_noise_ratio,
        "signal_to_noise_ratio",
    )


def validate_signal_and_metadata(
    samples: np.ndarray,
    sample_rate: float,
    metadata: Optional[SignalMetadata] = None,
) -> None:
    """
    Validate the complete loaded signal.

    This is the main validation function that should be
    called before passing a signal to the next stage
    of the signal-processing pipeline.

    It validates:
    - Signal samples
    - Sampling rate
    - Metadata, if provided
    - Number of samples against metadata
    - Sampling rate against metadata
    """

    # Validate actual signal samples
    validate_samples(
        samples,
        sample_rate,
    )

    # Metadata is optional
    if metadata is not None:

        # Validate metadata itself
        validate_metadata(metadata)

        # Check sample count consistency
        if metadata.number_of_samples != len(samples):
            raise ValueError(
                "Number of samples does not match metadata."
            )

        # Check sampling-rate consistency
        if not np.isclose(
            metadata.sample_rate,
            sample_rate,
        ):
            raise ValueError(
                "Sampling frequency does not match metadata."
            )


def _validate_optional_finite(
    value: Optional[float],
    field_name: str,
) -> None:
    """
    Validate an optional numeric parameter that must be finite.

    Positive, zero, and negative values are allowed.

    This is appropriate for parameters such as:
    - Carrier frequency
    - Frequency offset
    - Phase offset
    - Signal-to-noise ratio
    """

    if value is not None:

        if not np.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite."
            )


def _validate_optional_positive(
    value: Optional[float],
    field_name: str,
) -> None:
    """
    Validate an optional numeric parameter that must be
    greater than zero when provided.

    Appropriate for:
    - Symbol rate
    - Bandwidth
    - Sampling rate
    """

    if value is not None:

        if not np.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than zero."
            )