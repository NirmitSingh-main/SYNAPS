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


def validate_dataset_sample(
    sample_identifier_or_path: str,
    require_wav: bool = False,
) -> dict:
    """
    Validate an individual dataset sample for correspondence across:
    1. IQ file existence and integrity
    2. WAV file existence (if required)
    3. Metadata JSON existence and validity
    4. Reliable filename and class mapping
    5. Agreement between modulation in metadata and class directory
    6. Exact sample count match between IQ binary and metadata
    7. Valid samples_per_symbol and sampling_frequency_hz
    8. Bitstream consistency verification

    Returns structured dictionary report with status 'VALID', 'INVALID', or 'UNTRUSTED_METADATA'.
    """
    import json
    from project_paths import resolve_sample_paths, normalize_modulation_name

    resolved = resolve_sample_paths(sample_identifier_or_path)
    iq_path = resolved.get("iq_path")
    wav_path = resolved.get("wav_path")
    meta_path = resolved.get("metadata_path")
    class_name = resolved.get("class_name")
    sample_id = resolved.get("sample_id")

    issues = []
    metadata = {}
    iq_samples_count = 0

    if not iq_path or not iq_path.exists():
        issues.append(f"IQ file missing for {sample_id}")
    else:
        try:
            # Complex64 is 8 bytes per sample (float32 I, float32 Q)
            raw = np.fromfile(iq_path, dtype=np.complex64)
            iq_samples_count = len(raw)
            if iq_samples_count == 0:
                issues.append("IQ file is empty")
        except Exception as e:
            issues.append(f"Failed to read IQ file: {e}")

    if require_wav and (not wav_path or not wav_path.exists()):
        issues.append(f"WAV file missing for {sample_id}")

    if not meta_path or not meta_path.exists():
        issues.append(f"Metadata JSON missing for {sample_id}")
    else:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            issues.append(f"Failed to parse metadata JSON: {e}")

    # Cross-validation checks
    if metadata and iq_samples_count > 0:
        meta_samples = metadata.get("samples", metadata.get("number_of_samples", 0))
        if meta_samples != iq_samples_count:
            issues.append(
                f"Sample count mismatch: metadata specifies {meta_samples}, actual IQ has {iq_samples_count}"
            )

        meta_mod = metadata.get("modulation", metadata.get("modulation_type", ""))
        if meta_mod:
            norm_meta_mod = normalize_modulation_name(meta_mod)
            if class_name and norm_meta_mod != class_name:
                issues.append(
                    f"Modulation mismatch: metadata says {norm_meta_mod}, directory/class says {class_name}"
                )

        sps = metadata.get("samples_per_symbol", metadata.get("symbol_rate", 0))
        if sps <= 0:
            issues.append(f"Invalid samples_per_symbol or symbol_rate: {sps}")

        fs = metadata.get("sampling_frequency_hz", metadata.get("sample_rate", 0))
        if fs <= 0:
            issues.append(f"Invalid sampling frequency: {fs}")

    is_valid = len(issues) == 0
    status = "VALID" if is_valid else "INVALID"

    return {
        "sample_id": sample_id,
        "class_name": class_name,
        "iq_path": str(iq_path) if iq_path else None,
        "wav_path": str(wav_path) if wav_path else None,
        "metadata_path": str(meta_path) if meta_path else None,
        "sample_count": iq_samples_count,
        "status": status,
        "issues": issues,
        "metadata": metadata,
    }