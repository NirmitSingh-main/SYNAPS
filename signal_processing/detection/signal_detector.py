from typing import Optional, Tuple

import numpy as np

from .energy_detection import (
    calculate_average_power,
    detect_energy,
)

from .segmentation import (
    find_signal_region,
    extract_signal,
)


def detect_signal(
    samples: np.ndarray,
    threshold: Optional[float] = None,
    threshold_factor: float = 1.0,
    minimum_samples: int = 1,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Detect and extract the active signal region.

    This function is the main interface for the signal
    detection stage.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    threshold : float, optional
        Absolute power threshold.

        If None, the threshold is calculated from the
        average signal power multiplied by threshold_factor.

    threshold_factor : float
        Multiplier used when threshold is not explicitly
        provided.

    minimum_samples : int
        Minimum number of detected samples required for
        a valid signal region.

    Returns
    -------
    detected_signal : numpy.ndarray
        Extracted signal samples.

    region : tuple
        (start_index, end_index)

        end_index is exclusive.

    Raises
    ------
    TypeError
        If samples is not a NumPy array.

    ValueError
        If samples are empty or no signal is detected.
    """

    # ---------------------------------------------------------
    # INPUT VALIDATION
    # ---------------------------------------------------------

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

    if threshold_factor < 0:
        raise ValueError(
            "threshold_factor cannot be negative."
        )

    if minimum_samples <= 0:
        raise ValueError(
            "minimum_samples must be greater than zero."
        )

    # ---------------------------------------------------------
    # DETERMINE THRESHOLD
    # ---------------------------------------------------------

    if threshold is None:

        average_power = calculate_average_power(
            samples
        )

        threshold = (
            average_power * threshold_factor
        )

    else:

        if not np.isfinite(threshold):
            raise ValueError(
                "threshold must be finite."
            )

        if threshold < 0:
            raise ValueError(
                "threshold cannot be negative."
            )

    # ---------------------------------------------------------
    # ENERGY DETECTION
    # ---------------------------------------------------------

    detection_mask = detect_energy(
        samples,
        threshold,
    )

    # ---------------------------------------------------------
    # FIND SIGNAL REGION
    # ---------------------------------------------------------

    region = find_signal_region(
        detection_mask,
        minimum_samples,
    )

    if region is None:
        raise ValueError(
            "No active signal detected."
        )

    # ---------------------------------------------------------
    # EXTRACT SIGNAL
    # ---------------------------------------------------------

    detected_signal = extract_signal(
        samples,
        detection_mask,
        minimum_samples,
    )

    return detected_signal, region