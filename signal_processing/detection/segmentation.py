from typing import Optional, Tuple

import numpy as np


def find_signal_region(
    detection_mask: np.ndarray,
    minimum_samples: int = 1,
) -> Optional[Tuple[int, int]]:
    """
    Find the first and last detected samples.

    Parameters
    ----------
    detection_mask : numpy.ndarray
        Boolean array where True represents detected signal.

    minimum_samples : int
        Minimum number of detected samples required
        to consider a signal region valid.

    Returns
    -------
    tuple or None
        (start_index, end_index)

        end_index is exclusive, so the region can be
        directly used as:

            samples[start_index:end_index]

        Returns None if no valid signal region exists.
    """

    if not isinstance(detection_mask, np.ndarray):
        raise TypeError(
            "detection_mask must be a NumPy array."
        )

    if detection_mask.size == 0:
        raise ValueError(
            "Detection mask is empty."
        )

    if minimum_samples <= 0:
        raise ValueError(
            "minimum_samples must be greater than zero."
        )

    # Make sure the mask is Boolean.
    mask = detection_mask.astype(bool)

    detected_indices = np.flatnonzero(mask)

    if detected_indices.size < minimum_samples:
        return None

    start_index = int(detected_indices[0])
    end_index = int(detected_indices[-1]) + 1

    return start_index, end_index


def extract_signal(
    samples: np.ndarray,
    detection_mask: np.ndarray,
    minimum_samples: int = 1,
) -> np.ndarray:
    """
    Extract the detected signal region from the input samples.

    Parameters
    ----------
    samples : numpy.ndarray
        Complex-valued signal samples.

    detection_mask : numpy.ndarray
        Boolean detection mask.

    minimum_samples : int
        Minimum number of detected samples required.

    Returns
    -------
    numpy.ndarray
        Extracted signal region.

    Raises
    ------
    ValueError
        If no valid signal region is detected.
    """

    if not isinstance(samples, np.ndarray):
        raise TypeError(
            "samples must be a NumPy array."
        )

    if samples.size == 0:
        raise ValueError(
            "samples contains no data."
        )

    if len(samples) != len(detection_mask):
        raise ValueError(
            "samples and detection_mask must have "
            "the same length."
        )

    region = find_signal_region(
        detection_mask,
        minimum_samples,
    )

    if region is None:
        raise ValueError(
            "No valid signal region detected."
        )

    start_index, end_index = region

    return samples[start_index:end_index]


def get_signal_duration(
    start_index: int,
    end_index: int,
    sample_rate: float,
) -> float:
    """
    Calculate the duration of a detected signal region.

    Parameters
    ----------
    start_index : int
        Starting sample index.

    end_index : int
        Ending sample index (exclusive).

    sample_rate : float
        Sampling frequency in samples per second.

    Returns
    -------
    float
        Signal duration in seconds.
    """

    if start_index < 0:
        raise ValueError(
            "start_index cannot be negative."
        )

    if end_index <= start_index:
        raise ValueError(
            "end_index must be greater than start_index."
        )

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be greater than zero."
        )

    number_of_samples = end_index - start_index

    return number_of_samples / float(sample_rate)