"""
Higher-Order Cumulants (HOC) for communication signals.

This module calculates higher-order statistical features from
real-valued and complex IQ signals.

Higher-order cumulants are useful in communication-signal analysis
because they can help distinguish modulation types while being
relatively insensitive to Gaussian noise.

Supported inputs:
    - Real-valued NumPy arrays
    - Complex IQ NumPy arrays

Main functions:
    calculate_hoc()
    higher_order_cumulants()
    hoc()
"""


from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def _validate_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Validate and convert the input signal.
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

    if not np.all(
        np.isfinite(signal)
    ):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if np.iscomplexobj(signal):

        return np.asarray(
            signal,
            dtype=np.complex128,
        )

    return np.asarray(
        signal,
        dtype=np.float64,
    )


# ---------------------------------------------------------------------
# Moment calculation
# ---------------------------------------------------------------------

def _calculate_moments(
    signal: np.ndarray,
) -> dict[str, complex]:
    """
    Calculate second-, fourth-, and sixth-order moments.

    The signal is centered before calculating the moments.
    """

    signal = np.asarray(
        signal,
        dtype=np.complex128,
    )

    # Remove the mean.
    centered = (
        signal
        - np.mean(signal)
    )

    conjugate = np.conjugate(
        centered
    )

    moments: dict[str, complex] = {
        "m20": complex(
            np.mean(centered ** 2)
        ),

        "m21": complex(
            np.mean(
                centered * conjugate
            )
        ),

        "m22": complex(
            np.mean(
                centered ** 2
                * conjugate ** 2
            )
        ),

        "m40": complex(
            np.mean(centered ** 4)
        ),

        "m41": complex(
            np.mean(
                centered ** 3
                * conjugate
            )
        ),

        "m42": complex(
            np.mean(
                centered ** 2
                * conjugate ** 2
            )
        ),

        "m60": complex(
            np.mean(centered ** 6)
        ),

        "m61": complex(
            np.mean(
                centered ** 5
                * conjugate
            )
        ),

        "m62": complex(
            np.mean(
                centered ** 4
                * conjugate ** 2
            )
        ),

        "m63": complex(
            np.mean(
                centered ** 3
                * conjugate ** 3
            )
        ),
    }

    return moments


# ---------------------------------------------------------------------
# Cumulant calculation
# ---------------------------------------------------------------------

def _calculate_cumulants(
    moments: dict[str, complex],
) -> dict[str, complex]:
    """
    Calculate higher-order cumulants from signal moments.

    The commonly used complex cumulants are:

        C20
        C21
        C40
        C41
        C42
        C60
        C61
        C62
        C63
    """

    m20 = moments["m20"]
    m21 = moments["m21"]

    m40 = moments["m40"]
    m41 = moments["m41"]
    m42 = moments["m42"]

    m60 = moments["m60"]
    m61 = moments["m61"]
    m62 = moments["m62"]
    m63 = moments["m63"]

    # Second-order cumulants.
    c20 = m20

    c21 = m21

    # Fourth-order cumulants.
    c40 = (
        m40
        - 3.0 * (m20 ** 2)
    )

    c41 = (
        m41
        - 3.0 * m20 * m21
    )

    c42 = (
        m42
        - abs(m20) ** 2
        - 2.0 * (m21 ** 2)
    )

    # Sixth-order cumulants.
    c60 = (
        m60
        - 15.0 * m40 * m20
        + 30.0 * (m20 ** 3)
    )

    c61 = (
        m61
        - 10.0 * m40 * m21
        - 5.0 * m41 * m20
        + 30.0 * m20 * m21 * m20
    )

    c62 = (
        m62
        - 6.0 * m42 * m20
        - 8.0 * m41 * m21
        - 3.0 * m40 * np.conjugate(m20)
        + 24.0 * (m20 ** 2) * m21
        + 6.0 * m20 * (m21 ** 2)
    )

    c63 = (
        m63
        - 9.0 * m42 * m21
        + 12.0 * (m21 ** 3)
        - 3.0 * abs(m20) ** 2 * m21
    )

    return {
        "C20": complex(c20),
        "C21": complex(c21),
        "C40": complex(c40),
        "C41": complex(c41),
        "C42": complex(c42),
        "C60": complex(c60),
        "C61": complex(c61),
        "C62": complex(c62),
        "C63": complex(c63),
    }


# ---------------------------------------------------------------------
# Normalized cumulants
# ---------------------------------------------------------------------

def _calculate_normalized_cumulants(
    cumulants: dict[str, complex],
) -> dict[str, float]:
    """
    Calculate normalized magnitudes of higher-order cumulants.

    Normalization uses C21, the signal power.

    This makes the values less dependent on signal amplitude.
    """

    power = abs(
        cumulants["C21"]
    )

    if power <= 0.0:

        return {
            "C40_normalized": 0.0,
            "C41_normalized": 0.0,
            "C42_normalized": 0.0,
            "C60_normalized": 0.0,
            "C61_normalized": 0.0,
            "C62_normalized": 0.0,
            "C63_normalized": 0.0,
        }

    return {
        "C40_normalized": float(
            abs(cumulants["C40"])
            / (power ** 2)
        ),

        "C41_normalized": float(
            abs(cumulants["C41"])
            / (power ** 2)
        ),

        "C42_normalized": float(
            abs(cumulants["C42"])
            / (power ** 2)
        ),

        "C60_normalized": float(
            abs(cumulants["C60"])
            / (power ** 3)
        ),

        "C61_normalized": float(
            abs(cumulants["C61"])
            / (power ** 3)
        ),

        "C62_normalized": float(
            abs(cumulants["C62"])
            / (power ** 3)
        ),

        "C63_normalized": float(
            abs(cumulants["C63"])
            / (power ** 3)
        ),
    }


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------

def calculate_hoc(
    signal: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate higher-order cumulants for a signal.

    Parameters
    ----------
    signal:
        One-dimensional real or complex signal.

    Returns
    -------
    dict
        Dictionary containing:

            moments
            cumulants
            normalized_cumulants
            signal_power
            signal_mean
            number_of_samples
            is_complex
    """

    signal = _validate_signal(
        signal
    )

    original_mean = np.mean(
        signal
    )

    moments = _calculate_moments(
        signal
    )

    cumulants = _calculate_cumulants(
        moments
    )

    normalized_cumulants = (
        _calculate_normalized_cumulants(
            cumulants
        )
    )

    signal_power = float(
        np.mean(
            np.abs(signal) ** 2
        )
    )

    result: dict[str, Any] = {
        "moments": moments,

        "cumulants": cumulants,

        "normalized_cumulants": (
            normalized_cumulants
        ),

        "signal_mean": complex(
            original_mean
        ),

        "signal_power": signal_power,

        "number_of_samples": int(
            signal.size
        ),

        "is_complex": bool(
            np.iscomplexobj(signal)
        ),
    }

    # Also expose commonly used cumulants at the top level.
    for name, value in cumulants.items():

        result[name] = value

    for name, value in normalized_cumulants.items():

        result[name] = value

    return result


# ---------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------

def higher_order_cumulants(
    signal: np.ndarray,
) -> dict[str, Any]:
    """
    Compatibility wrapper for calculate_hoc().
    """

    return calculate_hoc(
        signal
    )


def hoc(
    signal: np.ndarray,
) -> dict[str, Any]:
    """
    Short compatibility wrapper for calculate_hoc().
    """

    return calculate_hoc(
        signal
    )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def get_cumulants(
    signal: np.ndarray,
) -> dict[str, complex]:
    """
    Return only the calculated cumulants.
    """

    result = calculate_hoc(
        signal
    )

    cumulants = result[
        "cumulants"
    ]

    return dict(cumulants)