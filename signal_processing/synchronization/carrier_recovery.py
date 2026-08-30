import numpy as np

from .frequency_sync import correct_frequency_offset
from .phase_sync import correct_phase_offset


def recover_carrier(
    iq,
    frequency_offset_hz,
    phase_offset_degrees,
    sampling_frequency_hz,
):
    """
    Perform carrier recovery on a complex IQ signal.

    Carrier recovery consists of:
        1. Frequency offset correction
        2. Phase offset correction

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ samples.

    frequency_offset_hz : float
        Known frequency offset in Hz.

    phase_offset_degrees : float
        Known phase offset in degrees.

    sampling_frequency_hz : float
        Sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Carrier-corrected IQ samples.

    Raises
    ------
    TypeError
        If iq is not a NumPy array or is not complex-valued.

    ValueError
        If the signal or parameters are invalid.
    """

    # ---------------------------------------------------------
    # Validate IQ signal
    # ---------------------------------------------------------

    if not isinstance(iq, np.ndarray):
        raise TypeError(
            "iq must be a NumPy array."
        )

    if iq.size == 0:
        raise ValueError(
            "IQ signal cannot be empty."
        )

    if not np.iscomplexobj(iq):
        raise TypeError(
            "iq must contain complex-valued samples."
        )

    if not np.isfinite(iq.real).all():
        raise ValueError(
            "IQ signal contains invalid real values."
        )

    if not np.isfinite(iq.imag).all():
        raise ValueError(
            "IQ signal contains invalid imaginary values."
        )

    # ---------------------------------------------------------
    # Validate sampling frequency
    # ---------------------------------------------------------

    if not np.isfinite(sampling_frequency_hz):
        raise ValueError(
            "sampling_frequency_hz must be finite."
        )

    if sampling_frequency_hz <= 0:
        raise ValueError(
            "sampling_frequency_hz must be greater than zero."
        )

    # ---------------------------------------------------------
    # Validate frequency offset
    # ---------------------------------------------------------

    if not np.isfinite(frequency_offset_hz):
        raise ValueError(
            "frequency_offset_hz must be finite."
        )

    # ---------------------------------------------------------
    # Validate phase offset
    # ---------------------------------------------------------

    if not np.isfinite(phase_offset_degrees):
        raise ValueError(
            "phase_offset_degrees must be finite."
        )

    # ---------------------------------------------------------
    # Step 1: Correct frequency offset
    # ---------------------------------------------------------

    frequency_corrected = correct_frequency_offset(
        iq,
        frequency_offset_hz,
        sampling_frequency_hz,
    )

    # ---------------------------------------------------------
    # Step 2: Correct phase offset
    # ---------------------------------------------------------

    carrier_recovered = correct_phase_offset(
        frequency_corrected,
        phase_offset_degrees,
    )

    return carrier_recovered