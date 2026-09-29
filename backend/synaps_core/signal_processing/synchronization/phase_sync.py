import numpy as np


def correct_phase_offset(iq, phase_offset_degrees):
    """
    Correct a known phase offset in a complex IQ signal.

    Parameters:
        iq: Complex IQ samples
        phase_offset_degrees: Phase offset in degrees

    Returns:
        Phase-corrected IQ samples
    """

    phase_offset_radians = np.deg2rad(phase_offset_degrees)

    correction = np.exp(-1j * phase_offset_radians)

    corrected_iq = iq * correction

    return corrected_iq