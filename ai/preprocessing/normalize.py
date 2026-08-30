import numpy as np


def normalize_iq(iq: np.ndarray) -> np.ndarray:
    """
    Normalize complex IQ samples by maximum magnitude.

    Input:
        iq: Complex-valued IQ samples.

    Output:
        Normalized complex64 IQ samples.
    """

    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("IQ signal cannot be empty")

    if not np.iscomplexobj(iq):
        raise ValueError("IQ input must be complex-valued")

    max_magnitude = np.max(np.abs(iq))

    if max_magnitude == 0:
        return iq.astype(np.complex64)

    return (iq / max_magnitude).astype(np.complex64)