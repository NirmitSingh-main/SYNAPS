import numpy as np


def prepare_iq_features(iq: np.ndarray) -> np.ndarray:
    """
    Prepare normalized IQ samples for the AI pipeline.

    Input:
        iq:
            Complex-valued IQ samples.

    Output:
        Feature matrix with columns:
            [I, Q, magnitude, phase]

        Shape:
            (number_of_samples, 4)
    """

    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("IQ signal cannot be empty")

    if not np.iscomplexobj(iq):
        raise ValueError("IQ input must contain complex-valued samples")

    # In-phase and quadrature components
    i = np.real(iq)
    q = np.imag(iq)

    # Magnitude
    magnitude = np.abs(iq)

    # Phase
    phase = np.angle(iq)

    # Normalize I/Q using maximum magnitude
    scale = np.max(magnitude)

    if scale > 0:
        i = i / scale
        q = q / scale
        magnitude = magnitude / scale

    features = np.column_stack(
        (
            i,
            q,
            magnitude,
            phase,
        )
    )

    return features.astype(np.float32)