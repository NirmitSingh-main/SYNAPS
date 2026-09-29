import numpy as np


def prepare_iq_features(iq: np.ndarray) -> np.ndarray:
    """
    Prepare normalized IQ samples for the AI pipeline.

    Input:
        iq:
            Complex-valued IQ samples (x[n] = I[n] + j*Q[n]).

    Output:
        Feature matrix with 6 columns:
            [I, Q, magnitude, absolute_phase, diff_phase_cos, diff_phase_sin]

        Shape:
            (number_of_samples, 6)

    Features:
        1. I: In-phase component normalized by max magnitude in [-1, 1].
        2. Q: Quadrature component normalized by max magnitude in [-1, 1].
        3. magnitude: Normalized signal envelope |x[n]| in [0, 1].
        4. absolute_phase: Instantaneous phase angle(x[n]) in [-pi, pi].
        5. diff_phase_cos: In-phase component of normalized differential phasor:
           cos(Delta theta[n]) = Re(x[n] * conj(x[n-1]) / |x[n] * conj(x[n-1])|) in [-1, 1].
        6. diff_phase_sin: Quadrature component of normalized differential phasor:
           sin(Delta theta[n]) = Im(x[n] * conj(x[n-1]) / |x[n] * conj(x[n-1])|) in [-1, 1].

    Why Circular Representation (cos, sin) for Differential Phase:
        - Resolves phase wrapping: Angle of conjugate product avoids modulo-2pi boundary cuts.
        - Suitable for linear window aggregation: Averaging [cos(Delta theta), sin(Delta theta)]
          computes the true circular first trigonometric moment (mean resultant phasor).
        - Encodes phase coherence: Resultant length R = sqrt(mean_cos^2 + mean_sin^2) directly
          measures phase stability / frequency consistency without destructive angle cancellation.
        - Robust under CFO: A static carrier offset Delta f manifests as a constant unit phasor
          (cos(2pi*Delta f/fs), sin(2pi*Delta f/fs)), leaving discrete symbol transitions separable.
        - Boundary handling: For n=0, initialized to unit phasor (cos=1.0, sin=0.0).
        - Zero/silence handling: Zero-magnitude samples safely map to (cos=0.0, sin=0.0) without NaN/Inf.
    """

    iq = np.asarray(iq)

    if iq.size == 0:
        raise ValueError("IQ signal cannot be empty")

    if not np.iscomplexobj(iq):
        raise ValueError("IQ input must contain complex-valued samples")

    # In-phase and quadrature components
    i = np.real(iq).astype(np.float32)
    q = np.imag(iq).astype(np.float32)

    # Magnitude
    magnitude = np.abs(iq).astype(np.float32)

    # Absolute Phase in [-pi, pi]
    phase = np.angle(iq).astype(np.float32)

    # Local Differential Phasor: x[n] * conj(x[n-1]) / |x[n] * conj(x[n-1])|
    diff_cos = np.ones(len(iq), dtype=np.float32)
    diff_sin = np.zeros(len(iq), dtype=np.float32)

    if len(iq) > 1:
        # Conjugate product computes local phase rotation directly
        prod = iq[1:] * np.conj(iq[:-1])
        prod_mag = np.abs(prod).astype(np.float32)

        # Normalize to unit circle on non-zero samples
        valid_mask = prod_mag > 1e-12
        unit_phasor = np.zeros(len(prod), dtype=np.complex64)
        unit_phasor[valid_mask] = prod[valid_mask] / prod_mag[valid_mask]

        diff_cos[1:] = np.nan_to_num(np.real(unit_phasor), nan=0.0, posinf=0.0, neginf=0.0)
        diff_sin[1:] = np.nan_to_num(np.imag(unit_phasor), nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize I/Q and magnitude using maximum magnitude
    scale = float(np.max(magnitude))

    if scale > 0:
        i = i / scale
        q = q / scale
        magnitude = magnitude / scale

    # Stack all 6 features: [I, Q, magnitude, phase, diff_phase_cos, diff_phase_sin]
    features = np.column_stack(
        (
            i,
            q,
            magnitude,
            phase,
            diff_cos,
            diff_sin,
        )
    )

    return features.astype(np.float32)


def tokenize_signal_features(
    features: np.ndarray,
    num_tokens: int = 256,
) -> np.ndarray:
    """
    Convert a variable-length feature matrix into a fixed-length
    token sequence suitable for Transformer input.

    Strategy:
        Divide the full signal into `num_tokens` non-overlapping windows.
        For each window, compute the mean of each feature column.
        This produces exactly `num_tokens` tokens regardless of input length.

    Why window averaging on circular features:
        - Columns 0-2 (I, Q, magnitude): standard temporal envelope averaging.
        - Column 3 (phase): mean instantaneous phase.
        - Columns 4-5 (diff_cos, diff_sin): exact circular resultant vector averaging.
          The token vector (mean_cos, mean_sin) preserves both mean phase progression
          and local circular phase concentration (coherence).

    Input:
        features:
            Shape (N, feature_dim) where N is the raw sample count.
            Typically (8000, 6) or (9600, 6) from prepare_iq_features().

    Output:
        Shape (num_tokens, feature_dim).
        Typically (256, 6).
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix (N, feature_dim), got shape {features.shape}"
        )

    n_samples = features.shape[0]
    feature_dim = features.shape[1]

    if n_samples == 0:
        raise ValueError("Cannot tokenize empty feature matrix")

    if n_samples < num_tokens:
        # If signal is shorter than num_tokens, pad with zeros
        padded = np.zeros((num_tokens, feature_dim), dtype=np.float32)
        padded[:n_samples] = features
        return padded

    # Compute window boundaries for even splitting
    # np.array_split handles non-evenly-divisible lengths correctly
    chunks = np.array_split(features, num_tokens, axis=0)

    # Aggregate each window by computing its mean
    tokens = np.array(
        [chunk.mean(axis=0) for chunk in chunks],
        dtype=np.float32,
    )

    return tokens