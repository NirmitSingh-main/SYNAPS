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


# def tokenize_signal_features(
#     features: np.ndarray,
#     num_tokens: int = 256,
# ) -> np.ndarray:
#     """
#     Convert a variable-length feature matrix into a fixed-length
#     token sequence suitable for Transformer input.

#     Strategy:
#         Divide the full signal into `num_tokens` non-overlapping windows.
#         For each window, aggregate:
#           - Columns 0-5 (means): [mean_I, mean_Q, mean_magnitude, mean_phase, mean_diff_cos, mean_diff_sin]
#           - Local amplitude & dispersion statistics:
#             * std_magnitude: local envelope standard deviation within window
#             * std_I: local in-phase dispersion within window
#             * std_Q: local quadrature dispersion within window
#             * ptp_magnitude: local envelope peak-to-peak dynamic range (max - min)

#         This produces exactly `num_tokens` tokens (shape: `(num_tokens, 10)`),
#         preserving local amplitude distribution and constellation ring variance
#         while maintaining circular phase representation.

#     Input:
#         features:
#             Shape (N, feature_dim) where N is the raw sample count.
#             Typically (8000, 6) or (9600, 6) from prepare_iq_features().

#     Output:
#         Shape (num_tokens, 10) for standard 6-feature input.
#     """
#     if features.ndim != 2:
#         raise ValueError(
#             f"Expected 2D feature matrix (N, feature_dim), got shape {features.shape}"
#         )

#     n_samples = features.shape[0]
#     feature_dim = features.shape[1]

#     if n_samples == 0:
#         raise ValueError("Cannot tokenize empty feature matrix")

#     # If 6 features, extended dimension is 10
#     out_dim = feature_dim + 4 if feature_dim >= 6 else feature_dim

#     if n_samples < num_tokens:
#         padded = np.zeros((num_tokens, out_dim), dtype=np.float32)
#         if feature_dim >= 6:
#             base_mean = features
#             std_mag = np.zeros((n_samples, 1), dtype=np.float32)
#             std_i = np.zeros((n_samples, 1), dtype=np.float32)
#             std_q = np.zeros((n_samples, 1), dtype=np.float32)
#             ptp_mag = np.zeros((n_samples, 1), dtype=np.float32)
#             ext = np.hstack([base_mean, std_mag, std_i, std_q, ptp_mag])
#             padded[:n_samples] = ext
#         else:
#             padded[:n_samples] = features
#         return padded

#     # Compute window boundaries for even splitting
#     chunks = np.array_split(features, num_tokens, axis=0)

#     # Aggregate each window with means and local dispersion statistics
#     tokens = []
#     for chunk in chunks:
#         c_mean = chunk.mean(axis=0)
#         if feature_dim >= 6:
#             # chunk columns: [I, Q, magnitude, phase, diff_cos, diff_sin]
#             std_mag = float(np.std(chunk[:, 2]))
#             std_i = float(np.std(chunk[:, 0]))
#             std_q = float(np.std(chunk[:, 1]))
#             ptp_mag = float(np.max(chunk[:, 2]) - np.min(chunk[:, 2]))
#             tok = np.concatenate([c_mean, [std_mag, std_i, std_q, ptp_mag]])
#         else:
#             tok = c_mean
#         tokens.append(tok)

#     return np.array(tokens, dtype=np.float32)


def tokenize_signal_features(
    features: np.ndarray,
    num_tokens: int = 256,
) -> np.ndarray:
    """
    Convert a variable-length feature matrix into a fixed-length
    token sequence suitable for Transformer input.

    Strategy:
        Divide the full signal into `num_tokens` non-overlapping windows.

        For each window, aggregate:
          - Columns 0-5 (means):
            [mean_I, mean_Q, mean_magnitude, mean_phase,
             mean_diff_cos, mean_diff_sin]

          - Local amplitude & dispersion statistics:
            * std_magnitude
            * std_I
            * std_Q
            * ptp_magnitude

        This produces exactly `num_tokens` tokens with shape
        `(num_tokens, 10)` for the standard 6-feature input.

    Input:
        features:
            Shape (N, feature_dim).

    Output:
        Shape (num_tokens, 10) for standard 6-feature input.
    """

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix (N, feature_dim), "
            f"got shape {features.shape}"
        )

    n_samples = features.shape[0]
    feature_dim = features.shape[1]

    if n_samples == 0:
        raise ValueError("Cannot tokenize empty feature matrix")

    # For the standard 6-feature IQ representation,
    # append 4 local statistics.
    out_dim = feature_dim + 4 if feature_dim >= 6 else feature_dim

    # Handle signals shorter than the requested number of tokens.
    if n_samples < num_tokens:
        padded = np.zeros(
            (num_tokens, out_dim),
            dtype=np.float32
        )

        if feature_dim >= 6:
            base_mean = features

            std_mag = np.zeros(
                (n_samples, 1),
                dtype=np.float32
            )
            std_i = np.zeros(
                (n_samples, 1),
                dtype=np.float32
            )
            std_q = np.zeros(
                (n_samples, 1),
                dtype=np.float32
            )
            ptp_mag = np.zeros(
                (n_samples, 1),
                dtype=np.float32
            )

            ext = np.hstack(
                [
                    base_mean,
                    std_mag,
                    std_i,
                    std_q,
                    ptp_mag,
                ]
            )

            padded[:n_samples] = ext
        else:
            padded[:n_samples] = features

        return padded

    # Divide the signal into exactly num_tokens windows.
    chunks = np.array_split(
        features,
        num_tokens,
        axis=0
    )

    tokens = []

    for chunk in chunks:

        # Existing six mean features.
        c_mean = chunk.mean(axis=0)

        if feature_dim >= 6:

            # Columns:
            # 0 = I
            # 1 = Q
            # 2 = magnitude
            # 3 = phase
            # 4 = differential phase cosine
            # 5 = differential phase sine

            std_mag = float(
                np.std(chunk[:, 2])
            )

            std_i = float(
                np.std(chunk[:, 0])
            )

            std_q = float(
                np.std(chunk[:, 1])
            )

            ptp_mag = float(
                np.max(chunk[:, 2])
                - np.min(chunk[:, 2])
            )

            # Final token:
            # [6 original means +
            #  4 local statistics]
            tok = np.concatenate(
                [
                    c_mean,
                    [
                        std_mag,
                        std_i,
                        std_q,
                        ptp_mag,
                    ],
                ]
            )

        else:
            tok = c_mean

        tokens.append(tok)

    tokens = np.array(
        tokens,
        dtype=np.float32
    )

    # Safety check.
    if not np.all(np.isfinite(tokens)):
        raise ValueError(
            "Tokenization produced NaN or Inf values"
        )

    return tokens


def prepare_synchronized_symbol_features(
    iq: np.ndarray,
    sampling_rate: float = 1_000_000.0,
    max_symbols: int = 1024,
    num_channels: int = 8,
) -> np.ndarray:
    """
    Extract synchronized symbol-center representation.

    Pipeline:
        Raw IQ
          ↓
        RMS Normalization
          ↓
        Blind CFO Estimation & Derotation
          ↓
        Blind SPS Estimation
          ↓
        Oerder-Meyr Fractional Timing Recovery
          ↓
        Symbol-Level Features: [I, Q, mag, diff_cos, diff_sin, power, 4th_moment, 8th_moment]

    Returns:
        Tensor array of shape (num_channels, max_symbols), dtype float32.
    """
    from signal_processing.synchronization.sync_pipeline import extract_synchronized_symbols
    from ai.features.symbol_features import extract_symbol_features

    # 1. Blind synchronization (no metadata leakage)
    sync_res = extract_synchronized_symbols(iq, sampling_rate=sampling_rate)
    symbols = sync_res.symbols

    # 2. Extract statistical symbol features
    feats = extract_symbol_features(symbols)

    num_s = len(symbols)
    if num_s < max_symbols:
        pad_len = max_symbols - num_s
        i_arr = np.pad(feats["i"], (0, pad_len))
        q_arr = np.pad(feats["q"], (0, pad_len))
        mag_arr = np.pad(feats["magnitude"], (0, pad_len))
        pwr_arr = np.pad(feats["power"], (0, pad_len))
    else:
        i_arr = feats["i"][:max_symbols]
        q_arr = feats["q"][:max_symbols]
        mag_arr = feats["magnitude"][:max_symbols]
        pwr_arr = feats["power"][:max_symbols]

    # Differential phase cos/sin
    diff_cos = np.ones(max_symbols, dtype=np.float32)
    diff_sin = np.zeros(max_symbols, dtype=np.float32)
    if num_s > 1:
        sym_sub = symbols[:min(num_s, max_symbols)]
        prod = sym_sub[1:] * np.conj(sym_sub[:-1])
        prod_mag = np.abs(prod)
        valid = prod_mag > 1e-12
        unit = np.zeros(len(prod), dtype=np.complex64)
        unit[valid] = prod[valid] / prod_mag[valid]
        diff_cos[1:len(sym_sub)] = np.nan_to_num(np.real(unit), nan=0.0, posinf=0.0, neginf=0.0)
        diff_sin[1:len(sym_sub)] = np.nan_to_num(np.imag(unit), nan=0.0, posinf=0.0, neginf=0.0)

    m4_arr = np.full(max_symbols, feats["fourth_moment_ratio"], dtype=np.float32)
    m8_arr = np.full(max_symbols, feats["eighth_moment_ratio"], dtype=np.float32)

    if num_channels == 8:
        tensor = np.stack([i_arr, q_arr, mag_arr, diff_cos, diff_sin, pwr_arr, m4_arr, m8_arr], axis=0)
    elif num_channels == 5:
        tensor = np.stack([i_arr, q_arr, mag_arr, diff_cos, diff_sin], axis=0)
    else:
        raise ValueError(f"Unsupported num_channels: {num_channels}")

    return tensor.astype(np.float32)