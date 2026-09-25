"""
Symbol-level representation and statistical feature extraction for synchronized communication signals.

Extracts:
    - I, Q
    - Magnitude |s|
    - Power |s|^2
    - Phase angle(s)
    - Magnitude variance Var(|s|)
    - Amplitude histogram & modality
    - Normalized 4th moment: E[|s|^4] / (E[|s|^2])^2
    - Normalized 8th moment: E[|s|^8] / (E[|s|^2])^4
"""

from typing import Any, Dict, Tuple
import numpy as np


def extract_symbol_features(symbols: np.ndarray, num_hist_bins: int = 10) -> Dict[str, Any]:
    """
    Extract symbol-level statistical and geometric constellation features.

    Parameters:
    -----------
    symbols : np.ndarray
        1D array of complex symbol-spaced samples (e.g. from sync pipeline).
    num_hist_bins : int
        Number of magnitude histogram bins (default: 10).

    Returns:
    --------
    Dict[str, Any] containing scalar features and raw symbol channels.
    """
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.size == 0:
        raise ValueError("Symbols array cannot be empty.")

    i_vals = symbols.real.astype(np.float32)
    q_vals = symbols.imag.astype(np.float32)
    mag = np.abs(symbols).astype(np.float32)
    pwr = (mag ** 2).astype(np.float32)
    phase = np.angle(symbols).astype(np.float32)

    # Basic statistics
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_var = float(np.var(mag))
    pwr_mean = float(np.mean(pwr))
    pwr_std = float(np.std(pwr))

    # Normalized moments
    # 4th moment: E[|s|^4] / E[|s|^2]^2
    pwr_sq_mean = float(np.mean(pwr ** 2))
    fourth_moment_ratio = float(pwr_sq_mean / (max(pwr_mean, 1e-12) ** 2))

    # 8th moment: E[|s|^8] / E[|s|^2]^4 (clipped to prevent overflow)
    pwr_4th_mean = float(np.mean(pwr ** 4))
    eighth_moment_ratio = float(np.clip(pwr_4th_mean / (max(pwr_mean, 1e-12) ** 4), 0.0, 1000.0))

    # Amplitude histogram (normalized to sum to 1)
    hist_counts, bin_edges = np.histogram(mag, bins=num_hist_bins, density=True)
    hist_counts = (hist_counts / (np.sum(hist_counts) + 1e-12)).astype(np.float32)

    # Modality / number of significant amplitude peaks in histogram
    # Smooth histogram with 3-tap moving average
    smoothed = np.convolve(hist_counts, [0.25, 0.5, 0.25], mode="same")
    peak_count = 0
    for k in range(1, len(smoothed) - 1):
        if smoothed[k] > smoothed[k - 1] and smoothed[k] > smoothed[k + 1] and smoothed[k] > 0.05:
            peak_count += 1
    num_amp_modes = max(1, peak_count)

    # Radius percentiles
    p25, p50, p75 = [float(x) for x in np.percentile(mag, [25, 50, 75])]
    mag_iqr = p75 - p25

    return {
        "i": i_vals,
        "q": q_vals,
        "magnitude": mag,
        "power": pwr,
        "phase": phase,
        "mag_mean": mag_mean,
        "mag_std": mag_std,
        "mag_variance": mag_var,
        "mag_iqr": mag_iqr,
        "pwr_mean": pwr_mean,
        "pwr_std": pwr_std,
        "fourth_moment_ratio": fourth_moment_ratio,
        "eighth_moment_ratio": eighth_moment_ratio,
        "amplitude_histogram": hist_counts,
        "num_amplitude_modes": num_amp_modes,
        "num_symbols": int(symbols.size),
    }


def compute_symbol_tensor_features(symbols: np.ndarray, target_length: int = 256) -> np.ndarray:
    """
    Format synchronized symbol channels into a fixed-length multi-channel tensor
    for downstream neural network ingestion:
    [I, Q, magnitude, phase, power] of shape (5, target_length).

    Parameters:
    -----------
    symbols : np.ndarray
        Extracted symbol samples.
    target_length : int
        Desired sequence length (default: 256).

    Returns:
    --------
    np.ndarray of shape (5, target_length), dtype float32.
    """
    feats = extract_symbol_features(symbols)
    num_s = len(symbols)

    if num_s >= target_length:
        i_sub = feats["i"][:target_length]
        q_sub = feats["q"][:target_length]
        mag_sub = feats["magnitude"][:target_length]
        phase_sub = feats["phase"][:target_length]
        pwr_sub = feats["power"][:target_length]
    else:
        # Zero-pad
        pad_len = target_length - num_s
        i_sub = np.pad(feats["i"], (0, pad_len))
        q_sub = np.pad(feats["q"], (0, pad_len))
        mag_sub = np.pad(feats["magnitude"], (0, pad_len))
        phase_sub = np.pad(feats["phase"], (0, pad_len))
        pwr_sub = np.pad(feats["power"], (0, pad_len))

    tensor = np.stack([i_sub, q_sub, mag_sub, phase_sub, pwr_sub], axis=0).astype(np.float32)
    return tensor
