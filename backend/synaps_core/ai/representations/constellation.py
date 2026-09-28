"""
2D Constellation grid and density representations.
"""

from typing import Tuple
import numpy as np
import torch


def compute_constellation_histogram(
    symbols_or_iq: np.ndarray,
    grid_size: int = 64,
    range_limit: float = 2.0,
) -> np.ndarray:
    """
    Compute 2D histogram / density grid of constellation points in range [-range_limit, range_limit].
    """
    samples = np.asarray(symbols_or_iq, dtype=np.complex128)
    if samples.size == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    # Normalize RMS
    rms = np.sqrt(np.mean(np.abs(samples) ** 2))
    if rms > 1e-12:
        norm_samples = samples / rms
    else:
        norm_samples = samples

    i = norm_samples.real
    q = norm_samples.imag

    bins = np.linspace(-range_limit, range_limit, grid_size + 1)
    hist, _, _ = np.histogram2d(i, q, bins=[bins, bins])

    # Normalize density
    max_h = np.max(hist)
    if max_h > 0:
        hist = hist / max_h

    return hist.astype(np.float32)


def extract_constellation_tensor(
    symbols_or_iq: np.ndarray,
    grid_size: int = 64,
) -> torch.Tensor:
    """
    Extract normalized constellation 2D density tensor of shape (1, grid_size, grid_size).
    """
    grid = compute_constellation_histogram(symbols_or_iq, grid_size=grid_size)
    return torch.from_numpy(grid).unsqueeze(0)