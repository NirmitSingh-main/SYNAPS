"""
Raw IQ representation utilities for neural network inputs.
"""

from typing import Tuple
import numpy as np
import torch


def extract_raw_iq_tensor(
    iq_samples: np.ndarray,
    target_length: int = 1000,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extract a (2, target_length) or (target_length, 2) normalized IQ tensor.
    """
    samples = np.asarray(iq_samples, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError("IQ signal cannot be empty.")

    i = samples.real
    q = samples.imag

    if normalize:
        peak = max(float(np.max(np.abs(samples))), 1e-12)
        i = i / peak
        q = q / peak

    if len(i) >= target_length:
        i = i[:target_length]
        q = q[:target_length]
    else:
        pad_len = target_length - len(i)
        i = np.pad(i, (0, pad_len))
        q = np.pad(q, (0, pad_len))

    stacked = np.stack([i, q], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)