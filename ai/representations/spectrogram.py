"""
Time-frequency / spectrogram representations for signal intelligence.
"""

from typing import Optional, Tuple
import numpy as np
import scipy.signal
import torch


def compute_spectrogram(
    iq_samples: np.ndarray,
    sample_rate: float = 1_000_000.0,
    nperseg: int = 128,
    noverlap: int = 64,
    nfft: Optional[int] = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT spectrogram (frequencies, times, Sxx).
    """
    samples = np.asarray(iq_samples, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError("Signal cannot be empty.")

    f, t, zxx = scipy.signal.stft(
        samples,
        fs=sample_rate,
        nperseg=min(nperseg, len(samples)),
        noverlap=min(noverlap, len(samples) - 1),
        nfft=nfft,
        return_onesided=False,
    )
    # Centered frequency spectrum
    f = np.fft.fftshift(f)
    zxx = np.fft.fftshift(zxx, axes=0)
    magnitude_db = 20 * np.log10(np.abs(zxx) + 1e-12)

    return f, t, magnitude_db


def extract_spectrogram_tensor(
    iq_samples: np.ndarray,
    sample_rate: float = 1_000_000.0,
    target_shape: Tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """
    Extract normalized 2D spectrogram tensor of shape (1, H, W).
    """
    f, t, mag_db = compute_spectrogram(iq_samples, sample_rate=sample_rate)

    # Normalize to [0, 1]
    min_val = np.min(mag_db)
    max_val = np.max(mag_db)
    if max_val > min_val:
        norm = (mag_db - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(mag_db)

    # Simple bilinear/grid interpolation or resize if needed
    h, w = target_shape
    curr_h, curr_w = norm.shape
    if curr_h != h or curr_w != w:
        from scipy.ndimage import zoom
        zoom_y = h / curr_h
        zoom_x = w / curr_w
        norm = zoom(norm, (zoom_y, zoom_x), order=1)

    tensor = torch.from_numpy(norm.astype(np.float32)).unsqueeze(0)
    return tensor