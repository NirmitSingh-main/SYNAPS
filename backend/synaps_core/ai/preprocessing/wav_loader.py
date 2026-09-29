import numpy as np
from scipy.io import wavfile


def load_wav_file(filepath):
    """
    Load a 2-channel WAV containing I/Q samples.

    Channel 0 -> I
    Channel 1 -> Q

    Returns:
        complex64 NumPy array
    """

    sample_rate, data = wavfile.read(filepath)

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(
            "WAV file must contain two channels: I and Q."
        )

    if data.shape[1] != 2:
        raise ValueError(
            "WAV file must have exactly 2 channels."
        )

    i = data[:, 0].astype(np.float32)
    q = data[:, 1].astype(np.float32)

    iq = i + 1j * q

    return sample_rate, iq.astype(np.complex64)