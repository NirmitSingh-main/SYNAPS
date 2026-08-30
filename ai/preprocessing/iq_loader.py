import numpy as np


def load_iq_file(filepath):
    """
    Load an interleaved float32 IQ file.

    Format:
        I0, Q0, I1, Q1, I2, Q2, ...

    Returns:
        complex64 NumPy array
    """

    data = np.fromfile(filepath, dtype="<f4")

    if len(data) % 2 != 0:
        raise ValueError("IQ file contains an odd number of float values.")

    i = data[0::2]
    q = data[1::2]

    iq = i + 1j * q

    return iq.astype(np.complex64)


if __name__ == "__main__":
    filepath = "data/iq/signal_0004_bpsk.iq"

    iq = load_iq_file(filepath)

    print("Successfully loaded IQ file")
    print("Number of complex samples:", len(iq))
    print("First 5 samples:")
    print(iq[:5])