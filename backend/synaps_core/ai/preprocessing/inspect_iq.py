import numpy as np
import matplotlib.pyplot as plt

from iq_loader import load_iq_file


filepath = "data/iq/BPSK/signal_0004_bpsk.iq"

iq = load_iq_file(filepath)

# -----------------------------
# 1. Constellation
# -----------------------------
plt.figure()
plt.scatter(iq.real, iq.imag, s=3)
plt.xlabel("In-phase")
plt.ylabel("Quadrature")
plt.title("BPSK Constellation")
plt.grid(True)
plt.axis("equal")
plt.show()


# -----------------------------
# 2. IQ waveform
# -----------------------------
plt.figure()
plt.plot(iq.real[:1000], label="In-phase")
plt.plot(iq.imag[:1000], label="Quadrature")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("IQ Waveform")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# 3. Frequency spectrum
# -----------------------------
spectrum = np.fft.fftshift(np.fft.fft(iq))
frequencies = np.fft.fftshift(
    np.fft.fftfreq(len(iq), d=1 / 1_000_000)
)

plt.figure()
plt.plot(frequencies, 20 * np.log10(np.abs(spectrum) + 1e-12))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Signal Spectrum")
plt.grid(True)
plt.show()