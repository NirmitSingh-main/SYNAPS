SIH26147 Synthetic Signal Starter Dataset
40 paired signals: 10 BPSK, 10 QPSK, 10 2FSK, 10 16QAM.

IQ files: interleaved little-endian float32 I,Q.
WAV files: two-channel float32 WAV, channel 1=I and channel 2=Q.
All signals are complex-baseband; nominal carrier is 0 Hz.
Frequency offset, phase offset, amplitude and signal-to-noise ratio are varied.
Each JSON and dataset.csv row contains exact ground truth, including transmitted bits.

This is a validation batch. Confirm Dev 1 can load it before scaling.
