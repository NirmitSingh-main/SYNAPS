// ============================================================
// DEMO DATA ONLY
// All values below are illustrative and fabricated for
// demonstration purposes. They do NOT represent real analysis
// output from any backend, model, or signal processing system.
// ============================================================

export interface DemoAnalysis {
  id: string;
  timestamp: string;
  isDemoData: boolean;
  label: string;
  filename: string;
  format: string;
  classification: string;
  confidence: number;
  sampleRate: number;
  duration: number;
  bandwidth: number;
  snr: number;
  peakFrequency: number;
  numSamples: number;
  predictionBreakdown: { classLabel: string; probability: number }[];
  features: { name: string; value: string; unit: string; description: string }[];
  explanation: string;
  waveformSamples: number[];
  spectrumBins: number[];
  spectrogramRows: number[][];
}

export const demoAnalysis: DemoAnalysis = {
  // ---- Meta ----
  id: "demo-00001",
  timestamp: "2026-08-30T17:41:00Z",
  isDemoData: true,
  label: "Illustrative analysis",

  // ---- File info ----
  filename: "sample_signal.iq",
  format: "IQ",

  // ---- Classification ----
  classification: "BPSK",
  confidence: 0.913,

  // ---- Signal metrics ----
  sampleRate: 1000000,       // Hz
  duration: 0.008192,        // seconds
  bandwidth: 120000,         // Hz
  snr: 17.4,                 // dB
  peakFrequency: 6100,       // Hz
  numSamples: 8192,

  // ---- Prediction breakdown ----
  predictionBreakdown: [
    { classLabel: "BPSK",     probability: 0.913 },
    { classLabel: "QPSK",     probability: 0.043 },
    { classLabel: "QAM16",    probability: 0.035 },
    { classLabel: "FSK",      probability: 0.009 },
  ],

  // ---- Extracted features ----
  features: [
    { name: "Carrier Frequency Offset",      value: "20.6",   unit: "kHz",  description: "Mean instantaneous frequency deviation" },
    { name: "Signal-to-Noise Ratio",         value: "17.4",   unit: "dB",   description: "Estimated SNR" },
    { name: "3dB Occupied Bandwidth",        value: "120.0",  unit: "kHz",  description: "Estimated signal bandwidth" },
    { name: "Higher-Order Cumulant C40",     value: "0.025",  unit: "",     description: "4th-order cumulant for constellation symmetry" },
    { name: "Higher-Order Cumulant C42",     value: "0.435",  unit: "",     description: "4th-order cumulant for power variance" },
    { name: "Peak-to-Average Power Ratio",   value: "5.4",    unit: "dB",   description: "Crest factor of the signal envelope" },
  ],

  // ---- Illustrative explanation ----
  explanation: "The extracted Higher-Order Cumulants and narrow spectral profile are consistent with a BPSK modulated carrier. The neural Transformer model classified this signal with 91.3% confidence, corroborated by DSP cumulant analysis.",

  // ---- Waveform samples (256 points) ----
  waveformSamples: Array.from({ length: 256 }, (_, i) => {
    const t = i / 256;
    return (
      Math.sin(t * Math.PI * 2 * 12 + Math.sin(t * Math.PI * 2 * 1.5) * 3.2) * 0.7 +
      Math.sin(t * Math.PI * 2 * 37 + 0.4) * 0.12 +
      Math.sin(t * Math.PI * 2 * 73) * 0.05
    );
  }),

  // ---- Spectrum bins (64 bins) ----
  spectrumBins: Array.from({ length: 64 }, (_, i) => {
    const x = i / 64;
    return (
      Math.exp(-Math.pow((x - 0.22) * 6, 2)) * 0.92 +
      Math.exp(-Math.pow((x - 0.62) * 9, 2)) * 0.54 +
      Math.max(0, 0.08 - Math.abs(x - 0.5) * 0.3) +
      0.03 * Math.abs(Math.sin(i * 1.7))
    );
  }),

  // ---- Spectrogram (24 rows x 48 columns) ----
  spectrogramRows: Array.from({ length: 24 }, (_, row) =>
    Array.from({ length: 48 }, (_, col) => {
      const t = row / 24;
      const f = col / 48;
      return Math.max(
        0,
        Math.min(
          1,
          0.5 +
          0.5 * Math.sin(col * 0.35 + t * 4 + Math.cos(row * 0.5 + t * 2) * 1.6) *
            Math.exp(-Math.pow((f - 0.4) * 2.8, 2))
        )
      );
    })
  ),
};
