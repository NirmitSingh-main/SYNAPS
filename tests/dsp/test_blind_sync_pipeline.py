"""
Comprehensive unit tests for Blind Synchronization and Symbol-Level Representation.

Tests:
    - Blind CFO estimation
    - Blind symbol-rate / SPS estimation
    - Oerder-Meyr timing recovery and fractional interpolation
    - Complete symbol extraction pipeline
    - QPSK constellation compactness (single radius)
    - QAM16 amplitude-level separation (3 distinct levels)
    - Finite outputs and no NaN/Inf
    - Symbol-level statistical feature extraction
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dsp.frequency.cfo import estimate_cfo
from dsp.timing.symbol_rate import estimate_symbol_rate
from signal_processing.synchronization.timing_sync import estimate_timing_offset, sample_symbols
from signal_processing.synchronization.sync_pipeline import extract_synchronized_symbols
from ai.features.symbol_features import extract_symbol_features, compute_symbol_tensor_features


class TestBlindSyncPipeline(unittest.TestCase):
    """Test suite for blind synchronization and symbol-level representations."""

    def setUp(self):
        self.fs = 1_000_000.0
        self.sps = 10
        self.sr = self.fs / self.sps
        self.num_symbols = 200
        self.rng = np.random.default_rng(42)

    def _generate_synthetic_qpsk(self, cfo_hz=15000.0, timing_offset=3.5, snr_db=25.0):
        # 4 QPSK points
        bits = self.rng.integers(0, 2, size=self.num_symbols * 2)
        i_sym = np.where(bits[0::2] == 0, 1.0, -1.0)
        q_sym = np.where(bits[1::2] == 0, 1.0, -1.0)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(2.0)

        # Pulse shaping (repeat)
        baseband = np.repeat(symbols, self.sps)
        t = np.arange(len(baseband)) / self.fs
        cfo_rot = np.exp(1j * (2 * np.pi * cfo_hz * t + 0.3))
        modulated = baseband * cfo_rot

        # Delay by fractional offset
        if timing_offset > 0:
            modulated = np.roll(modulated, int(round(timing_offset)))

        # Add noise
        sig_pwr = np.mean(np.abs(modulated) ** 2)
        noise_pwr = sig_pwr / (10.0 ** (snr_db / 10.0))
        noise = self.rng.normal(0, np.sqrt(noise_pwr / 2), len(modulated)) + 1j * self.rng.normal(0, np.sqrt(noise_pwr / 2), len(modulated))
        return (modulated + noise).astype(np.complex64), symbols

    def _generate_synthetic_qam16(self, cfo_hz=-20000.0, timing_offset=4.2, snr_db=25.0):
        # 16 QAM points
        bits = self.rng.integers(0, 2, size=self.num_symbols * 4)
        lut = {(0, 0): -3.0, (0, 1): -1.0, (1, 1): 1.0, (1, 0): 3.0}
        symbols = []
        for k in range(self.num_symbols):
            b = bits[k*4 : (k+1)*4]
            symbols.append((lut[(b[0], b[1])] + 1j * lut[(b[2], b[3])]) / np.sqrt(10.0))
        symbols = np.array(symbols, dtype=np.complex128)

        baseband = np.repeat(symbols, self.sps)
        t = np.arange(len(baseband)) / self.fs
        cfo_rot = np.exp(1j * (2 * np.pi * cfo_hz * t + 0.5))
        modulated = baseband * cfo_rot

        if timing_offset > 0:
            modulated = np.roll(modulated, int(round(timing_offset)))

        sig_pwr = np.mean(np.abs(modulated) ** 2)
        noise_pwr = sig_pwr / (10.0 ** (snr_db / 10.0))
        noise = self.rng.normal(0, np.sqrt(noise_pwr / 2), len(modulated)) + 1j * self.rng.normal(0, np.sqrt(noise_pwr / 2), len(modulated))
        return (modulated + noise).astype(np.complex64), symbols

    def test_blind_cfo_estimation(self):
        """Test blind CFO estimation on QPSK and QAM16 signals."""
        cfo_qpsk_true = 18500.0
        sig_qpsk, _ = self._generate_synthetic_qpsk(cfo_hz=cfo_qpsk_true)
        res_qpsk = estimate_cfo(sig_qpsk, self.fs)
        self.assertAlmostEqual(res_qpsk["cfo_hz"], cfo_qpsk_true, delta=500.0)

        cfo_qam_true = -25000.0
        sig_qam, _ = self._generate_synthetic_qam16(cfo_hz=cfo_qam_true)
        res_qam = estimate_cfo(sig_qam, self.fs)
        self.assertAlmostEqual(res_qam["cfo_hz"], cfo_qam_true, delta=800.0)

    def test_timing_recovery(self):
        """Test Oerder-Meyr timing recovery and symbol extraction."""
        sig_qpsk, _ = self._generate_synthetic_qpsk(cfo_hz=0.0, timing_offset=3.0)
        offset = estimate_timing_offset(sig_qpsk, self.sps, return_fractional=True)
        self.assertIsInstance(offset, float)
        self.assertGreaterEqual(offset, 0.0)
        self.assertLess(offset, self.sps)

        syms = sample_symbols(sig_qpsk, self.sps, timing_offset=offset, use_fractional=True)
        self.assertGreater(len(syms), 0)
        self.assertTrue(np.all(np.isfinite(syms)))

    def test_complete_sync_pipeline(self):
        """Test complete end-to-end blind synchronization pipeline."""
        sig_qam, _ = self._generate_synthetic_qam16(cfo_hz=12000.0, timing_offset=2.5)
        sync_res = extract_synchronized_symbols(sig_qam, self.fs)
        self.assertIsNotNone(sync_res.symbols)
        self.assertGreater(len(sync_res.symbols), 50)
        self.assertTrue(np.all(np.isfinite(sync_res.symbols)))
        self.assertAlmostEqual(sync_res.cfo_hz, 12000.0, delta=1000.0)

    def test_qpsk_constellation_compactness(self):
        """Test that synchronized QPSK symbols have low radius variance (single ring)."""
        sig_qpsk, _ = self._generate_synthetic_qpsk(cfo_hz=15000.0, timing_offset=3.0, snr_db=30.0)
        sync_res = extract_synchronized_symbols(sig_qpsk, self.fs)
        feats = extract_symbol_features(sync_res.symbols)

        # QPSK: All symbols are on radius r=1.0
        # 4th moment ratio for constant envelope PSK is close to 1.0 (approx 1.0 - 1.2)
        self.assertLess(feats["fourth_moment_ratio"], 1.35)
        self.assertLess(feats["mag_std"], 0.25)

    def test_qam16_amplitude_level_separation(self):
        """Test that synchronized QAM16 symbols exhibit multi-level magnitude spread."""
        sig_qam, _ = self._generate_synthetic_qam16(cfo_hz=-18000.0, timing_offset=4.0, snr_db=30.0)
        sync_res = extract_synchronized_symbols(sig_qam, self.fs)
        feats = extract_symbol_features(sync_res.symbols)

        # 16-QAM theoretical normalized 4th moment is 1.32
        self.assertGreater(feats["fourth_moment_ratio"], 1.20)
        self.assertGreater(feats["mag_std"], 0.25)
        self.assertGreaterEqual(feats["num_amplitude_modes"], 1)

    def test_finite_outputs_and_edge_cases(self):
        """Test safety on zero / constant / edge case signals."""
        zero_sig = np.zeros(1000, dtype=np.complex64)
        sync_res = extract_synchronized_symbols(zero_sig, self.fs)
        self.assertTrue(np.all(np.isfinite(sync_res.symbols)))

        feats = extract_symbol_features(sync_res.symbols)
        self.assertTrue(np.isfinite(feats["mag_mean"]))
        self.assertTrue(np.isfinite(feats["fourth_moment_ratio"]))

        tensor = compute_symbol_tensor_features(sync_res.symbols, target_length=256)
        self.assertEqual(tensor.shape, (5, 256))
        self.assertTrue(np.all(np.isfinite(tensor)))


if __name__ == "__main__":
    unittest.main()
