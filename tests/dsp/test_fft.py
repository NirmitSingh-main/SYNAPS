"""
Tests for the spectral DSP modules.

Tests:
    - fft.py
    - psd.py
    - bandwidth.py

Run with:

    py -m unittest tests.dsp.test_fft
"""

import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dsp.spectral.fft import compute_fft
from dsp.spectral.psd import (
    estimate_psd,
    find_peak_frequency,
    calculate_total_power,
    psd_to_db,
)
from dsp.spectral.bandwidth import (
    estimate_bandwidth,
    estimate_bandwidth_from_psd,
)


class TestSpectralProcessing(unittest.TestCase):
    """
    Test FFT, PSD, and bandwidth functionality.
    """

    def setUp(self) -> None:
        """
        Create a synthetic communication-style signal.
        """

        self.sampling_rate = 1000.0

        self.number_of_samples = 4096

        self.frequency = 100.0

        self.time = (
            np.arange(
                self.number_of_samples,
                dtype=np.float64,
            )
            / self.sampling_rate
        )

        self.signal = np.sin(
            2.0
            * np.pi
            * self.frequency
            * self.time
        )

        self.complex_signal = np.exp(
            1j
            * 2.0
            * np.pi
            * self.frequency
            * self.time
        )

    # ================================================================
    # FFT TESTS
    # ================================================================

    def test_fft_real_signal(self) -> None:
        """
        Test FFT processing of a real-valued signal.
        """

        result = compute_fft(
            self.signal,
            self.sampling_rate,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "frequencies_hz",
            result,
        )

        self.assertIn(
            "spectrum",
            result,
        )

        frequencies = np.asarray(
            result["frequencies_hz"]
        )

        spectrum = np.asarray(
            result["spectrum"]
        )

        self.assertEqual(
            frequencies.ndim,
            1,
        )

        self.assertEqual(
            spectrum.ndim,
            1,
        )

        self.assertEqual(
            frequencies.size,
            spectrum.size,
        )

        self.assertGreater(
            frequencies.size,
            0,
        )

    def test_fft_detects_frequency(self) -> None:
        """
        Verify that FFT identifies the dominant signal frequency.
        """

        result = compute_fft(
            self.signal,
            self.sampling_rate,
        )

        self.assertIn(
            "peak_frequency_hz",
            result,
        )

        detected_frequency = float(
            result["peak_frequency_hz"]
        )

        frequency_error = abs(
            detected_frequency
            - self.frequency
        )

        frequency_resolution = (
            self.sampling_rate
            / self.number_of_samples
        )

        self.assertLessEqual(
            frequency_error,
            frequency_resolution * 2.0,
        )

    def test_fft_complex_signal(self) -> None:
        """
        Test FFT processing of a complex IQ signal.
        """

        result = compute_fft(
            self.complex_signal,
            self.sampling_rate,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "frequencies_hz",
            result,
        )

        self.assertIn(
            "spectrum",
            result,
        )

    def test_fft_rejects_empty_signal(self) -> None:
        """
        FFT must reject an empty signal.
        """

        with self.assertRaises(
            ValueError
        ):

            compute_fft(
                np.array([]),
                self.sampling_rate,
            )

    # ================================================================
    # PSD TESTS
    # ================================================================

    def test_psd_returns_required_values(self) -> None:
        """
        Verify PSD output structure.
        """

        result = estimate_psd(
            self.signal,
            self.sampling_rate,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        required_keys = [
            "frequencies_hz",
            "psd",
            "sampling_rate_hz",
            "segment_length",
            "overlap",
            "overlap_samples",
            "total_power",
            "peak_frequency_hz",
            "peak_psd",
        ]

        for key in required_keys:

            self.assertIn(
                key,
                result,
            )

    def test_psd_detects_frequency(self) -> None:
        """
        Verify that PSD identifies the dominant frequency.
        """

        result = estimate_psd(
            self.signal,
            self.sampling_rate,
        )

        detected_frequency = float(
            result["peak_frequency_hz"]
        )

        self.assertAlmostEqual(
            detected_frequency,
            self.frequency,
            delta=5.0,
        )

    def test_psd_power_is_positive(self) -> None:
        """
        Verify that PSD produces positive signal power.
        """

        result = estimate_psd(
            self.signal,
            self.sampling_rate,
        )

        total_power = float(
            result["total_power"]
        )

        self.assertGreater(
            total_power,
            0.0,
        )

    def test_psd_complex_signal(self) -> None:
        """
        Test PSD calculation for complex IQ data.
        """

        result = estimate_psd(
            self.complex_signal,
            self.sampling_rate,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        frequencies = np.asarray(
            result["frequencies_hz"]
        )

        psd = np.asarray(
            result["psd"]
        )

        self.assertEqual(
            frequencies.size,
            psd.size,
        )

        self.assertTrue(
            np.all(
                np.isfinite(psd)
            )
        )

    def test_find_peak_frequency(self) -> None:
        """
        Test the standalone peak-frequency helper.
        """

        frequencies = np.array(
            [
                0.0,
                50.0,
                100.0,
                150.0,
            ]
        )

        psd = np.array(
            [
                0.1,
                0.2,
                10.0,
                0.3,
            ]
        )

        peak = find_peak_frequency(
            frequencies,
            psd,
        )

        self.assertEqual(
            peak,
            100.0,
        )

    def test_calculate_total_power(self) -> None:
        """
        Test total power calculation from a PSD.
        """

        frequencies = np.array(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        )

        psd = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

        power = calculate_total_power(
            frequencies,
            psd,
        )

        self.assertAlmostEqual(
            power,
            3.0,
            places=6,
        )

    def test_psd_to_db(self) -> None:
        """
        Test PSD-to-decibel conversion.
        """

        psd = np.array(
            [
                1.0,
                10.0,
                100.0,
            ]
        )

        result = psd_to_db(
            psd
        )

        self.assertTrue(
            np.allclose(
                result,
                np.array(
                    [
                        0.0,
                        10.0,
                        20.0,
                    ]
                ),
            )
        )

    # ================================================================
    # BANDWIDTH TESTS
    # ================================================================

    def test_bandwidth_returns_required_values(
        self,
    ) -> None:
        """
        Verify bandwidth output structure.
        """

        result = estimate_bandwidth(
            self.signal,
            self.sampling_rate,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        required_keys = [
            "bandwidth_hz",
            "lower_frequency_hz",
            "upper_frequency_hz",
            "occupied_fraction",
            "sampling_rate_hz",
        ]

        for key in required_keys:

            self.assertIn(
                key,
                result,
            )

    def test_bandwidth_is_non_negative(
        self,
    ) -> None:
        """
        Bandwidth must never be negative.
        """

        result = estimate_bandwidth(
            self.signal,
            self.sampling_rate,
        )

        bandwidth = float(
            result["bandwidth_hz"]
        )

        self.assertGreaterEqual(
            bandwidth,
            0.0,
        )

    def test_bandwidth_frequency_order(
        self,
    ) -> None:
        """
        Verify lower frequency is not greater than upper frequency.
        """

        result = estimate_bandwidth(
            self.signal,
            self.sampling_rate,
        )

        lower_frequency = float(
            result["lower_frequency_hz"]
        )

        upper_frequency = float(
            result["upper_frequency_hz"]
        )

        self.assertLessEqual(
            lower_frequency,
            upper_frequency,
        )

    def test_bandwidth_occupied_fraction(
        self,
    ) -> None:
        """
        Test a custom occupied-power percentage.
        """

        result = estimate_bandwidth(
            self.signal,
            self.sampling_rate,
            occupied_fraction=0.95,
        )

        self.assertAlmostEqual(
            float(
                result["occupied_fraction"]
            ),
            0.95,
        )

    def test_bandwidth_from_psd(
        self,
    ) -> None:
        """
        Test bandwidth calculation using an already calculated PSD.
        """

        psd_result = estimate_psd(
            self.signal,
            self.sampling_rate,
        )

        result = estimate_bandwidth_from_psd(
            psd_result["frequencies_hz"],
            psd_result["psd"],
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "bandwidth_hz",
            result,
        )

        self.assertIn(
            "lower_frequency_hz",
            result,
        )

        self.assertIn(
            "upper_frequency_hz",
            result,
        )

        bandwidth = float(
            result["bandwidth_hz"]
        )

        self.assertGreaterEqual(
            bandwidth,
            0.0,
        )

    def test_bandwidth_rejects_invalid_fraction(
        self,
    ) -> None:
        """
        Bandwidth must reject invalid occupied fractions.
        """

        with self.assertRaises(
            ValueError
        ):

            estimate_bandwidth(
                self.signal,
                self.sampling_rate,
                occupied_fraction=1.5,
            )

    # ================================================================
    # CROSS-MODULE TEST
    # ================================================================

    def test_fft_psd_and_bandwidth_work_together(
        self,
    ) -> None:
        """
        Verify that FFT, PSD, and bandwidth modules can process
        the same communication signal.
        """

        fft_result = compute_fft(
            self.signal,
            self.sampling_rate,
        )

        psd_result = estimate_psd(
            self.signal,
            self.sampling_rate,
        )

        bandwidth_result = estimate_bandwidth(
            self.signal,
            self.sampling_rate,
        )

        self.assertIsInstance(
            fft_result,
            dict,
        )

        self.assertIsInstance(
            psd_result,
            dict,
        )

        self.assertIsInstance(
            bandwidth_result,
            dict,
        )

        fft_frequency = float(
            fft_result["peak_frequency_hz"]
        )

        psd_frequency = float(
            psd_result["peak_frequency_hz"]
        )

        bandwidth = float(
            bandwidth_result["bandwidth_hz"]
        )

        self.assertAlmostEqual(
            fft_frequency,
            self.frequency,
            delta=5.0,
        )

        self.assertAlmostEqual(
            psd_frequency,
            self.frequency,
            delta=5.0,
        )

        self.assertGreaterEqual(
            bandwidth,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()