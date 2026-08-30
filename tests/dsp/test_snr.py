"""
Tests for SNR and Higher-Order Cumulants (HOC).

Tests:
    1. SNR module independently
    2. HOC module independently
    3. Integration between HOC and SNR
"""

import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dsp.signal_quality.snr import (
    calculate_signal_power,
    estimate_noise_power,
    estimate_snr,
    get_snr_db,
    snr,
    snr_from_powers,
)

from dsp.statistical.hoc import (
    calculate_hoc,
    get_cumulants,
    hoc,
    higher_order_cumulants,
)


class TestSNR(unittest.TestCase):
    """Tests for dsp.signal_quality.snr."""

    # -------------------------------------------------------------
    # Test signal power
    # -------------------------------------------------------------

    def test_signal_power_real(self):
        """Test power calculation for a real signal."""

        signal = np.ones(1000)

        power = calculate_signal_power(
            signal
        )

        self.assertAlmostEqual(
            power,
            1.0,
            places=6,
        )

    def test_signal_power_complex(self):
        """Test power calculation for a complex IQ signal."""

        signal = (
            np.ones(1000)
            + 1j * np.ones(1000)
        )

        power = calculate_signal_power(
            signal
        )

        self.assertAlmostEqual(
            power,
            2.0,
            places=6,
        )

    # -------------------------------------------------------------
    # Test SNR from known powers
    # -------------------------------------------------------------

    def test_snr_from_powers(self):
        """Test direct SNR calculation."""

        snr_db = snr_from_powers(
            signal_power=100.0,
            noise_power=1.0,
        )

        self.assertAlmostEqual(
            snr_db,
            20.0,
            places=6,
        )

    def test_zero_noise(self):
        """Zero noise should produce infinite SNR."""

        result = snr_from_powers(
            signal_power=1.0,
            noise_power=0.0,
        )

        self.assertTrue(
            np.isinf(result)
        )

    # -------------------------------------------------------------
    # Test noise estimation
    # -------------------------------------------------------------

    def test_noise_estimation_difference(self):
        """Test difference-based noise estimation."""

        rng = np.random.default_rng(42)

        noise = (
            rng.normal(
                0.0,
                0.1,
                5000,
            )
        )

        estimated_noise = estimate_noise_power(
            noise,
            method="difference",
        )

        self.assertGreater(
            estimated_noise,
            0.0,
        )

        self.assertLess(
            estimated_noise,
            0.03,
        )

    def test_noise_estimation_variance(self):
        """Test variance-based noise estimation."""

        rng = np.random.default_rng(42)

        noise = (
            rng.normal(
                0.0,
                0.1,
                5000,
            )
        )

        estimated_noise = estimate_noise_power(
            noise,
            method="variance",
        )

        self.assertGreater(
            estimated_noise,
            0.0,
        )

        self.assertAlmostEqual(
            estimated_noise,
            0.01,
            delta=0.003,
        )

    # -------------------------------------------------------------
    # Test complete SNR estimator
    # -------------------------------------------------------------

    def test_estimate_snr_real(self):
        """Test complete SNR estimation for a real signal."""

        rng = np.random.default_rng(10)

        signal = (
            np.ones(5000)
            + rng.normal(
                0.0,
                0.05,
                5000,
            )
        )

        result = estimate_snr(
            signal,
            method="difference",
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "snr_db",
            result,
        )

        self.assertIn(
            "signal_power",
            result,
        )

        self.assertIn(
            "noise_power",
            result,
        )

        self.assertGreater(
            result["snr_db"],
            0.0,
        )

        self.assertGreater(
            result["signal_power"],
            0.0,
        )

        self.assertGreater(
            result["noise_power"],
            0.0,
        )

    def test_estimate_snr_complex(self):
        """Test complete SNR estimation for complex IQ."""

        rng = np.random.default_rng(20)

        clean_signal = (
            np.ones(5000)
            + 1j * np.ones(5000)
        )

        noise = (
            rng.normal(
                0.0,
                0.05,
                5000,
            )
            + 1j
            * rng.normal(
                0.0,
                0.05,
                5000,
            )
        )

        signal = (
            clean_signal
            + noise
        )

        result = estimate_snr(
            signal,
            method="difference",
        )

        self.assertTrue(
            result["is_complex"]
        )

        self.assertGreater(
            result["snr_db"],
            0.0,
        )

    # -------------------------------------------------------------
    # Test compatibility wrappers
    # -------------------------------------------------------------

    def test_snr_wrapper(self):
        """Test the snr() compatibility wrapper."""

        signal = np.ones(1000)

        result = snr(
            signal
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "snr_db",
            result,
        )

    def test_get_snr_db(self):
        """Test get_snr_db()."""

        signal = np.ones(1000)

        value = get_snr_db(
            signal
        )

        self.assertTrue(
            np.isfinite(value)
            or np.isinf(value)
        )

    # -------------------------------------------------------------
    # Test invalid input
    # -------------------------------------------------------------

    def test_empty_signal_rejected(self):
        """Empty signals must be rejected."""

        with self.assertRaises(
            ValueError
        ):
            estimate_snr(
                np.array([])
            )

    def test_multidimensional_signal_rejected(self):
        """Multidimensional signals must be rejected."""

        signal = np.ones(
            (10, 2)
        )

        with self.assertRaises(
            ValueError
        ):
            estimate_snr(
                signal
            )

    def test_invalid_noise_method_rejected(self):
        """Invalid noise-estimation method must be rejected."""

        signal = np.ones(100)

        with self.assertRaises(
            ValueError
        ):
            estimate_noise_power(
                signal,
                method="invalid",
            )


class TestHOC(unittest.TestCase):
    """Tests for dsp.statistical.hoc."""

    # -------------------------------------------------------------
    # Test basic HOC calculation
    # -------------------------------------------------------------

    def test_hoc_real_signal(self):
        """Test HOC calculation for a real signal."""

        rng = np.random.default_rng(100)

        signal = rng.normal(
            0.0,
            1.0,
            5000,
        )

        result = calculate_hoc(
            signal
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "moments",
            result,
        )

        self.assertIn(
            "cumulants",
            result,
        )

        self.assertIn(
            "normalized_cumulants",
            result,
        )

        self.assertIn(
            "C40",
            result,
        )

        self.assertIn(
            "C42",
            result,
        )

    def test_hoc_complex_signal(self):
        """Test HOC calculation for complex IQ."""

        rng = np.random.default_rng(101)

        signal = (
            rng.normal(
                0.0,
                1.0,
                5000,
            )
            + 1j
            * rng.normal(
                0.0,
                1.0,
                5000,
            )
        )

        result = calculate_hoc(
            signal
        )

        self.assertTrue(
            result["is_complex"]
        )

        self.assertEqual(
            result["number_of_samples"],
            5000,
        )

        self.assertIn(
            "C20",
            result,
        )

        self.assertIn(
            "C21",
            result,
        )

        self.assertIn(
            "C40",
            result,
        )

        self.assertIn(
            "C60",
            result,
        )

    # -------------------------------------------------------------
    # Test HOC values
    # -------------------------------------------------------------

    def test_hoc_contains_finite_values(self):
        """HOC results should contain finite numerical values."""

        rng = np.random.default_rng(102)

        signal = (
            rng.normal(
                0.0,
                1.0,
                3000,
            )
            + 1j
            * rng.normal(
                0.0,
                1.0,
                3000,
            )
        )

        result = calculate_hoc(
            signal
        )

        for value in result[
            "cumulants"
        ].values():

            self.assertTrue(
                np.isfinite(
                    value.real
                )
            )

            self.assertTrue(
                np.isfinite(
                    value.imag
                )
            )

    def test_normalized_hoc_values(self):
        """Normalized HOC values should be finite."""

        rng = np.random.default_rng(103)

        signal = (
            rng.normal(
                0.0,
                1.0,
                3000,
            )
            + 1j
            * rng.normal(
                0.0,
                1.0,
                3000,
            )
        )

        result = calculate_hoc(
            signal
        )

        for value in result[
            "normalized_cumulants"
        ].values():

            self.assertTrue(
                np.isfinite(value)
            )

    # -------------------------------------------------------------
    # Test HOC wrappers
    # -------------------------------------------------------------

    def test_higher_order_cumulants_wrapper(self):
        """Test higher_order_cumulants()."""

        signal = np.ones(
            1000
        )

        result = higher_order_cumulants(
            signal
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "cumulants",
            result,
        )

    def test_hoc_wrapper(self):
        """Test hoc()."""

        signal = np.ones(
            1000
        )

        result = hoc(
            signal
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "C40",
            result,
        )

    def test_get_cumulants(self):
        """Test get_cumulants()."""

        signal = np.ones(
            1000
        )

        result = get_cumulants(
            signal
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "C20",
            result,
        )

        self.assertIn(
            "C40",
            result,
        )

        self.assertIn(
            "C42",
            result,
        )


# =====================================================================
# Integration tests
# =====================================================================

class TestHOCAndSNRIntegration(unittest.TestCase):
    """
    Integration tests for HOC and SNR.

    Both modules receive the same communication signal.
    """

    def setUp(self):
        """Create reproducible test signals."""

        rng = np.random.default_rng(999)

        number_of_samples = 8000

        # QPSK-like signal.
        symbol_indices = rng.integers(
            0,
            4,
            number_of_samples,
        )

        constellation = np.array(
            [
                1.0 + 1.0j,
                1.0 - 1.0j,
                -1.0 + 1.0j,
                -1.0 - 1.0j,
            ],
            dtype=np.complex128,
        )

        clean_signal = (
            constellation[
                symbol_indices
            ]
        )

        noise = (
            rng.normal(
                0.0,
                0.15,
                number_of_samples,
            )
            + 1j
            * rng.normal(
                0.0,
                0.15,
                number_of_samples,
            )
        )

        self.signal = (
            clean_signal
            + noise
        )

    # -------------------------------------------------------------
    # Integration test 1
    # -------------------------------------------------------------

    def test_hoc_and_snr_process_same_signal(self):
        """
        Verify that HOC and SNR can process the same IQ signal.
        """

        hoc_result = calculate_hoc(
            self.signal
        )

        snr_result = estimate_snr(
            self.signal,
            method="difference",
        )

        self.assertIsInstance(
            hoc_result,
            dict,
        )

        self.assertIsInstance(
            snr_result,
            dict,
        )

        self.assertIn(
            "C40",
            hoc_result,
        )

        self.assertIn(
            "C42",
            hoc_result,
        )

        self.assertIn(
            "snr_db",
            snr_result,
        )

        self.assertTrue(
            np.isfinite(
                snr_result["snr_db"]
            )
        )

    # -------------------------------------------------------------
    # Integration test 2
    # -------------------------------------------------------------

    def test_hoc_features_and_snr_are_consistent(self):
        """
        Verify that HOC features and SNR values are numerically valid.
        """

        hoc_result = calculate_hoc(
            self.signal
        )

        snr_result = estimate_snr(
            self.signal,
            method="difference",
        )

        c40 = hoc_result[
            "C40_normalized"
        ]

        c42 = hoc_result[
            "C42_normalized"
        ]

        snr_db = snr_result[
            "snr_db"
        ]

        self.assertTrue(
            np.isfinite(c40)
        )

        self.assertTrue(
            np.isfinite(c42)
        )

        self.assertTrue(
            np.isfinite(snr_db)
        )

        self.assertGreater(
            snr_result[
                "signal_power"
            ],
            0.0,
        )

        self.assertGreater(
            snr_result[
                "noise_power"
            ],
            0.0,
        )

    # -------------------------------------------------------------
    # Integration test 3
    # -------------------------------------------------------------

    def test_real_signal_hoc_and_snr(self):
        """
        Verify both modules also work with a real-valued signal.
        """

        rng = np.random.default_rng(500)

        signal = (
            np.cos(
                2.0
                * np.pi
                * 0.1
                * np.arange(6000)
            )
            + rng.normal(
                0.0,
                0.1,
                6000,
            )
        )

        hoc_result = calculate_hoc(
            signal
        )

        snr_result = estimate_snr(
            signal,
            method="difference",
        )

        self.assertFalse(
            hoc_result["is_complex"]
        )

        self.assertFalse(
            snr_result["is_complex"]
        )

        self.assertIn(
            "C40",
            hoc_result,
        )

        self.assertIn(
            "snr_db",
            snr_result,
        )

        self.assertTrue(
            np.isfinite(
                snr_result["snr_db"]
            )
        )


# =====================================================================
# Test runner
# =====================================================================

if __name__ == "__main__":
    unittest.main()