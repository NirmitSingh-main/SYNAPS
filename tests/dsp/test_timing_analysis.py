"""
Combined tests for:

    dsp.timing.symbol_rate
    dsp.cyclostationary.analysis

Tests:
    - Individual symbol-rate estimation
    - Individual cyclostationary analysis
    - Real-valued signals
    - Complex IQ signals
    - Input validation
    - Cyclic autocorrelation
    - Cyclic-frequency estimation
    - Integration between symbol-rate and
      cyclostationary analysis
"""


import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from dsp.timing.symbol_rate import (
    estimate_symbol_rate,
    estimate_symbol_rate_from_signal,
    get_symbol_rate,
)

from dsp.cyclostationary.analysis import (
    calculate_cyclic_autocorrelation,
    estimate_cyclic_frequencies,
    analyze_cyclostationarity,
    get_cyclic_frequencies,
)


class TestSymbolRateEstimation(unittest.TestCase):
    """
    Tests for symbol_rate.py only.
    """

    def setUp(self):
        self.sampling_rate = 10000.0
        self.symbol_rate = 1000.0
        self.samples_per_symbol = 10

        self.number_of_symbols = 200

        rng = np.random.default_rng(42)

        self.symbols = rng.choice(
            [-1.0, 1.0],
            size=self.number_of_symbols,
        )

        self.signal = np.repeat(
            self.symbols,
            self.samples_per_symbol,
        )

        self.signal = (
            self.signal
            + 0.02
            * rng.normal(
                size=self.signal.size
            )
        )

        self.iq_signal = (
            self.signal
            + 1j
            * np.zeros_like(
                self.signal
            )
        )

    # -------------------------------------------------------------
    # Basic real signal test
    # -------------------------------------------------------------

    def test_symbol_rate_real_signal(self):
        """
        Symbol-rate estimator should process a real-valued signal.
        """

        result = estimate_symbol_rate(
            self.signal,
            self.sampling_rate,
            minimum_symbol_rate_hz=100.0,
            maximum_symbol_rate_hz=3000.0,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "symbol_rate_hz",
            result,
        )

        self.assertIn(
            "samples_per_symbol",
            result,
        )

        self.assertIn(
            "confidence",
            result,
        )

        self.assertGreater(
            result["symbol_rate_hz"],
            0.0,
        )

        self.assertGreater(
            result["samples_per_symbol"],
            0.0,
        )

    # -------------------------------------------------------------
    # Complex IQ signal test
    # -------------------------------------------------------------

    def test_symbol_rate_complex_signal(self):
        """
        Symbol-rate estimator should process complex IQ data.
        """

        result = estimate_symbol_rate(
            self.iq_signal,
            self.sampling_rate,
            minimum_symbol_rate_hz=100.0,
            maximum_symbol_rate_hz=3000.0,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertTrue(
            result["is_complex"]
        )

        self.assertGreater(
            result["symbol_rate_hz"],
            0.0,
        )

    # -------------------------------------------------------------
    # Wrapper test
    # -------------------------------------------------------------

    def test_symbol_rate_wrapper(self):
        """
        Compatibility wrapper should produce a valid result.
        """

        result = estimate_symbol_rate_from_signal(
            self.signal,
            self.sampling_rate,
            minimum_symbol_rate_hz=100.0,
            maximum_symbol_rate_hz=3000.0,
        )

        self.assertIn(
            "symbol_rate_hz",
            result,
        )

        self.assertGreater(
            result["symbol_rate_hz"],
            0.0,
        )

    # -------------------------------------------------------------
    # Convenience function
    # -------------------------------------------------------------

    def test_get_symbol_rate(self):
        """
        get_symbol_rate() should return a float.
        """

        result = get_symbol_rate(
            self.signal,
            self.sampling_rate,
            minimum_symbol_rate_hz=100.0,
            maximum_symbol_rate_hz=3000.0,
        )

        self.assertIsInstance(
            result,
            float,
        )

        self.assertGreater(
            result,
            0.0,
        )

    # -------------------------------------------------------------
    # Sampling-rate validation
    # -------------------------------------------------------------

    def test_invalid_sampling_rate(self):
        """
        Invalid sampling rate should raise ValueError.
        """

        with self.assertRaises(
            ValueError
        ):

            estimate_symbol_rate(
                self.signal,
                0.0,
            )

    # -------------------------------------------------------------
    # Empty signal validation
    # -------------------------------------------------------------

    def test_empty_signal(self):
        """
        Empty input should raise ValueError.
        """

        with self.assertRaises(
            ValueError
        ):

            estimate_symbol_rate(
                np.array([]),
                self.sampling_rate,
            )

    # -------------------------------------------------------------
    # Multidimensional signal validation
    # -------------------------------------------------------------

    def test_multidimensional_signal(self):
        """
        Multidimensional input should raise ValueError.
        """

        invalid_signal = np.zeros(
            (10, 2)
        )

        with self.assertRaises(
            ValueError
        ):

            estimate_symbol_rate(
                invalid_signal,
                self.sampling_rate,
            )


class TestCyclostationaryAnalysis(unittest.TestCase):
    """
    Tests for analysis.py only.
    """

    def setUp(self):
        self.sampling_rate = 10000.0
        self.symbol_rate = 1000.0
        self.samples_per_symbol = 10

        self.number_of_symbols = 200

        rng = np.random.default_rng(123)

        self.symbols = rng.choice(
            [-1.0, 1.0],
            size=self.number_of_symbols,
        )

        self.signal = np.repeat(
            self.symbols,
            self.samples_per_symbol,
        )

        self.signal = (
            self.signal
            + 0.02
            * rng.normal(
                size=self.signal.size
            )
        )

        self.iq_signal = (
            self.signal
            + 1j
            * np.zeros_like(
                self.signal
            )
        )

    # -------------------------------------------------------------
    # Cyclic autocorrelation
    # -------------------------------------------------------------

    def test_cyclic_autocorrelation_real(self):
        """
        Test cyclic autocorrelation on a real signal.
        """

        result = calculate_cyclic_autocorrelation(
            self.signal,
            self.sampling_rate,
            self.symbol_rate,
            max_lag_samples=50,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "lags_samples",
            result,
        )

        self.assertIn(
            "cyclic_autocorrelation",
            result,
        )

        self.assertIn(
            "magnitude",
            result,
        )

        self.assertEqual(
            len(
                result["lags_samples"]
            ),
            len(
                result[
                    "cyclic_autocorrelation"
                ]
            ),
        )

    # -------------------------------------------------------------
    # Cyclic autocorrelation complex
    # -------------------------------------------------------------

    def test_cyclic_autocorrelation_complex(self):
        """
        Test cyclic autocorrelation on complex IQ data.
        """

        result = calculate_cyclic_autocorrelation(
            self.iq_signal,
            self.sampling_rate,
            self.symbol_rate,
            max_lag_samples=50,
        )

        self.assertTrue(
            np.iscomplexobj(
                result[
                    "cyclic_autocorrelation"
                ]
            )
        )

        self.assertGreater(
            len(
                result["lags_samples"]
            ),
            0,
        )

    # -------------------------------------------------------------
    # Cyclic-frequency estimation
    # -------------------------------------------------------------

    def test_estimate_cyclic_frequencies(self):
        """
        Test cyclic-frequency estimation.
        """

        result = estimate_cyclic_frequencies(
            self.signal,
            self.sampling_rate,
            minimum_cyclic_frequency_hz=100.0,
            maximum_cyclic_frequency_hz=3000.0,
            number_of_frequencies=128,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "cyclic_frequencies_hz",
            result,
        )

        self.assertIn(
            "spectral_correlation",
            result,
        )

        self.assertIn(
            "spectral_correlation_magnitude",
            result,
        )

        self.assertIn(
            "peak_cyclic_frequency_hz",
            result,
        )

        self.assertGreater(
            result[
                "peak_cyclic_frequency_hz"
            ],
            0.0,
        )

    # -------------------------------------------------------------
    # Complex cyclic-frequency estimation
    # -------------------------------------------------------------

    def test_estimate_cyclic_frequencies_complex(self):
        """
        Test cyclic-frequency estimation on IQ data.
        """

        result = estimate_cyclic_frequencies(
            self.iq_signal,
            self.sampling_rate,
            minimum_cyclic_frequency_hz=100.0,
            maximum_cyclic_frequency_hz=3000.0,
            number_of_frequencies=128,
        )

        self.assertIn(
            "peak_cyclic_frequency_hz",
            result,
        )

        self.assertGreater(
            result[
                "peak_magnitude"
            ],
            0.0,
        )

    # -------------------------------------------------------------
    # Complete analysis
    # -------------------------------------------------------------

    def test_complete_cyclostationary_analysis(self):
        """
        Test the complete analysis pipeline.
        """

        result = analyze_cyclostationarity(
            self.signal,
            self.sampling_rate,
            minimum_cyclic_frequency_hz=100.0,
            maximum_cyclic_frequency_hz=3000.0,
            number_of_frequencies=128,
            max_lag_samples=50,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "peak_cyclic_frequency_hz",
            result,
        )

        self.assertIn(
            "detection_score",
            result,
        )

        self.assertIn(
            "cyclic_autocorrelation",
            result,
        )

        self.assertIn(
            "cyclic_autocorrelation_magnitude",
            result,
        )

        self.assertGreater(
            result[
                "peak_cyclic_frequency_hz"
            ],
            0.0,
        )

        self.assertGreaterEqual(
            result[
                "detection_score"
            ],
            0.0,
        )

    # -------------------------------------------------------------
    # Convenience function
    # -------------------------------------------------------------

    def test_get_cyclic_frequencies(self):
        """
        Test cyclic-frequency convenience function.
        """

        result = get_cyclic_frequencies(
            self.signal,
            self.sampling_rate,
            minimum_cyclic_frequency_hz=100.0,
            maximum_cyclic_frequency_hz=3000.0,
            number_of_frequencies=64,
        )

        self.assertIsInstance(
            result,
            np.ndarray,
        )

        self.assertEqual(
            result.size,
            64,
        )

        self.assertTrue(
            np.all(
                np.isfinite(result)
            )
        )

    # -------------------------------------------------------------
    # Invalid signal
    # -------------------------------------------------------------

    def test_invalid_signal(self):
        """
        Invalid signal should raise ValueError.
        """

        with self.assertRaises(
            ValueError
        ):

            analyze_cyclostationarity(
                np.array([]),
                self.sampling_rate,
            )

    # -------------------------------------------------------------
    # Invalid sampling rate
    # -------------------------------------------------------------

    def test_invalid_sampling_rate(self):
        """
        Invalid sampling rate should raise ValueError.
        """

        with self.assertRaises(
            ValueError
        ):

            analyze_cyclostationarity(
                self.signal,
                0.0,
            )


class TestTimingAndCyclostationaryIntegration(
    unittest.TestCase
):
    """
    Integration tests for symbol-rate estimation and
    cyclostationary analysis.
    """

    def setUp(self):
        self.sampling_rate = 12000.0
        self.symbol_rate = 1000.0
        self.samples_per_symbol = 12

        self.number_of_symbols = 250

        rng = np.random.default_rng(999)

        symbols = rng.choice(
            [-1.0, 1.0],
            size=self.number_of_symbols,
        )

        self.iq_signal = np.repeat(
            symbols,
            self.samples_per_symbol,
        ).astype(
            np.complex128
        )

        # Add small complex noise.
        noise = (
            0.01
            * (
                rng.normal(
                    size=self.iq_signal.size
                )
                + 1j
                * rng.normal(
                    size=self.iq_signal.size
                )
            )
        )

        self.iq_signal += noise

    # -------------------------------------------------------------
    # Integration test
    # -------------------------------------------------------------

    def test_symbol_rate_and_cyclostationary_integration(
        self
    ):
        """
        Verify that symbol-rate estimation and
        cyclostationary analysis can operate on
        the same IQ signal.
        """

        symbol_result = estimate_symbol_rate(
            self.iq_signal,
            self.sampling_rate,
            minimum_symbol_rate_hz=200.0,
            maximum_symbol_rate_hz=3000.0,
        )

        self.assertIn(
            "symbol_rate_hz",
            symbol_result,
        )

        estimated_symbol_rate = float(
            symbol_result[
                "symbol_rate_hz"
            ]
        )

        self.assertGreater(
            estimated_symbol_rate,
            0.0,
        )

        # Use the symbol-rate estimate to guide
        # the cyclostationary analysis.
        lower_frequency = max(
            1.0,
            estimated_symbol_rate * 0.5,
        )

        upper_frequency = min(
            self.sampling_rate / 2.0,
            estimated_symbol_rate * 2.0,
        )

        cyclostationary_result = (
            analyze_cyclostationarity(
                self.iq_signal,
                self.sampling_rate,
                minimum_cyclic_frequency_hz=(
                    lower_frequency
                ),
                maximum_cyclic_frequency_hz=(
                    upper_frequency
                ),
                number_of_frequencies=128,
                max_lag_samples=50,
            )
        )

        self.assertIn(
            "peak_cyclic_frequency_hz",
            cyclostationary_result,
        )

        self.assertIn(
            "detection_score",
            cyclostationary_result,
        )

        self.assertGreater(
            cyclostationary_result[
                "peak_cyclic_frequency_hz"
            ],
            0.0,
        )

    # -------------------------------------------------------------
    # Cross-module consistency
    # -------------------------------------------------------------

    def test_symbol_rate_and_cyclic_frequency_are_consistent(
        self
    ):
        """
        The detected cyclic frequency should be in the same
        general frequency region as the estimated symbol rate.
        """

        symbol_result = estimate_symbol_rate(
            self.iq_signal,
            self.sampling_rate,
            minimum_symbol_rate_hz=200.0,
            maximum_symbol_rate_hz=3000.0,
        )

        estimated_symbol_rate = float(
            symbol_result[
                "symbol_rate_hz"
            ]
        )

        cyclic_result = (
            estimate_cyclic_frequencies(
                self.iq_signal,
                self.sampling_rate,
                minimum_cyclic_frequency_hz=(
                    estimated_symbol_rate * 0.5
                ),
                maximum_cyclic_frequency_hz=(
                    min(
                        self.sampling_rate / 2.0,
                        estimated_symbol_rate * 2.0,
                    )
                ),
                number_of_frequencies=128,
            )
        )

        cyclic_frequency = float(
            cyclic_result[
                "peak_cyclic_frequency_hz"
            ]
        )

        self.assertGreater(
            cyclic_frequency,
            0.0,
        )

        # The cyclic frequency can be a harmonic or another
        # periodic component, so allow a broad relationship.
        ratio = (
            cyclic_frequency
            / estimated_symbol_rate
        )

        self.assertGreater(
            ratio,
            0.25,
        )

        self.assertLess(
            ratio,
            4.0,
        )

    # -------------------------------------------------------------
    # Real + IQ integration
    # -------------------------------------------------------------

    def test_real_and_iq_pipeline(self):
        """
        Both real-valued and complex IQ signals should pass
        through the two modules.
        """

        real_signal = np.real(
            self.iq_signal
        )

        symbol_result_real = (
            estimate_symbol_rate(
                real_signal,
                self.sampling_rate,
                minimum_symbol_rate_hz=200.0,
                maximum_symbol_rate_hz=3000.0,
            )
        )

        symbol_result_iq = (
            estimate_symbol_rate(
                self.iq_signal,
                self.sampling_rate,
                minimum_symbol_rate_hz=200.0,
                maximum_symbol_rate_hz=3000.0,
            )
        )

        self.assertGreater(
            symbol_result_real[
                "symbol_rate_hz"
            ],
            0.0,
        )

        self.assertGreater(
            symbol_result_iq[
                "symbol_rate_hz"
            ],
            0.0,
        )

        real_analysis = (
            analyze_cyclostationarity(
                real_signal,
                self.sampling_rate,
                minimum_cyclic_frequency_hz=100.0,
                maximum_cyclic_frequency_hz=3000.0,
                number_of_frequencies=64,
                max_lag_samples=30,
            )
        )

        iq_analysis = (
            analyze_cyclostationarity(
                self.iq_signal,
                self.sampling_rate,
                minimum_cyclic_frequency_hz=100.0,
                maximum_cyclic_frequency_hz=3000.0,
                number_of_frequencies=64,
                max_lag_samples=30,
            )
        )

        self.assertGreater(
            real_analysis[
                "peak_cyclic_frequency_hz"
            ],
            0.0,
        )

        self.assertGreater(
            iq_analysis[
                "peak_cyclic_frequency_hz"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()