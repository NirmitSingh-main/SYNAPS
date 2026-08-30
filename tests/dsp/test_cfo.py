import os
import tempfile
import unittest
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.io import wavfile

from dsp.frequency.cfo import (
    estimate_cfo,
    estimate_cfo_from_file,
)


class TestCFO(unittest.TestCase):

    def setUp(self):
        """
        Common parameters used by the tests.
        """

        self.sampling_rate = 1_000_000
        self.true_frequency = 100_000
        self.reference_frequency = 99_000

        # 0.01 second signal.
        self.duration = 0.01

        self.time = np.arange(
            0,
            self.duration,
            1 / self.sampling_rate,
        )

        # Expected CFO:
        #
        # 100,000 - 99,000 = 1,000 Hz
        self.expected_cfo = (
            self.true_frequency
            - self.reference_frequency
        )

    def test_real_signal_cfo(self):
        """
        Test Carrier Frequency Offset estimation
        using a real-valued signal.
        """

        signal = np.cos(
            2 * np.pi
            * self.true_frequency
            * self.time
        )

        result = estimate_cfo(
            signal,
            self.sampling_rate,
            self.reference_frequency,
        )

        estimated_cfo = result["cfo_hz"]

        self.assertAlmostEqual(
            estimated_cfo,
            self.expected_cfo,
            delta=100,
        )

    def test_complex_iq_signal_cfo(self):
        """
        Test Carrier Frequency Offset estimation
        using a complex In-phase/Quadrature signal.
        """

        signal = np.exp(
            1j
            * 2
            * np.pi
            * self.true_frequency
            * self.time
        )

        result = estimate_cfo(
            signal,
            self.sampling_rate,
            self.reference_frequency,
        )

        estimated_cfo = result["cfo_hz"]

        self.assertAlmostEqual(
            estimated_cfo,
            self.expected_cfo,
            delta=100,
        )

    def test_wav_file_cfo(self):
        """
        Test Carrier Frequency Offset estimation
        using a WAV file.
        """

        signal = np.cos(
            2 * np.pi
            * self.true_frequency
            * self.time
        )

        # Convert signal to 16-bit WAV samples.
        wav_signal = np.int16(
            signal
            * 32767
        )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:

            file_path = temporary_file.name

        try:

            wavfile.write(
                file_path,
                self.sampling_rate,
                wav_signal,
            )

            result = estimate_cfo_from_file(
                file_path,
                self.reference_frequency,
            )

            estimated_cfo = result["cfo_hz"]

            self.assertAlmostEqual(
                estimated_cfo,
                self.expected_cfo,
                delta=100,
            )

            self.assertEqual(
                result["file_type"],
                ".wav",
            )

            self.assertEqual(
                result["sampling_rate_hz"],
                self.sampling_rate,
            )

        finally:

            if os.path.exists(file_path):
                os.remove(file_path)

    def test_iq_file_cfo(self):
        """
        Test Carrier Frequency Offset estimation
        using a raw interleaved IQ file.
        """

        signal = np.exp(
            1j
            * 2
            * np.pi
            * self.true_frequency
            * self.time
        )

        # Create interleaved:
        #
        # I, Q, I, Q, I, Q, ...
        iq_data = np.empty(
            signal.size * 2,
            dtype=np.float32,
        )

        iq_data[0::2] = np.real(signal)
        iq_data[1::2] = np.imag(signal)

        with tempfile.NamedTemporaryFile(
            suffix=".iq",
            delete=False,
        ) as temporary_file:

            file_path = temporary_file.name

        try:

            iq_data.tofile(file_path)

            result = estimate_cfo_from_file(
                file_path,
                self.reference_frequency,
                sampling_rate=self.sampling_rate,
            )

            estimated_cfo = result["cfo_hz"]

            self.assertAlmostEqual(
                estimated_cfo,
                self.expected_cfo,
                delta=100,
            )

            self.assertEqual(
                result["file_type"],
                ".iq",
            )

            self.assertEqual(
                result["sampling_rate_hz"],
                self.sampling_rate,
            )

        finally:

            if os.path.exists(file_path):
                os.remove(file_path)


if __name__ == "__main__":
    unittest.main()