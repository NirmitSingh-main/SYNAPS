import unittest
import tempfile
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.io import wavfile

from backend.api import signal
from dsp.frequency.frequency_estimation import (
    estimate_frequency,
    load_signal,
    estimate_frequency_from_file,
)
class TestFrequencyEstimation(unittest.TestCase):
    def test_wav_file_frequency(self):
        """Test frequency estimation from an actual WAV file."""

        sampling_rate = 1_000_000
        true_frequency = 100_000
        duration = 0.01

        time = np.arange(
            0,
            duration,
            1 / sampling_rate
        )

        # Create a real-valued test signal.
        signal = np.cos(
            2 * np.pi * true_frequency * time
        )

        # Convert to int16, similar to a normal WAV recording.
        wav_signal = np.int16(
            signal * 32767
        )

        # Create a temporary WAV file.
        with tempfile.TemporaryDirectory() as temp_directory:

            wav_path = Path(temp_directory) / "test_signal.wav"

            wavfile.write(
                wav_path,
                sampling_rate,
                wav_signal
            )

            # Run the complete file-based estimator.
            result = estimate_frequency_from_file(
                wav_path
            )

        estimated_frequency = result[
            "estimated_frequency_hz"
        ]

        self.assertLess(
            abs(
                estimated_frequency -
                true_frequency
            ),
            1_000
        )

        self.assertEqual(
            result["file_type"],
            ".wav"
        )

        self.assertEqual(
            result["sampling_rate_hz"],
            sampling_rate
        )


    def test_iq_file_frequency(self):
        """Test frequency estimation from an actual raw IQ file."""

        sampling_rate = 1_000_000
        true_frequency = 150_000
        duration = 0.01

        time = np.arange(
            0,
            duration,
            1 / sampling_rate
        )

        # Create a complex IQ signal.
        signal = np.exp(
            1j * 2 * np.pi * true_frequency * time
        )

        # Store as interleaved float32:
        #
        # I, Q, I, Q, I, Q, ...
        iq_data = np.empty(
            signal.size * 2,
            dtype=np.float32
        )

        iq_data[0::2] = np.real(signal)
        iq_data[1::2] = np.imag(signal)

        with tempfile.TemporaryDirectory() as temp_directory:

            iq_path = Path(temp_directory) / "test_signal.iq"

            iq_data.tofile(iq_path)

            # Run the complete file-based estimator.
            result = estimate_frequency_from_file(
                iq_path,
                sampling_rate=sampling_rate,
                iq_dtype=np.float32,
                iq_order="IQ"
            )

        estimated_frequency = result[
            "estimated_frequency_hz"
        ]

        self.assertLess(
            abs(
                estimated_frequency -
                true_frequency
            ),
            1_000
        )

        self.assertEqual(
            result["file_type"],
            ".iq"
        )

        self.assertEqual(
            result["sampling_rate_hz"],
            sampling_rate
        )

        self.assertEqual(
            result["signal_type"],
            "complex_iq"
        )
    
    def test_real_signal_frequency(self):
        sampling_rate = 1_000_000
        true_frequency = 100_000

        duration = 0.01

        time = np.arange(
            0,
            duration,
            1 / sampling_rate
        )

        signal = np.cos(
            2 * np.pi * true_frequency * time
        )

        result = estimate_frequency(
            signal,
            sampling_rate
        )

        estimated_frequency = result["estimated_frequency_hz"]

        self.assertLess(
            abs(estimated_frequency - true_frequency),
            1_000
        )

    def test_complex_iq_frequency(self):
        sampling_rate = 1_000_000
        true_frequency = 150_000

        duration = 0.01

        time = np.arange(
            0,
            duration,
            1 / sampling_rate
        )

        signal = np.exp(
            1j * 2 * np.pi * true_frequency * time
        )

        result = estimate_frequency(
            signal,
            sampling_rate
        )

        estimated_frequency = result["estimated_frequency_hz"]

        self.assertLess(
            abs(estimated_frequency - true_frequency),
            1_000
        )


if __name__ == "__main__":
    unittest.main()