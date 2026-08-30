import unittest
import sys
from pathlib import Path
import tempfile

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.io import wavfile

from dsp.phase.phase_estimation import (
    estimate_phase,
    estimate_phase_from_file,
)


class TestPhaseEstimation(unittest.TestCase):

    def setUp(self):
        self.sampling_rate = 1_000_000
        self.frequency = 100_000
        self.phase = np.pi / 4

        self.duration = 0.01

        self.time = np.arange(
            0,
            self.duration,
            1 / self.sampling_rate,
        )

        self.real_signal = np.cos(
            2 * np.pi * self.frequency * self.time
            + self.phase
        )

        self.complex_signal = np.exp(
            1j
            * (
                2 * np.pi
                * self.frequency
                * self.time
                + self.phase
            )
        )

    def test_real_signal_phase(self):
        """
        Test phase estimation on a real-valued signal.
        """

        result = estimate_phase(
            self.real_signal,
            self.sampling_rate,
            self.frequency,
        )

        estimated_phase = result[
            "estimated_phase_radians"
        ]

        phase_error = abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        estimated_phase
                        - self.phase
                    )
                )
            )
        )

        self.assertLess(
            phase_error,
            0.05,
        )

    def test_complex_iq_signal_phase(self):
        """
        Test phase estimation on a complex
        In-phase/Quadrature signal.
        """

        result = estimate_phase(
            self.complex_signal,
            self.sampling_rate,
            self.frequency,
        )

        estimated_phase = result[
            "estimated_phase_radians"
        ]

        phase_error = abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        estimated_phase
                        - self.phase
                    )
                )
            )
        )

        self.assertLess(
            phase_error,
            0.05,
        )

    def test_wav_file_phase(self):
        """
        Test phase estimation from a WAV file.
        """

        wav_signal = (
            self.real_signal * 32767
        ).astype(np.int16)

        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = (
                Path(temp_dir)
                / "test_signal.wav"
            )

            wavfile.write(
                file_path,
                self.sampling_rate,
                wav_signal,
            )

            result = estimate_phase_from_file(
                file_path,
                frequency_hz=self.frequency,
            )

            estimated_phase = result[
                "estimated_phase_radians"
            ]

            phase_error = abs(
                np.angle(
                    np.exp(
                        1j
                        * (
                            estimated_phase
                            - self.phase
                        )
                    )
                )
            )

            self.assertLess(
                phase_error,
                0.05,
            )

def test_iq_file_phase(self):
    """
    Test phase estimation from a raw IQ file.
    """

    i_samples = np.asarray(
        np.real(self.complex_signal),
        dtype=np.float32,
    )

    q_samples = np.asarray(
        np.imag(self.complex_signal),
        dtype=np.float32,
    )

    iq_data = np.empty(
        self.complex_signal.size * 2,
        dtype=np.float32,
    )

    iq_data[0::2] = i_samples
    iq_data[1::2] = q_samples

    with tempfile.TemporaryDirectory() as temp_dir:

        file_path = (
            Path(temp_dir)
            / "test_signal.iq"
        )

        iq_data.tofile(file_path)

        result = estimate_phase_from_file(
            file_path,
            frequency_hz=self.frequency,
            sampling_rate=self.sampling_rate,
        )

        estimated_phase = result[
            "estimated_phase_radians"
        ]

        phase_error = abs(
            np.angle(
                np.exp(
                    1j
                    * (
                        estimated_phase
                        - self.phase
                    )
                )
            )
        )

        self.assertLess(
            phase_error,
            0.05,
        )

    def test_invalid_empty_signal(self):
        """
        Empty signals should be rejected.
        """

        with self.assertRaises(ValueError):

            estimate_phase(
                np.array([]),
                self.sampling_rate,
                self.frequency,
            )

    def test_invalid_sampling_rate(self):
        """
        Invalid sampling rates should be rejected.
        """

        with self.assertRaises(ValueError):

            estimate_phase(
                self.real_signal,
                0,
                self.frequency,
            )

    def test_invalid_file_format(self):
        """
        Unsupported file formats should be rejected.
        """

        with tempfile.TemporaryDirectory() as temp_dir:

            file_path = (
                Path(temp_dir)
                / "test_signal.txt"
            )

            file_path.write_text(
                "invalid signal"
            )

            with self.assertRaises(ValueError):

                estimate_phase_from_file(
                    file_path,
                    frequency_hz=self.frequency,
                    sampling_rate=self.sampling_rate,
                )


if __name__ == "__main__":
    unittest.main()