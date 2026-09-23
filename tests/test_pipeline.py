"""
End-to-end integration test for the SYNAPS Signal Intelligence pipeline.
Tests pure signals (BPSK, QPSK, FSK, QAM16) and MIXED multi-carrier signals.
"""

import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pipeline import analyze_signal
from backend.schemas.response import IntelligenceReportResponse
from project_paths import (
    CLASS_NAMES,
    get_class_iq_dir,
    get_class_wav_dir,
    resolve_sample_paths,
)


class TestPipeline(unittest.TestCase):

    def test_pipeline_on_bpsk_sample(self):
        """Test full analysis pipeline on BPSK IQ file."""
        bpsk_dir = get_class_iq_dir("BPSK")
        bpsk_files = list(bpsk_dir.glob("*.iq"))
        self.assertTrue(len(bpsk_files) > 0, "No BPSK IQ files found")
        iq_file = bpsk_files[0]

        results = analyze_signal(str(iq_file))

        self.assertIn("input_info", results)
        self.assertIn("dsp_analysis", results)
        self.assertIn("synchronization", results)
        self.assertIn("ai_classification", results)
        self.assertIn("decision", results)
        self.assertIn("demodulation", results)
        self.assertIn("bit_recovery", results)
        self.assertIn("report", results)
        self.assertIn("frontend_data", results)

        rep = results["report"]
        self.assertEqual(rep["report_title"], "SYNAPS SIGNAL INTELLIGENCE REPORT")
        self.assertEqual(rep["format"], "IQ")
        self.assertEqual(results["signal_type"], "SINGLE")
        self.assertIn(rep["sample_count"], [8000, 9600])
        self.assertIn(rep["modulation_decision"]["final_modulation"], CLASS_NAMES + ["UNKNOWN"])
        self.assertGreater(rep["dsp_metrics"]["snr_db"], -100.0)

        # Validate bit recovery structure
        br = results["bit_recovery"]
        self.assertIn("validation_status", br)
        self.assertIsNotNone(br.get("recovered_bit_count"))

        # Verify Pydantic schema validation
        validated_schema = IntelligenceReportResponse(**rep)
        self.assertIsNotNone(validated_schema)

        print("[PASS] test_pipeline_on_bpsk_sample")

    def test_pipeline_on_qpsk_wav_sample(self):
        """Test full analysis pipeline on QPSK WAV file."""
        qpsk_dir = get_class_wav_dir("QPSK")
        qpsk_files = list(qpsk_dir.glob("*.wav"))
        self.assertTrue(len(qpsk_files) > 0, "No QPSK WAV files found")
        wav_file = qpsk_files[0]

        results = analyze_signal(str(wav_file))

        rep = results["report"]
        self.assertEqual(rep["format"], "WAV")
        self.assertEqual(results["signal_type"], "SINGLE")
        self.assertIn(rep["sample_count"], [8000, 9600])
        self.assertIn(rep["modulation_decision"]["final_modulation"], CLASS_NAMES + ["UNKNOWN"])

        # Verify Pydantic schema validation
        validated_schema = IntelligenceReportResponse(**rep)
        self.assertIsNotNone(validated_schema)

        print("[PASS] test_pipeline_on_qpsk_wav_sample")

    def test_pipeline_on_fsk_sample(self):
        """Test full analysis pipeline on FSK IQ file."""
        fsk_dir = get_class_iq_dir("FSK")
        fsk_files = list(fsk_dir.glob("*.iq"))
        self.assertTrue(len(fsk_files) > 0, "No FSK IQ files found")
        iq_file = fsk_files[0]

        results = analyze_signal(str(iq_file))
        self.assertEqual(results["signal_type"], "SINGLE")
        self.assertIn("fsk_tone_metrics", results["dsp_analysis"])
        print("[PASS] test_pipeline_on_fsk_sample")

    def test_pipeline_on_qam16_sample(self):
        """Test full analysis pipeline on QAM16 IQ file."""
        qam_dir = get_class_iq_dir("QAM16")
        qam_files = list(qam_dir.glob("*.iq"))
        self.assertTrue(len(qam_files) > 0, "No QAM16 IQ files found")
        iq_file = qam_files[0]

        results = analyze_signal(str(iq_file))
        self.assertEqual(results["signal_type"], "SINGLE")
        self.assertGreater(results["dsp_analysis"]["snr_db"], -100.0)
        print("[PASS] test_pipeline_on_qam16_sample")

    def test_pipeline_on_2component_mixed(self):
        """Test full analysis pipeline on 2-component MIXED IQ file."""
        mixed_dir = get_class_iq_dir("MIXED")
        mixed_files = list(mixed_dir.glob("*.iq"))
        self.assertTrue(len(mixed_files) > 0, "No MIXED IQ files found")
        iq_file = mixed_files[0]  # signal_03001_mixed.iq

        results = analyze_signal(str(iq_file))

        self.assertEqual(results["signal_type"], "MIXED")
        self.assertEqual(results["demodulation"]["bit_count"], 0)
        self.assertEqual(results["bit_recovery"]["validation_status"], "COMPONENT_RECOVERY_NOT_VALIDATED")
        self.assertTrue(len(results["detected_components"]) >= 2)
        self.assertTrue(len(results["component_results"]) >= 2)

        # Verify Pydantic schema validation
        rep = results["report"]
        self.assertEqual(rep["signal_type"], "MIXED")
        validated_schema = IntelligenceReportResponse(**rep)
        self.assertIsNotNone(validated_schema)

        print("[PASS] test_pipeline_on_2component_mixed")

    def test_pipeline_on_3component_mixed(self):
        """Test full analysis pipeline on 3-component MIXED IQ file."""
        mixed_dir = get_class_iq_dir("MIXED")
        target_file = mixed_dir / "signal_03301_mixed.iq"
        if not target_file.exists():
            target_file = list(mixed_dir.glob("*.iq"))[-1]

        results = analyze_signal(str(target_file))

        self.assertEqual(results["signal_type"], "MIXED")
        self.assertEqual(results["demodulation"]["bit_count"], 0)
        self.assertEqual(results["bit_recovery"]["validation_status"], "COMPONENT_RECOVERY_NOT_VALIDATED")
        self.assertTrue(len(results["detected_components"]) >= 3)
        self.assertTrue(len(results["component_results"]) >= 3)

        # Verify Pydantic schema validation
        rep = results["report"]
        validated_schema = IntelligenceReportResponse(**rep)
        self.assertIsNotNone(validated_schema)

        print("[PASS] test_pipeline_on_3component_mixed")


if __name__ == "__main__":
    unittest.main()