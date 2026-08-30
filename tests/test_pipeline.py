"""
End-to-end integration test for the SYNAPS Signal Intelligence pipeline.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pipeline import analyze_signal
from project_paths import IQ_ROOT, WAV_ROOT


def test_pipeline_on_iq_sample():
    """
    Test full analysis pipeline execution on a canonical IQ file.
    """
    iq_file = IQ_ROOT / "BPSK" / "signal_0001_bpsk.iq"
    assert iq_file.exists(), f"Sample file not found: {iq_file}"

    results = analyze_signal(str(iq_file))

    # Verify high-level structure
    assert "input_info" in results
    assert "dsp_analysis" in results
    assert "synchronization" in results
    assert "ai_classification" in results
    assert "decision" in results
    assert "demodulation" in results
    assert "fingerprint" in results
    assert "report" in results

    # Verify report contents
    rep = results["report"]
    assert rep["report_title"] == "SYNAPS SIGNAL INTELLIGENCE REPORT"
    assert rep["format"] == "IQ"
    assert rep["sample_count"] == 8192
    assert rep["modulation_decision"]["final_modulation"] in ["BPSK", "QPSK", "FSK", "QAM16"]
    assert rep["dsp_metrics"]["snr_db"] > -100.0
    assert rep["emitter_fingerprint"]["fingerprint_id"].startswith("RF-FP-")

    print("[PASS] test_pipeline_on_iq_sample: Successfully executed full pipeline on IQ file")


def test_pipeline_on_wav_sample():
    """
    Test full analysis pipeline execution on a canonical WAV file.
    """
    wav_file = WAV_ROOT / "QPSK" / "signal_0201_qpsk.wav"
    assert wav_file.exists(), f"Sample file not found: {wav_file}"

    results = analyze_signal(str(wav_file))

    rep = results["report"]
    assert rep["format"] == "WAV"
    assert rep["sample_count"] == 8192
    assert rep["modulation_decision"]["final_modulation"] in ["BPSK", "QPSK", "FSK", "QAM16"]
    print("[PASS] test_pipeline_on_wav_sample: Successfully executed full pipeline on WAV file")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING END-TO-END PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    test_pipeline_on_iq_sample()
    test_pipeline_on_wav_sample()
    print("\nALL PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY!")