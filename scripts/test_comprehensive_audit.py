import json
from pathlib import Path
import numpy as np

from backend.schemas.response import (
    IntelligenceReportResponse,
    ModulationPrediction,
    DspMetrics,
    RecoveryPipelineSummary,
    BitRecoverySummary,
)
from backend.services.pipeline import analyze_signal
from project_paths import resolve_sample_paths, DATA_ROOT


def run_comprehensive_audit_tests():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE INTEGRATION AUDIT TESTS")
    print("=" * 80)

    # 1. Pure BPSK
    print("\n--- 1. Testing Pure BPSK ---")
    p_bpsk = str(resolve_sample_paths("signal_00001_bpsk")["iq_path"])
    res_bpsk = analyze_signal(p_bpsk)
    assert res_bpsk["signal_type"] == "SINGLE"
    assert res_bpsk["decision"]["final_modulation"] in ("BPSK", "QPSK", "FSK", "QAM16")
    assert res_bpsk["bit_recovery"]["validation_status"] in ("VALIDATED", "SUBOPTIMAL", "POOR_ALIGNMENT")
    assert res_bpsk["bit_recovery"]["recovered_bit_count"] is not None
    assert res_bpsk["bit_recovery"]["recovered_bit_count"] > 0
    assert res_bpsk["decoding"]["decoding_status"] in ("DECODED_ASCII", "UNSTRUCTURED_BINARY")
    rep_bpsk = IntelligenceReportResponse(**res_bpsk["report"])
    assert rep_bpsk is not None
    print(f"[PASS] BPSK Recovered Bits: {res_bpsk['bit_recovery']['recovered_bit_count']} | Status: {res_bpsk['bit_recovery']['validation_status']}")

    # 2. Pure QPSK
    print("\n--- 2. Testing Pure QPSK ---")
    p_qpsk = str(resolve_sample_paths("signal_00751_qpsk")["iq_path"])
    res_qpsk = analyze_signal(p_qpsk)
    assert res_qpsk["signal_type"] == "SINGLE"
    assert res_qpsk["decision"]["final_modulation"] in ("BPSK", "QPSK", "FSK", "QAM16")
    assert res_qpsk["bit_recovery"]["validation_status"] in ("VALIDATED", "SUBOPTIMAL", "POOR_ALIGNMENT")
    assert res_qpsk["bit_recovery"]["recovered_bit_count"] is not None
    assert res_qpsk["bit_recovery"]["recovered_bit_count"] > 0
    rep_qpsk = IntelligenceReportResponse(**res_qpsk["report"])
    assert rep_qpsk is not None
    print(f"[PASS] QPSK Recovered Bits: {res_qpsk['bit_recovery']['recovered_bit_count']} | Status: {res_qpsk['bit_recovery']['validation_status']}")

    # 3. Pure FSK
    print("\n--- 3. Testing Pure FSK ---")
    p_fsk = str(resolve_sample_paths("signal_01501_fsk")["iq_path"])
    res_fsk = analyze_signal(p_fsk)
    assert res_fsk["signal_type"] == "SINGLE"
    assert res_fsk["decision"]["final_modulation"] in ("BPSK", "QPSK", "FSK", "QAM16")
    assert res_fsk["bit_recovery"]["validation_status"] in ("VALIDATED", "SUBOPTIMAL", "POOR_ALIGNMENT")
    assert res_fsk["bit_recovery"]["recovered_bit_count"] is not None
    assert res_fsk["bit_recovery"]["recovered_bit_count"] > 0
    rep_fsk = IntelligenceReportResponse(**res_fsk["report"])
    assert rep_fsk is not None
    print(f"[PASS] FSK Recovered Bits: {res_fsk['bit_recovery']['recovered_bit_count']} | Status: {res_fsk['bit_recovery']['validation_status']}")

    # 4. Pure QAM16
    print("\n--- 4. Testing Pure QAM16 ---")
    p_qam = str(resolve_sample_paths("signal_02251_qam16")["iq_path"])
    res_qam = analyze_signal(p_qam)
    assert res_qam["signal_type"] == "SINGLE"
    assert res_qam["decision"]["final_modulation"] in ("BPSK", "QPSK", "FSK", "QAM16")
    assert res_qam["bit_recovery"]["validation_status"] in ("VALIDATED", "SUBOPTIMAL", "POOR_ALIGNMENT")
    assert res_qam["bit_recovery"]["recovered_bit_count"] is not None
    assert res_qam["bit_recovery"]["recovered_bit_count"] > 0
    rep_qam = IntelligenceReportResponse(**res_qam["report"])
    assert rep_qam is not None
    print(f"[PASS] QAM16 Recovered Bits: {res_qam['bit_recovery']['recovered_bit_count']} | Status: {res_qam['bit_recovery']['validation_status']}")

    # 5. MIXED 2-component
    print("\n--- 5. Testing MIXED 2-Component ---")
    p_m2 = str(resolve_sample_paths("signal_03001_mixed")["iq_path"])
    res_m2 = analyze_signal(p_m2)
    assert res_m2["signal_type"] == "MIXED"
    assert res_m2["bit_recovery"]["validation_status"] == "COMPONENT_RECOVERY_NOT_VALIDATED"
    assert res_m2["bit_recovery"]["recovered_bit_count"] is None
    assert res_m2["frontend_data"]["recoveredBitCount"] is None
    assert res_m2["decoding"]["decoding_status"] == "NOT_APPLICABLE"
    assert len(res_m2["detected_components"]) >= 2
    rep_m2 = IntelligenceReportResponse(**res_m2["report"])
    assert rep_m2 is not None
    print(f"[PASS] MIXED-2 Components: {res_m2['detected_components']} | Bit Recovery: {res_m2['bit_recovery']['validation_status']}")

    # 6. MIXED 3-component
    print("\n--- 6. Testing MIXED 3-Component ---")
    p_m3 = str(resolve_sample_paths("signal_03301_mixed")["iq_path"])
    res_m3 = analyze_signal(p_m3)
    assert res_m3["signal_type"] == "MIXED"
    assert res_m3["bit_recovery"]["validation_status"] == "COMPONENT_RECOVERY_NOT_VALIDATED"
    assert res_m3["bit_recovery"]["recovered_bit_count"] is None
    assert res_m3["frontend_data"]["recoveredBitCount"] is None
    assert res_m3["decoding"]["decoding_status"] == "NOT_APPLICABLE"
    assert len(res_m3["detected_components"]) >= 3
    rep_m3 = IntelligenceReportResponse(**res_m3["report"])
    assert rep_m3 is not None
    print(f"[PASS] MIXED-3 Components: {res_m3['detected_components']} | Bit Recovery: {res_m3['bit_recovery']['validation_status']}")

    # 7. External File Without Metadata
    print("\n--- 7. Testing External File Without Metadata ---")
    # Generate a temporary synthetic IQ file without JSON metadata in uploads dir
    temp_dir = DATA_ROOT / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_iq_path = temp_dir / "external_test_signal.iq"
    
    t = np.linspace(0, 0.01, 10000)
    # Simple synthetic BPSK
    symbols = np.random.choice([-1.0, 1.0], size=1000)
    sig_ext = np.repeat(symbols, 10).astype(np.float32)
    sig_iq = np.zeros(20000, dtype=np.float32)
    sig_iq[0::2] = sig_ext
    sig_iq[1::2] = 0.0
    sig_iq.tofile(str(temp_iq_path))

    res_ext = analyze_signal(str(temp_iq_path), sample_rate=1000000.0)
    assert res_ext["bit_recovery"]["validation_status"] == "UNVALIDATED_NO_METADATA"
    assert res_ext["bit_recovery"]["reference_bit_count"] is None
    assert res_ext["bit_recovery"]["bit_accuracy_pct"] is None
    assert res_ext["bit_recovery"]["ber"] is None
    assert res_ext["bit_recovery"]["recovered_bit_count"] > 0
    assert res_ext["frontend_data"]["recoveredBitCount"] > 0
    rep_ext = IntelligenceReportResponse(**res_ext["report"])
    assert rep_ext is not None
    print(f"[PASS] External Signal Recovered Bits: {res_ext['bit_recovery']['recovered_bit_count']} | Status: {res_ext['bit_recovery']['validation_status']}")

    # 8. Backend Schema Serialization
    print("\n--- 8. Testing Backend Schema Serialization ---")
    json_str = rep_ext.model_dump_json()
    assert len(json_str) > 0
    print("[PASS] Pydantic model_dump_json succeeded")

    # 9. Decoding Status Serialization
    print("\n--- 9. Testing Decoding Status Serialization ---")
    assert rep_ext.recovery_pipeline.decoding_status is not None
    assert rep_ext.recovery_pipeline.fec_status == "NOT_CONFIGURED"
    print(f"[PASS] decoding_status: {rep_ext.recovery_pipeline.decoding_status}, fec_status: {rep_ext.recovery_pipeline.fec_status}")

    print("\n" + "=" * 80)
    print("ALL COMPREHENSIVE INTEGRATION AUDIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_audit_tests()
