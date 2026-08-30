"""
Comprehensive Signal Intelligence Report Generator (JSON & Text).
"""

from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path


def generate_intelligence_report(
    analysis_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Format and structure the final intelligence report.
    """
    input_info = analysis_result.get("input_info", {})
    ai_info = analysis_result.get("ai_classification", {})
    dsp_info = analysis_result.get("dsp_analysis", {})
    sync_info = analysis_result.get("synchronization", {})
    demod_info = analysis_result.get("demodulation", {})
    decode_info = analysis_result.get("decoding", {})
    decision_info = analysis_result.get("decision", {})
    fingerprint_info = analysis_result.get("fingerprint", {})
    evidence_info = analysis_result.get("evidence", {})

    report = {
        "report_title": "SYNAPS SIGNAL INTELLIGENCE REPORT",
        "signal_id": input_info.get("sample_id", "UNKNOWN"),
        "source_file": input_info.get("file_path"),
        "format": input_info.get("format"),
        "sample_count": input_info.get("sample_count", 0),
        "sample_rate_hz": input_info.get("sample_rate_hz", 1_000_000.0),
        
        "modulation_decision": {
            "final_modulation": decision_info.get("final_modulation", ai_info.get("predicted_class", "UNKNOWN")),
            "decision_status": decision_info.get("decision_status", "CONFIRMED"),
            "confidence_pct": ai_info.get("confidence", 0.0),
            "detection_status": ai_info.get("status", "KNOWN"),
            "probabilities": ai_info.get("probabilities", {}),
        },

        "dsp_metrics": {
            "snr_db": dsp_info.get("snr_db", 0.0),
            "carrier_frequency_offset_hz": dsp_info.get("cfo_hz", 0.0),
            "bandwidth_3db_hz": dsp_info.get("bandwidth_hz", 0.0),
            "symbol_rate": dsp_info.get("symbol_rate", 0.0),
            "peak_frequency_hz": dsp_info.get("peak_frequency_hz", 0.0),
        },

        "recovery_pipeline": {
            "synchronization_status": sync_info.get("status", "SUCCESS"),
            "recovered_symbols": demod_info.get("symbol_count", 0),
            "recovered_bits": demod_info.get("bit_count", 0),
            "decoded_message": decode_info.get("decoded_message"),
            "payload_entropy": decode_info.get("entropy", 0.0),
        },

        "emitter_fingerprint": fingerprint_info,
        "evidence_summary": {
            "overall_score": evidence_info.get("overall_evidence_score", 1.0),
            "supported_modulation": evidence_info.get("primary_supported_modulation"),
        },
    }

    return report


def format_text_report(report: Dict[str, Any]) -> str:
    """
    Format structured report into a clean, human-readable text document.
    """
    lines = [
        "=" * 60,
        "               SYNAPS SIGNAL INTELLIGENCE REPORT",
        "=" * 60,
        f"Signal ID       : {report.get('signal_id')}",
        f"Source File     : {report.get('source_file')}",
        f"Format          : {report.get('format')}",
        f"Sample Count    : {report.get('sample_count')} samples",
        f"Sampling Rate   : {report.get('sample_rate_hz'):,.0f} Hz",
        "",
        "MODULATION CLASSIFICATION & DECISION",
        "-" * 40,
        f"Predicted Class : {report['modulation_decision']['final_modulation']}",
        f"Confidence      : {report['modulation_decision']['confidence_pct']:.2f}%",
        f"Status          : {report['modulation_decision']['detection_status']} ({report['modulation_decision']['decision_status']})",
        "",
        "Class Probabilities:",
    ]

    for mod, prob in report["modulation_decision"]["probabilities"].items():
        lines.append(f"  - {mod:<8}: {prob:.2f}%")

    dsp = report["dsp_metrics"]
    lines.extend([
        "",
        "DSP PHYSICAL ESTIMATES",
        "-" * 40,
        f"SNR             : {dsp.get('snr_db', 0.0):.2f} dB",
        f"CFO             : {dsp.get('carrier_frequency_offset_hz', 0.0):.2f} Hz",
        f"Bandwidth (3dB) : {dsp.get('bandwidth_3db_hz', 0.0):,.0f} Hz",
        f"Symbol Rate     : {dsp.get('symbol_rate', 0.0):,.0f} Baud",
        "",
        "SIGNAL RECOVERY & DECODING",
        "-" * 40,
        f"Recovered Bits  : {report['recovery_pipeline']['recovered_bits']}",
        f"Decoded Message : {repr(report['recovery_pipeline']['decoded_message']) if report['recovery_pipeline']['decoded_message'] else 'None'}",
        "",
        "EMITTER FINGERPRINT",
        "-" * 40,
        f"Fingerprint ID  : {report['emitter_fingerprint'].get('fingerprint_id', 'N/A')}",
        f"PAPR            : {report['emitter_fingerprint'].get('papr_db', 0.0):.2f} dB",
        "=" * 60,
    ])

    return "\n".join(lines)


def save_report(
    report: Dict[str, Any],
    output_dir: Path,
    base_name: str = "report",
) -> Tuple[Path, Path]:
    """
    Save JSON and TXT reports to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{base_name}.json"
    txt_path = output_dir / f"{base_name}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_text_report(report))

    return json_path, txt_path