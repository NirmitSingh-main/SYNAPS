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
    bit_rec_info = analysis_result.get("bit_recovery", {})
    comp_results = analysis_result.get("component_results", [])
    signal_type = analysis_result.get("signal_type", "SINGLE")

    report = {
        "report_title": "SYNAPS SIGNAL INTELLIGENCE REPORT",
        "signal_id": input_info.get("sample_id", "UNKNOWN"),
        "source_file": input_info.get("file_path"),
        "format": input_info.get("format"),
        "sample_count": input_info.get("sample_count", 0),
        "sample_rate_hz": input_info.get("sample_rate_hz", 1_000_000.0),
        "signal_type": signal_type,
        
        "modulation_decision": {
            "final_modulation": decision_info.get("final_modulation", ai_info.get("predicted_class", "UNKNOWN")),
            "decision_status": decision_info.get("decision_status", "CONFIRMED"),
            "confidence_pct": ai_info.get("confidence", 0.0),
            "detection_status": ai_info.get("status", "KNOWN"),
            "probabilities": ai_info.get("probabilities", {}),
            "detected_components": analysis_result.get("detected_components", []),
        },

        "dsp_metrics": {
            "snr_db": dsp_info.get("snr_db", 0.0),
            "carrier_frequency_offset_hz": dsp_info.get("cfo_hz", 0.0),
            "occupied_bandwidth_hz": dsp_info.get("bandwidth_hz", 0.0),
            "symbol_rate": dsp_info.get("symbol_rate", 0.0),
            "peak_frequency_hz": dsp_info.get("peak_frequency_hz", 0.0),
        },

        "recovery_pipeline": {
            "synchronization_status": sync_info.get("status", "SUCCESS"),
            "recovered_symbols": demod_info.get("symbol_count", 0),
            "recovered_bits": demod_info.get("bit_count", 0),
            "decoded_message": decode_info.get("decoded_message"),
            "decoding_status": decode_info.get("decoding_status", "RAW_BITSTREAM_UNSTRUCTURED"),
            "fec_status": decode_info.get("fec_status", "NOT_CONFIGURED"),
            "payload_entropy": decode_info.get("entropy", 0.0),
        },

        "bit_recovery": bit_rec_info,
        "component_results": comp_results,

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
    signal_type = report.get("signal_type", "SINGLE")
    mod_decision = report.get("modulation_decision", {})
    detected_comps = mod_decision.get("detected_components", [])

    lines = [
        "=" * 60,
        "               SYNAPS SIGNAL INTELLIGENCE REPORT",
        "=" * 60,
        f"Signal ID       : {report.get('signal_id')}",
        f"Source File     : {report.get('source_file')}",
        f"Format          : {report.get('format')}",
        f"Signal Type     : {signal_type}",
        f"Sample Count    : {report.get('sample_count')} samples",
        f"Sampling Rate   : {report.get('sample_rate_hz'):,.0f} Hz",
        "",
        "MODULATION CLASSIFICATION & DECISION",
        "-" * 40,
        f"Final Decision  : {mod_decision.get('final_modulation')}",
        f"Confidence      : {mod_decision.get('confidence_pct', 0.0):.2f}%",
        f"Status          : {mod_decision.get('detection_status', 'KNOWN')} ({mod_decision.get('decision_status', 'CONFIRMED')})",
    ]

    if detected_comps:
        lines.append(f"Detected Comps  : {', '.join(detected_comps)}")

    lines.extend([
        "",
        "Class Probabilities:",
    ])

    for mod, prob in mod_decision.get("probabilities", {}).items():
        lines.append(f"  - {mod:<8}: {prob:.2f}%")

    dsp = report.get("dsp_metrics", {})
    lines.extend([
        "",
        "DSP PHYSICAL ESTIMATES",
        "-" * 40,
        f"SNR             : {dsp.get('snr_db', 0.0):.2f} dB",
        f"CFO             : {dsp.get('carrier_frequency_offset_hz', 0.0):.2f} Hz",
        f"Occupied BW (99%): {dsp.get('occupied_bandwidth_hz', 0.0):,.0f} Hz",
        f"Symbol Rate     : {dsp.get('symbol_rate', 0.0):,.0f} Baud",
    ])

    recovered_bits_count = report['recovery_pipeline'].get('recovered_bits', 0)
    recovered_bits_display = "—" if signal_type == "MIXED" else str(recovered_bits_count)

    lines.extend([
        "",
        "SIGNAL RECOVERY & DECODING",
        "-" * 40,
        f"Recovered Bits  : {recovered_bits_display}",
        f"Decoded Message : {repr(report['recovery_pipeline'].get('decoded_message')) if report['recovery_pipeline'].get('decoded_message') else 'None'}",
    ])

    bit_rec = report.get("bit_recovery", {})
    if bit_rec:
        lines.extend([
            "",
            "BIT RECOVERY VALIDATION",
            "-" * 40,
            f"Status          : {bit_rec.get('validation_status', 'N/A')}",
        ])
        if bit_rec.get("validation_status") in ("COMPONENT_RECOVERY_NOT_VALIDATED", "Component bit recovery not validated"):
            lines.append(f"Reference Bits  : —")
            lines.append(f"Recovered Bits  : —")
            lines.append(f"Bit Accuracy    : —")
            lines.append(f"BER             : —")
        elif bit_rec.get("reference_bit_count") is not None:
            lines.append(f"Reference Bits  : {bit_rec.get('reference_bit_count')}")
            lines.append(f"Recovered Bits  : {bit_rec.get('recovered_bit_count')}")
            lines.append(f"Matched Bits    : {bit_rec.get('matched_bit_count')}")
            lines.append(f"Bit Accuracy    : {bit_rec.get('bit_accuracy_pct', 0.0):.2f}%")
            lines.append(f"Bit Error Rate  : {bit_rec.get('ber', 0.0):.4f}")
        else:
            # UNVALIDATED_NO_METADATA or NO_RECOVERED_BITS
            lines.append(f"Reference Bits  : —")
            rec_count = bit_rec.get('recovered_bit_count')
            lines.append(f"Recovered Bits  : {rec_count if rec_count is not None else '—'}")
            lines.append(f"Bit Accuracy    : —")
            lines.append(f"BER             : —")

    lines.extend([
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