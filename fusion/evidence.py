"""
Multi-modal evidence aggregator for signal modulation assessment.
"""

from typing import Any, Dict, List
import numpy as np


def aggregate_evidence(
    fused_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate evidentiary support across AI predictions and physical DSP properties.
    """
    dsp_sum = fused_features.get("dsp_summary", {})
    ai_sum = fused_features.get("ai_summary", {})
    ai_pred = ai_sum.get("predicted_class", "UNKNOWN")
    ai_conf = ai_sum.get("confidence", 0.0)
    snr_db = dsp_sum.get("snr_db", 0.0)

    evidence_items: List[Dict[str, Any]] = []

    # 1. AI Evidence
    evidence_items.append({
        "source": "Transformer Neural Network",
        "supported_modulation": ai_pred,
        "weight": 0.50,
        "score": float(ai_conf) / 100.0 if ai_conf > 1.0 else float(ai_conf),
        "description": f"AI model assigned {ai_conf:.1f}% confidence to {ai_pred}.",
    })

    # 2. SNR Evidence
    snr_score = min(max((snr_db + 5.0) / 25.0, 0.0), 1.0)
    evidence_items.append({
        "source": "Signal-to-Noise Ratio (SNR)",
        "supported_modulation": ai_pred,
        "weight": 0.20,
        "score": float(snr_score),
        "description": f"Estimated SNR is {snr_db:.1f} dB.",
    })

    # 3. Statistical / HOC Evidence
    c42 = dsp_sum.get("hoc_c42", 0.0)
    hoc_mod = "QAM16" if c42 > 0.5 else ("FSK" if c42 < 0.1 else ai_pred)
    evidence_items.append({
        "source": "Higher-Order Cumulants",
        "supported_modulation": hoc_mod,
        "weight": 0.15,
        "score": 0.85,
        "description": f"Cumulant profile C42={c42:.3f}.",
    })

    # 4. Spectral Bandwidth Evidence
    bw_hz = dsp_sum.get("bandwidth_hz", 0.0)
    evidence_items.append({
        "source": "Spectral Bandwidth",
        "supported_modulation": ai_pred,
        "weight": 0.15,
        "score": 0.90,
        "description": f"Estimated 3dB bandwidth is {bw_hz / 1e3:.1f} kHz.",
    })

    # Overall evidence score
    total_score = sum(item["weight"] * item["score"] for item in evidence_items)

    return {
        "overall_evidence_score": float(total_score),
        "primary_supported_modulation": ai_pred,
        "evidence_breakdown": evidence_items,
    }