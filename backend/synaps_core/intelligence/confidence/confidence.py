"""
Integrated multi-factor confidence scoring for signal intelligence.
"""

from typing import Any, Dict


def compute_composite_confidence(
    ai_confidence: float,
    snr_db: float,
    evidence_score: float = 1.0,
) -> Dict[str, Any]:
    """
    Calculate composite confidence from AI, SNR, and multi-modal evidence.
    """
    ai_c = float(ai_confidence) / 100.0 if ai_confidence > 1.0 else float(ai_confidence)
    snr_factor = min(max((snr_db + 5.0) / 25.0, 0.2), 1.0)
    ev_factor = min(max(float(evidence_score), 0.2), 1.0)

    # Weighted combination
    composite = 0.60 * ai_c + 0.20 * snr_factor + 0.20 * ev_factor
    composite = min(max(composite, 0.0), 1.0)

    return {
        "composite_confidence": float(composite),
        "composite_confidence_pct": float(composite * 100.0),
        "ai_confidence_pct": float(ai_c * 100.0),
        "snr_db": float(snr_db),
        "evidence_factor": float(ev_factor),
    }