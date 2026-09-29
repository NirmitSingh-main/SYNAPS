"""
Final modulation decision engine combining AI predictions and DSP physical confirmation.
"""

from typing import Any, Dict, List, Optional
from project_paths import CLASS_NAMES, normalize_modulation_name


def make_modulation_decision(
    ai_prediction: str,
    ai_confidence: float,
    evidence: Dict[str, Any],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Produce the final verified modulation decision and confidence status.
    """
    norm_ai_pred = normalize_modulation_name(ai_prediction)
    conf = float(ai_confidence) / 100.0 if ai_confidence > 1.0 else float(ai_confidence)

    evidence_score = float(evidence.get("overall_evidence_score", conf))

    if conf >= threshold and evidence_score >= threshold:
        decision = norm_ai_pred
        decision_status = "CONFIRMED"
    elif conf >= 0.40:
        decision = norm_ai_pred
        decision_status = "TENTATIVE"
    else:
        decision = "UNKNOWN"
        decision_status = "UNRESOLVED"

    return {
        "final_modulation": decision,
        "decision_status": decision_status,
        "composite_confidence": float(conf),
        "evidence_score": float(evidence_score),
    }