"""
Out-of-distribution and unknown modulation detector.
"""

from typing import Dict, Any


def is_unknown_signal(
    confidence: float,
    threshold: float = 0.60,
) -> bool:
    """
    Check if signal confidence falls below the known threshold.
    """
    conf = float(confidence) / 100.0 if confidence > 1.0 else float(confidence)
    return conf < threshold


def evaluate_unknown_status(
    confidence: float,
    threshold: float = 0.60,
) -> Dict[str, Any]:
    """
    Evaluate signal known/unknown status.
    """
    unknown = is_unknown_signal(confidence, threshold)
    return {
        "status": "UNKNOWN" if unknown else "KNOWN",
        "is_unknown": unknown,
        "confidence": float(confidence),
        "threshold": float(threshold),
    }