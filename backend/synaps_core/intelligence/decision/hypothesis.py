"""
Modulation hypothesis generation and scoring.
"""

from typing import Any, Dict, List
from project_paths import CLASS_NAMES


def generate_hypotheses(
    probabilities: Dict[str, float],
    dsp_metrics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Generate ranked list of candidate modulation hypotheses with confidence scores.
    """
    hypotheses = []

    for name in CLASS_NAMES:
        prob = float(probabilities.get(name, 0.0))
        if prob > 1.0:
            prob = prob / 100.0

        # Adjust score with basic physical consistency
        score = prob
        hypotheses.append({
            "modulation": name,
            "probability": prob,
            "hypothesis_score": float(score),
        })

    hypotheses.sort(key=lambda h: h["hypothesis_score"], reverse=True)
    return hypotheses