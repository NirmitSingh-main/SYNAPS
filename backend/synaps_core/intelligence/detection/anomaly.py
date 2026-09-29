"""
Signal anomaly and physical inconsistency detection.
"""

from typing import Any, Dict, List
import numpy as np


def detect_anomalies(
    dsp_metrics: Dict[str, Any],
    ai_confidence: float,
) -> List[str]:
    """
    Detect physical anomalies such as severe SNR degradation, extreme CFO, or low confidence.
    """
    anomalies = []
    snr_db = float(dsp_metrics.get("snr_db", dsp_metrics.get("snr", {}).get("snr_db", 10.0)))
    cfo_hz = abs(float(dsp_metrics.get("cfo_hz", dsp_metrics.get("cfo", {}).get("cfo_hz", 0.0))))

    if snr_db < 0.0:
        anomalies.append(f"Low SNR warning: {snr_db:.1f} dB indicates heavy noise interference.")

    if cfo_hz > 100_000:
        anomalies.append(f"Large Carrier Frequency Offset: {cfo_hz:.0f} Hz.")

    conf = float(ai_confidence) / 100.0 if ai_confidence > 1.0 else float(ai_confidence)
    if conf < 0.50:
        anomalies.append(f"Low AI classification confidence: {conf * 100:.1f}%.")

    return anomalies