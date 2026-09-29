"""
Multi-domain feature fusion combining DSP physical metrics and AI learned features.
"""

from typing import Any, Dict, Optional
import numpy as np


def fuse_dsp_and_ai_features(
    dsp_metrics: Dict[str, Any],
    ai_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fuse DSP statistical/spectral features and AI classification outputs into a coherent feature vector.

    Parameters
    ----------
    dsp_metrics : Dict[str, Any]
        Dictionary containing DSP outputs: 'snr', 'cfo', 'bandwidth', 'hoc', 'constellation', etc.
    ai_features : Dict[str, Any], optional
        Dictionary containing AI outputs: 'predicted_class', 'probabilities', 'confidence', etc.

    Returns
    -------
    fused : Dict[str, Any]
        Aggregated feature dictionary with numerical vector and metadata.
    """
    ai_features = ai_features or {}

    # Extract scalar DSP indicators
    snr_db = float(dsp_metrics.get("snr_db", dsp_metrics.get("snr", {}).get("snr_db", 0.0)))
    cfo_hz = float(dsp_metrics.get("cfo_hz", dsp_metrics.get("cfo", {}).get("cfo_hz", 0.0)))
    bw_hz = float(dsp_metrics.get("bandwidth_hz", dsp_metrics.get("bandwidth", {}).get("bandwidth_hz", 0.0)))
    symbol_rate = float(dsp_metrics.get("symbol_rate", dsp_metrics.get("timing", {}).get("symbol_rate", 0.0)))

    # Higher-Order Cumulants
    hoc_dict = dsp_metrics.get("hoc", {})
    c40 = float(np.abs(hoc_dict.get("C40", 0.0)))
    c42 = float(np.abs(hoc_dict.get("C42", 0.0)))
    c63 = float(np.abs(hoc_dict.get("C63", 0.0)))

    # AI probabilities
    probs = ai_features.get("probabilities", {})
    p_bpsk = float(probs.get("BPSK", 0.0)) / 100.0 if probs.get("BPSK", 0.0) > 1.0 else float(probs.get("BPSK", 0.0))
    p_qpsk = float(probs.get("QPSK", 0.0)) / 100.0 if probs.get("QPSK", 0.0) > 1.0 else float(probs.get("QPSK", 0.0))
    p_fsk = float(probs.get("FSK", 0.0)) / 100.0 if probs.get("FSK", 0.0) > 1.0 else float(probs.get("FSK", 0.0))
    p_qam16 = float(probs.get("QAM16", 0.0)) / 100.0 if probs.get("QAM16", 0.0) > 1.0 else float(probs.get("QAM16", 0.0))

    feature_vector = np.array([
        snr_db,
        cfo_hz / 1e5,
        bw_hz / 1e6,
        symbol_rate / 1e5,
        c40,
        c42,
        c63,
        p_bpsk,
        p_qpsk,
        p_fsk,
        p_qam16,
    ], dtype=np.float32)

    return {
        "feature_vector": feature_vector,
        "dsp_summary": {
            "snr_db": snr_db,
            "cfo_hz": cfo_hz,
            "bandwidth_hz": bw_hz,
            "symbol_rate": symbol_rate,
            "hoc_c40": c40,
            "hoc_c42": c42,
            "hoc_c63": c63,
        },
        "ai_summary": {
            "predicted_class": ai_features.get("predicted_class"),
            "confidence": ai_features.get("confidence", 0.0),
            "probabilities": probs,
        },
    }