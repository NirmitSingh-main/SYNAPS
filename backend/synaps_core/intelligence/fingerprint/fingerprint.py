"""
RF Emitter and signal physical fingerprinting module.
"""

from typing import Any, Dict
import hashlib
import numpy as np


def extract_signal_fingerprint(
    iq_samples: np.ndarray,
    dsp_metrics: Dict[str, Any],
    modulation: str,
) -> Dict[str, Any]:
    """
    Generate unique physical and statistical RF fingerprint.
    """
    samples = np.asarray(iq_samples, dtype=np.complex64)
    cfo = float(dsp_metrics.get("cfo_hz", dsp_metrics.get("cfo", {}).get("cfo_hz", 0.0)))
    snr = float(dsp_metrics.get("snr_db", dsp_metrics.get("snr", {}).get("snr_db", 0.0)))
    bw = float(dsp_metrics.get("bandwidth_hz", dsp_metrics.get("bandwidth", {}).get("bandwidth_hz", 0.0)))
    
    # Statistical moments
    mean_pwr = float(np.mean(np.abs(samples) ** 2)) if len(samples) > 0 else 0.0
    papr_db = float(10 * np.log10(np.max(np.abs(samples) ** 2) / (mean_pwr + 1e-12))) if mean_pwr > 0 else 0.0

    # Short cryptographic hash of physical signature
    sig_bytes = f"{modulation}:{cfo:.1f}:{snr:.1f}:{bw:.1f}:{papr_db:.2f}".encode("utf-8")
    fingerprint_hash = hashlib.sha256(sig_bytes).hexdigest()[:16].upper()

    return {
        "fingerprint_id": f"RF-FP-{fingerprint_hash}",
        "modulation": modulation,
        "carrier_offset_hz": cfo,
        "snr_db": snr,
        "bandwidth_hz": bw,
        "papr_db": papr_db,
        "average_power": mean_pwr,
    }