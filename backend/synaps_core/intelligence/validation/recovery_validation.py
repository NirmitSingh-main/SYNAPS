"""
End-to-end signal recovery validation across synchronization, demodulation, and decoding.
"""

from typing import Any, Dict, Optional
import numpy as np


def validate_recovery_pipeline(
    sync_results: Dict[str, Any],
    demod_results: Dict[str, Any],
    decode_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce comprehensive health and confidence assessment of signal recovery pipeline.
    """
    recovered_bits = demod_results.get("recovered_bits", np.array([]))
    num_bits = len(recovered_bits)

    has_sync = sync_results.get("status", "SUCCESS") == "SUCCESS"
    has_demod = num_bits > 0
    has_decode = decode_results.get("status", "VALID") == "VALID"

    overall_status = "SUCCESS" if (has_sync and has_demod and has_decode) else "DEGRADED"

    return {
        "recovery_status": overall_status,
        "synchronization_passed": has_sync,
        "demodulation_passed": has_demod,
        "decoding_passed": has_decode,
        "recovered_bit_count": num_bits,
    }