"""
Decoded payload validation (ASCII, UTF-8, binary entropy, bitstream structure).
"""

from typing import Any, Dict, Union
import numpy as np


def validate_decoded_payload(
    decoded_data: Union[str, bytes, bytearray],
    recovered_bits: np.ndarray,
) -> Dict[str, Any]:
    """
    Validate quality, entropy, and character validity of recovered data payload.
    """
    bits = np.asarray(recovered_bits, dtype=np.uint8)
    bit_count = int(len(bits))

    if bit_count == 0:
        return {
            "valid": False,
            "status": "EMPTY",
            "bit_count": 0,
            "printable_ratio": 0.0,
            "entropy": 0.0,
        }

    # Bit entropy
    p1 = np.mean(bits == 1)
    p0 = 1.0 - p1
    entropy = float(-(p0 * np.log2(p0 + 1e-12) + p1 * np.log2(p1 + 1e-12)))

    # Printable character ratio
    if isinstance(decoded_data, str):
        text = decoded_data
        printable = sum(c.isprintable() for c in text)
        printable_ratio = float(printable / len(text)) if len(text) > 0 else 0.0
    elif isinstance(decoded_data, (bytes, bytearray)):
        printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in decoded_data)
        printable_ratio = float(printable / len(decoded_data)) if len(decoded_data) > 0 else 0.0
    else:
        printable_ratio = 0.0

    return {
        "valid": True,
        "status": "VALID",
        "bit_count": bit_count,
        "entropy": entropy,
        "printable_ratio": printable_ratio,
        "ones_ratio": float(p1),
    }