DEFAULT_THRESHOLD = 0.70


def is_unknown(confidence, threshold=DEFAULT_THRESHOLD):
    """
    Determine whether a prediction should be treated
    as uncertain/unknown.

    confidence:
        Value between 0 and 1.

    threshold:
        Minimum confidence required for a known class.
    """

    confidence = float(confidence)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    return confidence < threshold


def get_detection_status(
    confidence,
    threshold=DEFAULT_THRESHOLD
):
    """
    Return a human-readable detection status.
    """

    if is_unknown(confidence, threshold):
        return "UNKNOWN"

    return "KNOWN"