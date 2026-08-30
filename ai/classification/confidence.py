import torch


def calculate_confidence(logits):
    """
    Calculate class probabilities and confidence
    from model logits.

    Returns:
        probabilities
        predicted_index
        confidence
    """

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

    return (
        probabilities[0],
        int(predicted_index[0].item()),
        float(confidence[0].item())
    )


def confidence_percent(confidence):
    """
    Convert confidence from 0-1 to percentage.
    """

    return float(confidence) * 100.0