"""
Evaluation metrics for modulation classification and AI training.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch


def calculate_accuracy(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Calculate classification accuracy in range [0.0, 1.0].
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if predictions.size == 0:
        return 0.0

    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    return float(np.mean(predictions == targets))


def calculate_confusion_matrix(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
) -> np.ndarray:
    """
    Calculate confusion matrix of shape (num_classes, num_classes).
    Rows = True class, Columns = Predicted class.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = np.argmax(predictions, axis=1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1

    return cm


def calculate_per_class_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Precision, Recall, F1-Score, and Accuracy per class.
    """
    num_classes = len(class_names)
    cm = calculate_confusion_matrix(predictions, targets, num_classes=num_classes)

    results = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        total = np.sum(cm[i, :])

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc = float(tp / total) if total > 0 else 0.0

        results[name] = {
            "accuracy": acc * 100.0,
            "precision": precision * 100.0,
            "recall": recall * 100.0,
            "f1_score": f1 * 100.0,
            "correct": int(tp),
            "total": int(total),
        }

    return results