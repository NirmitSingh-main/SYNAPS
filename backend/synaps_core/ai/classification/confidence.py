"""
Torch-Free Confidence & Probability Calculations for SYNAPS Backend.
"""

import numpy as np


def calculate_confidence(logits):
    """
    Calculate class probabilities and confidence from model logits or probability arrays.

    Returns:
        probabilities (np.ndarray): 1D array of class probabilities
        predicted_index (int): Index of max probability
        confidence (float): Max probability value in [0, 1]
    """
    arr = np.asarray(logits, dtype=np.float64)

    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    # Check if inputs are already normalized probabilities (sum to ~1.0 and all >= 0)
    row = arr[0]
    if np.all(row >= 0) and np.isclose(np.sum(row), 1.0, atol=1e-2):
        probabilities = row
    else:
        # Compute numerically stable softmax
        max_val = np.max(arr, axis=1, keepdims=True)
        exp_arr = np.exp(arr - max_val)
        probabilities = (exp_arr / np.sum(exp_arr, axis=1, keepdims=True))[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return probabilities, predicted_index, confidence


def confidence_percent(confidence):
    """
    Convert confidence from 0-1 float to percentage (0-100).
    """
    return float(confidence) * 100.0