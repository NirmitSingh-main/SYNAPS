"""
Unit tests for AI models, representations, confidence scoring, and metrics.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.features.learned_features import prepare_iq_features
from ai.models.transformer import SignalTransformer
from ai.models.cnn import SignalCNN
from ai.models.ssl_mae import SignalMaskedAutoencoder
from ai.classification.modulation import classify_modulation, get_class_index
from ai.classification.confidence import calculate_confidence, confidence_percent
from ai.classification.unknown_detection import is_unknown, get_detection_status
from ai.training.metrics import calculate_accuracy, calculate_per_class_metrics, calculate_confusion_matrix
from ai.representations.raw_iq import extract_raw_iq_tensor
from ai.representations.spectrogram import extract_spectrogram_tensor
from ai.representations.constellation import extract_constellation_tensor


def test_learned_features():
    iq = np.array([1.0 + 2.0j, -0.5 + 0.5j, 0.0 + 1.0j], dtype=np.complex64)
    feats = prepare_iq_features(iq)
    assert feats.shape == (3, 4), f"Expected shape (3, 4), got {feats.shape}"
    assert np.all(np.isfinite(feats))
    print("[PASS] test_learned_features")


def test_transformer_model():
    model = SignalTransformer(input_features=4, num_classes=4, d_model=32, nhead=2, num_layers=1)
    x = torch.randn(2, 100, 4)
    logits = model(x)
    assert logits.shape == (2, 4), f"Expected (2, 4), got {logits.shape}"
    print("[PASS] test_transformer_model")


def test_cnn_model():
    model = SignalCNN(in_channels=4, num_classes=4, num_filters=16)
    x = torch.randn(2, 100, 4)
    logits = model(x)
    assert logits.shape == (2, 4), f"Expected (2, 4), got {logits.shape}"
    print("[PASS] test_cnn_model")


def test_ssl_mae_model():
    model = SignalMaskedAutoencoder(input_dim=4, embed_dim=32, encoder_layers=1, decoder_layers=1, nhead=2)
    x = torch.randn(2, 50, 4)
    recon, loss, encoded = model(x)
    assert recon.shape == x.shape
    assert loss.item() >= 0.0
    print("[PASS] test_ssl_mae_model")


def test_classification_and_confidence():
    logits = torch.tensor([[5.0, 1.0, 0.5, -2.0]])
    probs, idx, conf = calculate_confidence(logits)
    assert idx == 0
    assert conf > 0.80

    cname = classify_modulation(idx)
    assert cname == "BPSK"
    assert get_class_index("BPSK") == 0

    pct = confidence_percent(conf)
    assert pct > 80.0

    assert is_unknown(conf, threshold=0.70) is False
    assert is_unknown(0.40, threshold=0.70) is True
    assert get_detection_status(conf, threshold=0.70) == "KNOWN"
    assert get_detection_status(0.40, threshold=0.70) == "UNKNOWN"
    print("[PASS] test_classification_and_confidence")


def test_metrics():
    preds = np.array([0, 1, 2, 3, 0])
    targets = np.array([0, 1, 2, 3, 1])
    acc = calculate_accuracy(preds, targets)
    assert np.isclose(acc, 0.80)

    cm = calculate_confusion_matrix(preds, targets, num_classes=4)
    assert cm.shape == (4, 4)

    per_class = calculate_per_class_metrics(preds, targets, ["BPSK", "QPSK", "FSK", "QAM16"])
    assert "BPSK" in per_class
    assert per_class["BPSK"]["correct"] == 1
    print("[PASS] test_metrics")


def test_representations():
    iq = (np.random.randn(200) + 1j * np.random.randn(200)).astype(np.complex64)
    raw_tensor = extract_raw_iq_tensor(iq, target_length=100)
    assert raw_tensor.shape == (2, 100)

    spec_tensor = extract_spectrogram_tensor(iq, target_shape=(32, 32))
    assert spec_tensor.shape == (1, 32, 32)

    const_tensor = extract_constellation_tensor(iq, grid_size=32)
    assert const_tensor.shape == (1, 32, 32)
    print("[PASS] test_representations")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AI UNIT TESTS")
    print("=" * 60)
    test_learned_features()
    test_transformer_model()
    test_cnn_model()
    test_ssl_mae_model()
    test_classification_and_confidence()
    test_metrics()
    test_representations()
    print("\nALL AI UNIT TESTS PASSED SUCCESSFULLY!")