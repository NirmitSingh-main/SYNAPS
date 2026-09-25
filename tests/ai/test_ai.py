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

from ai.features.learned_features import prepare_iq_features, tokenize_signal_features
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
    assert feats.shape == (3, 6), f"Expected shape (3, 6), got {feats.shape}"
    assert np.all(np.isfinite(feats))
    print("[PASS] test_learned_features")


def test_tokenization_features():
    iq = (np.random.randn(8000) + 1j * np.random.randn(8000)).astype(np.complex64)
    feats = prepare_iq_features(iq)
    tokens = tokenize_signal_features(feats, num_tokens=256)
    assert tokens.shape == (256, 10), f"Expected shape (256, 10), got {tokens.shape}"
    assert np.all(np.isfinite(tokens)), "Tokens contain NaN or Inf"
    print("[PASS] test_tokenization_features")


from ai.models.transformer import SignalTransformer, IQConvFrontEnd


def test_transformer_model():
    # Test tokenized feature input (backward compatibility)
    model_tok = SignalTransformer(input_features=10, num_classes=5, d_model=32, nhead=2, num_layers=1, use_conv_frontend=False)
    x_tok = torch.randn(2, 256, 10)
    logits_tok = model_tok(x_tok)
    assert logits_tok.shape == (2, 5), f"Expected (2, 5), got {logits_tok.shape}"

    # Test Conv1D front-end with 2 channels [I, Q]
    frontend_2ch = IQConvFrontEnd(in_channels=2, out_channels=64, num_tokens=256)
    h_2ch = frontend_2ch(torch.randn(2, 2, 9600))
    assert h_2ch.shape == (2, 256, 64), f"Expected (2, 256, 64), got {h_2ch.shape}"
    h_8000 = frontend_2ch(torch.randn(2, 2, 8000))
    assert h_8000.shape == (2, 256, 64), f"Expected (2, 256, 64), got {h_8000.shape}"

    # Test Conv1D front-end with 3 channels [I, Q, I^2+Q^2] (run_015)
    frontend_3ch = IQConvFrontEnd(in_channels=3, out_channels=64, num_tokens=256)
    h_3ch = frontend_3ch(torch.randn(2, 3, 9600))
    assert h_3ch.shape == (2, 256, 64), f"Expected (2, 256, 64), got {h_3ch.shape}"

    # Test Conv1D front-end with 5 channels [I, Q, mag, diff_cos, diff_sin] (run_017)
    frontend_5ch = IQConvFrontEnd(in_channels=5, out_channels=64, num_tokens=256)
    h_5ch = frontend_5ch(torch.randn(2, 5, 9600))
    assert h_5ch.shape == (2, 256, 64), f"Expected (2, 256, 64), got {h_5ch.shape}"

    # Test raw IQ input (2-channel) end-to-end
    model_raw2 = SignalTransformer(input_features=2, num_classes=5, d_model=64, nhead=4, num_layers=2)
    x_raw2 = torch.randn(2, 2, 9600)
    logits_raw2 = model_raw2(x_raw2)
    assert logits_raw2.shape == (2, 5), f"Expected (2, 5), got {logits_raw2.shape}"

    # Test raw IQ + power input (3-channel) end-to-end
    model_raw3 = SignalTransformer(input_features=3, num_classes=5, d_model=64, nhead=4, num_layers=2)
    x_raw3 = torch.randn(2, 3, 9600)
    logits_raw3 = model_raw3(x_raw3)
    assert logits_raw3.shape == (2, 5), f"Expected (2, 5), got {logits_raw3.shape}"

    # Test raw IQ + mag + diff_phase input (5-channel) end-to-end (run_017 architecture)
    model_raw5 = SignalTransformer(input_features=5, num_classes=5, d_model=64, nhead=4, num_layers=2)
    x_raw5 = torch.randn(2, 5, 9600)
    logits_raw5 = model_raw5(x_raw5)
    assert logits_raw5.shape == (2, 5), f"Expected (2, 5), got {logits_raw5.shape}"

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



from ai.models.multi_branch import MultiBranchSignalClassifier, ConstellationGNN


def test_multi_branch_model():
    # 1. Test ConstellationGNN
    gnn = ConstellationGNN(node_in_features=5, hidden_dim=32, out_dim=64, dropout=0.35)
    nodes = torch.randn(4, 64, 5)
    g_emb = gnn(nodes)
    assert g_emb.shape == (4, 64), f"Expected (4, 64), got {g_emb.shape}"
    assert torch.all(torch.isfinite(g_emb)), "GNN embedding contains NaN or Inf"

    # 2. Test MultiBranchSignalClassifier with GNN (3 branches, dropout=0.35)
    model_3branch = MultiBranchSignalClassifier(input_channels=5, num_classes=5, d_model=64, dropout=0.35, use_gnn=True)
    x = torch.randn(4, 5, 9600)
    logits3 = model_3branch(x)
    assert logits3.shape == (4, 5), f"Expected (4, 5), got {logits3.shape}"
    assert torch.all(torch.isfinite(logits3)), "MultiBranch logits contain NaN or Inf"

    # 3. Test MultiBranchSignalClassifier without GNN (2 branches: CNN + Transformer)
    model_2branch = MultiBranchSignalClassifier(input_channels=5, num_classes=5, d_model=64, dropout=0.35, use_gnn=False)
    logits2 = model_2branch(x)
    assert logits2.shape == (4, 5), f"Expected (4, 5), got {logits2.shape}"

    # 4. Backward pass test with label smoothing and gradient clipping
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    loss = criterion(logits3, torch.tensor([0, 1, 2, 3]))
    assert torch.isfinite(loss)
    loss.backward()

    # Gradient clipping test
    grad_norm = torch.nn.utils.clip_grad_norm_(model_3branch.parameters(), max_norm=1.0)
    assert torch.isfinite(grad_norm)

    print("[PASS] test_multi_branch_model")


def test_optimization_and_regularization():
    model = MultiBranchSignalClassifier(input_channels=5, num_classes=5, d_model=64, dropout=0.35, use_gnn=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    # Simulate 5 steps with plateau
    for epoch in range(5):
        x = torch.randn(2, 5, 9600)
        y = torch.tensor([0, 1])
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

def test_rms_normalization():
    from ai.training.train import load_signal_raw_iq
    import tempfile

    # Test 1: Normal complex signal
    t = np.linspace(0, 1, 1000)
    iq_clean = (np.cos(2 * np.pi * 50 * t) + 1j * np.sin(2 * np.pi * 50 * t)).astype(np.complex64) * 4.5
    
    with tempfile.NamedTemporaryFile(suffix=".iq", delete=False) as f:
        tmp_path = Path(f.name)
        # Interleaved float32 convention
        interleaved = np.empty(len(iq_clean) * 2, dtype=np.float32)
        interleaved[0::2] = iq_clean.real
        interleaved[1::2] = iq_clean.imag
        interleaved.tofile(tmp_path)

    try:
        item = {"iq_path": tmp_path}
        feats = load_signal_raw_iq(item, max_length=1000)
        assert feats.shape == (5, 1000), f"Expected (5, 1000), got {feats.shape}"
        assert np.all(np.isfinite(feats)), "Features contain NaN or Inf"
        
        # Check that RMS power is approximately 1.0
        i_ch = feats[0]
        q_ch = feats[1]
        rms_pwr = np.mean(i_ch**2 + q_ch**2)
        assert np.isclose(rms_pwr, 1.0, atol=1e-3), f"Expected RMS power ~1.0, got {rms_pwr}"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    # Test 2: Zero/silent signal edge case
    iq_zero = np.zeros(100, dtype=np.complex64)
    with tempfile.NamedTemporaryFile(suffix=".iq", delete=False) as f:
        tmp_zero = Path(f.name)
        interleaved_z = np.zeros(200, dtype=np.float32)
        interleaved_z.tofile(tmp_zero)

    try:
        item_zero = {"iq_path": tmp_zero}
        feats_zero = load_signal_raw_iq(item_zero, max_length=100)
        assert feats_zero.shape == (5, 100)
        assert np.all(np.isfinite(feats_zero)), "Zero signal produced NaN/Inf"
    finally:
        if tmp_zero.exists():
            tmp_zero.unlink()

    print("[PASS] test_rms_normalization")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AI UNIT TESTS")
    print("=" * 60)
    test_learned_features()
    test_tokenization_features()
    test_transformer_model()
    test_multi_branch_model()
    test_optimization_and_regularization()
    test_rms_normalization()
    test_cnn_model()
    test_ssl_mae_model()
    test_classification_and_confidence()
    test_metrics()
    test_representations()
    print("\nALL AI UNIT TESTS PASSED SUCCESSFULLY!")