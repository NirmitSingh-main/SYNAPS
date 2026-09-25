"""
Transformer model training pipeline for modulation classification in SYNAPS.

Uses the new dataset (v2) with 5 classes and CSV-based train/validation/test splits.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from project_paths import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    NUM_CLASSES,
    DATA_ROOT,
    DATASET_CSV,
    PROJECT_ROOT,
    TRANSFORMER_CHECKPOINT,
    TRANSFORMER_RESULT_DIR,
    get_class_iq_dir,
    get_class_metadata_dir,
    normalize_modulation_name,
)
from ai.preprocessing.iq_loader import load_iq_file
from ai.features.learned_features import (
    prepare_iq_features,
    tokenize_signal_features,
    prepare_synchronized_symbol_features,
)
from ai.models.transformer import SignalTransformer
from ai.models.multi_branch import MultiBranchSignalClassifier
from ai.training.metrics import (
    calculate_accuracy,
    calculate_per_class_metrics,
)


# ============================================================
# CONFIGURATION DEFAULTS
# ============================================================

MAX_SIGNAL_LENGTH = 9600   # Maximum raw IQ samples in dataset
MAX_SYMBOLS = 1024         # Maximum symbol-center samples for synchronized representation
NUM_TOKENS = 256           # Fixed Transformer sequence length (produced by Conv1D front-end)
SEQUENCE_LENGTH = NUM_TOKENS  # Alias for compatibility
INPUT_REPRESENTATION = "raw_iq"  # "raw_iq" | "tokens" | "synchronized_symbols"
INPUT_FEATURES = 5         # raw_iq channels: I, Q, mag, diff_cos, diff_sin
INPUT_FEATURES_SYNC = 8    # synchronized_symbols channels: I, Q, mag, diff_cos, diff_sin, power, m4, m8
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

RANDOM_SEED = 42

MODEL_PATH = TRANSFORMER_CHECKPOINT
RESULT_FOLDER = TRANSFORMER_RESULT_DIR


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# ============================================================
# RESULT FILE MANAGEMENT
# ============================================================

def get_next_result_files() -> Tuple[Path, Path]:
    """
    Returns (txt_path, json_path) with auto-incremented run number:
        result/transformer/run_001.txt
        result/transformer/run_001.json
    """
    RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
    existing_files = list(RESULT_FOLDER.glob("run_*.txt"))
    numbers = []

    for file in existing_files:
        try:
            number = int(file.stem.split("_")[1])
            numbers.append(number)
        except (IndexError, ValueError):
            pass

    next_number = max(numbers) + 1 if numbers else 1
    txt_file = RESULT_FOLDER / f"run_{next_number:03d}.txt"
    json_file = RESULT_FOLDER / f"run_{next_number:03d}.json"
    return txt_file, json_file


# ============================================================
# DATASET DISCOVERY (CSV-BASED)
# ============================================================

def load_dataset_csv() -> List[Dict]:
    """
    Load the master dataset.csv and resolve each sample to its IQ path.
    Returns a list of dicts with keys:
        iq_path, metadata_path, label, class_name, sample_id, split
    """
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {DATASET_CSV}\n"
            f"Expected at: {DATASET_CSV.resolve()}"
        )

    dataset = []

    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            modulation_raw = row["modulation"]
            split = row["split"]
            class_name = normalize_modulation_name(modulation_raw)

            if class_name not in CLASS_TO_INDEX:
                raise ValueError(
                    f"Unknown modulation '{modulation_raw}' "
                    f"(normalized: '{class_name}') in dataset.csv "
                    f"for sample {filename}"
                )

            label = CLASS_TO_INDEX[class_name]

            # Resolve IQ and metadata paths
            iq_dir = get_class_iq_dir(class_name)
            meta_dir = get_class_metadata_dir(class_name)

            # Filename in CSV matches stem exactly
            iq_path = iq_dir / f"{filename}.iq"
            meta_path = meta_dir / f"{filename}.json"

            if not iq_path.exists():
                raise FileNotFoundError(
                    f"IQ file not found: {iq_path}\n"
                    f"Referenced by dataset.csv row: {filename}"
                )

            dataset.append({
                "iq_path": iq_path,
                "metadata_path": meta_path,
                "label": label,
                "class_name": class_name,
                "sample_id": filename,
                "split": split,
            })

    return dataset


def split_dataset(dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split the dataset using the 'split' column from CSV.
    """
    train = [s for s in dataset if s["split"] == "train"]
    validation = [s for s in dataset if s["split"] == "validation"]
    test = [s for s in dataset if s["split"] == "test"]

    if not train:
        raise RuntimeError("No training samples found in dataset.csv")
    if not validation:
        raise RuntimeError("No validation samples found in dataset.csv")
    if not test:
        raise RuntimeError("No test samples found in dataset.csv")

    return train, validation, test


# ============================================================
# FEATURE EXTRACTION & DATASET BUILDING
# ============================================================

def load_signal_raw_iq(item: Dict, max_length: int = MAX_SIGNAL_LENGTH) -> np.ndarray:
    """
    Load IQ file -> extract (5, max_length) raw [I, Q, mag, diff_cos, diff_sin] float32 tensor.
    Channel 0 = I (RMS-normalized real)
    Channel 1 = Q (RMS-normalized imag)
    Channel 2 = magnitude (RMS-normalized signal envelope)
    Channel 3 = diff_cos (in-phase component of differential phasor)
    Channel 4 = diff_sin (quadrature component of differential phasor)
    """
    iq = load_iq_file(item["iq_path"])
    if len(iq) < max_length:
        pad_len = max_length - len(iq)
        iq = np.pad(iq, (0, pad_len), mode="constant")
    else:
        iq = iq[:max_length]

    i = np.real(iq).astype(np.float32)
    q = np.imag(iq).astype(np.float32)
    mag = np.abs(iq).astype(np.float32)

    # RMS-power normalization: scale = sqrt(E[|x|^2])
    scale = float(np.sqrt(np.mean(mag ** 2)))
    if scale > 1e-12 and np.isfinite(scale):
        i = i / scale
        q = q / scale
        mag = mag / scale

    diff_cos = np.ones(max_length, dtype=np.float32)
    diff_sin = np.zeros(max_length, dtype=np.float32)

    if len(iq) > 1:
        prod = iq[1:] * np.conj(iq[:-1])
        prod_mag = np.abs(prod).astype(np.float32)
        valid = prod_mag > 1e-12
        unit_phasor = np.zeros(len(prod), dtype=np.complex64)
        unit_phasor[valid] = prod[valid] / prod_mag[valid]

        diff_cos[1:] = np.nan_to_num(np.real(unit_phasor), nan=0.0, posinf=0.0, neginf=0.0)
        diff_sin[1:] = np.nan_to_num(np.imag(unit_phasor), nan=0.0, posinf=0.0, neginf=0.0)

    return np.stack([i, q, mag, diff_cos, diff_sin], axis=0)


def load_signal_synchronized(item: Dict, max_symbols: int = MAX_SYMBOLS) -> np.ndarray:
    """
    Load IQ file -> blind synchronization -> symbol-center feature tensor.

    Returns shape (8, max_symbols): [I, Q, mag, diff_cos, diff_sin, power, m4, m8].
    All synchronization is fully blind (no metadata leakage).
    """
    iq = load_iq_file(item["iq_path"])
    tensor = prepare_synchronized_symbol_features(
        iq, max_symbols=max_symbols, num_channels=8
    )
    return tensor


def load_signal_features(item: Dict, num_tokens: int = NUM_TOKENS) -> np.ndarray:
    """
    Load IQ file -> extract handcrafted features -> tokenize into
    fixed-length sequence via window averaging (backward compatibility).
    """
    iq = load_iq_file(item["iq_path"])
    features = prepare_iq_features(iq)
    tokens = tokenize_signal_features(features, num_tokens=num_tokens)
    return tokens


def build_split_arrays(
    samples: List[Dict],
    representation: str = INPUT_REPRESENTATION,
    num_tokens: int = NUM_TOKENS,
    max_length: int = MAX_SIGNAL_LENGTH,
    max_symbols: int = MAX_SYMBOLS,
    split_name: str = "dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from a list of sample dicts.
    Supports:
        representation="raw_iq": X shape (N, 5, max_length)  [I, Q, mag, diff_cos, diff_sin]
        representation="synchronized_symbols": X shape (N, 8, max_symbols)
        representation="tokens": X shape (N, num_tokens, num_features)
    """
    X = []
    y = []
    errors = 0

    print(f"\nLoading {representation} inputs for {len(samples)} {split_name} signals...")
    for idx, item in enumerate(samples):
        try:
            if representation == "raw_iq":
                feats = load_signal_raw_iq(item, max_length=max_length)
            elif representation == "synchronized_symbols":
                feats = load_signal_synchronized(item, max_symbols=max_symbols)
            else:
                feats = load_signal_features(item, num_tokens=num_tokens)
        except Exception as e:
            # Synchronized pipeline may fail on edge cases; log and use zeros
            if representation == "synchronized_symbols":
                errors += 1
                print(f"  [WARN] Sync failed for {item['sample_id']}: {e}")
                feats = np.zeros((INPUT_FEATURES_SYNC, max_symbols), dtype=np.float32)
            else:
                raise
        X.append(feats)
        y.append(item["label"])

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            print(f"  Loaded {idx + 1}/{len(samples)}")

    if errors > 0:
        print(f"  [WARN] {errors}/{len(samples)} sync failures (zero-filled)")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    print(f"  {split_name} data: X shape={X.shape}, y shape={y.shape}")
    return X, y


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(dataset: List[Dict]) -> None:
    """
    Run sanity checks on the dataset before training.
    Verifies: file existence, IQ length, NaN/Inf, label validity, class counts.
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    class_counts = {name: 0 for name in CLASS_NAMES}
    errors = []

    for i, item in enumerate(dataset):
        # Check IQ file exists
        if not item["iq_path"].exists():
            errors.append(f"IQ file missing: {item['iq_path']}")
            continue

        # Check metadata exists
        if item["metadata_path"] and not item["metadata_path"].exists():
            errors.append(f"Metadata missing: {item['metadata_path']}")

        # Check label is valid
        if item["label"] < 0 or item["label"] >= NUM_CLASSES:
            errors.append(f"Invalid label {item['label']} for {item['sample_id']}")

        class_counts[item["class_name"]] += 1

    # Print class distribution
    print("\nClass distribution:")
    for name in CLASS_NAMES:
        print(f"  {name}: {class_counts[name]}")
    print(f"  Total: {sum(class_counts.values())}")

    # Spot-check first sample from each class for NaN/Inf
    checked_classes = set()
    for item in dataset:
        if item["class_name"] not in checked_classes:
            try:
                iq = load_iq_file(item["iq_path"])
                if np.any(np.isnan(iq.view(np.float32))) or np.any(np.isinf(iq.view(np.float32))):
                    errors.append(f"NaN/Inf detected in {item['iq_path']}")
                checked_classes.add(item["class_name"])
            except Exception as e:
                errors.append(f"Error loading {item['iq_path']}: {e}")

    if errors:
        print(f"\n[WARNING] {len(errors)} validation errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        raise RuntimeError(f"Dataset validation failed with {len(errors)} errors")

    print("\n[OK] Dataset validation passed")


# ============================================================
# SMOKE TEST
# ============================================================

def run_smoke_test(
    model: torch.nn.Module,
    device: torch.device,
    representation: str = INPUT_REPRESENTATION,
) -> None:
    """
    Verify model forward/backward pass with one sample per class.
    """
    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # Load one sample per class
    dataset_all = load_dataset_csv()
    class_samples = {}
    for item in dataset_all:
        if item["class_name"] not in class_samples:
            class_samples[item["class_name"]] = item

    if len(class_samples) != NUM_CLASSES:
        missing = set(CLASS_NAMES) - set(class_samples.keys())
        raise RuntimeError(f"Smoke test: missing classes: {missing}")

    print(f"Loaded 1 sample from each of {NUM_CLASSES} classes")

    # Build input tensors
    X_list = []
    y_list = []
    for class_name in CLASS_NAMES:
        item = class_samples[class_name]
        if representation == "raw_iq":
            feats = load_signal_raw_iq(item)
        elif representation == "synchronized_symbols":
            feats = load_signal_synchronized(item)
        else:
            feats = load_signal_features(item)
        X_list.append(feats)
        y_list.append(item["label"])

    X = torch.tensor(np.array(X_list), dtype=torch.float32).to(device)
    y = torch.tensor(y_list, dtype=torch.long).to(device)

    # Forward pass
    model.train()
    outputs = model(X)

    print(f"Input shape:  {X.shape}")
    print(f"Output shape: {outputs.shape}")
    assert outputs.shape == (NUM_CLASSES, NUM_CLASSES), \
        f"Expected output shape ({NUM_CLASSES}, {NUM_CLASSES}), got {outputs.shape}"

    # Check softmax sums
    probs = torch.softmax(outputs, dim=1)
    sums = probs.sum(dim=1)
    print(f"Softmax sums: {sums.detach().cpu().numpy()}")
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        "Softmax probabilities do not sum to ~1.0"

    # Backward pass
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, y)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Backward pass: OK (gradients computed)")

    # Verify gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No non-zero gradients found after backward pass"

    model.zero_grad()

    print("\n[OK] Smoke test passed — model forward/backward verified")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = calculate_accuracy(all_preds, all_targets)

    return avg_loss, accuracy, all_preds, all_targets


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    save_path: Path = MODEL_PATH,
    class_weights: Optional[List[float]] = None,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.05,
    early_stopping_patience: int = 8,
) -> Tuple[List[Dict], float, int]:
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
        print(f"  Class weights: {dict(zip(CLASS_NAMES, class_weights))}")
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"  Class weights: None (uniform, label_smoothing={label_smoothing})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    print(f"\nStarting training on {device} for {epochs} epochs (lr={learning_rate}, weight_decay={weight_decay})...")
    print(f"  Early stopping patience: {early_stopping_patience} epochs (monitoring val_loss)")
    print(f"  ReduceLROnPlateau: factor=0.5, patience=4, min_lr=1e-6")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        val_loss, val_acc, _, _ = evaluate_model(
            model, validation_loader, criterion, device
        )

        # Step scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc * 100.0),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc * 100.0),
            "learning_rate": float(current_lr),
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100.0:6.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100.0:6.2f}% | LR: {current_lr:.1e}"
        )

        # Track best model by lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_architecture": model.__class__.__name__,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "class_to_index": CLASS_TO_INDEX,
                "input_features": getattr(model, "input_features", getattr(model, "input_channels", INPUT_FEATURES)),
                "use_conv_frontend": getattr(model, "use_conv_frontend", True),
                "d_model": model.d_model,
                "sequence_length": SEQUENCE_LENGTH,
                "epoch": epoch + 1,
                "validation_accuracy": float(val_acc * 100.0),
                "validation_loss": float(val_loss),
            }
            torch.save(checkpoint, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n[EARLY STOPPING] Validation loss did not improve for {early_stopping_patience} consecutive epochs.")
                print(f"Stopping early at epoch {epoch + 1}. Restoring best checkpoint from epoch {best_epoch}.")
                break

    # Restore best checkpoint
    if save_path.exists():
        best_cp = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(best_cp, dict) and "model_state_dict" in best_cp:
            model.load_state_dict(best_cp["model_state_dict"])
        else:
            model.load_state_dict(best_cp)
        print(f"\n[OK] Restored best checkpoint from epoch {best_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc * 100.0:.2f}%)")

    return history, best_val_acc, best_epoch


# ============================================================
# SAVE COMPLETE RESULTS
# ============================================================

def save_training_results(
    txt_file: Path,
    json_file: Path,
    config_dict: Dict,
    history: List[Dict],
    best_val_accuracy: float,
    best_epoch: int,
    test_loss: float,
    test_accuracy: float,
    per_class_results: Dict,
    model_path: Path,
):
    structured_result = {
        "model_architecture": "SignalTransformer",
        "model_path": str(model_path),
        "configuration": config_dict,
        "class_mapping": CLASS_TO_INDEX,
        "num_classes": NUM_CLASSES,
        "best_epoch": best_epoch,
        "best_validation_accuracy": float(best_val_accuracy * 100.0),
        "final_test_loss": float(test_loss),
        "final_test_accuracy": float(test_accuracy * 100.0),
        "per_class_metrics": per_class_results,
        "training_history": history,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(structured_result, f, indent=2)

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("SYNAPS TRANSFORMER TRAINING RESULT\n")
        f.write("====================================\n\n")
        f.write(f"Model checkpoint: {model_path}\n")
        f.write(f"Total epochs: {config_dict.get('epochs')}\n")
        f.write(f"Batch size: {config_dict.get('batch_size')}\n")
        f.write(f"Learning rate: {config_dict.get('learning_rate')}\n")
        f.write(f"Sequence length: {config_dict.get('sequence_length')}\n")
        f.write(f"Num classes: {NUM_CLASSES}\n")
        f.write(f"Classes: {CLASS_NAMES}\n\n")
        f.write("TRAINING HISTORY:\n")
        for rec in history:
            f.write(
                f"Epoch {rec['epoch']:02d} | "
                f"Train Loss: {rec['train_loss']:.4f} | Train Acc: {rec['train_accuracy']:.2f}% | "
                f"Val Loss: {rec['validation_loss']:.4f} | Val Acc: {rec['validation_accuracy']:.2f}%\n"
            )
        f.write(f"\nBest Epoch: {best_epoch} (Val Acc: {best_val_accuracy * 100.0:.2f}%)\n")
        f.write(f"Final Test Loss: {test_loss:.4f}\n")
        f.write(f"Final Test Accuracy: {test_accuracy * 100.0:.2f}%\n\n")
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':>8}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>8}  {'F1':>8}  {'Count':>6}\n")
        f.write("-" * 60 + "\n")
        for cname in CLASS_NAMES:
            if cname in per_class_results:
                met = per_class_results[cname]
                f.write(
                    f"{cname:>8}  {met['accuracy']:7.2f}%  {met['precision']:8.2f}%  "
                    f"{met['recall']:7.2f}%  {met['f1_score']:7.2f}%  {met['total']:>5d}\n"
                )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_training_pipeline(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    representation: str = INPUT_REPRESENTATION,
    max_samples: Optional[int] = None,
    save_checkpoint: bool = True,
    smoke_test: bool = False,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Dict:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_checkpoint = Path(checkpoint_path) if checkpoint_path else MODEL_PATH

    # --------------------------------------------------------
    # 1. Load and validate dataset from CSV
    # --------------------------------------------------------
    print("=" * 60)
    print("SYNAPS TRANSFORMER TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Input representation: {representation}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Classes: {CLASS_NAMES}")

    dataset_all = load_dataset_csv()
    validate_dataset(dataset_all)

    # --------------------------------------------------------
    # 2. Split using CSV split column
    # --------------------------------------------------------
    train_samples, val_samples, test_samples = split_dataset(dataset_all)

    print(f"\nSplit distribution:")
    print(f"  Train:      {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"  Test:       {len(test_samples)}")

    # Per-class split distribution
    print("\nPer-class split:")
    for class_name in CLASS_NAMES:
        n_train = sum(1 for s in train_samples if s["class_name"] == class_name)
        n_val = sum(1 for s in val_samples if s["class_name"] == class_name)
        n_test = sum(1 for s in test_samples if s["class_name"] == class_name)
        print(f"  {class_name:>6}: train={n_train:4d}  val={n_val:3d}  test={n_test:3d}")

    # --------------------------------------------------------
    # 3. Limit samples if requested
    # --------------------------------------------------------
    if max_samples:
        train_samples = train_samples[:max_samples]
        val_samples = val_samples[:max(max_samples // 5, 1)]
        test_samples = test_samples[:max(max_samples // 5, 1)]
        print(f"\n[MAX_SAMPLES] Limited to: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    # --------------------------------------------------------
    # 4. Build feature arrays
    # --------------------------------------------------------
    X_train, y_train = build_split_arrays(train_samples, representation=representation, num_tokens=NUM_TOKENS, split_name="train")
    X_val, y_val = build_split_arrays(val_samples, representation=representation, num_tokens=NUM_TOKENS, split_name="validation")
    X_test, y_test = build_split_arrays(test_samples, representation=representation, num_tokens=NUM_TOKENS, split_name="test")

    # --------------------------------------------------------
    # 5. Create DataLoaders
    # --------------------------------------------------------
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # 6. Create model
    # --------------------------------------------------------
    if representation == "raw_iq" or representation == "synchronized_symbols":
        if representation == "synchronized_symbols":
            input_dim = INPUT_FEATURES_SYNC  # 8: [I, Q, mag, diff_cos, diff_sin, power, m4, m8]
            channel_desc = "I, Q, mag, diff_cos, diff_sin, power, m4, m8"
        else:
            input_dim = INPUT_FEATURES  # 5: [I, Q, mag, diff_cos, diff_sin]
            channel_desc = "I, Q, mag, diff_cos, diff_sin"
        model = MultiBranchSignalClassifier(
            input_channels=input_dim,
            num_classes=NUM_CLASSES,
            d_model=64,
            nhead=4,
            transformer_layers=2,
            dropout=0.35,
            num_tokens=NUM_TOKENS,
            num_gnn_nodes=64,
            use_gnn=True,
        ).to(device)
        arch_name = f"MultiBranchSignalClassifier (CNN + Transformer + Constellation GNN, {input_dim}ch)"
    else:
        input_dim = X_train.shape[-1]
        channel_desc = f"{input_dim} token features"
        model = SignalTransformer(input_features=input_dim, num_classes=NUM_CLASSES).to(device)
        arch_name = "SignalTransformer (Token Linear Projection)"

    print(f"\nModel: {arch_name}")
    print(f"  Input features: {input_dim} (channels: {channel_desc})")
    print(f"  Num classes: {NUM_CLASSES}")
    print(f"  Class weights: None (uniform — standard CrossEntropyLoss)")
    print(f"  Model output shape: [batch_size, {NUM_CLASSES}]")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------------
    # 7. Smoke test
    # --------------------------------------------------------
    run_smoke_test(model, device, representation=representation)

    # --------------------------------------------------------
    # 8. Train
    # --------------------------------------------------------
    actual_epochs = 2 if smoke_test else epochs
    history, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        epochs=actual_epochs,
        learning_rate=learning_rate,
        save_path=target_checkpoint if save_checkpoint else RESULT_FOLDER / "temp_model.pth",
        class_weights=None,  # run_016: standard unweighted CrossEntropyLoss
    )

    # --------------------------------------------------------
    # 9. Evaluate best model on test set
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    # Load best checkpoint
    save_path = target_checkpoint if save_checkpoint else RESULT_FOLDER / "temp_model.pth"
    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded best checkpoint from epoch {best_epoch}")

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, targets = evaluate_model(model, test_loader, criterion, device)
    per_class_results = calculate_per_class_metrics(preds, targets, CLASS_NAMES)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100.0:.2f}%")
    print(f"\nPer-class metrics:")
    print(f"{'Class':>8}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>8}  {'F1':>8}  {'Count':>6}")
    print("-" * 60)
    for cname in CLASS_NAMES:
        if cname in per_class_results:
            m = per_class_results[cname]
            print(
                f"{cname:>8}  {m['accuracy']:7.2f}%  {m['precision']:8.2f}%  "
                f"{m['recall']:7.2f}%  {m['f1_score']:7.2f}%  {m['total']:>5d}"
            )

    # Confusion matrix
    from ai.training.metrics import calculate_confusion_matrix
    cm = calculate_confusion_matrix(preds, targets, num_classes=NUM_CLASSES)
    print(f"\nConfusion Matrix (rows=actual, columns=predicted):")
    header = "         " + " ".join(f"{name:>8}" for name in CLASS_NAMES)
    print(header)
    for i, cname in enumerate(CLASS_NAMES):
        row = f"{cname:>8} " + " ".join(f"{cm[i, j]:>8}" for j in range(NUM_CLASSES))
        print(row)

    # --------------------------------------------------------
    # 10. Save results
    # --------------------------------------------------------
    txt_file, json_file = get_next_result_files()
    config_dict = {
        "epochs": actual_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "input_representation": representation,
        "sequence_length": SEQUENCE_LENGTH,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "smoke_test": smoke_test,
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "test_samples": len(test_samples),
    }

    save_training_results(
        txt_file=txt_file,
        json_file=json_file,
        config_dict=config_dict,
        history=history,
        best_val_accuracy=best_val_acc,
        best_epoch=best_epoch,
        test_loss=test_loss,
        test_accuracy=test_acc,
        per_class_results=per_class_results,
        model_path=target_checkpoint,
    )

    print(f"\n[SUCCESS] Training results saved to:\n  {txt_file}\n  {json_file}")
    return {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "history": history,
        "result_txt": str(txt_file),
        "result_json": str(json_file),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SYNAPS Training Pipeline")
    parser.add_argument(
        "--representation",
        type=str,
        default=INPUT_REPRESENTATION,
        choices=["raw_iq", "tokens", "synchronized_symbols"],
        help="Input representation mode",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    args = parser.parse_args()

    run_training_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        representation=args.representation,
        checkpoint_path=args.checkpoint,
    )