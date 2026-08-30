"""
Transformer model training pipeline for modulation classification in SYNAPS.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from project_paths import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    IQ_ROOT,
    METADATA_ROOT,
    PROJECT_ROOT,
    TRANSFORMER_CHECKPOINT,
    TRANSFORMER_RESULT_DIR,
    resolve_sample_paths,
)
from ai.preprocessing.iq_loader import load_iq_file
from ai.features.learned_features import prepare_iq_features
from ai.models.transformer import SignalTransformer
from ai.training.metrics import (
    calculate_accuracy,
    calculate_per_class_metrics,
)


# ============================================================
# CONFIGURATION DEFAULTS
# ============================================================

SEQUENCE_LENGTH = 1000
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

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
# DATASET DISCOVERY
# ============================================================

def find_dataset() -> List[Dict]:
    """
    Discover all dataset samples using canonical path resolution.
    """
    dataset = []

    for class_index, class_key in enumerate(CLASS_NAMES):
        metadata_folder = METADATA_ROOT / class_key
        iq_folder = IQ_ROOT / class_key

        if not metadata_folder.exists():
            raise FileNotFoundError(f"Metadata folder not found: {metadata_folder}")
        if not iq_folder.exists():
            raise FileNotFoundError(f"IQ folder not found: {iq_folder}")

        metadata_files = sorted(metadata_folder.glob("*.json"))
        print(f"{class_key}: {len(metadata_files)} metadata files found")

        for metadata_path in metadata_files:
            resolved = resolve_sample_paths(metadata_path)
            iq_path = resolved["iq_path"]

            if not iq_path or not iq_path.exists():
                raise FileNotFoundError(
                    f"IQ file corresponding to {metadata_path.name} not found in {iq_folder}"
                )

            dataset.append({
                "iq_path": iq_path,
                "metadata_path": metadata_path,
                "label": class_index,
                "class_name": class_key,
                "sample_id": resolved["sample_id"],
            })

    print(f"\nTotal dataset samples discovered: {len(dataset)}")
    return dataset


# ============================================================
# FEATURE EXTRACTION & DATASET BUILDING
# ============================================================

def load_signal_features(item: Dict, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    iq = load_iq_file(item["iq_path"])
    features = prepare_iq_features(iq)

    if len(features) >= sequence_length:
        features = features[:sequence_length]
    else:
        padded = np.zeros((sequence_length, 4), dtype=np.float32)
        padded[:len(features)] = features
        features = padded

    return features.astype(np.float32)


def build_dataset(
    dataset: List[Dict],
    sequence_length: int = SEQUENCE_LENGTH,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays from dataset list.
    """
    if max_samples and max_samples < len(dataset):
        dataset = dataset[:max_samples]

    X = []
    y = []

    print(f"\nExtracting features for {len(dataset)} signals...")
    for idx, item in enumerate(dataset):
        features = load_signal_features(item, sequence_length=sequence_length)
        X.append(features)
        y.append(item["label"])

        if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
            print(f"Loaded {idx + 1}/{len(dataset)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    print(f"Features loaded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_idx in range(len(CLASS_NAMES)):
        class_indices = np.where(y == class_idx)[0]
        rng.shuffle(class_indices)
        n = len(class_indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_indices.extend(class_indices[:train_end])
        val_indices.extend(class_indices[train_end:val_end])
        test_indices.extend(class_indices[val_end:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices], y[train_indices],
        X[val_indices], y[val_indices],
        X[test_indices], y[test_indices],
    )


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
) -> Tuple[List[Dict], float, int]:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    print(f"\nStarting training on {device} for {epochs} epochs (lr={learning_rate})...")

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

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

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
        f.write(f"Learning rate: {config_dict.get('learning_rate')}\n\n")
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
        f.write("PER-CLASS ACCURACY:\n")
        for cname, met in per_class_results.items():
            f.write(f"{cname}: {met['accuracy']:.2f}% ({met['correct']}/{met['total']})\n")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_training_pipeline(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_samples: Optional[int] = None,
    save_checkpoint: bool = True,
    smoke_test: bool = False,
) -> Dict:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = find_dataset()
    X, y = build_dataset(dataset, sequence_length=SEQUENCE_LENGTH, max_samples=max_samples)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(X, y)

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

    model = SignalTransformer(input_features=4, num_classes=len(CLASS_NAMES)).to(device)

    actual_epochs = 2 if smoke_test else epochs
    history, best_val_acc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        epochs=actual_epochs,
        learning_rate=learning_rate,
        save_path=MODEL_PATH if save_checkpoint else RESULT_FOLDER / "temp_model.pth",
    )

    # Evaluate best model
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, preds, targets = evaluate_model(model, test_loader, criterion, device)
    per_class_results = calculate_per_class_metrics(preds, targets, CLASS_NAMES)

    txt_file, json_file = get_next_result_files()
    config_dict = {
        "epochs": actual_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "sequence_length": SEQUENCE_LENGTH,
        "smoke_test": smoke_test,
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
        model_path=MODEL_PATH,
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
    run_training_pipeline()