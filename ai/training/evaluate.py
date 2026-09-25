import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from ai.training.train import (
    load_dataset_csv,
    split_dataset,
    build_split_arrays,
    NUM_TOKENS,
)
from ai.models.transformer import SignalTransformer
from ai.training.metrics import (
    calculate_confusion_matrix,
    calculate_per_class_metrics,
)
from project_paths import (
    CLASS_NAMES,
    NUM_CLASSES,
    TRANSFORMER_CHECKPOINT,
    TRANSFORMER_RESULT_DIR,
)


MODEL_PATH = TRANSFORMER_CHECKPOINT
RESULT_DIR = TRANSFORMER_RESULT_DIR


def main():

    print("==============================")
    print("TRANSFORMER EVALUATION")
    print("==============================")

    # --------------------------------------------------
    # Load trained Transformer
    # --------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Load checkpoint (supports rich checkpoint, legacy state_dict, and arbitrary feature counts)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    from ai.models.multi_branch import MultiBranchSignalClassifier

    in_features = 5
    arch = "SignalTransformer"
    if isinstance(checkpoint, dict):
        if "input_features" in checkpoint:
            in_features = checkpoint["input_features"]
        if "model_architecture" in checkpoint:
            arch = checkpoint["model_architecture"]
        elif "model_state_dict" in checkpoint:
            if "fusion_head.0.weight" in checkpoint["model_state_dict"]:
                arch = "MultiBranchSignalClassifier"
            elif "input_projection.weight" in checkpoint["model_state_dict"]:
                in_features = checkpoint["model_state_dict"]["input_projection.weight"].shape[1]

    if arch == "MultiBranchSignalClassifier":
        model = MultiBranchSignalClassifier(
            input_channels=in_features,
            num_classes=NUM_CLASSES,
        )
    else:
        model = SignalTransformer(
            input_features=in_features,
            num_classes=NUM_CLASSES,
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded rich checkpoint ({arch}, epoch {checkpoint.get('epoch', '?')}, in_features={in_features})")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded legacy state_dict checkpoint ({arch}, in_features={in_features})")

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Load test dataset using CSV splits
    # --------------------------------------------------

    dataset_all = load_dataset_csv()
    train_samples, val_samples, test_samples = split_dataset(dataset_all)

    representation = "raw_iq" if in_features in (2, 3, 5) else "tokens"
    X_test, y_test = build_split_arrays(
        test_samples,
        representation=representation,
        num_tokens=NUM_TOKENS,
        split_name="test",
    )

    print("\nTest samples:", len(X_test))
    print("Test shape:", X_test.shape)

    # Slice features to match model input dimension if needed for handcrafted tokens
    if representation == "tokens" and X_test.shape[-1] > in_features:
        X_eval = X_test[:, :, :in_features]
    else:
        X_eval = X_test

    # --------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------

    test_dataset = TensorDataset(
        torch.tensor(X_eval, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for features, labels in test_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    test_loss = total_loss / total_samples
    test_accuracy = (
        total_correct / total_samples
    ) * 100

    # --------------------------------------------------
    # Confusion Matrix (5 classes)
    # --------------------------------------------------

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    confusion_matrix = calculate_confusion_matrix(
        all_predictions,
        all_labels,
        num_classes=NUM_CLASSES,
    )

    # --------------------------------------------------
    # Per-class metrics
    # --------------------------------------------------

    per_class = calculate_per_class_metrics(
        all_predictions,
        all_labels,
        CLASS_NAMES,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n==============================")
    print("EVALUATION RESULT")
    print("==============================")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    print(f"\nPer-class metrics:")
    print(f"{'Class':>8}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>8}  {'F1':>8}  {'Count':>6}")
    print("-" * 60)

    for class_name in CLASS_NAMES:
        result = per_class[class_name]
        print(
            f"{class_name:>8}  {result['accuracy']:7.2f}%  {result['precision']:8.2f}%  "
            f"{result['recall']:7.2f}%  {result['f1_score']:7.2f}%  {result['total']:>5d}"
        )

    # --------------------------------------------------
    # Print confusion matrix (5x5)
    # --------------------------------------------------

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print("Rows = Actual")
    print("Columns = Predicted\n")

    print(
        "         "
        + " ".join(
            f"{name:>8}"
            for name in CLASS_NAMES
        )
    )

    for index, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"{class_name:>8} "
            + " ".join(
                f"{confusion_matrix[index, j]:>8}"
                for j in range(NUM_CLASSES)
            )
        )

    # --------------------------------------------------
    # Save evaluation
    # --------------------------------------------------

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = list(
        RESULT_DIR.glob(
            "evaluation_*.txt"
        )
    )

    numbers = []

    for file in existing:

        try:
            numbers.append(
                int(
                    file.stem.split("_")[1]
                )
            )
        except (ValueError, IndexError):
            pass

    next_number = (
        max(numbers) + 1
        if numbers
        else 1
    )

    result_file = (
        RESULT_DIR
        / f"evaluation_{next_number:03d}.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SYNAPS TRANSFORMER EVALUATION\n"
        )

        f.write(
            "=============================\n\n"
        )

        f.write(
            f"Model: {MODEL_PATH}\n"
        )

        f.write(
            f"Num classes: {NUM_CLASSES}\n"
        )

        f.write(
            f"Classes: {CLASS_NAMES}\n"
        )

        f.write(
            f"Dataset samples: {len(dataset_all)}\n"
        )

        f.write(
            f"Test samples: {len(X_test)}\n"
        )

        f.write(
            f"Test shape: {X_test.shape}\n"
        )

        f.write(
            f"Device: {device}\n\n"
        )

        f.write(
            f"Test Loss: {test_loss:.4f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy:.2f}%\n\n"
        )

        f.write(
            "PER-CLASS METRICS\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            f"{'Class':>8}  {'Accuracy':>8}  {'Precision':>9}  {'Recall':>8}  {'F1':>8}  {'Count':>6}\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        for class_name in CLASS_NAMES:
            result = per_class[class_name]
            f.write(
                f"{class_name:>8}  {result['accuracy']:7.2f}%  {result['precision']:8.2f}%  "
                f"{result['recall']:7.2f}%  {result['f1_score']:7.2f}%  {result['total']:>5d}\n"
            )

        f.write("\n")

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            "----------------\n"
        )

        f.write(
            "Rows = Actual, "
            "Columns = Predicted\n\n"
        )

        f.write(
            "         "
            + " ".join(
                f"{name:>8}"
                for name in CLASS_NAMES
            )
            + "\n"
        )

        for index, class_name in enumerate(
            CLASS_NAMES
        ):
            f.write(
                f"{class_name:>8} "
                + " ".join(
                    f"{confusion_matrix[index, j]:>8}"
                    for j in range(NUM_CLASSES)
                )
                + "\n"
            )

    print(
        f"\nEvaluation saved to:\n"
        f"{result_file}"
    )


if __name__ == "__main__":
    main()