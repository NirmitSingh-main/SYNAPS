import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from ai.training.train import (
    find_dataset,
    build_dataset,
    stratified_split,
    CLASS_NAMES,
    SEQUENCE_LENGTH,
)
from ai.models.transformer import SignalTransformer


MODEL_PATH = Path("ai/models/transformer.pth")
RESULT_DIR = Path("result/transformer")


def main():

    print("==============================")
    print("TRANSFORMER EVALUATION")
    print("==============================")

    # --------------------------------------------------
    # Load existing dataset
    # --------------------------------------------------

    dataset = find_dataset()

    X, y = build_dataset(dataset)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = stratified_split(X, y)

    print("\nTest samples:", len(X_test))
    print("Test shape:", X_test.shape)

    # --------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Load trained Transformer
    # --------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = SignalTransformer()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

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
    # Confusion Matrix
    # --------------------------------------------------

    confusion_matrix = np.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=int
    )

    for true_label, predicted_label in zip(
        all_labels,
        all_predictions
    ):
        confusion_matrix[
            true_label,
            predicted_label
        ] += 1

    # --------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------

    per_class = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        total = confusion_matrix[
            index
        ].sum()

        correct = confusion_matrix[
            index,
            index
        ]

        accuracy = (
            correct / total * 100
            if total > 0
            else 0.0
        )

        per_class[class_name] = {
            "correct": int(correct),
            "total": int(total),
            "accuracy": accuracy,
        }

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

    print("\nPer-class accuracy:")

    for class_name in CLASS_NAMES:

        result = per_class[class_name]

        print(
            f"{class_name}: "
            f"{result['accuracy']:.2f}% "
            f"({result['correct']}/"
            f"{result['total']})"
        )

    # --------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted\n"
    )

    print(
        "             "
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
                f"{value:>8}"
                for value in confusion_matrix[index]
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
            f"Dataset samples: {len(dataset)}\n"
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
            "PER-CLASS ACCURACY\n"
        )

        f.write(
            "------------------\n"
        )

        for class_name in CLASS_NAMES:

            result = per_class[class_name]

            f.write(
                f"{class_name}: "
                f"{result['accuracy']:.2f}% "
                f"({result['correct']}/"
                f"{result['total']})\n"
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
            "             "
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
                    f"{value:>8}"
                    for value in confusion_matrix[index]
                )
                + "\n"
            )

    print(
        f"\nEvaluation saved to:\n"
        f"{result_file}"
    )


if __name__ == "__main__":
    main()