import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from ai.preprocessing.iq_loader import load_iq_file
from ai.features.learned_features import prepare_iq_features
from ai.models.transformer import SignalTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "BPSK",
    "QPSK",
    "FSK",
    "QAM16",
]

CLASS_TO_INDEX = {
    "BPSK": 0,
    "QPSK": 1,
    "FSK": 2,
    "QAM16": 3,
}

SEQUENCE_LENGTH = 1000

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

MODEL_PATH = Path("ai/models/transformer.pth")

# IMPORTANT:
# Your project currently uses "result", not "results".
RESULT_FOLDER = Path("result/transformer")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# RESULT FILE
# ============================================================

def get_next_result_file():
    """
    Automatically create:
        run_001.txt
        run_002.txt
        run_003.txt
        ...

    inside:
        result/transformer/
    """

    RESULT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_files = list(
        RESULT_FOLDER.glob("run_*.txt")
    )

    numbers = []

    for file in existing_files:

        try:
            number = int(
                file.stem.split("_")[1]
            )

            numbers.append(number)

        except (IndexError, ValueError):
            pass

    if numbers:
        next_number = max(numbers) + 1
    else:
        next_number = 1

    return (
        RESULT_FOLDER
        / f"run_{next_number:03d}.txt"
    )


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset():

    dataset = []

    for class_index, class_key in enumerate(
        CLASS_NAMES
    ):

        metadata_folder = (
            Path("data/metadata")
            / class_key
        )

        iq_folder = (
            Path("data/iq")
            / class_key
        )

        metadata_files = sorted(
            metadata_folder.glob("*.json")
        )

        print(
            f"{class_key}: "
            f"{len(metadata_files)} metadata files"
        )

        if not metadata_files:

            raise FileNotFoundError(
                f"No metadata files found in:\n"
                f"{metadata_folder}"
            )

        if not iq_folder.exists():

            raise FileNotFoundError(
                f"IQ folder not found:\n"
                f"{iq_folder}"
            )

        for metadata_path in metadata_files:

            with open(
                metadata_path,
                "r"
            ) as f:

                metadata = json.load(f)

            filename = metadata["filename"]

            if not filename.endswith(".iq"):
                filename += ".iq"

            # ------------------------------------------------
            # Exact filename
            # ------------------------------------------------

            iq_path = (
                iq_folder / filename
            )

            # ------------------------------------------------
            # FSK compatibility
            # ------------------------------------------------

            if (
                not iq_path.exists()
                and class_key == "FSK"
            ):

                alternative_name = filename.replace(
                    "_2fsk.iq",
                    "_fsk.iq"
                )

                iq_path = (
                    iq_folder
                    / alternative_name
                )

            # ------------------------------------------------
            # QAM16 compatibility
            # ------------------------------------------------

            if (
                not iq_path.exists()
                and class_key == "QAM16"
            ):

                alternative_name = filename.replace(
                    "_16qam.iq",
                    "_qam16.iq"
                )

                iq_path = (
                    iq_folder
                    / alternative_name
                )

            # ------------------------------------------------
            # Final validation
            # ------------------------------------------------

            if not iq_path.exists():

                raise FileNotFoundError(
                    f"\nIQ file not found.\n"
                    f"Metadata: {metadata_path}\n"
                    f"Expected: {filename}\n"
                    f"Folder: {iq_folder}"
                )

            dataset.append(
                {
                    "iq_path": iq_path,
                    "metadata_path": metadata_path,
                    "label": class_index,
                    "class_name": class_key,
                }
            )

    print(
        f"\nTotal dataset samples: "
        f"{len(dataset)}"
    )

    return dataset


# ============================================================
# LOAD ONE SIGNAL
# ============================================================

def load_signal(item):

    iq = load_iq_file(
        item["iq_path"]
    )

    features = prepare_iq_features(
        iq
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    if len(features) >= SEQUENCE_LENGTH:

        features = features[
            :SEQUENCE_LENGTH
        ]

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    else:

        padded = np.zeros(
            (
                SEQUENCE_LENGTH,
                4
            ),
            dtype=np.float32
        )

        padded[
            :len(features)
        ] = features

        features = padded

    return features.astype(
        np.float32
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(dataset):

    X = []
    y = []

    print("\nLoading IQ signals...")

    for index, item in enumerate(
        dataset
    ):

        features = load_signal(
            item
        )

        X.append(features)
        y.append(item["label"])

        if (
            index + 1
        ) % 100 == 0:

            print(
                f"Loaded "
                f"{index + 1}/"
                f"{len(dataset)}"
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    print("\nDataset loaded.")
    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    return X, y


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def stratified_split(X, y):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    train_indices = []
    validation_indices = []
    test_indices = []

    for class_index in range(
        len(CLASS_NAMES)
    ):

        class_indices = np.where(
            y == class_index
        )[0]

        rng.shuffle(
            class_indices
        )

        n = len(
            class_indices
        )

        train_end = int(
            n * TRAIN_RATIO
        )

        validation_end = (
            train_end
            + int(
                n * VALIDATION_RATIO
            )
        )

        train_indices.extend(
            class_indices[
                :train_end
            ]
        )

        validation_indices.extend(
            class_indices[
                train_end:
                validation_end
            ]
        )

        test_indices.extend(
            class_indices[
                validation_end:
            ]
        )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        validation_indices
    )

    rng.shuffle(
        test_indices
    )

    X_train = X[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    X_validation = X[
        validation_indices
    ]

    y_validation = y[
        validation_indices
    ]

    X_test = X[
        test_indices
    ]

    y_test = y[
        test_indices
    ]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate_model(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += (
                labels.size(0)
            )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

def calculate_per_class_accuracy(
    model,
    loader,
    device
):

    model.eval()

    correct_per_class = np.zeros(
        len(CLASS_NAMES),
        dtype=int
    )

    total_per_class = np.zeros(
        len(CLASS_NAMES),
        dtype=int
    )

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                features
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            for class_index in range(
                len(CLASS_NAMES)
            ):

                mask = (
                    labels
                    == class_index
                )

                total_per_class[
                    class_index
                ] += mask.sum().item()

                correct_per_class[
                    class_index
                ] += (
                    (
                        predictions[mask]
                        == labels[mask]
                    )
                    .sum()
                    .item()
                )

    results = {}

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        total = total_per_class[
            class_index
        ]

        correct = correct_per_class[
            class_index
        ]

        if total > 0:

            accuracy = (
                correct
                / total
                * 100
            )

        else:

            accuracy = 0.0

        results[class_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }

    return results


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    train_loader,
    validation_loader,
    device
):

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\n==============================")
    print("TRANSFORMER TRAINING")
    print("==============================")

    print(
        "Device:",
        device
    )

    print(
        "Epochs:",
        EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    best_validation_accuracy = 0.0

    history = []

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for features, labels in train_loader:

            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += (
                labels.size(0)
            )

        train_loss = (
            total_loss
            / total_samples
        )

        train_accuracy = (
            total_correct
            / total_samples
        )

        (
            validation_loss,
            validation_accuracy
        ) = evaluate_model(
            model,
            validation_loader,
            criterion,
            device
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": (
                train_accuracy * 100
            ),
            "validation_loss": (
                validation_loss
            ),
            "validation_accuracy": (
                validation_accuracy * 100
            ),
        }

        history.append(
            epoch_result
        )

        print(
            f"Epoch "
            f"{epoch + 1}/{EPOCHS} | "
            f"Loss: "
            f"{train_loss:.4f} | "
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}% | "
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

    print(
        "\nBest validation accuracy:",
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_PATH
    )

    return (
        history,
        best_validation_accuracy
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    result_file,
    dataset,
    X_train,
    X_validation,
    X_test,
    history,
    best_validation_accuracy,
    test_loss,
    test_accuracy,
    per_class_results,
    device
):

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SYNAPS TRANSFORMER TRAINING RESULT\n"
        )

        f.write(
            "====================================\n\n"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        f.write(
            "DATASET\n"
        )

        f.write(
            "-------\n"
        )

        f.write(
            f"Total samples: {len(dataset)}\n"
        )

        for class_name in CLASS_NAMES:

            count = sum(
                item["class_name"]
                == class_name
                for item in dataset
            )

            f.write(
                f"{class_name}: "
                f"{count}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        f.write(
            "CONFIGURATION\n"
        )

        f.write(
            "-------------\n"
        )

        f.write(
            f"Sequence length: "
            f"{SEQUENCE_LENGTH}\n"
        )

        f.write(
            "Features: "
            "I, Q, magnitude, phase\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Epochs: "
            f"{EPOCHS}\n"
        )

        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        f.write(
            f"Random seed: "
            f"{RANDOM_SEED}\n"
        )

        f.write(
            f"Device: "
            f"{device}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        f.write(
            "DATASET SPLIT\n"
        )

        f.write(
            "-------------\n"
        )

        f.write(
            f"Training samples: "
            f"{len(X_train)}\n"
        )

        f.write(
            f"Validation samples: "
            f"{len(X_validation)}\n"
        )

        f.write(
            f"Test samples: "
            f"{len(X_test)}\n"
        )

        f.write(
            f"Training shape: "
            f"{X_train.shape}\n"
        )

        f.write(
            f"Validation shape: "
            f"{X_validation.shape}\n"
        )

        f.write(
            f"Test shape: "
            f"{X_test.shape}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Epoch history
        # ----------------------------------------------------

        f.write(
            "TRAINING HISTORY\n"
        )

        f.write(
            "----------------\n"
        )

        for item in history:

            f.write(
                f"Epoch {item['epoch']}/{EPOCHS} | "
                f"Loss: "
                f"{item['train_loss']:.4f} | "
                f"Train Accuracy: "
                f"{item['train_accuracy']:.2f}% | "
                f"Validation Loss: "
                f"{item['validation_loss']:.4f} | "
                f"Validation Accuracy: "
                f"{item['validation_accuracy']:.2f}%\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Best validation
        # ----------------------------------------------------

        f.write(
            "BEST VALIDATION RESULT\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            f"Best Validation Accuracy: "
            f"{best_validation_accuracy * 100:.2f}%\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        f.write(
            "FINAL TEST\n"
        )

        f.write(
            "----------\n"
        )

        f.write(
            f"Test Loss: "
            f"{test_loss:.4f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Per-class
        # ----------------------------------------------------

        f.write(
            "PER-CLASS ACCURACY\n"
        )

        f.write(
            "------------------\n"
        )

        for class_name in CLASS_NAMES:

            result = per_class_results[
                class_name
            ]

            f.write(
                f"{class_name}: "
                f"{result['accuracy']:.2f}% "
                f"("
                f"{result['correct']}/"
                f"{result['total']}"
                f")\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        f.write(
            "MODEL\n"
        )

        f.write(
            "-----\n"
        )

        f.write(
            f"Best model: "
            f"{MODEL_PATH}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=============================="
    )

    print(
        "SYNAPS AI DATASET"
    )

    print(
        "=============================="
    )

    # --------------------------------------------------------
    # Find dataset
    # --------------------------------------------------------

    dataset = find_dataset()

    # --------------------------------------------------------
    # Load signals
    # --------------------------------------------------------

    X, y = build_dataset(
        dataset
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = stratified_split(
        X,
        y
    )

    print(
        "\n=============================="
    )

    print(
        "DATASET SPLIT"
    )

    print(
        "=============================="
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_validation)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Validation shape:",
        X_validation.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Tensor datasets
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.long
        )
    )

    validation_dataset = TensorDataset(
        torch.tensor(
            X_validation,
            dtype=torch.float32
        ),
        torch.tensor(
            y_validation,
            dtype=torch.long
        )
    )

    test_dataset = TensorDataset(
        torch.tensor(
            X_test,
            dtype=torch.float32
        ),
        torch.tensor(
            y_test,
            dtype=torch.long
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SignalTransformer()

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    (
        history,
        best_validation_accuracy
    ) = train_model(
        model,
        train_loader,
        validation_loader,
        device
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    if MODEL_PATH.exists():

        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device
            )
        )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    criterion = (
        torch.nn.CrossEntropyLoss()
    )

    (
        test_loss,
        test_accuracy
    ) = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        "\n=============================="
    )

    print(
        "FINAL TEST"
    )

    print(
        "=============================="
    )

    print(
        "Test Loss:",
        f"{test_loss:.4f}"
    )

    print(
        "Test Accuracy:",
        f"{test_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    per_class_results = (
        calculate_per_class_accuracy(
            model,
            test_loader,
            device
        )
    )

    print(
        "\nPer-class accuracy:"
    )

    for class_name in CLASS_NAMES:

        result = (
            per_class_results[
                class_name
            ]
        )

        print(
            f"{class_name}: "
            f"{result['accuracy']:.2f}% "
            f"("
            f"{result['correct']}/"
            f"{result['total']}"
            f")"
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result_file = (
        get_next_result_file()
    )

    save_results(
        result_file=result_file,
        dataset=dataset,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        history=history,
        best_validation_accuracy=(
            best_validation_accuracy
        ),
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        per_class_results=(
            per_class_results
        ),
        device=device
    )

    print(
        "\n=============================="
    )

    print(
        "RESULT SAVED"
    )

    print(
        "=============================="
    )

    print(
        result_file
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()