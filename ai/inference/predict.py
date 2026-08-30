import sys
from pathlib import Path

import numpy as np
import torch

from project_paths import (
    TRANSFORMER_CHECKPOINT,
    TRANSFORMER_RESULT_DIR,
    CLASS_NAMES,
)

from ai.preprocessing.iq_loader import load_iq_file
from ai.preprocessing.wav_loader import load_wav_file
from ai.features.learned_features import prepare_iq_features
from ai.models.transformer import SignalTransformer

from ai.classification.modulation import classify_modulation
from ai.classification.confidence import (
    calculate_confidence,
    confidence_percent,
)
from ai.classification.unknown_detection import (
    get_detection_status,
)


MODEL_PATH = TRANSFORMER_CHECKPOINT
RESULT_DIR = TRANSFORMER_RESULT_DIR
SEQUENCE_LENGTH = 1000


def load_model(model_path=None):

    target_path = Path(model_path) if model_path else MODEL_PATH
    if not target_path.exists():
        # Check alternate model location
        alt_path = Path("models/transformer/transformer.pth")
        if alt_path.exists():
            target_path = alt_path
        else:
            raise FileNotFoundError(
                f"Model not found: {target_path}"
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SignalTransformer()

    model.load_state_dict(
        torch.load(
            target_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, device


def prepare_features(iq):

    features = prepare_iq_features(iq)

    if len(features) >= SEQUENCE_LENGTH:
        features = features[:SEQUENCE_LENGTH]

    else:
        padded = np.zeros(
            (SEQUENCE_LENGTH, 4),
            dtype=np.float32
        )

        padded[:len(features)] = features
        features = padded

    return features.astype(np.float32)


def load_input(filepath):

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Input file not found: {filepath}"
        )

    extension = filepath.suffix.lower()

    if extension == ".iq":
        iq = load_iq_file(filepath)
        sample_rate = None
        input_format = "IQ"

    elif extension == ".wav":
        sample_rate, iq = load_wav_file(filepath)
        input_format = "WAV"

    else:
        raise ValueError(
            "Only .iq and .wav files are supported."
        )

    return iq, sample_rate, input_format


def save_result(
    filepath,
    input_format,
    sample_count,
    sample_rate,
    features,
    predicted_class,
    confidence,
    status,
    probabilities,
):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing = list(
        RESULT_DIR.glob("inference_*.txt")
    )

    numbers = []

    for file in existing:
        try:
            numbers.append(
                int(file.stem.split("_")[1])
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
        / f"inference_{next_number:03d}.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("SYNAPS TRANSFORMER INFERENCE\n")
        f.write("============================\n\n")

        f.write(f"Input file: {filepath}\n")
        f.write(f"Input format: {input_format}\n")
        f.write(f"Samples: {sample_count}\n")

        if sample_rate is not None:
            f.write(
                f"Sample rate: {sample_rate} Hz\n"
            )

        f.write(
            f"Feature shape: {features.shape}\n\n"
        )

        f.write("PREDICTION\n")
        f.write("----------\n")

        f.write(
            f"Predicted class: {predicted_class}\n"
        )

        f.write(
            f"Confidence: {confidence:.2f}%\n"
        )

        f.write(
            f"Detection status: {status}\n\n"
        )

        f.write("PROBABILITIES\n")
        f.write("-------------\n")

        for class_name in CLASS_NAMES:
            f.write(
                f"{class_name}: "
                f"{probabilities[class_name]:.2f}%\n"
            )

    return result_file


def predict(filepath):

    print("==============================")
    print("SYNAPS SIGNAL INFERENCE")
    print("==============================")

    print("Input:", filepath)

    model, device = load_model()

    print("Device:", device)

    iq, sample_rate, input_format = load_input(
        filepath
    )

    print("Samples:", len(iq))

    if sample_rate is not None:
        print("Sample rate:", sample_rate)

    features = prepare_features(iq)

    print("Feature shape:", features.shape)

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)

    probabilities, predicted_index, confidence = (
        calculate_confidence(output)
    )

    predicted_class = classify_modulation(
        predicted_index
    )

    confidence_pct = confidence_percent(
        confidence
    )

    status = get_detection_status(
        confidence
    )

    probability_dict = {}

    print("\n==============================")
    print("PREDICTION")
    print("==============================")

    print(
        "Predicted class:",
        predicted_class
    )

    print(
        f"Confidence: {confidence_pct:.2f}%"
    )

    print(
        "Detection status:",
        status
    )

    print("\nProbabilities:")

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            probabilities[index].item() * 100
        )

        probability_dict[class_name] = probability

        print(
            f"{class_name}: "
            f"{probability:.2f}%"
        )

    result_file = save_result(
        filepath=filepath,
        input_format=input_format,
        sample_count=len(iq),
        sample_rate=sample_rate,
        features=features,
        predicted_class=predicted_class,
        confidence=confidence_pct,
        status=status,
        probabilities=probability_dict,
    )

    print("\n==============================")
    print("RESULT SAVED")
    print("==============================")

    print(result_file)

    return {
        "file": str(filepath),
        "input_format": input_format,
        "predicted_class": predicted_class,
        "confidence": confidence_pct,
        "status": status,
        "probabilities": probability_dict,
        "result_file": str(result_file),
    }


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python -m ai.inference.predict "
            "<path_to_iq_or_wav>"
        )

        sys.exit(1)

    predict(sys.argv[1])