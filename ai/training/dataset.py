from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ai.preprocessing.iq_loader import load_iq_file
from ai.features.learned_features import prepare_iq_features, tokenize_signal_features
from project_paths import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    NUM_CLASSES,
    get_class_iq_dir,
)


NUM_TOKENS = 256
SEQUENCE_LENGTH = NUM_TOKENS


class IQSignalDataset(Dataset):
    """
    Dataset for modulation classification.

    Loads IQ files and converts them into
    [I, Q, magnitude, phase, diff_phase_cos, diff_phase_sin] features tokenized into NUM_TOKENS windows.

    Supports 5 classes: BPSK, QPSK, FSK, QAM16, MIXED.
    """

    def __init__(self, root=None, num_tokens=NUM_TOKENS):
        self.num_tokens = num_tokens

        self.samples = []

        for class_name in CLASS_NAMES:
            if root is not None:
                class_dir = Path(root) / class_name
            else:
                class_dir = get_class_iq_dir(class_name)

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing dataset directory: {class_dir}"
                )

            for iq_file in sorted(class_dir.glob("*.iq")):
                self.samples.append(
                    (
                        iq_file,
                        CLASS_TO_INDEX[class_name],
                    )
                )

        if not self.samples:
            raise RuntimeError("No IQ files found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        iq_file, label = self.samples[index]

        iq = load_iq_file(iq_file)
        features = prepare_iq_features(iq)
        tokens = tokenize_signal_features(features, num_tokens=self.num_tokens)

        x = torch.tensor(tokens, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


if __name__ == "__main__":
    dataset = IQSignalDataset()

    print("==============================")
    print("IQ DATASET TEST")
    print("==============================")

    print("Total samples:", len(dataset))

    counts = {name: 0 for name in CLASS_NAMES}

    for _, label in dataset.samples:
        counts[CLASS_NAMES[label]] += 1

    for name, count in counts.items():
        print(f"{name}: {count}")

    x, y = dataset[0]

    print("\nFirst sample:")
    print("Feature shape:", x.shape)
    print("Label:", y.item())
    print("Class:", CLASS_NAMES[y.item()])
    print("Feature dtype:", x.dtype)