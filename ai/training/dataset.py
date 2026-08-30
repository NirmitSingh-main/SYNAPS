from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ai.preprocessing.iq_loader import load_iq_file
from ai.features.learned_features import prepare_iq_features


CLASS_NAMES = ["BPSK", "QPSK", "FSK", "QAM16"]

CLASS_TO_INDEX = {
    "BPSK": 0,
    "QPSK": 1,
    "FSK": 2,
    "QAM16": 3,
}


class IQSignalDataset(Dataset):
    """
    Dataset for modulation classification.

    Loads IQ files and converts them into
    [I, Q, magnitude, phase] features.
    """

    def __init__(self, root="data/iq", sequence_length=1000):
        self.root = Path(root)
        self.sequence_length = sequence_length

        self.samples = []

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name

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

        # Fixed sequence length for Transformer
        if len(features) >= self.sequence_length:
            features = features[:self.sequence_length]
        else:
            padded = np.zeros(
                (self.sequence_length, features.shape[1]),
                dtype=np.float32,
            )

            padded[:len(features)] = features
            features = padded

        x = torch.tensor(features, dtype=torch.float32)
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
    