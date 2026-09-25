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

    Supports representations:
        - "tokens": [mean_I, mean_Q, mag, phase, cos, sin, ...] tokenized into NUM_TOKENS windows.
        - "raw_iq": (5, max_length) raw normalized continuous waveform.
        - "synchronized_symbols": (8, max_symbols) synchronized symbol-center representation.

    Supports 5 classes: BPSK, QPSK, FSK, QAM16, MIXED.
    """

    def __init__(
        self,
        root=None,
        num_tokens=NUM_TOKENS,
        representation: str = "tokens",
        max_length: int = 9600,
        max_symbols: int = 1024,
    ):
        self.num_tokens = num_tokens
        self.representation = representation
        self.max_length = max_length
        self.max_symbols = max_symbols

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

        if self.representation == "synchronized_symbols":
            from ai.features.learned_features import prepare_synchronized_symbol_features
            tensor = prepare_synchronized_symbol_features(iq, max_symbols=self.max_symbols, num_channels=8)
            x = torch.tensor(tensor, dtype=torch.float32)
        elif self.representation == "raw_iq":
            if len(iq) < self.max_length:
                iq_pad = np.pad(iq, (0, self.max_length - len(iq)), mode="constant")
            else:
                iq_pad = iq[:self.max_length]
            i_val = np.real(iq_pad).astype(np.float32)
            q_val = np.imag(iq_pad).astype(np.float32)
            mag_val = np.abs(iq_pad).astype(np.float32)
            scale = float(np.sqrt(np.mean(mag_val ** 2)))
            if scale > 1e-12:
                i_val /= scale
                q_val /= scale
                mag_val /= scale
            diff_cos = np.ones(self.max_length, dtype=np.float32)
            diff_sin = np.zeros(self.max_length, dtype=np.float32)
            if len(iq_pad) > 1:
                prod = iq_pad[1:] * np.conj(iq_pad[:-1])
                pmag = np.abs(prod).astype(np.float32)
                valid = pmag > 1e-12
                unit = np.zeros(len(prod), dtype=np.complex64)
                unit[valid] = prod[valid] / pmag[valid]
                diff_cos[1:] = np.nan_to_num(np.real(unit), nan=0.0, posinf=0.0, neginf=0.0)
                diff_sin[1:] = np.nan_to_num(np.imag(unit), nan=0.0, posinf=0.0, neginf=0.0)
            raw_arr = np.stack([i_val, q_val, mag_val, diff_cos, diff_sin], axis=0)
            x = torch.tensor(raw_arr, dtype=torch.float32)
        else:
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