CLASS_NAMES = [
    "BPSK",
    "QPSK",
    "FSK",
    "QAM16",
]


def classify_modulation(predicted_index):
    """
    Convert the Transformer class index into
    the corresponding modulation name.
    """

    predicted_index = int(predicted_index)

    if predicted_index < 0 or predicted_index >= len(CLASS_NAMES):
        raise ValueError(
            f"Invalid class index: {predicted_index}"
        )

    return CLASS_NAMES[predicted_index]


def get_class_index(modulation):
    """
    Convert a modulation name into its class index.
    """

    modulation = str(modulation).upper()

    if modulation not in CLASS_NAMES:
        raise ValueError(
            f"Unknown modulation: {modulation}"
        )

    return CLASS_NAMES.index(modulation)