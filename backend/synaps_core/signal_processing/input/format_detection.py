from pathlib import Path
from typing import Dict


SUPPORTED_FORMATS = {
    ".wav": "WAV",
    ".iq": "IQ",
}


def detect_format(file_path: str) -> str:
    """
    Detect the signal format from the file extension.

    Supported formats:
    - .wav
    - .iq

    Returns
    -------
    str
        "WAV" or "IQ"

    Raises
    ------
    ValueError
        If the format is unsupported.
    """

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension not in SUPPORTED_FORMATS:
        supported = ", ".join(SUPPORTED_FORMATS.keys())

        raise ValueError(
            f"Unsupported signal format '{extension}'. "
            f"Supported formats are: {supported}"
        )

    return SUPPORTED_FORMATS[extension]


def get_file_extension(file_path: str) -> str:
    """
    Return the lowercase file extension.

    Example
    -------
    signal_001.IQ -> ".iq"
    """

    return Path(file_path).suffix.lower()


def is_supported_format(file_path: str) -> bool:
    """
    Check whether a file has a supported signal format.

    Returns
    -------
    bool
        True for WAV or IQ, otherwise False.
    """

    extension = get_file_extension(file_path)

    return extension in SUPPORTED_FORMATS


def get_format_info(file_path: str) -> Dict[str, str]:
    """
    Return basic information about the signal file format.

    Returns
    -------
    dict
        Contains:
        - filename
        - extension
        - format
    """

    path = Path(file_path)

    signal_format = detect_format(file_path)

    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "format": signal_format,
    }