from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class SignalMetadata:
    """
    Metadata associated with one synthetic signal.

    Ground-truth fields describe how the signal was generated.
    Estimated fields can be filled later by the signal-processing
    and Artificial Intelligence pipelines.

    Units
    -----
    sample_rate              : samples/second (Hz)
    carrier_frequency        : Hz
    symbol_rate              : symbols/second
    amplitude                : relative amplitude
    phase_offset             : radians
    frequency_offset         : Hz
    signal_to_noise_ratio   : decibels (dB)
    bandwidth                : Hz
    """

    # ---------------------------------------------------------
    # FILE INFORMATION
    # ---------------------------------------------------------

    signal_id: str
    file_name: str
    file_format: str

    # ---------------------------------------------------------
    # BASIC SIGNAL INFORMATION
    # ---------------------------------------------------------

    sample_rate: float
    number_of_samples: int

    # ---------------------------------------------------------
    # GROUND-TRUTH TRANSMITTER PARAMETERS
    # ---------------------------------------------------------

    modulation_type: Optional[str] = None

    carrier_frequency: Optional[float] = None
    symbol_rate: Optional[float] = None

    amplitude: Optional[float] = None
    phase_offset: Optional[float] = None
    frequency_offset: Optional[float] = None

    signal_to_noise_ratio: Optional[float] = None
    bandwidth: Optional[float] = None

    # Original information before modulation.
    # Used only as synthetic ground truth.
    original_message: Optional[str] = None

    # Encoding / channel-coding information
    encoding: Optional[str] = None
    fec_type: Optional[str] = None
    interleaving: Optional[str] = None

    # ---------------------------------------------------------
    # MACHINE LEARNING ESTIMATES
    # ---------------------------------------------------------

    predicted_modulation: Optional[str] = None
    predicted_modulation_confidence: Optional[float] = None

    estimated_carrier_frequency: Optional[float] = None
    estimated_symbol_rate: Optional[float] = None
    estimated_phase: Optional[float] = None
    estimated_frequency_offset: Optional[float] = None
    estimated_snr: Optional[float] = None
    estimated_bandwidth: Optional[float] = None

    # ---------------------------------------------------------
    # SIGNAL PROCESSING RESULTS
    # ---------------------------------------------------------

    synchronization_status: Optional[str] = None
    demodulation_status: Optional[str] = None
    decoding_status: Optional[str] = None

    recovered_bits: Optional[str] = None
    recovered_data: Optional[str] = None

    validation_status: Optional[str] = None
    bit_error_rate: Optional[float] = None

    # ---------------------------------------------------------
    # ADDITIONAL INFORMATION
    # ---------------------------------------------------------

    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------
    # CONVERSION FUNCTIONS
    # ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata into a Python dictionary.
        """
        return asdict(self)

    def to_json(self, output_path: str) -> None:
        """
        Save metadata as a JSON file.
        """

        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=4,
            )

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "SignalMetadata":
        """
        Create SignalMetadata from a dictionary.
        """

        return cls(**data)

    @classmethod
    def from_json(
        cls,
        input_path: str,
    ) -> "SignalMetadata":
        """
        Load metadata from a JSON file.
        """

        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {input_path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return cls.from_dict(data)


def create_metadata(
    signal_id: str,
    file_name: str,
    file_format: str,
    sample_rate: float,
    number_of_samples: int,
    **kwargs: Any,
) -> SignalMetadata:
    """
    Convenient function for creating metadata.

    Additional signal parameters can be supplied through kwargs.
    """

    return SignalMetadata(
        signal_id=signal_id,
        file_name=file_name,
        file_format=file_format,
        sample_rate=sample_rate,
        number_of_samples=number_of_samples,
        **kwargs,
    )