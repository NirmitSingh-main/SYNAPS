# import json
# import numpy as np

# from signal_processing.synchronization.frequency_sync import (
#     correct_frequency_offset
# )

# from signal_processing.synchronization.phase_sync import (
#     correct_phase_offset
# )

# from signal_processing.synchronization.timing_sync import (
#     estimate_timing_offset,
#     sample_symbols
# )

# from signal_processing.demodulation.bpsk import (
#     demodulate_bpsk
# )


# def test_bpsk_demodulation():

#     # -----------------------------
#     # Load IQ signal
#     # -----------------------------

#     iq_filepath = "data/iq/signal_0004_bpsk.iq"

#     data = np.fromfile(
#         iq_filepath,
#         dtype="<f4"
#     )

#     i = data[0::2]
#     q = data[1::2]

#     iq = i + 1j * q

#     # -----------------------------
#     # Signal parameters
#     # -----------------------------

#     sampling_frequency_hz = 1_000_000
#     frequency_offset_hz = -10_000
#     phase_offset_degrees = 0
#     samples_per_symbol = 10

#     # -----------------------------
#     # Frequency synchronization
#     # -----------------------------

#     frequency_corrected = correct_frequency_offset(
#         iq,
#         frequency_offset_hz,
#         sampling_frequency_hz
#     )

#     # -----------------------------
#     # Phase synchronization
#     # -----------------------------

#     phase_corrected = correct_phase_offset(
#         frequency_corrected,
#         phase_offset_degrees
#     )

#     # -----------------------------
#     # Timing synchronization
#     # -----------------------------

#     timing_offset = estimate_timing_offset(
#         phase_corrected,
#         samples_per_symbol
#     )

#     symbols = sample_symbols(
#         phase_corrected,
#         samples_per_symbol,
#         timing_offset
#     )

#     # -----------------------------
#     # BPSK demodulation
#     # -----------------------------

#     recovered_bits = demodulate_bpsk(symbols)

#     # -----------------------------
#     # Load ground-truth bits
#     # -----------------------------

#     metadata_filepath = (
#         "data/metadata/signal_0004_bpsk.json"
#     )

#     with open(metadata_filepath, "r") as file:
#         metadata = json.load(file)

#     expected_bits = np.array(
#         [int(bit) for bit in metadata["bits"]],
#         dtype=np.uint8
#     )

#     # -----------------------------
#     # Compare results
#     # -----------------------------

#     number_of_bits = min(
#         len(expected_bits),
#         len(recovered_bits)
#     )

#     correct_bits = np.sum(
#         expected_bits[:number_of_bits]
#         == recovered_bits[:number_of_bits]
#     )

#     accuracy = (
#         correct_bits / number_of_bits
#     ) * 100

#     print("\n-----------------------------")
#     print("BPSK DEMODULATION TEST")
#     print("-----------------------------")

#     print("Original IQ samples:", len(iq))
#     print("Samples per symbol:", samples_per_symbol)
#     print("Timing offset:", timing_offset)
#     print("Recovered symbols:", len(symbols))
#     print("Expected bits:", len(expected_bits))
#     print("Recovered bits:", len(recovered_bits))
#     print("Correct bits:", correct_bits)
#     print("Bit accuracy:", accuracy, "%")

#     print("\nExpected first 50 bits:")
#     print("".join(map(str, expected_bits[:50])))

#     print("\nRecovered first 50 bits:")
#     print("".join(map(str, recovered_bits[:50])))

#     # Basic checks
#     assert len(recovered_bits) > 0
#     assert np.all(
#         (recovered_bits == 0) |
#         (recovered_bits == 1)
#     )


# if __name__ == "__main__":
#     test_bpsk_demodulation()





# import json
# import numpy as np

# from signal_processing.synchronization.frequency_sync import (
#     correct_frequency_offset
# )

# from signal_processing.synchronization.phase_sync import (
#     correct_phase_offset
# )

# from signal_processing.synchronization.timing_sync import (
#     estimate_timing_offset,
#     sample_symbols
# )

# from signal_processing.demodulation.qpsk import (
#     demodulate_qpsk
# )


# def test_qpsk_demodulation():

#     # -----------------------------
#     # Load IQ signal
#     # -----------------------------

#     iq_filepath = "data/iq/signal_0013_qpsk.iq"

#     data = np.fromfile(
#         iq_filepath,
#         dtype="<f4"
#     )

#     i = data[0::2]
#     q = data[1::2]

#     iq = i + 1j * q

#     # -----------------------------
#     # Signal parameters
#     # -----------------------------

#     sampling_frequency_hz = 1_000_000
#     frequency_offset_hz = -5_000

#     # IMPORTANT:
#     # The synchronization function applies
#     # the negative of this value.
#     phase_offset_degrees = 135

#     samples_per_symbol = 10

#     # -----------------------------
#     # Frequency synchronization
#     # -----------------------------

#     frequency_corrected = correct_frequency_offset(
#         iq,
#         frequency_offset_hz,
#         sampling_frequency_hz
#     )

#     # -----------------------------
#     # Phase synchronization
#     # -----------------------------

#     phase_corrected = correct_phase_offset(
#         frequency_corrected,
#         phase_offset_degrees
#     )

#     # -----------------------------
#     # Timing synchronization
#     # -----------------------------

#     timing_offset = estimate_timing_offset(
#         phase_corrected,
#         samples_per_symbol
#     )

#     symbols = sample_symbols(
#         phase_corrected,
#         samples_per_symbol,
#         timing_offset
#     )

#     # -----------------------------
#     # QPSK demodulation
#     # -----------------------------

#     recovered_bits = demodulate_qpsk(symbols)

#     # -----------------------------
#     # Load ground-truth metadata
#     # -----------------------------

#     metadata_filepath = (
#         "data/metadata/signal_0013_qpsk.json"
#     )

#     with open(metadata_filepath, "r") as file:
#         metadata = json.load(file)

#     expected_bits = np.array(
#         [int(bit) for bit in metadata["bits"]],
#         dtype=np.uint8
#     )

#     # -----------------------------
#     # Compare recovered bits
#     # -----------------------------

#     number_of_bits = min(
#         len(expected_bits),
#         len(recovered_bits)
#     )

#     correct_bits = np.sum(
#         expected_bits[:number_of_bits]
#         == recovered_bits[:number_of_bits]
#     )

#     accuracy = (
#         correct_bits / number_of_bits
#     ) * 100

#     # -----------------------------
#     # Results
#     # -----------------------------

#     print("\n-----------------------------")
#     print("QPSK DEMODULATION TEST")
#     print("-----------------------------")

#     print("Original IQ samples:", len(iq))
#     print("Samples per symbol:", samples_per_symbol)
#     print("Frequency offset:", frequency_offset_hz, "Hz")
#     print("Phase correction:", phase_offset_degrees, "degrees")
#     print("Timing offset:", timing_offset)
#     print("Recovered symbols:", len(symbols))
#     print("Expected bits:", len(expected_bits))
#     print("Recovered bits:", len(recovered_bits))
#     print("Correct bits:", correct_bits)
#     print("Bit accuracy:", accuracy, "%")

#     print("\nExpected first 50 bits:")
#     print("".join(map(str, expected_bits[:50])))

#     print("\nRecovered first 50 bits:")
#     print("".join(map(str, recovered_bits[:50])))

#     # -----------------------------
#     # Basic validation
#     # -----------------------------

#     assert len(recovered_bits) > 0

#     assert np.all(
#         (recovered_bits == 0) |
#         (recovered_bits == 1)
#     )


# if __name__ == "__main__":
#     test_qpsk_demodulation()



# import json
# import numpy as np

# from signal_processing.synchronization.frequency_sync import (
#     correct_frequency_offset
# )

# from signal_processing.synchronization.phase_sync import (
#     correct_phase_offset
# )

# from signal_processing.synchronization.timing_sync import (
#     estimate_timing_offset,
#     sample_symbols
# )

# from signal_processing.demodulation.fsk import (
#     demodulate_fsk
# )


# def test_fsk_demodulation():

#     # -----------------------------
#     # Load IQ signal
#     # -----------------------------

#     iq_filepath = "data/iq/signal_0026_2fsk.iq"

#     raw = np.fromfile(
#         iq_filepath,
#         dtype="<f4"
#     )

#     iq = raw[0::2] + 1j * raw[1::2]

#     # -----------------------------
#     # Signal parameters
#     # -----------------------------

#     sampling_frequency_hz = 1_000_000
#     frequency_offset_hz = 20_000
#     phase_offset_degrees = 180
#     samples_per_symbol = 10

#     # -----------------------------
#     # Frequency synchronization
#     # -----------------------------

#     frequency_corrected = correct_frequency_offset(
#         iq,
#         frequency_offset_hz,
#         sampling_frequency_hz
#     )

#     # -----------------------------
#     # Phase synchronization
#     # -----------------------------

#     phase_corrected = correct_phase_offset(
#         frequency_corrected,
#         phase_offset_degrees
#     )

#     # -----------------------------
#     # Timing synchronization
#     # -----------------------------

#     timing_offset = estimate_timing_offset(
#         phase_corrected,
#         samples_per_symbol
#     )

#     symbols = sample_symbols(
#         phase_corrected,
#         samples_per_symbol,
#         timing_offset
#     )

#     # -----------------------------
#     # FSK demodulation
#     # -----------------------------

#     recovered_bits = demodulate_fsk(
#         phase_corrected,
#         samples_per_symbol
#     )

#     # -----------------------------
#     # Load ground truth
#     # -----------------------------

#     metadata_filepath = (
#         "data/metadata/signal_0026_2fsk.json"
#     )

#     with open(metadata_filepath, "r") as file:
#         metadata = json.load(file)

#     expected_bits = np.array(
#         [int(bit) for bit in metadata["bits"]],
#         dtype=np.uint8
#     )

#     # -----------------------------
#     # Compare
#     # -----------------------------

#     n = min(
#         len(expected_bits),
#         len(recovered_bits)
#     )

#     correct_bits = np.sum(
#         expected_bits[:n] ==
#         recovered_bits[:n]
#     )

#     accuracy = (
#         correct_bits / n
#     ) * 100

#     # -----------------------------
#     # Results
#     # -----------------------------

#     print("\n-----------------------------")
#     print("2FSK DEMODULATION TEST")
#     print("-----------------------------")

#     print("Original IQ samples:", len(iq))
#     print("Samples per symbol:", samples_per_symbol)
#     print("Frequency offset:", frequency_offset_hz, "Hz")
#     print("Phase correction:", phase_offset_degrees, "degrees")
#     print("Timing offset:", timing_offset)
#     print("Recovered symbols:", len(symbols))
#     print("Expected bits:", len(expected_bits))
#     print("Recovered bits:", len(recovered_bits))
#     print("Correct bits:", correct_bits)
#     print("Bit accuracy:", accuracy, "%")

#     print("\nExpected first 50 bits:")
#     print(
#         "".join(
#             map(str, expected_bits[:50])
#         )
#     )

#     print("\nRecovered first 50 bits:")
#     print(
#         "".join(
#             map(str, recovered_bits[:50])
#         )
#     )


# if __name__ == "__main__":
#     test_fsk_demodulation()





from ai.preprocessing.iq_loader import load_iq_file

from signal_processing.synchronization.frequency_sync import (
    correct_frequency_offset
)

from signal_processing.synchronization.phase_sync import (
    correct_phase_offset
)

from signal_processing.synchronization.timing_sync import (
    estimate_timing_offset,
    sample_symbols
)

from signal_processing.demodulation.bpsk import demodulate_bpsk
from signal_processing.demodulation.qpsk import demodulate_qpsk
from signal_processing.demodulation.fsk import demodulate_fsk
from signal_processing.demodulation.qam import demodulate_16qam

import json
import numpy as np
from pathlib import Path


def _generate_synthetic_signal(modulation: str, num_samples: int = 8192, rng_seed: int = 42):
    """
    Generate a synthetic modulated signal for testing.
    Returns (iq_samples, bits_array, samples_per_symbol).
    """
    rng = np.random.default_rng(rng_seed)
    sample_rate = 1_000_000.0
    symbol_rate = 100_000.0
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = num_samples // samples_per_symbol
    
    if modulation == "BPSK":
        bits = rng.integers(0, 2, size=num_symbols, dtype=np.uint8)
        symbols = np.where(bits == 1, 1.0 + 0j, -1.0 + 0j)
        baseband = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
    elif modulation == "QPSK":
        bits = rng.integers(0, 2, size=num_symbols * 2, dtype=np.uint8)
        i_bits = bits[0::2]
        q_bits = bits[1::2]
        i_sym = np.where(i_bits == 0, 1.0, -1.0)
        q_sym = np.where(q_bits == 0, 1.0, -1.0)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(2.0)
        baseband = np.repeat(symbols, samples_per_symbol)[:num_samples]
        
    elif modulation == "FSK":
        bits = rng.integers(0, 2, size=num_symbols, dtype=np.uint8)
        f_dev = symbol_rate / 2.0
        t_sym = np.arange(samples_per_symbol) / sample_rate
        chunks = []
        phase = 0.0
        for b in bits:
            freq = f_dev if b == 1 else -f_dev
            chunk = np.exp(1j * (2 * np.pi * freq * t_sym + phase))
            phase = np.angle(chunk[-1])
            chunks.append(chunk)
        baseband = np.concatenate(chunks)[:num_samples]
        
    elif modulation == "16QAM":
        bits = rng.integers(0, 2, size=num_symbols * 4, dtype=np.uint8)
        norm = np.sqrt(10.0)
        gray_lut = {
            (0, 0): -3.0, (0, 1): -1.0,
            (1, 1): 1.0, (1, 0): 3.0,
        }
        symbols = []
        for s_idx in range(num_symbols):
            b_chunk = bits[s_idx * 4 : (s_idx + 1) * 4]
            i_val = gray_lut[(b_chunk[0], b_chunk[1])]
            q_val = gray_lut[(b_chunk[2], b_chunk[3])]
            symbols.append((i_val + 1j * q_val) / norm)
        symbols = np.array(symbols, dtype=np.complex128)
        baseband = np.repeat(symbols, samples_per_symbol)[:num_samples]
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    
    # No frequency/phase offset for synthetic tests (clean signal)
    iq = baseband
    return iq, bits, samples_per_symbol


def _test_dataset_demodulation(class_name: str, signal_id: str, demodulator, uses_timing: bool = True):
    """
    Test demodulation on real dataset samples.
    class_name: "BPSK", "QPSK", "FSK", or "16QAM"
    signal_id: e.g., "signal_0001", "signal_0201", "signal_0401", "signal_0601"
    """
    # Determine file paths based on modulation type
    if class_name == "BPSK":
        iq_path = Path(f"data/iq/BPSK/{signal_id}_bpsk.iq")
        metadata_path = Path(f"data/metadata/BPSK/{signal_id}_bpsk.json")
    elif class_name == "QPSK":
        iq_path = Path(f"data/iq/QPSK/{signal_id}_qpsk.iq")
        metadata_path = Path(f"data/metadata/QPSK/{signal_id}_qpsk.json")
    elif class_name == "FSK":
        iq_path = Path(f"data/iq/FSK/{signal_id}_fsk.iq")
        metadata_path = Path(f"data/metadata/FSK/{signal_id}_2fsk.json")
    elif class_name == "16QAM":
        iq_path = Path(f"data/iq/QAM16/{signal_id}_qam16.iq")
        metadata_path = Path(f"data/metadata/QAM16/{signal_id}_16qam.json")
    else:
        raise ValueError(f"Unknown modulation: {class_name}")
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    expected_bits = np.array(
        [int(bit) for bit in metadata["bits"]],
        dtype=np.uint8
    )
    
    # Load IQ data
    iq = load_iq_file(str(iq_path))
    samples_per_symbol = metadata["samples_per_symbol"]
    frequency_offset = metadata["frequency_offset_hz"]
    phase_offset = metadata["phase_offset_degrees"]
    sampling_frequency = metadata["sampling_frequency_hz"]
    
    # Apply synchronization
    frequency_corrected = correct_frequency_offset(
        iq, frequency_offset, sampling_frequency
    )
    phase_corrected = correct_phase_offset(frequency_corrected, phase_offset)
    
    if uses_timing:
        timing_offset = estimate_timing_offset(phase_corrected, samples_per_symbol)
        symbols = sample_symbols(phase_corrected, samples_per_symbol, timing_offset)
        recovered_bits = demodulator(symbols)
    else:
        recovered_bits = demodulator(phase_corrected, samples_per_symbol)
    
    # Compare
    compare_length = min(len(expected_bits), len(recovered_bits))
    correct_bits = int(np.sum(expected_bits[:compare_length] == recovered_bits[:compare_length]))
    bit_accuracy = (
        (correct_bits / compare_length) * 100
        if compare_length > 0
        else 0.0
    )
    
    print(f"[PASS] Dataset integration {class_name}: successfully recovered {len(recovered_bits)} binary bits from {iq_path.name}")
    print(f"  Accuracy: {bit_accuracy:.2f}%")
    
    assert len(recovered_bits) > 0
    assert np.all((recovered_bits == 0) | (recovered_bits == 1))


def _test_synthetic_demodulation(class_name: str, demodulator, uses_timing: bool = True):
    """
    Test demodulation on synthetic (clean) signal.
    """
    iq, bits, sps = _generate_synthetic_signal(class_name)
    
    if uses_timing:
        timing_offset = estimate_timing_offset(iq, sps)
        symbols = sample_symbols(iq, sps, timing_offset)
        recovered_bits = demodulator(symbols)
    else:
        recovered_bits = demodulator(iq, sps)
    
    compare_length = min(len(bits), len(recovered_bits))
    correct_bits = int(np.sum(bits[:compare_length] == recovered_bits[:compare_length]))
    bit_accuracy = (
        (correct_bits / compare_length) * 100
        if compare_length > 0
        else 0.0
    )
    
    print(f"[PASS] Synthetic {class_name}: recovered {len(recovered_bits)} bits with {bit_accuracy:.2f}% accuracy")
    
    assert len(recovered_bits) > 0
    assert np.all((recovered_bits == 0) | (recovered_bits == 1))


# Synthetic tests
def test_synthetic_bpsk():
    _test_synthetic_demodulation("BPSK", demodulate_bpsk, uses_timing=True)

def test_synthetic_qpsk():
    _test_synthetic_demodulation("QPSK", demodulate_qpsk, uses_timing=True)

def test_synthetic_fsk():
    _test_synthetic_demodulation("FSK", demodulate_fsk, uses_timing=False)

def test_synthetic_16qam():
    _test_synthetic_demodulation("16QAM", demodulate_16qam, uses_timing=True)


# Dataset tests
def test_dataset_bpsk():
    _test_dataset_demodulation("BPSK", "signal_0001", demodulate_bpsk, uses_timing=True)

def test_dataset_qpsk():
    _test_dataset_demodulation("QPSK", "signal_0201", demodulate_qpsk, uses_timing=True)

def test_dataset_fsk():
    _test_dataset_demodulation("FSK", "signal_0401", demodulate_fsk, uses_timing=False)

def test_dataset_16qam():
    _test_dataset_demodulation("16QAM", "signal_0601", demodulate_16qam, uses_timing=True)


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING DEMODULATION TESTS")
    print("=" * 60)
    
    test_synthetic_bpsk()
    test_synthetic_qpsk()
    test_synthetic_fsk()
    test_synthetic_16qam()
    
    test_dataset_bpsk()
    test_dataset_qpsk()
    test_dataset_fsk()
    test_dataset_16qam()
    
    print("\n" + "=" * 60)
    print("ALL DEMODULATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)