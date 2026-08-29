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





# from ai.preprocessing.iq_loader import load_iq_file

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

# from signal_processing.demodulation.qam import (
#     demodulate_16qam
# )

# import json
# import numpy as np
# import matplotlib.pyplot as plt


# def test_16qam_demodulation():

#     print("\n-----------------------------")
#     print("16QAM DEMODULATION TEST")
#     print("-----------------------------")

#     # --------------------------------------------------
#     # File paths
#     # --------------------------------------------------

#     iq_path = "data/iq/signal_0033_16qam.iq"
#     metadata_path = "data/metadata/signal_0033_16qam.json"

#     # --------------------------------------------------
#     # Load metadata
#     # --------------------------------------------------

#     with open(metadata_path, "r") as f:
#         metadata = json.load(f)

#     expected_bits_string = metadata["bits"]

#     expected_bits = np.array(
#         [int(bit) for bit in expected_bits_string],
#         dtype=np.uint8
#     )

#     samples_per_symbol = metadata["samples_per_symbol"]
#     frequency_offset = metadata["frequency_offset_hz"]
#     phase_offset = metadata["phase_offset_degrees"]

#     sampling_frequency = metadata["sampling_frequency_hz"]

#     # --------------------------------------------------
#     # Load IQ
#     # --------------------------------------------------

#     iq = load_iq_file(iq_path)

#     print("Original IQ samples:", len(iq))
#     print("Samples per symbol:", samples_per_symbol)
#     print("Frequency offset:", frequency_offset, "Hz")
#     print("Phase correction:", phase_offset, "degrees")

#     # --------------------------------------------------
#     # Frequency synchronization
#     # --------------------------------------------------

#     frequency_corrected = correct_frequency_offset(
#         iq,
#         frequency_offset,
#         sampling_frequency
#     )

#     # --------------------------------------------------
#     # Phase synchronization
#     # --------------------------------------------------

#     phase_corrected = correct_phase_offset(
#         frequency_corrected,
#         phase_offset
#     )

#     # --------------------------------------------------
#     # Timing synchronization
#     # --------------------------------------------------

#     timing_offset = estimate_timing_offset(
#         phase_corrected,
#         samples_per_symbol
#     )

#     symbols = sample_symbols(
#         phase_corrected,
#         samples_per_symbol,
#         timing_offset
#     )

#     print("Timing offset:", timing_offset)
#     print("Recovered symbols:", len(symbols))

#     # --------------------------------------------------
#     # 16QAM demodulation
#     # --------------------------------------------------

#     recovered_bits = demodulate_16qam(symbols)

#     # --------------------------------------------------
#     # Compare expected and recovered bits
#     # --------------------------------------------------

#     compare_length = min(
#         len(expected_bits),
#         len(recovered_bits)
#     )

#     expected_compare = expected_bits[:compare_length]
#     recovered_compare = recovered_bits[:compare_length]

#     correct_bits = np.sum(
#         expected_compare == recovered_compare
#     )

#     accuracy = (
#         correct_bits / compare_length * 100
#         if compare_length > 0
#         else 0
#     )

#     # --------------------------------------------------
#     # Results
#     # --------------------------------------------------

#     print("Expected bits:", len(expected_bits))
#     print("Recovered bits:", len(recovered_bits))
#     print("Correct bits:", correct_bits)
#     print("Bit accuracy:", accuracy, "%")

#     print("\nExpected first 50 bits:")
#     print("".join(map(str, expected_bits[:50])))

#     print("\nRecovered first 50 bits:")
#     print("".join(map(str, recovered_bits[:50])))

#     # --------------------------------------------------
#     # Validation
#     # --------------------------------------------------

#     assert len(recovered_bits) == len(expected_bits), (
#         f"Bit length mismatch: "
#         f"expected {len(expected_bits)}, "
#         f"got {len(recovered_bits)}"
#     )

#     assert accuracy >= 95.0, (
#         f"16QAM accuracy too low: {accuracy:.2f}%"
#     )


# if __name__ == "__main__":
#     test_16qam_demodulation()