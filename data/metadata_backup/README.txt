SIH26147 METADATA - OLDER CODE-COMPATIBLE FORMAT
=================================================

800 JSON metadata files:
BPSK  : 200
QPSK  : 200
2FSK  : 200
16QAM : 200

This follows the metadata schema of the uploaded
sih26147_synthetic_dataset_starter.zip.

Important naming convention:
BPSK  -> signal_XXXX_bpsk.json
QPSK  -> signal_XXXX_qpsk.json
2FSK  -> signal_XXXX_2fsk.json
16QAM -> signal_XXXX_16qam.json

Metadata fields intentionally use the older names:
filename, modulation, sampling_frequency_hz,
symbol_rate_hz, nominal_carrier_frequency_hz,
frequency_offset_hz, phase_offset_degrees,
amplitude_scale, signal_to_noise_ratio_db,
samples, samples_per_symbol, iq_file_convention,
wav_file_convention, baseband, bits.
