"""
Synthetic communication signal dataset generator for SYNAPS (SIH26147).

Generates BPSK, QPSK, FSK, and 16QAM signals with configurable parameters,
saving corresponding .iq, .wav, and .json metadata files.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy.io import wavfile

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import DATA_ROOT, IQ_ROOT, WAV_ROOT, METADATA_ROOT, CLASS_NAMES


def generate_signal(
    modulation: str,
    num_samples: int = 8192,
    sample_rate: float = 1_000_000.0,
    symbol_rate: float = 100_000.0,
    snr_db: float = 20.0,
    carrier_offset_hz: float = 0.0,
    phase_offset_deg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if rng is None:
        rng = np.random.default_rng()

    sps = int(round(sample_rate / symbol_rate))
    num_symbols = int(np.ceil(num_samples / sps))

    if modulation == "BPSK":
        bits = rng.integers(0, 2, size=num_symbols, dtype=np.uint8)
        symbols = np.where(bits == 1, 1.0 + 0j, -1.0 + 0j)
        baseband = np.repeat(symbols, sps)[:num_samples]

    elif modulation == "QPSK":
        bits = rng.integers(0, 2, size=num_symbols * 2, dtype=np.uint8)
        # Gray mapping
        i_bits = bits[0::2]
        q_bits = bits[1::2]
        i_sym = np.where(i_bits == 0, 1.0, -1.0)
        q_sym = np.where(q_bits == 0, 1.0, -1.0)
        symbols = (i_sym + 1j * q_sym) / np.sqrt(2.0)
        baseband = np.repeat(symbols, sps)[:num_samples]

    elif modulation == "FSK":
        bits = rng.integers(0, 2, size=num_symbols, dtype=np.uint8)
        f_dev = symbol_rate / 2.0
        t_sym = np.arange(sps) / sample_rate
        chunks = []
        phase = 0.0
        for b in bits:
            freq = f_dev if b == 1 else -f_dev
            chunk = np.exp(1j * (2 * np.pi * freq * t_sym + phase))
            phase = np.angle(chunk[-1])
            chunks.append(chunk)
        baseband = np.concatenate(chunks)[:num_samples]

    elif modulation == "QAM16":
        bits = rng.integers(0, 2, size=num_symbols * 4, dtype=np.uint8)
        norm = np.sqrt(10.0)
        gray_lut = {
            (0, 0): -3.0,
            (0, 1): -1.0,
            (1, 1): 1.0,
            (1, 0): 3.0,
        }
        symbols = []
        for s_idx in range(num_symbols):
            b_chunk = bits[s_idx * 4 : (s_idx + 1) * 4]
            i_val = gray_lut[(b_chunk[0], b_chunk[1])]
            q_val = gray_lut[(b_chunk[2], b_chunk[3])]
            symbols.append((i_val + 1j * q_val) / norm)
        symbols = np.array(symbols, dtype=np.complex128)
        baseband = np.repeat(symbols, sps)[:num_samples]

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    # Apply Carrier Frequency Offset and Phase Offset
    t = np.arange(len(baseband)) / sample_rate
    cfo_rad = 2 * np.pi * carrier_offset_hz * t
    phase_rad = np.deg2rad(phase_offset_deg)
    modulated = baseband * np.exp(1j * (cfo_rad + phase_rad))

    # Add AWGN
    sig_pwr = np.mean(np.abs(modulated) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_pwr = sig_pwr / (snr_linear + 1e-12)
    noise = (rng.normal(0, np.sqrt(noise_pwr / 2), len(modulated)) +
             1j * rng.normal(0, np.sqrt(noise_pwr / 2), len(modulated)))
    noisy_iq = (modulated + noise).astype(np.complex64)

    meta = {
        "modulation": modulation,
        "sampling_frequency_hz": sample_rate,
        "symbol_rate_hz": symbol_rate,
        "samples": num_samples,
        "samples_per_symbol": sps,
        "signal_to_noise_ratio_db": snr_db,
        "frequency_offset_hz": carrier_offset_hz,
        "phase_offset_degrees": phase_offset_deg,
        "bits": "".join(map(str, bits.tolist())),
    }

    return noisy_iq, bits, meta


def save_generated_sample(
    sample_id: str,
    modulation: str,
    iq_samples: np.ndarray,
    metadata: Dict,
    output_root: Path = DATA_ROOT,
):
    norm_mod = modulation.upper()
    iq_dir = output_root / "iq" / norm_mod
    wav_dir = output_root / "wav" / norm_mod
    meta_dir = output_root / "metadata" / norm_mod

    iq_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{sample_id}_{norm_mod.lower()}"
    iq_file = iq_dir / f"{base_name}.iq"
    wav_file = wav_dir / f"{base_name}.wav"
    meta_file = meta_dir / f"{base_name}.json"

    # 1. Save IQ as interleaved float32
    iq_interleaved = np.empty(len(iq_samples) * 2, dtype=np.float32)
    iq_interleaved[0::2] = iq_samples.real
    iq_interleaved[1::2] = iq_samples.imag
    iq_interleaved.tofile(iq_file)

    # 2. Save WAV stereo float32
    wav_stereo = np.column_stack([iq_samples.real, iq_samples.imag]).astype(np.float32)
    wavfile.write(str(wav_file), int(metadata["sampling_frequency_hz"]), wav_stereo)

    # 3. Save Metadata JSON
    metadata["filename"] = base_name
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return iq_file, wav_file, meta_file


def main():
    parser = argparse.ArgumentParser(description="SYNAPS Synthetic Signal Generator")
    parser.add_argument("--modulation", type=str, default="BPSK", choices=CLASS_NAMES)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--count", type=int, default=1, help="Number of files to generate")
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--cfo", type=float, default=0.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=str(DATA_ROOT))
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print(f"Generating {args.count} {args.modulation} signals...")

    for i in range(args.count):
        sample_id = f"synth_{i+1:04d}"
        iq, bits, meta = generate_signal(
            modulation=args.modulation,
            num_samples=args.samples,
            snr_db=args.snr,
            carrier_offset_hz=args.cfo,
            phase_offset_deg=args.phase,
        )
        iq_f, wav_f, meta_f = save_generated_sample(sample_id, args.modulation, iq, meta, out_root)
        print(f"Saved: {iq_f.name}")

    print("Signal generation complete.")


if __name__ == "__main__":
    from typing import Dict, Optional, Tuple
    main()