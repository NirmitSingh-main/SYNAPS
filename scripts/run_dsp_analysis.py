"""
Standalone DSP analysis runner script for SYNAPS (SIH26147).
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import resolve_sample_paths
from signal_processing.input.loader import load_signal
from signal_processing.input.format_detection import detect_format
from dsp.spectral.fft import compute_fft
from dsp.spectral.psd import estimate_psd
from dsp.spectral.bandwidth import estimate_bandwidth
from dsp.signal_quality.snr import estimate_snr
from dsp.frequency.cfo import estimate_cfo
from dsp.frequency.frequency_estimation import estimate_frequency
from dsp.phase.phase_estimation import estimate_phase
from dsp.timing.symbol_rate import estimate_symbol_rate
from dsp.constellation.constellation import analyze_constellation
from dsp.statistical.hoc import calculate_hoc


def run_dsp(file_path: str, sample_rate: float = None) -> dict:
    resolved = resolve_sample_paths(file_path)
    target = resolved["iq_path"] or resolved["wav_path"] or Path(file_path)

    if not target.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    fmt = detect_format(str(target))
    samples, fs = load_signal(str(target), iq_sample_rate=sample_rate)

    print("=" * 60)
    print(f"DSP ANALYSIS: {target.name} ({fmt})")
    print("=" * 60)
    print(f"Samples      : {len(samples)}")
    print(f"Sampling Rate: {fs:,.0f} Hz")

    fft_res = compute_fft(samples, sampling_rate=fs)
    bw_res = estimate_bandwidth(samples, sampling_rate=fs)
    snr_res = estimate_snr(samples)
    freq_res = estimate_frequency(samples, sampling_rate=fs)
    est_f = float(freq_res.get("estimated_frequency_hz", 0.0))
    cfo_res = estimate_cfo(samples, sampling_rate=fs, reference_frequency_hz=0.0)
    phase_res = estimate_phase(samples, sampling_rate=fs, frequency_hz=est_f)
    timing_res = estimate_symbol_rate(samples, sampling_rate=fs)
    constellation_res = analyze_constellation(samples)
    hoc_res = calculate_hoc(samples)

    summary = {
        "file": str(target),
        "format": fmt,
        "sample_count": len(samples),
        "sampling_rate_hz": fs,
        "peak_frequency_hz": float(fft_res.get("peak_frequency_hz", 0.0)),
        "bandwidth_3db_hz": float(bw_res.get("bandwidth_3db_hz", bw_res.get("bandwidth_hz", 0.0))),
        "snr_db": float(snr_res.get("snr_db", 0.0)),
        "carrier_offset_hz": float(cfo_res.get("cfo_hz", 0.0)),
        "symbol_rate": float(timing_res.get("symbol_rate", 0.0)),
        "hoc_c40": float(abs(hoc_res.get("C40", 0.0))),
        "hoc_c42": float(abs(hoc_res.get("C42", 0.0))),
    }

    print("\n--- DSP ESTIMATES ---")
    print(f"  SNR (dB)       : {summary['snr_db']:.2f}")
    print(f"  CFO (Hz)       : {summary['carrier_offset_hz']:.2f}")
    print(f"  Bandwidth (Hz) : {summary['bandwidth_3db_hz']:,.0f}")
    print(f"  Symbol Rate    : {summary['symbol_rate']:,.0f}")
    print(f"  Peak Frequency : {summary['peak_frequency_hz']:,.0f} Hz")
    print(f"  HOC C40 / C42  : {summary['hoc_c40']:.3f} / {summary['hoc_c42']:.3f}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run DSP Analysis on a Signal")
    parser.add_argument("file_path", type=str, help="Path to .iq or .wav signal")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sampling rate in Hz")
    args = parser.parse_args()

    run_dsp(args.file_path, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()